"""イブの今の感情（S5・ユーザ裁定 2026-09-26）。

ユーザ（原文）:「、。ごとに感情を分析してその感情を現感情として保存、その感情によってirodoriの声トーンや
応答LLM,フィードバックLLMに入れたりして感情にそった応答をさせる。フィードバックでも感情分析してるから
フィードバックがきたらそれに更新。イメージは、。でもフィードバックでも更新タイミングが来たら更新。
（現感情を、。やフィードバックに入れれば大きく不自然な感情切り替わりは少ないと予想）。感情ラベルは喜怒哀楽笑いでいい。」
以前の裁定: 2026-09-23「ユーザー音声から読み取れる感情とeveの現在の感情を考慮して…どういう感情になるか推察して
感情出力」/ 2026-09-24「会話途中とフィードバックから分析して現感情出させる」/「感情判定はルナでよさそう」

- `EmotionState` … 今の感情（喜・怒・哀・楽・笑い のどれか・まだ判定していなければ None）。loop 上でだけ書く。
  **新しい情報が勝つ**: 更新ごとに「判定に使った会話の時点」（as_of）を持ち、今の感情より古い時点の判定は捨てる
  （例: ユーザの新しい発話で哀になった後に、その前の会話を振り返ったフィードバックが届いても上書きしない）
- `EmotionJudge` … ユーザの話の「、」「。」ごと（ストリーミング文字起こしの途中の文字・笑い判定と同じ合図）に
  gpt-6-luna（`emotion_judge` 役）へ「この発話を聞いたイブが次に返す声に乗る感情」を聞く。**今の感情も渡す**。
  質問文は実測（`tools/luna_emotion_eval.py`＝`tools/jvnv_eval.py` の FB_SYSTEM・JVNV / 名大会話コーパス）と同じ
- 応答は判定を待たない（間に合わなければ前の感情のまま・判定の失敗は「変えない」だけ）
- 使う所: 応答LLM（`context_assembler` の「# イブの今の気持ち」）・フィードバックLLM（入力と `emotion_label`）・
  Irodori の声（`emotion.yaml` の voice_emoji。未決の間は付けない）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .clock import now_mono

logger = logging.getLogger(__name__)

# ユーザ指定「感情ラベルは喜怒哀楽笑いでいい」。説明は実測に使った定義（tools/jvnv_eval.py の L5）
LABELS: dict[str, str] = {
    "喜": "嬉しい・喜んでいる",
    "怒": "怒っている・苛立っている",
    "哀": "悲しい・つらい",
    "楽": "穏やかで楽しい・リラックスしている",
    "笑い": "笑っている・可笑しがっている",
}
# 感情ごとのフィラーのバンク = bank/<FILLER_VOICE><suffix>/（ユーザ 2026-09-26「フィラーも感情載せるでいいと思う」）。
# `tools/build_filler_bank.py --emoji <emotion.yaml の voice_emoji>` で作る。フォルダ名は ASCII にする
BANK_SUFFIX: dict[str, str] = {
    "喜": "_emo_joy", "怒": "_emo_anger", "哀": "_emo_sad", "楽": "_emo_relaxed", "笑い": "_emo_laugh",
}
# 実測の質問文が一緒に聞いていた「話者の感情」（使わないが、質問文を実測と同じにするため残す）
_USER6 = ("怒り", "嫌悪", "恐れ", "喜び", "悲しみ", "驚き")
_NONE = "なし（初回）"

# 実測（2026-09-25・gpt-6-luna・reasoning_effort=none）と同じ文面。**変えると実測が当てはまらなくなる**
JUDGE_SYSTEM = """\
あなたはAI VTuber「イブ」の内省（フィードバック）モジュールです。
相手（ユーザ）の発話を読み、次の3つを判定して JSON だけを出力してください。
各判定には、その判定にどれくらい自信があるかを 0〜100 の整数で添えること。

user6: 発話している人の感情。次のどれか1つ: {l6}
user5: 発話している人の感情。次のどれか1つ: {l5}
eve5: この発話を聞いたイブが次に返す声に乗る感情。次のどれか1つ: {l5}
  イブの今の感情: {cur}

出力形式:
{{"user6": {{"label": "...", "confidence": 0}}, "user5": {{"label": "...", "confidence": 0}}, "eve5": {{"label": "...", "confidence": 0}}}}"""

JudgeFn = Callable[[list, str, Optional[str]], Awaitable[Optional[str]]]


def describe(label: Optional[str]) -> Optional[str]:
    """プロンプト用の1行（例「楽（穏やかで楽しい・リラックスしている）」）。未判定は None。"""
    if label not in LABELS:
        return None
    return f"{label}（{LABELS[label]}）"


class EmotionState:
    """イブの今の感情（loop 所有・単一スレッド・ロックなし）。"""

    def __init__(self) -> None:
        self.label: Optional[str] = None
        self.source = ""       # phrase（ユーザの話の途中）/ feedback
        self.as_of = 0.0       # この感情の判定に使った会話の時点（monotonic）
        self.updated = 0.0     # 書いた時刻（monotonic）

    def note(self, label: Optional[str], *, source: str, as_of: float) -> bool:
        """判定結果を反映する。今より古い時点の判定・知らないラベルは捨てる（False）。"""
        if label not in LABELS:
            return False
        if as_of < self.as_of:
            return False
        prev = self.label
        self.label, self.source, self.as_of, self.updated = label, source, as_of, now_mono()
        if prev != label:
            logger.info("🎭 イブの感情 %s → %s（%s）", prev or "なし", label, source)
        return True

    def describe(self) -> Optional[str]:
        return describe(self.label)


PERSONA_HOW = "感情は、この性格のイブとして判定する"


def judge_messages(context_lines: list, said: str, current: Optional[str], *, persona: str = "") -> list:
    """実測と同じ形の messages（system に今の感情・user に直前の会話と相手の発話）。

    persona（`load_persona()` の文字列）があれば system の最後に性格を添える。無いと、いじりや煽りを
    普通の人の感じ方で読み「哀」にする（性格つきの応答と食い違う・2026-10-05 実測）。
    """
    from .persona import persona_for

    system = JUDGE_SYSTEM.format(l6=" / ".join(_USER6), l5=" / ".join(LABELS),
                                 cur=current if current in LABELS else _NONE) + persona_for(persona, PERSONA_HOW)
    ctx = ("直前の会話:\n" + "\n".join(context_lines) + "\n\n") if context_lines else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": ctx + f"相手の発話: {said}"}]


def parse_eve5(content: str) -> Optional[str]:
    """判定の JSON から eve5 のラベルを取り出す（知らないラベル・壊れた JSON は None）。"""
    try:
        d = json.loads(content or "")
        label = str(((d.get("eve5") or {}).get("label")) or "").strip()
    except Exception:
        return None
    return label if label in LABELS else None


def _content(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        pass
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except Exception:
            return ""
    return ""


def make_judge_fn(registry, *, timeout_sec: float = 10.0, persona: str = "") -> JudgeFn:
    async def judge(context_lines: list, said: str, current: Optional[str]) -> Optional[str]:
        # reasoning_effort=none は実測と同じ条件。litellm は gpt-6-luna がこの引数を受けることを
        # 知らず弾く（UnsupportedParamsError・笑い判定で実測 2026-09-26）ので、通すことを明示する。
        r = await asyncio.wait_for(registry.complete(
            "emotion_judge", judge_messages(context_lines, said, current, persona=persona),
            response_format={"type": "json_object"}, reasoning_effort="none",
            allowed_openai_params=["reasoning_effort"]), timeout=timeout_sec)
        return parse_eve5(_content(r))

    return judge


class EmotionJudge:
    """「、」「。」ごとの途中の文字を受け取り、イブの感情を判定して `EmotionState` を更新する。

    判定は並行（上限あり）。後から始めた判定ほど新しい情報なので、古い判定が遅れて届いても
    上書きしない（`EmotionState.note` の as_of）。同じ区間の同じ文字は2回聞かない。
    """

    def __init__(self, *, state: EmotionState, judge_fn: JudgeFn, context_fn: Callable[[], list],
                 max_inflight: int = 3) -> None:
        self._state = state
        self._judge = judge_fn
        self._context = context_fn
        self._max = max(1, int(max_inflight))
        self._tasks: set = set()
        self._seen: dict = {}

    def on_phrase(self, seg: int, text: str) -> None:
        """途中の文字（その発話の頭からここまで）。「、」「。」が来た時と、確定した時に呼ばれる。"""
        text = (text or "").strip()
        if not text:
            return
        seen = self._seen.setdefault(seg, set())
        if text in seen:
            return
        if len(self._seen) > 32:
            for k in sorted(self._seen)[:-16]:
                self._seen.pop(k, None)
        if len(self._tasks) >= self._max:
            logger.debug("感情の判定が混んでいるので見送り（#%d）: %.20s", seg, text)
            return
        seen.add(text)
        try:
            ctx = list(self._context() or [])
        except Exception:
            logger.exception("感情の判定の会話の取得に失敗（会話なしで判定）")
            ctx = []
        t = asyncio.get_running_loop().create_task(
            self._run(seg, text, ctx, self._state.label, now_mono()))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _run(self, seg: int, text: str, ctx: list, current: Optional[str], as_of: float) -> None:
        t0 = time.monotonic()
        try:
            label = await self._judge(ctx, text, current)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 判定できなければ変えないだけ（本流は止めない）
            logger.warning("感情の判定に失敗（今の感情のまま）: %s", e)
            return
        if label is None:
            logger.info("🎭 感情の判定 #%d: 読めない出力（今の感情のまま・%.2f秒）", seg, time.monotonic() - t0)
            return
        applied = self._state.note(label, source="phrase", as_of=as_of)
        logger.info("🎭 感情の判定 #%d: %s%s（%.2f秒）「%s」", seg, label,
                    "" if applied else "・後の判定が先に届いたので使わない", time.monotonic() - t0, text[-30:])

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            try:
                await t
            except BaseException:
                pass
        self._tasks.clear()


def load_voice_emoji(path: Path) -> dict:
    """`emotion.yaml` の voice_emoji（感情 → Irodori の絵文字）。未決（null）・読めない時は空。"""
    try:
        import yaml

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("感情の表（%s）を読めない（声のトーンは付けない）: %s", path, e)
        return {}
    out = {}
    for k, v in (cfg.get("voice_emoji") or {}).items():
        if k in LABELS and v:
            out[k] = str(v)
    return out
