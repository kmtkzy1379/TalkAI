"""イブの今の感情（S5・ユーザ裁定 2026-09-26・数値と時間で薄れる形は 2026-10-09）。

ユーザ（原文）:「、。ごとに感情を分析してその感情を現感情として保存、その感情によってirodoriの声トーンや
応答LLM,フィードバックLLMに入れたりして感情にそった応答をさせる。フィードバックでも感情分析してるから
フィードバックがきたらそれに更新。イメージは、。でもフィードバックでも更新タイミングが来たら更新。
（現感情を、。やフィードバックに入れれば大きく不自然な感情切り替わりは少ないと予想）。感情ラベルは喜怒哀楽笑いでいい。」
以前の裁定: 2026-09-23「ユーザー音声から読み取れる感情とeveの現在の感情を考慮して…どういう感情になるか推察して
感情出力」/ 2026-09-24「会話途中とフィードバックから分析して現感情出させる」/「感情判定はルナでよさそう」

2026-10-09（ユーザ原文）:「喜怒哀楽笑…に数値0〜100の数値をつけて起動終了時〜起動開始時までの時間で数値を減少させる。
…0が平常。50が明確にその感情を抱いてる。100がとても激しくその感情を抱いてる。最大合計数値100で割り振る。」
（きっかけ: 起動直後、前回の取りこぼし分のフィードバックが 10日前の謝り合いから「哀」を出し、それが「今の感情」として
自発発話の指示と声（😭）に使われていた。直近フィードバックの欄は「10日前の会話」と正しく書いていたのに感情だけ現在扱い）

- 感情は5つの数値（各 0〜100・合計100以下・0＝平常）。判定（「、」「。」ごと・フィードバック）は5つに割り振って返す
- **判定に使った会話の時点（as_of・壁時計）から時間がたつほど薄れる**（1時間あたり `EMOTION_DECAY_PER_HOUR` ずつ引く・
  0 で止まる）。起動していない間も同じ時計で減るので、何日も前の会話の感情は平常に戻っている
  （ユーザの言う「終了〜起動の間で減らす」を、判定した時点からの経過で数える＝起動中に長く黙っていた間も同じように減る）
- **新しい情報が勝つ**: 今の感情より古い時点の判定は捨てる（例: ユーザの新しい発話で変わった後に、その前の会話を
  振り返ったフィードバックが届いても上書きしない）。フィードバックの時点は振り返った会話の最後の発話の時刻
- 判定のたびにファイルへ保存し（`EMOTION_STATE_FILE`）、起動時に読む＝前回の続きから、経過の分だけ薄れた状態で始まる
- 使う所: 応答LLM（`context_assembler` の「# イブの今の気持ち」に数値で）・フィードバックLLM・感情判定（今の感情として）・
  声とフィラー（いちばん強い感情が `EMOTION_TONE_MIN` 以上の時だけ、その感情の声。弱い時は普通の声）
- 応答は判定を待たない（間に合わなければ前の感情のまま・判定の失敗は「変えない」だけ）
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .config import Config

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
MAX_TOTAL = 100.0
# 数値の意味（ユーザ 2026-10-09 の定義そのまま）。判定の指示とプロンプトの両方で同じ言い方を使う
SCALE = "0=平常（その感情は無い）・50=はっきりその感情を抱いている・100=とても激しくその感情を抱いている"
SCALE_SHORT = "0=平常・50=はっきり・100=とても激しい"  # 応答LLM に毎ターン入る方は短く
NEUTRAL = "平常（どの感情も0）"

# 実測の質問文が一緒に聞いていた「話者の感情」（使わないが、判定の前段として残す）
_USER6 = ("怒り", "嫌悪", "恐れ", "喜び", "悲しみ", "驚き")

# 判定（gpt-6-luna・reasoning_effort=none）。2026-10-09 に eve5 を「1つのラベル＋自信」から「5つに数値を割り振る」へ
JUDGE_SYSTEM = """\
あなたはAI VTuber「イブ」の内省（フィードバック）モジュールです。
相手（ユーザ）の発話を読み、次の3つを判定して JSON だけを出力してください。

user6: 発話している人の感情。次のどれか1つと、その判定にどれくらい自信があるか（0〜100 の整数）: {l6}
user5: 発話している人の感情。次のどれか1つと、その判定にどれくらい自信があるか（0〜100 の整数）: {l5}
eve5: この発話を聞いたイブが次に返す声に乗る感情を、{l5} の5つに数値で割り振る。
  各 0〜100 の整数（{scale}）。5つの合計は100以下。どの感情も無ければすべて0。
  イブの今の感情: {cur}

出力形式:
{{"user6": {{"label": "...", "confidence": 0}}, "user5": {{"label": "...", "confidence": 0}}, "eve5": {eve5_shape}}}"""

JudgeFn = Callable[[list, str, str], Awaitable[Optional[dict]]]

_ALIAS = {"笑": "笑い"}


def normalize(raw) -> Optional[dict[str, float]]:
    """判定の数値を整える: 5つのラベルだけ・各 0〜100・合計が100を超えたら比率を保って100に縮める。

    知っているラベルが1つも無い（壊れた出力・形が違う）時は None（＝その判定は使わない）。全部0は平常として有効。
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        key = _ALIAS.get(str(k).strip(), str(k).strip())
        if key not in LABELS:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        out[key] = min(100.0, max(0.0, x))
    if not out:
        return None
    total = sum(out.values())
    if total > MAX_TOTAL:
        out = {k: v * MAX_TOTAL / total for k, v in out.items()}
    return {k: out.get(k, 0.0) for k in LABELS}


def _ordered(values: dict) -> list[tuple[str, int]]:
    """0 を除いて強い順（同じ強さはラベルの順）。"""
    order = list(LABELS)
    items = [(k, int(v)) for k, v in values.items() if int(v) > 0]
    return sorted(items, key=lambda kv: (-kv[1], order.index(kv[0])))


def brief(values: dict) -> str:
    """判定・フィードバックに「今の感情」として渡す短い形（例「喜 40・笑い 15」・全部0なら平常）。"""
    items = _ordered(values)
    return "・".join(f"{k} {v}" for k, v in items) if items else NEUTRAL


def describe(values: dict) -> Optional[str]:
    """応答LLM 用の1行（説明と数値の意味つき）。全部0（平常）なら None＝気持ちのブロックを出さない。"""
    items = _ordered(values)
    if not items:
        return None
    body = "・".join(f"{k}（{LABELS[k]}）{v}" for k, v in items)
    return f"{body}（各0〜100。{SCALE_SHORT}）"


def _iso_to_epoch(iso: str) -> Optional[float]:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


class EmotionState:
    """イブの今の感情（loop 所有・単一スレッド・ロックなし）。

    `_raw` は判定した時点（as_of）の値。読む時に、そこからの経過で薄れた値（`values()`）を計算する。
    """

    def __init__(self, *, path: Optional[str] = None, decay_per_hour: Optional[float] = None,
                 tone_min: Optional[float] = None, now_fn: Callable[[], float] = time.time) -> None:
        self._raw: dict[str, float] = {}
        self.as_of = 0.0       # この感情の判定に使った会話の時点（壁時計・epoch 秒）
        self.source = ""       # phrase（ユーザの話の途中）/ feedback / 前回の起動から引き継いだ時は保存時のまま
        self._decay = float(Config.EMOTION_DECAY_PER_HOUR if decay_per_hour is None else decay_per_hour)
        self._tone_min = float(Config.EMOTION_TONE_MIN if tone_min is None else tone_min)
        self._now = now_fn
        self._path: Optional[Path] = None
        if path:
            p = Path(path)
            self._path = p if p.is_absolute() else Path(__file__).resolve().parent.parent / p
            self._load()

    # --- 読む -------------------------------------------------------------

    def values(self, now: Optional[float] = None) -> dict[str, int]:
        """今の値（判定した時点からの経過で薄れた分を引いた整数・0 は省く）。"""
        if not self._raw:
            return {}
        hours = max(0.0, (self._now() if now is None else now) - self.as_of) / 3600.0
        cut = self._decay * hours
        out = {}
        for k in LABELS:
            v = int(round(self._raw.get(k, 0.0) - cut))
            if v > 0:
                out[k] = v
        return out

    def brief(self) -> str:
        return brief(self.values())

    def describe(self) -> Optional[str]:
        return describe(self.values())

    def dominant(self) -> Optional[str]:
        """声とフィラーに乗せる感情: いちばん強い感情（`EMOTION_TONE_MIN` 以上の時だけ・弱ければ None＝普通の声）。"""
        items = _ordered(self.values())
        if not items or items[0][1] < self._tone_min:
            return None
        return items[0][0]

    @property
    def label(self) -> Optional[str]:
        """1つのラベルが要る所（フィラーの感情の声・場面の判定）用の別名。`dominant()` と同じ。"""
        return self.dominant()

    # --- 書く -------------------------------------------------------------

    def note(self, values, *, source: str, as_of: float) -> bool:
        """判定結果を反映する。今より古い時点の判定・読めない値は捨てる（False）。as_of は壁時計（epoch 秒）。"""
        vals = normalize(values)
        if vals is None:
            return False
        if as_of < self.as_of:
            return False
        before = self.brief()
        self._raw, self.source, self.as_of = vals, source, float(as_of)
        after = self.brief()
        if before != after:
            age = self._now() - self.as_of
            old = f"・{age / 3600:.1f}時間前の会話から" if age > 600 else ""
            logger.info("🎭 イブの感情 %s → %s（%s%s）", before, after, source, old)
        self._save()
        return True

    # --- 保存 -------------------------------------------------------------

    def _save(self) -> None:
        if self._path is None:
            return
        rec = {"values": {k: round(v, 2) for k, v in self._raw.items()}, "as_of": self.as_of,
               "as_of_iso": datetime.fromtimestamp(self.as_of).astimezone().isoformat(), "source": self.source}
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)  # 書きかけで落ちても前の内容が残る
        except OSError:
            logger.exception("感情の保存に失敗（メモリ上は保持して継続）")

    def _load(self) -> None:
        try:
            rec = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("保存した感情（%s）を読めない（平常から始める）", self._path)
            return
        vals = normalize((rec or {}).get("values"))
        try:
            as_of = float(rec.get("as_of"))
        except (TypeError, ValueError, AttributeError):
            as_of = None
        if vals is None or as_of is None:
            logger.warning("保存した感情（%s）の形が違う（平常から始める）", self._path)
            return
        self._raw, self.as_of, self.source = vals, as_of, str(rec.get("source") or "")
        hours = max(0.0, self._now() - as_of) / 3600.0
        logger.info("🎭 前回の感情を読み込み: %s（%.1f時間前の判定）→ 今は %s",
                    brief({k: int(round(v)) for k, v in vals.items()}), hours, self.brief())


PERSONA_HOW = "感情は、この性格のイブとして判定する"


def judge_messages(context_lines: list, said: str, current: str, *, persona: str = "") -> list:
    """messages（system に今の感情・user に直前の会話と相手の発話）。current は `EmotionState.brief()`。

    persona（`load_persona()` の文字列）があれば system の最後に性格を添える。無いと、いじりや煽りを
    普通の人の感じ方で読み「哀」にする（性格つきの応答と食い違う・2026-10-05 実測）。
    """
    from .persona import persona_for

    shape = json.dumps({k: 0 for k in LABELS}, ensure_ascii=False)
    system = JUDGE_SYSTEM.format(l6=" / ".join(_USER6), l5=" / ".join(LABELS), scale=SCALE,
                                 cur=current or NEUTRAL, eve5_shape=shape) + persona_for(persona, PERSONA_HOW)
    ctx = ("直前の会話:\n" + "\n".join(context_lines) + "\n\n") if context_lines else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": ctx + f"相手の発話: {said}"}]


def parse_eve5(content: str) -> Optional[dict[str, float]]:
    """判定の JSON から eve5 の5つの数値を取り出す（壊れた JSON・形が違う時は None）。"""
    try:
        d = json.loads(content or "")
        return normalize(d.get("eve5"))
    except Exception:
        return None


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
    async def judge(context_lines: list, said: str, current: str) -> Optional[dict]:
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
                 max_inflight: int = 3, now_fn: Callable[[], float] = time.time) -> None:
        self._state = state
        self._judge = judge_fn
        self._context = context_fn
        self._max = max(1, int(max_inflight))
        self._now = now_fn
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
            self._run(seg, text, ctx, self._state.brief(), self._now()))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _run(self, seg: int, text: str, ctx: list, current: str, as_of: float) -> None:
        t0 = time.monotonic()
        try:
            values = await self._judge(ctx, text, current)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 判定できなければ変えないだけ（本流は止めない）
            logger.warning("感情の判定に失敗（今の感情のまま）: %s", e)
            return
        if values is None:
            logger.info("🎭 感情の判定 #%d: 読めない出力（今の感情のまま・%.2f秒）", seg, time.monotonic() - t0)
            return
        applied = self._state.note(values, source="phrase", as_of=as_of)
        logger.info("🎭 感情の判定 #%d: %s%s（%.2f秒）「%s」", seg, brief({k: int(round(v)) for k, v in values.items()}),
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


def span_end_epoch(turns) -> float:
    """フィードバックの判定の時点＝振り返った会話の最後の発話の時刻（読めなければ今）。"""
    try:
        t = _iso_to_epoch(turns[-1].stamp.iso)
    except (AttributeError, IndexError, TypeError):
        t = None
    return t if t is not None else time.time()


# ラベルの直後が区切り（= ＝ : ：・JSON の引用符も可）の時だけ数値を拾う（「楽しい」等の文中の字は拾わない）
_VALUES_LINE = {k: re.compile(rf"{re.escape(k)}\"?\s*[=＝:：]\s*(\d+(?:\.\d+)?)") for k in ("喜", "怒", "哀", "楽")}
_VALUES_LINE["笑い"] = re.compile(r"笑い?\"?\s*[=＝:：]\s*(\d+(?:\.\d+)?)")


def parse_values_line(line: str) -> Optional[dict[str, float]]:
    """フィードバックの `emotion_values: 喜=50, 怒=0, …` の行を5つの数値に（読めなければ None）。"""
    raw = {}
    for k, rx in _VALUES_LINE.items():
        m = rx.search(line or "")
        if m:
            raw[k] = m.group(1)
    return normalize(raw)
