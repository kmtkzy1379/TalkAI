"""自己形成サイドカー — イブの好み・趣向を会話から育てる（PIPELINE_DESIGN.md §9.4 契約）。

## なぜ FeedbackWorker と別なのか（裁定 2026-08-15）
| | FeedbackWorker | ここ |
|---|---|---|
| 起動 | 応答完了ごと（実測 8分で11回） | 合図が出た時だけ（実測 イブ発話の 15.5%） |
| 速度 | **要る**（出力が surprise になり発話判定を駆動する） | 要らない（次の会話に間に合えばよい） |
| 壊れた時 | surprise が stale 化＝中核原理に影響 | ブロックが更新されないだけ |
| 操作 | append-only | **read-modify-write**（更新・削除） |
相乗りさせると、重い read-modify-write が surprise を待たせる。だから分ける。

## 「無限に増える」への答え（ユーザ懸念 2026-08-15）
**LLM に自由を与えない**。スロットは `memory/self_model.py` の閉じた一覧で、1スロット
最大2件。これはコード側の不変条件で、LLM が何を言っても超えられない。よって
system に載るブロックの長さは**証明可能に有界**。

## ユーザの好みを取り込まない守り（2段）
1. トリガが見るのは `speaker == "eve"` のターンだけ（`trigger.py`）。
2. LLM に渡すのも**イブ自身の発話だけ**。ユーザ発話は文脈として渡さない。
   実データ 2026-08-15: ユーザの「俺はおかかが好き」が2か月かけてイブの自己申告に
   転写されていた（`conversation_history.jsonl:32` → `:515`）。
   日本語の主語解析には頼らない（表記ゆれ・主語省略に弱い＝ユーザ指摘）。
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from ..memory.self_model import MAX_PER_SLOT, MAX_TOPICS, POLARITIES, SelfModelStore
from .trigger import SPEAKER_EVE, should_examine

logger = logging.getLogger(__name__)

SELF_MODEL_SYSTEM = (
    "あなたはAI VTuber「イブ」の自己形成係です。イブ**自身の発話**だけを読み、"
    "イブの好み・趣向として台帳に残すべきものを決めます。\n"
    "台帳の1行は「極性・対象・値」の3つです。\n"
    "- 極性は " + " / ".join(POLARITIES) + " のどちらか（これ以外は使えません）\n"
    "- 対象は自由に決めてよい（食べ物・ゲーム・ポケモン・色・音楽 など）\n"
    f"- 1つの対象につき値は最大{MAX_PER_SLOT}件。2件あるのは「どちらも好きで迷っている」状態です\n"
    f"- 台帳全体で対象は最大{MAX_TOPICS}個。溢れると古いものが自動で消えます\n"
    "\n"
    "出力は次の行だけを必要な数だけ。**変更が要らなければ何も出力しないでください**。\n"
    "set: 極性 / 対象 = 値\n"
    "remove: 極性 / 対象 = 値\n"
    "例) set: 好き / 食べ物 = おかかおにぎり\n"
    "\n"
    "規則:\n"
    "- 値は短い名詞句（例『おかかおにぎり』『藍色』）。文にしない。\n"
    "- **今の台帳に近い対象が既にあるなら、必ずそれを使う**。"
    "『食べ物』があるのに『食材』『料理』『好物』のような言い換えで新しい対象を作らない"
    "（枠を無駄に使うだけで、古い項目が消えます）。\n"
    "- **イブが自分の好みとして述べたものだけ**。ユーザの好みは絶対に入れない。\n"
    "- その場の話題に触れただけ・相手に合わせただけのものは入れない"
    "（例『チーズケーキ美味しそうだね』は感想であって好物ではない）。\n"
    "- イブが以前の好みを否定した / 違うと言われた時だけ remove を出す。\n"
    "- 迷ったら何も出力しない。台帳は少ない方がよい。"
)

# `set: 好き / 食べ物 = おかかおにぎり` を拾う。全角の区切りも許す
# （feedback のパーサと同じ方針＝小さな表記ゆれで全損させない）。
_OP = re.compile(
    r"^\s*(set|remove)\s*[:：]\s*(.+?)\s*[/／]\s*(.+?)\s*[=＝]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _g(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _content(resp) -> str:
    choices = _g(resp, "choices") or []
    if not choices:
        return ""
    msg = _g(choices[0], "message")
    return (_g(msg, "content") or "") if msg is not None else ""


def parse_ops(text: str) -> list[tuple[str, str, str, str]]:
    """LLM 出力 → [(op, 極性, 対象, 値)]。**raise しない**（解析できない行は捨てる）。"""
    out: list[tuple[str, str, str, str]] = []
    for m in _OP.finditer(text or ""):
        out.append((m.group(1).lower(), m.group(2).strip(),
                    m.group(3).strip(), m.group(4).strip()))
    return out


def render_eve_turns(turns) -> str:
    """LLM に渡すイブ自身の発話（**ユーザ発話は渡さない**＝取り込みの経路を作らない）。"""
    return "\n".join(
        f"- {t.text}" for t in turns
        if getattr(t, "speaker", None) == SPEAKER_EVE and getattr(t, "text", "")
    )


class SelfModelWorker:
    """single-flight サイドカー。既定 no-op（未 start）。"""

    def __init__(self, *, model_registry, store: SelfModelStore, cache) -> None:
        self._model = model_registry
        self._store = store
        self._cache = cache
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._watermark: str = ""

    # --- ライフサイクル（FeedbackWorker と同形）---

    def start(self) -> None:
        if self._task is None:
            self._stopping = False
            # stop() が wait を解くために立てた起床イベントは**立ったまま残る**
            # （set から cancel まで await を挟まないため）。消さないと再開直後に
            # 無駄な1周が走る（実測 2026-08-14: 沈黙監視が発火できず初回が11秒遅れた）。
            self._event.clear()
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stopping = True
        self._event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def trigger(self) -> None:
        """未処理ありを通知（O(1)・非ブロッキング＝応答の邪魔をしない）。"""
        self._event.set()

    # --- 本体 ---

    async def _run_forever(self) -> None:
        while not self._stopping:
            await self._event.wait()
            self._event.clear()
            if self._stopping:
                break
            try:
                await self._process_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # サイドカーの失敗で worker を殺さない
                logger.exception("自己形成で例外（継続）")

    async def _process_once(self) -> None:
        span = self._cache.turns_since(self._watermark) if self._cache is not None else []
        if not span:
            return
        # **watermark は先に進める**。合図が無いスパンを毎回読み直さないため。
        # 取りこぼしても実害が無い（好みは繰り返し語られる）＝FeedbackWorker と違い
        # ここは no-skip を要求しない。
        self._watermark = span[-1].stamp.iso
        if not should_examine(span):
            return
        body = render_eve_turns(span)
        if not body:
            return
        current = self._store.render() or "（まだ何も無い）"
        messages = [
            {"role": "system", "content": SELF_MODEL_SYSTEM},
            {"role": "user", "content": (
                f"# 今の台帳\n{current}\n\n# イブ自身の発話\n{body}"
            )},
        ]
        try:
            resp = await self._model.complete("self_model", messages)
        except Exception:
            logger.exception("自己形成 LLM 失敗（台帳は変えずに継続）")
            return
        ops = parse_ops(_content(resp))
        if not ops:
            return
        applied = 0
        for op, polarity, topic, value in ops:
            # 極性の閉じ性・項目数の上限・件数上限・長さは**すべて store 側のコードが持つ**。
            # ここで弾かないのは、判定を1か所に集めるため（LLM 出力を信用しない）。
            ok = (self._store.remove(polarity, topic, value) if op == "remove"
                  else self._store.set(polarity, topic, value))
            if ok:
                applied += 1
                logger.info("🪞 自己形成: %s %sな%s = %s", op, polarity, topic, value)
        if applied:
            logger.info("🪞 台帳を更新（%d件） → %s", applied, self._store.as_dict())
