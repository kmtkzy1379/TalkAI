"""フィラーの1段目を場面に寄せる（S8・ユーザ指定 2026-09-27・表は fillers.yaml の mood）。

ユーザ:「今はフィラーランダムにしてるがjevで最適化とかできない？例えばお系は驚く、う系は迷う、悩むなどで多くなります」
「通常…/悩む…/驚く、お/悲しむ、う、無言/不満、えー、無言」「できればほとんど通常にさせて強いときのみその他感情にしたい」
「閾値0.9でいい。ただ1段目はそれでいいけど2段目以降は通常通りランダムでいい」「通常のえーは低め、不満のえーは高め」

- ユーザの話の「、」「。」ごとに（笑い・感情の判定と同じ合図）、Jev（OpenRouter `typesafe/jev-1.13`・1回 約0.25秒・
  約$0.00002）へ、聞き手の反応の種類（通常・悩む・驚く・悲しむ・不満）を確率つきで聞く
- フィラーを鳴らし始める時（`mood(seg)`）、その区間の最新の判定で「通常」以外の種類の確率が threshold（0.9）以上ならその種類。
  それ以外・判定が間に合わない・失敗は「通常」。1段目の選び方は `bank.draw_first`
- 実測（`tools/filler_select_probe.py`・名大会話コーパス＋JVNV・2026-09-27）: 0.9 で雑談の約2.4%だけ通常以外（驚く 1.9%・
  悲しむ 0.5%）/ はっきり悲しい話 30/30・驚き 7/30・怒り嫌悪 13/60・喜びで誤って切り替わる 2/30。
  「イブの今の感情」を渡すと強く通常に寄る（驚き 27/30 → 3/30）ので、既定は渡さない（fillers.yaml の with_eve_emotion）
- 質問文と各種類の説明は実測と同じ（説明は fillers.yaml の desc）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

from ..clock import now_mono
from .bank import MOOD_DEFAULT

logger = logging.getLogger(__name__)

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
# 実測と同じ質問文（tools/filler_select_probe.py の CAT_INSTR）
INSTR = "相手の発話を聞いた直後の、聞き手の反応（話し始める前の気持ち）"

JudgeFn = Callable[[list, str, Optional[str]], Awaitable[dict]]


class JevClient:
    """Jev（OpenRouter の decisions API）に1問だけ聞く。接続は使い回す。失敗は例外（呼び出し側が「通常」にする）。"""

    def __init__(self, api_key: str, categories: dict, *, timeout_sec: float = 5.0,
                 url: str = JEV_URL, model: str = JEV_MODEL) -> None:
        self._key = api_key
        self._criteria = dict(categories)
        self._timeout = timeout_sec
        self._url = url
        self._model = model
        self._session = None

    def body(self, context_lines: list, said: str, eve_emotion: Optional[str]) -> dict:
        state: dict = {}
        if eve_emotion:
            state["イブの今の感情"] = eve_emotion
        if context_lines:
            state["直前の会話"] = list(context_lines)
        state["相手の発話"] = said
        return {"model": self._model, "state": state,
                "questions": {"filler": {"type": "choice", "instructions": INSTR, "criteria": self._criteria}}}

    async def ask(self, context_lines: list, said: str, eve_emotion: Optional[str] = None) -> dict:
        """種類 → 確率。"""
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        async with self._session.post(
                self._url, data=json.dumps(self.body(context_lines, said, eve_emotion), ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}) as r:
            if r.status != 200:
                raise RuntimeError(f"Jev HTTP {r.status}: {(await r.text())[:120]}")
            d = await r.json()
        a = (d.get("answers") or {}).get("filler") or {}
        probs = a.get("probabilities") or {}
        return {str(k): float(v) for k, v in probs.items()}

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class MoodJudge:
    """「、」「。」ごとの途中の文字を受け取り、Jev の判定を区間ごとに持つ（loop 上・判定は並行・上限あり）。"""

    def __init__(self, *, judge_fn: JudgeFn, context_fn: Callable[[], list], categories: list,
                 threshold: float, emotion_fn: Optional[Callable[[], Optional[str]]] = None,
                 max_inflight: int = 3) -> None:
        self._judge = judge_fn
        self._context = context_fn
        self._cats = [c for c in categories if c != MOOD_DEFAULT]
        self.threshold = float(threshold)
        self._emotion = emotion_fn
        self._max = max(1, int(max_inflight))
        self._tasks: set = set()
        self._seen: dict = {}
        self._latest: dict = {}  # seg → (判定に使った文字の時点, 種類 → 確率)

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
            logger.debug("場面の判定が混んでいるので見送り（#%d）: %.20s", seg, text)
            return
        seen.add(text)
        try:
            ctx = list(self._context() or [])
        except Exception:
            logger.exception("場面の判定の会話の取得に失敗（会話なしで判定）")
            ctx = []
        emo = None
        if self._emotion is not None:
            try:
                emo = self._emotion()
            except Exception:
                emo = None
        t = asyncio.get_running_loop().create_task(self._run(seg, text, ctx, emo, now_mono()))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _run(self, seg: int, text: str, ctx: list, emo: Optional[str], as_of: float) -> None:
        t0 = time.monotonic()
        try:
            probs = await self._judge(ctx, text, emo)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 判定できなければ「通常」のまま（本流は止めない）
            logger.warning("場面の判定に失敗（通常のまま）: %s", e)
            return
        cur = self._latest.get(seg)
        if cur is not None and cur[0] > as_of:
            return  # 後から始めた判定が先に届いている
        self._latest[seg] = (as_of, probs)
        if len(self._latest) > 32:
            for k in sorted(self._latest)[:-16]:
                self._latest.pop(k, None)
        mood = self._pick(probs)
        top = max(self._cats, key=lambda c: probs.get(c, 0.0), default=None)
        logger.info("🎭 場面の判定 #%d: %s（%s %.2f・%.2f秒）「%s」", seg, mood, top,
                    probs.get(top, 0.0) if top else 0.0, time.monotonic() - t0, text[-30:])

    def _pick(self, probs: dict) -> str:
        best = max(self._cats, key=lambda c: probs.get(c, 0.0), default=None)
        if best is not None and probs.get(best, 0.0) >= self.threshold:
            return best
        return MOOD_DEFAULT

    def mood(self, seg: int) -> str:
        """その区間のフィラーの1段目の種類（強い判定が無ければ「通常」）。"""
        cur = self._latest.get(seg)
        return MOOD_DEFAULT if cur is None else self._pick(cur[1])

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            try:
                await t
            except BaseException:
                pass
        self._tasks.clear()
