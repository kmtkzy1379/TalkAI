"""笑い（S6・ユーザ裁定 2026-09-26。2026-09-25「笑いの出し方」の閾値 0.85 / 0.70 を 0.9 / 0.75 に更新）。

ユーザの話を**ストリーミング文字起こしの途中の文字**で見て、「、」「。」ごとに gpt-6-luna に
「聞き手のイブが思わず笑う場面か」を yes/no で聞き、logprobs を確率にする。
  ≥ immediate（0.9）… その場で笑う（話の途中でも。遮りうるので、つい笑ってしまう場面だけ）
  ≥ filler（0.75） … 話し終わりのフィラーに笑いを含ませる（フィラーの中で普通に1回笑う・2026-09-26
                     「ただ普通に笑うほうがよさそう」。`laugh.filler_style: tone` なら語を笑いの声色で）
- **応答は判定を待たない**（ユーザ裁定 2026-09-25「笑い判定ができなくても待たない」）。間に合わなければ笑わないだけ
- 途中の文字は**笑い判定専用**（応答は確定した文字から作る）
- 質問文は実測に使った `tools/laugh_judge_probe.py` と同じ（名大会話コーパス・AUC 0.69。
  閾値 0.9 で 笑った所の23%を拾い・笑わなかった所の7%で笑う＝話の途中）

判定の呼び出しは `ModelRegistry` の `laugh_judge` 役（既定 gpt-6-luna）。
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

QUESTION = ("相手の話のこの時点で、聞き手のイブが思わず笑う場面である"
            "（面白い・微笑ましい・呆れて笑う・気まずさを笑ってごまかす など。"
            "相手がつらい話・悲しい話をしているなど、笑うと不適切な場面では笑わない）")
BOUNDARY = "、。？！?!"

JudgeFn = Callable[[list, str], Awaitable[float]]


def laugh_messages(context_lines: list, said: str) -> list:
    return [{"role": "system", "content": "yes か no の1語だけで答える。"},
            {"role": "user", "content": "直前の会話:\n" + ("\n".join(context_lines) or "（なし）")
             + f"\n\n相手が今話している途中の内容:\n{said}\n\n"
             + f"判定: {QUESTION}。そうなら yes、違うなら no。"}]


def yes_probability(resp) -> float:
    """1語目の top_logprobs から yes の確率（yes/はい ÷ (yes/はい + no/いいえ)）。取れなければ 0。"""
    try:
        top = resp.choices[0].logprobs.content[0].top_logprobs
    except Exception:
        return 0.0
    py = pn = 0.0
    for x in top or ():
        tok = (getattr(x, "token", None) if not isinstance(x, dict) else x.get("token")) or ""
        lp = getattr(x, "logprob", None) if not isinstance(x, dict) else x.get("logprob")
        if lp is None:
            continue
        t = tok.strip().lower()
        if t in ("yes", "はい"):
            py += math.exp(lp)
        elif t in ("no", "いいえ"):
            pn += math.exp(lp)
    return py / (py + pn) if (py + pn) > 0 else 0.0


def make_judge_fn(registry) -> JudgeFn:
    async def judge(context_lines: list, said: str) -> float:
        # reasoning_effort=none は実測と同じ条件（約0.6〜0.8秒）。litellm は gpt-6-luna がこの引数を
        # 受けることを知らず弾く（UnsupportedParamsError・実測 2026-09-26）ので、通すことを明示する。
        r = await registry.complete(
            "laugh_judge", laugh_messages(context_lines, said),
            logprobs=True, top_logprobs=5, reasoning_effort="none", max_completion_tokens=5,
            allowed_openai_params=["reasoning_effort"])
        return yes_probability(r)

    return judge


class LaughJudge:
    """「、」「。」ごとの途中の文字を受け取り、判定して結果を知らせる（loop 上・判定は並行・上限あり）。

    on_immediate(seg, p) … 確率 ≥ immediate（その場で笑う）
    on_filler(seg, p)    … 確率 ≥ filler（話し終わりのフィラーに笑いを含ませる）
    seg = マイクの区間の番号（どの発話への判定か）。同じ区間の同じ文字は2回判定しない。
    """

    def __init__(self, *, judge_fn: JudgeFn, context_fn: Callable[[], list],
                 on_immediate: Callable[[int, float], None], on_filler: Callable[[int, float], None],
                 immediate: float = 0.9, filler: float = 0.75, max_inflight: int = 3) -> None:
        self._judge = judge_fn
        self._context = context_fn
        self._on_immediate = on_immediate
        self._on_filler = on_filler
        self.immediate = float(immediate)
        self.filler = float(filler)
        self._max = max(1, int(max_inflight))
        self._tasks: set = set()
        self._seen: dict = {}  # seg → 判定した文字の集合（同じ文字を2回聞かない）

    def on_phrase(self, seg: int, text: str) -> None:
        """途中の文字（その発話の頭からここまで）。「、」「。」が来た時と、確定した時に呼ばれる。"""
        text = (text or "").strip()
        if not text:
            return
        seen = self._seen.setdefault(seg, set())
        if text in seen:
            return
        if len(self._seen) > 32:  # 古い区間の記録は捨てる
            for k in sorted(self._seen)[:-16]:
                self._seen.pop(k, None)
        if len(self._tasks) >= self._max:
            logger.debug("笑い判定が混んでいるので見送り（#%d）: %.20s", seg, text)
            return
        seen.add(text)
        try:
            ctx = list(self._context() or [])
        except Exception:
            logger.exception("笑い判定の会話の取得に失敗（会話なしで判定）")
            ctx = []
        t = asyncio.get_running_loop().create_task(self._run(seg, text, ctx))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _run(self, seg: int, text: str, ctx: list) -> None:
        t0 = time.monotonic()
        try:
            p = float(await self._judge(ctx, text))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 判定できなければ笑わないだけ（本流は止めない）
            logger.warning("笑い判定に失敗（笑わない）: %s", e)
            return
        act = ("その場で笑う" if p >= self.immediate else
               "フィラーに笑い" if p >= self.filler else "笑わない")
        logger.info("😄 笑い判定 #%d: %.2f（%s・%.2f秒）「%s」", seg, p, act, time.monotonic() - t0, text[-30:])
        for fn, th in ((self._on_immediate, self.immediate), (self._on_filler, self.filler)):
            if p >= th:
                try:
                    fn(seg, p)
                except Exception:
                    logger.exception("笑いの合図で例外（無視）")

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            try:
                await t
            except BaseException:
                pass
        self._tasks.clear()
