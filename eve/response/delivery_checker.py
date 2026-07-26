"""J-2 P1-1 — 配達確認サイドカー（CancelResolver の近クローン・別コルーチン）。

実機事故(2026-07-18 E2E S2): 検索完了の報告ターンが、直前の未応答ユーザ発話への返事を
優先し、報告内容(payload.content)を一度も発話せずに正常完了した（barge-in ではないため
既存の `_maybe_redeliver`(世代変化/CancelledError 専用)は対象外＝検出装置が無かった）。

ここでは barge-in なしで完了した CALLFUNCTION_RESULT ターンについて、実際の発話が
報告内容を伝えたか、安価な LLM に3択で確認させる:
  - delivered: 内容を伝えた（言い換え/要約でもよい）
  - withheld : ユーザの取消/中止等により意図的に控えた（例: 「わかった、やめておくね」）
  - forgotten: 伝えておらず、控える理由も無い（別の話への返事だけ等）
forgotten の時だけ、既存の再配達経路(redeliver_fn)をそのまま再利用して再送する
（attempts 上限・dedup_key は redeliver_fn 側/StimulusQueue 側の既存契約に委ねる）。

判定不能（LLM失敗/出力不明瞭）は **delivered 扱い**（安全側＝過剰な再配達をしない。
v1 教訓: 不明瞭を speak/redeliver 側に倒すと暴走する）。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind

logger = logging.getLogger(__name__)

DELIVERY_CHECK_SYSTEM = (
    "あなたはAI VTuber「イブ」の報告配達の確認係です。イブには機能実行結果（報告すべき内容）が渡され、"
    "直後にイブが実際に発話しました。次の3択で判定し、1語だけ返してください:\n"
    "- delivered: 発話が報告内容の要点をユーザに伝えている（言い換え・要約でもよい）\n"
    "- withheld: ユーザが直前に不要・中止と言ったため、報告を意図的に控えた（その旨の発話がある）\n"
    "- forgotten: 報告内容を伝えておらず、控える理由もない（別の話への返事だけ等）\n"
    "出力は delivered / withheld / forgotten のどれか1語のみ。"
)

_VERDICTS = ("delivered", "withheld", "forgotten")


def parse_delivery_verdict(text: Optional[str]) -> str:
    """判定LLM出力→verdict。解析失敗/不明瞭は 'delivered'（安全側・過剰な再配達をしない）。"""
    t = (text or "").strip().lower()
    for v in _VERDICTS:
        if v in t:
            return v
    return "delivered"


def _g(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _content(resp) -> str:
    choices = _g(resp, "choices") or []
    if not choices:
        return ""
    msg = _g(choices[0], "message")
    return (_g(msg, "content") or "") if msg is not None else ""


@dataclass
class DeliveryCheckRequest:
    stimulus: Stimulus
    recent_text: str
    spoken_text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class DeliveryChecker:
    """single-flight サイドカー（FeedbackWorker/CancelResolver と同形）。既定 no-op（未 start）。"""

    def __init__(self, *, model_registry, redeliver_fn, max_attempts: int) -> None:
        self._model = model_registry
        self._redeliver_fn = redeliver_fn
        self._max_attempts = max_attempts
        self._inbox: "asyncio.Queue[DeliveryCheckRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # --- ライフサイクル（dispatcher/cancel_resolver と同形）---

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self, drain_timeout: float = 3.0) -> None:
        self._stopping = True
        try:
            await asyncio.wait_for(self._inbox.join(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def submit(self, stimulus: Stimulus, *, recent_text: str, spoken_text: str) -> None:
        """判定要求を積むだけ（O(1)・非ブロッキング・orchestrator.handle をブロックしない）。"""
        self._inbox.put_nowait(DeliveryCheckRequest(stimulus, recent_text, spoken_text))

    async def _run(self) -> None:
        while not self._stopping:
            req = await self._inbox.get()
            try:
                await self._handle_one(req)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("配達確認で例外（継続）")
            finally:
                self._inbox.task_done()

    async def _handle_one(self, req: DeliveryCheckRequest) -> None:
        payload = req.stimulus.payload
        if not isinstance(payload, CallFunctionResult):
            return
        messages = [
            {"role": "system", "content": DELIVERY_CHECK_SYSTEM},
            {"role": "user", "content": (
                f"# 直近の会話\n{req.recent_text or '（なし）'}\n\n"
                f"# 機能実行結果（報告すべき内容）\n{payload.content}\n\n"
                f"# イブの実際の発話\n{req.spoken_text or '（発話なし）'}"
            )},
        ]
        try:
            resp = await self._model.complete("delivery_check", messages)
            verdict = parse_delivery_verdict(_content(resp))
        except Exception:
            logger.exception("配達確認 LLM 失敗（再配達しない・安全側）")
            return
        logger.info("📮 配達確認: %s (dedup=%s)", verdict, req.stimulus.dedup_key)
        if verdict != "forgotten":
            return
        if payload.attempts >= self._max_attempts:
            logger.warning("⚠ 配達確認で forgotten だが再配達上限のため断念: %.40s", payload.content)
            return
        if self._redeliver_fn is None:
            return
        retry = Stimulus(
            kind=StimulusKind.CALLFUNCTION_RESULT,
            payload=CallFunctionResult(payload.function_name, payload.content, payload.ok, payload.attempts + 1),
            dedup_key=req.stimulus.dedup_key,
        )
        # ターンは既に完了済み（audio.join 後）＝put 直前に再考すべきレースは無い
        # （barge-in 用の delivered() クロージャと違い、ここでの spoken は既に確定している）。
        self._redeliver_fn(retry, lambda: False)
