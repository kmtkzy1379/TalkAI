"""FeedbackWorker — サイドカー本体（single-flight + watermark/span）。

PIPELINE_DESIGN.md §9.4 サイドカー契約の実装。応答完了点から `trigger()` で起こされ、
会話の [watermark, latest] スパンを **値コピー**で読み、FeedbackLLM.run に渡す。run が成功
（RAG 保存済）した時だけ watermark を最新へ前進させる。

不変条件:
- **single-flight**: 長寿命タスク1本が直列処理。per-turn create_task を作らない（PredictionState の
  同時 read-modify-write を構造的に排除）。
- **トリガは O(1) 非ブロッキング**: `trigger()` は Event.set() のみ（応答経路をブロックしない）。
- **no-skip**: busy 中に増えたターンも次サイクルが [watermark, latest] でまとめてカバー
  （ユーザ要件「フィードバックしてない会話＝記憶喪失を作らない」）。watermark は run 成功時のみ前進。
- **値コピー**: live deque/Turn 参照を worker に渡さない（将来の Turn 変更に対する防御）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..context_assembler import OMITTED_SPEAKER, Turn
from .feedback_llm import FeedbackLLM

logger = logging.getLogger(__name__)


class FeedbackWorker:
    def __init__(self, feedback: FeedbackLLM, cache) -> None:
        self._fb = feedback
        self._cache = cache
        self._state = feedback.state
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._idle = asyncio.Event()
        self._idle.set()  # 初期は idle（処理中でない）

    # --- ライフサイクル ---------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self, drain_timeout: float = 2.0) -> None:
        """新規処理を止め、進行中 run を短く drain してから cancel（best-effort）。"""
        self._stopping = True
        self._event.set()  # wait を解く
        if not self._idle.is_set():
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                pass  # 期限超過なら諦めて cancel（未保存分は次回起動 catch-up が回収）
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # --- トリガ（応答完了点から・O(1)）-----------------------------------

    def trigger(self) -> None:
        """未処理ありを通知（非ブロッキング）。応答クリティカルパスを汚さない。"""
        self._event.set()

    # --- worker 本体 ------------------------------------------------------

    async def _run_forever(self) -> None:
        while not self._stopping:
            await self._event.wait()
            self._event.clear()
            if self._stopping:
                break
            try:
                # 1トリガで「今ある未処理スパン」を処理。処理中に増えた分は
                # trigger() が再び Event を立てるので次イテレーションで拾う（no-skip）。
                await self._process_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # サイドカーの失敗で worker を殺さない
                logger.exception("FeedbackWorker 処理で例外（継続）")

    async def _process_once(self) -> None:
        span = self._cache.turns_since(self._state.watermark)
        # 値コピー（live deque を渡さない）+ 省略マーカ除去
        snapshot = [
            Turn(t.speaker, t.text, t.stamp) for t in span if t.speaker != OMITTED_SPEAKER
        ]
        if not snapshot:
            return
        new_watermark = snapshot[-1].stamp.iso  # 読んだ時点の最新（以後の新着は次サイクル）
        self._idle.clear()
        try:
            result = await self._fb.run(snapshot)
        finally:
            self._idle.set()
        if result is not None:
            # RAG 保存成功 → watermark 前進（no-skip・記憶喪失なし）
            self._state.watermark = new_watermark
        # result is None（失敗）→ watermark 据え置き。次トリガで同スパンを再カバー。
