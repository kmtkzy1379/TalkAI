"""応答オーケストレータの口（Protocol）と F1 スタブ + drain ループ。

実 ResponseOrchestrator（F2）は同じ `handle()` で差し替わる（将来互換）。
F1 はスタブで「刺激→文→音声キュー」の配線だけを通す。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Protocol, runtime_checkable

from .audio_play_queue import AudioPlayQueue
from .stimulus import Stimulus, StimulusKind
from .stimulus_queue import StimulusQueue

logger = logging.getLogger(__name__)


@runtime_checkable
class Orchestrator(Protocol):
    async def handle(self, stimulus: Stimulus) -> None: ...


class StubOrchestrator:
    """F1 用スタブ。刺激を N 個のフェイク文に変換し、現世代の seq を予約して投入する。

    work_delay_s で「処理中」を再現できる（busy 中は溜まることの確認に使う）。
    """

    def __init__(self, audio: AudioPlayQueue, sentences: int = 1, work_delay_s: float = 0.0) -> None:
        self._audio = audio
        self._sentences = sentences
        self._work_delay = work_delay_s
        self.handled: list[Stimulus] = []

    async def handle(self, stimulus: Stimulus) -> None:
        if self._work_delay:
            await asyncio.sleep(self._work_delay)
        gen = self._audio.current_generation()
        for _ in range(self._sentences):
            seq = self._audio.reserve_seq(gen)  # 生成順に予約（再生は seq 昇順）
            self._audio.enqueue(gen, seq, f"<{stimulus.kind.name}:{seq}>")
        self.handled.append(stimulus)


class PipelineRunner:
    """StimulusQueue から1件ずつ drain して orchestrator に渡す単一 consumer。"""

    def __init__(
        self,
        queue: StimulusQueue,
        orchestrator: Orchestrator,
        audio: AudioPlayQueue,
        max_consecutive_errors: int = 3,
    ) -> None:
        self._queue = queue
        self._orch = orchestrator
        self._audio = audio
        self._max_consecutive_errors = max_consecutive_errors
        self.processed = 0
        self._active: Optional[asyncio.Task] = None

    async def _next(self) -> Stimulus:
        s = await self._queue.get()
        if s.kind == StimulusKind.USER_UTTERANCE:
            # coalesce: 入力前に溜まったユーザ発話を1つに繋げる（pileup防止・「繋げて入れる」）
            extra = self._queue.drain_user_texts()
            if extra:
                merged = " ".join([str(s.payload), *extra])
                s = Stimulus(StimulusKind.USER_UTTERANCE, payload=merged)
        return s

    async def run_once(self) -> Stimulus:
        s = await self._next()
        # handle はキャンセル可能なタスクで走らせる（barge-in で中断するため）
        self._active = asyncio.create_task(self._orch.handle(s))
        await asyncio.wait({self._active})  # 完了 or barge-in キャンセルで戻る
        task = self._active
        self._active = None
        self.processed += 1
        if not task.cancelled() and task.exception() is not None:
            raise task.exception()  # handle 内未捕捉の例外 → A2 循環ブレーカへ
        return s

    def interrupt(self) -> None:
        """barge-in: 進行中の応答生成を即キャンセル（Eve が止まる）。発話開始の瞬間に呼ぶ。"""
        if self._active and not self._active.done():
            self._active.cancel()

    def is_busy(self) -> bool:
        """応答生成が進行中か（F5 沈黙監視が応答中に発火しないためのガード）。"""
        return self._active is not None and not self._active.done()

    async def run(self) -> None:
        # A2: 1刺激の失敗で単一 consumer を殺さない。例外は traceback 付きでログし継続。
        # 連続失敗（想定外が頻発）は無理に回さず停止＝循環ブレーカ。
        consecutive = 0
        while True:
            try:
                await self.run_once()
                consecutive = 0
            except asyncio.CancelledError:
                if self._active and not self._active.done():
                    self._active.cancel()  # シャットダウン時は進行中も片付ける
                break
            except Exception:
                consecutive += 1
                logger.exception("刺激処理に失敗（%d 回連続）", consecutive)
                if consecutive >= self._max_consecutive_errors:
                    logger.error("連続失敗が上限(%d)に達したため停止", self._max_consecutive_errors)
                    raise
