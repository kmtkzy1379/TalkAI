"""応答オーケストレータの口（Protocol）と F1 スタブ + drain ループ。

実 ResponseOrchestrator（F2）は同じ `handle()` で差し替わる（将来互換）。
F1 はスタブで「刺激→文→音声キュー」の配線だけを通す。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

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

    async def run_once(self) -> Stimulus:
        s = await self._queue.get()
        # barge-in 配線: ユーザ発話は言いかけを止める（世代を進める）。
        # 実トリガ（発話中検知）は F2/F4 で接続。ここは機構の発火点。
        if s.kind == StimulusKind.USER_UTTERANCE:
            self._audio.bump_generation()
        await self._orch.handle(s)
        self.processed += 1
        return s

    async def run(self) -> None:
        # A2: 1刺激の失敗で単一 consumer を殺さない。例外は traceback 付きでログし継続。
        # ただし連続失敗（想定外が頻発）は無理に回さず停止＝循環ブレーカ。
        consecutive = 0
        while True:
            try:
                await self.run_once()
                consecutive = 0
            except asyncio.CancelledError:
                break
            except Exception:
                consecutive += 1
                logger.exception("刺激処理に失敗（%d 回連続）", consecutive)
                if consecutive >= self._max_consecutive_errors:
                    logger.error("連続失敗が上限(%d)に達したため停止", self._max_consecutive_errors)
                    raise
