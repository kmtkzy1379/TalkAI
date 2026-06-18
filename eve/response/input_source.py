"""入力源の抽象（STT の差込口）。

- TextInputSource: テキスト発話を USER_UTTERANCE として投入（テスト/タイプ入力）。
- MicSttInputSource: mic → VAD → STT → USER_UTTERANCE。発話開始で barge-in（Eveが譲る）。

いずれも asyncio ループ内で動く（mic のブロッキング read のみ executor）。よって
StimulusQueue.put を直接 await でき、別スレッド用の thread-safe put は不要。
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from ..pipeline.audio_play_queue import AudioPlayQueue
from ..pipeline.stimulus import Stimulus, StimulusKind
from ..pipeline.stimulus_queue import StimulusQueue
from ..stt import Stt

logger = logging.getLogger(__name__)


class InputSource(ABC):
    def __init__(self, queue: StimulusQueue) -> None:
        self._queue = queue

    @abstractmethod
    async def start(self) -> None:
        """取り込みを開始する（mic 版はタスクを起動して即 return）。"""
        ...


class TextInputSource(InputSource):
    async def submit(self, text: str) -> None:
        await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))

    async def start(self) -> None:
        return  # テキスト版は外部から submit() を呼ぶだけ


class MicSttInputSource(InputSource):
    """mic → VAD → STT → 刺激。発話開始で AudioPlayQueue.interrupt()（barge-in＝Eveが譲る）。"""

    def __init__(self, queue: StimulusQueue, stt: Stt, audio: AudioPlayQueue) -> None:
        super().__init__(queue)
        self._stt = stt
        self._audio = audio
        self._ai = None
        self._task = None

    async def start(self) -> None:
        from ..audio_input import AudioInput  # torch/pyaudio を要する→遅延 import

        # 発話開始の瞬間に barge-in（STT 完了を待たず即停止＝自然な割り込み）
        self._ai = AudioInput(callback_on_speech_start=self._audio.interrupt)
        self._ai.start()
        self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        assert self._ai is not None
        while True:
            pcm = await self._ai.get_audio()
            try:
                text = await self._stt.transcribe(pcm)
            except Exception:  # STT 失敗はこの発話を捨てて継続（起こりうる）
                logger.exception("STT 失敗（この発話を捨てて継続）")
                continue
            if text:
                await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._ai:
            self._ai.stop()
