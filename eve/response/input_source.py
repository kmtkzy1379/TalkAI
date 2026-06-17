"""入力源の抽象（STT の差込口）。

F2 は TextInputSource（テキスト発話を USER_UTTERANCE として queue に流す）のみ。
F2.5 で MicSttInputSource が同じインターフェースで置き換わる（フルループの継ぎ目）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..pipeline.stimulus import Stimulus, StimulusKind
from ..pipeline.stimulus_queue import StimulusQueue


class InputSource(ABC):
    def __init__(self, queue: StimulusQueue) -> None:
        self._queue = queue

    @abstractmethod
    async def start(self) -> None:
        """常駐の取り込みを開始する（mic 版で使用）。"""
        ...


class TextInputSource(InputSource):
    async def submit(self, text: str) -> None:
        await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))

    async def start(self) -> None:
        return  # テキスト版は外部から submit() を呼ぶだけ
