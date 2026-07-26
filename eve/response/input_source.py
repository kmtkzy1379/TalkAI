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
    """mic → VAD → STT → 刺激。発話開始で on_speech_start（barge-in＝Eveが即譲る）を呼ぶ。

    on_speech_start には「音声停止＋進行中応答キャンセル」をまとめた barge-in を渡す
    （VoiceLoop が audio.interrupt + runner.interrupt を束ねて渡す）。
    """

    def __init__(self, queue: StimulusQueue, stt: Stt, on_speech_start=None, on_utterance=None,
                 on_stt_start=None, on_stt_end=None) -> None:
        super().__init__(queue)
        self._stt = stt
        self._on_speech_start = on_speech_start
        # 発話セグメント到着（＝ユーザが話し終えた）時に呼ぶ。F5 の沈黙時計リセット用。
        self._on_utterance = on_utterance
        # J-2 ③-A: STT 待ち窓（発話終了〜テキスト投入）の可視化フック。この窓は user_speaking=False
        # かつ沈黙時計リセット済みで「静か」に見え、自発発話が差し込まれてユーザ応答を遅らせた
        # （E2E S8 実測）。start=transcribe 直前 / end=投入完了（失敗・空認識含む）後に呼ぶ。
        self._on_stt_start = on_stt_start
        self._on_stt_end = on_stt_end
        self._ai = None
        self._task = None

    async def start(self) -> None:
        from ..audio_input import AudioInput  # torch/pyaudio を要する→遅延 import

        # 発話開始の瞬間に barge-in（STT 完了を待たず即停止＝自然な割り込み）
        self._ai = AudioInput(callback_on_speech_start=self._on_speech_start)
        self._ai.start()
        self._task = asyncio.create_task(self._consume())

    def _hook(self, fn, name: str) -> None:
        if fn is None:
            return
        try:
            fn()
        except Exception:
            logger.exception("%s フックで例外（無視して継続）", name)

    async def _consume(self) -> None:
        assert self._ai is not None
        while True:
            pcm = await self._ai.get_audio()
            # 発話セグメントが届いた＝ユーザは話し終えた（F5: user_speaking 解除 + 沈黙時計リセット）。
            self._hook(self._on_utterance, "on_utterance")
            # J-2 ③-A: STT 待ち窓を開く（この間は自発発話を差し込まない）。
            self._hook(self._on_stt_start, "on_stt_start")
            text = None
            try:
                text = await self._stt.transcribe(pcm)
            except Exception:  # STT 失敗はこの発話を捨てて継続（起こりうる）
                logger.exception("STT 失敗（この発話を捨てて継続）")
            if text:
                logger.info("🧑 %s", text)
                await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))
            # J-2 ③-A: 窓を閉じるのは**投入完了後**（put 前に閉じると put の await 中に
            # 自発発話が滑り込む微小窓が残る）。失敗/空認識でも必ず閉じる（固着防止）。
            self._hook(self._on_stt_end, "on_stt_end")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._ai:
            self._ai.stop()
