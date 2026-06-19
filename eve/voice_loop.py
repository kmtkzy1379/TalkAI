"""VoiceLoop — フルループ組み立て（mic→STT→応答LLM→TTS→再生）。

F0–F2.5 の部品を結線する本体。UI(F6) はこれを包む想定。レイテンシ重視:
- 起動時ウォームアップで cold-start を消す。
- 全段 asyncio タスク（ブロッキングI/Oのみ executor）。
- barge-in は MicSttInputSource が発話開始で audio.interrupt()。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .context_assembler import ContextAssembler
from .memory import ConversationCache
from .model_registry import ModelRegistry
from .pipeline.audio_play_queue import AudioPlayQueue
from .pipeline.orchestrator import PipelineRunner
from .pipeline.stimulus_queue import StimulusQueue
from .response.input_source import MicSttInputSource
from .response.orchestrator import ResponseOrchestrator
from .response.player import RealAudioPlayer
from .response.tts import VoicevoxTTS
from .stt import make_stt

logger = logging.getLogger(__name__)


class VoiceLoop:
    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.player = RealAudioPlayer()
        self.audio = AudioPlayQueue(play_fn=self.player.play_fn)
        self.queue = StimulusQueue()
        self.tts = VoicevoxTTS()
        self.stt = make_stt()
        self.cache = ConversationCache()  # 短期記憶（会話ログ・直近注入・実発話記録）

        async def stream_fn(messages):
            async for delta in self.registry.stream("response", messages):
                yield delta

        self.orchestrator = ResponseOrchestrator(
            self.audio, stream_fn, self.tts.generate, ContextAssembler(),
            conversation_cache=self.cache,
        )
        self.runner = PipelineRunner(self.queue, self.orchestrator, self.audio)
        self.input = MicSttInputSource(self.queue, self.stt, on_speech_start=self._barge_in)
        self._tasks: list[asyncio.Task] = []

    def _barge_in(self) -> None:
        """発話開始の瞬間: 音声停止＋進行中応答キャンセル（Eve が即譲る）。"""
        logger.info("⏸ 発話検知（割り込み）")
        self.audio.interrupt()
        self.runner.interrupt()

    async def warmup(self) -> None:
        """STT/LLM/TTS を1回空打ちして cold-start（初回の数秒遅延）を消す。"""
        logger.info("ウォームアップ開始")
        await self.stt.warmup()
        try:
            await self.registry.complete("response", [{"role": "user", "content": "hi"}], max_tokens=1)
        except Exception as e:
            logger.warning("LLM ウォームアップ失敗（続行）: %s", e)
        try:
            await self.tts.generate("。")
        except Exception as e:
            logger.warning("TTS ウォームアップ失敗（続行）: %s", e)
        logger.info("ウォームアップ完了")

    async def run(self) -> None:
        await self.cache.initialize()  # 既存ログ復元 + 書き込み worker 起動
        await self.warmup()
        self._tasks.append(asyncio.create_task(self.audio.play_worker()))
        self._tasks.append(asyncio.create_task(self.runner.run()))
        await self.input.start()  # mic + STT 消費タスクを起動
        logger.info("VoiceLoop 稼働。話しかけてください。")
        await asyncio.Event().wait()  # キャンセルされるまで稼働

    async def stop(self) -> None:
        self.input.stop()
        for t in self._tasks:
            t.cancel()
        try:
            await self.cache.shutdown()  # 書き込みキューをドレイン（記録を取りこぼさない）
        except Exception:
            pass
        try:
            await self.tts.close()
        except Exception:
            pass
        self.player.close()
