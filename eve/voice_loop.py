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
from .feedback import FeedbackLLM, FeedbackWorker, PredictionState
from .memory import ConversationCache, RagStore
from .memory.embed import make_embedder
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
        self.rag = RagStore(make_embedder())  # 長期記憶（連想想起）

        # F4 内分泌系: 各応答後に非同期で内省 → RAG 書込 + surprise + 直近フィードバック注入。
        self.prediction = PredictionState()  # loop 所有・単一書込（feedback worker のみ書く）
        self.feedback = FeedbackLLM(self.registry, rag_store=self.rag, prediction_state=self.prediction)
        self.feedback_worker = FeedbackWorker(self.feedback, self.cache)

        async def stream_fn(messages):
            async for delta in self.registry.stream("response", messages):
                yield delta

        self.orchestrator = ResponseOrchestrator(
            self.audio, stream_fn, self.tts.generate, ContextAssembler(),
            conversation_cache=self.cache, rag_store=self.rag,
            prediction_state=self.prediction,
            on_response_complete=self.feedback_worker.trigger,  # 正常完了で feedback を起こす
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
            await self.rag.warmup()  # 埋め込みモデルの初回ロードを先に済ませる
        except Exception as e:
            logger.warning("RAG ウォームアップ失敗（続行）: %s", e)
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
        await self.rag.initialize()  # 既存 RAG 記憶を復元 + 書き込み worker 起動
        # F4 起動時 catch-up: watermark を永続 RAG の最新 timestamp から復元し、
        # それより新しい復元会話（前回 feedback 途中で落ちた tail）を1回取り戻す。
        self.prediction.watermark = self.rag.latest_timestamp()
        self.feedback_worker.start()
        if self.cache.turns_since(self.prediction.watermark):
            self.feedback_worker.trigger()
        await self.warmup()
        self._tasks.append(asyncio.create_task(self.audio.play_worker()))
        self._tasks.append(asyncio.create_task(self.runner.run()))
        await self.input.start()  # mic + STT 消費タスクを起動
        logger.info("VoiceLoop 稼働。話しかけてください。")
        await asyncio.Event().wait()  # キャンセルされるまで稼働

    async def stop(self) -> None:
        self.input.stop()
        # feedback worker を先に drain/停止（進行中 add_chunk を rag.shutdown 前に flush 機会を与える。
        # 未完分は watermark 未前進なので次回起動の catch-up が回収＝記憶喪失を作らない）。
        try:
            await self.feedback_worker.stop()
        except Exception:
            pass
        for t in self._tasks:
            t.cancel()
        try:
            await self.cache.shutdown()  # 書き込みキューをドレイン（記録を取りこぼさない）
        except Exception:
            pass
        try:
            await self.rag.shutdown()
        except Exception:
            pass
        try:
            await self.tts.close()
        except Exception:
            pass
        self.player.close()
