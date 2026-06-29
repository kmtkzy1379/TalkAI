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

from .config import Config
from .context_assembler import ContextAssembler
from .feedback import FeedbackLLM, FeedbackWorker, PredictionState
from .memory import ConversationCache, RagStore
from .memory.embed import make_embedder
from .model_registry import ModelRegistry
from .pipeline.audio_play_queue import AudioPlayQueue
from .pipeline.orchestrator import PipelineRunner
from .pipeline.stimulus import StimulusKind
from .pipeline.stimulus_queue import StimulusQueue
from .capability import CapabilityRegistry
from .response.function_dispatcher import FunctionDispatcher
from .task import ReconcileTimer, TaskAgent, TaskExecutor, TaskStore, register_task_capabilities
from .response.input_source import MicSttInputSource
from .response.orchestrator import ResponseOrchestrator
from .response.player import RealAudioPlayer
from .response.style import SPEECH_STYLE
from .response.tts import VoicevoxTTS
from .speech import SilenceMonitor, SpeechDecider, SpeechState, make_decide_fn
from .stt import make_stt
from .vlm import ChangeDetector, VisionState, VlmWorker, make_narrate_fn
from .vlm.capture import ScreenCapture
from .vlm.capture_thread import CaptureThread

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
        self.prediction = PredictionState()  # loop 所有・単一書込（feedback/VLM が書く）
        self.feedback = FeedbackLLM(self.registry, rag_store=self.rag, prediction_state=self.prediction)
        self.feedback_worker = FeedbackWorker(self.feedback, self.cache)

        # F6 画面認識: loop 所有 VisionState（ring=全キャプチャ・latest_vision）。
        self.vision_state = VisionState(ring_max=Config.VLM_RING_MAX)

        # F5 発話判定: 5秒沈黙で should_speak → True で AUTONOMOUS_SPEECH を投入。
        self.speech_state = SpeechState()  # loop 所有 ephemeral（沈黙計測/発話判定ログ）
        self.speech_decider = SpeechDecider(
            state=self.speech_state, cache=self.cache, rag=self.rag,
            prediction_state=self.prediction, queue=self.queue,
            decide_fn=make_decide_fn(self.registry),  # role=speech_decide
            vision_state=self.vision_state,  # F6: 発話判定に直近画面を入れる
        )

        # F6 VLM サイドカー: 専用スレッドが capture→gate→on_frame、single-flight worker が
        # 1回の multi-frame 呼び出しで実況化→latest_vision/note_vlm_surprise→ガード付き発話トリガ。
        self.vlm_worker = VlmWorker(
            vision_state=self.vision_state, prediction_state=self.prediction,
            narrate_fn=make_narrate_fn(self.registry),  # role=vlm_leaf
            speak_trigger=self.speech_decider.trigger,
            speak_guard=self._vision_can_speak,  # A5/Q4: busy/ユーザ発話中/decider 処理中なら叩かない
            frames_per_call=Config.VLM_MAX_FRAMES_PER_CALL,
            min_interval_sec=Config.VLM_MIN_INTERVAL_SEC,
            dedup_ratio=Config.VLM_DEDUP_RATIO,
        )
        self.vlm_capture = ScreenCapture(
            monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
            jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD,
        )
        self.vlm_change_detector = ChangeDetector(
            phash_threshold=Config.VLM_PHASH_THRESHOLD, periodic_frames=Config.VLM_PERIODIC_FRAMES,
        )
        self.capture_thread: Optional[CaptureThread] = None  # run() で loop 取得後に生成（VLM_ENABLED 時）

        # J Call-Function: read-only Capability 層 + 実行 Dispatcher（既定 off）。
        # self_status の live 値は runner/queue を lazy lambda で読む（runner はこの後で生成）。
        self.capabilities = CapabilityRegistry(
            is_busy=lambda: self.runner.is_busy(),
            qsize=lambda: self.queue.qsize(),
        )
        self.dispatcher = FunctionDispatcher(registry=self.capabilities, queue=self.queue)

        # J-1 タスク管理（既定 off・CALLFUNCTION_ENABLED 前提）: read-only 能力に予約/自動実行を足す。
        # create_task/list_tasks/cancel_task は同じ registry に登録＝dispatcher 経由で応答LLM に提示される。
        self.task_store = None
        self.task_executor = None
        self.reconcile_timer = None
        if Config.TASK_ENABLED:
            self.task_store = TaskStore(
                task_file=Config.TASK_FILE, max_tasks=Config.TASK_MAX,
                orphan_timeout_sec=Config.TASK_ORPHAN_TIMEOUT_SEC,
            )
            register_task_capabilities(self.capabilities, self.task_store)
            # TaskAgent（inc2）: delegate_task の自然文ゴールを賢い task LLM が境界つきループで完遂。
            task_agent = TaskAgent(
                registry=self.capabilities, model_registry=self.registry, store=self.task_store,
                max_steps=Config.TASK_AGENT_MAX_STEPS, timeout_sec=Config.TASK_AGENT_TIMEOUT_SEC,
            )
            self.task_executor = TaskExecutor(
                store=self.task_store, registry=self.capabilities, queue=self.queue, agent=task_agent,
            )
            self.reconcile_timer = ReconcileTimer(
                store=self.task_store, executor=self.task_executor, tick_sec=Config.TASK_RECONCILE_TICK_SEC,
            )

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            if tools:
                async for delta in self.registry.stream_with_tools(
                    "response", messages, tools=tools, tool_sink=tool_sink
                ):
                    yield delta
            else:
                async for delta in self.registry.stream("response", messages):
                    yield delta

        self.orchestrator = ResponseOrchestrator(
            # system プロンプト= SPEECH_STYLE（最小スタイル指示・ペルソナではない）。
            # 空 ContextAssembler() を渡すと system 無しで応答が "システム応答…" と自己ラベル化する
            # leak が出るため、明示的に SPEECH_STYLE を与える。
            self.audio, stream_fn, self.tts.generate,
            ContextAssembler(system_prompt=SPEECH_STYLE),
            conversation_cache=self.cache, rag_store=self.rag,
            prediction_state=self.prediction,
            on_response_complete=self._on_response_complete,  # 正常完了で feedback + 沈黙時計リセット
            vision_state=self.vision_state,  # F6: 応答文脈に直近画面を注入
            dispatcher=self.dispatcher,  # J: tool_calls を応答完了後に submit（gate は CALLFUNCTION_ENABLED）
        )
        self.runner = PipelineRunner(self.queue, self.orchestrator, self.audio)
        # 沈黙監視は応答中(runner busy)/ユーザ発話中は発火しない（is_busy をガードに使う）。
        self.silence_monitor = SilenceMonitor(
            state=self.speech_state, decider=self.speech_decider, is_busy_fn=self.runner.is_busy,
        )
        self.input = MicSttInputSource(
            self.queue, self.stt,
            on_speech_start=self._barge_in,                      # ユーザ発話開始（barge-in）
            on_utterance=self.speech_state.mark_user_utterance,  # 発話終了→沈黙時計リセット
        )
        self._tasks: list[asyncio.Task] = []

    def _barge_in(self) -> None:
        """発話開始の瞬間: 音声停止＋進行中応答キャンセル（Eve が即譲る）。"""
        logger.info("⏸ 発話検知（割り込み）")
        self.audio.interrupt()
        self.runner.interrupt()  # 進行中応答(自発含む)を cancel＝実発話分のみ記録(C5)
        # F5: ユーザ優先。判定中の自発発話は seq 変化で自己破棄、キュー済みの自発刺激は削除。
        self.speech_state.mark_user_speech_start()
        self.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)

    def _on_response_complete(self) -> None:
        """応答 正常完了: F4 feedback を起こす + F5 沈黙時計をリセット（Eve が喋った）。"""
        self.feedback_worker.trigger()
        self.speech_state.mark_eve_activity()

    def _vision_can_speak(self) -> bool:
        """F6 画面起因の発話ガード（A5/Q4）: 応答中/ユーザ発話中/判定処理中なら起こさない。"""
        return (
            not self.runner.is_busy()
            and not self.speech_state.user_speaking
            and self.speech_decider.is_idle()
        )

    async def warmup(self) -> None:
        """STT/LLM/TTS を1回空打ちして cold-start（初回の数秒遅延）を消す。"""
        logger.info("ウォームアップ開始")
        await self.stt.warmup()
        try:
            await self.rag.warmup()  # 埋め込みモデルの初回ロードを先に済ませる
        except Exception as e:
            logger.warning("RAG ウォームアップ失敗（続行）: %s", e)
        try:
            # max_tokens は付けない: reasoning 系(gpt-5.x)は1トークンで完了できず BadRequest 警告に
            # なる。warmup は一度きり・"hi" への短応答なのでコストは無視できる。
            await self.registry.complete("response", [{"role": "user", "content": "hi"}])
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
        # F5: 発話判定 worker + 沈黙監視を起動（沈黙時計の baseline をここでリセット）。
        self.speech_state.mark_eve_activity()
        self.speech_decider.start()
        self.silence_monitor.start()
        # J: Call-Function 実行サイドカー（既定 off）。read-only 能力のみ。
        if Config.CALLFUNCTION_ENABLED:
            self.dispatcher.start()
            logger.info("Call-Function 稼働（read-only 能力）")
        # J-1: タスク管理（既定 off）。store 復元 + executor/scheduler 起動。
        if self.task_store is not None:
            await self.task_store.initialize()
            self.task_executor.start()
            self.reconcile_timer.start()
            logger.info("タスク管理 稼働（予約タスク）")
        # F6: 画面認識を起動（既定 off）。capture スレッドは loop 確定後に生成し on_frame を橋渡し。
        if Config.VLM_ENABLED:
            self.vlm_worker.start()
            self.capture_thread = CaptureThread(
                capture=self.vlm_capture, change_detector=self.vlm_change_detector,
                deliver=self.vlm_worker.on_frame, loop=asyncio.get_running_loop(),
                target_fps=Config.VLM_TARGET_FPS,
            )
            self.capture_thread.start()
            logger.info("画面認識(VLM) 稼働")
        await self.input.start()  # mic + STT 消費タスクを起動
        logger.info("VoiceLoop 稼働。話しかけてください。")
        await asyncio.Event().wait()  # キャンセルされるまで稼働

    async def stop(self) -> None:
        self.input.stop()
        # F6 A3: capture スレッドを**最初に**止める（閉じるループへフレームを送らない）→ vlm worker drain。
        if self.capture_thread is not None:
            try:
                await asyncio.to_thread(self.capture_thread.stop)  # join をループ外で
            except Exception:
                pass
        try:
            await self.vlm_worker.stop()
        except Exception:
            pass
        # F5: 沈黙監視を止め（新規トリガを断つ）→ 発話判定 worker を drain/停止。
        try:
            await self.silence_monitor.stop()
        except Exception:
            pass
        try:
            await self.speech_decider.stop()
        except Exception:
            pass
        # J: Call-Function 実行サイドカーを drain/停止（進行中の能力実行を取りこぼさない）。
        try:
            await self.dispatcher.stop()
        except Exception:
            pass
        # J-1: タスク管理を停止（timer 停止→executor drain→store flush の順）。
        if self.reconcile_timer is not None:
            try:
                await self.reconcile_timer.stop()
            except Exception:
                pass
        if self.task_executor is not None:
            try:
                await self.task_executor.stop()
            except Exception:
                pass
        if self.task_store is not None:
            try:
                await self.task_store.shutdown()
            except Exception:
                pass
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
