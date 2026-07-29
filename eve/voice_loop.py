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

from .clock import now_mono
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
from .response.delivery_checker import DeliveryChecker
from .response.function_dispatcher import FunctionDispatcher
from .task import (
    CancelResolver, ReconcileTimer, TaskAgent, TaskExecutor, TaskStore,
    active_tasks_for_context, register_task_capabilities,
)
from .response.input_source import MicSttInputSource
from .response.orchestrator import REDELIVER_MAX_ATTEMPTS, ResponseOrchestrator
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
        # 埋め込みは RAG と自発発話の同内容抑制（意味の重複）で共有する（モデルの二重ロードを避ける）。
        self.embedder = make_embedder()
        self.rag = RagStore(self.embedder)  # 長期記憶（連想想起）

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
            embedder=self.embedder,  # J-2 ②-2: 言い換えただけの再提案を意味類似で抑える二段目
            # J-2 P2-3: self.task_store は下の TASK_ENABLED ブロックで後から確定するため、
            # 呼び出し時点の値を読む遅延 lambda にする（is_busy=lambda: self.runner... と同じ流儀）。
            tasks_provider=lambda: (
                active_tasks_for_context(self.task_store) if self.task_store is not None else None
            ),
        )

        # F6 VLM サイドカー: 専用スレッドが capture→gate→on_frame、single-flight worker が
        # 1回の multi-frame 呼び出しで実況化→latest_vision/note_vlm_surprise→ガード付き発話トリガ。
        self.vlm_worker = VlmWorker(
            vision_state=self.vision_state, prediction_state=self.prediction,
            narrate_fn=make_narrate_fn(self.registry),  # role=vlm_leaf
            speak_trigger=lambda: self.speech_decider.trigger("vlm"),  # source は観測用（③分析の反省）
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
        # J-2 P1-1: barge-in なしで完了した機能報告ターンの配達確認（redeliver_fn は下で定義する
        # self._redeliver_stimulus をそのまま束縛・既存の再配達経路を再利用する）。
        self.delivery_checker = DeliveryChecker(
            model_registry=self.registry, redeliver_fn=self._redeliver_stimulus,
            max_attempts=REDELIVER_MAX_ATTEMPTS,
        )

        # J-1 タスク管理（既定 off・CALLFUNCTION_ENABLED 前提）: read-only 能力に予約/自動実行を足す。
        # create_task/list_tasks/cancel_task は同じ registry に登録＝dispatcher 経由で応答LLM に提示される。
        self.task_store = None
        self.task_executor = None
        self.reconcile_timer = None
        self.cancel_resolver = None
        self.search_client = None  # J-2（SEARCH_ENABLED かつ TASK_ENABLED 時のみ生成）
        tasks_provider = None  # Fix#2: TASK_ENABLED 時のみ配線（None ならブロック自体を注入しない）
        if Config.TASK_ENABLED:
            self.task_store = TaskStore(
                task_file=Config.TASK_FILE, max_tasks=Config.TASK_MAX,
                orphan_timeout_sec=Config.TASK_ORPHAN_TIMEOUT_SEC,
            )
            # 取消はタスク側で解決（別コルーチン・executor と並行）: 応答LLM は reference を渡すだけ。
            self.cancel_resolver = CancelResolver(
                store=self.task_store, model_registry=self.registry, queue=self.queue,
            )
            register_task_capabilities(self.capabilities, self.task_store, cancel_resolver=self.cancel_resolver)
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
            # Fix#2: 予約タスクの現在状態を応答LLM の system に毎ターン注入（完了済み変更の
            # 再実行・状態と矛盾する約束の防止＝2026-07-13 実機事故の根本原因対応）。
            store = self.task_store
            tasks_provider = lambda: active_tasks_for_context(store)  # noqa: E731
            # J-2 search（既定 off・TASK_ENABLED ブロック内に置く＝TaskAgent 専用能力なので
            # タスク管理なしでは誰も呼べない状態を構造的に不能にする）。
            if Config.SEARCH_ENABLED:
                import importlib.util
                if importlib.util.find_spec("ddgs") is None:
                    # 起動時 fail-soft（VOICEVOX 未起動と同流儀）: 検索だけ無効化して続行。
                    logger.warning("SEARCH_ENABLED=1 だが ddgs 未導入のため検索を無効化（pip install ddgs）")
                else:
                    from .search import SearchClient, register_search_capability
                    from .search.deep import DeepResearcher
                    self.search_client = SearchClient(
                        deep_researcher=DeepResearcher(self.registry))  # inc2: 隔離要約はツールなしロール
                    register_search_capability(self.capabilities, self.search_client)
        elif Config.SEARCH_ENABLED:
            logger.warning("SEARCH_ENABLED=1 だが TASK_ENABLED=0 のため検索は無効（TaskAgent 専用能力）")

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
            tasks_provider=tasks_provider,  # J-1/Fix#2: 予約タスク状態の毎ターン注入（TASK_ENABLED 時のみ）
            capabilities_provider=self.capabilities.outward_actions,  # D3: 外の世界への手段（registry 導出）
            redeliver_fn=self._redeliver_stimulus,  # barge-in で潰れたタスク報告の再配達（WHEN はこちらが所有）
            delivery_checker=self.delivery_checker,  # J-2 P1-1: barge-in なし完了ターンの配達確認
            is_suppressed=self.queue.is_suppressed,  # D2: 取消済み報告は発話前に捨てる（put の第2関門）
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
            on_stt_start=self.speech_state.mark_stt_pending,     # J-2 ③-A: STT待ち窓を開く
            on_stt_end=self.speech_state.clear_stt_pending,      # J-2 ③-A: 投入完了で窓を閉じる
        )
        self._tasks: list[asyncio.Task] = []
        # 再配達（barge-in で潰れた機能報告）の待機タスク。done で自己除去・stop() で cancel。
        self._redeliver_waiters: "set[asyncio.Task]" = set()
        self._redeliver_grace_sec = 2.0  # user_speaking 解除→put までの STT 完了猶予
        self._redeliver_max_wait_sec = 60.0  # 話し終わり待ちの上限（餓死防止）

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

    def _redeliver_stimulus(self, stim, should_abort) -> None:
        """barge-in で発話前に潰れた機能報告の再投入（WHEN 制御＝2026-07-13 21:20 事故対応）。

        ユーザが話し終わる（user_speaking=False）まで待ち、さらに STT 完了猶予をおいてから
        queue へ戻す。これでユーザの割り込み発話が先にキューに並び、priority（USER=0 <
        CALLFUNCTION_RESULT=1）で「ユーザへの応答が先・再報告が後」の順序が構造的に成立する。
        should_abort = put 直前の最終判定（「結局再生されていた」レースを拾い二重発話を防ぐ。
        on_played は cancel 伝播より数十ms 遅れて発火し得るため、判定はここまで遅延させる）。
        同期・非ブロッキング（orchestrator の CancelledError 経路から呼ばれる）。
        既知の限界: STT が猶予を超えて遅いとユーザ発話刺激より先に並び順序が入れ替わる
        （両方配達はされる・再報告マーキングで発話は自然に繋がる＝許容）。
        """
        async def _wait_and_put() -> None:
            t0 = now_mono()
            while self.speech_state.user_speaking and now_mono() - t0 < self._redeliver_max_wait_sec:
                await asyncio.sleep(0.1)  # 話し終わり待ち（上限＝報告を餓死させない）
            if self.speech_state.user_speaking:
                logger.warning("⚠ ユーザ発話が待機上限を超過 — 報告を再投入する（barge で再中断され得る）")
            await asyncio.sleep(self._redeliver_grace_sec)  # STT 完了猶予＝ユーザ発話刺激が先に並ぶ
            if should_abort():
                logger.info("🔁 再配達を中止（再生済みが確定＝二重発話防止）")
                return
            await self.queue.put(stim)

        task = asyncio.create_task(_wait_and_put())
        # done で自己除去（溜め込まない）+ stop() で明示 cancel（シャットダウン時の孤児化防止）。
        self._redeliver_waiters.add(task)
        task.add_done_callback(self._redeliver_waiters.discard)

    def _vision_can_speak(self) -> bool:
        """F6 画面起因の発話ガード（A5/Q4）: 応答中/ユーザ発話中/STT待ち/判定処理中なら起こさない。

        stt_pending は J-2 ③-A: VLM 経由トリガは沈黙5秒閾値を迂回するため、発話終了〜STT完了の
        窓（1〜3秒）に自発発話が差し込まれ、直後に届くユーザ発話の処理を遅らせていた（E2E S8）。
        """
        return (
            not self.runner.is_busy()
            and not self.speech_state.user_speaking
            and not self.speech_state.stt_pending
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
            self.delivery_checker.start()  # J-2 P1-1: 報告ターンの配達確認
            logger.info("Call-Function 稼働（read-only 能力）")
        # J-1: タスク管理（既定 off）。store 復元 + executor/scheduler 起動。
        if self.task_store is not None:
            await self.task_store.initialize()
            self.task_executor.start()
            self.reconcile_timer.start()
            if self.cancel_resolver is not None:
                self.cancel_resolver.start()  # executor と並列（取消が実行中タスクの後ろで待たない）
            logger.info("タスク管理 稼働（予約タスク）")
            if self.search_client is not None:
                logger.info("Web検索 稼働（search_web・TaskAgent 専用）")
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
        # J-2 P1-1: 配達確認サイドカーを drain/停止（進行中の判定が再配達を予約する猶予を与える。
        # 予約された再配達タスクは _redeliver_waiters に乗るので下の孤児化防止で回収される）。
        try:
            await self.delivery_checker.stop()
        except Exception:
            pass
        # J-1: タスク管理を停止（取消解決 drain→timer 停止→executor drain→store flush の順）。
        if self.cancel_resolver is not None:
            try:
                await self.cancel_resolver.stop()
            except Exception:
                pass
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
        for t in list(self._redeliver_waiters):  # 再配達待機の孤児化防止（未配達分は消える＝許容）
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
