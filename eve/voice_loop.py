"""VoiceLoop — フルループ組み立て（mic→STT→応答LLM→TTS→再生）。

F0–F2.5 の部品を結線する本体。UI(F6) はこれを包む想定。レイテンシ重視:
- 起動時ウォームアップで cold-start を消す。
- 全段 asyncio タスク（ブロッキングI/Oのみ executor）。
- barge-in は MicSttInputSource が発話開始で audio.interrupt()（鳴っている節は言い切る）。
  ユーザが話している途中にイブの音が鳴り始めた時も、最初の節で譲る（`_yield_if_user_speaking`）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from .clock import now_iso, now_mono
from .config import Config
from .context_assembler import ContextAssembler
from .feedback import FeedbackLLM, FeedbackWorker, PredictionState
from .memory import ConversationCache, RagStore
from .memory.embed import make_embedder
from .model_registry import ModelRegistry
from .pipeline.audio_play_queue import AudioPlayQueue
from .pipeline.orchestrator import PipelineRunner
from .pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
from .pipeline.stimulus_queue import StimulusQueue
from .capability import CapabilityRegistry
from .cleanup import sweep as cleanup_sweep
from .cleanup import sweep_orphan_tts_servers
from .response.delivery_checker import DeliveryChecker
from .response.function_dispatcher import FunctionDispatcher
from .task import (
    CancelResolver, ReconcileTimer, TaskAgent, TaskExecutor, TaskStore,
    active_tasks_for_context, register_task_capabilities, undelivered_reports,
)
from .response.input_source import MicSttInputSource
from .response.orchestrator import REDELIVER_MAX_ATTEMPTS, ResponseOrchestrator
from .response.player import RealAudioPlayer
from .response.style import SPEECH_STYLE, SPEECH_STYLE_STREAM
from .response.tts import VoicevoxTTS
from .emotion import EmotionState
from .speech import SilenceMonitor, SpeechDecider, SpeechState, make_decide_fn
from .memory.self_model import SelfModelStore, SelfRecall
from .persona import load_persona, persona_for_stream, persona_path
from .persona_examples import StyleExamples, examples_path, load_examples
from .self_model.worker import SelfModelWorker
from .stt import make_stt
from .vlm import ChangeDetector, VisionState, VlmWorker, make_narrate_fn
from .vlm.capture import ScreenCapture
from .vlm.capture_thread import CaptureThread

logger = logging.getLogger(__name__)

# 配信の始め・終わりの区切りで持ち越す要約（`VoiceLoop._carry_text`）: 配信の前は手前30分・配信の後は30分間・最大6件
CARRY_PRE_SEC = 1800.0
CARRY_POST_SEC = 1800.0
CARRY_MAX = 6


def _iso_shift(iso: str, sec: float) -> str:
    from datetime import datetime, timedelta

    return (datetime.fromisoformat(iso) + timedelta(seconds=sec)).isoformat()


def _minutes(sec: float) -> int:
    return max(1, int(round(sec / 60.0)))


def _make_tts():
    """`TTS_BACKEND` で応答の TTS を選ぶ（**自動フォールバックはしない**＝ユーザ裁定 2026-09-25）。"""
    backend = (Config.TTS_BACKEND or "").strip().lower()
    if backend == "irodori":
        from .response.tts_irodori import IrodoriTTS

        return IrodoriTTS()
    if backend == "voicevox":
        return VoicevoxTTS()
    raise ValueError(f"unknown TTS_BACKEND: {Config.TTS_BACKEND}（irodori | voicevox）")


def _preimport_sdks() -> None:
    """LLM 系 SDK を**1スレッドで順に** import しておく（並列ウォームアップの前提）。

    どれが失敗しても続行する（オフライン構成や未導入の provider があり得る）。
    ここで温めた import は各 SDK 側でキャッシュされるので、後段の並列呼び出しは
    ネットワーク待ちだけになる。
    """
    for mod in ("pydantic", "openai", "litellm"):
        try:
            __import__(mod)
        except Exception:
            logger.debug("事前 import に失敗（続行）: %s", mod, exc_info=True)


class VoiceLoop:
    def __init__(self) -> None:
        # UI の機能 ON/OFF（`eve/control.py`）。**ヘッドレスでは None のまま**＝挙動は不変。
        # UI が起動時に差し込む。応答文脈の `features_off` はこれを遅延参照する。
        self.control = None
        # Alchemy（ユーザ作のゲーム・既定 off）。実体は下の Call-Function の後で作る。発話判定・画面認識・応答は
        # この参照を遅延 lambda で読む（作る前に呼ばれても None＝何も足さない）
        self.alchemy = None
        self._alchemy_react_task: Optional[asyncio.Task] = None
        # YouTube 配信モード（既定 off）。実体は下で作る。応答・発話判定はこの参照を遅延 lambda で読む（配信していない時は None）
        self.stream = None
        # 配信の始め・終わりの区切りで持ち越す「直前に話していたこと」（`_carry_text`・無ければ None）
        self._carry: Optional[dict] = None
        # API 使用量メーター（既定 off）。ON の時だけ ModelRegistry/STT/埋め込みが記録する。
        self.usage = None
        if Config.USAGE_LOG_ENABLED:
            from .usage import UsageMeter, install

            self.usage = UsageMeter(Path(Config.USAGE_LOG_FILE), tag=os.getenv("USAGE_TAG", ""))
            install(self.usage)
            logger.info("API 使用量の記録 稼働（%s）", self.usage.path)
        self.registry = ModelRegistry()
        self.player = RealAudioPlayer()
        self.audio = AudioPlayQueue(play_fn=self.player.play_fn)
        # 出力は開きっぱなし（2026-09-26）。区切りの間は直前の音（応答・フィラー）の終わりから測る
        self.audio.item_gap_sec = Config.SPEECH_GAP_SEC
        self.audio.last_audio_end = lambda: self.player.last_end
        self.queue = StimulusQueue()
        self.tts = _make_tts()  # 既定 Irodori（サーバは warmup で用意する＝ここでは起動しない）
        self.stt = make_stt()
        # 性格（persona.md）は応答だけでなくフィードバック・感情判定・笑い判定にも渡す（食い違わないため・
        # 2026-10-05・笑い判定は 2026-10-09）。笑い判定を作る前に読む
        self.persona = load_persona(os.getenv("PERSONA_FILE"))
        # S3 応答非依存フィラー（既定 off）。runner は下で作るので is_busy は遅延 lambda。
        self.filler = self._build_filler()
        # S6 笑い（既定 off）。フィラーの仕組みで鳴らすので、フィラーが無ければ作らない。
        self.laugh = self._build_laugh()
        # S5 イブの今の感情（既定 off）。「、」「。」ごとの判定（ストリーミング文字起こしの時）とフィードバックで更新。
        # 5つの数値で、判定した会話の時点から時間とともに薄れる。判定のたびに保存し、起動時に前回の分を読む（2026-10-09）
        self.emotion = EmotionState(path=Config.EMOTION_STATE_FILE) if Config.EMOTION_ENABLED else None
        self.emotion_judge = self._build_emotion_judge()
        # S8 フィラーの1段目を場面に寄せる（既定 off・Jev）
        self.mood = self._build_mood()
        self._voice_tone = self._build_voice_tone()  # 応答の声のトーン（orchestrator がターンごとに1回読む）
        self.cache = ConversationCache()  # 短期記憶（会話ログ・直近注入・実発話記録）
        # 埋め込みは RAG と自発発話の同内容抑制（意味の重複）で共有する（モデルの二重ロードを避ける）。
        self.embedder = make_embedder()
        self.rag = RagStore(self.embedder)  # 長期記憶（連想想起）
        # X で見たこと（`eve/xfeed/`・既定 off）。会話の記憶とは別ファイル・別の上限で持ち、思い出す時だけ合わせる
        # （同じファイルにすると、起動時の内省の位置・配信の区切りの要約・件数の上限が X の記憶で狂う＝`RagStore.attach_side`）
        self.x_memory = None
        if Config.X_ENABLED:
            self.x_memory = RagStore(self.embedder, rag_file=Config.X_MEMORY_FILE, max_chunks=Config.X_MEMORY_MAX)
            self.rag.attach_side(self.x_memory)
        # 話し方の実例（persona_examples.yaml・人が選ぶ）。無い・0 なら None＝性格の文だけで動く
        self.style_examples = self._build_style_examples()
        # 自己モデル（イブの好み・趣向）。**性格はここに入らない**（ペルソナ=persona.md の担当）。
        # 既定 off。ON なら応答文脈と TaskAgent の両方に `# イブ自身のこと` が渡る。
        self.self_model = None
        self.self_model_worker = None
        self.self_recall = None
        if Config.SELF_MODEL_ENABLED:
            self.self_model = SelfModelStore(Config.SELF_MODEL_FILE)
            self.self_model_worker = SelfModelWorker(
                model_registry=self.registry, store=self.self_model, cache=self.cache,
            )
            # 応答には今の話しかけに近い項目だけを入れる（記憶の検索と同じ埋め込み・ユーザ 2026-10-17）
            self.self_recall = SelfRecall(self.self_model, self.embedder)
        # X で話題を仕入れるサイドカー（6時間ごと・3件・起動時に前回からの分を1回取り戻す）。性格と好みの控えはイブの感想に使う
        self.x_collector = None
        if self.x_memory is not None:
            from .xfeed import build_collector

            self.x_collector = build_collector(
                registry=self.registry, store=self.x_memory, memory=self.rag, persona=self.persona,
                # 情報収集AI には好みの全部・感想には話題に近い項目だけ（応答と同じ選び方）。
                # `is not None` で見る（空の台帳でも配線する・TaskAgent と同じ理由）
                self_model_fn=(lambda: self.self_model.render()) if self.self_model is not None else None,
                self_model_pick=self.self_recall.block_for if self.self_recall is not None else None,
                # X で見た物への感想からも好みを育てる（ユーザ 2026-10-17「情報収集中でも更新はしていい」）
                on_impression=(self.self_model_worker.note_text if self.self_model_worker is not None else None),
                embedder=self.embedder)  # 1回の3件が似た話題なら次の候補にする（記憶と同じ埋め込み）
            if self.x_collector is None:
                logger.warning("X_ENABLED=1 だが X_BEARER_TOKEN が無いので X には取りに行かない（Xで見た記憶は思い出す）")

        # F4 内分泌系: 各応答後に非同期で内省 → RAG 書込 + surprise + 直近フィードバック注入。
        self.prediction = PredictionState()  # loop 所有・単一書込（feedback/VLM が書く）
        self.feedback = FeedbackLLM(self.registry, rag_store=self.rag, prediction_state=self.prediction,
                                    emotion_state=self.emotion, persona=self.persona)
        self.feedback_worker = FeedbackWorker(self.feedback, self.cache)

        # F6 画面認識: loop 所有 VisionState（ring=全キャプチャ・latest_vision）。
        self.vision_state = VisionState(ring_max=Config.VLM_RING_MAX)

        # F5 発話判定: 5秒沈黙で should_speak → True で AUTONOMOUS_SPEECH を投入。
        self.speech_state = SpeechState()  # loop 所有 ephemeral（沈黙計測/発話判定ログ）
        self.speech_decider = SpeechDecider(
            state=self.speech_state, cache=self.cache, rag=self.rag,
            prediction_state=self.prediction, queue=self.queue,
            decide_fn=make_decide_fn(self.registry, persona=self.persona),  # role=speech_decide・性格つき
            vision_state=self.vision_state,  # F6: 発話判定に直近画面を入れる
            embedder=self.embedder,  # J-2 ②-2: 言い換えただけの再提案を意味類似で抑える二段目
            # J-2 P2-3: self.task_store は下の TASK_ENABLED ブロックで後から確定するため、
            # 呼び出し時点の値を読む遅延 lambda にする（is_busy=lambda: self.runner... と同じ流儀）。
            tasks_provider=lambda: (
                active_tasks_for_context(
                    self.task_store, undelivered=undelivered_reports(self.task_store, self.queue))
                if self.task_store is not None else None
            ),
            # Alchemy の部屋・対戦中だけ事実を渡す（普段は None＝発話判定の入力は変わらない）
            game_provider=lambda: self.alchemy.decider_block() if self.alchemy is not None else None,
            # YouTube 配信中だけ事実を渡す（配信していない時は None）
            stream_provider=lambda: self.stream.decider_block() if self.stream is not None else None,
        )

        # F6 VLM サイドカー: 専用スレッドが capture→gate→on_frame、single-flight worker が
        # 1回の multi-frame 呼び出しで実況化→latest_vision/note_vlm_surprise→ガード付き発話トリガ。
        self.vlm_worker = VlmWorker(
            vision_state=self.vision_state, prediction_state=self.prediction,
            # E11: 自分が開いた窓を「ユーザの画面変化」として実況しないよう、毎回名前を渡す
            # （`self.workspace` は下の WS_ENABLED ブロックで後から確定するので遅延 lambda）。
            narrate_fn=make_narrate_fn(  # role=vlm_leaf
                self.registry,
                own_windows_fn=lambda: (self.workspace.open_window_names()
                                        if self.workspace is not None else ()),
                # 前回の予想（画面認識の中だけで回す・ユーザ裁定 2026-09-30）。worker はこの直後に作るので遅延 lambda
                prediction_fn=lambda: self.vlm_worker.last_prediction,
                # Alchemy: イブ自身のゲーム画面を見ている間の断り書き / ユーザの画面に Alchemy が映っているかの1行
                screen_note_fn=lambda: self.alchemy.screen_note() if self.alchemy is not None else None,
                detect_alchemy=Config.ALCHEMY_ENABLED and Config.CALLFUNCTION_ENABLED,
            ),
            speak_trigger=lambda: self.speech_decider.trigger("vlm"),  # source は観測用（③分析の反省）
            speak_guard=self._vision_can_speak,  # A5/Q4: busy/ユーザ発話中/decider 処理中なら叩かない
            frames_per_call=Config.VLM_MAX_FRAMES_PER_CALL,
            min_interval_sec=Config.VLM_MIN_INTERVAL_SEC,
            dedup_ratio=Config.VLM_DEDUP_RATIO,
            on_alchemy_seen=lambda: self.alchemy.note_seen_on_screen() if self.alchemy is not None else None,
        )
        self.vlm_capture = ScreenCapture(
            monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
            jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD,
        )
        # デスクトップの写し方を覚えておく（Alchemy の対戦中はイブのゲーム画面に差し替え、終わったら戻す）
        self._desktop_capture = self.vlm_capture
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
        self.workspace = None  # J-3（WS_ENABLED かつ TASK_ENABLED 時のみ生成）
        tasks_provider = None  # Fix#2: TASK_ENABLED 時のみ配線（None ならブロック自体を注入しない）
        # UI「タスク ON/OFF」用: **登録し直すための手続き**を覚えておく（`registry.unregister` の
        # docstring の指針どおり、束＝タスク+検索+作業フォルダ をまとめて出し入れする）。
        # 手書きの名前リストにしないのは、構成（SEARCH/WS/EXEC/VIEWER）で束が変わるため。
        self._task_cap_registrars: "list[Callable[[], None]]" = []
        _caps_before = set(self.capabilities.names())

        def _reg(fn: "Callable[[], None]") -> None:
            """今すぐ登録し、あとで**同じ登録をやり直せる**ように覚えておく。"""
            self._task_cap_registrars.append(fn)
            fn()

        if Config.TASK_ENABLED:
            self.task_store = TaskStore(
                task_file=Config.TASK_FILE, max_tasks=Config.TASK_MAX,
                orphan_timeout_sec=Config.TASK_ORPHAN_TIMEOUT_SEC,
            )
            # 取消はタスク側で解決（別コルーチン・executor と並行）: 応答LLM は reference を渡すだけ。
            self.cancel_resolver = CancelResolver(store=self.task_store, queue=self.queue)
            # queue は1件制ゲートが「未配達の完了報告」を数えるのに要る（`capabilities.py`）。
            _reg(lambda: register_task_capabilities(
                self.capabilities, self.task_store,
                cancel_resolver=self.cancel_resolver, queue=self.queue))
            # TaskAgent（inc2）: delegate_task の自然文ゴールを賢い task LLM が境界つきループで完遂。
            task_agent = TaskAgent(
                registry=self.capabilities, model_registry=self.registry, store=self.task_store,
                max_steps=Config.TASK_AGENT_MAX_STEPS, timeout_sec=Config.TASK_AGENT_TIMEOUT_SEC,
                # goal の「イブの好きな〜」を受け手側で解決できるようにする（会話履歴は渡さない）。
                # `is not None` で見る（台帳は len を持つので、空で起動すると `if self.self_model` が偽になり、
                # その起動中に好みが書かれても渡らなかった・2026-10-09）
                self_model_fn=(lambda: self.self_model.render()) if self.self_model is not None else None,
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
            # 1件制: 枠を握るのはアクティブだけでなく「終わったが未配達の報告」も含むので、
            # 応答LLM にも同じ母集団を見せる（受付ゲートの拒否理由と食い違わせない）。
            queue = self.queue
            tasks_provider = lambda: active_tasks_for_context(  # noqa: E731
                store, undelivered=undelivered_reports(store, queue))
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
                    _reg(lambda: register_search_capability(self.capabilities, self.search_client))
            # J-3 作業フォルダ（既定 off・search と同じ理由で TASK_ENABLED ブロック内に置く）。
            if Config.WS_ENABLED:
                from .workspace import Workspace, register_workspace_capabilities, resolve_workspace_root
                from .workspace.paths import exec_compatible
                ws_root, why = resolve_workspace_root(Config.WS_ROOT)
                if ws_root is None:
                    # 起動時 fail-soft（ddgs 未導入時と同流儀）: 作業フォルダだけ無効化して続行。
                    logger.warning("WS_ENABLED=1 だが作業フォルダの場所が不適切なため無効化（%s）", why)
                else:
                    self.workspace = Workspace(
                        ws_root, quota_bytes=int(Config.WS_QUOTA_GB * 1024 ** 3),
                        max_write_chars=Config.WS_WRITE_MAX_CHARS,
                        max_read_chars=Config.WS_READ_MAX_CHARS,
                        max_files=Config.WS_MAX_FILES, rescan_sec=Config.WS_RESCAN_SEC,
                    )
                    _reg(lambda: register_workspace_capabilities(self.capabilities, self.workspace))
                    # プログラム実行（別フラグ）。setup_sandbox.py apply 済みが前提だが、
                    # 未セットアップでも起動は続ける（実行時に SandboxUnavailable で
                    # 「サンドボックスが使えない」と正直に返る＝fail-soft）。
                    exec_ok, exec_why = exec_compatible(ws_root)
                    if Config.WS_EXEC_ENABLED and not exec_ok:
                        # ファイル能力は親プロセスで動くので生かしたまま、実行だけ無効化する
                        # （E2E で「作成はできたけど実行環境エラー」が出た経路を起動時に潰す）。
                        logger.warning("WS_EXEC_ENABLED=1 だが実行を無効化: %s", exec_why)
                    elif Config.WS_EXEC_ENABLED:
                        from .workspace import register_exec_capability
                        _reg(lambda: register_exec_capability(
                            self.capabilities, self.workspace,
                            allow=Config.WS_EXEC_ALLOW.split(","),
                            timeout_sec=Config.WS_RUN_TIMEOUT_SEC,
                            grace_sec=Config.WS_RUN_GRACE_SEC,
                            mem_mb=Config.WS_RUN_MEM_MB,
                            max_procs=Config.WS_RUN_MAX_PROCS,
                            max_output_chars=Config.WS_MAX_OUTPUT_CHARS,
                        ))
                    if Config.WS_VIEWER_ENABLED:
                        from .workspace import register_viewer_capability
                        _reg(lambda: register_viewer_capability(
                            self.capabilities, self.workspace,
                            max_windows=Config.WS_VIEWER_MAX_WINDOWS))
        elif Config.SEARCH_ENABLED or Config.WS_ENABLED:
            if Config.SEARCH_ENABLED:
                logger.warning("SEARCH_ENABLED=1 だが TASK_ENABLED=0 のため検索は無効（TaskAgent 専用能力）")
            if Config.WS_ENABLED:
                logger.warning("WS_ENABLED=1 だが TASK_ENABLED=0 のため作業フォルダは無効（TaskAgent 専用能力）")
        # 束の実体は「登録の前後で増えた名前」。手書きしないので構成差で腐らない。
        self._task_cap_names: "list[str]" = sorted(set(self.capabilities.names()) - _caps_before)

        # Alchemy（ユーザ作のブラウザ対戦ゲーム・既定 off）。タスクの束とは別に登録する（UI のタスク ON/OFF で消えない）。
        # 部屋の出入りは応答AI の機能なので CALLFUNCTION_ENABLED が前提
        if Config.ALCHEMY_ENABLED:
            if not Config.CALLFUNCTION_ENABLED:
                logger.warning("ALCHEMY_ENABLED=1 だが CALLFUNCTION_ENABLED=0 のため Alchemy は無効"
                               "（部屋に入る・やめるは応答AI の機能）")
            else:
                from .alchemy import AlchemyManager, register_alchemy_capabilities

                self.alchemy = AlchemyManager(
                    self.registry, react=self._alchemy_react, report=self._alchemy_report,
                    screen_hook=self._alchemy_screen, say=self._alchemy_say,
                    user_seq=lambda: self.speech_state.user_activity_seq,
                    # 見切りの30秒を止めるのは、イブが実際に話している・ユーザが話している間だけ
                    # （発話判定が考えている間まで数えると、判定のたびに時計が戻り79秒かかった・実測 2026-10-11）
                    busy=lambda: (self.runner.is_busy() or self.speech_state.user_speaking
                                  or not self.audio.is_quiet() or self.audio.is_playing()))
                register_alchemy_capabilities(self.capabilities, self.alchemy)

        # YouTube 配信モード（ユーザ指定 2026-10-15・既定 off）。UI の［配信］ボタンから ControlPlane が start/stop する。
        # コメントは「イブの手が空いた時」に1件ずつユーザ発話として入る（割り込まない）
        if Config.STREAM_ENABLED:
            from .stream import StreamManager, StreamSettings

            self.stream = StreamManager(
                StreamSettings.from_config(), is_free=self._free_for_comment, submit=self.submit_stream_comment,
                before_stream=self._stream_session_start, on_live=self._stream_live,
                after_stream=self._stream_session_end)
            self.stream.on_end = self._stream_ended
            # 配信中にイブが画面に出した窓（作業の結果）を数えて閉じるため（タスクが無い構成では空）
            self.stream.viewers_fn = lambda: self.workspace.live_viewers() if self.workspace is not None else []
            # この配信の会話（区切りより後）。イブの発言に何度も出ている言葉を数える（`StreamManager._repeat_fact`）
            self.stream.turns_fn = self.cache.since_boundary

        # 応答LLM の reasoning_effort（Config.RESPONSE_REASONING_EFFORT・gpt-6-luna は none 必須）。
        # litellm は gpt-6-luna がこの引数を受けることを知らないので allowed_openai_params で通す（emotion.py と同じ）
        effort = Config.RESPONSE_REASONING_EFFORT.strip()
        response_kw = ({"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]}
                       if effort else {})
        model = self.registry.resolve("response")
        if Config.CALLFUNCTION_ENABLED and "gpt-6-luna" in model and effort != "none":
            logger.warning("RESPONSE_MODEL=%s はツール付きだと reasoning_effort=none でないと失敗する"
                           "（ユーザ発話への応答が出ない）→ .env に RESPONSE_REASONING_EFFORT=none", model)

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            if tools:
                async for delta in self.registry.stream_with_tools(
                    "response", messages, tools=tools, tool_sink=tool_sink, **response_kw
                ):
                    yield delta
            else:
                async for delta in self.registry.stream("response", messages, **response_kw):
                    yield delta

        self.orchestrator = ResponseOrchestrator(
            # system プロンプト= SPEECH_STYLE（最小スタイル指示・ペルソナではない）。
            # 空 ContextAssembler() を渡すと system 無しで応答が "システム応答…" と自己ラベル化する
            # leak が出るため、明示的に SPEECH_STYLE を与える。
            # tts は `generate_with_meta`（= (WAV, 口の開閉計画)）を渡す。UI が無い時は
            # 計画が誰にも読まれないだけで挙動は同じ（orchestrator が tuple を吸収する）。
            self.audio, stream_fn, self.tts.generate_with_meta,
            # ペルソナ（性格・口調）は**別ファイル**から後付けする。規律(SPEECH_STYLE)と
            # 分けておくのが要点で、混ぜると v1 の「プロンプト肥大＋ペルソナとロジックの癒着」
            # を繰り返す（CLAUDE.md の v1 問題5）。無ければ従来どおりペルソナ無しで動く。
            ContextAssembler(system_prompt="\n\n".join(
                p for p in (SPEECH_STYLE, self.persona) if p),
                stream_system_prompt="\n\n".join(
                    p for p in (SPEECH_STYLE_STREAM, persona_for_stream(self.persona)) if p),
                reply_opener=self._reply_opener()),  # S4: 応答依存フィラー（ユーザ発話への応答だけ）
            # 文ごとの合成を同時にいくつ投げるか（Irodori はサーバが1本ずつ処理するので 2・S1）
            tts_concurrency=getattr(self.tts, "concurrency", 3),
            conversation_cache=self.cache, rag_store=self.rag,
            prediction_state=self.prediction,
            on_response_complete=self._on_response_complete,  # 正常完了で feedback + 沈黙時計リセット
            # 空の台帳で起動しても配線する（`is not None`・上の TaskAgent と同じ理由）
            self_model_fn=(lambda: self.self_model.render()) if self.self_model is not None else None,
            self_model_pick=self.self_recall.block_for if self.self_recall is not None else None,
            vision_state=self.vision_state,  # F6: 応答文脈に直近画面を注入
            dispatcher=self.dispatcher,  # J: tool_calls を応答完了後に submit（gate は CALLFUNCTION_ENABLED）
            tasks_provider=tasks_provider,  # J-1/Fix#2: 予約タスク状態の毎ターン注入（TASK_ENABLED 時のみ）
            capabilities_provider=self.capabilities.outward_actions,  # D3: 外の世界への手段（registry 導出）
            redeliver_fn=self._redeliver_stimulus,  # barge-in で潰れたタスク報告の再配達（WHEN はこちらが所有）
            delivery_checker=self.delivery_checker,  # J-2 P1-1: barge-in なし完了ターンの配達確認
            is_suppressed=self.queue.is_suppressed,  # D2: 取消済み報告は発話前に捨てる（put の第2関門）
            # UI: 今オフの機能を毎ターン system に入れる（「機能が無い」と言わせないため）。
            # ヘッドレスでは control が None なので空＝ブロック自体が出ない。
            features_off_provider=lambda: (self.control.features_off()
                                           if self.control is not None else []),
            # S3: 1文目の合成開始＝「応答が近い」（フィラーは次の語を始めない）/ 直前のつなぎ言葉を応答LLM へ
            on_first_sentence=self._on_response_near,
            filler_provider=self.filler.take_spoken if self.filler is not None else None,
            # S5: イブの今の気持ち（全刺激種別で system に入れる・未判定なら入れない）
            emotion_provider=self.emotion.describe if self.emotion is not None else None,
            tone_provider=self._voice_tone,  # S5: 声のトーン（未決の間は None＝付けない）
            style_examples=self.style_examples,  # 話し方の実例（ユーザ発話への応答だけ・毎ターン選び直す）
            # Alchemy の準備・部屋・対戦中だけ `# Alchemy`（ルールと事実）を入れる（普段は None＝入れない）
            game_provider=lambda: self.alchemy.voice_block() if self.alchemy is not None else None,
            # 委譲の使い方は delegate_task が出ている時だけ（UI でタスクを切っても Alchemy の機能は残るため）
            delegation_offered=lambda: self.capabilities.has("delegate_task"),
            # YouTube 配信中だけ `# 配信`（誰が見ているか・コメントの形）を入れる（配信していない時は None）
            stream_provider=lambda: self.stream.voice_block() if self.stream is not None else None,
        )
        self.runner = PipelineRunner(self.queue, self.orchestrator, self.audio)
        # 沈黙監視は応答中(runner busy)/ユーザ発話中/フィラー中は発火しない。
        self.silence_monitor = SilenceMonitor(
            state=self.speech_state, decider=self.speech_decider, is_busy_fn=self._busy_for_autospeech,
        )
        self.input = MicSttInputSource(
            self.queue, self.stt,
            on_speech_start=self._barge_in,                      # ユーザ発話開始（barge-in）
            on_utterance=self.speech_state.mark_user_utterance,  # 発話終了→沈黙時計リセット
            on_stt_start=self.speech_state.mark_stt_pending,     # J-2 ③-A: STT待ち窓を開く
            on_stt_end=self.speech_state.clear_stt_pending,      # J-2 ③-A: 投入完了で窓を閉じる
            on_speech_end=self._on_speech_end,                   # S3: VAD の終了確定＝フィラーの発火点
            on_stt_result=self._on_stt_result,                   # S3: 空認識ならフィラーを止める
            # S6 笑い判定・S5 感情の判定: 途中の文字（「、」「。」ごと）
            on_phrase=(self._on_phrase if (self.laugh is not None or self.emotion_judge is not None
                                           or self.mood is not None) else None),
        )
        # イブの音が鳴り始める時、ユーザがまだ話していたら最初の節で譲る（割り込みのかけ直し）
        self.audio.add_start_listener(self._yield_if_user_speaking)
        if self.stream is not None:
            self.audio.add_text_listener(self.stream.on_sentence)  # 配信の字幕（配信していない時は何もしない）
            self.orchestrator.carry_provider = self._carry_text  # 配信の始め・終わりで直前の会話の要約を持ち越す
            # 配信中は自分から話すかを考え直す間隔を短く保つ（話すかどうかはイブが決める）
            self.silence_monitor.max_interval_fn = (
                lambda: Config.STREAM_SPEECH_MAX_INTERVAL_SEC if self.stream.active else None)
        self._tasks: list[asyncio.Task] = []
        # 再配達（barge-in で潰れた機能報告）の待機タスク。done で自己除去・stop() で cancel。
        self._redeliver_waiters: "set[asyncio.Task]" = set()
        self._ui_tasks: "set[asyncio.Task]" = set()  # UI からの投入（キーボード入力）の保持
        # 起動の進み具合を UI に見せるフック（既定 None＝ヘッドレスの挙動は不変）。
        # `fn(key: str, text: str)`。loop 上から同期で呼ぶ＝UI 側は emit だけして戻ること。
        self.on_progress = None
        self._redeliver_grace_sec = 2.0  # user_speaking 解除→put までの STT 完了猶予
        self._redeliver_max_wait_sec = 60.0  # 話し終わり待ちの上限（餓死防止）

    # --- S3 フィラー -------------------------------------------------------

    def _build_filler(self):
        """応答非依存フィラーを用意する（既定 off・使えなければ理由をログに出して無効＝起動は続ける）。"""
        if not Config.FILLER_ENABLED:
            return None
        try:
            from .filler.bank import FillerBank, FillerTable
            from .filler.player import FillerPlayer
            from .response.tts_irodori import resolve_repo_path

            table = FillerTable.from_yaml(Path(resolve_repo_path(Config.FILLER_TABLE)))
            bank_dir = Path(resolve_repo_path(str(Path("bank") / Config.FILLER_VOICE)))
            if table.join_gap_sec is None:
                # 推測で決めない: 聴き比べてユーザが決めるまで鳴らさない（listen/s2/gap_*.wav）
                logger.warning("FILLER_ENABLED=1 だが fillers.yaml の timing.join_gap_sec（つなぎの間）が"
                               "未決のためフィラーは鳴らさない")
                return None
            # S6 笑いの声色・笑いの語（あれば）。bank/<FILLER_VOICE><laugh.bank_suffix>/
            laugh_dir = bank_dir.parent / (bank_dir.name + table.laugh_bank_suffix)
            # S5 感情の声（ユーザ 2026-09-26「フィラーも感情載せる」）。emotion.yaml で絵文字が決まった感情だけ
            # bank/<FILLER_VOICE><BANK_SUFFIX>/ を読む（無い・絵文字が違う感情は普通の声）
            emotion_emoji: dict = {}
            emotion_dirs: dict = {}
            if Config.EMOTION_ENABLED:
                from .emotion import BANK_SUFFIX, load_voice_emoji

                emotion_emoji = load_voice_emoji(Path(resolve_repo_path(Config.EMOTION_TABLE)))
                emotion_dirs = {k: bank_dir.parent / (bank_dir.name + BANK_SUFFIX[k]) for k in emotion_emoji}
            bank = FillerBank.load(table, bank_dir, laugh_dir=laugh_dir,
                                   emotion_dirs=emotion_dirs, emotion_emoji=emotion_emoji)
        except Exception as e:
            logger.warning("FILLER_ENABLED=1 だがフィラーを無効化: %s", e)
            return None
        if (Config.TTS_BACKEND or "").strip().lower() != "irodori":
            logger.warning("フィラーは Irodori の声・応答は %s（声が揃わない）", Config.TTS_BACKEND)
        elif bank.voice_ref and Path(bank.voice_ref).resolve() != Path(
                resolve_repo_path(Config.IRODORI_VOICE_REF)).resolve():
            logger.warning("フィラーのバンクの話者（%s）と応答の話者（%s）が違う",
                           Path(bank.voice_ref).name, Config.IRODORI_VOICE_REF)
        player = FillerPlayer(
            bank, audio=self.audio, play_fn=self.player.play_fn,
            is_busy=lambda: self.runner.is_busy(),
            muted=lambda: bool(getattr(self.player, "muted", False)),
            # S5: 今の感情（チェーンを始める時に読む）。感情はこの後で作るので遅延 lambda
            emotion_fn=lambda: (self.emotion.label if getattr(self, "emotion", None) is not None else None),
            # フィラー前の待ち（ユーザ決定 2026-10-09「フィラーはA」・話の途中の間でフィラーを被せない）
            start_delay_sec=Config.FILLER_START_DELAY_SEC,
            # ログの秒を「話し終わり（声が止まった時）から」で出すため（話し終わりの判定 SILENCE_LIMIT の分を足す）
            judge_sec=Config.SILENCE_LIMIT,
        )
        # ゲートが閉じている間に応答の音声ができた＝畳む合図
        self.audio.on_head_waiting = player.on_head_ready
        logger.info("フィラー 稼働（%s・%d語・1語目は話し終わりから %.2f秒（判定 %.2f＋待ち %.2f）・つなぎの間 %.2f秒・"
                    "伸ばし棒 %s・感情の声 %s）",
                    bank_dir.name, len(bank.clips), Config.SILENCE_LIMIT + Config.FILLER_START_DELAY_SEC,
                    Config.SILENCE_LIMIT, Config.FILLER_START_DELAY_SEC, table.join_gap_sec,
                    "使う" if bank.stretch_ok() else "使わない",
                    "/".join(bank.by_emotion) or "なし")
        return player

    @staticmethod
    def _reply_opener() -> str:
        """S4: 応答の書き出しに置く応答依存フィラーの表（既定 off・読めなければ無しで続ける）。"""
        if not Config.FILLER_DEPENDENT_ENABLED:
            return ""
        try:
            from .filler.prompt import reply_opener_block
            from .response.tts_irodori import resolve_repo_path

            block = reply_opener_block(Path(resolve_repo_path(Config.FILLER_TABLE))) or ""
        except Exception as e:
            logger.warning("FILLER_DEPENDENT_ENABLED=1 だが表を読めないので無効: %s", e)
            return ""
        if block:
            logger.info("応答の書き出し（フィラーの表） 稼働（%d行を一例として渡す・fillers.yaml の opener_prompt）",
                        sum(1 for ln in block.splitlines() if ln.startswith("- ")))
        return block

    def _build_laugh(self):
        """S6 笑い（既定 off）。ストリーミング文字起こしの途中の文字を gpt-6-luna で判定する。"""
        if not Config.LAUGH_ENABLED:
            return None
        if self.filler is None:
            logger.warning("LAUGH_ENABLED=1 だがフィラーが無効なので笑いも無効（笑いはフィラーの仕組みで鳴らす）")
            return None
        if not getattr(self.stt, "streaming", False):
            logger.warning("LAUGH_ENABLED=1 だが文字起こしがストリーミングでないので笑いは無効"
                           "（途中の文字が要る・STT_BACKEND=openai_live）")
            return None
        bank = self.filler._bank
        if not bank.laughs and not bank.laugh_words:
            logger.warning("LAUGH_ENABLED=1 だが笑いのバンクが無いので笑いは無効（bank/<声>%s/ を作る）",
                           bank.table.laugh_bank_suffix)
            return None
        from .filler.laugh import LaughJudge, make_judge_fn

        t = bank.table
        judge = LaughJudge(
            judge_fn=make_judge_fn(self.registry, persona=self.persona), context_fn=self._laugh_context,
            on_immediate=lambda seg, p: self.filler.request_laugh(seg),
            on_filler=lambda seg, p: self.filler.laugh_in_filler(seg),
            immediate=t.laugh_immediate, filler=t.laugh_filler,
        )
        logger.info("笑い 稼働（%s・%s・その場で笑う %.2f 以上 / フィラーに笑い %.2f 以上（%s）・笑いの語 %d・声色 %d語）",
                    self.registry.resolve("laugh_judge"), "性格あり" if self.persona else "性格なし",
                    t.laugh_immediate, t.laugh_filler,
                    "笑いの声色" if t.laugh_filler_style == "tone" else "普通に笑う",
                    len(bank.laughs), len(bank.laugh_words))
        return judge

    def _build_emotion_judge(self):
        """S5 感情の判定（「、」「。」ごと）。ストリーミング文字起こしでなければ作らない（フィードバックだけで更新）。"""
        if self.emotion is None:
            return None
        if not getattr(self.stt, "streaming", False):
            logger.warning("EMOTION_ENABLED=1 だが文字起こしがストリーミングでないので「、」「。」ごとの判定は無効"
                           "（フィードバックでだけ更新・STT_BACKEND=openai_live で有効）")
            return None
        from .emotion import EmotionJudge, make_judge_fn

        logger.info("感情 稼働（%s・「、」「。」ごと＋フィードバックで更新・喜怒哀楽笑いを各0〜100で・1時間に %g ずつ薄れる）",
                    self.registry.resolve("emotion_judge"), Config.EMOTION_DECAY_PER_HOUR)
        return EmotionJudge(state=self.emotion, judge_fn=make_judge_fn(self.registry, persona=self.persona),
                            context_fn=self._laugh_context)

    def _build_mood(self):
        """S8: フィラーの1段目を場面に寄せる判定（Jev）。使えなければ理由をログに出して None（今までどおりの抽選）。"""
        if not Config.FILLER_MOOD_ENABLED:
            return None
        if self.filler is None:
            logger.warning("FILLER_MOOD_ENABLED=1 だがフィラーが無効なので場面の判定も無効")
            return None
        if not getattr(self.stt, "streaming", False):
            logger.warning("FILLER_MOOD_ENABLED=1 だが文字起こしがストリーミングでないので場面の判定は無効"
                           "（途中の文字が要る・STT_BACKEND=openai_live）")
            return None
        t = self.filler._bank.table
        if not t.mood_first:
            logger.warning("FILLER_MOOD_ENABLED=1 だが fillers.yaml に mood が無いので場面の判定は無効")
            return None
        if not Config.OPENROUTER_API_KEY:
            logger.warning("FILLER_MOOD_ENABLED=1 だが OPENROUTER_API_KEY が無いので場面の判定は無効（Jev は OpenRouter 経由）")
            return None
        from .filler.mood import JevClient, MoodJudge

        self._jev = JevClient(Config.OPENROUTER_API_KEY, t.mood_desc)
        emotion_fn = None
        if t.mood_with_emotion:
            emotion_fn = lambda: (self.emotion.label if getattr(self, "emotion", None) is not None else None)  # noqa: E731
        judge = MoodJudge(judge_fn=self._jev.ask, context_fn=self._laugh_context, categories=list(t.mood_first),
                          threshold=t.mood_threshold, emotion_fn=emotion_fn)
        self.filler.mood_fn = judge.mood
        logger.info("場面 稼働（Jev・「、」「。」ごと・%.2f 以上で1段目を %s に・それ以外は通常・%s）",
                    t.mood_threshold, "/".join(c for c in t.mood_first if c != "通常"),
                    "今の感情も渡す" if t.mood_with_emotion else "今の感情は渡さない")
        return judge

    def _build_voice_tone(self):
        """S5 声のトーン: 今の感情に合う Irodori の絵文字を返す関数（emotion.yaml・未決の間は None＝付けない）。

        Irodori の時だけ（VOICEVOX は絵文字を読み上げてしまう）。長さは絵文字を除いた文にそろえる（サーバの match_plain）。
        """
        if self.emotion is None or not hasattr(self.tts, "match_emoji_length"):
            return None
        from .emotion import load_voice_emoji
        from .response.tts_irodori import resolve_repo_path

        table = load_voice_emoji(Path(resolve_repo_path(Config.EMOTION_TABLE)))
        if not table:
            logger.info("声のトーン: emotion.yaml の voice_emoji が未決なので付けない")
            return None
        emo = self.emotion
        logger.info("声のトーン: %s（いちばん強い感情が %g 以上の時だけ・長さは絵文字なしにそろえる）",
                    " / ".join(f"{k}={v}" for k, v in table.items()), Config.EMOTION_TONE_MIN)
        return lambda: table.get(emo.dominant())

    def _on_phrase(self, seg: int, text: str) -> None:
        """途中の文字（「、」「。」ごと・確定時）を笑い判定と感情の判定へ配る（1つの失敗で他を止めない）。"""
        for judge in (self.laugh, self.emotion_judge, getattr(self, "mood", None)):
            if judge is None:
                continue
            try:
                judge.on_phrase(seg, text)
            except Exception:
                logger.exception("途中の文字の判定で例外（無視して継続）")

    def _laugh_context(self) -> list:
        """笑い・感情の判定に渡す直前の会話（実測と同じく最大10ターン・話者つき）。"""
        turns = [t for t in self.cache.recent() if t.speaker != "__omitted__"][-10:]
        return [f"{'イブ' if t.speaker == 'eve' else 'ユーザ'}: {t.text}" for t in turns]

    def _filler_active(self) -> bool:
        return self.filler is not None and self.filler.active

    def _busy_for_autospeech(self) -> bool:
        """自発発話のガード: 応答中・フィラー中は発火しない（S3）。

        Alchemy の部屋で待っている間も発話判定は止めない（ユーザ 2026-10-11「待機中も発話判定はON。1分半で催促してもいい。
        問題は5分以上待たせても居座り続ける不自然さ」）。5分の最終確認と抜けるのは待ちの見張り（`AlchemyManager`）。
        """
        return self.runner.is_busy() or self._filler_active()

    def _on_speech_end(self, speech_sec: float, seg: int = 0) -> None:
        if self.filler is not None:
            self.filler.on_speech_end(speech_sec, seg)

    def _on_stt_result(self, put: bool) -> None:
        if self.filler is not None:
            self.filler.on_stt_result(put)

    def _on_response_near(self) -> None:
        if self.filler is not None:
            self.filler.on_response_near()

    def _barge_in(self, why: str = "発話検知（割り込み）") -> None:
        """発話開始の瞬間: 次の節から鳴らさない＋進行中応答キャンセル（Eve が譲る）。

        鳴っている節（「、」「。」までの1項目）は言い切ってから黙る（ユーザ裁定 2026-09-22
        「今まではユーザーしゃべったら速攻黙ったけど不自然。、。までは喋らせる」・`AudioPlayQueue`）。
        """
        logger.info("⏸ %s", why)
        if self.filler is not None:
            self.filler.on_barge_in()  # 鳴っている語は言い切り、次の語は始めない
        self.audio.interrupt()
        self.runner.interrupt()  # 進行中応答(自発含む)を cancel＝実発話分のみ記録(C5)
        # F5: ユーザ優先。判定中の自発発話は seq 変化で自己破棄、キュー済みの自発刺激は削除。
        self.speech_state.mark_user_speech_start()
        self.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)

    def _yield_if_user_speaking(self) -> None:
        """イブの音が鳴り始める瞬間: ユーザがまだ話している途中なら、この節を言い切って譲る。

        ユーザ裁定 2026-09-26「話の途中で応答が始まってもいい。0.5秒も黙れば人間でも話し終わったと
        思って話し始めてしまう。相手がまだ話したら、。まで話して再度割り込ませればいい」。
        割り込みは発話の**始まり**でしか起きない。間（`SILENCE_LIMIT`）を置いて話し続けると VAD が
        発話を2つに分け、前半への応答が後半の途中で鳴り始める（後半の始まりの割り込みは、前半への
        応答ができる前に済んでいる）ので、鳴り始めるここで割り込みをかけ直す。鳴り始めた節は
        言い切る（`AudioPlayQueue` は鳴っている項目を止めない）＝「、。まで話して」譲る。
        `AudioPlayQueue` の再生開始通知（loop 上・同期）から呼ばれる。
        """
        try:
            speaking = self.input.user_in_speech()
        except Exception:
            logger.exception("話している途中かの確認で例外（譲らずに続ける）")
            return
        if speaking:
            self._barge_in("まだ話している途中で鳴り始めた（この節を言い切って譲る）")

    def submit_user_text(self, text: str) -> None:
        """キーボードで打った文を**マイク発話と同じ経路**でパイプラインへ入れる。

        **loop 上で呼ぶこと**（UI スレッドからは `call_soon_threadsafe` 経由）。
        マイク経路（`response/input_source.py:78-96`）と同じ順序で state を進める:
        1. `_barge_in()` — 喋っている途中なら譲る。口頭で割り込むのと挙動を揃える
           （黙って待たせると「打ったのに無視された」ように見える）。キュー済みの
           自発発話も捨てる。
        2. `mark_user_utterance()` — `_barge_in` が立てた `user_speaking` を戻し、
           沈黙時計をリセットする。これを忘れると自律発話が永久に止まる。
        3. 刺激を投入。会話履歴への記録は `response/orchestrator.py:378` が行う
           （マイク由来と同じ扱い＝記憶の中で区別されない）。

        STT を通らないので `stt_pending` の窓は開かない（待つ理由が無い）。
        """
        text = (text or "").strip()
        if not text:
            return
        self._barge_in()
        self.speech_state.mark_user_utterance()
        logger.info("⌨ %s", text)
        task = asyncio.create_task(self.queue.put(
            Stimulus(StimulusKind.USER_UTTERANCE, payload=text)))
        # 参照を保持する（create_task の戻り値を捨てると GC で消えうる）
        self._ui_tasks.add(task)
        task.add_done_callback(self._ui_tasks.discard)

    # --- YouTube 配信モード（`eve/stream/`）-----------------------------------

    def _free_for_comment(self) -> bool:
        """配信のコメントを渡してよい間か（イブが話していない・考えていない・ユーザ（持ち主）が話していない）。

        コメントは割り込まない（話している途中のイブを止めない・ユーザ指定は「音声合成完了後に最新コメントを取得」）。
        待っている自発発話の下書きはコメントに譲る（渡す時に捨てる＝`submit_stream_comment`）ので、それだけは数えない。
        """
        if self.runner.is_paused() or self.runner.is_busy():
            return False
        if not self.audio.is_quiet() or self.audio.is_playing():
            return False
        if self.speech_state.user_speaking or self.speech_state.stt_pending or self._filler_active():
            return False
        return all(s.kind == StimulusKind.AUTONOMOUS_SPEECH for s in self.queue.snapshot())

    def submit_stream_comment(self, text: str, dedup_key: str) -> None:
        """配信のコメントをユーザ発話として入れる（**loop 上・手が空いた時だけ呼ばれる＝割り込みはしない**）。

        `submit_user_text` と違って `_barge_in` を呼ばない。沈黙時計を戻し（＝判定中の自発発話は seq の変化で捨てられる）、
        判定済みでまだ話していない自発発話の下書きはコメントに譲って捨てる。会話の記録は orchestrator が行う。
        """
        self.speech_state.mark_user_utterance()
        self.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)
        task = asyncio.create_task(self.queue.put(
            Stimulus(StimulusKind.USER_UTTERANCE, payload=text, dedup_key=dedup_key)))
        self._ui_tasks.add(task)
        task.add_done_callback(self._ui_tasks.discard)

    async def _stream_session_start(self, wait_sec: float = 20.0) -> None:
        """配信の前: 直前の会話の内省を済ませてから会話に区切りを入れる（配信前の私的な会話を配信の文脈に入れない）。

        内省（直近フィードバック）は区切りより前の分を応答・発話判定に入れない（orchestrator / decider）。先に済ませて
        おかないと、配信の最初の内省が配信前の会話までまとめて要約し、それが配信の文脈に入ってしまう。
        """
        self.feedback_worker.trigger()
        t0 = now_mono()
        while now_mono() - t0 < wait_sec:
            await asyncio.sleep(0.3)
            if not self.cache.turns_since(self.prediction.watermark) and self.feedback_worker.is_idle():
                break
        else:
            logger.warning("配信前の会話の内省が %.0f 秒で終わらなかった（区切りは入れて配信を始める）", wait_sec)
        now = now_iso()
        # 区切りの手前30分に話していたことの要約を、配信中の返事に渡す（会話そのものは入れない＝前の短い返事に長さが
        # 引っ張られない・私的なことを言いにくい。入れないと「さっきの話」を忘れる。実測 2026-10-16
        # `tools/prompt_eval/stream_switch.py`: 覚えていた 区切る 0/4・混ぜる 4/4・要約 4/4／最初のコメントへの返事
        # 74・52・90字／私的な語 0/4・2/4・0/4）
        self._carry = {"kind": "pre", "since": _iso_shift(now, -CARRY_PRE_SEC), "until": now}
        self.cache.mark_boundary()

    def _stream_live(self) -> None:
        """配信が始まった（ここから先の判定には配信の事実が入る）: それまでに決めた自発発話は捨てる。

        実測 2026-10-15: 配信の準備中（OBS の用意の数秒）に決めた最初の話題が、配信の事実なしに配信前の持ち主との会話から
        選ばれ、配信の最初の一言になった。判定中の分は seq で、決まって待っている分はキューから捨てる。
        """
        self.speech_state.mark_session_start()
        self.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)

    def _stream_session_end(self) -> None:
        """配信の後: 会話に区切りを入れる（配信のコメントを、配信後の持ち主との会話の文脈に入れない）。

        配信中に決めた（視聴者に向けた）自発発話も、配信が終わった後には話さない。
        配信の様子は要約だけを、配信の後しばらく（CARRY_POST_SEC）返事に渡す（「配信どうだった？」に答えられるように）。
        """
        st = self.stream
        if st is not None and st.started_iso:
            note = (f"配信は約{_minutes(st.elapsed_sec())}分・コメント {st.total}件（{len(st.people)}人）。"
                    if st.started_mono else "")
            self._carry = {"kind": "post", "since": st.started_iso, "until": now_iso(), "note": note,
                           "expires": now_mono() + CARRY_POST_SEC}
        self.cache.mark_boundary()
        self.speech_state.mark_session_start()
        self.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)

    def _carry_text(self) -> Optional[str]:
        """区切りの手前で話していたことの要約（見出し込み）。配信中は配信の前の会話、配信の後しばらくは配信のこと。"""
        c = self._carry
        if c is None:
            return None
        streaming = self.stream is not None and self.stream.active
        if c["kind"] == "pre" and not streaming:
            return None
        if c["kind"] == "post":
            if streaming:
                return None
            if now_mono() > c["expires"]:
                self._carry = None
                return None
        lines = self.rag.summaries_between(c["since"], c["until"], k=CARRY_MAX)
        if not lines and not c.get("note"):
            return None
        head = ("# 配信の直前に開発者と話していたこと（要約・視聴者はこの会話を知らない）" if c["kind"] == "pre"
                else "# さっきまでの配信（要約）")
        body = "\n".join(f"- {s}" for s in lines)
        return "\n".join(x for x in (head, body, c.get("note") or "") if x)

    def _stream_ended(self, reason: str) -> None:
        """OBS で止めた・YouTube で終えた等、外で配信が終わった（loop 上）。機能の戻しは ControlPlane に任せる。"""
        if self.control is not None:
            self.control.request("stream", False)
            return
        task = asyncio.create_task(self.stream.stop())  # UI が無い（ControlPlane が無い）時はここで止める
        self._ui_tasks.add(task)
        task.add_done_callback(self._ui_tasks.discard)

    def _on_response_complete(self) -> None:
        """応答 正常完了: F4 feedback を起こす + F5 沈黙時計をリセット（Eve が喋った）。"""
        self.feedback_worker.trigger()
        if self.self_model_worker is not None:
            self.self_model_worker.trigger()  # 合図が無ければ LLM は呼ばれない（trigger.py）
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
        # Alchemy の対戦中（画面認識がイブのゲーム画面を見ている間）は、画面の変化では話しかけを起こさない。
        # 演出で画面は絶えず変わるので、毎行動の実況になる（ユーザ指定「毎行動うるさく反応するのはうざい」）。
        # ゲームの話しかけは、大きく動いた時の合図（`_alchemy_react`）だけが起こす
        if self.alchemy is not None and self.alchemy.owns_screen():
            return False
        return self._quiet_for_speech()

    def _quiet_for_speech(self) -> bool:
        """自分から話しかけてよい間か（応答中・ユーザ発話中・STT待ち・判定処理中・フィラー中でない）。"""
        return (
            not self.runner.is_busy()
            and not self.speech_state.user_speaking
            and not self.speech_state.stt_pending
            and self.speech_decider.is_idle()
            and not self._filler_active()  # S3: フィラー中は実況を差し込まない
        )

    # --- Alchemy（ユーザ作のゲーム）------------------------------------------

    def _alchemy_report(self, name: str, content: str, ok: bool) -> None:
        """部屋に入れた・入れなかった等を、機能の報告として応答AI へ（loop 上・同期）。"""
        stim = Stimulus(
            StimulusKind.CALLFUNCTION_RESULT,
            payload=CallFunctionResult(function_name=name, content=content, ok=ok),
            dedup_key=f"alchemy:{name}:{now_mono():.3f}",
        )
        task = asyncio.create_task(self.queue.put(stim))
        self._ui_tasks.add(task)  # 参照を保持（stop() で片付く）
        task.add_done_callback(self._ui_tasks.discard)

    def _alchemy_react(self, text: str) -> None:
        """ゲームが大きく動いた: 発話判定を起こす（話すかどうかは発話判定が決める）。

        応答中・ユーザが話している間は起こさずに少し待つ（割り込まない）。待っても間が来なければ見送る
        （出来事は `# Alchemy` の「最近の出来事」に残るので、次に話す時の材料にはなる）。
        """
        if self._alchemy_react_task is not None and not self._alchemy_react_task.done():
            return  # 既に待っている（発話判定は最新の出来事を読む）
        self._alchemy_react_task = asyncio.create_task(self._alchemy_react_when_quiet())
        self._ui_tasks.add(self._alchemy_react_task)
        self._alchemy_react_task.add_done_callback(self._ui_tasks.discard)

    async def _alchemy_react_when_quiet(self, max_wait: float = 15.0) -> None:
        t0 = now_mono()
        while now_mono() - t0 < max_wait:
            if self._quiet_for_speech():
                self.speech_decider.trigger("game")
                return
            await asyncio.sleep(0.3)
        logger.info("🎮 話せる間が来なかったので、出来事への反応は見送った")

    def _alchemy_say(self, content: str, reason: str) -> None:
        """Alchemy の部屋で待っている時の催促・見切り: 発話判定を通さずにイブが自分から話す（何を言うかは応答AI）。

        「いつ」はユーザ指定（5分待って来なければ確かめる・30秒返事が無ければ抜ける）なので判定AI に委ねない。
        既出の用件に載せる（発話判定が同じ催促を重ねない）。
        """
        from .speech.decider import AutonomousSpeech

        async def _put() -> None:
            t0 = now_mono()
            while not self._quiet_for_speech() and now_mono() - t0 < 15.0:
                await asyncio.sleep(0.3)
            self.speech_state.record_prior_item(content, played=True)
            await self.queue.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, payload=AutonomousSpeech(content, reason)))

        task = asyncio.create_task(_put())
        self._ui_tasks.add(task)
        task.add_done_callback(self._ui_tasks.discard)

    async def _alchemy_screen(self, feed) -> None:
        """画面認識の見る先を切り替える（feed＝イブのゲーム画面 / None＝デスクトップに戻す）。

        ユーザ指定「画面認識はAIが操作するはずの画面を認識させる。ユーザーのPC画面は認識させない」。
        動いていれば止めてから差し替えて動かし直す（`stop_vision` が画面の記憶も消す＝デスクトップの実況を
        対戦中に持ち込まない・逆も同じ）。画面認識を UI で切っている間は差し替えだけ（点けた時にこの先を見る）。
        """
        running = self.capture_thread is not None
        if running:
            await self.stop_vision()
        if feed is not None:
            self.vlm_capture = ScreenCapture(
                monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
                jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD,
                grab_fn=feed.grab,
            )
        else:
            self.vlm_capture = self._desktop_capture
        if running:
            self.start_vision()

    def _progress(self, key: str, text: str) -> None:
        """起動の進み具合を UI へ知らせる（既定 None＝ヘッドレスでは何もしない）。

        同期・非ブロッキング契約（UI 側は Qt signal を emit するだけ）。例外は握る
        ＝表示のための機能が起動そのものを壊さない。
        """
        if self.on_progress is None:
            return
        try:
            self.on_progress(key, text)
        except Exception:
            logger.exception("起動進捗の通知で例外（起動は継続）")

    async def warmup(self) -> None:
        """STT/RAG/LLM/TTS の cold-start を**並列に**潰す。

        直列だと合計 15〜36秒かかっていた（実測8ラン・2026-08-14）。4つは互いに独立で、
        しかも重い RAG（Ruri のロード）は `run_in_executor` の中で動く＝**ループを塞がない**
        ので、`gather` すれば実質「いちばん遅い1つ」ぶんで済む。
        失敗は各ステップ内で握るので `gather` に例外は上がらない（1つ落ちても他は進む）。
        """
        logger.info("ウォームアップ開始（並列）")
        t0 = now_mono()
        # **重い import だけは先に直列で済ませる**（実起動で踏んだ）。
        # litellm と openai SDK はどちらも pydantic を遅延 import するため、別スレッドから
        # 同時に触ると partially-initialized module の競合になる:
        #   「cannot import name 'import_string' from partially initialized module
        #     pydantic._internal._validators (most likely due to a circular import)」
        #   「module 'pydantic' has no attribute 'BaseModel'」
        # 直列だった頃は起きなかった＝並列化が持ち込んだ問題なので、並列化の内側で塞ぐ。
        await asyncio.to_thread(_preimport_sdks)

        async def step(key: str, label: str, coro_factory) -> None:
            s = now_mono()
            self._progress(key, f"{label}を準備中…")
            try:
                await coro_factory()
                ok = True
            except Exception as e:
                logger.warning("%s ウォームアップ失敗（続行）: %s", label, e)
                ok = False
            d = now_mono() - s
            logger.info("ウォームアップ %s: %.1fs%s", label, d, "" if ok else "（失敗・続行）")
            self._progress(key, f"{label} {'OK' if ok else '失敗（続行）'} {d:.1f}s")

        async def embed_warmup() -> None:
            # 埋め込みモデル（Ruri）のロードの後で、話し方の実例の話しかけを埋め込んでおく（似た場面の除外に使う）。
            # 並べて走らせるとモデルを2回ロードしうるので、同じ手順の中で順に
            await self.rag.warmup()
            examples = getattr(self, "style_examples", None)
            if examples is not None:
                await examples.prepare()

        extra = []
        laugh = getattr(self, "laugh", None)
        if laugh is not None:
            # S6: 初回の判定だけ約4.7秒かかった（実測 2026-09-26・以降は約0.7秒）。最初の「、」で遅れないよう温める
            from .filler.laugh import make_judge_fn

            extra.append(step("laugh", "笑い判定",
                              lambda: make_judge_fn(self.registry, persona=self.persona)(
                                  ["ユーザ: こんにちは"], "こんにちは。")))
        if getattr(self, "mood", None) is not None:
            # S8 Jev（OpenRouter）の接続を温める（初回だけ TLS の分遅い）
            extra.append(step("mood", "場面の判定",
                              lambda: self._jev.ask(["ユーザ: こんにちは"], "こんにちは。", None)))
        if getattr(self, "emotion_judge", None) is not None:
            # 感情の判定も同じモデルの初回だけ遅い（笑い判定と同じ理由）。形が違う呼び出しなので別に温める
            from .emotion import make_judge_fn as make_emotion_judge_fn

            extra.append(step("emotion", "感情の判定",
                              lambda: make_emotion_judge_fn(self.registry, persona=self.persona)(["ユーザ: こんにちは"], "こんにちは。", None)))
        await asyncio.gather(
            *extra,
            step("stt", "音声認識", self.stt.warmup),
            # 埋め込みモデル（Ruri）の初回ロード。実測でここが最長になりやすい（＋話し方の実例の準備）
            step("rag", "記憶（埋め込み）", embed_warmup),
            # max_tokens は付けない: reasoning 系(gpt-5.x)は1トークンで完了できず BadRequest
            # 警告になる。warmup は一度きり・"hi" への短応答なのでコストは無視できる。
            step("llm", "応答モデル",
                 lambda: self.registry.complete("response", [{"role": "user", "content": "hi"}])),
            # VOICEVOX は「。」を1回合成 / Irodori はサーバを用意（無ければ起動してロードを待つ）
            step("tts", "音声合成", self.tts.warmup),
            # 出力を開いておく（開きっぱなし・1回目の応答で開く約0.2秒を払わない）。
            # レートが違えば最初の再生で開き直すだけ（Irodori 48kHz / VOICEVOX 24kHz）
            step("player", "音声出力",
                 lambda: asyncio.to_thread(self.player.prepare, self._output_rate())),
        )
        logger.info("ウォームアップ完了 %.1fs", now_mono() - t0)
        self._progress("warmup", f"ウォームアップ完了 {now_mono() - t0:.1f}秒")

    def _build_style_examples(self):
        """話し方の実例（`eve/persona_examples.py`）。ファイルが無い・空・数が 0 なら None。"""
        if Config.PERSONA_EXAMPLES_K <= 0:
            return None
        examples = load_examples()
        if not examples:
            logger.info("話し方の実例: なし（%s が無いか空）", examples_path())
            return None
        # 似た場面の除外のしきい値は Ruri で校正した値（言い換え 50/50 を 0.88 で捕まえる）。
        # 別の埋め込みでは値の分布が違うので、文字の似かたで代用する
        emb = self.embedder if (Config.EMBED_BACKEND or "ruri").strip().lower() == "ruri" else None
        logger.info("話し方の実例: %d組から毎回%d組（似た場面の除外=%s）", len(examples),
                    Config.PERSONA_EXAMPLES_K, "意味の近さ（Ruri）" if emb is not None else "文字の似かた")
        return StyleExamples(examples, k=Config.PERSONA_EXAMPLES_K, embedder=emb)

    @staticmethod
    def _output_rate() -> int:
        """応答の音声のサンプルレート（出力を先に開くため）。フィラーのバンクも Irodori＝48kHz。"""
        return 48000 if (Config.TTS_BACKEND or "").strip().lower() == "irodori" else 24000

    # --- 画面認識の起動/停止（起動時と UI の ON/OFF の両方から使う）------------

    def _build_text_gate(self):
        """文字ゲートを作る（OCR が使えなければ None）。

        pHash は枠内の文字変化を拾えない（実測 pHash 一致 15/15 ペアで画素完全一致 0/15）。
        OCR の読み取りは **LLM に一切渡さない**（外に出るのは bool だけ＝誤読がハルシネの元）。
        `WindowsOcr()` の初期化は winsdk を触るので **loop 上で**行う（キャプチャスレッドに
        持ち込まない）。ON/OFF のたびに作り直すのは、前回テキストを持ち越すと OFF 中の変化を
        ON 直後に「今変わった」として拾ってしまうため。
        """
        if not Config.VLM_OCR_ENABLED:
            return None
        from .vlm.text_gate import ScreenTextGate, WindowsOcr

        ocr = WindowsOcr()
        if not ocr.available:
            # fail-soft（ddgs 未導入時と同流儀）: 文字ゲートだけ無効化して続行。
            logger.warning("VLM_OCR_ENABLED=1 だが OCR を初期化できないため文字ゲートを"
                           "無効化（pip install winsdk / OCR 言語パックを確認）")
            return None
        cap = self.vlm_capture
        gate = ScreenTextGate(
            ocr=ocr,
            grab_png=lambda: cap.grab_png(Config.VLM_OCR_DOWNSCALE),
            diff_threshold_pct=Config.VLM_OCR_DIFF_PCT,
            interval_sec=Config.VLM_OCR_INTERVAL_SEC,
        )
        logger.info("文字ゲート 稼働（%s・%.0f%%以上の変化・最短%.0f秒間隔）",
                    ocr.lang, Config.VLM_OCR_DIFF_PCT, Config.VLM_OCR_INTERVAL_SEC)
        return gate

    def start_vision(self) -> None:
        """VLM worker + capture スレッドを起動する（loop 上から呼ぶ・二重起動しない）。"""
        if self.capture_thread is not None:
            return
        self.vlm_worker.start()
        self.capture_thread = CaptureThread(
            capture=self.vlm_capture, change_detector=self.vlm_change_detector,
            deliver=self.vlm_worker.on_frame, loop=asyncio.get_running_loop(),
            target_fps=Config.VLM_TARGET_FPS, text_gate=self._build_text_gate(),
        )
        self.capture_thread.start()
        logger.info("画面認識(VLM) 稼働")

    async def stop_vision(self) -> None:
        """capture スレッド → worker の順で止め、**画面の記憶を消す**。

        順序は `stop()` と同じ理由: 閉じるループへフレームを送らせない。
        state を消すのは、OFF 後も `latest_vision` が残ると「今の画面」として古い実況を
        語り続けるため（VisionState 自体にも鮮度ガードはあるが、OFF は即座に効かせる）。
        """
        th, self.capture_thread = self.capture_thread, None
        if th is not None:
            try:
                await asyncio.to_thread(th.stop)  # join をループ外で
            except Exception:
                logger.exception("capture スレッドの停止で例外（続行）")
        try:
            await self.vlm_worker.stop()
        except Exception:
            logger.exception("VLM worker の停止で例外（続行）")
        self.vision_state.ring.clear()
        self.vision_state.latest_vision = None
        self.vision_state.latest_vision_mono = 0.0
        self.vision_state.latest_is_narration = False
        self.vision_state.last_frame_mono = 0.0
        self.vision_state.last_change_mono = 0.0
        # 変化検出の内部状態（前回 pHash）も捨てる。持ち越すと OFF 中に起きた変化を
        # ON 直後の1枚目と比較して「今変わった」と誤って拾う。
        # worker 側の前回実況も捨てる。残すと ON 直後の実況が重複判定で消える。
        # （文字ゲートは start_vision() が毎回作り直すので持ち越さない。）
        for name, obj in (("ChangeDetector", self.vlm_change_detector), ("VlmWorker", self.vlm_worker)):
            try:
                obj.reset()
            except Exception:
                logger.exception("%s のリセットで例外（続行）", name)

    async def run(self) -> None:
        self._progress("memory", "記憶を読み込み中…")
        await self.cache.initialize()  # 既存ログ復元 + 書き込み worker 起動
        await self.rag.initialize()  # 既存 RAG 記憶を復元 + 書き込み worker 起動
        if self.x_memory is not None:
            await self.x_memory.initialize()  # X で見たこと（別ファイル）
        x_note = f" / Xで見た {len(self.x_memory)}" if self.x_memory is not None else ""
        self._progress("memory", f"記憶 OK（会話 {len(self.cache.recent())} / RAG {len(self.rag)}{x_note}）")
        # F4 起動時 catch-up: watermark を永続 RAG の最新 timestamp から復元し、
        # それより新しい復元会話（前回 feedback 途中で落ちた tail）を1回取り戻す。
        self.prediction.watermark = self.rag.latest_timestamp()
        self.feedback_worker.start()
        if self.cache.turns_since(self.prediction.watermark):
            self.feedback_worker.trigger()
        if Config.TTS_BACKEND.strip().lower() == "irodori":
            # 前回の EVE が残した Irodori サーバ（持ち主がもういない）を**ウォームアップの前に**止める。
            # 後だと、残骸を「既にあるサーバ」として使ってしまい、終了時に誰も止めない。
            try:
                n = await asyncio.to_thread(sweep_orphan_tts_servers)
                if n:
                    logger.info("🧹 前回の Irodori サーバを %d 個止めた", n)
            except Exception:
                logger.warning("Irodori サーバの掃除に失敗（続行）", exc_info=True)
        await self.warmup()
        self._progress("features", "機能を起動中…")
        self._tasks.append(asyncio.create_task(self.audio.play_worker()))
        self._tasks.append(asyncio.create_task(self.runner.run()))
        if self.filler is not None:
            self.filler.start()  # 常駐1本（ターンごとにタスクを乱立させない）
        # F5: 発話判定 worker + 沈黙監視を起動（沈黙時計の baseline をここでリセット）。
        # `AUTOSPEECH_ENABLED=0` なら起動しない（UI メニューから後で ON にできる）。
        self.speech_state.mark_eve_activity()
        if Config.AUTOSPEECH_ENABLED:
            self.speech_decider.start()
            self.silence_monitor.start()
        else:
            logger.info("自律発話は起動時 OFF（AUTOSPEECH_ENABLED=0）")
        # J: Call-Function 実行サイドカー（既定 off）。read-only 能力のみ。
        if Config.CALLFUNCTION_ENABLED:
            self.dispatcher.start()
            self.delivery_checker.start()  # J-2 P1-1: 報告ターンの配達確認
            logger.info("Call-Function 稼働（read-only 能力）")
        # 自己形成（好み・趣向の台帳）。合図が出た時だけ LLM を呼ぶ＝普段は無音のサイドカー。
        if self.self_model_worker is not None:
            self.self_model_worker.start()
            logger.info("自己形成 稼働（%s・%d項目）",
                        self.self_model.path, self.self_model.topics)
        # X の仕入れ（既定 off）。起動直後は待ってから、前回から間が空いていれば1回取り戻す
        if self.x_collector is not None:
            self.x_collector.start()
            wait = self.x_collector.seconds_until_due()
            logger.info("🐦 X の仕入れ 稼働（%g時間ごと・%d件・タイムライン %s・次は%s）",
                        Config.X_INTERVAL_HOURS, Config.X_PER_RUN,
                        "あり" if self.x_collector.has_timeline else "なし",
                        "起動の少し後" if wait <= 0 else f"約{wait / 3600:.1f}時間後")
        # J-1: タスク管理（既定 off）。store 復元 + executor/scheduler 起動。
        if self.task_store is not None:
            await self.task_store.initialize()
            # 起動時の掃除（2026-08-08 ユーザ裁定）: **executor を起動する前**に未終了タスクを
            # 片付ける。後だと reconcile が前回の Pending を掴んで走り出し、1件制では
            # 起動直後からユーザの最初の依頼が断られる。異常終了した前回の残りもここで回収する。
            _swept = cleanup_sweep(task_store=self.task_store, workspace=self.workspace,
                                   phase="startup")
            if any(_swept.values()):
                logger.info("🧹 起動時の掃除: %s", _swept)
            self.task_executor.start()
            self.reconcile_timer.start()
            if self.cancel_resolver is not None:
                self.cancel_resolver.start()  # executor と並列（取消が実行中タスクの後ろで待たない）
            logger.info("タスク管理 稼働（予約タスク）")
            if self.search_client is not None:
                logger.info("Web検索 稼働（search_web・TaskAgent 専用）")
            if self.workspace is not None:
                # mkdir + 起動時フルスキャン（スキャンは to_thread＝loop を塞がない）。
                await self.workspace.ensure_ready()
                logger.info("作業フォルダ 稼働（%s・%s）",
                            self.workspace.paths.root, self.workspace.usage())
                if Config.WS_EXEC_ENABLED:
                    logger.info("プログラム実行 稼働（run_program・%s）", Config.WS_EXEC_ALLOW)
                if Config.WS_VIEWER_ENABLED:
                    logger.info("画面表示 稼働（show_on_screen・別ウィンドウ）")
        # Alchemy（既定 off）。ブラウザは遊ぶ時まで開かない（ここでは準備の期限の見張りだけ）
        if self.alchemy is not None:
            self.alchemy.start()
        # F6: 画面認識を起動（既定 off）。capture スレッドは loop 確定後に生成し on_frame を橋渡し。
        if Config.VLM_ENABLED:
            self.start_vision()
        self._progress("mic", "マイクを開いています…")
        await self.input.start()  # mic + STT 消費タスクを起動
        logger.info("VoiceLoop 稼働。話しかけてください。")
        self._progress("ready", "準備完了。話しかけてください")
        await asyncio.Event().wait()  # キャンセルされるまで稼働

    async def stop(self) -> None:
        self.input.stop()
        if self.stream is not None:
            # 配信中なら OBS の配信を止める（イブが居なくなった画面を流し続けない）・OBS の音を戻す
            try:
                await self.stream.shutdown()
            except Exception:
                logger.exception("配信の後片付けで例外（続行）")
        if self.laugh is not None:
            try:
                await self.laugh.stop()  # 判定中の問い合わせを止める
            except Exception:
                pass
        if getattr(self, "emotion_judge", None) is not None:
            try:
                await self.emotion_judge.stop()
            except Exception:
                pass
        if getattr(self, "mood", None) is not None:
            try:
                await self.mood.stop()
                await self._jev.close()
            except Exception:
                pass
        if self.filler is not None:
            try:
                await self.filler.stop()  # 発話権ゲートも必ず開ける
            except Exception:
                pass
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
        # Alchemy: 部屋から抜けてイブのゲーム画面（ブラウザ）を閉じる（画面認識は上で止めてある）
        if self.alchemy is not None:
            try:
                await self.alchemy.stop()
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
        # 自己形成サイドカー（好み・趣向の台帳）。止まらなくても本流に影響しない。
        if self.self_model_worker is not None:
            try:
                await self.self_model_worker.stop()
            except Exception:
                pass
        # X の仕入れ（途中なら打ち切る。記憶に入れた分と状態は1件ごとに保存済み）
        if self.x_collector is not None:
            try:
                await self.x_collector.stop()
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
            # 終了時の掃除（2026-08-08 ユーザ裁定）: **executor を止めた後・store を閉じる前**。
            # 未終了タスクを terminal にしておかないと、次の起動で 1件制の枠を塞ぐ。
            # 異常終了ではここに来ないが、その分は起動時の掃除が回収する（両端でやる理由）。
            try:
                swept = cleanup_sweep(task_store=self.task_store, workspace=self.workspace,
                                      phase="shutdown")
                if any(swept.values()):
                    logger.info("🧹 終了時の掃除: %s", swept)
            except Exception:
                pass
            try:
                await self.task_store.shutdown()
            except Exception:
                pass
        if self.workspace is not None:
            # E2: 開いたビューアを必ず閉じる。残すと**ランを跨いで画面に居座り**、次セッションの
            # VLM がそれを「ユーザの画面」として拾って、イブが自分の過去の出力を相手の作業画面
            # として実況する（実測 2/3ラン）。窓が消える不便より自己観察ループの実害を取る。
            try:
                # `force=True`: ユーザが操作中の窓もここでは閉じる（プロセスごと終わる
                # ので残しても意味が無く、別プロセスのビューアだけ孤児になる）。
                n = self.workspace.close_viewers(force=True)
                if n:
                    logger.info("表示ウィンドウを %d 個閉じた", n)
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
        for t in list(self._ui_tasks):  # UI 投入の孤児化防止
            t.cancel()
        try:
            await self.cache.shutdown()  # 書き込みキューをドレイン（記録を取りこぼさない）
        except Exception:
            pass
        try:
            await self.rag.shutdown()
        except Exception:
            pass
        if self.x_memory is not None:
            try:
                await self.x_memory.shutdown()  # X で見た記憶の書き込みキューを流しきる
            except Exception:
                pass
        try:
            await self.tts.close()
        except Exception:
            pass
        try:
            await self.stt.close()  # ストリーミング文字起こしの接続を閉じる（一括方式は何もしない）
        except Exception:
            pass
        self.player.close()
        if self.usage is not None:
            # 終了時に必ず総額を出す（ログを掘らなくても「今のランでいくら使ったか」が分かる）。
            try:
                logger.info("💰 このランの API 使用量\n%s", self.usage.format_summary())
            except Exception:
                logger.exception("使用量の集計に失敗（無視）")
