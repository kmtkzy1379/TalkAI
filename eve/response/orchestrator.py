"""ResponseOrchestrator — 応答の背骨（pipeline.Orchestrator Protocol を実装）。

刺激 → ContextAssembler で messages → stream_fn で token stream → SentenceSplitter で文 →
文ごとに seq 予約 + TTS(並行,上限) → AudioPlayQueue へ enqueue（seq 再整列）。
PipelineRunner には StubOrchestrator の代わりにこれを注入する。

barge-in（応答側）: stream 中に世代が変わったら生成停止。TTS 完了時にも世代を確認して
古い世代の音声は enqueue しない（AudioPlayQueue 側でも世代で破棄される＝二重の安全）。

依存性注入（stream_fn / tts_fn）で実 API とテスト用フェイクを切替（Tier-1 は実依存ゼロ）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Optional

from ..config import Config
from ..context_assembler import OMITTED_SPEAKER, ContextAssembler, RagChunk, Turn
from ..pipeline.audio_play_queue import AudioPlayQueue
from ..pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
from ..speech.decider import AutonomousSpeech
from .splitter import JapaneseSentenceSplitter
from .style import SPEECH_STYLE, sanitize_for_speech

if TYPE_CHECKING:  # 実行時の循環 import を避ける（型注釈のみ）
    from ..feedback.prediction_state import PredictionState
    from ..memory.conversation_cache import ConversationCache
    from ..memory.long_term import RagStore

logger = logging.getLogger(__name__)

StreamFn = Callable[[list[dict]], AsyncIterator[str]]  # messages -> 文字列デルタの async iter
TtsFn = Callable[[str], Awaitable[Optional[bytes]]]  # 文 -> 音声バイト(or None)

# 再配達（2026-07-13 21:20 実機事故対応）: barge-in で潰れたタスク報告の救済条件。
# 実発話（C5: 再生開始した文は途中停止でも「聞かれた扱い」）がこの文字数以下なら
# 「実質何も伝わっていない」とみなし再配達する。超えていれば配達済み扱い（二重発話防止）。
REDELIVER_MAX_SPOKEN_CHARS = 5
# 再配達の上限回数（マイクノイズ等で barge-in が連発しても無限に再試行しない）。
REDELIVER_MAX_ATTEMPTS = 2

# J-2 P2-1: 機能呼び出しのみで発話ゼロの無言ターン対策（フォールバック相槌）。
# 応答LLM(gpt-5.4-mini 系)は前置き無しで tool だけ呼ぶことがある（実測 8回中6回・
# プロンプトの必須化だけでは治らない＝規律はコードで強制）。tool_schemas() で応答LLMに
# 見えるのは delegate_task/cancel_task のみ（他は agent_tool 専用で tool_sink に出ない）。
_FALLBACK_ACK = {"delegate_task": "やっとくね。", "cancel_task": "確認するね。"}
_DEFAULT_ACK = "うん。"


def _fallback_ack(tool_sink: list) -> str:
    """機能呼び出しのみで発話が空だったターン用の最小限の相槌（J-2 P2-1）。"""
    for tc in tool_sink:
        fn = tc.get("function") if isinstance(tc, dict) else None
        name = (fn or {}).get("name") if fn else None
        if name in _FALLBACK_ACK:
            return _FALLBACK_ACK[name]
    return _DEFAULT_ACK


def _render_recent(turns: Optional[list]) -> str:
    """J-2 P1-1: 配達確認 judge へ渡す直近会話の簡易描画（省略マーカは除く）。"""
    if not turns:
        return ""
    lines = []
    for t in turns:
        if getattr(t, "speaker", None) == OMITTED_SPEAKER:
            continue
        label = "イブ" if getattr(t, "speaker", None) == "eve" else "ユーザ"
        lines.append(f"[{label}] {getattr(t, 'text', '')}")
    return "\n".join(lines)


class ResponseOrchestrator:
    def __init__(
        self,
        audio: AudioPlayQueue,
        stream_fn: StreamFn,
        tts_fn: TtsFn,
        context_assembler: Optional[ContextAssembler] = None,
        tts_concurrency: int = 3,
        conversation_cache: Optional["ConversationCache"] = None,
        rag_store: Optional["RagStore"] = None,
        prediction_state: Optional["PredictionState"] = None,
        on_response_complete: Optional[Callable[[], None]] = None,
        vision_state=None,  # F6 任意: 直近の画面ナレーション源（latest_vision を文脈へ注入）
        dispatcher=None,  # J 任意: FunctionDispatcher（tool_calls を応答完了後に submit）
        tasks_provider: Optional[Callable[[], list[str]]] = None,  # J-1 任意: 予約タスク一覧（整形済み行）
        redeliver_fn: Optional[Callable[[Stimulus], None]] = None,  # 任意: 潰れた報告の再投入（同期・非ブロッキング）
        delivery_checker=None,  # J-2 P1-1 任意: barge-in なしで完了した報告ターンの配達確認サイドカー
        is_suppressed: Optional[Callable[[Optional[str]], bool]] = None,  # D2 任意: 取消済み報告の判定（同期）
    ) -> None:
        self._audio = audio
        self._stream_fn = stream_fn
        self._tts_fn = tts_fn
        self._ctx = context_assembler or ContextAssembler(system_prompt=SPEECH_STYLE)
        self._tts_concurrency = tts_concurrency
        # 任意注入: None なら短期記憶を使わない（F2 系テストは従来どおりの経路）。
        self._cache = conversation_cache
        # 任意注入: 長期記憶(連想RAG)。None なら使わない。
        self._rag = rag_store
        # 任意注入(F4): 直近フィードバックの同期読み元（last_feedback を文脈へ注入）。
        self._state = prediction_state
        # 任意注入(F4): 正常完了時に feedback worker を起こすトリガ（O(1)・非ブロッキング）。
        self._on_complete = on_response_complete
        # 任意注入(F6): 直近の画面ナレーション（latest_vision）を build 時に同期読みして注入。
        self._vision_state = vision_state
        # 任意注入(J): Call-Function の実行サイドカー。tool_calls を応答完了後に submit（非ブロッキング）。
        self._dispatcher = dispatcher
        # 任意注入(J-1/Fix#2): 予約タスクの現在状態（整形済み行）。**全刺激種別**で system に注入する
        # （USER=完了済み変更の再実行防止 / CALLFUNCTION_RESULT=結果と矛盾する約束の防止 /
        # AUTONOMOUS=存在しないタスクの再約束防止。2026-07-13 実機事故の根本原因対応）。
        self._tasks_provider = tasks_provider
        # 任意注入: barge-in で発話前に潰れた CALLFUNCTION_RESULT を再投入するコールバック。
        # 「いつ再投入するか」（ユーザ発話が先に並ぶ保証）は提供側（VoiceLoop）が所有する。
        self._redeliver_fn = redeliver_fn
        # 任意注入(J-2 P1-1): barge-in の有無に関わらず、報告内容が実際に発話されたか確認するサイドカー。
        self._delivery_checker = delivery_checker
        # 任意注入(D2): この dedup_key の報告が取消済みかを同期判定（StimulusQueue.is_suppressed）。
        # 依存の向きは tasks_provider と同型のクロージャ注入＝response 層は task 層を import しない。
        self._is_suppressed = is_suppressed
        self.last_response = ""  # 生成済み全文（自然さの目視・テスト用。記憶には使わない＝C5）

    def _tools_enabled_for(self, stimulus: Stimulus) -> bool:
        """この刺激の応答で tool（Call-Function）を有効にするか。

        USER 発話のみ＝CALLFUNCTION_RESULT/自発発話には渡さない（1ホップ抑制＝二次注入/無限ループ防止）。
        """
        return (
            self._dispatcher is not None
            and Config.CALLFUNCTION_ENABLED
            and stimulus.kind == StimulusKind.USER_UTTERANCE
        )

    def _vision_for(self, stimulus: Stimulus) -> Optional[str]:
        """この刺激の応答文脈に入れる画面情報（層分離の唯一の分岐点）。

        3値: None=VLM 未配線（ブロックを出さない）/ ""=今は画面情報が無い**明示** / 本文。
        「無い」を明示しないと、RAG に入った過去の画面描写を現在として語る（実機事故 2026-07-27）。
        """
        if self._vision_state is None:
            return None
        if stimulus.kind == StimulusKind.USER_UTTERANCE:
            v = self._vision_state.vision_for_user(
                Config.VLM_VISION_TTL_SEC, Config.VLM_STATIC_VISION_MAX_SEC)
        else:
            v = self._vision_state.fresh_vision(Config.VLM_VISION_TTL_SEC)
        return v if v else ""

    def _build_messages(
        self,
        stimulus: Stimulus,
        recent_turns: Optional[list[Turn]] = None,
        rag_chunks: Optional[list[RagChunk]] = None,
    ) -> list[dict]:
        last_feedback = self._state.last_feedback if self._state is not None else None
        # D1: 内省が**いつの会話**を対象にしたか（watermark = そのスパン末尾の iso）。
        # 起動時 catch-up では前セッション末尾を要約するので、これが無いと
        # 「# 直近フィードバック」が5時間前の依頼を現在の依頼として提示する。
        last_feedback_iso = self._state.watermark if self._state is not None else None
        # 画面の注入は**刺激種別で鮮度の意味を変える**（層分離）:
        # - USER 発話: 変化していない間は据え置き可（静止画面でも「今何が見える?」に答えられる。
        #   注入文言に「N秒前・以降変化なし」が付くので捏造にならない）。
        # - 自発発話/機能報告: 鮮度 TTL 内＝「今さっき変化した画面」だけ。静止中に画面が入り続けて
        #   同じ画面要素の話を繰り返す固執を断つ（実機実測で判定136回中135回に画面が入っていた）。
        vision = self._vision_for(stimulus)
        # F5 自発発話: payload は AutonomousSpeech(content, reason)。content は**ユーザ発話でなく
        # イブ自身の下書き**として autonomous_content へ（ユーザ枠に入れると応答LLMが自分の発話に
        # 返事して話者を取り違える＝Fix3）。reason は発話判定理由。USER 等は従来どおり user_text。
        payload = stimulus.payload
        speech_reason = None
        user_text = None
        autonomous_content = None
        callfunction_result = None
        stale_reference = False
        if stimulus.kind == StimulusKind.AUTONOMOUS_SPEECH and isinstance(payload, AutonomousSpeech):
            autonomous_content = payload.content
            speech_reason = payload.reason
            stale_reference = payload.stale_reference
        elif stimulus.kind == StimulusKind.CALLFUNCTION_RESULT and isinstance(payload, CallFunctionResult):
            # 機能実行結果はユーザ発話ではない＝user 枠でなく「# 機能実行結果」ブロックへ（誤話者防止）。
            callfunction_result = payload.content
            if payload.attempts > 0:
                # 再配達: 「遮られた再報告」であることを認識させる（さっき言いかけた、等の自然な言い直し。
                # 既に伝えた前提の話し方や、同じ内容の二重報告を防ぐ）。
                callfunction_result = (
                    "（この報告は直前に読み上げが遮られて、まだユーザに伝わっていない。"
                    "今の会話の流れに合わせて、改めて一度だけ自然に伝える。既に伝えた前提では話さない）\n"
                    + payload.content
                )
        else:
            user_text = str(payload)
        # Fix#2: 予約タスクの現在状態（ターン開始時の同期スナップショット・loop 単一所有なので安全）。
        active_tasks = None
        if self._tasks_provider is not None:
            try:
                active_tasks = self._tasks_provider()
            except Exception:
                logger.exception("予約タスク一覧の取得に失敗（注入なしで継続）")
        # native ロール messages（system + user/assistant ターン列 + 最終 user発話/自発指示）。
        return self._ctx.assemble(
            user_text=user_text,
            autonomous_content=autonomous_content,
            recent_turns=recent_turns,
            rag_chunks=rag_chunks,
            last_feedback=last_feedback,
            last_feedback_iso=last_feedback_iso,
            vision=vision,
            speech_decision_reason=speech_reason,
            stale_reference=stale_reference,
            callfunction_result=callfunction_result,
            tools_active=self._tools_enabled_for(stimulus),
            active_tasks=active_tasks,
        )

    def _notify_complete(self) -> None:
        """正常完了時に feedback worker を起こす（O(1)・例外は握りつぶす）。"""
        if self._on_complete is None:
            return
        try:
            self._on_complete()
        except Exception:
            logger.exception("feedback トリガで例外（無視して継続）")

    def _maybe_redeliver(self, stimulus: Stimulus, gen: int, spoken: list[str], parts: list[str]) -> None:
        """barge-in で発話前に潰れた機能実行報告の再配達判定（2026-07-13 21:20 実機事故対応）。

        事故: 報告ターンが最初の一文を出す前に barge-in で中断され、queue から取り出し済みの
        刺激は戻されず報告が無音消失した（タスクは Done・結果生成済みだった）。
        対象は CALLFUNCTION_RESULT 全般（予約タスク/即時能力/取消報告）。

        「配達済みか」の**最終判定は put 直前に再投入側が行う**（delivered クロージャ）:
        cancel の伝播は即時だが on_played（C5: 再生開始した文は途中停止でも「聞かれた扱い」）は
        数十ms 遅れて発火するため、この時点の spoken は未確定。再投入は最短でも2秒後なので
        その時点では確定している。delivered は False→True にしか変化しない（spoken は増える一方・
        parts は cancel 時点で確定）ため、ここでの早期 return は安全側のみ。
        """
        if self._redeliver_fn is None or stimulus.kind != StimulusKind.CALLFUNCTION_RESULT:
            return
        payload = stimulus.payload
        if not isinstance(payload, CallFunctionResult):
            return
        if self._audio.current_generation() == gen:
            return  # 中断されていない＝通常配達済み
        if payload.attempts >= REDELIVER_MAX_ATTEMPTS:
            logger.warning("⚠ 機能報告の再配達を断念（%d回中断）: %.40s", payload.attempts + 1, payload.content)
            return

        def delivered() -> bool:
            """実質伝わったか（True なら再配達しない＝同じ報告を2回言わない）。

            parts は「stream 完走時のみ全文」が渡される契約（途中 break/cancel では空）。
            よって parts 一致は「≤5文字の短い報告を最後まで再生し切った」場合だけ成立する。
            """
            s = "".join(spoken)
            if len(s) > REDELIVER_MAX_SPOKEN_CHARS:
                return True  # 聞かれた扱いの文がある（C5 準拠＝会話記憶とも整合）
            if parts and s == "".join(parts):
                return True  # 全文（完走保証つき）を再生済み＝完全配達
            return False

        if delivered():
            return
        retry = Stimulus(
            kind=StimulusKind.CALLFUNCTION_RESULT,
            payload=CallFunctionResult(payload.function_name, payload.content, payload.ok, payload.attempts + 1),
            dedup_key=stimulus.dedup_key,  # queue 内 dedup が二重再投入も防ぐ
        )
        logger.info("🔁 機能報告を再配達予約（%d回目・判定時発話%d文字）: %.40s",
                    payload.attempts + 1, len("".join(spoken)), payload.content)
        try:
            self._redeliver_fn(retry, delivered)
        except Exception:
            logger.exception("再配達の予約に失敗（この報告は断念）")

    def _maybe_check_delivery(self, stimulus: Stimulus, gen: int, spoken: list[str], recent) -> None:
        """J-2 P1-1: barge-in **なし**で完了した機能報告ターンが、実際に内容を伝えたか確認する。

        barge-in された場合は `_maybe_redeliver` が既に再配達を判断する（世代変化が条件）ため、
        ここは「正常完了はしたが報告内容に触れず終わった」（例: 直前の別発話への返事を優先した）
        という barge-in では捉えられないケースだけを対象にする。判定はサイドカーへ投げて
        非ブロッキングで返る（LLM 呼び出しで本ターンの完了を遅らせない）。
        """
        if self._delivery_checker is None or stimulus.kind != StimulusKind.CALLFUNCTION_RESULT:
            return
        if not isinstance(stimulus.payload, CallFunctionResult):
            return
        if self._audio.current_generation() != gen:
            return  # barge-in された（_maybe_redeliver 側の担当・二重投入しない）
        try:
            self._delivery_checker.submit(
                stimulus, recent_text=_render_recent(recent), spoken_text="".join(spoken))
        except Exception:
            logger.exception("配達確認の投入に失敗（この報告の確認は断念）")

    async def handle(self, stimulus: Stimulus) -> None:
        gen = self._audio.current_generation()
        # 記憶: 現ターンを記録する**前**に直近会話をスナップショット（現発話が二重表示されない）。
        recent = self._cache.recent_for_injection() if self._cache is not None else None
        # 長期記憶: 今の発話に連想する過去記憶を取得（クエリ埋め込みはここで・≤3s 内）。
        rag_chunks = None
        if self._rag is not None:
            try:
                if stimulus.kind == StimulusKind.AUTONOMOUS_SPEECH:
                    # 自発発話: 今の画面/内容に**関連する記憶** ＋ **新しい切り口** を混ぜる（user search とは別系統）。
                    vis = self._vision_state.fresh_vision(Config.VLM_VISION_TTL_SEC) if self._vision_state is not None else None
                    content = stimulus.payload.content if isinstance(stimulus.payload, AutonomousSpeech) else None
                    # J-2 ②-3: 判定側と同じく、今注入している会話区間から生まれた記憶は除外する
                    # （実測でエコーは判定経路と応答経路の**両方**に出ていた）。
                    since = min(
                        (t.stamp.iso for t in (recent or []) if t.speaker != OMITTED_SPEAKER), default=None)
                    rag_chunks = await self._rag.autonomous_memories(
                        " ".join(filter(None, [vis, content])), Config.RAG_RANDOM_K,
                        context_since_iso=since)
                elif stimulus.kind == StimulusKind.CALLFUNCTION_RESULT and isinstance(stimulus.payload, CallFunctionResult):
                    # 機能実行結果は **content** で関連記憶を引く（payload の repr 文字列で検索しない）。
                    rag_chunks = await self._rag.search(stimulus.payload.content)
                else:
                    rag_chunks = await self._rag.search(str(stimulus.payload))
            except asyncio.CancelledError:
                # 応答開始前（RAG 検索中）の barge-in: まだ一文字も発話していない＝報告は再配達対象。
                # ここを覆わないと「開始直後の相槌」（実機事故 2026-07-13 21:20 の形）で報告が消える。
                self._maybe_redeliver(stimulus, gen, [], [])
                raise
            except Exception:
                logger.exception("RAG 取得に失敗（記憶なしで継続）")
        # D2 二重関門: RAG の await 中に取消が走った場合、この刺激は put の関門より後ろに居るので
        # 墓標を素通りしている。**まだ一文字も喋っていない**この地点で捨てる（実機 D2 では
        # pop→初回発話に 1.3 秒あった）。put 側だけに頼らないのは、resolver の suppress と
        # runner の pop の順序差が call_soon 1スロットしかなく、設計上の保証にならないため。
        if self._is_suppressed is not None and stimulus.kind == StimulusKind.CALLFUNCTION_RESULT:
            try:
                if self._is_suppressed(stimulus.dedup_key):
                    logger.info("⚰ 抑止済みの報告ターンを中止: dedup=%s", stimulus.dedup_key)
                    return  # 再配達もしない（_maybe_redeliver を通さずに抜ける）
            except Exception:
                logger.exception("抑止判定で例外（無視して通常配達）")
        messages = self._build_messages(stimulus, recent, rag_chunks)
        # J Call-Function: USER ターンのみ tools を渡す。tool_calls はここに溜め、応答完了後に submit。
        tools = self._dispatcher.tool_schemas() if self._tools_enabled_for(stimulus) else None
        tool_sink: list = []
        # ユーザ発話なら user ターンを記録（自律/vision/callfunction には user ターンは無い）。
        if self._cache is not None and stimulus.kind == StimulusKind.USER_UTTERANCE:
            self._cache.add_turn("user", str(stimulus.payload))
        splitter = JapaneseSentenceSplitter()
        sem = asyncio.Semaphore(self._tts_concurrency)
        tasks: list[asyncio.Task] = []
        parts: list[str] = []
        spoken: list[str] = []  # C5: 実際に再生し終えた文だけが入る（生成≠発話）
        stream_complete = False  # stream が中断されず最後まで生成し切ったか（再配達の完全配達判定に使う）
        eve_recorded = False

        def _record_eve() -> None:
            nonlocal eve_recorded
            if eve_recorded or self._cache is None:
                return
            eve_recorded = True
            self._cache.add_turn("eve", "".join(spoken))  # 空なら add_turn 側で無視

        async def _tts_and_enqueue(seq: int, sentence: str) -> None:
            # A1: 同一世代である限り、TTS が None/例外でも **必ず seq を埋める**。
            # 埋めないと AudioPlayQueue._drain_buffer が連続 seq で永久停止
            # （= head-of-line デッドロック / 1文目失敗でターン丸ごと無音）。
            wav = None
            try:
                async with sem:
                    if self._audio.current_generation() != gen:
                        return  # barge-in 済み: 起こさない（古い世代は破棄される）
                    wav = await self._tts_fn(sentence)
            except Exception:
                logger.exception("TTS 失敗（文をスキップして継続）: %.20s", sentence)
                wav = None
            if wav is None:
                logger.warning("TTS が音声を返さず（文をスキップ）: %.20s", sentence)
            if self._audio.current_generation() == gen:
                # on_played は「実際に再生し終えた時」だけ呼ばれる → spoken に積む（C5）。
                self._audio.enqueue(gen, seq, wav, text=sentence, on_played=spoken.append)

        def _emit(sentence: str, *, auto: bool = False) -> None:
            clean = sanitize_for_speech(sentence)  # 残留マークダウン除去（コードゲート）
            if not clean:
                return  # 記号だけの行は読み上げない
            # J-2 P2-1: auto=True(コード生成のフォールバック相槌)はログタグで区別する
            # （発話内容・記憶には印を付けない＝会話文脈を汚さない。デバッグ識別専用）。
            logger.info("🤖(auto-ack) %s" if auto else "🤖 %s", clean)
            parts.append(clean)
            seq = self._audio.reserve_seq(gen)  # stream 順に予約（再生は seq 昇順）
            tasks.append(asyncio.create_task(_tts_and_enqueue(seq, clean)))

        try:
            try:
                # tools 有効時のみ tool 対応 stream（content は従来どおり TTS・tool_calls は sink へ）。
                stream = (
                    self._stream_fn(messages, tools=tools, tool_sink=tool_sink)
                    if tools is not None else self._stream_fn(messages)
                )
                async for delta in stream:
                    if self._audio.current_generation() != gen:
                        break  # barge-in(世代変化): 生成停止（残りの文は出さない）
                    for sentence in splitter.feed(delta):
                        _emit(sentence)
                else:
                    # 正常終了時のみ末尾を flush（barge-in/エラー中断時は出さない）
                    for sentence in splitter.flush():
                        _emit(sentence)
                    stream_complete = True  # parts が「報告の全文」であることの保証（途中 break では偽）
            except Exception:
                # A3: LLM/stream の一時エラーは起こりうる → ログして途中までで打ち切り、継続。
                logger.exception("応答生成中にエラー（途中までで打ち切り）")
            if stream_complete and tool_sink and not parts:
                # J-2 P2-1: 機能を呼びつつ発話ゼロで終わったターンへの最小限のフォールバック
                # （barge-in等で stream_complete=False の時は追加しない＝古い世代を起こさない）。
                _emit(_fallback_ack(tool_sink), auto=True)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.last_response = "".join(parts)
            if self._cache is not None:
                # 現ターンの音声が再生し切るまで待ってから「実発話」を記録（C5）。
                await self._audio.join()
                _record_eve()
            # F4: 正常完了時のみ feedback をトリガ（barge-in=CancelledError 経路では呼ばない）。
            # 直近の eve ターンが記録された後なので、worker のスパンに今回応答が含まれる。
            self._notify_complete()
            # J Call-Function: 応答完了後に tool_calls を実行へ submit（O(1)・非ブロッキング＝Eve は即解放）。
            # 実行は背景サイドカーで進む。barge-in(CancelledError)経路はここに到達しない＝submit しない。
            if tool_sink and self._dispatcher is not None:
                self._dispatcher.submit(tool_sink)
            # 正常経路でも世代変化（audio.interrupt のみ・stream break）で無発話終了があり得る
            # → 潰れたタスク報告は再配達（CancelledError 経路と同じ判定）。
            # parts は stream 完走時のみ「全文」として渡す（途中 break の parts と spoken が偶然
            # 一致しても「短い前置きだけ再生済み＝配達済み」と誤判定しないため）。
            self._maybe_redeliver(stimulus, gen, spoken, parts if stream_complete else [])
            # J-2 P1-1: barge-in が無かった（gen 不変）場合のみ、内容を実際に伝えたか確認する
            # （barge-in された分は上の _maybe_redeliver が担当・二重チェック/二重配達を避ける）。
            self._maybe_check_delivery(stimulus, gen, spoken, recent)
        except asyncio.CancelledError:
            # barge-in: ここまでに**実際に喋った分だけ**を記憶に記録（生成途中の文は残さない）。
            _record_eve()
            for t in tasks:  # 進行中の TTS タスクも片付ける（孤児化防止）
                if not t.done():
                    t.cancel()
            # barge-in で発話前に潰れたタスク報告は再配達（同期呼び出しのみ・await しない）。
            self._maybe_redeliver(stimulus, gen, spoken, parts if stream_complete else [])
            raise
