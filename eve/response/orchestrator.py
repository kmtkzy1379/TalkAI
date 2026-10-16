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
import json
import logging
import re
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Optional

from ..config import Config
from ..clock import elapsed_wall
from ..context_assembler import (
    OMITTED_SPEAKER,
    SESSION_GAP_SEC,
    ContextAssembler,
    RagChunk,
    Turn,
)
from ..pipeline.audio_play_queue import AudioPlayQueue
from ..pipeline.stimulus import (
    REDELIVER_BARGE_IN,
    REDELIVER_NOT_CONVEYED,
    CallFunctionResult,
    Stimulus,
    StimulusKind,
)
from ..speech.decider import AutonomousSpeech
from ..stt.askback import UNINTELLIGIBLE_PLACEHOLDER
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

def _reply_from_tools(tool_sink: list) -> Optional[str]:
    """機能呼び出しのみで本文が空だったターンに言う一言（`delegate_task` / `cancel_task` の `reply`＝応答LLM が書いたもの）。
    無ければ None。

    ユーザ 2026-09-28「タスクやる前に｢やっとくね｣が定型文化してて機械的。出来ればその文脈にあった言い回しとかして
    欲しい」→ 2026-09-29 採用。応答LLM（gpt-6-luna）は本文を書かずにツールだけ返すことが多く（実測 6件中5件）、
    その度に定型の「やっとくね。」になっていた。ツールの引数なら書き落とさない（必須の引数・`task/capabilities.py`）。

    **定型の相槌（旧「やっとくね。」「確認するね。」「うん。」）は 2026-10-09 に撤去**（ユーザ「例文などで決まり文句や
    定型文は使わせたくない。なるべくレパートリー多く自由に」・性格は丁寧語なのに定型だけタメ口で口調が崩れていた）。
    本文も reply も無いターンは何も言わない（`handle` が警告を残す＝実機でどれくらい起きるか追える）。
    長さでは切らない（応答LLM が書いた一言なので、無言にするより言う方がよい）。
    """
    for tc in tool_sink:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not fn:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            continue
        reply = args.get("reply") if isinstance(args, dict) else None
        if isinstance(reply, str) and reply.strip():
            return reply.strip()
    return None


# 応答の二重化の関門（`handle._emit`）: この文字数以上の節が同じ応答の中でもう一度出たら「2回目に入った」とみなす。
# 短い節（「うん」「そっか」等）は普通に繰り返すので対象にしない。
REPEAT_MIN_CHARS = 8
_REPEAT_IGNORE = re.compile(r"[\s、。，．,.！？!?…・ー〜~「」『』（）()]")
# 文末の軽い終助詞は落として比べる（「…設定しておいて。」と「…設定しておいてね。」を同じ節とみなす・実測）
_REPEAT_TAIL = re.compile(r"[ねよなわ]+$")


def _repeat_key(clause: str) -> str:
    """節の比較用の形（空白・句読点・記号と、文末の終助詞を落とす）。"""
    return _REPEAT_TAIL.sub("", _REPEAT_IGNORE.sub("", clause or ""))


# 「できない」と言ったのに同じ返事で delegate_task も出す矛盾は、**指示で止める**（`context_assembler` の
# 「手段の無いことは delegate_task にも回さない…できないと伝えた依頼で delegate_task を呼ばない」）。以前は返事の語
# （〜できない・〜てみるね 等）で委譲を捨てる関門（`_says_cannot`）があったが、2026-10-09 に撤去した（ユーザ「そもそも
# タスクLLMに依頼出してるのは応答AIだから。指示で制御してみる。無理そうなら Jev で判定させる」）。語の一覧は性格の口調
# （10/05 から丁寧語）が変わると当たらなくなり、「ドラマは見られないんですけど、調べてみますね」の委譲まで捨てて、
# 言ったのに調べない状態を作っていた。撤去後の実測（Gemini）で食い違いは 0（手段の無い依頼 26/26 で委譲しない）。


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


def _recall_query(stimulus: Stimulus) -> str:
    """自分のことの控えの近さを測る文: ユーザ発話はその文・自発発話は判定のメモ・機能の報告は結果。"""
    payload = stimulus.payload
    if isinstance(payload, (AutonomousSpeech, CallFunctionResult)):
        return payload.content or ""
    return str(payload)


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
        capabilities_provider: Optional[Callable[[], list[str]]] = None,  # D3 任意: 外の世界への手段
        redeliver_fn: Optional[Callable[[Stimulus], None]] = None,  # 任意: 潰れた報告の再投入（同期・非ブロッキング）
        delivery_checker=None,  # J-2 P1-1 任意: barge-in なしで完了した報告ターンの配達確認サイドカー
        is_suppressed: Optional[Callable[[Optional[str]], bool]] = None,  # D2 任意: 取消済み報告の判定（同期）
        features_off_provider: Optional[Callable[[], list[str]]] = None,  # UI 任意: 今オフの機能
        self_model_fn: Optional[Callable[[], str]] = None,  # 任意: イブ自身の好み（`# イブ自身のこと`）
        # 任意: 今の話しかけ（自発発話はそのメモ・機能の報告は結果）に近い好みだけの `# イブ自身のこと`（0件なら ""）。
        # あれば self_model_fn より優先する（`memory/self_model.SelfRecall`・ユーザ 2026-10-17「好みの種類が近いときに読み込ませる」）
        self_model_pick: Optional[Callable[[str], Awaitable[str]]] = None,
        # S3 任意: 「応答が近い」合図（そのターンで最初の文の合成を始めた瞬間＝音声まで約0.2〜0.35秒）。
        # フィラーはこれで新しい語を始めない（同期・非ブロッキング・例外は握る）。
        on_first_sentence: Optional[Callable[[], None]] = None,
        # S3 任意: 直前にイブが口にしたつなぎ言葉（「あー」「あ、えー」等・無ければ None）。
        # USER 発話のときだけ読み、応答LLM への user メッセージの前置きにする（**payload は変えない**
        # ＝会話履歴・RAG・フィードバックには入らない）。
        filler_provider: Optional[Callable[[], Optional[str]]] = None,
        # S5 任意: イブの今の気持ち（`EmotionState.describe()`・未判定なら None）。全刺激種別で system に入れる
        emotion_provider: Optional[Callable[[], Optional[str]]] = None,
        # S5 任意: 声のトーン＝今の感情の絵文字（Irodori の注釈・未決の感情は None）。応答を組み立てた時点で
        # 1回だけ読み、そのターンの全部の節の**TTS に渡す文字にだけ**付ける（記憶・画面の文字には付けない）
        tone_provider: Optional[Callable[[], Optional[str]]] = None,
        # 任意: 話し方の実例（`eve/persona_examples.StyleExamples`）。ユーザ発話への応答でだけ、毎ターン選び直して
        # 性格の直後に入れる。今の話しかけと意味が近い実例は見せない（丸写しの元・2026-10-01 実測）
        style_examples=None,
        # 任意: Alchemy（ユーザ作のゲーム）の準備・対戦中の事実（`AlchemyManager.voice_block`・それ以外は None）。全刺激種別で入れる
        game_provider: Optional[Callable[[], Optional[str]]] = None,
        # 任意: 委譲（delegate_task）が今出ているか。タスクを UI で切っても Alchemy の機能は残るので「tool がある」だけでは
        # 委譲の使い方を出す理由にならない（出すと無い delegate_task を使うつもりで喋る）。None なら従来どおり tool の有無で決める
        delegation_offered: Optional[Callable[[], bool]] = None,
        # 任意: YouTube 配信中の事実（`StreamManager.voice_block`・配信していない時は None）。全刺激種別で入れる
        stream_provider: Optional[Callable[[], Optional[str]]] = None,
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
        # 任意注入(D3): 「外の世界に手を出せる手段」を registry から取る（散文手書きは腐る）。
        # tasks_provider と同型のクロージャ注入＝response 層は capability 層を import しない。
        self._capabilities_provider = capabilities_provider
        # 任意注入: barge-in で発話前に潰れた CALLFUNCTION_RESULT を再投入するコールバック。
        # 「いつ再投入するか」（ユーザ発話が先に並ぶ保証）は提供側（VoiceLoop）が所有する。
        self._redeliver_fn = redeliver_fn
        # 任意注入(J-2 P1-1): barge-in の有無に関わらず、報告内容が実際に発話されたか確認するサイドカー。
        self._delivery_checker = delivery_checker
        # 任意注入(D2): この dedup_key の報告が取消済みかを同期判定（StimulusQueue.is_suppressed）。
        # 依存の向きは tasks_provider と同型のクロージャ注入＝response 層は task 層を import しない。
        self._is_suppressed = is_suppressed
        # 任意注入(UI): 今オフにしている機能の名前。ヘッドレスでは None＝ブロック自体が出ない。
        self._features_off_provider = features_off_provider
        self._self_model_fn = self_model_fn
        self._self_model_pick = self_model_pick
        self._on_first_sentence = on_first_sentence
        self._filler_provider = filler_provider
        self._emotion_provider = emotion_provider
        self._tone_provider = tone_provider
        self._style_examples = style_examples
        self._game_provider = game_provider
        self._delegation_offered = delegation_offered
        self._stream_provider = stream_provider
        # 配信の始め・終わりの区切りの手前で話していたことの要約（`eve/stream/` が作る・無ければ None）
        self.carry_provider: Optional[Callable[[], Optional[str]]] = None
        self.last_response = ""  # 生成済み全文（自然さの目視・テスト用。記憶には使わない＝C5）

    def _tools_enabled_for(self, stimulus: Stimulus) -> bool:
        """この刺激の応答で tool（Call-Function）を有効にするか。

        USER 発話のみ＝CALLFUNCTION_RESULT/自発発話には渡さない（1ホップ抑制＝二次注入/無限ループ防止）。

        **提示できる tool が1つも無い時は False**（UI でタスクを OFF にすると
        `delegate_task`/`cancel_task` が unregister され `tool_schemas()` が空になる）。
        ここを見ないと system プロンプトの `tools_active` だけが True のまま残り、
        「使える tool があるつもりで喋るのに実際には渡っていない」という嘘の状態になる。
        """
        if self._dispatcher is None or not Config.CALLFUNCTION_ENABLED:
            return False
        if stimulus.kind != StimulusKind.USER_UTTERANCE:
            return False
        try:
            return bool(self._dispatcher.tool_schemas())
        except Exception:
            logger.exception("tool_schemas の取得に失敗（tool なしで応答）")
            return False

    def _delegation_for(self, stimulus: Stimulus) -> bool:
        """委譲（タスク）の使い方・実行できる手段のブロックを出すか（tool が渡り、かつ delegate_task が出ている）。"""
        if not self._tools_enabled_for(stimulus):
            return False
        if self._delegation_offered is None:
            return True
        try:
            return bool(self._delegation_offered())
        except Exception:
            logger.exception("委譲の有無の取得に失敗（出すとみなす）")
            return True

    def _vision_for(self, stimulus: Stimulus) -> tuple[Optional[str], bool]:
        """この刺激の応答文脈に入れる画面情報（層分離の唯一の分岐点）。

        `(本文, 暫定か)` を返す。本文は3値: None=VLM 未配線（ブロックを出さない）/
        ""=今は画面情報が無い**明示** / 本文。
        「無い」を明示しないと、RAG に入った過去の画面描写を現在として語る（実機事故 2026-07-27）。

        暫定=True は「画面が変わったが実況がまだ追いついていない」区間（`vision_after_change`）。
        ここを空白にすると応答LLM が能力ごと否認するため、断り付きで渡す（実測 2026-09-21）。

        **自発発話・機能報告も同じ規則にした（ユーザ提案 2026-09-30）**: 以前は `fresh_vision`（7秒）だけで、
        画面が静止しているだけで「画面情報なし・今この瞬間の画面は届いていない」になっていた（誤解を生む文言）。
        静止中は「N秒前の画面。以降ずっと変化なし」を付けて渡す。同話題固執の層分離は**発話判定の入力と
        記憶の検索クエリ**（どちらも `fresh_vision` のまま）で守られている。自発発話の応答は判定が決めた下書きを
        言い直すだけで、実測（下書きが画面と無関係な場面×6）では静止画面を足しても画面の話を足さなかった。
        """
        if self._vision_state is None:
            return None, False
        v = self._vision_state.vision_for_user(
            Config.VLM_VISION_TTL_SEC, Config.VLM_STATIC_VISION_MAX_SEC)
        if v:
            return v, False
        v = self._vision_state.vision_after_change(
            Config.VLM_VISION_TTL_SEC, Config.VLM_STALE_VISION_MAX_SEC)
        if v:
            return v, True
        return "", False

    def _build_messages(
        self,
        stimulus: Stimulus,
        recent_turns: Optional[list[Turn]] = None,
        rag_chunks: Optional[list[RagChunk]] = None,
        style_examples: Optional[str] = None,
        self_model: Optional[str] = None,
    ) -> list[dict]:
        last_feedback = self._state.last_feedback if self._state is not None else None
        # D1: 内省が**いつの会話**を対象にしたか（watermark = そのスパン末尾の iso）。
        # 起動時 catch-up では前セッション末尾を要約するので、これが無いと
        # 「# 直近フィードバック」が5時間前の依頼を現在の依頼として提示する。
        last_feedback_iso = self._state.watermark if self._state is not None else None
        if last_feedback and self._cache is not None and getattr(self._cache, "before_boundary", None) is not None:
            # 起動中の区切り（YouTube 配信の始まり・終わり）より前の会話の内省は入れない
            # （配信前の私的な会話の要約を配信の文脈に持ち込まない）
            if self._cache.before_boundary(last_feedback_iso):
                last_feedback = None
        # D1（案2）: 前セッション最終発話からの経過。会話は注入しないので、境界の事実だけ渡す。
        prev_gap = None
        if self._cache is not None:
            prev_iso = getattr(self._cache, "restored_last_iso", lambda: None)()
            if prev_iso:
                age = elapsed_wall(prev_iso)
                prev_gap = age if age > SESSION_GAP_SEC else None
        # 画面の注入: 変化していない間は据え置き（「N秒前の画面・以降変化なし」が付くので捏造にならない）。
        # 2026-09-30 から自発発話・機能報告も同じ（以前は静止中に「画面情報なし」になっていた）。
        # 静止中の同話題固執の層分離は、発話判定の入力と記憶の検索クエリ（`fresh_vision`）が担う。
        vision, vision_stale = self._vision_for(stimulus)
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
                # 再配達の前置き。**理由で文言を分ける**（実機事故 2026-08-15 02:44）:
                # 旧実装は `attempts > 0` の単一条件で barge-in 用の文言を付けており、
                # 「実際には喋り終えたが要点が伝わらなかった」ケースにも
                # 「まだユーザに伝わっていない」という**事実と食い違う前提**を渡していた。
                # 結果、応答LLM は直前のユーザ発話への謝罪ごと作り直し、二重回答になった。
                if getattr(payload, "reason", REDELIVER_BARGE_IN) == REDELIVER_NOT_CONVEYED:
                    callfunction_result = (
                        "（この報告は一度発話したが、要点がユーザに伝わっていない可能性がある。"
                        "**同じ話を繰り返さず**、まだ伝えていない要点だけを一言で伝える。"
                        "謝り直したり、直前のやりとりに改めて返事をしたりしない）\n"
                        + payload.content
                    )
                else:
                    # barge-in: 「遮られた再報告」であることを認識させる（さっき言いかけた、等の
                    # 自然な言い直し。既に伝えた前提の話し方や、同じ内容の二重報告を防ぐ）。
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
        # D3: 外の世界への手段。**tools を渡すターン（USER 発話）でだけ**注入する。
        # 自発発話/機能報告ターン（応答32件中14件＝44%）には出さない: 承諾すべき依頼が無いのに
        # 能力の枠だけ渡すと、自発発話が「私にできるのは〜だけ」の自己言及に寄る（②-6 の
        # 訂正指示を自発経路だけに限定したのと同じ判断）。
        # また tools 無効時（CALLFUNCTION_ENABLED=0 等）にも出さない: 手段ゼロの構成では
        # 2026-07-28 12:24-12:26 の実機ログのとおり**既に過剰拒否側へ倒れており**
        # （5連続で誤拒否・直前に内部知識で答えた話題まで拒否）、そこへ「手段が無い」を
        # 足すのは逆効果。
        capabilities = None
        if self._capabilities_provider is not None and self._delegation_for(stimulus):
            try:
                capabilities = self._capabilities_provider()
            except Exception:
                logger.exception("能力一覧の取得に失敗（注入なしで継続）")
        # UI で今オフにしている機能。**全刺激種別で出す**（自発発話でも「見えないから」と
        # 言い出す場面があり、そこで「機能が無い」と言われるのが最も紛らわしいため）。
        # イブ自身の好み。今の話しかけに近い項目だけ（`handle` が `self_model_pick` で選んで渡す・0件は ""）。
        # 渡されなかった時（選ぶ口が無い構成・テスト・評価の道具）は従来どおり全部入れる。長さはストア側の上限で有界。
        if self_model is None and self._self_model_fn is not None:
            try:
                self_model = self._self_model_fn() or None
            except Exception:
                logger.exception("自己モデルの取得に失敗（無しで継続）")
        self_model = self_model or None
        features_off = None
        if self._features_off_provider is not None:
            try:
                features_off = self._features_off_provider() or None
            except Exception:
                logger.exception("オフ機能一覧の取得に失敗（注入なしで継続）")
        # S3: この発話の後にイブが既に口にしたつなぎ言葉（USER 発話のときだけ）。
        spoken_filler = None
        if self._filler_provider is not None and stimulus.kind == StimulusKind.USER_UTTERANCE:
            try:
                spoken_filler = self._filler_provider() or None
            except Exception:
                logger.exception("つなぎ言葉の取得に失敗（無しで継続）")
        emotion = None
        if self._emotion_provider is not None:
            try:
                emotion = self._emotion_provider() or None
            except Exception:
                logger.exception("今の感情の取得に失敗（無しで継続）")
        game = None
        if self._game_provider is not None:
            try:
                game = self._game_provider() or None
            except Exception:
                logger.exception("Alchemy の事実の取得に失敗（無しで継続）")
        stream = None
        if self._stream_provider is not None:
            try:
                stream = self._stream_provider() or None
            except Exception:
                logger.exception("配信の事実の取得に失敗（無しで継続）")
        carry = None
        if self.carry_provider is not None:
            try:
                carry = self.carry_provider() or None
            except Exception:
                logger.exception("区切りの手前の会話の要約の取得に失敗（無しで継続）")
        # native ロール messages（system + user/assistant ターン列 + 最終 user発話/自発指示）。
        return self._ctx.assemble(
            user_text=user_text,
            autonomous_content=autonomous_content,
            recent_turns=recent_turns,
            rag_chunks=rag_chunks,
            last_feedback=last_feedback,
            last_feedback_iso=last_feedback_iso,
            vision=vision,
            vision_stale=vision_stale,
            speech_decision_reason=speech_reason,
            stale_reference=stale_reference,
            callfunction_result=callfunction_result,
            tools_active=self._delegation_for(stimulus),
            active_tasks=active_tasks,
            capabilities=capabilities,
            prev_session_gap_sec=prev_gap,
            features_off=features_off,
            self_model=self_model,
            spoken_filler=spoken_filler,
            emotion=emotion,
            style_examples=style_examples if stimulus.kind == StimulusKind.USER_UTTERANCE else None,
            game=game,
            stream=stream,
            carry_over=carry,
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
                elif str(stimulus.payload) == UNINTELLIGIBLE_PLACEHOLDER:
                    pass  # 聞き取れなかった発話＝中身が無いので連想する記憶も無い（代替の文字で検索しない）
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
        # 話し方の実例（ユーザ発話への応答だけ）。記憶の検索の後に選ぶ＝同じ発話の埋め込みを使い回せる
        examples, shown = None, []
        if self._style_examples is not None and stimulus.kind == StimulusKind.USER_UTTERANCE:
            try:
                examples, shown = await self._style_examples.pick(str(stimulus.payload))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("話し方の実例の選択に失敗（実例なしで継続）")
                examples, shown = None, []
        # 自分のことの控え: 今の話しかけに近い項目だけ（実例の後＝話しかけの埋め込みは記憶の検索の物を使い回す）
        self_model = None
        if self._self_model_pick is not None:
            try:
                self_model = await self._self_model_pick(_recall_query(stimulus))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("自分のことの控えの選択に失敗（全部入れる）")
                self_model = None
        messages = self._build_messages(stimulus, recent, rag_chunks, style_examples=examples or None,
                                        self_model=self_model)
        # S5 声のトーン: 応答を組み立てた時点の今の感情で**このターンは1つに決める**（ユーザ 2026-09-26
        # 「ユーザー話し終わる前までの現感情→応答生成→発話」）。応答の途中で感情の判定が届いても声だけ
        # 変わらない＝応答の文字（上の「# イブの今の気持ち」）と声の感情がそろう。
        tone = None
        if self._tone_provider is not None:
            try:
                tone = self._tone_provider() or None
            except Exception:
                logger.exception("声のトーンの取得に失敗（付けずに合成）")
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
        # 同じ応答の中で答えを2回言い始めたら、そこから先は読み上げない（`_emit` の関門）。
        said_keys: set[str] = set()
        repeating = False

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
            meta = None  # UI のリップシンク計画（TTS が返した時だけ非 None）
            try:
                async with sem:
                    if self._audio.current_generation() != gen:
                        return  # barge-in 済み: 起こさない（古い世代は破棄される）
                    wav = await self._tts_fn(sentence + tone if tone else sentence)
                # tts_fn が `(wav, meta)` を返す実装（`VoicevoxTTS.generate_with_meta`）にも対応。
                # 単一値を返す従来の tts_fn（テスト/tools の偽物・`generate`）はここを素通りする
                # ＝ meta は None のまま＝以降の UI 経路は完全に休眠する。
                if isinstance(wav, tuple) and len(wav) == 2:
                    wav, meta = wav
            except Exception:
                logger.exception("TTS 失敗（文をスキップして継続）: %.20s", sentence)
                wav = None
                meta = None
            if wav is None:
                logger.warning("TTS が音声を返さず（文をスキップ）: %.20s", sentence)
            if self._audio.current_generation() == gen:
                # UI 用メタは enqueue の**前**に預ける（再生開始時に pop される）。
                # `enqueue` のシグネチャは変えない（E2E ハーネスが固定引数で wrap している）。
                # **必ず握り潰す**: ここで例外が出ると下の enqueue に到達せず seq に穴が空き、
                # `_drain_buffer` が contiguous で止まってターン丸ごと無音になる（A1 の規律）。
                # しかもこの関数の例外は `gather(return_exceptions=True)` に吸われlog も出ない。
                if meta is not None:
                    try:
                        self._audio.note_meta(gen, seq, meta)
                    except Exception:
                        logger.exception("note_meta 失敗（口パクを諦めて音声は続行）")
                # on_played は「実際に再生し終えた時」だけ呼ばれる → spoken に積む（C5）。
                self._audio.enqueue(gen, seq, wav, text=sentence, on_played=spoken.append)

        def _emit(sentence: str) -> None:
            nonlocal repeating
            clean = sanitize_for_speech(sentence)  # 残留マークダウン除去（コードゲート）
            if not clean or repeating:
                return  # 記号だけの行は読み上げない／繰り返しに入った後は読み上げない
            # 応答の二重化の関門（実測 2026-09-30・gpt-6-luna reasoning_effort=none の 1308応答中 18件＝1.4%）:
            # 答えを言い切った後に、言い回しを少し変えた**同じ答えをもう一度**出すことがある（間に
            # 「vel」「/final_element」「<|END|>」が挟まることも）。そのまま読み上げると同じ説明を2回喋る。
            # 前に言った節（8文字以上）とまったく同じ節が出たら「2回目に入った」とみなし、以降は読み上げない。
            # stream は止めない（最後に来る tool_calls を取りこぼさないため・読み飛ばすだけ）。
            key = _repeat_key(clean)
            if len(key) >= REPEAT_MIN_CHARS and key in said_keys:
                repeating = True
                logger.warning("🔁 応答が同じ内容を繰り返し始めた→以降は読み上げない: %.30s", clean)
                return
            said_keys.add(key)
            logger.info("🤖 %s", clean)
            parts.append(clean)
            seq = self._audio.reserve_seq(gen)  # stream 順に予約（再生は seq 昇順）
            tasks.append(asyncio.create_task(_tts_and_enqueue(seq, clean)))
            if len(parts) == 1 and self._on_first_sentence is not None:
                # S3「応答が近い」: 1文目の合成を始めた＝音声まで約0.2〜0.35秒（フィラーは次の語を始めない）
                try:
                    self._on_first_sentence()
                except Exception:
                    logger.exception("on_first_sentence フックで例外（無視して継続）")

        async def _run_stream(msgs, tls, sink) -> bool:
            """1回分の生成を読み上げへ流す。最後まで生成し切ったら True（parts が全文である保証）。"""
            try:
                # tools 有効時のみ tool 対応 stream（content は従来どおり TTS・tool_calls は sink へ）。
                stream = self._stream_fn(msgs, tools=tls, tool_sink=sink) if tls is not None else self._stream_fn(msgs)
                async for delta in stream:
                    if self._audio.current_generation() != gen:
                        return False  # barge-in(世代変化): 生成停止（残りの文は出さない）
                    for sentence in splitter.feed(delta):
                        _emit(sentence)
                # 正常終了時のみ末尾を flush（barge-in/エラー中断時は出さない）
                for sentence in splitter.flush():
                    _emit(sentence)
                return True
            except Exception:
                # A3: LLM/stream の一時エラーは起こりうる → ログして途中までで打ち切り、継続。
                logger.exception("応答生成中にエラー（途中までで打ち切り）")
                return False

        try:
            stream_complete = await _run_stream(messages, tools, tool_sink)
            redo = (self._dispatcher.reanswer_only(tool_sink)
                    if stream_complete and tool_sink and not parts and hasattr(self._dispatcher, "reanswer_only")
                    else [])
            if redo:
                # 会話に事実を足すだけの機能（Alchemy のルール準備）だけを呼んで本文ゼロ: 一言（reply）は言わずに
                # その場で実行し、足された事実（ルール）を見て同じ発話への返事を作り直す（1回だけ）。
                # 前置き「ちょっと待ってね」で止まらず、そのまま答える（ユーザ 2026-10-11）
                first_reply = _reply_from_tools(tool_sink)
                for name, args in redo:
                    logger.info("⚙ 返事の前に実行: %s → %s", name, self._dispatcher.execute_now(name, args))
                tool_sink = []
                messages = self._build_messages(stimulus, recent, rag_chunks, style_examples=examples or None,
                                                self_model=self_model)
                tools = (self._dispatcher.tool_schemas(exclude={n for n, _a in redo})
                         if tools is not None else None) or None
                stream_complete = await _run_stream(messages, tools, tool_sink)
                if stream_complete and not parts and not tool_sink and first_reply:
                    # 作り直しが何も言わなかった時だけ、最初の一言を言う（無言にしない）
                    for sentence in splitter.feed(first_reply) + splitter.flush():
                        _emit(sentence)
            if stream_complete and tool_sink and not parts:
                # 機能を呼びつつ本文ゼロで終わったターン: 応答LLM が引数に書いた一言（reply）を言う
                # （「、」「。」で区切るのは本文と同じ。barge-in等で stream_complete=False の時は言わない＝
                # 古い世代を起こさない）。定型の相槌は使わない（`_reply_from_tools` の注記）。
                reply = _reply_from_tools(tool_sink)
                if reply:
                    reply_splitter = JapaneseSentenceSplitter()
                    for sentence in reply_splitter.feed(reply) + reply_splitter.flush():
                        _emit(sentence)
                else:
                    names = [((tc.get("function") or {}) if isinstance(tc, dict) else {}).get("name") for tc in tool_sink]
                    logger.warning("🤐 機能を呼んだが何も言わなかった（本文も reply も無い）: %s",
                                   ",".join(n or "?" for n in names))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.last_response = "".join(parts)
            if shown:
                # 見せた実例をほぼそのまま言っていないか（止めはしない＝もう声に出ている。実例の選び直しの材料にログへ）
                hit = self._style_examples.copy_suspect(self.last_response, shown)
                if hit:
                    logger.warning("📋 話し方の実例をほぼそのまま言った疑い: 「%.40s」≒ 実例「%.40s」",
                                   self.last_response, hit)
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
