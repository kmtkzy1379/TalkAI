"""F5 沈黙監視 + 発話判定サイドカー（loop タスク2本・OS スレッド0・ロック0＝§9.4）。

- SpeechState: loop 所有 ephemeral（単一書込）。沈黙計測・再評価カデンス・発話判定ログ。
- SilenceMonitor: 周期 tick で「5秒沈黙 + 応答中でない + ユーザ発話中でない + decider idle」を
  満たす時だけ SpeechDecider をトリガ（O(1)）。バックオフ/カテゴリ/再挨拶抑制は持たない（Q3）。
- SpeechDecider: single-flight（FeedbackWorker を踏襲）。値コピー snapshot → should_speak →
  発話判定ログに True/False とも記録 → speak なら AUTONOMOUS_SPEECH 刺激を投入。

モノローグや再挨拶が出たら抑制で隠さず should_speak/文脈/feedback の不具合として直す方針。
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from typing import Callable, Optional

from ..clock import humanize, now_iso, now_mono
from ..config import Config
from ..context_assembler import OMITTED_SPEAKER, Turn
from ..pipeline.stimulus import Stimulus, StimulusKind
from .decider import (
    AutonomousSpeech,
    DecideFn,
    is_topic_outsourcing,
    should_speak,
    stale_recency_deixis,
)

logger = logging.getLogger(__name__)

# J-2 ②-1: 同内容の自発発話の抑制閾値（一段目＝文字bigram Jaccard）。
# 実測校正(2026-07-23・E2E実データ): 抑制すべき重複ペア=0.286/0.453、
# 通すべき別話題ペア=最大0.074 → 0.25 で両側に余裕をもって分離できる。
# ②-2(2026-07-26): 実発話4文の実測で、語彙を変えた同内容の再提案が bigram 最大0.23＝
# 閾値をすり抜けた。文字の一致だけでは足りないので二段目に埋め込み cos を足した
# （閾値は Config.SPEECH_DUP_EMB_THRESHOLD。同話題0.909-0.932 vs 別話題0.780-0.834 で分離）。
DUP_SIM_THRESHOLD = 0.25


def _char_bigrams(s: str) -> set:
    s = "".join((s or "").split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


def cosine(a, b) -> float:
    """正規化を仮定しないコサイン類似度（純Python・件数は時間窓で数件なので十分速い）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def content_similarity(a: str, b: str) -> float:
    """文字bigram Jaccard（決定論・依存なし）。日本語の「言い換えただけの同内容」検出用。

    実機事故(E2E S10 2026-07-22): 放置中に判定LLMが6連続で重複を正しく拒否した後7回目で
    陥落し、前回自発発話とほぼ同内容を再生成した。プロンプト規則は既にあり破られたため、
    コードゲート（本指標）が保証層になる（規律はコードで強制）。
    """
    ba, bb = _char_bigrams(a), _char_bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


class SpeechState:
    """発話判定の loop 所有 ephemeral 状態（単一書込・同期読み・ロック不要）。"""

    def __init__(self, *, now_fn: Callable[[], float] = now_mono, log_max: Optional[int] = None,
                 stt_pending_max_sec: float = 10.0, dup_window_sec: Optional[float] = None) -> None:
        self._now = now_fn
        # 最後に「誰かが喋った」時刻（user 発話開始/到着 + eve 応答/自発発話 完了）。
        # eve も含めるのは、Eve が喋った直後に 5秒で自分にモノローグしないため（沈黙=誰も話さない時間）。
        self.last_activity_mono = now_fn()
        self.last_eval_mono = now_fn()  # 最後に発話判定した時刻（フラット再評価カデンス）
        self.user_speaking = False
        # ユーザ活動の単調カウンタ。発話判定の最中にユーザが話し始めたら（seq 変化）
        # 自発発話を中止・破棄するために使う（ユーザ優先）。eve 活動では増やさない。
        self.user_activity_seq = 0
        # J-2 ③-A: 発話終了〜STT完了(テキスト投入)の1〜3秒窓。この間は user_speaking=False かつ
        # 沈黙時計リセット済みで全ガードに「静か」と見えるため、VLM 経由トリガの自発発話が
        # 差し込まれ、直後に届くユーザ発話の処理を数十秒遅らせた（E2E S8 実測 2026-07-22）。
        # 最大寿命つき: STT ハング/クリア漏れで自発発話が永久停止しないための安全弁。
        self.stt_pending_since: Optional[float] = None
        self._stt_pending_max = float(stt_pending_max_sec)
        self.speech_log: deque = deque(maxlen=log_max or Config.SPEECH_LOG_MAX)
        # J-2 ②-2: 同内容抑制の比較元＝**自発発話専用の時間窓履歴**（観測用 speech_log と分離）。
        # speech_log は黙る判定も入る件数上限バッファなので、放置中は 5-6秒毎の False 判定で
        # 実効約50-60秒まで縮み、前回の自発発話が押し出されて同じ提案が通っていた（実測 8回中4回）。
        # 保持するのは判定LLMの下書き（＝判定と同期・連続同発話も比較できる）とその埋め込み。
        self.autonomous_log: deque = deque(maxlen=Config.SPEECH_DUP_LOG_MAX)
        # J-2 ①: 判定LLM に見せる「既出の用件」（実発話 + コードゲートが止めた下書き）。
        self.prior_log: deque = deque(maxlen=Config.SPEECH_DUP_LOG_MAX)
        self._dup_window = dup_window_sec if dup_window_sec is not None else Config.SPEECH_DUP_WINDOW_SEC

    # --- 活動マーク（VoiceLoop から呼ぶ）---
    def mark_user_speech_start(self) -> None:
        self.user_speaking = True
        self.user_activity_seq += 1
        self.last_activity_mono = self._now()

    def mark_user_utterance(self) -> None:
        """STT 結果が届いた（発話終了）。"""
        self.user_speaking = False
        self.user_activity_seq += 1
        self.last_activity_mono = self._now()

    def mark_stt_pending(self) -> None:
        """発話セグメント到着＝STT 開始（テキスト投入までの「見えない待ち窓」を可視化）。"""
        self.stt_pending_since = self._now()

    def clear_stt_pending(self) -> None:
        """STT 完了（テキスト投入後 / 失敗 / 空認識）＝窓を閉じる。"""
        self.stt_pending_since = None

    @property
    def stt_pending(self) -> bool:
        """STT 実行中か（ユーザ発話テキストが到着直前＝自発発話を差し込んではいけない窓）。"""
        if self.stt_pending_since is None:
            return False
        return (self._now() - self.stt_pending_since) < self._stt_pending_max

    def mark_eve_activity(self) -> None:
        """Eve が喋った（応答 or 自発発話 完了）→ 沈黙時計をリセット。"""
        self.last_activity_mono = self._now()

    # --- 計測 ---
    def silence_seconds(self) -> float:
        return max(0.0, self._now() - self.last_activity_mono)

    def eval_due(self, threshold: float) -> bool:
        """5秒沈黙 + フラット再評価カデンス（前回判定/活動から threshold 秒）。"""
        base = max(self.last_eval_mono, self.last_activity_mono)
        return (self._now() - base) >= threshold

    def record_decision(self, *, speak: bool, reason: str, content: str) -> None:
        """発話判定ログに記録（True/False とも・処理非関与）+ 再評価時刻を進める。"""
        self.last_eval_mono = self._now()
        self.speech_log.append(
            {"ts": now_iso(), "speak": speak, "reason": reason, "content": content}
        )

    def record_prior_item(self, content: str, *, played: bool) -> None:
        """既出の用件を記録（J-2 ①）。played=True は実際に投入した下書き、False はコードゲートが
        止めた下書き（丸投げ/同内容）。**同内容抑制の比較元(`autonomous_log`)とは別に持つ**
        （抑制下書きを比較元に混ぜると ②-2 の閾値校正 0.25/0.87 が無効化されるため）。"""
        if not (content or "").strip():
            return
        self.prior_log.append({"mono": self._now(), "content": content, "played": played})

    def prior_items(self, cap: int = 8) -> list[str]:
        """判定LLM へ見せる既出一覧（新しい順に整形・時間窓は同内容抑制と同じ）。

        実発話は窓内の全件を必ず残し、残り枠を直近の「やめた」下書きで埋める（放置が続くと
        抑制下書きが窓内に十数件たまるため上限が要る・実測 idle_fix5 で15件）。
        """
        cutoff = self._now() - self._dup_window
        while self.prior_log and self.prior_log[0]["mono"] < cutoff:
            self.prior_log.popleft()
        entries = list(self.prior_log)
        played = [e for e in entries if e["played"]]
        others = [e for e in entries if not e["played"]]
        keep = played + others[-max(0, cap - len(played)):] if cap > len(played) else played[-cap:]
        keep.sort(key=lambda e: e["mono"])
        now = self._now()
        return [
            f"[{humanize(now - e['mono'])}・{'発話済' if e['played'] else 'やめた'}] {e['content']}"
            for e in keep
        ]

    def record_autonomous_speech(self, content: str, embedding=None) -> None:
        """実際に刺激投入まで進んだ自発発話を、同内容抑制の比較元として記録。

        既知の限界: 投入後に barge-in で潰れた分も残るため、言っていない話題が最大 窓ぶん
        抑制され得る（頻度が低く実害も小さいので許容。目立つなら実発話との連携に進む）。
        """
        self.autonomous_log.append({"mono": self._now(), "content": content, "emb": embedding})

    def recent_autonomous(self) -> list[dict]:
        """時間窓内の自発発話（窓から出た古い記録は捨てる＝比較元は常に「直近」）。"""
        cutoff = self._now() - self._dup_window
        while self.autonomous_log and self.autonomous_log[0]["mono"] < cutoff:
            self.autonomous_log.popleft()
        return list(self.autonomous_log)


class SpeechDecider:
    """single-flight 発話判定 worker（FeedbackWorker と同形）。"""

    def __init__(
        self,
        *,
        state: SpeechState,
        cache,
        rag,
        prediction_state,
        queue,
        decide_fn: DecideFn,
        rag_random_k: Optional[int] = None,
        vision_state=None,  # F6 任意: 直近の画面ナレーション源（latest_vision を読む）
        tasks_provider=None,  # J-2 P2-3 任意: 実行中タスク一覧（Fix#2 の active_tasks_for_context 相当）
        dup_threshold: Optional[float] = None,  # J-2 ②-1: 同内容抑制の類似度閾値（テスト注入用）
        embedder=None,  # J-2 ②-2 任意: 二段目（意味の重複）判定用。None なら文字bigram のみ
        dup_emb_threshold: Optional[float] = None,
    ) -> None:
        self._state = state
        self._cache = cache
        self._rag = rag
        self._pred = prediction_state
        self._queue = queue
        self._decide_fn = decide_fn
        self._vision_state = vision_state
        self._tasks_provider = tasks_provider
        self._dup_threshold = dup_threshold if dup_threshold is not None else DUP_SIM_THRESHOLD
        self._embedder = embedder
        self._dup_emb_threshold = (
            dup_emb_threshold if dup_emb_threshold is not None else Config.SPEECH_DUP_EMB_THRESHOLD)
        self._k = rag_random_k if rag_random_k is not None else Config.RAG_RANDOM_K
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._idle = asyncio.Event()
        self._idle.set()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self, drain_timeout: float = 2.0) -> None:
        self._stopping = True
        self._event.set()
        if not self._idle.is_set():
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def trigger(self, source: str = "unknown") -> None:
        """発話判定を起こす（O(1)）。source は観測用（silence=沈黙監視 / vlm=画面変化。
        J-2 ③の分析でトリガ元がログに無く追跡困難だった反省）。"""
        if not self._event.is_set():
            logger.info("🔔 発話判定トリガ(%s)", source)
        self._event.set()

    def is_idle(self) -> bool:
        """処理中でなく、保留トリガも無い（single-flight の二重起動防止に使う）。"""
        return self._idle.is_set() and not self._event.is_set()

    async def _run_forever(self) -> None:
        while not self._stopping:
            await self._event.wait()
            self._event.clear()
            if self._stopping:
                break
            try:
                await self._decide_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SpeechDecider 処理で例外（継続）")

    async def _decide_once(self) -> None:
        # 値コピー snapshot（live deque を渡さない・省略マーカ除去）
        recent = self._cache.recent_for_injection() if self._cache is not None else []
        recent = [Turn(t.speaker, t.text, t.stamp) for t in recent if t.speaker != OMITTED_SPEAKER]
        surprise = int(self._pred.surprise)  # 必須・int（指標として渡す）
        last_feedback = getattr(self._pred, "last_feedback", None)  # イブの今の感情/要約
        vision = self._vision_state.fresh_vision(Config.VLM_VISION_TTL_SEC) if self._vision_state else None
        # 自律発話の材料: 今の画面/直近に**関連する記憶** ＋ **新しい切り口**（user 会話の search とは別系統）。
        # 「甘いもの検索→前のチーズケーキの話」のように画面↔記憶を結べるよう、画面+直近をクエリに引く。
        if self._rag is not None:
            # クエリの会話側は**直近のユーザ発話**（直近ターンではない）。直近ターンだと自発発話が
            # 続いた時にイブ自身の発話でイブの記憶を引く自己強化ループになり、同じ話題に留まる。
            last_user = next((t.text for t in reversed(recent) if t.speaker == "user"), None)
            q = " ".join(filter(None, [vision, last_user]))
            # J-2 ②-3: 今 recent として見せている会話区間から生まれた記憶は種にしない
            # （「さっき自分が書いた今の会話」を過去の記憶として再提示しても新しい話題にならない）。
            since = min((getattr(getattr(t, "stamp", None), "iso", "") or "") for t in recent) if recent else None
            seeds = await self._rag.autonomous_memories(q, self._k, context_since_iso=since or None)
        else:
            seeds = []
        silence = self._state.silence_seconds()
        # J-2 P2-3: 実行中タスク一覧（あれば先回り回答を避ける材料に・未配線/エラー時は None）。
        active_tasks = None
        if self._tasks_provider is not None:
            try:
                active_tasks = self._tasks_provider()
            except Exception:
                logger.exception("実行中タスク一覧の取得に失敗（注入なしで継続）")
        prior = self._state.prior_items()  # J-2 ①: 既出の用件（実発話 + 止めた下書き）
        seq0 = self._state.user_activity_seq  # 判定中にユーザが話したか検出する基準
        self._idle.clear()
        try:
            decision = await should_speak(
                surprise=surprise, silence_seconds=silence,
                recent_turns=recent, topic_seeds=seeds, decide_fn=self._decide_fn,
                last_feedback=last_feedback, vision=vision, active_tasks=active_tasks,
                prior_items=prior,
            )
        finally:
            self._idle.set()
        # ユーザ優先: 判定中にユーザが話し始めたら自発発話を**中止・破棄**（put しない）。
        if self._state.user_activity_seq != seq0:
            self._state.record_decision(speak=False, reason="ユーザ発話により自発発話を中止（削除）", content="")
            return
        # J-2 ③-A: 判定完了時に STT 待ち窓なら破棄。seq は発話終了（＝窓の開始）時に増えるため、
        # 保留イベントで判定が窓の**中**から開始すると seq0 は増えた後の値＝seq チェックを
        # すり抜ける。その最後の隙間をここで塞ぐ（tick/vlm ガードは窓内の新規トリガのみ防ぐ）。
        if self._state.stt_pending:
            self._state.record_decision(
                speak=False, reason="STT処理中（ユーザ発話が到着直前）のため自発発話を破棄", content="")
            return
        # J-2 ②-4: 根拠なく話題を相手へ丸投げするだけの下書きは抑制（同内容抑制より**前**＝
        # 埋め込み1回分を節約）。材料は判定LLMへ実際に渡したもの（種は `seed_text` と同じ表現）。
        if decision.speak:
            materials = [t.text for t in recent]
            materials += [s.seed_text() if hasattr(s, "seed_text") else getattr(s, "text", "")
                          for s in (seeds or [])]
            if vision:
                materials.append(vision)
            if is_topic_outsourcing(decision.content, materials):
                self._state.record_decision(
                    speak=False,
                    reason="観測できる根拠なしに話題を相手へ丸投げするため抑制",
                    content=decision.content,  # 何を言おうとしたかは残す（同内容抑制と同じ規約）
                )
                self._state.record_prior_item(decision.content, played=False)
                logger.info("🔇 空振り発話を抑制（材料に接地しない話題の丸投げ）: %.40s", decision.content)
                return
        # J-2 ②-1/②-2: 直近（時間窓）の自発発話と同内容なら抑制（保証層のコードゲート）。
        emb = None
        if decision.speak:
            is_dup, score, kind, emb = await self._duplicate_of(decision.content)
            if is_dup:
                # content は残す（何を言おうとして抑制されたかのデバッグ材料）。
                self._state.record_decision(
                    speak=False,
                    reason=f"直近の自発発話と同内容のため抑制（{kind}類似度{score:.2f}）",
                    content=decision.content,
                )
                self._state.record_prior_item(decision.content, played=False)
                logger.info("🔇 自発発話を同内容抑制（%s類似度%.2f）: %.40s", kind, score, decision.content)
                return
            # 二段目は await を挟む（埋め込み）。その間にユーザが話し始めていないか再確認。
            if self._state.user_activity_seq != seq0 or self._state.stt_pending:
                self._state.record_decision(
                    speak=False, reason="ユーザ発話により自発発話を中止（削除）", content="")
                return
        # True/False とも発話判定ログに記録（観測専用・処理非関与）+ 再評価カデンス更新
        self._state.record_decision(speak=decision.speak, reason=decision.reason, content=decision.content)
        if decision.speak:
            self._state.record_autonomous_speech(decision.content, emb)  # 次回以降の比較元
            self._state.record_prior_item(decision.content, played=True)
            # ②-6: 「さっき」等が実際の経過と食い違うなら、応答側に訂正許可を渡す
            stale = stale_recency_deixis(decision.content, recent)
            if stale:
                logger.info("🕒 下書きの時間表現が経過と食い違う（訂正を指示）: %.40s", decision.content)
            await self._queue.put(
                Stimulus(
                    StimulusKind.AUTONOMOUS_SPEECH,
                    AutonomousSpeech(decision.content, decision.reason, stale_reference=stale),
                    merge_key="autonomous",
                )
            )


    async def _duplicate_of(self, content: str) -> tuple[bool, float, str, Optional[list]]:
        """直近の自発発話との重複判定。返り値 = (抑制するか, スコア, 手段ラベル, 候補の埋め込み)。

        二段構え: ①文字bigram（依存なし・即時）→ ②埋め込み cos（言い換えを捕まえる）。
        ②は①を通った時だけ走り、埋め込みは投入時の記録にも再利用する（1判定につき最大1回）。
        埋め込みが使えない/失敗した場合は①のみで継続（機能を落とさない）。
        """
        entries = self._state.recent_autonomous()
        bigram = max((content_similarity(content, e["content"]) for e in entries), default=0.0)
        if bigram >= self._dup_threshold:
            return True, bigram, "文字", None
        if self._embedder is None:
            return False, bigram, "", None
        # 比較元が無くても埋め込みは取る（次回以降の比較元として記録に持たせるため。
        # ここを省くと最初の1件が emb 無しになり、二段目が永久に空振りする）。
        try:
            vec = await self._embedder.embed_query(content)
        except Exception:
            logger.exception("同内容抑制の埋め込みに失敗（文字bigram のみで継続）")
            return False, bigram, "", None
        cos = max((cosine(vec, e["emb"]) for e in entries if e.get("emb")), default=0.0)
        if cos >= self._dup_emb_threshold:
            return True, cos, "意味", vec
        return False, bigram, "", vec


class SilenceMonitor:
    """周期 tick で発話判定をトリガする loop タスク（ガードは全て code）。"""

    def __init__(
        self,
        *,
        state: SpeechState,
        decider: SpeechDecider,
        is_busy_fn: Callable[[], bool],
        tick_sec: Optional[float] = None,
        threshold_sec: Optional[float] = None,
    ) -> None:
        self._state = state
        self._decider = decider
        self._is_busy = is_busy_fn
        self._tick = tick_sec if tick_sec is not None else Config.SILENCE_TICK_SEC
        self._threshold = threshold_sec if threshold_sec is not None else Config.SILENCE_THRESHOLD_SEC
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._tick)
            try:
                self.tick()
            except Exception:
                logger.exception("SilenceMonitor tick で例外（継続）")

    def tick(self) -> bool:
        """1回分のガード判定 + 必要ならトリガ。トリガしたら True（テスト用）。"""
        if self._is_busy():
            return False  # 応答中は発火しない
        if self._state.user_speaking:
            return False  # ユーザ発話中は発火しない
        if self._state.stt_pending:
            return False  # J-2 ③-A: STT待ち窓（ユーザ発話テキストが到着直前）は発火しない
        if not self._state.eval_due(self._threshold):
            return False  # 5秒沈黙 + フラット再評価カデンス未満
        if not self._decider.is_idle():
            return False  # single-flight（処理中/保留トリガあり）
        self._decider.trigger("silence")
        return True
