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
from collections import deque
from typing import Callable, Optional

from ..clock import now_iso, now_mono
from ..config import Config
from ..context_assembler import OMITTED_SPEAKER, Turn
from ..pipeline.stimulus import Stimulus, StimulusKind
from .decider import AutonomousSpeech, DecideFn, should_speak

logger = logging.getLogger(__name__)


class SpeechState:
    """発話判定の loop 所有 ephemeral 状態（単一書込・同期読み・ロック不要）。"""

    def __init__(self, *, now_fn: Callable[[], float] = now_mono, log_max: Optional[int] = None) -> None:
        self._now = now_fn
        # 最後に「誰かが喋った」時刻（user 発話開始/到着 + eve 応答/自発発話 完了）。
        # eve も含めるのは、Eve が喋った直後に 5秒で自分にモノローグしないため（沈黙=誰も話さない時間）。
        self.last_activity_mono = now_fn()
        self.last_eval_mono = now_fn()  # 最後に発話判定した時刻（フラット再評価カデンス）
        self.user_speaking = False
        # ユーザ活動の単調カウンタ。発話判定の最中にユーザが話し始めたら（seq 変化）
        # 自発発話を中止・破棄するために使う（ユーザ優先）。eve 活動では増やさない。
        self.user_activity_seq = 0
        self.speech_log: deque = deque(maxlen=log_max or Config.SPEECH_LOG_MAX)

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
    ) -> None:
        self._state = state
        self._cache = cache
        self._rag = rag
        self._pred = prediction_state
        self._queue = queue
        self._decide_fn = decide_fn
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

    def trigger(self) -> None:
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
        seeds = self._rag.random(self._k) if self._rag is not None else []
        surprise = int(self._pred.surprise)  # 必須・int
        silence = self._state.silence_seconds()
        seq0 = self._state.user_activity_seq  # 判定中にユーザが話したか検出する基準
        self._idle.clear()
        try:
            decision = await should_speak(
                surprise=surprise, silence_seconds=silence,
                recent_turns=recent, topic_seeds=seeds, decide_fn=self._decide_fn,
            )
        finally:
            self._idle.set()
        # ユーザ優先: 判定中にユーザが話し始めたら自発発話を**中止・破棄**（put しない）。
        if self._state.user_activity_seq != seq0:
            self._state.record_decision(speak=False, reason="ユーザ発話により自発発話を中止（削除）", content="")
            return
        # True/False とも発話判定ログに記録（観測専用・処理非関与）+ 再評価カデンス更新
        self._state.record_decision(speak=decision.speak, reason=decision.reason, content=decision.content)
        if decision.speak:
            await self._queue.put(
                Stimulus(
                    StimulusKind.AUTONOMOUS_SPEECH,
                    AutonomousSpeech(decision.content, decision.reason),
                    merge_key="autonomous",
                )
            )


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
        if not self._state.eval_due(self._threshold):
            return False  # 5秒沈黙 + フラット再評価カデンス未満
        if not self._decider.is_idle():
            return False  # single-flight（処理中/保留トリガあり）
        self._decider.trigger()
        return True
