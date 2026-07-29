"""StimulusQueue — 応答起動刺激の単一窓口。

企画書の drain 規則（沈黙=最速刺激 / busy=溜める / CallFunction逐次 / vision・feedbackマージ）
をここに集約する。標準の asyncio.PriorityQueue では put 時マージ/dedup と get 時 aging が
できないため自前実装する。

- put: dedup_key 一致は捨てる / merge_key 一致は最新で置換（1件に畳む） / それ以外は追加。
- get: 待機中から実効優先度（基本優先度 − aging ブースト）が最も高いものを返す。
       aging は待ち時間が閾値を超えた低優先刺激を徐々に底上げし starvation を防ぐ（控えめ）。
- 単一 consumer が1件ずつ取り処理完了まで次を取らない → 「CallFunction 逐次」「busy 中は溜める」
  が自然に成立する。
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Callable

from .. import clock
from .stimulus import Stimulus, StimulusKind

logger = logging.getLogger(__name__)

# 墓標(tombstone)の寿命。**正しさの装置ではない**（長短どちらでも誤破棄は起きない）:
# 抑止対象の dedup_key は "task:{task_id}" で task_id は uuid4 由来の一意値
# （`task/schema.py` new_task_id）、かつ対象タスクは terminal（store の terminal-stays-terminal）。
# つまり**そのキーで将来「正当な」put が発生することは構造上あり得ない**ので、TTL は
# メモリ回収のためだけにある。逆に短い TTL は危険で、抑止からの再配達は
#   配達中ターンの完走（実測 15.0s: E2E 2026-07-29 t=449.39→464.42）
#   ＋ 配達確認LLM（model_registry に timeout 設定が無い）
#   ＋ 再配達待ち（voice_loop の _redeliver_max_wait_sec=60 + _redeliver_grace_sec=2）
# と積み上がるため、当初案の 90 秒では貫通する。
SUPPRESS_TTL_SEC = 3600.0
SUPPRESS_MAX_KEYS = 256  # 上限超過分は失効が近い順に捨てる（一意キーなので取り違えは起きない）


class StimulusQueue:
    def __init__(
        self,
        aging_threshold_s: float = 30.0,
        aging_step_s: float = 30.0,
        clock_fn: Callable[[], float] = clock.now_mono,
    ) -> None:
        self._items: list[tuple[float, Stimulus]] = []  # (enqueue_mono, stimulus)
        self._cond = asyncio.Condition()
        self._aging_threshold = aging_threshold_s
        self._aging_step = aging_step_s
        self._clock = clock_fn
        self._tombstones: dict[str, float] = {}  # dedup_key -> 失効 mono 時刻

    # --- 墓標（取消済み報告の関門）-----------------------------------------

    def _sweep(self, now: float) -> None:
        """失効した墓標を落とす（put/suppress のたびに呼ぶ遅延掃除・タイマー不要）。"""
        if self._tombstones:
            self._tombstones = {k: e for k, e in self._tombstones.items() if e > now}

    def is_suppressed(self, dedup_key: str | None) -> bool:
        """この dedup_key は抑止済みか（同期・consumer 側ゲートからも使う）。

        失効判定は `now >= expiry`（境界ちょうどは**失効**）。
        """
        if not dedup_key:
            return False  # None/空 = 「dedup しない」の意味。墓標の対象外
        expiry = self._tombstones.get(dedup_key)
        if expiry is None:
            return False
        if self._clock() >= expiry:
            self._tombstones.pop(dedup_key, None)
            return False
        return True

    def suppress(self, dedup_key: str, ttl_sec: float = SUPPRESS_TTL_SEC) -> int:
        """この dedup_key の報告を「待機中も将来分も」配達しない（同期・loop 上から呼ぶ）。

        `discard_kind` / `drain_user_texts` と同じ契約で **await を挟まない**＝単一ループ上で
        atomic。呼び手（CancelResolver）は起床から最初の await までにこれを終える必要がある
        （猶予は call_soon 1スロット：resolver 起床の直後に runner が刺激を pop する）。

        戻り値 = キューから実際に取り除いた件数。`put` の dedup により同一 dedup_key は
        高々1件しか待機できない（38-41行と同じ不変条件）ので、実質 0/1 の bool。
        0 = 「既に配達した / 今まさに配達中 / 再配達待ちで保留中」のいずれかで、呼び手は
        この3つを区別できない（報告文で断定してはいけない理由）。
        """
        if not dedup_key:
            return 0  # None キーを墓標にすると dedup_key 無しの刺激（ユーザ発話）を全部捨てる
        now = self._clock()
        self._sweep(now)
        before = len(self._items)
        self._items = [(e, s) for (e, s) in self._items if s.dedup_key != dedup_key]
        self._tombstones[dedup_key] = now + ttl_sec  # 二重 suppress は後勝ちで延長
        if len(self._tombstones) > SUPPRESS_MAX_KEYS:
            doomed = sorted(self._tombstones.items(), key=lambda kv: kv[1])
            for k, _ in doomed[: len(self._tombstones) - SUPPRESS_MAX_KEYS]:
                self._tombstones.pop(k, None)
        return before - len(self._items)

    async def put(self, s: Stimulus) -> None:
        async with self._cond:
            # 墓標は dedup/merge より**先**に見る（merge_key を持つ抑止済み刺激が
            # merge 分岐から入り込むのを防ぐ）。
            if self.is_suppressed(s.dedup_key):
                # 取消系の無ログは 2026-07-13 事故で追跡困難だった（cancel_resolver.py の反省）。
                logger.info("⚰ 抑止済み刺激を破棄: dedup=%s", s.dedup_key)
                return
            if s.dedup_key is not None:
                for _, existing in self._items:
                    if existing.dedup_key == s.dedup_key:
                        return  # 二重投入を捨てる
            if s.merge_key is not None:
                for i, (_, existing) in enumerate(self._items):
                    if existing.merge_key == s.merge_key:
                        self._items[i] = (self._clock(), s)  # 最新で置換＝1件に畳む
                        self._cond.notify()
                        return
            self._items.append((self._clock(), s))
            self._cond.notify()

    def _effective_priority(self, enq: float, s: Stimulus, now: float) -> float:
        """実効優先度（小さいほど高優先）。待ち時間が閾値超で base を底上げ。"""
        wait = max(0.0, now - enq)
        boost = 0
        if self._aging_step > 0 and wait > self._aging_threshold:
            boost = math.floor((wait - self._aging_threshold) / self._aging_step) + 1
        return s.base_priority - boost

    async def get(self) -> Stimulus:
        async with self._cond:
            while not self._items:
                await self._cond.wait()
            now = self._clock()
            best_i = 0
            best_key: tuple[float, float] | None = None
            for i, (enq, s) in enumerate(self._items):
                # tie-break は enqueue 時刻（古い順 = FIFO）
                key = (self._effective_priority(enq, s, now), enq)
                if best_key is None or key < best_key:
                    best_key = key
                    best_i = i
            _, chosen = self._items.pop(best_i)
            return chosen

    def qsize(self) -> int:
        return len(self._items)

    def snapshot(self) -> list[Stimulus]:
        """テスト/デバッグ用。待機中の刺激（順序保証なし）。"""
        return [s for _, s in self._items]

    def drain_user_texts(self) -> list[str]:
        """待機中の USER_UTTERANCE を全て取り出しテキストを返す（coalesce 用・同期）。

        単一 consumer がドレイン直後に呼ぶ前提（await を挟まないので atomic）。
        """
        texts = [str(s.payload) for _, s in self._items if s.kind == StimulusKind.USER_UTTERANCE]
        self._items = [(e, s) for (e, s) in self._items if s.kind != StimulusKind.USER_UTTERANCE]
        return texts

    def discard_kind(self, kind: StimulusKind) -> int:
        """待機中の指定 kind の刺激を全削除し削除件数を返す（同期・loop 上から呼ぶ）。

        barge-in でユーザが話し始めた時に未処理の AUTONOMOUS_SPEECH を消す等（ユーザ優先）。
        await を挟まずに呼ぶ前提（atomic）。
        """
        before = len(self._items)
        self._items = [(e, s) for (e, s) in self._items if s.kind != kind]
        return before - len(self._items)
