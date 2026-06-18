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
import math
from typing import Callable

from .. import clock
from .stimulus import Stimulus, StimulusKind


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

    async def put(self, s: Stimulus) -> None:
        async with self._cond:
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
