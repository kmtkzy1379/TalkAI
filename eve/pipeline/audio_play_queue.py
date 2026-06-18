"""AudioPlayQueue — できた音声を順に流し、割り込みで古い音声を捨てる。

旧 modules/player.py（asyncio.Queue worker + 割り込み時 drain + 別スレッド対応）を PORT し、
停止判定を **seq + generation トークン**に置換する。

なぜ generation 化か（v1 からの改善）:
  v1 は単一フラグ `interrupt_signal` を立て 200ms 後にタイマーでリセットしていたため、
  多重割り込みでリセットタスクの競合対策が必要だった。generation は単調増加する世代番号で、
  「世代が古い項目は捨てる」を番号比較だけで無曖昧に判定でき、タイマーや完了待ち Event が不要。
  順序(seq)/世代(generation)の権威をこのクラスに一元化する。

- reserve_seq(gen): 文分割時に**生成順**で seq を予約（実 TTS は順不同に完了してよい）。
- enqueue(gen, seq, audio): TTS 完了時に投入（順不同で来てよい）。
- play_worker: 同一 generation 内を seq 昇順で再生（順不同は buffer して contiguous で出す）。
                世代が古い項目は再生せず破棄。
- bump_generation(): barge-in / 新ターンで世代+1。待機・buffer の古い世代を破棄。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class _Item:
    generation: int
    seq: int
    audio: Any


class AudioPlayQueue:
    def __init__(self, play_fn: Optional[Callable[[Any], Awaitable[None]]] = None) -> None:
        # play_fn 注入: F1 はフェイク（順序記録）、F2 で実 TTS 再生に差替（将来互換）。
        self._play_fn = play_fn
        self._queue: "asyncio.Queue[_Item]" = asyncio.Queue()
        self._generation = 0
        self._next_seq = 0  # 現 generation で次に再生すべき seq
        self._buffer: dict[int, _Item] = {}  # 順不同で来た seq の待避（現 generation 分）
        self._reserve: dict[int, int] = {}  # generation -> 次に予約する seq
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """別スレッド（VAD/STT スレッド等）からの barge-in 用にメインループを渡す。"""
        self._loop = loop

    def current_generation(self) -> int:
        return self._generation

    def reserve_seq(self, generation: int) -> int:
        """文分割時に生成順で seq を予約する（再生は seq 昇順で保証される）。"""
        n = self._reserve.get(generation, 0)
        self._reserve[generation] = n + 1
        return n

    def bump_generation(self) -> int:
        """barge-in / 新ターン。世代を進め、古い世代の待機・buffer を破棄する。"""
        self._generation += 1
        self._next_seq = 0
        self._buffer.clear()
        kept: list[_Item] = []
        while not self._queue.empty():
            try:
                it = self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
            if it.generation >= self._generation:
                kept.append(it)
        for it in kept:
            self._queue.put_nowait(it)
        return self._generation

    def interrupt(self) -> None:
        """どのスレッドからでも安全に barge-in（世代を進める）。

        generation モデルなので完了待ち Event やタイマーリセットは不要（v1 比の簡素化）。
        """
        if self._loop is not None and self._loop.is_running():
            try:
                running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is self._loop:
                self.bump_generation()
            else:
                self._loop.call_soon_threadsafe(self.bump_generation)
        else:
            self.bump_generation()

    def enqueue(self, generation: int, seq: int, audio: Any) -> None:
        self._queue.put_nowait(_Item(generation, seq, audio))

    async def play_worker(self) -> None:
        while True:
            try:
                it = await self._queue.get()
                try:
                    if it.generation != self._generation:
                        continue  # 古い世代は捨てる（未来世代は発生しない）
                    self._buffer[it.seq] = it
                    await self._drain_buffer()
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:  # 再生1件の失敗で worker を止めない（起こりやすい・軽微）
                logger.exception("AudioPlayQueue 再生エラー（この項目を飛ばして継続）")

    async def _drain_buffer(self) -> None:
        # seq が現在の next_seq から contiguous な間だけ順に再生する
        while self._next_seq in self._buffer:
            it = self._buffer.pop(self._next_seq)
            self._next_seq += 1
            await self._play(it.audio)

    async def _play(self, audio: Any) -> None:
        if audio is None:
            return  # A1: スキップ文の番兵（TTS 失敗等）。seq 連続性のため枠だけ消費
        if self._play_fn is None:
            raise NotImplementedError("実再生は F2 で実装。F1 は play_fn を注入する。")
        await self._play_fn(audio)

    async def join(self) -> None:
        """投入済みの全項目が処理されるまで待つ（テスト用）。"""
        await self._queue.join()
