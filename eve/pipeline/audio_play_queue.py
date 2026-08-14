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
    text: str = ""  # この音声が読み上げる文（C5: 実発話の記憶用）
    # 実際に再生し終えた時だけ呼ばれる（barge-in で捨てられた文では呼ばれない＝C5）。
    on_played: Optional[Callable[[str], None]] = None


class AudioPlayQueue:
    def __init__(self, play_fn: Optional[Callable[..., Awaitable[None]]] = None) -> None:
        # play_fn 注入: F1 はフェイク（順序記録）、F2 で実 TTS 再生に差替（将来互換）。
        self._play_fn = play_fn
        # play_fn が should_stop（mid-sentence barge-in 用）を受け取れるか自動判定
        self._play_takes_stop = False
        if play_fn is not None:
            try:
                import inspect

                params = inspect.signature(play_fn).parameters.values()
                self._play_takes_stop = len(params) >= 2 or any(
                    p.kind == p.VAR_POSITIONAL for p in params
                )
            except (TypeError, ValueError):
                self._play_takes_stop = False
        self._queue: "asyncio.Queue[_Item]" = asyncio.Queue()
        self._generation = 0
        self._next_seq = 0  # 現 generation で次に再生すべき seq
        self._buffer: dict[int, _Item] = {}  # 順不同で来た seq の待避（現 generation 分）
        self._reserve: dict[int, int] = {}  # generation -> 次に予約する seq
        # UI（リップシンク）用の再生開始フック。**既定 None＝既存挙動は完全に不変**。
        # 属性として持つ（`__init__`/`enqueue` のシグネチャは変えない）＝ E2E ハーネスが
        # `enqueue` を固定シグネチャで wrap している（tools/search_e2e_test.py:381 /
        # tools/task_incident_replay.py:183）ため、引数を足すと **ログ0行で全ターン無音**に
        # なる（`response/orchestrator.py:411` は try の外・`:449` の gather が例外を吸う）。
        self.on_sentence_start: Optional[Callable[[str, Any], None]] = None
        # 再生が終わった/打ち切られた通知。**barge-in で口が開きっぱなしになるのを防ぐ**
        # （途中停止でも `_play` は正常 return するので必ずここに来る）。
        self.on_sentence_end: Optional[Callable[[str], None]] = None
        self._meta: dict[tuple[int, int], Any] = {}  # (generation, seq) -> 口の開閉計画
        self._meta_max = 64  # bound（barge-in で再生されなかった分の溜まり込みを防ぐ）

    def current_generation(self) -> int:
        return self._generation

    def reserve_seq(self, generation: int) -> int:
        """文分割時に生成順で seq を予約する（再生は seq 昇順で保証される）。"""
        n = self._reserve.get(generation, 0)
        self._reserve[generation] = n + 1
        return n

    def note_meta(self, generation: int, seq: int, meta: Any) -> None:
        """UI 用のメタ（口の開閉計画）を (generation, seq) に紐づけて預かる。

        **新規メソッド**なので既存の呼び出し元・モンキーパッチを一切壊さない（`enqueue` に
        引数を足す案を捨てた理由は `__init__` のコメント参照）。再生開始時に `pop` して渡す。
        """
        if meta is None:
            return
        if len(self._meta) >= self._meta_max:  # 溜まり込み防止（古いものから捨てる）
            for k in list(self._meta)[: len(self._meta) - self._meta_max + 1]:
                self._meta.pop(k, None)
        self._meta[(generation, seq)] = meta

    def bump_generation(self) -> int:
        """barge-in / 新ターン。世代を進め、古い世代の待機・buffer を破棄する。"""
        self._generation += 1
        self._next_seq = 0
        self._buffer.clear()
        # 再生されなかった文のメタを捨てる（世代が変わればもう使わない）
        self._meta = {k: v for k, v in self._meta.items() if k[0] >= self._generation}
        # 古い世代の予約カウンタを破棄（世代ごとに1エントリ増え続ける無制限増加を防ぐ）。
        self._reserve = {g: n for g, n in self._reserve.items() if g >= self._generation}
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
        """barge-in（世代を進める）。**asyncio ループ上から呼ぶ前提**。

        単一ループ所有を正とする（P2 裁定 a）。将来 OS スレッドから割り込むときは
        スレッド側で直接呼ばず、PIPELINE_DESIGN.md §9.3 の橋渡し契約
        （call_soon_threadsafe / queue→loop drain）経由にすること。
        generation モデルなので完了待ち Event やタイマーリセットは不要（v1 比の簡素化）。
        """
        self.bump_generation()

    def enqueue(
        self,
        generation: int,
        seq: int,
        audio: Any,
        text: str = "",
        on_played: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._queue.put_nowait(_Item(generation, seq, audio, text, on_played))

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
            # UI: **音が出る直前**に「この文を今から喋る」を通知する（既定 None なら何もしない）。
            # 番兵(audio=None)では鳴らないので通知しない＝on_played と同じ判定に揃える。
            # 同期・非ブロッキング契約（await を挟まない＝ここに世代変化の隙間を作らない）。
            meta = self._meta.pop((it.generation, it.seq), None)
            if it.audio is not None and self.on_sentence_start is not None:
                try:
                    self.on_sentence_start(it.text, meta)
                except Exception:
                    logger.exception("on_sentence_start コールバックでエラー（再生は継続）")
            await self._play(it.audio, it.generation)
            if it.audio is not None and self.on_sentence_end is not None:
                try:
                    self.on_sentence_end(it.text)
                except Exception:
                    logger.exception("on_sentence_end コールバックでエラー（再生は継続）")
            # C5: 実際に再生した文だけ「喋った」と報告（B3 の途中停止も聞かれた扱い）。
            # 番兵(audio=None)は再生していないので報告しない。
            if it.audio is not None and it.on_played is not None:
                try:
                    it.on_played(it.text)
                except Exception:
                    logger.exception("on_played コールバックでエラー（再生は継続）")

    async def _play(self, audio: Any, generation: int) -> None:
        if audio is None:
            return  # A1: スキップ文の番兵（TTS 失敗等）。seq 連続性のため枠だけ消費
        if self._play_fn is None:
            raise NotImplementedError("実再生は F2 で実装。F1 は play_fn を注入する。")
        if self._play_takes_stop:
            # B3: 再生中に世代が変わったら（barge-in）文の途中でも止める
            await self._play_fn(audio, lambda: generation != self._generation)
        else:
            await self._play_fn(audio)

    def is_quiet(self) -> bool:
        """再生待ちが無いか（待機キューも順不同 buffer も空）。

        `join()` と違って**待たない**。UI の停止処理は「上限つきで様子を見る」ので、
        待ち切る `join()` は使えない（1件でも詰まると永久に戻らない）。
        """
        return self._queue.empty() and not self._buffer

    async def join(self) -> None:
        """投入済みの全項目が処理されるまで待つ（テスト用）。"""
        await self._queue.join()
