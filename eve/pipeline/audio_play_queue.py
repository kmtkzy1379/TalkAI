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
  **鳴っている項目は止めない（言い切る）**。項目は「、」「。」で区切った節（`response/splitter.py`）なので、
  割り込まれたら「、」「。」まで話して譲る（ユーザ裁定 2026-09-22「今まではユーザーしゃべったら速攻黙ったけど
  不自然。、。までは喋らせる」）。次の項目からは鳴らさない。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..clock import now_mono

logger = logging.getLogger(__name__)


@dataclass
class _Item:
    generation: int
    seq: int
    audio: Any
    text: str = ""  # この音声が読み上げる文（C5: 実発話の記憶用）
    # 実際に再生した時だけ呼ばれる（barge-in で**一度も鳴らなかった**文では呼ばれない＝C5）。
    on_played: Optional[Callable[[str], None]] = None
    # on_played を既に呼んだか（barge-in 時の同期フラッシュと通常完了の二重発火を防ぐ）。
    reported: bool = False


class AudioPlayQueue:
    def __init__(self, play_fn: Optional[Callable[..., Awaitable[None]]] = None) -> None:
        # play_fn 注入: F1 はフェイク（順序記録）、F2 で実 TTS 再生に差替（将来互換）。
        # 渡すのは音声だけ（割り込みでも鳴っている項目は止めない＝止める合図を渡す必要が無い）。
        self._play_fn = play_fn
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
        # 今まさに鳴っている項目（`_drain_buffer` の `await self._play` 中だけ非 None）。
        # barge-in が来た瞬間に「途中まで喋った文」を確定させるために持つ（下記 `_flush_playing`）。
        self._playing: Optional[_Item] = None
        # --- 発話権ゲート（S3 フィラー）。**既定は開いている＝既存の挙動は不変** ---
        # フィラーはこのキューの外で鳴らす（キューに入れると後から伸ばせない・応答が先に seq を
        # 予約する）。鳴らしている間はゲートを閉じ、応答の音声は**取り出す前に**待たせる。
        # 世代(generation)・seq には触らない＝割り込みの仕組みはそのまま効く。
        self._floor = asyncio.Event()
        self._floor.set()
        # ゲートが閉じたまま戻らない事故（全ターン無音）の最後の安全弁。フィラーの上限4秒＋
        # 言い切る1語＋余裕。超えたらログに残して開ける。
        self.floor_timeout_sec = 8.0
        # ゲートが閉じている間に「次に鳴らす応答の音声」ができた時の通知（フィラーを畳む合図）。
        self.on_head_waiting: Optional[Callable[[], None]] = None
        # 音が鳴り始めた時の通知（UI の on_sentence_start とは別枠＝UI の差し込みを壊さない）。
        self._start_listeners: list[Callable[[], None]] = []
        # --- 区切りの間（S7・2026-09-26）。**既定 0＝待たない（既存の挙動は不変）** ---
        # 再生器の出力を開きっぱなしにすると、項目ごとの開き直しで空いていた約0.29秒の間が消える。
        # 試聴で良しとされた「、」「。」ごとの音はこの間込みなので、直前の音（応答・フィラー・笑い）が
        # 終わってから `item_gap_sec` 空けて次の項目を鳴らす（VoiceLoop が `SPEECH_GAP_SEC` を入れる）。
        self.item_gap_sec = 0.0
        # 直前の音が終わった時刻（monotonic・0=まだ）を返す。再生器（`RealAudioPlayer.last_end`）から読む
        self.last_audio_end: Optional[Callable[[], float]] = None

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

    def _flush_playing(self) -> None:
        """鳴っている最中の項目を**その場で**「喋った」と確定させる（C5・同期・raise しない）。

        鳴っている項目は割り込みでも最後まで鳴る（言い切る・2026-09-26）ので、確定させた内容は
        実際に聞かれた内容と一致する。

        実機事故 2026-08-15（イブの発話15グループ中4グループが会話履歴に1文字も残らなかった）:
        C5 は元々「再生開始した文は途中停止でも聞かれた扱い」だが、`on_played` は
        `_drain_buffer` が `await self._play(...)` から戻った後＝**数十ms 遅れて**発火する。
        一方 barge-in は `voice_loop._barge_in` が `audio.interrupt()` → `runner.interrupt()` と
        進み、cancel を受けた `response/orchestrator.py` の `except CancelledError:` が
        **同期で** `_record_eve()` を呼ぶ。この順序だと `spoken` はまだ空で、
        「ほぼ最後まで喋った文」ごと記憶から消えていた（ユーザ指摘 2026-08-15）。

        `interrupt()` は cancel より前に走るので、ここで確定させれば記録に間に合う。
        """
        it = self._playing
        if it is None or it.reported or it.audio is None or it.on_played is None:
            return
        it.reported = True
        try:
            it.on_played(it.text)
        except Exception:
            logger.exception("on_played コールバックでエラー（barge-in 時の確定・継続）")

    def bump_generation(self) -> int:
        """barge-in / 新ターン。世代を進め、古い世代の待機・buffer を破棄する。

        鳴っている項目は止めない（言い切る）。止まるのは次の項目から。
        """
        # 世代を進める**前**に、鳴っている文を「喋った」と確定させる（C5・上記の順序問題）。
        self._flush_playing()
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

    # --- 発話権ゲート（S3 フィラー）----------------------------------------

    def hold_floor(self) -> None:
        """フィラーが発話権を取る（応答の音声は取り出す前に待たせる）。"""
        self._floor.clear()

    def release_floor(self) -> None:
        self._floor.set()

    def floor_held(self) -> bool:
        return not self._floor.is_set()

    def head_ready(self) -> bool:
        """次に鳴らす応答の音声（今の世代の next_seq）ができているか。"""
        return self._next_seq in self._buffer

    def is_playing(self) -> bool:
        """このキューの音声が今まさに鳴っているか（`is_quiet` は待ち行列しか見ない）。"""
        return self._playing is not None

    def pending_audio(self) -> bool:
        """鳴らす予定の音声が残っているか（待機キュー・順不同 buffer。ゲートの状態は見ない）。"""
        return not self._queue.empty() or bool(self._buffer)

    def add_start_listener(self, fn: Callable[[], None]) -> None:
        self._start_listeners.append(fn)

    async def _wait_floor(self) -> None:
        if self.on_head_waiting is not None:
            try:
                self.on_head_waiting()
            except Exception:
                logger.exception("on_head_waiting コールバックでエラー（待機は継続）")
        try:
            await asyncio.wait_for(self._floor.wait(), timeout=self.floor_timeout_sec)
        except asyncio.TimeoutError:
            logger.error("発話権ゲートが %.0f 秒開かない — 開けて再生を続ける（フィラー側の不具合）",
                         self.floor_timeout_sec)
            self._floor.set()

    def _gap_wait(self) -> float:
        """次の項目を鳴らすまでに待つ秒（区切りの間の残り）。0 以下なら待たない。raise しない。"""
        if self.item_gap_sec <= 0 or self.last_audio_end is None:
            return 0.0
        try:
            end = float(self.last_audio_end() or 0.0)
        except Exception:
            logger.exception("直前の音の終わりの取得で例外（間を空けずに続ける）")
            return 0.0
        if end <= 0:
            return 0.0
        return self.item_gap_sec - (now_mono() - end)

    async def _drain_buffer(self) -> None:
        # seq が現在の next_seq から contiguous な間だけ順に再生する
        while self._next_seq in self._buffer:
            if not self._floor.is_set():
                # フィラーが鳴っている。**取り出す前に**待つ（取り出してから待つと、その間の
                # 割り込みで世代が変わっても古い文を鳴らしてしまう）。待った後は while の条件から
                # 評価し直す（待っている間に割り込みで buffer が空になっていることがある）。
                await self._wait_floor()
                continue
            gap = self._gap_wait()
            if gap > 0:
                # 区切りの間。フィラーと同じく**取り出す前に**待ち、待った後は評価し直す
                await asyncio.sleep(gap)
                continue
            it = self._buffer.pop(self._next_seq)
            self._next_seq += 1
            # UI: **音が出る直前**に「この文を今から喋る」を通知する（既定 None なら何もしない）。
            # 番兵(audio=None)では鳴らないので通知しない＝on_played と同じ判定に揃える。
            # 同期・非ブロッキング契約（await を挟まない＝ここに世代変化の隙間を作らない）。
            meta = self._meta.pop((it.generation, it.seq), None)
            # 鳴っている間だけ自分を指しておく（barge-in が来たら `_flush_playing` が確定させる）。
            # **開始の通知より前に**立てる: 通知の中で割り込みが起きても（ユーザが話している途中で
            # 鳴り始めた＝この節を言い切って譲る・`voice_loop._yield_if_user_speaking`）、この項目を
            # 「喋った」と確定させてから応答が cancel される（順序が逆だと記憶から消える・C5）。
            self._playing = it
            try:
                if it.audio is not None and self.on_sentence_start is not None:
                    try:
                        self.on_sentence_start(it.text, meta)
                    except Exception:
                        logger.exception("on_sentence_start コールバックでエラー（再生は継続）")
                if it.audio is not None:
                    for fn in self._start_listeners:
                        try:
                            fn()
                        except Exception:
                            logger.exception("再生開始の通知でエラー（再生は継続）")
                await self._play(it.audio)
            finally:
                self._playing = None
            if it.audio is not None and self.on_sentence_end is not None:
                try:
                    self.on_sentence_end(it.text)
                except Exception:
                    logger.exception("on_sentence_end コールバックでエラー（再生は継続）")
            # C5: 実際に再生した項目だけ「喋った」と報告（割り込まれても鳴っている項目は言い切る）。
            # 番兵(audio=None)は再生していないので報告しない。
            # `reported` は barge-in 時に `_flush_playing` が既に報告した印＝二重に積まない。
            if it.audio is not None and it.on_played is not None and not it.reported:
                it.reported = True
                try:
                    it.on_played(it.text)
                except Exception:
                    logger.exception("on_played コールバックでエラー（再生は継続）")

    async def _play(self, audio: Any) -> None:
        if audio is None:
            return  # A1: スキップ文の番兵（TTS 失敗等）。seq 連続性のため枠だけ消費
        if self._play_fn is None:
            raise NotImplementedError("実再生は F2 で実装。F1 は play_fn を注入する。")
        # 鳴り始めたら世代が変わっても（barge-in）止めない＝この節を言い切る。
        # 旧 B3（文の途中でも即停止）はユーザ裁定 2026-09-22 で廃止。ミュートは再生器の中で即止まる。
        await self._play_fn(audio)

    def is_quiet(self) -> bool:
        """再生待ちが無いか（待機キューも順不同 buffer も空・**フィラー中でもない**）。

        `join()` と違って**待たない**。UI の停止処理は「上限つきで様子を見る」ので、
        待ち切る `join()` は使えない（1件でも詰まると永久に戻らない）。
        フィラーが発話権を持っている間は False（喋っている最中＝静かではない・S3）。
        """
        return self._queue.empty() and not self._buffer and self._floor.is_set()

    async def join(self) -> None:
        """投入済みの全項目が処理されるまで待つ（テスト用）。"""
        await self._queue.join()
