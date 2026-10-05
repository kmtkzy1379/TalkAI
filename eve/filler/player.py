"""フィラーを鳴らす常駐の再生係（S3・ユーザ指定 2026-09-25・`docs/FILLER_IMPL_PLAN.md` §1.4 / §6）。

## 1発話の流れ
`on_speech_end(L)` … VAD が発話の終わりを確定した瞬間（loop 上で同期に呼ばれる）
  鳴らさない: つなぎの間が未決 / 発話 L ≤ 0.6秒（「うん」「へー」）/ 応答の生成中 / 応答の再生中 / ミュート中
  鳴らす: **その場で発話権ゲートを閉じ**（この瞬間から応答の音声は待たされる）、常駐タスクを起こす
常駐タスク … 語を1つずつ鳴らす。**鳴っている語は途中で切らない**（言い切り）。次の語を始める前に畳む条件を見る:
  応答の音声ができた / 応答が近い合図（1文目の合成開始）/ ユーザが話し出した / 空認識 / 上限4秒（次の語で超えるなら始めない）
  無言は待っている途中でも畳める（音が無いので切っても不自然にならない）
  finally でゲートを開ける（例外でも・キャンセルでも。**開かないと全ターン無音**）

再生は `AudioPlayQueue` の外（プレイヤーは共有・キューに入れると後から伸ばせない）。
世代・seq には触らない＝割り込みで古い応答が捨てられる仕組みはそのまま効く。

## 合図の宛先（1発話ごとの番号）
割り込み・空認識の合図は「どの発話のフィラーを止めるか」を番号で持つ。番号を持たないと、
前の発話への合図（例: 割り込まれた前のフィラーを止める合図）が次の発話のフィラーまで止めてしまう。

## 笑い（S6・ユーザ裁定 2026-09-26・判定は `laugh.py`）
- `request_laugh(seg)` … その場で笑う（笑いの確率 ≥ 0.9）。区間 seg のチェーンが鳴っていれば次の1つとして挟む。
  まだ話している途中なら単独で鳴らす（言い切り・ゲートを閉じて）。1発話1回まで
- `laugh_in_filler(seg)` … 話し終わりのフィラーの中で普通に1回笑う（≥ 0.75・ユーザ 2026-09-26「ただ普通に笑うほうが
  よさそう」）。チェーンが鳴っていれば次の1つとして、まだ話している途中ならチェーンの頭で。フィラーが鳴らない発話では
  笑わない。「その場で笑う」と合わせて1発話1回（先に予約していても ≥ 0.9 が来たら、その場で笑う方に回す）。
  `laugh.filler_style: tone` の時は代わりに `set_laughing(seg)`（その発話のフィラーを笑いの声色にする・旧方式）
- 笑った語も応答LLM に伝える（`take_spoken`）。笑いの声色で鳴らした時は「（笑いながら）」を添える
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from . import audio as fa
from .bank import MOOD_DEFAULT, FillerBank, Piece, _pick_word, draw, draw_first

logger = logging.getLogger(__name__)

# 応答LLM に伝えるつなぎ言葉の鮮度（話し終わりからこれ以上経ったものは渡さない）
_SPOKEN_FRESH_SEC = 15.0


def _remember(s: set, seg: int) -> None:
    """区間番号の集合に足す（古い区間は捨てる＝溜め込まない）。"""
    s.add(seg)
    if len(s) > 64:
        for k in sorted(s)[:-32]:
            s.discard(k)


@dataclass
class _Req:
    rid: int
    speech_sec: float
    t_end: float  # 発話終了（VAD 確定）の時刻（monotonic）
    wall: float = 0.0  # 同じ瞬間の壁時計（ログ用）
    seg: int = 0  # マイクの区間番号（笑い判定と対応づける）
    laugh_first: bool = False  # チェーンの頭で笑う（話し終わる直前に「その場で笑う」が来た）


@dataclass
class _ChainLog:
    rid: int
    t_end: float
    speech_sec: float
    wall: float = 0.0
    pieces: list = field(default_factory=list)  # (表示, 始まり, 終わり)＝話し終わりからの秒
    stop_reason: str = ""
    stop_at: Optional[float] = None
    near_at: Optional[float] = None
    head_at: Optional[float] = None
    limit: bool = False
    ended_at: Optional[float] = None
    emotion: str = ""  # このチェーンの声の感情（S5・無ければ空）
    mood: str = ""     # 1段目の場面の種類（S8・通常以外の時だけ）


class FillerPlayer:
    def __init__(
        self,
        bank: FillerBank,
        *,
        audio,
        play_fn: Callable[..., Awaitable[None]],
        is_busy: Callable[[], bool],
        muted: Callable[[], bool] = lambda: False,
        rng: Optional[random.Random] = None,
        emotion_fn: Callable[[], Optional[str]] = lambda: None,
    ) -> None:
        self._bank = bank
        self._t = bank.table
        self._audio = audio
        self._play = play_fn
        self._is_busy = is_busy
        self._muted = muted
        # S5: イブの今の感情（喜怒哀楽笑い・無ければ None）。チェーンを始める時に1回読み、その発話のフィラーは
        # その感情の声で鳴らす（途中で判定が届いても1つのチェーンの中では変えない）
        self._emotion_fn = emotion_fn
        # S8: 区間番号 → 1段目の場面の種類（`MoodJudge.mood`・VoiceLoop が入れる）。None なら今までどおりの抽選
        self.mood_fn: Optional[Callable[[int], str]] = None
        self._rng = rng or random.Random()
        self._task: Optional[asyncio.Task] = None
        self._wake: Optional[asyncio.Event] = None
        self._poke: Optional[asyncio.Event] = None
        self._pending: Optional[_Req] = None
        self._holding = False
        self._running_rid = 0
        self._req_seq = 0      # 採用された発話の通し番号（on_speech_end ごとに +1）
        self._stt_seq = 0      # 文字起こしの結果の通し番号（同じ発話の並びを FIFO で追う）
        self._stop_rid = 0     # この番号以下の発話のフィラーは止める
        self._stop_why: dict[int, str] = {}
        self._spoken: list[str] = []
        self._spoken_rid = 0
        self._spoken_at = 0.0
        self._spoken_taken = False
        self._log: Optional[_ChainLog] = None
        self._await_start: Optional[_ChainLog] = None  # 応答の鳴り始めを待っているチェーン
        # S6 笑い
        self._running_seg = 0
        self._laugh_req: Optional[int] = None   # 単独で笑う区間（話している途中）
        self._laugh_next = False                # 走っているチェーンに笑いを挟む
        self._laughed: set = set()              # 笑った（その場で笑う予定の）区間＝1発話1回
        self._laugh_chain: set = set()          # フィラーの中で笑う予定の区間（≥ filler・まだ鳴らしていない）
        self._laughing: set = set()             # フィラーを笑いの声色にする区間
        self._pre_spoken: dict = {}             # 話し終わる前に笑った語（区間 → [文字]）
        self._spoken_seg = 0
        self._spoken_tone = False               # 笑いの声色で鳴らした（応答LLM に「笑いながら」と伝える）
        audio.add_start_listener(self._on_audio_start)

    # ------------------------------------------------------------------
    # 起動・停止
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """フィラー（笑いを含む）を鳴らしている/鳴らす予定がある（自発発話・画面実況のガード用）。"""
        return (self._running_rid != 0 or self._running_seg != 0 or self._pending is not None
                or self._laugh_req is not None)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._events()
            self._task = asyncio.create_task(self._run(), name="filler")

    async def stop(self) -> None:
        t, self._task = self._task, None
        if t is not None:
            t.cancel()
            try:
                await t
            except BaseException:
                pass
        self._pending = None
        self._laugh_req = None
        if self._holding:
            self._holding = False
            self._audio.release_floor()

    def _events(self) -> None:
        if self._wake is None:
            self._wake = asyncio.Event()
            self._poke = asyncio.Event()

    # ------------------------------------------------------------------
    # 合図（すべて loop 上で同期に呼ばれる・例外を外に出さない）
    # ------------------------------------------------------------------

    def on_speech_end(self, speech_sec: float, seg: int = 0) -> None:
        """VAD が発話の終わりを確定した（採用した区間のみ）。鳴らすなら即ゲートを閉じる。"""
        self._events()
        now = time.monotonic()
        self._req_seq += 1
        rid = self._req_seq
        pre = list(self._pre_spoken.pop(seg, [])) if seg else []  # 話している途中に笑った分
        self._spoken, self._spoken_rid, self._spoken_at, self._spoken_taken = pre, rid, now, False
        self._spoken_seg, self._spoken_tone = seg, False
        why = None
        if self._t.join_gap_sec is None:
            why = "つなぎの間が未決（fillers.yaml の timing.join_gap_sec）"
        elif speech_sec <= self._t.min_utterance_sec:
            why = f"発話 {speech_sec:.2f}秒 ≤ {self._t.min_utterance_sec:.1f}秒"
        elif self._is_busy():
            why = "応答の生成中"
        elif self._audio.is_playing() or self._audio.pending_audio():
            why = "応答の再生中"
        elif self._muted():
            why = "ミュート中"
        if why:
            logger.info("🫧 フィラーなし #%d（%s）", rid, why)
            self._laugh_chain.discard(seg)  # フィラーの中で笑う予定だった分は笑わない（フィラーが無い）
            return
        self._audio.hold_floor()
        self._holding = True
        req = _Req(rid, speech_sec, now, time.time(), seg=seg)
        if seg and self._laugh_req == seg:
            # 単独で笑う予定だったがまだ鳴らしていない＝チェーンの頭で笑う（2回笑わない）
            self._laugh_req, req.laugh_first = None, True
        elif seg and seg in self._laugh_chain:
            req.laugh_first = True  # 話している途中に「フィラーで笑う」と決まっていた＝頭で笑う
        self._pending = req
        self._wake.set()

    # --- S6 笑い ---------------------------------------------------------

    def request_laugh(self, seg: int) -> None:
        """その場で笑う（笑いの確率 ≥ immediate）。1発話1回。"""
        if seg <= 0 or seg in self._laughed or not self._bank.laughs:
            return
        self._events()
        _remember(self._laughed, seg)
        self._laugh_chain.discard(seg)  # フィラーで笑う予定だった＝その場で笑う方に回す（1回だけ）
        if self._running_rid and self._running_seg == seg:
            self._laugh_next = True  # その発話のフィラーが鳴っている＝次の1つとして挟む
            self._poke.set()
            return
        if self._pending is not None and self._pending.seg == seg:
            self._pending.laugh_first = True
            return
        why = None
        if self._is_busy():
            why = "応答の生成中"
        elif self._audio.is_playing() or self._audio.pending_audio():
            why = "応答の再生中"
        elif self._muted():
            why = "ミュート中"
        if why:
            logger.info("😄 笑わない #%d（%s）", seg, why)
            return
        self._audio.hold_floor()
        self._holding = True
        self._laugh_req = seg
        self._wake.set()

    def laugh_in_filler(self, seg: int) -> None:
        """話し終わりのフィラーに笑いを含ませる（笑いの確率 ≥ filler）。

        既定（`laugh.filler_style: laugh`）はフィラーの中で普通に1回笑う。`tone` なら語を笑いの声色で鳴らす。
        """
        if self._t.laugh_filler_style == "tone":
            self.set_laughing(seg)
            return
        if seg <= 0 or seg in self._laughed or seg in self._laugh_chain or not self._bank.laughs:
            return
        self._events()
        _remember(self._laugh_chain, seg)
        if self._running_rid and self._running_seg == seg:
            self._laugh_next = True  # その発話のフィラーが鳴っている＝次の1つとして挟む
            self._poke.set()
        elif self._pending is not None and self._pending.seg == seg:
            self._pending.laugh_first = True
        # それ以外（まだ話している途中）は、話し終わりのフィラーの頭で笑う（on_speech_end）

    def set_laughing(self, seg: int) -> None:
        """その発話のフィラーを笑いの声色にする（`laugh.filler_style: tone` の時・笑いの確率 ≥ filler）。"""
        if seg <= 0 or not self._bank.laugh_words:
            return
        _remember(self._laughing, seg)

    def on_response_near(self) -> None:
        """応答の1文目の合成を始めた（音声まで約0.2〜0.35秒）。"""
        log = self._log
        if log is not None and log.near_at is None:
            log.near_at = time.monotonic() - log.t_end
        self._signal("応答が近い", self._req_seq)

    def on_head_ready(self) -> None:
        """ゲートが閉じている間に、次に鳴らす応答の音声ができた。"""
        log = self._log
        if log is not None and log.head_at is None:
            log.head_at = time.monotonic() - log.t_end
        self._signal("応答の音声ができた", self._req_seq)

    def on_barge_in(self) -> None:
        """ユーザが話し出した: 鳴っている語は言い切り、次の語は始めない。"""
        self._signal("ユーザが話し出した", self._req_seq)

    def on_stt_result(self, put: bool) -> None:
        """文字起こしの結果（採用した発話の順に1回ずつ）。put=False は応答が来ない＝止める。"""
        self._stt_seq += 1
        if put:
            return
        sid = self._stt_seq
        if self._spoken_rid == sid:
            self._spoken_taken = True  # この発話への応答は無い＝つなぎ言葉を次に持ち越さない
        self._signal("空認識", sid)

    def take_spoken(self) -> Optional[str]:
        """応答LLM に伝えるつなぎ言葉（**応答のメッセージを組み立てる時点までに鳴らし始めた語**）。

        1回だけ渡す（次の刺激に持ち越さない）。最新の発話のものだけ・話し終わりから15秒以内。
        """
        if (self._spoken_taken or not self._spoken or self._spoken_rid != self._req_seq
                or time.monotonic() - self._spoken_at > _SPOKEN_FRESH_SEC):
            return None
        self._spoken_taken = True
        return "、".join(self._spoken) + ("（笑いながら）" if self._spoken_tone else "")

    def _signal(self, why: str, rid: int) -> None:
        if rid <= 0:
            return
        if rid > self._stop_rid:
            self._stop_rid = rid
        self._stop_why.setdefault(rid, why)
        if len(self._stop_why) > 32:
            for k in sorted(self._stop_why)[:-16]:
                self._stop_why.pop(k, None)
        if self._poke is not None:
            self._poke.set()

    def _stopped(self, rid: int) -> bool:
        return self._stop_rid >= rid or self._audio.head_ready()

    # ------------------------------------------------------------------
    # 常駐タスク
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            # 笑い（話している途中）→ チェーン の順に、溜まっている分を全部こなしてからゲートを開ける
            while self._laugh_req is not None or self._pending is not None:
                try:
                    if self._laugh_req is not None:
                        seg, self._laugh_req = self._laugh_req, None
                        await self._lone_laugh(seg)
                    else:
                        req, self._pending = self._pending, None
                        await self._chain(req)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("フィラーで例外（ゲートを開けて続行）")
                finally:
                    self._running_rid = 0
                    self._running_seg = 0
            if self._holding:
                self._holding = False
                self._audio.release_floor()

    async def _lone_laugh(self, seg: int) -> None:
        """話している途中に単独で笑う（言い切り）。"""
        clip = self._bank.pick_laugh(self._rng)
        if clip is None:
            return
        self._running_seg = seg
        t0 = time.monotonic()
        await self._play_word(fa.to_wav(clip.samples, clip.sr), clip.text)
        if self._spoken_seg == seg:
            self._spoken.append(clip.text)  # 話し終わった後（フィラー無しの短い発話など）
        else:
            self._pre_spoken.setdefault(seg, []).append(clip.text)
            if len(self._pre_spoken) > 16:
                for k in sorted(self._pre_spoken)[:-8]:
                    self._pre_spoken.pop(k, None)
        logger.info("😄 笑った #%d: %s（%.2f秒・話の途中）", seg, clip.text, time.monotonic() - t0)

    async def _chain(self, req: _Req) -> None:
        t, sr = self._t, self._bank.sr
        self._running_rid = req.rid
        self._running_seg = req.seg
        self._laugh_next = False
        log = _ChainLog(req.rid, req.t_end, req.speech_sec, req.wall)
        self._log = log
        try:
            emo = self._emotion_fn() or None
        except Exception:
            logger.exception("今の感情の取得に失敗（普通の声で鳴らす）")
            emo = None
        log.emotion = emo or ""
        # このチェーンが鳴らした語（応答LLM に伝える分）。自分専用のリストに積み、最新の発話の
        # チェーンの時だけ公開する（前の発話のチェーンの語が次の発話に混ざらないように）。
        words: list[str] = []
        public = self._spoken_rid == req.rid
        if public:
            words = self._spoken  # 話している途中に笑った分（あれば）の続きに積む
        t0 = time.monotonic()
        last_word: Optional[str] = None
        last_kind: Optional[str] = None
        first = True
        nxt: Optional[Piece] = None
        used: dict = {}  # このチェーンで鳴らした語 → 回数（同じ語の繰り返しを少し減らす・repeat_factor）
        drew_first = False  # S8: 場面で選ぶのは最初の抽選（1段目）だけ。2段目以降は今までどおり
        try:
            while True:
                if self._stopped(req.rid):
                    log.stop_reason = self._stop_why.get(req.rid) or (
                        "応答の音声ができた" if self._audio.head_ready() else "止める合図")
                    log.stop_at = time.monotonic() - req.t_end
                    break
                if req.laugh_first or self._laugh_next:
                    # 笑い（その場で笑う ≥ immediate / フィラーに笑い ≥ filler）をこの発話のフィラーの中で鳴らす
                    req.laugh_first = self._laugh_next = False
                    if req.seg:
                        _remember(self._laughed, req.seg)  # 1発話1回（後から ≥ immediate が来ても笑わない）
                        self._laugh_chain.discard(req.seg)
                    clip = self._bank.pick_laugh(self._rng)
                    if clip is not None:
                        elapsed = time.monotonic() - t0
                        gap = self._gap_before(first, last_kind)
                        if elapsed + gap + clip.sec > t.max_chain_sec:
                            log.limit = True
                            break
                        if gap > 0 and not await self._wait_gap(req.rid, gap):
                            continue  # 間の途中で止める合図（ループの先頭で理由を残して抜ける）
                        words.append(clip.text)
                        s = time.monotonic() - req.t_end
                        await self._play_word(fa.to_wav(clip.samples, sr), clip.text)
                        log.pieces.append((("、" if gap > 0 else "") + clip.text + "(笑)", s,
                                           time.monotonic() - req.t_end))
                        last_kind = "laugh"
                        first = False
                        continue
                piece_mood = None
                if nxt is None and not drew_first and self.mood_fn is not None:
                    piece_mood = self._mood_for(req.seg)
                    if piece_mood != MOOD_DEFAULT:
                        log.mood = piece_mood
                    piece = draw_first(t, self._rng, piece_mood)
                else:
                    piece = nxt or draw(t, self._rng, first=first, last_word=last_word,
                                        last_kind=last_kind, stretch_ok=self._bank.stretch_ok(), used=used)
                drew_first = True
                nxt = None
                if piece.kind == "stretch":  # 伸ばし棒は先読みでしか鳴らせない（下）→ 別の語に回す
                    piece = Piece("word", _pick_word(t, self._rng, last_word, used))
                elapsed = time.monotonic() - t0
                if piece.kind == "silence":
                    sec = self._silence_len(piece.sec, first, last_kind)
                    if elapsed + sec > t.max_chain_sec:
                        log.limit = True
                        break
                    s = time.monotonic() - req.t_end
                    await self._wait_silence(req.rid, sec)
                    log.pieces.append((f"〔無言{sec:.2f}〕", s, time.monotonic() - req.t_end))
                    last_kind = "silence"
                    first = False
                    continue
                tone = req.seg in self._laughing and piece.word_id in self._bank.laugh_words
                k, clip = self._bank.pick(piece.word_id, self._rng, laughing=tone, emotion=emo, mood=piece_mood)
                samples, text, kind = clip.samples, clip.text, piece.kind
                # 笑いの声色・感情の声の語には伸ばし棒の素材が無い＝伸ばさない（次の1つは改めて抽選）
                emo_voice = not tone and self._bank.has_emotion(emo, piece.word_id)
                mood_voice = (piece_mood, piece.word_id) in self._bank.mood_clips
                if piece.kind == "word" and not tone:
                    # 先読み: 次の1つをここで決める（言い切った後では継げない）。
                    # 伸ばし棒 → この語に「ー」を継いだものを鳴らす（普通の声の語だけ）
                    after = dict(used)
                    after[piece.word_id] = after.get(piece.word_id, 0) + 1  # これから鳴らす語も数える
                    la = draw(t, self._rng, first=False, last_word=piece.word_id, last_kind="word",
                              stretch_ok=self._bank.stretch_ok() and not emo_voice and not mood_voice,
                              used=after)
                    if la.kind == "stretch":
                        st = self._bank.stretched.get((piece.word_id, k))
                        if st is not None:
                            samples, text, kind = st.samples, st.text, "stretch"
                    else:
                        nxt = la
                gap = self._gap_before(first, last_kind)
                dur = gap + len(samples) / sr
                if elapsed + dur > t.max_chain_sec:
                    log.limit = True  # 次の語で上限を超える＝始めない（4秒以上は不自然）
                    break
                if gap > 0 and not await self._wait_gap(req.rid, gap):
                    continue  # 間の途中で止める合図（ループの先頭で理由を残して抜ける）
                wav = fa.to_wav(samples, sr)
                words.append(text)
                if tone and public:
                    self._spoken_tone = True
                s = time.monotonic() - req.t_end
                await self._play_word(wav, text)  # 言い切り（止めない）
                log.pieces.append((("、" if gap > 0 else "") + text + ("🤭" if tone else ""), s,
                                   time.monotonic() - req.t_end))
                if piece.kind == "word":
                    last_word = piece.word_id
                    used[piece.word_id] = used.get(piece.word_id, 0) + 1
                last_kind = kind
                first = False
        finally:
            log.ended_at = time.monotonic() - req.t_end
            self._running_rid = 0
            self._running_seg = 0
            self._log = None
            self._write_log(log)
            if log.pieces:
                self._await_start = log  # 応答が鳴り始めたら時刻を1行足す

    def _mood_for(self, seg: int) -> str:
        """S8: その区間の1段目の種類（判定が無い・失敗は「通常」）。"""
        try:
            return (self.mood_fn(seg) if seg else MOOD_DEFAULT) or MOOD_DEFAULT
        except Exception:
            logger.exception("場面の種類の取得に失敗（通常で鳴らす）")
            return MOOD_DEFAULT

    def _gap_before(self, first: bool, last_kind: Optional[str]) -> float:
        """次に鳴らすものの前の「、」の間（1つ目と無言の直後は無し）。

        特殊フィラー（例 えーっと）は語と っと・っとね をまとめて合成した1つの音＝他の語と同じく間を空ける。
        """
        if first or last_kind == "silence":
            return 0.0
        return self._t.join_gap_sec or 0.0

    def _silence_len(self, sec: float, first: bool, last_kind: Optional[str]) -> float:
        """抽選で出た無言の長さ。**語の直後の無言は、語と語の間（join_gap_sec）より短くしない**。

        無言の後の語には「、」の間を付けないので、語 → 無言0.3〜0.5秒 → 語 だと、語と語が 0.3〜0.5秒で
        続いてしまう（実測 2026-09-28 通常会話: え → 無言0.45 → あ）。ユーザ 2026-09-28「間を1.2秒にして
        そもそも2回目ほぼ出させない」に合わせ、2つのフィラーの間は無言が挟まっても 1.2秒あける。
        1段目の無言（最初に黙る・0.3〜0.5秒）はそのまま。
        """
        if first or last_kind == "silence":
            return sec
        return max(sec, self._t.join_gap_sec or 0.0)

    async def _wait_gap(self, rid: int, sec: float) -> bool:
        """「、」の間を待つ。**間の途中で止める合図（応答が近い 等）が来たら False**＝次の語は言わない。

        以前は間を次の語の音の頭に無音として付けて1つの音で鳴らしていたので、間の途中で合図が来ても
        次の語まで言い切っていた（実測 2026-09-28: 合図 +1.98 → 次の あー を言い切って +2.83）。
        間を 1.2秒にすると（ユーザ 2026-09-28）この遅れが最大で「間＋1語」になるため、間は止められるようにした。
        語そのものは今までどおり言い切る（途中で切らない）。
        """
        await self._wait_silence(rid, sec)
        return not self._stopped(rid)

    async def _wait_silence(self, rid: int, sec: float) -> None:
        """無言の間。止める合図が来たら途中でやめる（音が無いので切っても不自然にならない）。"""
        end = time.monotonic() + sec
        while not self._stopped(rid):
            remain = end - time.monotonic()
            if remain <= 0:
                return
            self._poke.clear()
            try:
                await asyncio.wait_for(self._poke.wait(), timeout=remain)
            except asyncio.TimeoutError:
                return

    async def _play_word(self, wav: bytes, text: str) -> None:
        start = getattr(self._audio, "on_sentence_start", None)
        end = getattr(self._audio, "on_sentence_end", None)
        if start is not None:
            try:
                from ..response.lipsync import plan_from_wav

                start(text, plan_from_wav(wav))  # 口パク（UI）。再生と同じ音声から作る
            except Exception:
                logger.exception("フィラーの口パク通知で例外（再生は続行）")
        try:
            await self._play(wav)
        finally:
            if end is not None:
                try:
                    end(text)
                except Exception:
                    logger.exception("フィラーの口パク終了通知で例外（無視）")

    # ------------------------------------------------------------------
    # ログ（ターンごと・1行）
    # ------------------------------------------------------------------

    def _write_log(self, log: _ChainLog) -> None:
        pieces = " ".join(f"{p}[{a:.2f}-{b:.2f}]" for p, a, b in log.pieces) or "（鳴らさず）"
        end = (f"{log.stop_reason}(+{log.stop_at:.2f})" if log.stop_reason
               else ("上限" if log.limit else "—"))
        extra = []
        if log.near_at is not None:
            extra.append(f"応答が近い +{log.near_at:.2f}")
        if log.head_at is not None:
            extra.append(f"1文目の音声 +{log.head_at:.2f}")
        wall = time.strftime("%H:%M:%S", time.localtime(log.wall)) + f".{int(log.wall * 1000) % 1000:03d}"
        logger.info("🫧 フィラー #%d: 発話終了 %s・発話 %.2f秒%s | %s | 終わり: %s | 畳み終え +%.2f%s",
                    log.rid, wall, log.speech_sec,
                    (f"・感情 {log.emotion}" if log.emotion else "") + (f"・場面 {log.mood}" if log.mood else ""),
                    pieces, end,
                    log.ended_at or 0.0, (" | " + " / ".join(extra)) if extra else "")

    def _on_audio_start(self) -> None:
        log, self._await_start = self._await_start, None
        if log is None:
            return
        now = time.monotonic() - log.t_end
        logger.info("🫧 応答の鳴り始め #%d: 話し終わり +%.2f秒（フィラーを畳んでから +%.2f秒）",
                    log.rid, now, now - (log.ended_at or now))
