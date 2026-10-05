"""OpenAI gpt-live-transcribe（ストリーミング文字起こし）バックエンド（S0・ユーザ裁定 2026-09-25）。

話している間に音声を送り、VAD が話し終わりを確定した瞬間に commit して確定文字を待つ。
一括方式（`openai_transcribe.py`）は話し終わってから音声全体を送るので、その分だけ遅い。
途中の文字（delta）は今は使わない（応答は確定した文字から作る＝ユーザ裁定）。

## 1発話の流れ（マイクの区間フック `audio_input.py` → `MicSttInputSource` から loop 上で同期に呼ぶ）
    start(pcm)        発話開始の確定（pre-roll 分の音声を添える）
    feed(pcm)         話している間のチャンク
    commit()          採用 → 確定文字の「待ち札」を積む    /   clear()  破棄（短すぎ・小さすぎ）
    await transcribe(pcm)   待ち札の順に確定文字を受け取る（pcm は送信済みなので使わない）
待ち札が無い時の `transcribe(pcm)` は、その場で pcm を丸ごと送って commit する
（ツール・E2E ハーネスが一括で呼ぶ経路を壊さない。同じ接続を使うので一括方式への切り替えではない）。

## 接続（実測 2026-09-25・`tools/stream_stt_probe.py` と同じ設定）
- `wss://api.openai.com/v1/realtime?intent=transcription`・`session.type=transcription`・
  `turn_detection: null`（区切りは EVE の VAD が決める）・`languages: ["ja"]`・`delay: low`
- **rate は 24000 以上しか受け付けない**（16000 を送ると session.update ごと拒否）→ 24kHz に上げて送る
- セッションの寿命は 3600 秒（`expires_at`）。空いている時に張り直す
- 1本を使い回す。切れたら次の発話に向けて張り直す（黙っている間は aiohttp の ping で保つ）
- commit → `input_audio_buffer.committed`（item_id が決まる）→ `...transcription.completed`。
  commit の失敗は error の `event_id` に送った commit の event_id が入る（突き合わせに使う）

## 失敗時（ユーザ裁定 2026-09-25「フォールバックはしなくていい」）
一括方式へ自動で切り替えない。その発話はログに残して捨て（空文字を返す＝聞き返しにも回さない）、
接続は次の発話に向けて張り直す。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..config import Config
from .base import Stt
from .filter import clean_transcript
from .resample import StreamResampler

logger = logging.getLogger(__name__)

LIVE_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
# 実測 2026-09-25: `rate` に 24000 未満を送ると session.update ごと拒否される
# （"Expected a value >= 24000, but got 16000 instead."）。マイク(16kHz)は上げて送る。
API_RATE = 24000
# サーバは 100ms 未満の commit を拒否する（input_audio_buffer_commit_empty・実測）。
_MIN_COMMIT_BYTES = API_RATE * 2 // 10
# セッションの寿命は 3600 秒（実測: session.created の expires_at − 現在 ≒ 3599秒）。
# 残りがこれを切ったら、空いている時（確定待ちが無い時）に張り直す。
_RENEW_BEFORE_SEC = 300.0
_CONNECT_TIMEOUT_SEC = 10.0      # 接続〜session.updated まで
_MAX_APPEND_BYTES = API_RATE * 2  # 1回の append の上限（1秒ぶん。溜まった分をまとめる時の上限）
# 張り直しを背景で試みるのは、これ以上生きていた接続が切れた時だけ
# （張った直後に切られる状態＝設定や鍵の問題で、背景で張り直すと無限に繰り返すため）。
_BG_RECONNECT_MIN_LIFE_SEC = 10.0
_SESSION_EVENT_ID = "eve_session_update"
# 途中の文字の区切り（S6 笑い: ここで笑い判定に送る。ユーザ裁定 2026-09-26「、。でルナに判定させて」）
_BOUNDARY = "、。？！?!"


class _WsConn:
    """aiohttp の WebSocket を「dict を送る / dict を受け取る」だけの形に包む。

    テストはこれと同じ形（`send` / `recv` / `close` / `closed`）の偽物を差し込む。
    """

    def __init__(self, session, ws) -> None:
        self._session = session
        self._ws = ws

    @property
    def closed(self) -> bool:
        return self._ws.closed

    async def send(self, obj: dict) -> None:
        await self._ws.send_str(json.dumps(obj, ensure_ascii=False))

    async def recv(self) -> Optional[dict]:
        """次のイベント。切れたら None。"""
        import aiohttp

        while True:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    return json.loads(msg.data)
                except ValueError:
                    logger.warning("文字起こし: JSON でないメッセージを無視")
                    continue
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                return None
            # それ以外（BINARY 等）は使わない

    async def close(self) -> None:
        try:
            await self._ws.close()
        finally:
            await self._session.close()


async def _connect_aiohttp(url: str, api_key: str) -> _WsConn:
    import aiohttp

    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=_CONNECT_TIMEOUT_SEC))
    try:
        # heartbeat: 黙っている間も ping で接続を保つ（返事が無ければ aiohttp が閉じる→張り直し）
        ws = await session.ws_connect(
            url, headers={"Authorization": f"Bearer {api_key}"}, heartbeat=20)
    except BaseException:
        await session.close()
        raise
    return _WsConn(session, ws)


@dataclass(eq=False)
class _Utt:
    """1発話ぶんの状態（loop 上だけで読み書きする）。"""

    n: int
    fut: "asyncio.Future[str]"
    audio_bytes: int = 0            # 送る予定の 24kHz PCM（commit 可否の判定用）
    sent_bytes: int = 0             # 実際に送り終えた分
    commit_at: Optional[float] = None
    extra_wait: float = 0.0         # 一括で渡された音声のぶん確定待ちを伸ばす（下記 transcribe）
    event_id: str = ""
    item_id: Optional[str] = None
    failed: bool = False
    cleared: bool = False
    tag: int = 0                    # 呼び出し側の区間番号（途中の文字をどの発話のものか知らせる用・S6 笑い）
    partial: str = ""               # 届いた途中の文字（delta）の積み上げ
    delta_item: Optional[str] = None  # 途中の文字が来た item_id


class OpenAILiveStt(Stt):
    streaming = True

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        language: str = "ja",
        delay: Optional[str] = None,
        commit_timeout_sec: Optional[float] = None,
        api_key: Optional[str] = None,
        url: str = LIVE_URL,
        connect_fn: Optional[Callable[[], Awaitable[Any]]] = None,
        src_rate: Optional[int] = None,
    ) -> None:
        self.model = model or Config.STT_LIVE_MODEL
        self.language = language
        self.delay = Config.STT_LIVE_DELAY if delay is None else delay
        self.commit_timeout = float(
            Config.STT_LIVE_COMMIT_TIMEOUT_SEC if commit_timeout_sec is None else commit_timeout_sec)
        self._api_key = Config.OPENAI_API_KEY if api_key is None else api_key
        self._url = url
        self._connect_fn = connect_fn or (lambda: _connect_aiohttp(self._url, self._api_key))
        self._src_rate = int(src_rate or Config.SAMPLE_RATE)
        self._resampler = StreamResampler(self._src_rate, API_RATE)
        # asyncio の部品は最初に使う時（loop の上）に作る（コンストラクタは loop の外でも呼ばれる）
        self._outq: deque = deque()                 # 送信待ち (utt, kind, data)。順序どおりに送る
        self._out_evt: Optional[asyncio.Event] = None
        self._conn_lock: Optional[asyncio.Lock] = None
        self._sender: Optional[asyncio.Task] = None
        self._reader: Optional[asyncio.Task] = None
        self._bg: Optional[asyncio.Task] = None
        self._conn: Any = None
        self._conn_since: float = 0.0
        self._ready: Optional[asyncio.Event] = None
        self._session_error: Optional[str] = None
        self._expires_at: Optional[float] = None    # 壁時計（time.time 基準）
        self._cur: Optional[_Utt] = None            # 送信中の発話（start 〜 commit/clear）
        self._pending: deque = deque()              # commit 済み・まだ transcribe で受け取られていない
        self._awaiting_item: deque = deque()        # commit を送った・item_id がまだ来ていない（送った順）
        self._by_item: dict[str, _Utt] = {}
        self._by_event: dict[str, _Utt] = {}
        self._early: dict[str, dict] = {}           # 対応づけより先に届いた確定（念のため）
        self._on_conn: set = set()                  # 今の接続に何か送った・まだ結果の出ていない発話
        self._item_owner: dict[str, _Utt] = {}      # 途中の文字の item_id → 発話
        self._n = 0
        self._closing = False
        # S6 笑い: 途中の文字が「、」「。」等で区切れた時／確定した時に `fn(tag, その発話の頭からここまで)`。
        # 既定 None＝途中の文字は使わない（応答は確定した文字から作る＝ユーザ裁定）。例外は握る。
        self.on_phrase: Optional[Callable[[int, str], None]] = None

    # ------------------------------------------------------------------
    # 1発話の入口（loop 上で同期に呼ぶ）
    # ------------------------------------------------------------------

    def start(self, pcm: bytes = b"", tag: int = 0) -> None:
        """発話開始の確定。`pcm` は pre-roll（確認チャンク込み）。`tag` は呼び出し側の区間番号。"""
        self._ensure_worker()
        if self._cur is not None:
            logger.warning("文字起こし: 発話#%d が閉じられないまま次が始まった（前のものは捨てる）",
                           self._cur.n)
            self.clear()
        self._n += 1
        self._cur = _Utt(n=self._n, fut=asyncio.get_running_loop().create_future(), tag=int(tag))
        self._resampler.reset()  # 発話と発話の間は時間が飛んでいる＝変換の履歴を繋がない
        if pcm:
            self.feed(pcm)

    def feed(self, pcm: bytes) -> None:
        utt = self._cur
        if utt is None or utt.failed or not pcm:
            return
        data = self._resampler.process(pcm)
        if data:
            utt.audio_bytes += len(data)
            self._enqueue(utt, "append", data)

    def commit(self) -> None:
        """採用。確定文字は `transcribe()` が待ち札の順に受け取る。"""
        utt, self._cur = self._cur, None
        if utt is None:
            logger.warning("文字起こし: commit したが開いている発話が無い（無視）")
            return
        utt.commit_at = time.monotonic()
        self._pending.append(utt)
        if utt.failed:
            return  # 既に捨てた（接続が切れた等）。受け取り側には "" が渡る
        if utt.audio_bytes < _MIN_COMMIT_BYTES:
            # サーバは 100ms 未満の commit を拒否する。送った分は消しておく（次の発話に混ざる）
            logger.warning("文字起こし: 発話#%d を捨てた（音声が 100ms 未満）", utt.n)
            utt.cleared = True
            self._settle(utt, "")
            self._enqueue(utt, "clear", None)
            return
        utt.event_id = f"eve_commit_{utt.n}"
        self._enqueue(utt, "commit", None)

    def clear(self) -> None:
        """破棄（0.3秒ゲート・音量ゲートで捨てた発話）。サーバ側のバッファも消す。"""
        utt, self._cur = self._cur, None
        if utt is None:
            return
        utt.cleared = True
        self._settle(utt, "")
        self._enqueue(utt, "clear", None)

    def reset(self) -> None:
        """マイクの停止・再開時。開いている発話と、受け取り手のいない待ち札を捨てる。"""
        if self._cur is not None:
            self.clear()
        while self._pending:
            self._settle(self._pending.popleft(), "")

    # ------------------------------------------------------------------
    # 確定文字の受け取り
    # ------------------------------------------------------------------

    async def transcribe(self, audio_bytes: bytes) -> str:
        if self._pending:
            utt = self._pending.popleft()
        else:
            # 待ち札が無い＝マイクの区間フックを通らない呼び出し（ツール・E2E の一括経路）。
            if not audio_bytes:
                return ""
            if self._cur is not None:
                logger.warning("文字起こし: マイクの発話を送信中に一括の呼び出しが来た（捨てる）")
                return ""
            step = self._src_rate * 2  # 1秒ずつに分けて渡す（1通のメッセージを巨大にしない）
            self.start(b"")
            for i in range(0, len(audio_bytes), step):
                self.feed(audio_bytes[i:i + step])
            utt = self._cur
            self.commit()
            if self._pending and self._pending[-1] is utt:
                self._pending.pop()
            # 一括で送った音声は実時間より速く読まれるが一瞬ではない
            # （実測: 10秒の音声を一気に送って commit → 確定まで 約3.0秒）。長さのぶん待つ。
            utt.extra_wait = len(audio_bytes) / 2.0 / self._src_rate
        return await self._await_result(utt)

    async def _await_result(self, utt: _Utt) -> str:
        if not utt.fut.done():
            base = utt.commit_at if utt.commit_at is not None else time.monotonic()
            limit = self.commit_timeout + utt.extra_wait
            remain = base + limit - time.monotonic()
            try:
                await asyncio.wait_for(asyncio.shield(utt.fut), timeout=max(0.0, remain))
            except asyncio.TimeoutError:
                was_on_conn = utt in self._on_conn
                self._fail(utt, f"確定が {limit:.1f} 秒来ない")
                if was_on_conn:
                    # この接続は信用できない（対応づけもずれうる）→ 張り直す
                    async with self._lock():
                        await self._close_conn("確定待ちの打ち切り")
        try:
            return utt.fut.result()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 起動・終了
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """接続と session.update だけ済ませる（音声は送らない＝費用がかからない）。"""
        self._ensure_worker()
        await self._ensure_conn()

    async def close(self) -> None:
        self._closing = True
        for t in (self._bg, self._sender):
            if t is not None and not t.done():
                t.cancel()
        self._bg = self._sender = None
        if self._cur is not None:
            self._settle(self._cur, "")
            self._cur = None
        while self._pending:
            self._settle(self._pending.popleft(), "")
        self._outq.clear()
        await self._close_conn("終了")

    # ------------------------------------------------------------------
    # 送信
    # ------------------------------------------------------------------

    def _lock(self) -> asyncio.Lock:
        if self._conn_lock is None:
            self._conn_lock = asyncio.Lock()
        return self._conn_lock

    def _ensure_worker(self) -> None:
        self._closing = False
        if self._out_evt is None:
            self._out_evt = asyncio.Event()
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_loop())

    def _enqueue(self, utt: _Utt, kind: str, data: Optional[bytes]) -> None:
        self._outq.append((utt, kind, data))
        if self._out_evt is not None:
            self._out_evt.set()

    async def _send_loop(self) -> None:
        while True:
            while not self._outq:
                self._out_evt.clear()
                await self._out_evt.wait()
            utt, kind, data = self._outq.popleft()
            if utt.failed or (utt.cleared and kind == "append"):
                continue  # 捨てた発話の残りは送らない
            if kind == "clear" and utt.sent_bytes == 0:
                continue  # まだ何も送っていない＝消すものが無い
            if kind == "append":
                # 溜まっている同じ発話の append をまとめる（接続待ちの間に溜まった分など）
                parts = [data]
                size = len(data)
                while self._outq and size < _MAX_APPEND_BYTES:
                    u2, k2, d2 = self._outq[0]
                    if u2 is not utt or k2 != "append":
                        break
                    self._outq.popleft()
                    parts.append(d2)
                    size += len(d2)
                data = b"".join(parts)
            try:
                conn = await self._ensure_conn()
                if kind == "append":
                    self._on_conn.add(utt)
                    await conn.send({"type": "input_audio_buffer.append",
                                     "audio": base64.b64encode(data).decode("ascii")})
                    utt.sent_bytes += len(data)
                elif kind == "commit":
                    self._by_event[utt.event_id] = utt
                    self._awaiting_item.append(utt)
                    await conn.send({"type": "input_audio_buffer.commit", "event_id": utt.event_id})
                elif kind == "clear":
                    self._on_conn.discard(utt)
                    await conn.send({"type": "input_audio_buffer.clear"})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("文字起こし: 送信に失敗（発話#%d を捨てて張り直す）: %s", utt.n, e)
                self._fail(utt, "送信失敗")
                async with self._lock():
                    await self._close_conn("送信失敗")

    # ------------------------------------------------------------------
    # 接続
    # ------------------------------------------------------------------

    def _renew_due(self) -> bool:
        return (self._expires_at is not None
                and self._expires_at - time.time() < _RENEW_BEFORE_SEC)

    def _idle_for_renew(self) -> bool:
        """張り直しても失う発話が無いか（今の接続に送りかけ・確定待ちが無い）。"""
        return not self._on_conn

    async def _ensure_conn(self):
        async with self._lock():
            conn = self._conn
            if conn is not None:
                if conn.closed:
                    # 受信側が気づく前に切れていた。後始末（載っていた発話を捨てる・対応づけの列を
                    # 空にする）を先に済ませる。済ませずに張ると、古い接続の commit 待ちが残って
                    # 新しい接続の committed と対応づけがずれる。
                    await self._close_conn("切れていた")
                elif self._renew_due() and self._idle_for_renew():
                    logger.info("文字起こし: セッションの寿命が近いので張り直す")
                    await self._close_conn("寿命")
                else:
                    return conn
            return await self._connect()

    async def _connect(self):
        t0 = time.monotonic()
        self._ready = asyncio.Event()
        self._session_error = None
        self._expires_at = None
        self._conn_since = 0.0
        conn = await asyncio.wait_for(self._connect_fn(), timeout=_CONNECT_TIMEOUT_SEC)
        self._conn = conn
        self._reader = asyncio.create_task(self._read_loop(conn))
        tr: dict = {"model": self.model}
        if self.language:
            tr["languages"] = [self.language]
        if self.delay:
            tr["delay"] = self.delay
        try:
            await conn.send({"type": "session.update", "event_id": _SESSION_EVENT_ID, "session": {
                "type": "transcription",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": API_RATE},
                                    "transcription": tr, "turn_detection": None}}}})
            remain = _CONNECT_TIMEOUT_SEC - (time.monotonic() - t0)
            await asyncio.wait_for(self._ready.wait(), timeout=max(0.1, remain))
        except BaseException:
            await self._close_conn("セッション設定の送信に失敗")
            raise
        if self._session_error is not None:
            err = self._session_error
            await self._close_conn("セッション設定が拒否された")
            raise ConnectionError(f"セッション設定が拒否された: {err}")
        self._conn_since = time.monotonic()
        logger.info("文字起こし: 接続した（%.2fs・%s・delay=%s）",
                    time.monotonic() - t0, self.model, self.delay or "既定")
        return conn

    async def _close_conn(self, reason: str) -> None:
        """今の接続を閉じる（ロックは呼び出し側が持つ）。送りかけ・確定待ちの発話は失う。"""
        conn, self._conn = self._conn, None
        rd, self._reader = self._reader, None
        if conn is not None:
            self._on_disconnect(reason)
        if rd is not None and rd is not asyncio.current_task():
            rd.cancel()
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                logger.debug("文字起こし: 接続を閉じる時に例外（無視）", exc_info=True)

    def _on_disconnect(self, reason: str) -> None:
        """接続が無くなった: その接続に載っていた発話を捨てる（ユーザ裁定: 張り直しはするが救済はしない）。"""
        for utt in list(self._on_conn):
            self._fail(utt, f"接続が切れた（{reason}）")
        self._on_conn.clear()
        self._awaiting_item.clear()
        self._by_item.clear()
        self._by_event.clear()
        self._early.clear()
        if self._ready is not None and not self._ready.is_set():
            self._session_error = self._session_error or f"接続が切れた（{reason}）"
            self._ready.set()

    def _schedule_reconnect(self) -> None:
        """空いている時に切れた接続を背景で張り直す（次の発話で接続待ちをしないため）。"""
        if self._closing or (self._bg is not None and not self._bg.done()):
            return

        async def _go() -> None:
            try:
                await self._ensure_conn()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("文字起こし: 張り直しに失敗（次の発話で再試行）: %s", e)

        self._bg = asyncio.create_task(_go())

    # ------------------------------------------------------------------
    # 受信
    # ------------------------------------------------------------------

    async def _read_loop(self, conn) -> None:
        try:
            while True:
                ev = await conn.recv()
                if ev is None:
                    break
                try:
                    self._on_event(ev)
                except Exception:
                    logger.exception("文字起こし: イベント処理で例外（無視して継続）")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("文字起こし: 受信が途切れた: %s", e)
        finally:
            if self._conn is conn:  # 自分で閉じたのではない＝向こうから切れた
                lived = time.monotonic() - self._conn_since if self._conn_since else 0.0
                self._conn = None
                self._reader = None
                if not self._closing:
                    logger.warning("文字起こし: 接続が切れた（次の発話に向けて張り直す）")
                self._on_disconnect("向こうから切れた")
                if not self._closing and lived >= _BG_RECONNECT_MIN_LIFE_SEC:
                    self._schedule_reconnect()
                try:
                    await conn.close()
                except Exception:
                    pass

    def _on_event(self, ev: dict) -> None:
        ty = ev.get("type") or ""
        if ty in ("session.created", "session.updated"):
            exp = (ev.get("session") or {}).get("expires_at")
            if isinstance(exp, (int, float)) and exp > 0:
                self._expires_at = float(exp)
            if ty == "session.updated" and self._ready is not None:
                self._ready.set()
        elif ty == "input_audio_buffer.committed":
            item = ev.get("item_id")
            if not self._awaiting_item:
                logger.warning("文字起こし: 心当たりの無い committed（%s）", item)
                return
            utt = self._awaiting_item.popleft()  # commit は送った順に確定する
            self._by_event.pop(utt.event_id, None)
            utt.item_id = item
            if item:
                self._own_item(item, utt)
                early = self._early.pop(item, None)
                if early is not None:
                    self._complete(utt, early)
                elif not utt.fut.done():
                    self._by_item[item] = utt
        elif ty.endswith("input_audio_transcription.delta"):
            utt = self._delta_owner(ev.get("item_id"))
            delta = ev.get("delta") or ""
            if utt is not None and delta:
                utt.partial += delta
                if any(ch in _BOUNDARY for ch in delta):
                    self._emit_phrase(utt, utt.partial)
        elif ty.endswith("input_audio_transcription.completed"):
            item = ev.get("item_id")
            utt = self._by_item.pop(item, None) if item else None
            if utt is None:
                if item and len(self._early) < 16:
                    self._early[item] = ev
                return
            self._complete(utt, ev)
        elif ty.endswith("input_audio_transcription.failed"):
            item = ev.get("item_id")
            utt = self._by_item.pop(item, None) if item else None
            err = ev.get("error") or {}
            why = f"文字起こしに失敗: {err.get('code') or ''} {err.get('message') or ''}".strip()
            if utt is not None:
                self._fail(utt, why)
            else:
                logger.warning("文字起こし: %s（item=%s）", why, item)
        elif ty == "error":
            err = ev.get("error") or {}
            desc = f"{err.get('code') or err.get('type') or ''}: {err.get('message') or ''}"
            if self._ready is not None and not self._ready.is_set():
                # 接続直後（session.update への返事待ち）の error は設定の拒否として扱う
                self._session_error = desc
                self._ready.set()
                return
            eid = err.get("event_id")
            utt = self._by_event.pop(eid, None) if eid else None
            if utt is not None:
                # commit が拒否された＝committed は来ない。対応づけの列から外す（外さないと以後ずれる）
                try:
                    self._awaiting_item.remove(utt)
                except ValueError:
                    pass
                self._fail(utt, desc)
            else:
                logger.warning("文字起こし: API からエラー: %s", desc)

    def _own_item(self, item: str, utt: _Utt) -> None:
        self._item_owner[item] = utt
        if len(self._item_owner) > 64:  # 古い対応は捨てる（有界）
            for k in list(self._item_owner)[:-32]:
                self._item_owner.pop(k, None)

    def _delta_owner(self, item: Optional[str]) -> Optional[_Utt]:
        """途中の文字がどの発話のものか。初めての item_id は、送信中の発話の最初の途中の文字とみなす。"""
        if not item:
            return None
        utt = self._item_owner.get(item)
        if utt is not None:
            return utt
        cur = self._cur
        if cur is not None and cur.delta_item is None and not cur.failed and not cur.cleared:
            cur.delta_item = item
            self._own_item(item, cur)
            return cur
        return None

    def _emit_phrase(self, utt: _Utt, text: str) -> None:
        fn = self.on_phrase
        if fn is None or not utt.tag or not text.strip():
            return
        try:
            fn(utt.tag, text)
        except Exception:
            logger.exception("on_phrase フックで例外（無視）")

    def _complete(self, utt: _Utt, ev: dict) -> None:
        text = clean_transcript(ev.get("transcript") or "")
        if text:
            self._emit_phrase(utt, text)  # 確定した文字でも1回（最後の句点が途中の文字に無い時の判定）
        lat = time.monotonic() - utt.commit_at if utt.commit_at is not None else float("nan")
        audio_sec = utt.sent_bytes / 2.0 / API_RATE
        logger.info("📝 文字起こし確定 #%d（commit→確定 %.2fs・音声 %.1fs・%d文字）",
                    utt.n, lat, audio_sec, len(text))
        self._note_usage(ev.get("usage"), audio_sec)
        self._settle(utt, text)

    # ------------------------------------------------------------------
    # 結果の確定
    # ------------------------------------------------------------------

    def _settle(self, utt: _Utt, text: str) -> None:
        self._on_conn.discard(utt)
        if not utt.fut.done():
            utt.fut.set_result(text)

    def _fail(self, utt: _Utt, why: str) -> None:
        """その発話を捨てる（ログに残す）。受け取り側には "" が渡る。"""
        self._on_conn.discard(utt)
        if utt.item_id:
            self._by_item.pop(utt.item_id, None)
        if utt.failed:
            return
        utt.failed = True
        if not utt.fut.done():
            logger.warning("文字起こし: 発話#%d を捨てた（%s）", utt.n, why)
            utt.fut.set_result("")

    def _note_usage(self, usage: Any, audio_sec: float) -> None:
        """使用量メーターへ（あれば）。課金は音声の長さ（$0.017/分）なので秒数を残す。"""
        try:
            from ..usage import meter

            m = meter()
            if m is None:
                return
            sec, acc = audio_sec, "approx"
            if isinstance(usage, dict) and usage.get("type") == "duration" and usage.get("seconds"):
                sec, acc = float(usage["seconds"]), "exact"
            m.record(role="stt", model=self.model, prompt_tokens=0, completion_tokens=0,
                     accuracy=acc, extra={"audio_sec": round(sec, 2)})
        except Exception:
            logger.debug("STT 使用量の記録に失敗（無視）", exc_info=True)
