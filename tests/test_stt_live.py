"""S0 ストリーミング文字起こしの決定論テスト（API 不要・フェイクの WebSocket）。

- 24kHz への変換（長さ・利得・チャンクをまたいだ連続性）
- マイクの区間フック（開始→途中→終了の順・採用/破棄・声の長さ・speech_queue より先に終了を知らせる）
- OpenAILiveStt: 開始→送信→commit→確定 / 破棄→clear / 切断→再接続 / 確定が来ない→打ち切り /
  commit の拒否で対応づけがずれない / 設定の拒否 / 一括の呼び出し / 寿命前の張り直し / reset
- MicSttInputSource: ストリーミングの配線（区間フック→STT・発話終了→on_speech_end・投入の有無）

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_stt_live.py
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

from eve.config import Config  # noqa: E402
from eve.pipeline.stimulus import StimulusKind  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.stt.openai_live import API_RATE, OpenAILiveStt  # noqa: E402
from eve.stt.resample import StreamResampler  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}  {detail}")


def pcm16(sec: float, rate: int = 16000, freq: float = 440.0, amp: float = 0.3) -> bytes:
    t = np.arange(int(sec * rate)) / rate
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# フェイクの gpt-live-transcribe
# ---------------------------------------------------------------------------

class FakeConn:
    def __init__(self, server: "FakeServer") -> None:
        self.server = server
        self.sent: list[dict] = []
        self._in: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.buf = 0

    async def send(self, obj: dict) -> None:
        if self.closed:
            raise ConnectionResetError("closed")
        self.sent.append(obj)
        self.server.on_client(self, obj)

    async def recv(self):
        ev = await self._in.get()
        return ev

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._in.put_nowait(None)

    def push(self, ev) -> None:
        if not self.closed:
            self._in.put_nowait(ev)

    def drop(self) -> None:
        """サーバ側から切る。"""
        self.closed = True
        self._in.put_nowait(None)


class FakeServer:
    def __init__(self, *, reply: bool = True, reject_session: bool = False,
                 commit_error_at: set | None = None, expires_in: float = 3600.0,
                 delay: float = 0.0, fail_connect: int = 0) -> None:
        self.reply = reply
        self.reject_session = reject_session
        self.commit_error_at = commit_error_at or set()
        self.expires_in = expires_in
        self.delay = delay
        self.fail_connect = fail_connect
        self.conns: list[FakeConn] = []
        self.items = 0
        self.commits = 0
        self.completed_reversed = False
        self._held: list = []

    async def connect(self) -> FakeConn:
        if self.fail_connect > 0:
            self.fail_connect -= 1
            raise OSError("connect refused")
        c = FakeConn(self)
        self.conns.append(c)
        c.push({"type": "session.created",
                "session": {"expires_at": time.time() + self.expires_in}})
        return c

    def on_client(self, c: FakeConn, obj: dict) -> None:
        ty = obj.get("type")
        if ty == "session.update":
            if self.reject_session:
                c.push({"type": "error", "error": {"code": "bad_rate", "message": "rejected",
                                                    "event_id": obj.get("event_id")}})
            else:
                c.push({"type": "session.updated",
                        "session": {"expires_at": time.time() + self.expires_in}})
        elif ty == "input_audio_buffer.append":
            c.buf += len(base64.b64decode(obj["audio"]))
        elif ty == "input_audio_buffer.clear":
            c.buf = 0
            c.push({"type": "input_audio_buffer.cleared"})
        elif ty == "input_audio_buffer.commit":
            self.commits += 1
            if self.commits in self.commit_error_at or c.buf == 0:
                c.push({"type": "error", "error": {"code": "input_audio_buffer_commit_empty",
                                                    "message": "x", "event_id": obj.get("event_id")}})
                c.buf = 0
                return
            self.items += 1
            item = f"item_{self.items}"
            nbytes, c.buf = c.buf, 0
            c.push({"type": "input_audio_buffer.committed", "item_id": item})
            if not self.reply:
                return
            ev = {"type": "conversation.item.input_audio_transcription.completed",
                  "item_id": item, "transcript": f"{nbytes}バイト",
                  "usage": {"type": "duration", "seconds": 1}}
            if self.completed_reversed:
                self._held.append((c, ev))
                if len(self._held) == 2:  # 2件そろったら逆順で返す
                    for cc, e in reversed(self._held):
                        cc.push(e)
                    self._held.clear()
            elif self.delay:
                asyncio.get_running_loop().call_later(self.delay, c.push, ev)
            else:
                c.push(ev)


def mk(server: FakeServer, **kw) -> OpenAILiveStt:
    kw.setdefault("commit_timeout_sec", 1.0)
    return OpenAILiveStt(api_key="x", connect_fn=server.connect, src_rate=16000, **kw)


def expected_bytes(pcm: bytes) -> int:
    return len(StreamResampler(16000, 24000).process(pcm))


def stream_utt(stt: OpenAILiveStt, pcm: bytes, chunk: int = 1024) -> None:
    """マイクと同じ形で流す（最初の塊を start に添え、残りを feed）。"""
    stt.start(pcm[:chunk])
    for i in range(chunk, len(pcm), chunk):
        stt.feed(pcm[i:i + chunk])


# ---------------------------------------------------------------------------
# 変換
# ---------------------------------------------------------------------------

def t_resample() -> None:
    x = pcm16(1.0, freq=1000)
    r = StreamResampler(16000, 24000)
    whole = r.process(x)
    check("変換: 16k→24k で長さ 1.5倍", len(whole) == len(x) * 3 // 2, f"{len(whole)}")
    r2 = StreamResampler(16000, 24000)
    chunked = b"".join(r2.process(x[i:i + 1024]) for i in range(0, len(x), 1024))
    a = np.frombuffer(whole, "<i2").astype(int)
    b = np.frombuffer(chunked, "<i2").astype(int)
    check("変換: チャンクに分けても一括と同じ（継ぎ目で切れない）",
          len(a) == len(b) and int(np.max(np.abs(a - b))) <= 1)
    y = np.frombuffer(whole, "<i2").astype(float)[3000:-3000]
    amp = np.sqrt(2 * np.mean(y ** 2)) / (0.3 * 32767)
    check("変換: 1kHz の振幅を保つ", abs(amp - 1.0) < 0.02, f"{amp:.3f}")
    same = StreamResampler(24000, 24000)
    check("変換: 同じレートは素通し", same.process(x) == x)


# ---------------------------------------------------------------------------
# マイクの区間フック
# ---------------------------------------------------------------------------

def t_tracker() -> None:
    from eve.audio_input import _SegmentTracker

    ev: list = []
    puts: list = []
    tr = _SegmentTracker(
        rate=16000, chunk=512, put=lambda b: (ev.append(("put", len(b))), puts.append(b)),
        on_speech_start=lambda: ev.append(("barge",)),
        on_segment_start=lambda b: ev.append(("start", len(b))),
        on_segment_audio=lambda b: ev.append(("audio", len(b))),
        on_segment_end=lambda ok, sec: ev.append(("end", ok, round(sec, 3))),
    )
    ch = b"\x01\x00" * 512  # 1チャンク = 1024 バイト
    now = 0.0
    for _ in range(5):  # 無音（pre-roll に溜まる）
        tr.step(ch, 0.1, 0.01, now)
        now += 0.032
    for _ in range(3):  # 発話 3チャンク目で開始確定（VAD_ONSET_SEC=0.12→3チャンク）
        tr.step(ch, 0.9, 0.2, now)
        now += 0.032
    check("区間: 発話開始の確定で start（pre-roll 5+確認3=8チャンク）",
          ev[:2] == [("start", 8 * 1024), ("barge",)], f"{ev[:3]}")
    for _ in range(10):
        tr.step(ch, 0.9, 0.2, now)
        now += 0.032
    n_audio_speech = sum(1 for e in ev if e[0] == "audio")
    check("区間: 発話中のチャンクを audio で流す", n_audio_speech == 10, f"{n_audio_speech}")
    while not puts:
        tr.step(ch, 0.1, 0.01, now)
        now += 0.032
    end = [e for e in ev if e[0] == "end"]
    check("区間: 無音 0.5秒で終了・採用", end and end[0][1] is True, f"{end}")
    # 声の長さ = 確認窓の頭〜最後の発話チャンク = 3+10 チャンク
    check("区間: 声の長さは pre-roll の手前と無音の尾を含まない（13×32ms）",
          end and abs(end[0][2] - 13 * 512 / 16000) < 1e-6, f"{end}")
    i_end = next(i for i, e in enumerate(ev) if e[0] == "end")
    i_put = next(i for i, e in enumerate(ev) if e[0] == "put")
    check("区間: speech_queue に流す前に終了を知らせる（commit が待ち札より先）", i_end < i_put)
    total_audio = sum(e[1] for e in ev if e[0] in ("start", "audio"))
    check("区間: フックに流した音声 = speech_queue に流した区間（中身が同じ）",
          total_audio == len(puts[0]), f"{total_audio} vs {len(puts[0])}")

    # 最短発話ゲート（ユーザ 2026-09-29「ノイズカットは修正」）: **声の長さ**で判定する。以前は pre-roll と
    # 無音の尾込みの長さを 0.3秒と比べていたので、本番の値（SILENCE_LIMIT=0.5）では一度も効いていなかった。
    # ここは本番と同じ SILENCE_LIMIT・確認窓のまま確かめる（値を 0 にして届かせるテストにしない）。
    import logging

    class _Grab(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.msgs: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.msgs.append(record.getMessage())

    def run_burst(voice_chunks: int, preroll_chunks: int = 12) -> tuple:
        ev3: list = []
        puts3: list = []
        tr3 = _SegmentTracker(rate=16000, chunk=512, put=puts3.append,
                              on_segment_end=lambda ok, sec: ev3.append((ok, round(sec, 3))))
        t = 0.0
        for _ in range(preroll_chunks):  # 無音（pre-roll に溜まる）
            tr3.step(ch, 0.1, 0.01, t)
            t += 0.032
        for _ in range(voice_chunks):  # 声
            tr3.step(ch, 0.9, 0.2, t)
            t += 0.032
        while not ev3:  # 無音で区間が終わるまで
            tr3.step(ch, 0.1, 0.01, t)
            t += 0.032
        return ev3[0][0], ev3[0][1], bool(puts3)

    grab = _Grab()
    audio_logger = logging.getLogger("eve.audio_input")
    saved_level = audio_logger.level
    saved_disable = logging.root.manager.disable  # このファイルは冒頭でログを止めている
    logging.disable(logging.NOTSET)
    audio_logger.setLevel(logging.INFO)
    audio_logger.propagate = False  # 取るだけ（テストの出力には出さない）
    audio_logger.addHandler(grab)
    saved = (Config.MIN_SPEECH_SEC, Config.SILENCE_LIMIT, Config.VAD_ONSET_SEC)
    try:
        Config.SILENCE_LIMIT, Config.VAD_ONSET_SEC = 0.5, 0.12
        Config.MIN_SPEECH_SEC = 0.0
        ok0, sec0, put0 = run_burst(3)
        check("最短発話: 既定 0 は長さで捨てない（確認窓ちょうどの 0.096秒でも採用＝今までの挙動）",
              ok0 is True and put0 and sec0 == 0.096, f"{(ok0, sec0, put0)}")
        Config.MIN_SPEECH_SEC = 0.2
        ok1, sec1, put1 = run_burst(4)
        check("最短発話: 声の長さが閾値未満は破棄（accepted=False・speech_queue に流さない）",
              ok1 is False and not put1 and sec1 == 0.128, f"{(ok1, sec1, put1)}")
        check("最短発話: 破棄はログに残す（黙って捨てない）",
              any("最短発話ゲートで破棄" in m for m in grab.msgs), f"{grab.msgs}")
        ok2, sec2, put2 = run_burst(7)
        check("最短発話: 閾値以上は採用", ok2 is True and put2 and sec2 == 0.224, f"{(ok2, sec2, put2)}")
        Config.SILENCE_LIMIT = 1.5
        ok3, sec3, put3 = run_burst(4)
        check("最短発話: 無音の尾を長くしても判定は変わらない（声の長さだけで決まる）",
              ok3 is False and not put3 and sec3 == 0.128, f"{(ok3, sec3, put3)}")
        Config.SILENCE_LIMIT = 0.5
        ok4, sec4, put4 = run_burst(4, preroll_chunks=0)
        check("最短発話: pre-roll が空でも判定は変わらない",
              ok4 is False and not put4 and sec4 == 0.128, f"{(ok4, sec4, put4)}")
    finally:
        Config.MIN_SPEECH_SEC, Config.SILENCE_LIMIT, Config.VAD_ONSET_SEC = saved
        audio_logger.removeHandler(grab)
        audio_logger.setLevel(saved_level)
        audio_logger.propagate = True
        logging.disable(saved_disable)

    # フックが例外を投げてもマイク監視は止まらない
    boom = _SegmentTracker(rate=16000, chunk=512, put=lambda b: None,
                           on_segment_start=lambda b: 1 / 0, on_segment_audio=lambda b: 1 / 0,
                           on_segment_end=lambda ok, sec: 1 / 0)
    ok = True
    try:
        t = 0.0
        for p in [0.9] * 5 + [0.1] * 40:
            boom.step(ch, p, 0.2 if p > 0.5 else 0.01, t)
            t += 0.032
    except Exception:
        ok = False
    check("区間: フックの例外はマイク監視に持ち込まない", ok)


# ---------------------------------------------------------------------------
# OpenAILiveStt
# ---------------------------------------------------------------------------

async def t_basic() -> None:
    srv = FakeServer()
    stt = mk(srv)
    pcm = pcm16(1.0)
    stream_utt(stt, pcm)
    stt.commit()
    text = await stt.transcribe(pcm)
    exp = expected_bytes(pcm)
    check("基本: 開始→送信→commit→確定文字", text == f"{exp}バイト", f"{text} (期待 {exp})")
    c = srv.conns[0]
    su = c.sent[0]
    fmt = su["session"]["audio"]["input"]["format"]
    tr = su["session"]["audio"]["input"]["transcription"]
    check("基本: session.update は 24kHz・区切りは EVE 側（turn_detection=None）",
          su["type"] == "session.update" and fmt == {"type": "audio/pcm", "rate": API_RATE}
          and su["session"]["audio"]["input"]["turn_detection"] is None
          and su["session"]["type"] == "transcription")
    check("基本: モデル・言語・delay", tr.get("model") == Config.STT_LIVE_MODEL
          and tr.get("languages") == ["ja"] and tr.get("delay") == Config.STT_LIVE_DELAY, f"{tr}")
    commits = [m for m in c.sent if m["type"] == "input_audio_buffer.commit"]
    check("基本: commit に event_id（error の突き合わせ用）", commits and commits[0].get("event_id"))
    # 2発話目は同じ接続を使い回す
    stream_utt(stt, pcm16(0.5))
    stt.commit()
    t2 = await stt.transcribe(b"")
    check("基本: 2発話目も同じ接続（接続は1本）", len(srv.conns) == 1 and t2.endswith("バイト"))
    await stt.close()
    check("基本: close で接続を閉じる", c.closed)


async def t_clear() -> None:
    srv = FakeServer()
    stt = mk(srv)
    stream_utt(stt, pcm16(0.5))
    await asyncio.sleep(0.05)  # 送信を進める
    stt.clear()
    await asyncio.sleep(0.05)
    c = srv.conns[0]
    check("破棄: clear を送る", any(m["type"] == "input_audio_buffer.clear" for m in c.sent))
    check("破棄: 待ち札を作らない", not stt._pending)
    pcm = pcm16(0.8)
    stream_utt(stt, pcm)
    stt.commit()
    text = await stt.transcribe(pcm)
    check("破棄: 次の発話は前の音声が混ざらない", text == f"{expected_bytes(pcm)}バイト", text)
    # 送る前に破棄した発話は何も送らない
    n_before = len(c.sent)
    stt.start(pcm16(0.2))
    stt.clear()
    await asyncio.sleep(0.05)
    check("破棄: 送る前に捨てた発話は append も clear も送らない", len(c.sent) == n_before,
          f"{[m['type'] for m in c.sent[n_before:]]}")
    await stt.close()


async def t_disconnect() -> None:
    srv = FakeServer()
    stt = mk(srv)
    await stt.warmup()
    check("切断: warmup で接続だけ済ませる（音声は送らない）",
          len(srv.conns) == 1 and all(m["type"] == "session.update" for m in srv.conns[0].sent))
    # 途中で切れる → その発話は捨てる
    stream_utt(stt, pcm16(0.5))
    await asyncio.sleep(0.05)
    srv.conns[0].drop()
    await asyncio.sleep(0.05)
    stt.feed(pcm16(0.2))
    stt.commit()
    t1 = await stt.transcribe(b"")
    check("切断: 話している途中で切れた発話は捨てる（空文字）", t1 == "", t1)
    # 次の発話で張り直す
    pcm = pcm16(0.6)
    stream_utt(stt, pcm)
    stt.commit()
    t2 = await stt.transcribe(pcm)
    check("切断: 次の発話に向けて張り直す", len(srv.conns) == 2 and t2 == f"{expected_bytes(pcm)}バイト",
          f"conns={len(srv.conns)} {t2}")
    await stt.close()


async def t_idle_drop() -> None:
    srv = FakeServer()
    stt = mk(srv)
    await stt.warmup()
    srv.conns[0].drop()  # 黙っている間に切れる（寿命・回線）
    await asyncio.sleep(0.05)
    pcm = pcm16(0.6)
    stream_utt(stt, pcm)
    stt.commit()
    text = await stt.transcribe(pcm)
    check("切断: 黙っている間に切れても次の発話は通る", text == f"{expected_bytes(pcm)}バイト", text)
    await stt.close()


async def t_timeout() -> None:
    srv = FakeServer(reply=False)
    stt = mk(srv, commit_timeout_sec=0.2)
    stream_utt(stt, pcm16(0.5))
    stt.commit()
    t0 = time.monotonic()
    text = await stt.transcribe(b"")
    dt = time.monotonic() - t0
    check("打ち切り: 確定が来なければ空文字", text == "", text)
    check("打ち切り: 待つのは打ち切り時間まで", dt < 1.0, f"{dt:.2f}s")
    check("打ち切り: 接続を張り直す（今の接続を閉じる）", srv.conns[0].closed)
    srv.reply = True
    pcm = pcm16(0.6)
    stream_utt(stt, pcm)
    stt.commit()
    t2 = await stt.transcribe(pcm)
    check("打ち切り: 次の発話は新しい接続で通る", len(srv.conns) == 2 and t2 == f"{expected_bytes(pcm)}バイト",
          f"{len(srv.conns)} {t2}")
    await stt.close()


async def t_commit_error() -> None:
    srv = FakeServer(commit_error_at={1})
    stt = mk(srv)
    stream_utt(stt, pcm16(0.5))
    stt.commit()
    pcm = pcm16(0.7)
    stream_utt(stt, pcm)
    stt.commit()
    t1 = await stt.transcribe(b"")
    t2 = await stt.transcribe(b"")
    check("拒否: commit を拒否された発話は空文字", t1 == "", t1)
    check("拒否: 次の発話の対応づけはずれない", t2 == f"{expected_bytes(pcm)}バイト", t2)
    await stt.close()


async def t_order() -> None:
    srv = FakeServer()
    srv.completed_reversed = True
    stt = mk(srv)
    a, b = pcm16(0.5), pcm16(0.9)
    stream_utt(stt, a)
    stt.commit()
    stream_utt(stt, b)
    stt.commit()
    ta = await stt.transcribe(b"")
    tb = await stt.transcribe(b"")
    check("順序: 確定が逆順に届いても item_id で正しく対応づける",
          ta == f"{expected_bytes(a)}バイト" and tb == f"{expected_bytes(b)}バイト", f"{ta} {tb}")
    await stt.close()


async def t_session_reject() -> None:
    srv = FakeServer(reject_session=True)
    stt = mk(srv)
    stream_utt(stt, pcm16(0.5))
    stt.commit()
    text = await stt.transcribe(b"")
    check("設定拒否: その発話は空文字（自動で一括方式に切り替えない）", text == "", text)
    check("設定拒否: 接続は閉じる", srv.conns and srv.conns[0].closed)
    await stt.close()


async def t_connect_fail() -> None:
    srv = FakeServer(fail_connect=1)
    stt = mk(srv)
    stream_utt(stt, pcm16(0.5))
    stt.commit()
    t1 = await stt.transcribe(b"")
    pcm = pcm16(0.6)
    stream_utt(stt, pcm)
    stt.commit()
    t2 = await stt.transcribe(pcm)
    check("接続失敗: その発話は捨て、次の発話で張り直す",
          t1 == "" and t2 == f"{expected_bytes(pcm)}バイト", f"{t1!r} {t2!r}")
    await stt.close()


async def t_direct() -> None:
    srv = FakeServer()
    stt = mk(srv)
    pcm = pcm16(2.5)
    text = await stt.transcribe(pcm)
    check("一括: 待ち札が無い呼び出しは音声を丸ごと送って確定させる",
          text == f"{expected_bytes(pcm)}バイト", text)
    appends = [m for m in srv.conns[0].sent if m["type"] == "input_audio_buffer.append"]
    biggest = max(len(base64.b64decode(m["audio"])) for m in appends)
    check("一括: 1通は1秒ぶんまでに分ける", biggest <= API_RATE * 2, f"{biggest}")
    check("一括: 空の音声は送らず空文字", await stt.transcribe(b"") == "")
    await stt.close()


async def t_renew() -> None:
    srv = FakeServer(expires_in=100.0)  # 寿命の残りが張り直しの閾値(300秒)を切っている
    stt = mk(srv)
    await stt.warmup()
    pcm = pcm16(0.5)
    stream_utt(stt, pcm)
    stt.commit()
    text = await stt.transcribe(pcm)
    check("寿命: 空いている時に張り直してから送る",
          len(srv.conns) == 2 and srv.conns[0].closed and text == f"{expected_bytes(pcm)}バイト",
          f"{len(srv.conns)} {text}")
    await stt.close()


async def t_reset() -> None:
    srv = FakeServer(reply=False)
    stt = mk(srv)
    stream_utt(stt, pcm16(0.5))
    stt.commit()
    stream_utt(stt, pcm16(0.3))
    stt.reset()
    check("reset: 開いている発話と待ち札を捨てる", stt._cur is None and not stt._pending)
    srv.reply = True
    await stt.close()


async def t_short() -> None:
    srv = FakeServer()
    stt = mk(srv)
    stt.start(pcm16(0.05))  # 100ms 未満
    stt.commit()
    text = await stt.transcribe(b"")
    check("短すぎ: 100ms 未満は commit せず空文字（サーバに拒否される形を送らない）",
          text == "" and srv.commits == 0, f"{text!r} commits={srv.commits}")
    await stt.close()


# ---------------------------------------------------------------------------
# MicSttInputSource の配線
# ---------------------------------------------------------------------------

class _FakeAI:
    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()

    async def get_audio(self) -> bytes:
        return await self.q.get()

    def stop(self) -> None:
        pass


async def t_input_source() -> None:
    from eve.response.input_source import MicSttInputSource

    srv = FakeServer()
    stt = mk(srv)
    q = StimulusQueue()
    log: list = []
    src = MicSttInputSource(
        q, stt,
        on_utterance=lambda: log.append("utterance"),
        on_stt_start=lambda: log.append("stt_start"),
        on_stt_end=lambda: log.append("stt_end"),
        on_speech_end=lambda sec, seg: log.append(("speech_end", sec)),
        on_stt_result=lambda put: log.append(("result", put)),
    )
    check("配線: ストリーミング backend を認識", src._streaming is True)
    ai = _FakeAI()
    src._ai = ai
    src._task = asyncio.create_task(src._consume())
    pcm = pcm16(1.0)
    src._segment_start(pcm[:1024])
    for i in range(1024, len(pcm), 1024):
        src._segment_audio(pcm[i:i + 1024])
    src._segment_end(True, 0.9)
    ai.q.put_nowait(pcm)
    s = await asyncio.wait_for(q.get(), 2.0)
    await asyncio.sleep(0.01)
    check("配線: 区間フック→STT→USER_UTTERANCE", s.kind == StimulusKind.USER_UTTERANCE
          and s.payload == f"{expected_bytes(pcm)}バイト", f"{s.payload}")
    check("配線: 発話終了の確定で on_speech_end(声の長さ)", ("speech_end", 0.9) in log, f"{log}")
    check("配線: 刺激にしたら on_stt_result(True) を on_stt_end の前に",
          ("result", True) in log and log.index(("result", True)) < log.index("stt_end"), f"{log}")
    # 破棄された区間: clear・on_speech_end は呼ばない
    log.clear()
    src._segment_start(pcm16(0.2))
    src._segment_end(False, 0.1)
    check("配線: 破棄した区間は on_speech_end を呼ばない", not any(
        isinstance(e, tuple) and e[0] == "speech_end" for e in log), f"{log}")
    check("配線: 破棄した区間は待ち札を作らない", not stt._pending)
    src._task.cancel()
    await stt.close()

    # 空認識 → on_stt_result(False)
    class _EmptyStt:
        streaming = False

        async def transcribe(self, pcm):
            return ""

    q2 = StimulusQueue()
    log2: list = []
    src2 = MicSttInputSource(q2, _EmptyStt(), on_stt_result=lambda put: log2.append(put),
                             on_stt_end=lambda: log2.append("end"))
    ai2 = _FakeAI()
    src2._ai = ai2
    src2._task = asyncio.create_task(src2._consume())
    ai2.q.put_nowait(b"\x00" * 3200)
    await asyncio.sleep(0.05)
    check("配線: 空認識は on_stt_result(False)", log2 == [False, "end"], f"{log2}")
    src2._task.cancel()


def t_factory() -> None:
    check("factory: openai_live", isinstance(make_stt("openai_live"), OpenAILiveStt))
    check("factory: 既定は openai_live（ユーザ裁定 2026-09-25）", Config.STT_BACKEND == "openai_live"
          or os.getenv("STT_BACKEND") is not None)
    s = OpenAILiveStt(api_key="x")
    check("factory: コンストラクタは loop の外でも作れる（接続しない）", s._conn is None and s._sender is None)


async def main() -> None:
    t_resample()
    t_tracker()
    t_factory()
    for fn in (t_basic, t_clear, t_disconnect, t_idle_drop, t_timeout, t_commit_error, t_order,
               t_session_reject, t_connect_fail, t_direct, t_renew, t_reset, t_short, t_input_source):
        try:
            await asyncio.wait_for(fn(), 10.0)
        except Exception as e:
            check(f"{fn.__name__} が例外なく終わる", False, f"{type(e).__name__}: {e}")


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
