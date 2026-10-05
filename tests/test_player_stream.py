"""出力を開きっぱなしにする再生器と、区切りの間（API・音声デバイス不要の決定論テスト）。

ユーザ 2026-09-26「開きっぱなしがいい」: 以前は項目ごとに出力を開いて閉じており、開くのに約0.2秒・出力の遅れ
約0.09秒かかって、毎回1回目の音が遅れ・区切りの間も空いていた。試聴（listen/clause/）で良しとされた音は、
この開き直しの間（約0.29秒）込みなので、開きっぱなしにした分の間は AudioPlayQueue が作る（SPEECH_GAP_SEC）。

  [1] 再生器: 同じレートなら開き直さない・レートが変われば開き直す・書けなくなったら閉じて次で開き直す・
      ミュート中は開かない・最後に鳴らし終えた時刻を持つ
  [2] 区切りの間: 直前の音が終わってから item_gap_sec 空けて次の項目・間を待っている間の割り込みでは
      次の項目を鳴らさない・直前の音が十分前なら待たない・既定 0 は待たない
  [3] 配線

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_player_stream.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

from eve.clock import now_mono  # noqa: E402
from eve.filler import audio as fa  # noqa: E402
from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402
from eve.response.player import RealAudioPlayer  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}" + (f"  ({detail})" if detail else ""))


class FakeStream:
    def __init__(self, rate: int, fail_after: int = -1) -> None:
        self.rate = rate
        self.writes = 0
        self.closed = False
        self.fail_after = fail_after

    def write(self, buf) -> None:
        if self.fail_after >= 0 and self.writes >= self.fail_after:
            raise OSError("device lost")
        self.writes += 1

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakePA:
    def __init__(self) -> None:
        self.opened: list = []
        self.fail_next_after = -1

    def open(self, format, channels, rate, output):
        st = FakeStream(rate, self.fail_next_after)
        self.fail_next_after = -1
        self.opened.append(st)
        return st

    def terminate(self) -> None:
        pass


def wav(sec: float, sr: int = 48000) -> bytes:
    return fa.to_wav(np.zeros(int(sec * sr), dtype=np.float32), sr)


def t_player() -> None:
    print("\n[1] 再生器（出力を開きっぱなし）")

    async def run() -> None:
        p = RealAudioPlayer()
        pa = FakePA()
        p._pa = pa
        check("まだ鳴らしていなければ最後の時刻は 0", p.last_end == 0.0)
        await p.play_fn(wav(0.05))
        await p.play_fn(wav(0.05))
        check("同じレートなら1回だけ開く（開きっぱなし）", len(pa.opened) == 1 and not pa.opened[0].closed,
              f"{len(pa.opened)}")
        check("鳴らし終えた時刻を持つ（区切りの間の基準）", 0 < now_mono() - p.last_end < 1.0)
        await p.play_fn(wav(0.05, 24000))
        check("レートが変われば開き直す（前の出力は閉じる）",
              len(pa.opened) == 2 and pa.opened[0].closed and pa.opened[1].rate == 24000)
        pa.fail_next_after = 0
        p._close_stream()
        failed = False
        try:
            await p.play_fn(wav(0.05))
        except OSError:
            failed = True
        check("書けなくなったら例外を上げ（キューがその項目を飛ばす）、出力を閉じる",
              failed and p._stream is None and pa.opened[-1].closed)
        await p.play_fn(wav(0.05))
        check("次の項目で開き直して鳴らす", p._stream is pa.opened[-1] and pa.opened[-1].writes > 0)
        n = len(pa.opened)
        p.muted = True
        await p.play_fn(wav(0.05))
        check("ミュート中は開かない・書かない", len(pa.opened) == n)
        p.muted = False
        p.prepare(48000)
        check("起動時に開いておける（prepare）", p._stream is not None and p._rate == 48000)
        p.close()
        check("終了時に閉じる", p._stream is None and p._pa is None)

    asyncio.run(run())


def t_gap() -> None:
    print("\n[2] 区切りの間")

    async def run() -> None:
        starts: list = []
        state = {"end": 0.0}

        async def play(audio):
            starts.append((audio, now_mono()))
            await asyncio.sleep(0.05)
            state["end"] = now_mono()

        q = AudioPlayQueue(play_fn=play)
        q.item_gap_sec = 0.2
        q.last_audio_end = lambda: state["end"]
        w = asyncio.create_task(q.play_worker())
        g = q.current_generation()
        t0 = now_mono()
        for x in ("節1", "節2", "節3"):
            q.enqueue(g, q.reserve_seq(g), x)
        await asyncio.sleep(0.8)
        names = [a for a, _ in starts]
        gaps = [round(starts[i + 1][1] - starts[i][1] - 0.05, 3) for i in range(len(starts) - 1)]
        check("順に鳴る", names == ["節1", "節2", "節3"], f"{names}")
        check("1つ目は待たない（直前の音が無い）", starts[0][1] - t0 < 0.05, f"{starts[0][1] - t0:.3f}")
        check("節と節の間は item_gap_sec（0.2秒）空く", all(0.18 <= x <= 0.3 for x in gaps), f"{gaps}")

        # 直前の音（フィラー等）が十分前に終わっていれば待たない
        state["end"] = now_mono() - 5.0
        starts.clear()
        t1 = now_mono()
        q.enqueue(g, q.reserve_seq(g), "次の応答")
        await asyncio.sleep(0.1)
        check("直前の音が十分前なら待たない", starts and starts[0][1] - t1 < 0.05)

        # 間を待っている間に割り込まれたら、次の項目は鳴らさない（取り出す前に待つ）
        starts.clear()
        await asyncio.sleep(0.1)
        state["end"] = now_mono()  # たった今、音が終わった
        g = q.current_generation()
        q.enqueue(g, q.reserve_seq(g), "間の後の節")
        await asyncio.sleep(0.05)  # 間を待っている最中
        q.interrupt()
        await asyncio.sleep(0.4)
        check("間を待っている間の割り込み: その節は鳴らさない", starts == [], f"{starts}")
        w.cancel()

        # 既定（0）は待たない＝既存の挙動は不変
        starts.clear()
        q2 = AudioPlayQueue(play_fn=play)
        q2.last_audio_end = lambda: now_mono()
        w2 = asyncio.create_task(q2.play_worker())
        t2 = now_mono()
        q2.enqueue(0, q2.reserve_seq(0), "a")
        q2.enqueue(0, q2.reserve_seq(0), "b")
        await asyncio.sleep(0.2)
        check("既定 0 は待たない", len(starts) == 2 and starts[1][1] - t2 < 0.12, f"{starts}")
        w2.cancel()

    asyncio.run(run())


def t_wiring() -> None:
    print("\n[3] 配線")
    import inspect

    from eve.config import Config
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    check("区切りの間は SPEECH_GAP_SEC（既定 0.29＝試聴で良しとされた値）",
          vl.audio.item_gap_sec == Config.SPEECH_GAP_SEC and abs(Config.SPEECH_GAP_SEC - 0.29) < 1e-9,
          f"{vl.audio.item_gap_sec}")
    vl.player.last_end = 123.0
    check("直前の音の終わりは再生器から読む（応答・フィラー・笑いを通して）", vl.audio.last_audio_end() == 123.0)
    check("フィラーも同じ再生器で鳴らす", vl.filler is None or vl.filler._play == vl.player.play_fn)
    src = inspect.getsource(VoiceLoop.warmup)
    check("起動時に出力を開いておく（ウォームアップ）", "self.player.prepare" in src)
    check("出力のレート: Irodori 48kHz / VOICEVOX 24kHz", VoiceLoop._output_rate() in (48000, 24000))


t_player()
t_gap()
t_wiring()
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
