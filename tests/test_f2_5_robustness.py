"""F2.5 Stage0 堅牢化テスト（API 不要・決定論）。

A1: TTS が None/例外でも以降の文が止まらない（head-of-line デッドロック解消）。
A2: PipelineRunner は1刺激の失敗で死なず継続。連続失敗は循環ブレーカで停止。
A3: stream 中の例外が handle() を貫通しない（途中まででクリーンに打ち切り）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f2_5_robustness.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 意図的に起こすエラーログ（traceback）でテスト出力が汚れるのを抑制
logging.disable(logging.CRITICAL)

from eve.pipeline import (  # noqa: E402
    AudioPlayQueue,
    PipelineRunner,
    Stimulus,
    StimulusKind,
    StimulusQueue,
)
from eve.response import ResponseOrchestrator  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


def _recorder():
    played: list = []

    async def play_fn(a):
        played.append(a)

    return played, play_fn


async def _run_orch(stream_fn, tts_fn):
    played, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())
    orch = ResponseOrchestrator(audio, stream_fn, tts_fn)
    raised = False
    try:
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    except Exception:
        raised = True
    await audio.join()
    worker.cancel()
    return played, orch, raised


# ---------- A1: TTS None/例外でも以降が止まらない ----------
async def t_a1_first_none() -> bool:
    async def stream(messages):
        for c in ["A。", "B。", "C。"]:
            yield c

    async def tts(s):
        return None if s == "A。" else f"[{s}]"  # 1文目だけ失敗

    played, _, _ = await _run_orch(stream, tts)
    return played == ["[B。]", "[C。]"]  # A はスキップ、B/C は再生


async def t_a1_mid_raise() -> bool:
    async def stream(messages):
        for c in ["A。", "B。", "C。"]:
            yield c

    async def tts(s):
        if s == "B。":
            raise RuntimeError("tts boom")  # 中間で例外
        return f"[{s}]"

    played, _, _ = await _run_orch(stream, tts)
    return played == ["[A。]", "[C。]"]  # B はスキップ、A/C は再生


# ---------- A3: stream 例外が handle を貫通しない ----------
async def t_a3_stream_raises() -> bool:
    async def stream(messages):
        yield "A。"
        raise RuntimeError("stream boom")  # 途中で stream が落ちる

    async def tts(s):
        return f"[{s}]"

    played, orch, raised = await _run_orch(stream, tts)
    return (not raised) and played == ["[A。]"] and orch.last_response == "A。"


# ---------- A2: ランナーの堅牢化 ----------
class _FlakyOrch:
    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.calls = 0
        self.handled = 0

    async def handle(self, stimulus) -> None:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("handler boom")
        self.handled += 1


async def t_a2_recovers() -> bool:
    q = StimulusQueue()
    _, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    orch = _FlakyOrch(fail_first=1)  # 1回失敗→以降成功
    runner = PipelineRunner(q, orch, audio)
    task = asyncio.create_task(runner.run())
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "a"))
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "b"))
    for _ in range(50):
        await asyncio.sleep(0)
        if orch.handled >= 1:
            break
    await asyncio.sleep(0.02)
    task.cancel()
    return orch.calls == 2 and orch.handled == 1  # 失敗1回後に2件目を処理


async def t_a2_breaker_stops() -> bool:
    q = StimulusQueue()
    _, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    orch = _FlakyOrch(fail_first=999)  # 常に失敗
    runner = PipelineRunner(q, orch, audio, max_consecutive_errors=3)
    for i in range(5):
        await q.put(Stimulus(StimulusKind.USER_UTTERANCE, str(i)))
    stopped = False
    try:
        await asyncio.wait_for(runner.run(), timeout=1.0)
    except RuntimeError:
        stopped = True
    except asyncio.TimeoutError:
        stopped = False
    return stopped and orch.calls == 3  # 3連続失敗で停止


async def main() -> None:
    check("A1 1文目TTS None でも以降再生", await t_a1_first_none())
    check("A1 中間TTS例外でも以降再生", await t_a1_mid_raise())
    check("A3 stream例外が handle を貫通しない", await t_a3_stream_raises())
    check("A2 1回失敗後に次刺激を処理", await t_a2_recovers())
    check("A2 連続失敗で循環ブレーカ停止", await t_a2_breaker_stops())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
