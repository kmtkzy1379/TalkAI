"""F2.5 統合の決定論テスト（API不要）: StimulusQueue→PipelineRunner→実ResponseOrchestrator
→AudioPlayQueue のフルループ結線を検証（監査 H7=フルループ未テスト を閉じる）。

実 mic/LLM/TTS は使わず stream_fn/tts_fn/play_fn を注入。
実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f2_5_integration.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


async def _drain_until(pred, limit: int = 400) -> None:
    for _ in range(limit):
        await asyncio.sleep(0)
        if pred():
            return
    await asyncio.sleep(0.05)


async def t_full_loop() -> bool:
    """USER 刺激 → runner → 実 orchestrator(fake stream/tts) → 再生（順序保持）。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    q = StimulusQueue()

    async def stream_fn(messages):
        for c in ["はい。", "元気です。"]:
            yield c

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn)
    runner = PipelineRunner(q, orch, audio)
    worker = asyncio.create_task(audio.play_worker())
    runner_task = asyncio.create_task(runner.run())

    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    await _drain_until(lambda: runner.processed >= 1)  # handle 完了=last_response確定+全enqueue
    await audio.join()  # 全再生完了まで待つ（playback と last_response の race を回避）

    runner_task.cancel()
    worker.cancel()
    return played == ["[はい。]", "[元気です。]"] and orch.last_response == "はい。元気です。"


async def t_two_turns() -> bool:
    """連続2ターンが順に処理される。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    q = StimulusQueue()

    async def stream_fn(messages):
        yield "おう。"

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn)
    runner = PipelineRunner(q, orch, audio)
    worker = asyncio.create_task(audio.play_worker())
    runner_task = asyncio.create_task(runner.run())

    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "1つ目"))
    await _drain_until(lambda: runner.processed >= 1)
    await audio.join()
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "2つ目"))
    await _drain_until(lambda: runner.processed >= 2)
    await audio.join()

    runner_task.cancel()
    worker.cancel()
    return runner.processed == 2 and played == ["[おう。]", "[おう。]"]


async def t_coalesce() -> bool:
    """入力前に溜まった USER 発話は1つに結合され、応答は1回（pileup防止）。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    q = StimulusQueue()
    seen: list = []

    async def stream_fn(messages):
        seen.append(messages[-1]["content"])  # user 内容
        yield "了解。"

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn)
    runner = PipelineRunner(q, orch, audio)
    # 2発話を drain 前に両方 queue へ入れてから runner を起動（確実に coalesce 対象に）
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "今日は"))
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "いい天気"))
    worker = asyncio.create_task(audio.play_worker())
    runner_task = asyncio.create_task(runner.run())

    await _drain_until(lambda: runner.processed >= 1)
    await audio.join()
    runner_task.cancel()
    worker.cancel()

    merged_ok = any(("今日は" in m and "いい天気" in m) for m in seen)
    return runner.processed == 1 and merged_ok and played == ["[了解。]"]


async def t_barge_in_cancel() -> bool:
    """barge-in（interrupt）で進行中の応答生成が即キャンセルされる（続きが出ない）。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    q = StimulusQueue()
    started = asyncio.Event()

    async def stream_fn(messages):
        started.set()
        yield "えーと。"
        await asyncio.sleep(5)  # 長い生成 → barge-in でキャンセルされるべき
        yield "つづき。"

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn)
    runner = PipelineRunner(q, orch, audio)
    worker = asyncio.create_task(audio.play_worker())
    runner_task = asyncio.create_task(runner.run())

    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "ねえ"))
    await started.wait()
    await asyncio.sleep(0)
    runner.interrupt()  # barge-in
    await _drain_until(lambda: runner.processed >= 1)
    await asyncio.sleep(0)
    runner_task.cancel()
    worker.cancel()

    return runner.processed == 1 and "[つづき。]" not in played


async def main() -> None:
    check("フルループ: USER→runner→実orchestrator→順次再生", await t_full_loop())
    check("連続2ターンを順に処理", await t_two_turns())
    check("coalesce: 溜まったUSERを1応答に結合", await t_coalesce())
    check("barge-in: interrupt で進行中応答を即キャンセル", await t_barge_in_cancel())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
