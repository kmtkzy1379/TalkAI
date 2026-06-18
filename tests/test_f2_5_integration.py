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


async def main() -> None:
    check("フルループ: USER→runner→実orchestrator→順次再生", await t_full_loop())
    check("連続2ターンを順に処理", await t_two_turns())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
