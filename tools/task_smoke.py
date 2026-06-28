r"""タスク管理 inc1 実機スモーク（実 pipeline・実LLM・要 .env・コスト小）。

検証: 応答LLM が「N秒後に〜」で **create_task を自律的に呼ぶ** → store に Pending → ReconcileTimer
（~1s）が when 到来を検知 → Executor が能力実行 → CALLFUNCTION_RESULT 再投入 → Eve が報告、まで
実 PipelineRunner+dispatcher+executor+scheduler+queue で通す。response=gpt-5.5（前置き＋tool 安定）。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402

Config.CALLFUNCTION_ENABLED = True
Config.TASK_ENABLED = True

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher, parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.task import ReconcileTimer, TaskExecutor, TaskStore, register_task_capabilities  # noqa: E402

T0 = time.monotonic()
LOG = []


def ev(tag, d):
    LOG.append((time.monotonic() - T0, tag, str(d)))


class CapturePlayer:
    async def play_fn(self, audio, should_stop=None):
        if not audio:
            return
        for _ in range(max(1, len(audio) // 12000)):
            if should_stop and should_stop():
                return
            await asyncio.sleep(0.02)


async def _tts(s):
    return b"x"


async def main():
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5"})
    queue = StimulusQueue()
    store = TaskStore(task_file=os.path.join(tempfile.mkdtemp(prefix="task_smoke_"), "t.jsonl"))
    caps = CapabilityRegistry(is_busy=lambda: runner.is_busy(), qsize=lambda: queue.qsize())
    register_task_capabilities(caps, store)
    dispatcher = FunctionDispatcher(registry=caps, queue=queue)
    _submit = dispatcher.submit

    def submit_wrap(tcs):
        ev("TOOL", [parse_tool_call(t)[0] for t in tcs])
        _submit(tcs)
    dispatcher.submit = submit_wrap  # type: ignore

    executor = TaskExecutor(store=store, registry=caps, queue=queue)
    scheduler = ReconcileTimer(store=store, executor=executor, tick_sec=1.0)
    player = CapturePlayer(); audio = AudioPlayQueue(play_fn=player.play_fn)

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        if tools:
            async for d in reg.stream_with_tools("response", messages, tools=tools, tool_sink=tool_sink, max_tokens=200):
                yield d
        else:
            async for d in reg.stream("response", messages, max_tokens=200):
                yield d

    orch = ResponseOrchestrator(audio, stream_fn, _tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                dispatcher=dispatcher)
    _handle = orch.handle

    async def handle_wrap(stim):
        tag = "EVE→cf" if stim.kind == StimulusKind.CALLFUNCTION_RESULT else "EVE"
        await _handle(stim)
        ev(tag, orch.last_response)
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    play_task = asyncio.create_task(audio.play_worker())
    run_task = asyncio.create_task(runner.run())
    await store.initialize(); dispatcher.start(); executor.start(); scheduler.start()
    ev("INFO", "パイプライン稼働（Call-Function + Task）")

    async def wait_idle(timeout=25.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0:
                await asyncio.sleep(0.2)
                if not runner.is_busy() and queue.qsize() == 0:
                    return
            await asyncio.sleep(0.1)

    async def says(line):
        ev("USER", line)
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, line))
        await wait_idle()

    await says("ねえイブ、3秒後に今の調子とキューの状況を教えて。")
    ev("TASKS", [(t.what, t.status, t.when is not None) for t in store.list_all()])
    await asyncio.sleep(5.0)  # scheduler(~1s)+executor が発火するのを待つ
    ev("TASKS-after", [(t.what, t.status) for t in store.list_all()])

    await says("あと、5秒後に『そろそろ休憩しよう』って言ってね。")
    await asyncio.sleep(7.0)
    ev("TASKS-final", [(t.what, t.status) for t in store.list_all()])

    scheduler and await scheduler.stop()
    await executor.stop()
    run_task.cancel(); play_task.cancel()
    await dispatcher.stop(); await store.shutdown()
    await asyncio.sleep(0.2)

    print("\n========== 時系列ログ（T+秒） ==========")
    icon = {"USER": "🧑USER ", "EVE": "🤖EVE  ", "EVE→cf": "🛠EVE(報告)", "TOOL": "  ⚙tool", "TASKS": "  📋", "TASKS-after": "  📋", "TASKS-final": "  📋", "INFO": "  …"}
    for t, tag, d in sorted(LOG, key=lambda x: x[0]):
        print(f"  {t:6.1f} {icon.get(tag, tag)} {d[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
