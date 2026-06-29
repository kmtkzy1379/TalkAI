r"""TaskAgent inc2 実践通しテスト（VOICEVOX ユーザ音声→STT→実 pipeline・実LLM gpt-5.5）。

応答LLM は delegate_task で自然文ゴールを委譲するだけ。賢い task LLM が境界つきループで完遂し
**最終結果を1つ**返す。これを実機同等で観察:
 A1 2手ゴール : 「PCとイブの状態を両方調べてまとめて」→ agent が pc_status+self_status→要約1つ。
 A2 失敗ゴール : 「外部サービスが使えるか調べてまとめて」(flaky)→ agent が試行→正直に失敗報告。
 A3 併用       : 予約(create_task)＋委譲(delegate_task)を同時→干渉なく両方完了。

⚙exec = registry.execute（delegate_task=応答LLM / pc_status等=agent ループ内）。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_agent_test.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402

Config.CALLFUNCTION_ENABLED = True
Config.TASK_ENABLED = True

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.memory import ConversationCache  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher, parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.speech import SpeechState  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.task import ReconcileTimer, TaskAgent, TaskExecutor, TaskStore, register_task_capabilities  # noqa: E402

T0 = time.monotonic()
LOG = []
CUR = ["init"]


def ev(tag, d):
    LOG.append((time.monotonic() - T0, CUR[0], tag, str(d)))


def synth(text, speaker=8, rate=16000):
    import requests
    q = requests.post(f"{Config.VOICEVOX_URL}/audio_query", params={"text": text, "speaker": speaker}, timeout=10).json()
    q["outputSamplingRate"] = rate; q["outputStereoToMono"] = True
    return requests.post(f"{Config.VOICEVOX_URL}/synthesis", json=q, params={"speaker": speaker}, timeout=20).content


def wav_pcm(b):
    with wave.open(io.BytesIO(b), "rb") as wf:
        return wf.readframes(wf.getnframes())


class CapturePlayer:
    async def play_fn(self, audio, should_stop=None):
        if not audio:
            return
        for _ in range(max(1, len(audio) // 12000)):
            if should_stop and should_stop():
                return
            await asyncio.sleep(0.02)


async def _tts(s):
    return b"x" * 4000


async def main():
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5", "task": "openai/gpt-5.5"})
    art = tempfile.mkdtemp(prefix="task_agent_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    state = SpeechState(); queue = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art, "t.jsonl"))
    caps = CapabilityRegistry(is_busy=lambda: runner.is_busy(), qsize=lambda: queue.qsize())
    register_task_capabilities(caps, store)

    def flaky(a):
        raise ConnectionError("外部サービスに接続できませんでした")
    caps.register(Capability("external_service_status", "外部サービスの稼働状況を確認する。引数なし。", {}, flaky))

    _exec = caps.execute

    def exec_wrap(name, args=None):
        out = _exec(name, args)
        if name not in ("delegate_task", "create_task", "list_tasks", "cancel_task"):
            ev("AGENT-TOOL", f"{name} -> {str(out)[:60]}")
        return out
    caps.execute = exec_wrap  # type: ignore

    dispatcher = FunctionDispatcher(registry=caps, queue=queue)
    _submit = dispatcher.submit

    def submit_wrap(tcs):
        ev("TOOL", [f"{parse_tool_call(t)[0]}{parse_tool_call(t)[1]}" for t in tcs])
        _submit(tcs)
    dispatcher.submit = submit_wrap  # type: ignore

    agent = TaskAgent(registry=caps, model_registry=reg, store=store,
                      max_steps=Config.TASK_AGENT_MAX_STEPS, timeout_sec=Config.TASK_AGENT_TIMEOUT_SEC)
    executor = TaskExecutor(store=store, registry=caps, queue=queue, agent=agent)
    scheduler = ReconcileTimer(store=store, executor=executor, tick_sec=1.0)
    player = CapturePlayer(); audio = AudioPlayQueue(play_fn=player.play_fn); stt = make_stt()

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        if tools:
            async for x in reg.stream_with_tools("response", messages, tools=tools, tool_sink=tool_sink, max_tokens=220):
                yield x
        else:
            async for x in reg.stream("response", messages, max_tokens=220):
                yield x

    orch = ResponseOrchestrator(audio, stream_fn, _tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, dispatcher=dispatcher,
                                on_response_complete=lambda: state.mark_eve_activity())
    _handle = orch.handle

    async def handle_wrap(stim):
        tag = "EVE→報告" if stim.kind == StimulusKind.CALLFUNCTION_RESULT else "EVE"
        await _handle(stim)
        ev(tag, (orch.last_response or "（空）")[:90])
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    play_task = asyncio.create_task(audio.play_worker()); run_task = asyncio.create_task(runner.run())
    await store.initialize(); dispatcher.start(); executor.start(); scheduler.start()
    state.mark_eve_activity(); await stt.warmup()
    ev("INFO", "稼働")

    async def wait_idle(timeout=45.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0 and executor.is_idle():
                await asyncio.sleep(0.2)
                if not runner.is_busy() and queue.qsize() == 0 and executor.is_idle():
                    return
            await asyncio.sleep(0.1)

    async def says(line):
        wav = await asyncio.to_thread(synth, line)
        heard = await stt.transcribe(wav_pcm(wav)) or line
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev("USER", heard)
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        await wait_idle()

    def tasks():
        return [(t.goal or t.what, t.status) for t in store.list_all()[-3:]]

    async def a1():
        await says("ねえイブ、PCとイブ自身の状態を両方調べて、まとめて教えてくれる？")
        await asyncio.sleep(2.0); ev("TASKS", tasks())

    async def a2():
        await says("外部サービスが今使えるか調べて、状況をまとめて教えて。")
        await asyncio.sleep(2.0); ev("TASKS", tasks())

    async def a3():
        await says("10秒後にPCの状態を教えて。それと、イブの調子も調べてまとめておいて。")
        await asyncio.sleep(12.0); ev("TASKS", tasks())

    plan = [("A1_2手ゴール", a1, 3), ("A2_失敗ゴール", a2, 2), ("A3_予約と委譲併用", a3, 1)]
    for name, fn, n in plan:
        for i in range(1, n + 1):
            CUR[0] = f"{name}#{i}"; ev("PHASE", name)
            try:
                await fn()
            except Exception as e:
                ev("ERROR", f"{type(e).__name__}: {e}")
            await wait_idle()

    CUR[0] = "end"
    await scheduler.stop(); await executor.stop(); await dispatcher.stop()
    run_task.cancel(); play_task.cancel(); await asyncio.sleep(0.2)

    print("\n========== 時系列（T+秒） ==========")
    icon = {"USER": "🧑", "EVE": "🤖", "EVE→報告": "🛠報告", "TOOL": "⚙委譲", "AGENT-TOOL": "  ↳agent", "TASKS": "📋", "PHASE": "═", "INFO": "…", "ERROR": "❌"}
    last = None
    for t, sc, tg, d in sorted(LOG, key=lambda x: x[0]):
        if tg == "PHASE":
            if sc != last:
                print(f"\n  ═══ {sc} ═══"); last = sc
        else:
            print(f"  {t:6.1f} [{sc:14}] {icon.get(tg, tg)} {d}")
    await cache.shutdown(); await store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
