r"""タスクのキャンセル徹底調査（VOICEVOX ユーザ音声→STT→実 pipeline・実LLM）。

音声だと「キャンセル」が届くまで STT+応答LLM の遅延がある。その遅延を実測し、
 C1 待機キャンセル : 長め(15s)の予約を待機中(~5s)に取消 → 発火せず Cancelled になるか・遅延は何秒か。
 C2 ギリギリ取消   : 短め(8s)の予約を発火直前(~6.5s)に取消 → 間に合うか/報告がすり抜けるか・その時の挙動。
各3回。30秒/1分でも遅延は一定なので「発火の何秒前までに言えば間に合うか」を示す。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_cancel_test.py
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

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.memory import ConversationCache  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher, parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.speech import SpeechState  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.task import ReconcileTimer, TaskExecutor, TaskStore, register_task_capabilities  # noqa: E402

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
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5"})
    art = tempfile.mkdtemp(prefix="task_cancel_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    state = SpeechState(); queue = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art, "t.jsonl"))
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
    stt = make_stt()

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        if tools:
            async for x in reg.stream_with_tools("response", messages, tools=tools, tool_sink=tool_sink, max_tokens=200):
                yield x
        else:
            async for x in reg.stream("response", messages, max_tokens=200):
                yield x

    orch = ResponseOrchestrator(audio, stream_fn, _tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, dispatcher=dispatcher,
                                on_response_complete=lambda: state.mark_eve_activity())
    _handle = orch.handle

    async def handle_wrap(stim):
        tag = "EVE→報告" if stim.kind == StimulusKind.CALLFUNCTION_RESULT else "EVE"
        await _handle(stim)
        ev(tag, (orch.last_response or "（空）")[:80])
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    play_task = asyncio.create_task(audio.play_worker()); run_task = asyncio.create_task(runner.run())
    await store.initialize(); dispatcher.start(); executor.start(); scheduler.start()
    state.mark_eve_activity(); await stt.warmup()
    ev("INFO", "稼働")

    async def wait_idle(timeout=30.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0:
                await asyncio.sleep(0.15)
                if not runner.is_busy() and queue.qsize() == 0:
                    return
            await asyncio.sleep(0.08)

    async def says(line, label="USER"):
        spoke = time.monotonic() - T0  # 「話し始め」相当（synth 前）
        wav = await asyncio.to_thread(synth, line)
        heard = await stt.transcribe(wav_pcm(wav)) or line
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev(label, f"(発話開始 T+{spoke:.1f}s) {heard}")
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        return spoke

    def latest_status():
        ts = store.list_all()
        return ts[-1].status if ts else None

    async def run_cancel(task_secs, cancel_at, label):
        # 予約
        await says(f"{task_secs}秒後に今の調子を教えてね。")
        await wait_idle()
        sched_done = time.monotonic() - T0
        # cancel_at 秒まで待って取消発話
        await asyncio.sleep(max(0, cancel_at - (time.monotonic() - T0 - sched_done)))
        spoke = await says("あ、やっぱりさっきの予約キャンセルして。", label="CANCEL発話")
        await wait_idle()
        # 取消が landed した時刻（cancel_task tool）
        cancel_tool_t = next((t for t, sc, tg, d in LOG if sc == CUR[0] and tg == "TOOL" and "cancel" in d), None)
        if cancel_tool_t is not None:
            ev("計測", f"取消遅延 = {cancel_tool_t - spoke:.1f}s（発話→cancel_task）")
        # 予約時刻まで見届ける
        await asyncio.sleep(task_secs + 3)
        ev("最終状態", [(t.what, t.status) for t in store.list_all()[-2:]])

    # C1 待機キャンセル（15s 予約・5s で取消）
    for i in range(1, int(os.getenv("C1", "3")) + 1):
        CUR[0] = f"C1待機#{i}"; ev("PHASE", "C1")
        await run_cancel(15, 5.0, "CANCEL発話"); await wait_idle()
    # C2 ギリギリ（8s 予約・6.5s で取消）
    for i in range(1, int(os.getenv("C2", "3")) + 1):
        CUR[0] = f"C2ギリギリ#{i}"; ev("PHASE", "C2")
        await run_cancel(8, 6.5, "CANCEL発話"); await wait_idle()

    CUR[0] = "end"
    await scheduler.stop(); await executor.stop(); await dispatcher.stop()
    run_task.cancel(); play_task.cancel(); await asyncio.sleep(0.2)

    print("\n========== 時系列（T+秒） ==========")
    icon = {"USER": "🧑予約", "CANCEL発話": "🧑取消", "EVE": "🤖", "EVE→報告": "🛠報告", "TOOL": "⚙", "計測": "⏱", "最終状態": "📋", "PHASE": "═", "INFO": "…"}
    last = None
    for t, sc, tg, d in sorted(LOG, key=lambda x: x[0]):
        if tg == "PHASE":
            if sc != last:
                print(f"\n  ═══ {sc} ═══"); last = sc
        else:
            print(f"  {t:6.1f} [{sc:11}] {icon.get(tg, tg)} {d}")
    await cache.shutdown(); await store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
