r"""inc2 修正 実機リプレイ（VOICEVOX→STT→実 pipeline・gpt-5.5・RAG/画面 並行・全イベント時刻付き記録）。

ユーザの事故を再現し、直ったかを見る:
 D1 remind修正 : 「N秒後に今の時刻を教えて」→ delegate→発火時 agent が pc_status→**実時刻**（remind echo でない）。
 D2 dedup      : 同じ予約を混乱気味に再依頼→**発火は1回だけ**。
 D3 取消1件    : 予約1件→「やっぱいいや」→ コード即時キャンセル＋**名前つき**報告。
 D4 取消ファジー : 2件予約→「さっきの時間のやつキャンセル」→ タスクLLM が言い換えを照合してその1件を止める。
 D5 併用       : RAG想起＋画面＋予約＋待機中の無関係雑談が自然に共存。

⚙委譲 = 応答LLM の tool（delegate_task/cancel_task）。↳agent = agent の実行能力。🛠報告 = CALLFUNCTION_RESULT。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_dedup_test.py
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
from eve.feedback import FeedbackLLM, FeedbackWorker, PredictionState  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher, parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.speech import SpeechState  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.task import (CancelResolver, ReconcileTimer, TaskAgent, TaskExecutor,  # noqa: E402
                      TaskStore, register_task_capabilities)
from eve.vlm import VisionState  # noqa: E402

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
    art = tempfile.mkdtemp(prefix="task_dedup_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(art, "r.jsonl")); await rag.warmup()
    await rag.add_chunk(text="ユーザはチーズケーキが大好きで甘いカフェを探している", search_text="チーズケーキ 甘いもの", prediction_diff=60)
    prediction = PredictionState(); fworker = FeedbackWorker(FeedbackLLM(reg, rag_store=rag, prediction_state=prediction), cache)
    vision = VisionState(ring_max=Config.VLM_RING_MAX)
    state = SpeechState(); queue = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art, "t.jsonl"))
    caps = CapabilityRegistry(is_busy=lambda: runner.is_busy(), qsize=lambda: queue.qsize())
    cancel_resolver = CancelResolver(store=store, model_registry=reg, queue=queue)
    register_task_capabilities(caps, store, cancel_resolver=cancel_resolver)

    _exec = caps.execute

    def exec_wrap(name, args=None):
        out = _exec(name, args)
        if name in ("self_status", "pc_status", "list_tasks", "external_service_status"):
            ev("AGENT-TOOL", f"{name} -> {str(out)[:55]}")
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
            async for x in reg.stream_with_tools("response", messages, tools=tools, tool_sink=tool_sink, max_tokens=230):
                yield x
        else:
            async for x in reg.stream("response", messages, max_tokens=230):
                yield x

    orch = ResponseOrchestrator(audio, stream_fn, _tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, rag_store=rag, prediction_state=prediction,
                                on_response_complete=lambda: (fworker.trigger(), state.mark_eve_activity()),
                                vision_state=vision, dispatcher=dispatcher)
    _handle = orch.handle

    async def handle_wrap(stim):
        if stim.kind == StimulusKind.CALLFUNCTION_RESULT:
            fn = getattr(stim.payload, "function_name", "")
            tag = "取消報告" if fn == "cancel_task" else "報告"
        else:
            tag = "EVE"
        await _handle(stim)
        ev(tag, (orch.last_response or "（空）")[:95])
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    play_task = asyncio.create_task(audio.play_worker()); run_task = asyncio.create_task(runner.run())
    await store.initialize(); dispatcher.start(); executor.start(); scheduler.start()
    cancel_resolver.start(); fworker.start()
    state.mark_eve_activity(); await stt.warmup()
    ev("INFO", "稼働")

    async def wait_idle(timeout=40.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0 and executor.is_idle() and cancel_resolver._inbox.empty():
                await asyncio.sleep(0.25)
                if not runner.is_busy() and queue.qsize() == 0 and executor.is_idle() and cancel_resolver._inbox.empty():
                    return
            await asyncio.sleep(0.1)

    async def hear(line):
        wav = await asyncio.to_thread(synth, line)
        return await stt.transcribe(wav_pcm(wav)) or line

    async def enqueue(heard, label="USER"):
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev(label, heard)
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))

    async def says(line):
        await enqueue(await hear(line)); await wait_idle()

    def tasks():
        return [(((t.goal or t.what)[:14]), t.status) for t in store.list_all()[-4:]]

    # ---------- シナリオ ----------
    async def d1_remind_fix():
        await says("ねえイブ、10秒後に今の時刻を教えてね。")
        await asyncio.sleep(12.0); await wait_idle(); ev("TASKS", tasks())

    async def d2_dedup():
        await enqueue(await hear("10秒後に今の調子を教えてね。")); await wait_idle()
        await enqueue(await hear("やっぱり10秒後に今の調子を教えて。")); await wait_idle()  # 再依頼→dedup
        ev("TASKS", tasks())
        await asyncio.sleep(12.0); await wait_idle(); ev("TASKS", tasks())

    async def d3_cancel_one():
        await says("15秒後に今の時刻を教えてね。")
        await asyncio.sleep(1.0)
        await says("あ、やっぱり今のいいや。")  # 1件→コード即時キャンセル＋名前
        ev("TASKS", tasks())

    async def d4_cancel_fuzzy():
        await says("15秒後に今の時刻を教えてね。")
        await says("それと、15秒後にPCの状態も教えてね。")  # 2件（別 goal）
        await asyncio.sleep(1.0)
        await says("さっきの時間のやつだけキャンセルして。")  # 言い換え→タスクLLM が照合
        ev("TASKS", tasks())

    async def d5_coexist():
        vision.set_latest("画面に Steam の Stardew Valley のストアページが開いている")
        await says("前に甘いものが好きって言ってたよね。あと10秒後に今の調子も教えて。")
        await asyncio.sleep(2.0)
        await says("ところで豚キムチいいよね。")  # 待機中の無関係雑談
        await asyncio.sleep(11.0); await wait_idle(); ev("TASKS", tasks())

    plan = [("D1_remind修正", d1_remind_fix, 2), ("D2_dedup", d2_dedup, 1),
            ("D3_取消1件", d3_cancel_one, 2), ("D4_取消ファジー", d4_cancel_fuzzy, 2), ("D5_併用", d5_coexist, 1)]
    for name, fn, n in plan:
        for i in range(1, n + 1):
            CUR[0] = f"{name}#{i}"; ev("PHASE", name)
            try:
                await fn()
            except Exception as e:
                ev("ERROR", f"{type(e).__name__}: {e}")
            await wait_idle()

    CUR[0] = "end"
    await cancel_resolver.stop(); await scheduler.stop(); await executor.stop()
    await dispatcher.stop(); await fworker.stop()
    run_task.cancel(); play_task.cancel(); await asyncio.sleep(0.2)

    print("\n========== 全ログ 時系列（T+秒） ==========")
    icon = {"USER": "🧑", "EVE": "🤖", "報告": "🛠報告", "取消報告": "✋取消", "TOOL": "⚙委譲",
            "AGENT-TOOL": "  ↳agent", "TASKS": "📋", "PHASE": "═", "INFO": "…", "ERROR": "❌"}
    last = None
    for t, sc, tg, d in sorted(LOG, key=lambda x: x[0]):
        if tg == "PHASE":
            if sc != last:
                print(f"\n  ═══ {sc} ═══"); last = sc
        else:
            print(f"  {t:6.1f} [{sc:14}] {icon.get(tg, tg)} {d}")
    await rag.shutdown(); await cache.shutdown(); await store.shutdown()
    print(f"\nartifacts: {art}")


if __name__ == "__main__":
    asyncio.run(main())
