r"""inc2 実践総合テスト（VOICEVOX ユーザ音声→STT→実 pipeline・実LLM・RAG/画面/自律発話も並行）。

ユーザ指定の確認項目:
 S1 2手ゴール   : 「PCとイブの状態を両方調べてまとめて」→ Eve「調べるね」だけ→裏で agent→一言要約1つ。
 S2 予約+雑談+割込: 「5秒後に状態を教えて」→ 待機中に雑談→5秒で割り込んで報告。
 S3 検索なし否定 : 「今日の天気は？」→ 検索手段が無いので**正直に否定**（ハルシネしない）。
 S4 ソフトキャンセル: 「PCとイブを調べて！」→（タイミング多種で）「やっぱりキャンセル！」→ 結果破棄・無報告。
 S5 RAG+画面+全ツール: 記憶想起＋疑似画面コメント＋委譲＋list_tasks を統合できるか。
 S6 自律発話     : 沈黙→自律的に一言（他機能が壊れてないか）。
全ツール: delegate_task / create_task / cancel_task / list_tasks（応答LLM）＋ self_status/pc_status（agent）。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_full_test.py
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
from eve.speech import SilenceMonitor, SpeechDecider, SpeechState, make_decide_fn  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.task import RUNNING, ReconcileTimer, TaskAgent, TaskExecutor, TaskStore, register_task_capabilities  # noqa: E402
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


SEED = [("ユーザはチーズケーキが大好きで、よく甘いカフェを探している", "チーズケーキ 甘いもの カフェ", 60)]


async def main():
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5", "task": "openai/gpt-5.5"})
    art = tempfile.mkdtemp(prefix="task_full_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(art, "r.jsonl")); await rag.warmup()
    for d, k, pd in SEED:
        await rag.add_chunk(text=d, search_text=k, prediction_diff=pd)
    prediction = PredictionState()
    feedback = FeedbackLLM(reg, rag_store=rag, prediction_state=prediction)
    fworker = FeedbackWorker(feedback, cache)
    vision = VisionState(ring_max=Config.VLM_RING_MAX)
    state = SpeechState(); queue = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art, "t.jsonl"))
    caps = CapabilityRegistry(is_busy=lambda: runner.is_busy(), qsize=lambda: queue.qsize())
    register_task_capabilities(caps, store)

    _exec = caps.execute

    def exec_wrap(name, args=None):
        out = _exec(name, args)
        if name in ("self_status", "pc_status", "external_service_status"):
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

    def on_complete():
        fworker.trigger(); state.mark_eve_activity()
    orch = ResponseOrchestrator(audio, stream_fn, _tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, rag_store=rag, prediction_state=prediction,
                                on_response_complete=on_complete, vision_state=vision, dispatcher=dispatcher)
    _handle = orch.handle

    async def handle_wrap(stim):
        tag = {StimulusKind.USER_UTTERANCE: "EVE", StimulusKind.CALLFUNCTION_RESULT: "EVE→報告",
               StimulusKind.AUTONOMOUS_SPEECH: "EVE自発"}.get(stim.kind, "EVE?")
        await _handle(stim)
        ev(tag, (orch.last_response or "（空）")[:95])
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    decider = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=prediction,
                            queue=queue, decide_fn=make_decide_fn(reg), vision_state=vision)
    silence = SilenceMonitor(state=state, decider=decider, is_busy_fn=runner.is_busy,
                             tick_sec=1.0, threshold_sec=4.0)
    play_task = asyncio.create_task(audio.play_worker()); run_task = asyncio.create_task(runner.run())
    await store.initialize(); dispatcher.start(); executor.start(); scheduler.start()
    fworker.start(); decider.start()
    state.mark_eve_activity(); await stt.warmup()
    ev("INFO", "稼働")

    def barge():
        audio.interrupt(); runner.interrupt()
        state.mark_user_speech_start(); queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)

    async def wait_idle(timeout=60.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0 and executor.is_idle():
                await asyncio.sleep(0.25)
                if not runner.is_busy() and queue.qsize() == 0 and executor.is_idle():
                    return
            await asyncio.sleep(0.1)

    async def hear(line):
        wav = await asyncio.to_thread(synth, line)
        return await stt.transcribe(wav_pcm(wav)) or line

    async def enqueue_user(heard, label="USER"):
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev(label, heard)
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))

    async def says(line):
        await enqueue_user(await hear(line))
        await wait_idle()

    def tasks():
        return [(((t.goal or t.what)[:16]), t.status) for t in store.list_all()[-3:]]

    # ---------- シナリオ ----------
    async def s1():
        await says("ねえイブ、PCとイブ自身の状態を両方調べて、まとめて教えてくれる？")
        ev("TASKS", tasks())

    async def s2():
        await enqueue_user(await hear("5秒後に今の調子を教えてね。"))
        await wait_idle()
        await asyncio.sleep(1.5)
        await enqueue_user(await hear("ところで今日はいい天気だね。"))  # 待機中に雑談
        await asyncio.sleep(6.0)
        await wait_idle()
        ev("TASKS", tasks())

    async def s3():
        await says("ねえイブ、今日の天気はどう？")
        ev("TASKS", tasks())

    async def s4(extra_delay):
        cancel_heard = await hear("あ、やっぱり今のキャンセルして！")  # 事前合成
        await enqueue_user(await hear("PCとイブの状態を今すぐ調べて教えて！"))
        # 委譲→agent が走り出す(goal が Running)のを待ってから取消（ループ内ソフトキャンセルを狙う）
        t = time.monotonic(); started = False
        while time.monotonic() - t < 14:
            if any(x.goal and x.status == RUNNING for x in store.list_all()):
                started = True; break
            await asyncio.sleep(0.1)
        ev("INFO", f"agent開始={started} → +{extra_delay}s で取消")
        await asyncio.sleep(extra_delay)
        barge()
        await enqueue_user(cancel_heard, label="USER取消")
        await wait_idle()
        ev("TASKS", tasks())

    async def s5():
        vision.set_latest("画面に Steam の Stardew Valley のストアページが開いている")
        await says("前に甘いものが好きって言ってたよね。あと、PCの状態も調べておいて。今の画面も見える？")
        await says("今、予約してるタスクって何かある？")
        ev("TASKS", tasks())

    async def s6():
        # 沈黙させて自律発話が出るか（他機能が壊れてないか）。
        silence.start()
        ev("INFO", "沈黙開始（自律発話待ち）")
        state.mark_eve_activity()  # ここから沈黙計測
        await asyncio.sleep(11.0)
        await silence.stop()
        ev("TASKS", tasks())

    plan = [("S1_2手ゴール", s1, [None, None]),
            ("S2_予約雑談割込", s2, [None, None]),
            ("S3_検索なし否定", s3, [None, None]),
            ("S4_ソフトキャンセル", s4, [0.3, 1.5, 3.0]),
            ("S5_RAG画面全ツール", s5, [None, None]),
            ("S6_自律発話", s6, [None])]
    for name, fn, variants in plan:
        for i, v in enumerate(variants, 1):
            CUR[0] = f"{name}#{i}"; ev("PHASE", name)
            try:
                await (fn(v) if v is not None else fn())
            except Exception as e:
                ev("ERROR", f"{type(e).__name__}: {e}")
            await wait_idle()

    CUR[0] = "end"
    await silence.stop(); await decider.stop(); await scheduler.stop(); await executor.stop()
    await dispatcher.stop(); await fworker.stop()
    run_task.cancel(); play_task.cancel(); await asyncio.sleep(0.2)

    print("\n========== 全ログ 時系列（T+秒） ==========")
    icon = {"USER": "🧑", "USER取消": "🧑✋", "EVE": "🤖", "EVE→報告": "🛠報告", "EVE自発": "💭自発",
            "TOOL": "⚙委譲", "AGENT-TOOL": "  ↳agent", "TASKS": "📋", "PHASE": "═", "INFO": "…", "ERROR": "❌", "EVE?": "🤖?"}
    last = None
    for t, sc, tg, d in sorted(LOG, key=lambda x: x[0]):
        if tg == "PHASE":
            if sc != last:
                print(f"\n  ═══ {sc} ═══"); last = sc
        else:
            print(f"  {t:6.1f} [{sc:16}] {icon.get(tg, tg)} {d}")
    await rag.shutdown(); await cache.shutdown(); await store.shutdown()
    print(f"\nartifacts: {art}")


if __name__ == "__main__":
    asyncio.run(main())
