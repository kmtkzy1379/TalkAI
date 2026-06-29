r"""タスク管理 inc1 実践通しテスト（VOICEVOX ユーザ音声→STT→実 pipeline・実LLM・各3回）。

シナリオ（各3回計測・まぐれ判定）:
 S1 基本     : 「3秒後に状態を教えて」→ create_task→発火→報告。
 S2 失敗     : 「3秒後に外部サービスの状態を確認して」(flaky)→ 失敗→正直報告。
 S3 実行中会話: 予約後、待っている間に別の話題で話しかけ→両方さばけるか。
 S4 報告中割込: タスク報告が始まった瞬間に barge-in→破綻せず別話題へ切替か。
 S5 RAG+画面 : 記憶質問+疑似vision+予約タスクを同時に→干渉せず統合応答か。

response=gpt-5.5。Eve 音声は鳴らさず CapturePlayer で消化（barge は LLM stream 中の is_busy を狙う）。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_pipeline_test.py
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
from eve.task import ReconcileTimer, TaskExecutor, TaskStore, register_task_capabilities  # noqa: E402
from eve.vlm import VisionState  # noqa: E402

T0 = time.monotonic()
LOG: list[tuple[float, str, str, str]] = []  # (t, scenario, tag, detail)
CUR = ["init"]


def ev(tag, detail):
    LOG.append((time.monotonic() - T0, CUR[0], tag, str(detail)))


def sh(s, n=150):
    return (str(s) if s is not None else "—")[:n].replace("\n", " / ")


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


SEED = [("ユーザはチーズケーキが好きで、カフェを探していた", "チーズケーキ 甘いもの カフェ", 60)]


async def main():
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5", "feedback": "openai/gpt-4o-mini"})
    art = tempfile.mkdtemp(prefix="task_pipe_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(art, "r.jsonl")); await rag.warmup()
    for d, k, pd in SEED:
        await rag.add_chunk(text=d, search_text=k, prediction_diff=pd)
    pred = PredictionState()
    vision = VisionState(ring_max=Config.VLM_RING_MAX)
    state = SpeechState()
    queue = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art, "t.jsonl"))
    caps = CapabilityRegistry(is_busy=lambda: runner.is_busy(), qsize=lambda: queue.qsize())
    register_task_capabilities(caps, store)

    def flaky(a):
        raise ConnectionError("外部サービスに接続できませんでした")
    caps.register(Capability("external_service_status", "外部サービスの稼働状況を確認する。引数なし。", {}, flaky))

    dispatcher = FunctionDispatcher(registry=caps, queue=queue)
    _submit = dispatcher.submit

    def submit_wrap(tcs):
        ev("TOOL", [f"{parse_tool_call(t)[0]}{parse_tool_call(t)[1]}" for t in tcs])
        _submit(tcs)
    dispatcher.submit = submit_wrap  # type: ignore

    executor = TaskExecutor(store=store, registry=caps, queue=queue)
    scheduler = ReconcileTimer(store=store, executor=executor, tick_sec=1.0)
    feedback = FeedbackLLM(reg, rag_store=rag, prediction_state=pred); fworker = FeedbackWorker(feedback, cache)
    player = CapturePlayer(); audio = AudioPlayQueue(play_fn=player.play_fn)
    tts = _tts; stt = make_stt()

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        if tools:
            async for x in reg.stream_with_tools("response", messages, tools=tools, tool_sink=tool_sink, max_tokens=180):
                yield x
        else:
            async for x in reg.stream("response", messages, max_tokens=180):
                yield x

    def on_complete():
        fworker.trigger(); state.mark_eve_activity()
    orch = ResponseOrchestrator(audio, stream_fn, tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, rag_store=rag, prediction_state=pred,
                                on_response_complete=on_complete, vision_state=vision, dispatcher=dispatcher)
    _handle = orch.handle

    async def handle_wrap(stim):
        tag = {StimulusKind.USER_UTTERANCE: "EVE", StimulusKind.CALLFUNCTION_RESULT: "EVE→報告"}.get(stim.kind, "EVE?")
        await _handle(stim)
        ev(tag, sh(orch.last_response))
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    play_task = asyncio.create_task(audio.play_worker()); run_task = asyncio.create_task(runner.run())
    await store.initialize(); dispatcher.start(); executor.start(); scheduler.start(); fworker.start()
    state.mark_eve_activity()
    await stt.warmup()
    ev("INFO", "稼働")

    async def hear(line):
        wav = await asyncio.to_thread(synth, line)
        return await stt.transcribe(wav_pcm(wav)) or line

    async def wait_idle(timeout=25.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0:
                await asyncio.sleep(0.2)
                if not runner.is_busy() and queue.qsize() == 0:
                    return
            await asyncio.sleep(0.08)

    async def says(line, prehear=None):
        heard = prehear or await hear(line)
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev("USER", sh(heard))
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        await wait_idle()

    def task_states():
        return [(t.what, t.status) for t in store.list_all()]

    # ---- シナリオ ----
    async def s1():
        await says("ねえイブ、3秒後に今の調子を教えて。")
        await asyncio.sleep(5.0)
        ev("TASKS", task_states())

    async def s2():
        await says("3秒後に外部サービスの稼働状況を確認して教えて。")
        await asyncio.sleep(5.0)
        ev("TASKS", task_states())

    async def s3():
        await says("5秒後に今の調子を教えてね。")
        chat = await hear("ところで、今日はちょっと寒いね。")  # 待っている間に別の話題
        await asyncio.sleep(1.5)
        await says(None, prehear=chat)  # 実行待ちの間に話しかける
        await asyncio.sleep(5.0)
        ev("TASKS", task_states())

    async def s4():
        barge = await hear("あ、やっぱり今はいいや。それより今日の天気どう思う？")  # 事前合成
        await says("3秒後に今の調子を教えて。")
        # タスク報告が始まる(is_busy True)瞬間を狙って割り込み
        t = time.monotonic(); did = False
        while time.monotonic() - t < 8:
            if runner.is_busy():
                audio.interrupt(); runner.interrupt(); queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)
                ev("BARGE", "報告中に割り込み")
                state.mark_user_speech_start(); state.mark_user_utterance()
                ev("USER", sh(barge))
                await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, barge))
                did = True; break
            await asyncio.sleep(0.03)
        if not did:
            ev("INFO", "報告 busy を捉えられず（割込なし）")
        await wait_idle()
        ev("TASKS", task_states())

    async def s5():
        vision.set_latest("画面に Steam の Stardew Valley のストアページが表示されている")
        await says("前に甘いもの好きって言ってたよね。あと3秒後に今の調子も教えて。")
        await asyncio.sleep(5.0)
        ev("TASKS", task_states())

    scenarios = [("S1基本", s1), ("S2失敗", s2), ("S3実行中会話", s3), ("S4報告中割込", s4), ("S5RAG+画面", s5)]
    for name, fn in scenarios:
        for i in range(1, 4):
            CUR[0] = f"{name}#{i}"
            ev("PHASE", name)
            try:
                await fn()
            except Exception as e:
                ev("ERROR", f"{type(e).__name__}: {e}")
            await wait_idle()

    CUR[0] = "end"
    await scheduler.stop(); await executor.stop(); await dispatcher.stop(); await fworker.stop()
    run_task.cancel(); play_task.cancel(); await asyncio.sleep(0.2)

    print("\n========== 時系列（T+秒 / シナリオ / 事象） ==========")
    icon = {"USER": "🧑", "EVE": "🤖", "EVE→報告": "🛠報告", "TOOL": "⚙", "TASKS": "📋", "BARGE": "✋", "PHASE": "═", "INFO": "…", "ERROR": "❌", "EVE?": "🤖?"}
    last = None
    for t, sc, tag, d in sorted(LOG, key=lambda x: x[0]):
        if sc != last and tag == "PHASE":
            print(f"\n  ═══ {sc} ═══"); last = sc
        elif tag != "PHASE":
            print(f"  {t:6.1f} [{sc:12}] {icon.get(tag, tag)} {d}")
    await rag.shutdown(); await cache.shutdown(); await store.shutdown()
    print(f"\nartifacts: {art}")


if __name__ == "__main__":
    asyncio.run(main())
