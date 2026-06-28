r"""Call-Function 実機パイプラインテスト（VOICEVOX ユーザ音声→STT→実パイプライン・実LLM）。

Tier-1/scenarios では見えない**実 PipelineRunner + dispatcher + queue + orchestrator** の統合を検証:
 S1 単発     : 「今の調子は？」→ self_status 呼出→結果再投入→Eve 報告（end-to-end）。
 S2 マルチ   : 「時刻とシステム両方」→ 2 tool_call → 2 結果 → Eve が**何回**喋るか（drain 順）。
 S3 RAG干渉  : 記憶を引く質問（tool 無し）が Call-Function 有効下でも壊れず統合応答するか。
 S4 画面干渉 : 疑似 vision を入れた状態で質問→ tool/画面/RAG が混ざっても破綻しないか。
 S5 barge-in : tool を呼ばせた直後に別話題で割り込み→新文脈に切替できるか・結果到着で破綻しないか。

応答=gpt-4o（実機既定）。VOICEVOX 起動必須。スクショ無し（vision は疑似注入）。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\callfunction_pipeline.py
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

Config.CALLFUNCTION_ENABLED = True  # 本テストで有効化

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.feedback import FeedbackLLM, FeedbackWorker, PredictionState  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.response.tts import VoicevoxTTS  # noqa: E402
from eve.speech import SpeechState  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.vlm import VisionState  # noqa: E402

OVERRIDES = {"response": "openai/gpt-4o", "feedback": "openai/gpt-4o-mini"}
T0 = time.monotonic()
LOG: list[tuple[float, str, str]] = []


def ev(tag, detail):
    LOG.append((time.monotonic() - T0, tag, detail))


def sh(s, n=160):
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


SEED = [("ユーザはチーズケーキが好きで、カフェを探していた", "チーズケーキ 甘いもの カフェ", 60)]


async def main():
    reg = ModelRegistry(overrides=OVERRIDES)
    art = tempfile.mkdtemp(prefix="cf_pipe_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(art, "r.jsonl")); await rag.warmup()
    for d, k, pd in SEED:
        await rag.add_chunk(text=d, search_text=k, prediction_diff=pd)
    pred = PredictionState()
    vision = VisionState(ring_max=Config.VLM_RING_MAX)
    state = SpeechState()
    queue = StimulusQueue()

    caps = CapabilityRegistry(is_busy=lambda: runner.is_busy(), qsize=lambda: queue.qsize())

    def flaky(a):
        raise ConnectionError("外部サービスに接続できませんでした")

    caps.register(Capability("external_service_status", "外部サービスの稼働状況を確認する。引数なし。", {}, flaky))
    dispatcher = FunctionDispatcher(registry=caps, queue=queue)
    _submit = dispatcher.submit

    def submit_wrap(tcs):
        from eve.response.function_dispatcher import parse_tool_call
        ev("SUBMIT", f"tool_calls={[parse_tool_call(t)[0] for t in tcs]}")
        _submit(tcs)
    dispatcher.submit = submit_wrap  # type: ignore

    feedback = FeedbackLLM(reg, rag_store=rag, prediction_state=pred)
    fworker = FeedbackWorker(feedback, cache)
    player = CapturePlayer(); audio = AudioPlayQueue(play_fn=player.play_fn)
    tts = VoicevoxTTS(); stt = make_stt()

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        if tools:
            async for d in reg.stream_with_tools("response", messages, tools=tools, tool_sink=tool_sink, max_tokens=300):
                yield d
        else:
            async for d in reg.stream("response", messages, max_tokens=300):
                yield d

    def on_complete():
        fworker.trigger(); state.mark_eve_activity()

    orch = ResponseOrchestrator(audio, stream_fn, tts.generate, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, rag_store=rag, prediction_state=pred,
                                on_response_complete=on_complete, vision_state=vision, dispatcher=dispatcher)
    _handle = orch.handle

    async def handle_wrap(stim):
        tag = {StimulusKind.USER_UTTERANCE: "EVE→user", StimulusKind.CALLFUNCTION_RESULT: "EVE→cfresult"}.get(stim.kind, str(stim.kind))
        await _handle(stim)
        ev(tag, sh(orch.last_response))
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    play_task = asyncio.create_task(audio.play_worker())
    run_task = asyncio.create_task(runner.run())
    fworker.start(); state.mark_eve_activity(); dispatcher.start()
    await stt.warmup()
    ev("INFO", "パイプライン稼働（Call-Function 有効）")

    async def wait_idle(timeout=30.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0:
                await asyncio.sleep(0.3)  # 背景 dispatch→再投入の猶予
                if not runner.is_busy() and queue.qsize() == 0:
                    return
            await asyncio.sleep(0.1)

    async def says(line, barge=False):
        wav = await asyncio.to_thread(synth, line)
        heard = await stt.transcribe(wav_pcm(wav)) or line
        if barge:
            audio.interrupt(); runner.interrupt(); queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)
            state.mark_user_speech_start()
            ev("BARGE", "割り込み発火")
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev("USER", sh(heard))
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        await wait_idle()

    ev("PHASE", "S1 単発")
    await says("ねえイブ、今の調子とキューの状況どう？")
    ev("PHASE", "S2 マルチツール")
    await says("時刻とシステムの調子を両方教えて。")
    ev("PHASE", "S3 RAG干渉")
    await says("前に甘いものの話したよね、私が好きって言ってたやつ覚えてる？")
    ev("PHASE", "S4 画面干渉（疑似vision）")
    vision.set_latest("画面に Steam の Stardew Valley のストアページが表示されている")
    await says("今の画面に映ってるの、なんてゲームかわかる？")
    ev("PHASE", "S5 barge-in / cancel")
    await says("外部サービスの稼働状況を確認して。")
    await asyncio.sleep(0.4)
    await says("あ、やっぱいいや。それより今日の天気どう思う？", barge=True)

    await asyncio.sleep(1.5)
    run_task.cancel(); play_task.cancel()
    await dispatcher.stop(); await fworker.stop()
    await asyncio.sleep(0.2)

    print("\n========== 時系列ログ（T+秒） ==========")
    icon = {"USER": "🧑USER ", "EVE→user": "🤖EVE  ", "EVE→cfresult": "🛠EVE(結果)", "SUBMIT": "  ⚙submit", "BARGE": "  ✋barge", "PHASE": "═══", "INFO": "  …"}
    for t, tag, detail in sorted(LOG, key=lambda x: x[0]):
        if tag == "PHASE":
            print(f"\n  ═══ {t:6.1f} {detail} ═══")
        else:
            print(f"  {t:6.1f} {icon.get(tag, tag)} {detail}")
    await rag.shutdown(); await cache.shutdown(); await tts.close()
    print(f"\nartifacts: {art}")


if __name__ == "__main__":
    asyncio.run(main())
