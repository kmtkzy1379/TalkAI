r"""F5 自然さ評価ハーネス（複数モデルの組み合わせで応答/フィードバック/発話判定を比較）。

実パイプライン（mic の代わりに VOICEVOX 合成ユーザ音声→STT）に、挨拶/沈黙→自発発話/
驚き/割り込み/穏やかな間 を含むシナリオを流し、各プリセット（response・feedback・
speech_decide の3役にモデルを割当）で会話の流れを時系列で出力する。自然さを目視判定する用。

== 起動方法 ==
  事前: VOICEVOX 起動 + .env に OPENAI_API_KEY / GEMINI_API_KEY。
  PowerShell:
    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\f5_model_eval.py            # 既定セット
    ...python tools\f5_model_eval.py gpt5.5                # 1プリセットだけ
    ...python tools\f5_model_eval.py gpt4o,gemini          # 複数指定（カンマ区切り）
    ...python tools\f5_model_eval.py all                   # 全プリセット
  実 LLM 課金あり。1プリセット ~1-2分（gpt-5.5 は遅め）。
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.ERROR)

from eve.config import Config  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.feedback import FeedbackLLM, FeedbackWorker, PredictionState  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.response.tts import VoicevoxTTS  # noqa: E402
from eve.speech import SilenceMonitor, SpeechDecider, SpeechState, make_decide_fn  # noqa: E402
from eve.stt import make_stt  # noqa: E402

USER_SPEAKER = 8

# プリセット: 3役(response/feedback/speech_decide)へのモデル割当。
PRESETS: dict[str, dict[str, str]] = {
    "gpt4o": {"response": "openai/gpt-4o", "feedback": "openai/gpt-4o-mini", "speech_decide": "openai/gpt-4o-mini"},
    "gpt5.4": {"response": "openai/gpt-5.4", "feedback": "openai/gpt-5.4-mini", "speech_decide": "openai/gpt-5.4-mini"},
    "gpt5.5": {"response": "openai/gpt-5.5", "feedback": "openai/gpt-5.4", "speech_decide": "openai/gpt-5.4-mini"},
    "gemini": {"response": "gemini/gemini-2.5-flash", "feedback": "gemini/gemini-2.5-flash", "speech_decide": "gemini/gemini-2.5-flash"},
    "mixed": {"response": "openai/gpt-5.5", "feedback": "openai/gpt-4o-mini", "speech_decide": "openai/gpt-4o-mini"},
}
DEFAULT_RUN = ["gpt4o", "gpt5.4", "gpt5.5", "gemini"]

# シナリオ: ("say",text) 通常発話 / ("wait",sec) 沈黙 / ("bargein",text) 割り込み発話。
SCENARIO = [
    ("say", "やっほー、イブ！今日はなんだか元気だよ。"),
    ("wait", 9),
    ("say", "あのね、さっき道で子犬を拾っちゃって、今うちにいるの。"),
    ("wait", 3),
    ("bargein", "あ、ごめん、名前まだ決めてないんだけどね。"),
    ("wait", 8),
    ("say", "ごはんをあげたら、すぐ寝ちゃったよ。"),
    ("wait", 9),
]


async def synth_user_wav(url, text, speaker, rate=16000):
    import aiohttp

    async with aiohttp.ClientSession() as s:
        async with s.post(f"{url}/audio_query", params={"text": text, "speaker": speaker}) as r:
            q = await r.json()
        q["outputSamplingRate"] = rate
        q["outputStereoToMono"] = True
        async with s.post(f"{url}/synthesis", json=q, params={"speaker": speaker}) as r:
            return await r.read()


def wav_to_pcm(b):
    with wave.open(io.BytesIO(b), "rb") as wf:
        return wf.readframes(wf.getnframes())


def sh(s, n=70):
    return (str(s) if s is not None else "—")[:n].replace("\n", " ")


class CapturePlayer:
    def __init__(self):
        self.count = 0

    async def play_fn(self, audio, should_stop=None):
        if not audio:
            return
        self.count += 1
        for _ in range(max(1, len(audio) // 9000)):
            if should_stop is not None and should_stop():
                return
            await asyncio.sleep(0.03)


async def run_preset(name: str, models: dict[str, str]) -> None:
    print("\n" + "=" * 78)
    print(f"PRESET '{name}': response={models['response']} / feedback={models['feedback']} / decide={models['speech_decide']}")
    print("=" * 78)
    import tempfile

    art = tempfile.mkdtemp(prefix=f"eve_eval_{name}_")
    reg = ModelRegistry(overrides=dict(models))
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl"))
    await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(art, "r.jsonl"))
    await rag.initialize()
    await rag.warmup()
    pred = PredictionState()
    feedback = FeedbackLLM(reg, rag_store=rag, prediction_state=pred)

    T0 = time.monotonic()
    log: list[tuple[float, str, str]] = []

    def ev(kind, detail):
        log.append((time.monotonic() - T0, kind, detail))

    _frun = feedback.run

    async def frun_wrap(turns):
        r = await _frun(turns)
        if r:
            ev("fb", f"surprise={pred.surprise:>3} 感情={sh(r.emotions,10)} 要約={sh(r.summary,40)}")
        return r

    feedback.run = frun_wrap  # type: ignore
    fworker = FeedbackWorker(feedback, cache)
    state = SpeechState()
    queue = StimulusQueue()
    decider = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred, queue=queue, decide_fn=make_decide_fn(reg))
    player = CapturePlayer()
    audio = AudioPlayQueue(play_fn=player.play_fn)
    tts = VoicevoxTTS()
    stt = make_stt()

    async def stream_fn(messages):
        # 評価用に長さを軽く制限（本番は無制限）。Gemini 等 reasoning 系は小さいと途中で切れる。
        async for d in reg.stream("response", messages, max_tokens=400):
            yield d

    def on_complete():
        fworker.trigger()
        state.mark_eve_activity()

    orch = ResponseOrchestrator(
        audio, stream_fn, tts.generate, ContextAssembler(system_prompt=SPEECH_STYLE),
        conversation_cache=cache, rag_store=rag, prediction_state=pred, on_response_complete=on_complete,
    )
    _handle = orch.handle

    async def handle_wrap(stim):
        kind = "auto" if stim.kind == StimulusKind.AUTONOMOUS_SPEECH else "←U"
        await _handle(stim)
        ev(f"EVE{kind}", sh(orch.last_response))

    orch.handle = handle_wrap  # type: ignore
    runner = PipelineRunner(queue, orch, audio)
    monitor = SilenceMonitor(state=state, decider=decider, is_busy_fn=runner.is_busy)

    # 発話判定の中身も記録（speak/silence とも・speech_log を後で吐く）
    play_task = asyncio.create_task(audio.play_worker())
    run_task = asyncio.create_task(runner.run())
    fworker.start()
    state.mark_eve_activity()
    decider.start()
    monitor.start()
    await stt.warmup()

    url = Config.VOICEVOX_URL

    async def wait_idle(timeout=30.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0:
                return
            await asyncio.sleep(0.1)

    async def user_says(line, barge=False):
        wav = await synth_user_wav(url, line, USER_SPEAKER)
        heard = await stt.transcribe(wav_to_pcm(wav)) or line
        if barge:
            ev("BARGE", "ユーザ割り込み")
            audio.interrupt()
            runner.interrupt()
            state.mark_user_speech_start()
            queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)
        state.mark_user_speech_start()
        state.mark_user_utterance()
        ev("USER", sh(heard))
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        await wait_idle()

    for step in SCENARIO:
        kind, val = step
        if kind == "say":
            await user_says(val)
        elif kind == "bargein":
            await user_says(val, barge=True)
        elif kind == "wait":
            await asyncio.sleep(val)

    await monitor.stop()
    await decider.stop()
    await fworker.stop()
    run_task.cancel()
    play_task.cancel()
    await asyncio.sleep(0.2)

    print("\n-- 会話の流れ（T+秒）--")
    for t, kind, detail in sorted(log, key=lambda x: x[0]):
        tag = {"USER": "🧑USER ", "EVE←U": "🤖EVE  ", "EVEauto": "💭AUTO ", "fb": "  [fb] ", "BARGE": "⏸割込 "}.get(kind, kind)
        print(f"  {t:6.1f} {tag} {detail}")
    print("\n-- 発話判定ログ（speak/silence とも・理由/内容）--")
    for e in list(state.speech_log):
        print(f"   speak={str(e['speak']):5s} 理由={sh(e['reason'],40)} 内容={sh(e['content'],30)}")
    autos = [d for (_, k, d) in log if k == "EVEauto"]
    print(f"\n-- 概況 -- 自発発話={len(autos)} / 発話判定={len(state.speech_log)} / Eve音声={player.count}")

    await rag.shutdown()
    await cache.shutdown()
    await tts.close()


async def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "all":
        names = list(PRESETS)
    elif arg:
        names = [n.strip() for n in arg.split(",") if n.strip() in PRESETS]
    else:
        names = DEFAULT_RUN
    if not names:
        print(f"利用可能プリセット: {', '.join(PRESETS)}")
        return
    print(f"実行プリセット: {names}")
    for n in names:
        try:
            await run_preset(n, PRESETS[n])
        except Exception as e:
            print(f"[PRESET {n} 失敗] {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
