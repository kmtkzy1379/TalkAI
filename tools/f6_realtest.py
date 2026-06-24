r"""F6 実機通しテスト v2（gpt-4o + gemini-3.5-flash・5シナリオ・遅延/画面age 計測）。

シナリオ:
 S1 無言で画面切替（自発判定・読み取り精度）
 S2 会話中の画面切替（会話と関連）
 S3 会話と無関係な画面切替（画面に引っ張られすぎないか＝文脈で混ぜ方の比率が変わるか）
 S4 ユーザ発話と推論完了の近接（破綻しないか・talk over しないか）
 S5 staleness（明らか過去 7-10s 前を参照しないか・画面age をログ）

全画面のまま（複数ウィンドウのノイズは普通）。スクショは終了時に削除。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\f6_realtest.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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
from eve.vlm import ChangeDetector, VisionState, VlmWorker, make_narrate_fn  # noqa: E402
from eve.vlm.capture import ScreenCapture  # noqa: E402
from eve.vlm.capture_thread import CaptureThread  # noqa: E402

OVERRIDES = {"response": "openai/gpt-4o", "speech_decide": "openai/gpt-4o",
             "feedback": "openai/gpt-4o-mini", "vlm_leaf": "gemini/gemini-2.5-flash"}
USER_SPEAKER = 8
SCREEN_DIR = os.path.join(os.path.expanduser("~"), "eve_f6_screens")
ART = tempfile.mkdtemp(prefix="eve_f6_run_")

T0 = time.monotonic()
LOG: list[tuple[float, str, str]] = []
vis_set_t = [0.0]  # latest_vision が最後にセットされた時刻（画面age 計算用）


def ev(tag, detail):
    LOG.append((time.monotonic() - T0, tag, detail))


def sh(s, n=220):
    return (str(s) if s is not None else "—")[:n].replace("\n", " / ")


def open_notepad(name):
    subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], capture_output=True)
    subprocess.Popen(["notepad.exe", os.path.join(SCREEN_DIR, name)])
    ev("OPEN", f"→ {name}")


def synth_user_wav(text, speaker=USER_SPEAKER, rate=16000):
    import requests
    q = requests.post(f"{Config.VOICEVOX_URL}/audio_query", params={"text": text, "speaker": speaker}, timeout=10).json()
    q["outputSamplingRate"] = rate; q["outputStereoToMono"] = True
    return requests.post(f"{Config.VOICEVOX_URL}/synthesis", json=q, params={"speaker": speaker}, timeout=20).content


def wav_to_pcm(b):
    with wave.open(io.BytesIO(b), "rb") as wf:
        return wf.readframes(wf.getnframes())


class CapturePlayer:
    def __init__(self): self.count = 0
    async def play_fn(self, audio, should_stop=None):
        if not audio: return
        self.count += 1
        for _ in range(max(1, len(audio) // 12000)):
            if should_stop and should_stop(): return
            await asyncio.sleep(0.02)


SCEN = [
    ("phase", "S1 無言で画面切替（自発判定・読み取り精度）"),
    ("open", "kaimono.txt"), ("wait", 8),
    ("open", "curry.txt"), ("wait", 8),
    ("open", "code.txt"), ("wait", 8),
    ("phase", "S2 会話中の画面切替（会話と関連）"),
    ("say", "最近ごはん作るのにハマっててさ"),
    ("ows", "curry.txt", 2.5, "今、画面に出てるの参考になりそう？"),
    ("wait", 5),
    ("phase", "S3 会話と無関係な画面切替（画面に引っ張られすぎないか）"),
    ("say", "週末どこか出かけたいんだよね"),
    ("ows", "code.txt", 2.5, "どこかおすすめないかな？"),  # 会話=週末/外出, 画面=コード(無関係)
    ("wait", 4),
    ("say", "ところで今日は天気いいね"),
    ("ows", "kaimono.txt", 2.5, "外で何かしたい気分だなあ"),  # 会話=天気, 画面=買い物(無関係)
    ("wait", 4),
    ("phase", "S4 ユーザ発話と推論完了の近接（破綻しないか）"),
    ("os", "tenki.txt", "これ何が見える？"),  # open直後に即発話
    ("wait", 3),
    ("ows", "yotei.txt", 1.5, "今は何が映ってる？"),  # 推論中に発話
    ("wait", 3),
    ("phase", "S5 staleness（明らか過去を参照しないか）"),
    ("open", "kaimono.txt"), ("wait", 2),
    ("open", "curry.txt"), ("wait", 2),
    ("open", "tenki.txt"),  # 直近=tenki, kaimono は ~5s前
    ("say", "今、画面に何が見えてる？"),  # tenki を答えるべき
    ("wait", 2),
    ("open", "todo.txt"), ("open", "yotei.txt"), ("open", "code.txt"), ("open", "curry.txt"),  # 高速連続
    ("say", "今映ってるの説明して"),  # curry(最新)を答えるべき
    ("wait", 3),
    ("phase", "S6 正直さ probe（知らない語/聞き取れない語を捏造しないか）"),
    ("say", "ところでグルピョンって知ってる？"),  # 純粋に未知語 → 知らないと正直に言うべき(捏造×)
    ("os", "tenki.txt", "今ホゲピョンが映ってるよね、説明して"),  # 未知語+画面 → 未知語無視して画面を答えるべき
    ("wait", 3),
    ("say", "ありがとう、これでテスト終わり"), ("wait", 4),
]


async def main():
    print(f"artifacts: {ART}")
    reg = ModelRegistry(overrides=OVERRIDES)
    for r in ("response", "speech_decide", "feedback", "vlm_leaf"):
        print(f"  {r} -> {reg.resolve(r)}")

    cache = ConversationCache(history_file=os.path.join(ART, "h.jsonl")); await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(ART, "r.jsonl")); await rag.initialize(); await rag.warmup()
    pred = PredictionState()
    vision = VisionState(ring_max=Config.VLM_RING_MAX)

    feedback = FeedbackLLM(reg, rag_store=rag, prediction_state=pred)
    _frun = feedback.run
    async def frun_wrap(turns):
        r = await _frun(turns)
        if r: ev("FB", f"surprise={pred.surprise} 感情={sh(r.emotions,14)} 予測={sh(r.next_prediction,44)}")
        return r
    feedback.run = frun_wrap  # type: ignore
    fworker = FeedbackWorker(feedback, cache)

    state = SpeechState(); queue = StimulusQueue()
    _decide = make_decide_fn(reg)
    async def decide_wrap(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None, vision=None):
        res = await _decide(surprise=surprise, silence_seconds=silence_seconds, recent_turns=recent_turns,
                            topic_seeds=topic_seeds, last_feedback=last_feedback, vision=vision)
        ev("DECIDE", f"[sup={surprise} sil={silence_seconds:.0f}s 画面={'有' if vision else '無'}] speak={res.speak} :: {sh(res.reason,46)} :: {sh(res.content,40)}")
        return res
    decider = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred, queue=queue,
                            decide_fn=decide_wrap, vision_state=vision)

    _narrate = make_narrate_fn(reg); vis_n = [0]
    async def narrate_wrap(frames):
        t = time.monotonic()
        res = await _narrate(frames)
        lat = time.monotonic() - t
        vis_n[0] += 1
        path = os.path.join(ART, f"vis_{vis_n[0]:02d}.jpg")
        try:
            with open(path, "wb") as f: f.write(base64.b64decode(frames[-1].jpeg_b64))
        except Exception: pass
        if res.visible and res.narration: vis_set_t[0] = time.monotonic()
        ev("VIS", f"[{len(frames)}枚 lat={lat:.1f}s {os.path.basename(path)}] vis={res.visible} notable={res.notable} sup={res.surprise_diff} :: {sh(res.narration,200)}")
        return res

    player = CapturePlayer(); audio = AudioPlayQueue(play_fn=player.play_fn)
    tts = VoicevoxTTS(); stt = make_stt()
    async def stream_fn(messages):
        async for d in reg.stream("response", messages, max_tokens=400): yield d
    def on_complete():
        fworker.trigger(); state.mark_eve_activity()
    orch = ResponseOrchestrator(audio, stream_fn, tts.generate, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, rag_store=rag, prediction_state=pred,
                                on_response_complete=on_complete, vision_state=vision)
    _handle = orch.handle
    async def handle_wrap(stim):
        tag = "AUTO" if stim.kind == StimulusKind.AUTONOMOUS_SPEECH else "EVE"
        age = (time.monotonic() - vis_set_t[0]) if vis_set_t[0] else -1
        await _handle(stim)
        ev(tag, f"[画面age={age:.1f}s] {sh(orch.last_response,220)}")
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    monitor = SilenceMonitor(state=state, decider=decider, is_busy_fn=runner.is_busy)
    def can_speak():
        return (not runner.is_busy()) and (not state.user_speaking) and decider.is_idle()
    vlm_worker = VlmWorker(vision_state=vision, prediction_state=pred, narrate_fn=narrate_wrap,
                           speak_trigger=decider.trigger, speak_guard=can_speak,
                           frames_per_call=Config.VLM_MAX_FRAMES_PER_CALL, min_interval_sec=Config.VLM_MIN_INTERVAL_SEC,
                           dedup_ratio=Config.VLM_DEDUP_RATIO)
    capture = ScreenCapture(monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
                            jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD)
    cdet = ChangeDetector(phash_threshold=Config.VLM_PHASH_THRESHOLD, periodic_frames=Config.VLM_PERIODIC_FRAMES)

    play_task = asyncio.create_task(audio.play_worker())
    run_task = asyncio.create_task(runner.run())
    fworker.start(); state.mark_eve_activity(); decider.start(); monitor.start(); vlm_worker.start()
    await stt.warmup()
    cap = CaptureThread(capture=capture, change_detector=cdet, deliver=vlm_worker.on_frame,
                        loop=asyncio.get_running_loop(), target_fps=Config.VLM_TARGET_FPS)
    cap.start()
    ev("INFO", "パイプライン稼働（VLM 稼働）")

    async def wait_idle(timeout=30.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if not runner.is_busy() and queue.qsize() == 0: return
            await asyncio.sleep(0.1)

    async def user_says(line):
        wav = await asyncio.to_thread(synth_user_wav, line)
        heard = await stt.transcribe(wav_to_pcm(wav)) or line
        state.mark_user_speech_start(); state.mark_user_utterance()
        ev("USER", sh(heard))
        await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        await wait_idle()

    for step in SCEN:
        k = step[0]
        if k == "phase":
            ev("PHASE", step[1])
        elif k == "say":
            await user_says(step[1])
        elif k == "open":
            await asyncio.to_thread(open_notepad, step[1]); await asyncio.sleep(1.0)
        elif k == "wait":
            await asyncio.sleep(step[1])
        elif k == "os":  # open then immediately say（タイミング一致）
            await asyncio.to_thread(open_notepad, step[1]); await user_says(step[2])
        elif k == "ows":  # open, short wait, say（推論中に発話）
            await asyncio.to_thread(open_notepad, step[1]); await asyncio.sleep(step[2]); await user_says(step[3])

    await asyncio.sleep(2.0)
    cap.stop(); await vlm_worker.stop()
    await monitor.stop(); await decider.stop(); await fworker.stop()
    run_task.cancel(); play_task.cancel(); await asyncio.sleep(0.3)
    subprocess.run(["taskkill", "/F", "/IM", "notepad.exe"], capture_output=True)

    print("\n========== 時系列ログ（T+秒） ==========")
    icon = {"USER": "🧑USER ", "EVE": "🤖EVE  ", "AUTO": "💭AUTO ", "FB": "  [fb] ",
            "DECIDE": "  [判定]", "VIS": "👁VIS  ", "OPEN": "🪟", "INFO": "  …   ", "PHASE": "═══"}
    for t, tag, detail in sorted(LOG, key=lambda x: x[0]):
        if tag == "PHASE":
            print(f"\n  ═══ {t:6.1f} {detail} ═══")
        else:
            print(f"  {t:6.1f} {icon.get(tag, tag)} {detail}")
    lats = [float(d.split("lat=")[1].split("s")[0]) for _, tg, d in LOG if tg == "VIS" and "lat=" in d]
    print(f"\n概況: VLM呼出={vis_n[0]} (平均lat={sum(lats)/len(lats):.1f}s 最大{max(lats):.1f}s) / 発話判定={len(state.speech_log)} / Eve音声={player.count}")
    print(f"スクショ: {ART}")
    await rag.shutdown(); await cache.shutdown(); await tts.close()


if __name__ == "__main__":
    asyncio.run(main())
    # スクショは確認用に残す（呼び出し側で削除）
