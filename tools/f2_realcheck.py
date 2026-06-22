"""F2 Tier-2 実機チェック（任意・手動・コスト発生）。

実 ModelRegistry(response 役) + 実 VOICEVOX で1プロンプトを流し、応答全文（自然さ目視）と
レイテンシ（TTFT / 初音声まで / 全文完了まで）を表示する。

前提:
  - .env に LLM キー（Claude 停止中 → OPENAI / GROQ / GEMINI のいずれか）
  - VOICEVOX が 127.0.0.1:50021 で起動
実行例:
  $env:PYTHONIOENCODING="utf-8"; python tools\f2_realcheck.py "こんにちは、軽く自己紹介して"
  python tools\f2_realcheck.py "プロンプト" --model groq/llama-3.3-70b-versatile --play

判定: 初音声まで ≤3s=良 / ≤5s=許容 / >5s=要調査（厳守ではない）。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, Stimulus, StimulusKind  # noqa: E402
from eve.response import ResponseOrchestrator, VoicevoxTTS  # noqa: E402


async def run(prompt: str, model: str | None, do_play: bool) -> None:
    reg = ModelRegistry()
    if model:
        reg.set_override("response", model)
    tts = VoicevoxTTS()

    t0 = time.monotonic()
    timings: dict[str, float | None] = {"ttft": None, "first_audio": None}

    async def stream_fn(messages):
        first = True
        async for d in reg.stream("response", messages):
            if first:
                timings["ttft"] = time.monotonic() - t0
                first = False
            yield d

    real_player = None
    if do_play:
        from eve.response import RealAudioPlayer

        real_player = RealAudioPlayer()

    async def play_fn(audio):
        if timings["first_audio"] is None:
            timings["first_audio"] = time.monotonic() - t0
        if real_player is not None:
            await real_player.play_fn(audio)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())
    orch = ResponseOrchestrator(audio, stream_fn, tts.generate)

    try:
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, payload=prompt))
        await audio.join()
    finally:
        worker.cancel()
        await tts.close()
        if real_player is not None:
            real_player.close()

    total = time.monotonic() - t0
    fa = timings["first_audio"]
    verdict = "（音声なし）"
    if fa is not None:
        verdict = "良(≤3s)" if fa <= 3.0 else ("許容(≤5s)" if fa <= 5.0 else "要調査(>5s)")

    print("\n===== F2 realcheck =====")
    print(f"model       : {reg.resolve('response')}")
    print(f"prompt      : {prompt}")
    print(f"response    : {orch.last_response}")
    print(f"TTFT        : {timings['ttft']:.2f}s" if timings["ttft"] else "TTFT        : N/A")
    print(f"first audio : {fa:.2f}s  -> {verdict}" if fa else "first audio : N/A")
    print(f"total       : {total:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default=None, help="response 役のモデルID上書き（低レイテンシ狙いに groq 等）")
    ap.add_argument("--play", action="store_true", help="実音声を再生（pyaudio 必要）")
    args = ap.parse_args()
    asyncio.run(run(args.prompt, args.model, args.play))


if __name__ == "__main__":
    main()
