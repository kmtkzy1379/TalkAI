r"""マイク実検証ハーネス: mic → VAD → STT(gpt-4o-transcribe) → 文字表示。

自分の発話／クリック／タイピングが実際どう検出されるかを確認する。Ctrl+C で終了。
STT は Config.STT_BACKEND（既定 openai=gpt-4o-transcribe）。`--groq` で現行比較も可。

実行（旧venvに torch/pyaudio あり）:
  $env:PYTHONIOENCODING="utf-8"; ..\portfolio8-VLM-AI\venv\Scripts\python.exe tools\mic_check.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config
from eve.logsetup import configure
from eve.stt import make_stt, pcm_to_wav


async def main() -> None:
    configure()
    backend = "groq" if "--groq" in sys.argv else None
    save = "--save" in sys.argv
    save_dir = os.path.join(os.path.dirname(__file__), "audio_samples", "captured")
    if save:
        os.makedirs(save_dir, exist_ok=True)
        print(f"発話WAVを保存: {save_dir}")
    stt = make_stt(backend)
    print(f"STT backend = {backend or Config.STT_BACKEND} / model = {Config.STT_MODEL}")
    print("STT ウォームアップ中...")
    await stt.warmup()

    from eve.audio_input import AudioInput  # torch/pyaudio を要する→ここで遅延 import

    ai = AudioInput()
    ai.start()
    print("=== 話してください。クリック/タイピングも試してOK。Ctrl+C で終了 ===")
    n = 0
    try:
        while True:
            audio = await ai.get_audio()
            n += 1
            if save:
                path = os.path.join(save_dir, f"utt_{n:03d}.wav")
                with open(path, "wb") as f:
                    f.write(pcm_to_wav(audio))
            t0 = time.monotonic()
            text = await stt.transcribe(audio)
            dt = time.monotonic() - t0
            secs = len(audio) / (Config.SAMPLE_RATE * 2)
            tag = f" -> {os.path.basename(path)}" if save else ""
            print(f"[発話 {secs:.1f}s / STT {dt:.2f}s] -> {text!r}{tag}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        ai.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
