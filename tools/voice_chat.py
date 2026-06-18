r"""フルループ実行: 声で話して声で返る（mic→STT→応答LLM→TTS→再生）。

前提: .env に OPENAI/GROQ/GEMINI 鍵、VOICEVOX 起動、マイク+イヤホン推奨（AIの声をmicが拾わない）。
実行(旧venvに torch/pyaudio/openai あり):
  $env:PYTHONIOENCODING="utf-8"; ..\portfolio8-VLM-AI\venv\Scripts\python.exe tools\voice_chat.py
Ctrl+C で終了。
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.logsetup import configure
from eve.voice_loop import VoiceLoop


async def main() -> None:
    configure()
    loop = VoiceLoop()
    try:
        await loop.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await loop.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
