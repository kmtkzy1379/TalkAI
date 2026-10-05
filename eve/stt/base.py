"""STT の差し替え抽象。

差し替え境界は `transcribe(audio_bytes: bytes) -> str` の1メソッドのみ（v1 と同契約）。
audio_bytes は VAD が切り出した **生PCM int16（mono, SAMPLE_RATE）**。WAV 化は各backendで行う。
"""
from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod

from ..config import Config


def pcm_to_wav(pcm: bytes) -> bytes:
    """生PCM int16 を WAV バイトに包む（STT API へ送る形式）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(Config.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(Config.SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


class Stt(ABC):
    # 話している間に音声を受け取る方式か（S0）。True の backend は `start/feed/commit/clear/reset`
    # を持ち、`MicSttInputSource` がマイクの区間フックからそれらを呼ぶ（`openai_live.py`）。
    streaming = False

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """発話PCM → 文字列。失敗/非音声は "" を返す（落とさない）。"""
        ...

    async def warmup(self) -> None:
        """起動時のコールドスタート緩和（接続/クライアント初期化）。既定 no-op。"""
        return

    async def close(self) -> None:
        """終了時の後片付け（接続を閉じる等）。既定 no-op。"""
        return
