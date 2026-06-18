"""STT パッケージ。差し替えは make_stt(backend) で。"""
from __future__ import annotations

from ..config import Config
from .base import Stt, pcm_to_wav
from .filter import clean_transcript


def make_stt(backend: str | None = None) -> Stt:
    backend = backend or Config.STT_BACKEND
    if backend == "openai":
        from .openai_transcribe import OpenAITranscribeStt

        return OpenAITranscribeStt()
    if backend == "groq":
        from .groq_whisper import GroqWhisperStt

        return GroqWhisperStt()
    raise ValueError(f"unknown STT_BACKEND: {backend}")


__all__ = ["Stt", "pcm_to_wav", "clean_transcript", "make_stt"]
