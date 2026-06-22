"""Groq Whisper バックエンド（ロールバック/フォールバック用）。

現行 v1 と同じ。実測で無音/クリックに「ご視聴ありがとうございました」を出す＝幻聴多く
**非推奨**。緊急ロールバックや比較計測のために残す。
"""
from __future__ import annotations

import asyncio
import logging

from ..config import Config
from .base import Stt, pcm_to_wav
from .filter import clean_transcript

logger = logging.getLogger(__name__)


class GroqWhisperStt(Stt):
    def __init__(self, model: str = "whisper-large-v3", language: str = "ja") -> None:
        self.model = model
        self.language = language
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=Config.GROQ_API_KEY)
        return self._client

    def _request(self, wav: bytes):
        return self._get_client().audio.transcriptions.create(
            file=("speech.wav", wav), model=self.model, language=self.language, response_format="text"
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        wav = pcm_to_wav(audio_bytes)
        loop = asyncio.get_running_loop()
        try:
            r = await loop.run_in_executor(None, self._request, wav)
        except Exception as e:
            logger.warning("STT(groq) 失敗、空を返す: %s", e)
            return ""
        return clean_transcript(str(r))
