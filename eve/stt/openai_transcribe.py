"""OpenAI gpt-4o-transcribe バックエンド（既定）。

実測で無音/クリックに危険な定型句を出さない（無音→空文字）。`prompt` で固有語の文脈注入可。
同期SDKは executor でオフロード。クライアントは遅延生成＋使い回し。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..config import Config
from .base import Stt, pcm_to_wav
from .filter import clean_transcript

logger = logging.getLogger(__name__)


class OpenAITranscribeStt(Stt):
    def __init__(
        self,
        model: Optional[str] = None,
        language: str = "ja",
        prompt: Optional[str] = None,
    ) -> None:
        self.model = model or Config.STT_MODEL
        self.language = language
        self.prompt = prompt
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        return self._client

    def _request(self, wav: bytes):
        client = self._get_client()
        kwargs = dict(file=("speech.wav", wav), model=self.model)
        if self.language:
            kwargs["language"] = self.language
        if self.prompt:
            kwargs["prompt"] = self.prompt
        try:
            return client.audio.transcriptions.create(**kwargs)
        except Exception:
            kwargs.pop("language", None)  # language 非対応時はフォールバック
            return client.audio.transcriptions.create(**kwargs)

    async def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        wav = pcm_to_wav(audio_bytes)
        loop = asyncio.get_running_loop()
        try:
            r = await loop.run_in_executor(None, self._request, wav)
        except Exception as e:  # 一時失敗は起こりうる → ログして空（落とさない）
            logger.warning("STT(openai) 失敗、空を返す: %s", e)
            return ""
        return clean_transcript(getattr(r, "text", "") or "")

    async def warmup(self) -> None:
        try:
            import numpy as np

            await self.transcribe(np.zeros(Config.SAMPLE_RATE // 2, dtype=np.int16).tobytes())
        except Exception:
            pass
