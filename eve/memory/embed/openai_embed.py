"""OpenAI 埋め込み backend（A/B 比較用）。text-embedding-3-small（1536次元）。

OpenAI は query/document のプレフィックス区別が無いので両方そのまま埋め込む。
同期 SDK は executor へオフロード。クライアントは遅延生成＋使い回し（STT と同方式）。
採択しなかった場合はこのファイルごと削除できる（ストア本体は embedder 非依存）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ...config import Config
from .base import Embedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(Embedder):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or Config.EMBED_MODEL
        self._client = None
        self.dim = 1536  # text-embedding-3-small

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        r = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in r.data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed, texts)

    async def embed_query(self, text: str) -> list[float]:
        res = await self.embed_documents([text])
        return res[0]

    async def warmup(self) -> None:
        try:
            await self.embed_query("warmup")
        except Exception as e:
            logger.warning("OpenAI 埋め込みウォームアップ失敗（続行）: %s", e)
