"""Ruri v3 ローカル埋め込み（日本語特化・既定）。

cl-nagoya/ruri-v3-310m（768次元・cosine・mean pooling・JMTEBで mE5-large 超え）。
query/document でプレフィックスが必須（"検索クエリ: " / "検索文書: "）なのでここで付与する。
sentence-transformers / torch は遅延 import（マイク同様、使う時だけ重い依存を読む）。
同期 encode は executor へオフロード。`normalize_embeddings=True` で cosine=内積になる。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ...config import Config
from .base import Embedder

logger = logging.getLogger(__name__)

QUERY_PREFIX = "検索クエリ: "  # Ruri v3 のクエリ用プレフィックス（モデルカードで確認済）
DOC_PREFIX = "検索文書: "  # 文書（記憶）用プレフィックス


class RuriEmbedder(Embedder):
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or Config.RURI_MODEL
        self._model = None
        self.dim = 768  # ruri-v3-310m。ロード後に実値で上書きする

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Ruri 埋め込みモデルをロード中 (%s, device=%s)...", self.model_name, device)
            self._model = SentenceTransformer(self.model_name, device=device)
            try:
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                pass
            logger.info("Ruri ロード完了 (dim=%d)", self.dim)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [e.tolist() for e in embs]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        prefixed = [DOC_PREFIX + t for t in texts]
        return await loop.run_in_executor(None, self._encode, prefixed)

    async def embed_query(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self._encode, [QUERY_PREFIX + text])
        return res[0]

    async def warmup(self) -> None:
        try:
            await self.embed_query("ウォームアップ")
        except Exception as e:
            logger.warning("Ruri ウォームアップ失敗（続行）: %s", e)
