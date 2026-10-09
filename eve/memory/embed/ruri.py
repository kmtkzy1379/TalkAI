"""Ruri v3 ローカル埋め込み（日本語特化・既定）。

cl-nagoya/ruri-v3-310m（768次元・cosine・mean pooling・JMTEBで mE5-large 超え）。
query/document でプレフィックスが必須（"検索クエリ: " / "検索文書: "）なのでここで付与する。
sentence-transformers / torch は遅延 import（マイク同様、使う時だけ重い依存を読む）。
同期 encode は executor へオフロード。`normalize_embeddings=True` で cosine=内積になる。
"""
from __future__ import annotations

import asyncio
import logging
import threading
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
        # ロードは1回だけ（`_encode` は executor の別々のスレッドから同時に呼ばれる）。実測 2026-10-09: 起動時の
        # ウォームアップと前回分のフィードバックの記憶の書き込みが同時に来て、同じ時刻に「ロード中」が2回出た
        # （モデルを2つ読み込み・起動が数秒遅れ・メモリも一時的に倍）。
        self._load_lock = threading.Lock()
        self.dim = 768  # ruri-v3-310m。ロード後に実値で上書きする
        # 直近のクエリの埋め込み（同じターンで記憶の検索と話し方の実例の除外が同じ発話を埋め込む＝二度計算しない）
        self._recent: dict[str, list[float]] = {}

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:  # 待っている間に先のスレッドが読み終えていれば、それを使う
                from sentence_transformers import SentenceTransformer
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Ruri 埋め込みモデルをロード中 (%s, device=%s)...", self.model_name, device)
                model = SentenceTransformer(self.model_name, device=device)
                try:
                    self.dim = int(model.get_sentence_embedding_dimension())
                except Exception:
                    pass
                self._model = model  # 読み終えてから公開する（途中の物を他のスレッドに使わせない）
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
        hit = self._recent.get(text)
        if hit is not None:
            return hit
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self._encode, [QUERY_PREFIX + text])
        if len(self._recent) >= 8:
            self._recent.pop(next(iter(self._recent)))  # いちばん古いものを捨てる
        self._recent[text] = res[0]
        return res[0]

    async def warmup(self) -> None:
        try:
            await self.embed_query("ウォームアップ")
        except Exception as e:
            logger.warning("Ruri ウォームアップ失敗（続行）: %s", e)
