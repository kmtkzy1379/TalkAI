"""埋め込み backend パッケージ。差し替えは make_embedder(backend) で（STT と同方式）。"""
from __future__ import annotations

from ...config import Config
from .base import Embedder


def make_embedder(backend: str | None = None) -> Embedder:
    backend = backend or Config.EMBED_BACKEND
    if backend == "ruri":
        from .ruri import RuriEmbedder

        return RuriEmbedder()
    if backend == "openai":
        from .openai_embed import OpenAIEmbedder

        return OpenAIEmbedder()
    raise ValueError(f"unknown EMBED_BACKEND: {backend}")


__all__ = ["Embedder", "make_embedder"]
