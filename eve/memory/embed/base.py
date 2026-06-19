"""埋め込み backend の差し替え抽象（STT の `Stt` ABC と同じ思想）。

差し替え境界は embed_documents / embed_query の2メソッド。返すのは float ベクトル。
クエリと文書でプレフィックス/前処理が異なる backend があるため2つに分ける
（Ruri は "検索クエリ: " / "検索文書: " のプレフィックスが必須）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    dim: int = 0  # 埋め込み次元（ロード後に確定。検索時の次元整合チェックに使う）

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """記憶として保存するテキスト群 → ベクトル群。"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """検索クエリ → ベクトル。"""
        ...

    async def warmup(self) -> None:
        """起動時のコールドスタート緩和（モデルロード/接続）。既定 no-op。"""
        return
