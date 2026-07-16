"""J-2 Web 検索（inc1 = ddgs スニペット層）。"""
from __future__ import annotations

from .capability import register_search_capability
from .client import SearchClient

__all__ = ["SearchClient", "register_search_capability"]
