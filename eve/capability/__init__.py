"""J-0 共有 Capability(能力)層。本 increment は read-only サブセットのみ。"""
from __future__ import annotations

from .registry import MEANS_BOUNDARY, NEVER_MEANS, Capability, CapabilityRegistry

__all__ = ["Capability", "CapabilityRegistry", "MEANS_BOUNDARY", "NEVER_MEANS"]
