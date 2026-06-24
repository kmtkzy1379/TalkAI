"""F6 VLM 画面認識（単発・複数フレーム / capture-on-change・snapshot モード）。

capture(専用スレッド・dumb)→ pHash 変化ゲート → 直近 N 枚の連続フレーム(変化前アンカー含む)を
single-flight worker が1回の multi-frame VLM 呼び出しで narration 化 → latest_vision 更新 +
note_vlm_surprise + ガード付き should_speak トリガ。staleness は latest-window + single-flight +
drop-oldest ring で構造的に回避。詳細は docs/COMPONENT_LOGIC.md §H。
"""
from __future__ import annotations

from .change_detector import ChangeDetector, hamming
from .narrator import build_messages, make_narrate_fn
from .parser import parse_vision
from .state import VisionState
from .types import Frame, VisionResult
from .worker import BLANK_MARKER, VlmWorker

__all__ = [
    "Frame",
    "VisionResult",
    "parse_vision",
    "ChangeDetector",
    "hamming",
    "VisionState",
    "make_narrate_fn",
    "build_messages",
    "VlmWorker",
    "BLANK_MARKER",
]
