"""VLM(narrator)出力（タグ付きテキスト）のパーサ。

`feedback/parser.py` と同方針: 小モデルの出力は崩れがちなので JSON でなく**タグ付きテキスト
regex**でフィールド抽出し、**決して raise しない**（全フィールド安全既定・全角コロン `：` 許容）。

A11（ハルシネ防止）: モデルが黒/判読不能と言ったら `visible=False`。本文に「視認不可」等が
あれば本文有無に関わらず不可視扱い（中身を捏造させない）。
"""
from __future__ import annotations

import re
from typing import Optional

from .types import VisionResult

# 視認不可を表す語（本文 or visible タグ値に現れたら不可視扱い）。
_INVISIBLE_MARKERS = ("視認不可", "判読不能", "見えない", "取得できません", "真っ黒", "not visible", "unreadable")
_NEGATIVE = ("no", "false", "いいえ", "不可", "低", "なし", "0")

_NARR_ALIASES = ("narration", "ナレーション", "実況", "描写", "説明")
_NOTABLE_ALIASES = ("notable", "notability", "注目", "重要", "言及")
_VISIBLE_ALIASES = ("visible", "視認", "見える")
_DIFF_ALIASES = ("surprise", "surprise_diff", "予測差", "驚き")


def _extract_line(text: str, aliases: tuple[str, ...]) -> str:
    for a in aliases:
        m = re.search(
            rf"^\s*{re.escape(a)}\s*[:：]\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            return m.group(1).strip()
    return ""


def _extract_int(text: str, aliases: tuple[str, ...]) -> Optional[int]:
    for a in aliases:
        m = re.search(
            rf"^\s*{re.escape(a)}\s*[:：]\s*[^\d-]*(-?\d+)", text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _truthy(value: str, default: bool) -> bool:
    v = value.strip().lower()
    if not v:
        return default
    return not any(n in v for n in _NEGATIVE)


def parse_vision(text: Optional[str]) -> VisionResult:
    """タグ付きテキストを VisionResult に。raise しない・安全既定。"""
    text = text or ""
    lower = text.lower()
    narration = _extract_line(text, _NARR_ALIASES) or ""

    visible_raw = _extract_line(text, _VISIBLE_ALIASES)
    visible = _truthy(visible_raw, default=True)
    # 本文/タグに視認不可マーカがあれば不可視（A11・中身を信用しない）
    if any(m.lower() in lower for m in _INVISIBLE_MARKERS):
        visible = False

    notable = _truthy(_extract_line(text, _NOTABLE_ALIASES), default=False)
    diff = _extract_int(text, _DIFF_ALIASES)

    if not visible:
        # 視認不可: 中身は採用せず surprise も上げない（捏造防止）。
        return VisionResult(narration="", notable=False, surprise_diff=None, visible=False)
    return VisionResult(narration=narration, notable=notable, surprise_diff=diff, visible=True)
