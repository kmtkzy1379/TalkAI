"""FeedbackLLM 出力（タグ付きテキスト）のパーサ。

設計判断（CLAUDE.md「規律はコードで強制」/ v1 実測）:
- 小モデル（既定 gpt-4o-mini）の出力は**部分的/崩れがち**。JSON は1文字崩れで全損し
  FEP ループを黙って壊す（surprise が stale 化＝装飾化）。よって**タグ付きテキスト regex**で
  フィールド単位に頑健抽出する（v1 feedback.py の `_*_TAG_RE` 方式の形を踏襲・激縮）。
- **パーサは決して raise しない**。全フィールド安全既定。全角コロン `：` も受ける。
- `prediction_diff` は抽出失敗時 **None**（= carry-forward の合図。0 を捏造しない＝
  「完璧に予測した」という嘘で沈黙を誘発しない）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

MAX_TAGS = 6


@dataclass
class FeedbackResult:
    summary: str = ""
    emotions: str = ""  # イブの感情
    user_emotion: str = ""  # ユーザの感情（推定）
    next_prediction: str = ""  # 次に何が起きるかの予測（次 run の FEP 比較対象）
    prediction_diff: Optional[int] = None  # 0-100。None=抽出失敗（carry-forward）
    reason: str = ""
    topic_tags: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """有意な抽出が1つも無い（完全なゴミ/空応答）。"""
        return not (
            self.summary
            or self.emotions
            or self.user_emotion
            or self.next_prediction
            or self.reason
            or self.topic_tags
            or self.prediction_diff is not None
        )


# タグ別名（英日・全角コロン許容）。値は同一行（lean: 多行値は1行目のみ拾う）。
_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "要約"),
    "emotions": ("emotion", "emotions", "感情", "イブの感情"),
    "user_emotion": ("user_emotion", "ユーザ感情", "ユーザーの感情", "ユーザの感情"),
    "next_prediction": ("next_prediction", "prediction", "次の予測", "予測"),
    "reason": ("reason", "理由"),
    "topic_tags": ("topic_tags", "tags", "タグ", "話題タグ"),
}
_DIFF_ALIASES: tuple[str, ...] = ("prediction_diff", "surprise", "予測差", "驚き")


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


def _extract_tags(text: str, aliases: tuple[str, ...]) -> list[str]:
    raw = _extract_line(text, aliases)
    if not raw:
        return []
    parts = re.split(r"[,、，]", raw)
    return [p.strip() for p in parts if p.strip()][:MAX_TAGS]


def parse_feedback(text: Optional[str]) -> FeedbackResult:
    """タグ付きテキストを FeedbackResult に。raise しない・全フィールド安全既定。"""
    text = text or ""
    return FeedbackResult(
        summary=_extract_line(text, _ALIASES["summary"]),
        emotions=_extract_line(text, _ALIASES["emotions"]),
        user_emotion=_extract_line(text, _ALIASES["user_emotion"]),
        next_prediction=_extract_line(text, _ALIASES["next_prediction"]),
        prediction_diff=_extract_int(text, _DIFF_ALIASES),
        reason=_extract_line(text, _ALIASES["reason"]),
        topic_tags=_extract_tags(text, _ALIASES["topic_tags"]),
    )
