"""VLM(narrator)出力（タグ付きテキスト）のパーサ。

`feedback/parser.py` と同方針: 小モデルの出力は崩れがちなので JSON でなく**タグ付きテキスト
regex**でフィールド抽出し、**決して raise しない**（全フィールド安全既定・全角コロン `：` 許容）。

A11（ハルシネ防止）: モデルが `visible: no`（画面全体を読み取れなかった）と返したら `visible=False`
（中身を採用しない・捏造させない）。

本文の語（「見えない」「判読不能」等）では判定しない（2026-10-09 ユーザ「キーワードで説明文破棄をやめる。
そもそもここの仕組みを消す」）。画面認識の指示は「分からない所は、どこがどう分からないかを正直に言う」
（`narrator.py` の VLM_SYSTEM）なので、本文に「右下の字幕は判読不能」と書くのは正しい振る舞いで、
それを語で拾うと、読めていた残りの説明ごと捨てて「画面を取得できませんでした」に置き換えていた。
"""
from __future__ import annotations

import re
from typing import Optional

from .types import VisionResult

_NEGATIVE = ("no", "false", "いいえ", "不可", "低", "なし", "0")

_NARR_ALIASES = ("narration", "ナレーション", "実況", "描写", "説明")
_NOTABLE_ALIASES = ("notable", "notability", "注目", "重要", "言及")
_VISIBLE_ALIASES = ("visible", "視認", "見える")
_DIFF_ALIASES = ("surprise", "surprise_diff", "予測差", "驚き")
_PRED_ALIASES = ("prediction", "予想", "次の予想")


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
    narration = _extract_line(text, _NARR_ALIASES) or ""

    # 読み取れたかはモデル自身の visible タグだけで決める（本文の語では決めない・上の注記）
    visible = _truthy(_extract_line(text, _VISIBLE_ALIASES), default=True)

    notable = _truthy(_extract_line(text, _NOTABLE_ALIASES), default=False)
    diff = _extract_int(text, _DIFF_ALIASES)
    prediction = _extract_line(text, _PRED_ALIASES)

    if not visible:
        # 視認不可: 中身は採用せず surprise も上げない（捏造防止）。予想も持ち越さない（見えていない画面の予想）。
        return VisionResult(narration="", notable=False, surprise_diff=None, visible=False)
    return VisionResult(narration=narration, notable=notable, surprise_diff=diff, visible=True,
                        prediction=prediction)
