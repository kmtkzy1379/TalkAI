"""STT 出力の極小フィルタ。

gpt-4o-transcribe は非音声で危険な日本語定型句を出さない（実測）が、稀に音イベントタグ
`[drumming]` / `(music)` / `【拍手】` のような**括弧だけの注記**を返す。これは丸ごと破棄する。
それ以外（短い英語片 "Thank you." 等）は VAD 源ゲートで元々届きにくいので、ここでは触らない
（過剰フィルタで正規入力を消さないため）。
"""
from __future__ import annotations

import re

_BRACKET_ONLY = re.compile(r"^[\(\[（【].*[\)\]）】]$")


def clean_transcript(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if _BRACKET_ONLY.match(t):
        return ""  # 音イベントタグ等は破棄
    return t
