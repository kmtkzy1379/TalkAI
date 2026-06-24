"""F6 VLM の値型（純データ・依存なし）。

- Frame: キャプチャ1枚。**ループには ndarray を渡さず JPEG base64 文字列で持つ**
  （スナップショットの value-copy が安価・不変＝§9 のロック不要設計と相性が良い）。
  `blank=True` は黒/空白フレーム（キャプチャ禁止/保護コンテンツの可能性・A11）。
- VisionResult: narrator(VLM)の解析結果。`visible=False` は「視認不可」＝**捏造させない**（A11）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Frame:
    frame_id: int
    mono_ts: float
    jpeg_b64: str
    blank: bool = False  # 黒/空白（取得不可の可能性・A11）


@dataclass
class VisionResult:
    narration: str = ""
    notable: bool = False  # 言及に値する変化か（VLM/LLM の判断・指標）
    surprise_diff: Optional[int] = None  # 0-100。None=不明（surprise を carry・捏造しない）
    visible: bool = True  # False=視認不可（黒/判読不能）＝中身を語らせない（A11）

    def is_empty(self) -> bool:
        return not (self.narration or self.notable or self.surprise_diff is not None)
