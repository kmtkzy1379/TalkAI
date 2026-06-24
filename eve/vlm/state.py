"""VisionState — F6 の loop 所有 ephemeral 状態（単一書込・同期読み・ロック不要・§9）。

- ring: 直近 N 枚の**全キャプチャ**（変化フレームだけでなく＝変化前アンカーを含む・A8）。
  書込者は唯一 `on_frame`（capture スレッドから call_soon_threadsafe で loop 上に届く・means-1）。
- latest_vision: 最新ナレーション or 「視認不可」正直マーカ（A11）。書込者は唯一 VlmWorker。
  読み手は ResponseOrchestrator（# 画面 注入）/ SpeechDecider（発話判定の入力）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .types import Frame


class VisionState:
    def __init__(self, ring_max: int = 6) -> None:
        self.ring: deque[Frame] = deque(maxlen=ring_max)  # 自動 drop-oldest（上限=レイテンシ bound）
        self.latest_vision: Optional[str] = None

    def add_frame(self, frame: Frame) -> None:
        """全キャプチャを積む（変化前アンカー保持・A8）。唯一の ring 書込点。"""
        self.ring.append(frame)

    def snapshot(self, k: int) -> list[Frame]:
        """直近 k 枚を value-copy（list 新規・Frame は frozen 不変）。await 前に同期で呼ぶ（A2）。"""
        return list(self.ring)[-k:]
