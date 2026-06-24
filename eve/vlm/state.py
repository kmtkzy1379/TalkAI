"""VisionState — F6 の loop 所有 ephemeral 状態（単一書込・同期読み・ロック不要・§9）。

- ring: 直近 N 枚の**全キャプチャ**（変化フレームだけでなく＝変化前アンカーを含む・A8）。
  書込者は唯一 `on_frame`（capture スレッドから call_soon_threadsafe で loop 上に届く・means-1）。
- latest_vision: 最新ナレーション or 「視認不可」正直マーカ（A11）。書込者は唯一 VlmWorker。
  読み手は ResponseOrchestrator（# 画面 注入）/ SpeechDecider（発話判定の入力）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from ..clock import now_mono
from .types import Frame


class VisionState:
    def __init__(self, ring_max: int = 6) -> None:
        self.ring: deque[Frame] = deque(maxlen=ring_max)  # 自動 drop-oldest（上限=レイテンシ bound）
        self.latest_vision: Optional[str] = None
        self.latest_vision_mono: float = 0.0  # latest_vision をセットした時刻（鮮度判定用）

    def add_frame(self, frame: Frame) -> None:
        """全キャプチャを積む（変化前アンカー保持・A8）。唯一の ring 書込点。"""
        self.ring.append(frame)

    def snapshot(self, k: int) -> list[Frame]:
        """直近 k 枚を value-copy（list 新規・Frame は frozen 不変）。await 前に同期で呼ぶ（A2）。"""
        return list(self.ring)[-k:]

    def set_latest(self, text: str, mono: Optional[float] = None) -> None:
        """最新ナレーションを時刻付きでセット（唯一の latest_vision 書込点＝VlmWorker）。"""
        self.latest_vision = text
        self.latest_vision_mono = mono if mono is not None else now_mono()

    def fresh_vision(self, ttl: float, now: Optional[float] = None) -> Optional[str]:
        """ttl 秒以内に更新された latest_vision のみ返す。古ければ None（明らか過去を参照させない）。"""
        if self.latest_vision is None:
            return None
        n = now if now is not None else now_mono()
        if (n - self.latest_vision_mono) > ttl:
            return None  # 古すぎ → 画面情報なし扱い（応答LLMは正直に「今は見えない」と言える）
        return self.latest_vision
