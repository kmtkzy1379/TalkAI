"""画面変化ゲート（pHash hamming）— 「VLM を呼ぶトリガか」だけを判定する。

v1 の `vlm/capture/change_detector.py` を簡素化（SSIM/4段ChangeLevel は廃）。**リングに何を積むか
は決めない**（リングは全キャプチャを積む＝A8）。ここは純粋に「前回の参照から十分変わったか／
静止が続いたので periodic に1回見るか」を bool で返すだけ。状態は参照 pHash と idle カウンタのみ。
"""
from __future__ import annotations

DEFAULT_PHASH_THRESHOLD = 12  # hamming 距離（0-64）。小さいほど敏感。
DEFAULT_PERIODIC_FRAMES = 0  # >0 で「静止が N フレーム続いたら強制 True」。0=無効（静止はコスト0）。


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class ChangeDetector:
    def __init__(
        self,
        phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
        periodic_frames: int = DEFAULT_PERIODIC_FRAMES,
    ) -> None:
        self._threshold = phash_threshold
        self._periodic = periodic_frames
        self._reference: int | None = None
        self._idle = 0

    def evaluate(self, phash: int) -> bool:
        """True = この pHash は VLM 呼び出しに値する（変化 or periodic 強制）。"""
        if self._reference is None:
            self._reference = phash
            self._idle = 0
            return True  # 初回は必ず見る
        if hamming(self._reference, phash) < self._threshold:
            self._idle += 1
            if self._periodic and self._idle >= self._periodic:
                self._idle = 0
                return True  # 静止が続いた → たまに見直す（既定 無効）
            return False
        # 変化あり → 参照更新
        self._reference = phash
        self._idle = 0
        return True
