"""画面キャプチャ（mss + downscale + pHash + JPEG b64 + 黒/空白検出）。

重い依存（mss/cv2/PIL/imagehash/numpy）は**関数内で遅延 import**（`eve.vlm` の他モジュールは
これに依存せず Tier-1 で軽い）。`grab_fn` を注入すれば mss 無し（テスト用に BGR ndarray を渡す）。

返すのは `(Frame, phash)`: Frame はループが持つ軽い値（b64 文字列・blank フラグ）、phash は
スレッド側の変化ゲート用。A11: 分散が小さい（黒/空白）フレームは `blank=True`（取得不可の可能性）。
"""
from __future__ import annotations

import base64
import logging
from typing import Callable, Optional

from ..clock import now_mono
from .types import Frame

logger = logging.getLogger(__name__)


class ScreenCapture:
    def __init__(
        self,
        monitor: int = 1,
        downscale_max: int = 1280,
        jpeg_quality: int = 70,
        blank_std_threshold: float = 6.0,
        grab_fn: Optional[Callable[[], "object"]] = None,
    ) -> None:
        self._monitor = monitor
        self._downscale_max = downscale_max
        self._jpeg_quality = jpeg_quality
        self._blank_std = blank_std_threshold
        self._grab_fn = grab_fn  # None → mss（遅延・スレッド毎インスタンス）
        self._sct = None
        self._counter = 0

    def _grab_mss(self):
        from mss import mss  # 遅延 import（headless/Tier-1 では読み込まない）
        import numpy as np

        if self._sct is None:
            self._sct = mss()  # 1スレッドに1インスタンス（共有しない）
        mon = self._sct.monitors[self._monitor]
        raw = self._sct.grab(mon)
        return np.array(raw)[:, :, :3].copy()  # BGRA → BGR

    def capture_one(self) -> tuple[Frame, int]:
        """1枚キャプチャ → (Frame, phash)。失敗時は例外（呼び出し側 A10 で安全停止）。"""
        import cv2
        import imagehash
        import numpy as np
        from PIL import Image

        bgr = (self._grab_fn or self._grab_mss)()
        h, w = bgr.shape[:2]
        if max(h, w) > self._downscale_max:
            scale = self._downscale_max / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        blank = float(np.std(bgr)) < self._blank_std  # A11: 黒/空白（取得不可の可能性）
        phash = int(str(imagehash.phash(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))), 16)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

        frame = Frame(frame_id=self._counter, mono_ts=now_mono(), jpeg_b64=jpeg_b64, blank=blank)
        self._counter += 1
        return frame, phash

    def grab_png(self, downscale_max: int = 1600) -> bytes:
        """OCR（文字ゲート）用に1枚を **PNG=可逆** で返す。失敗は空 bytes。

        VLM 送信用は 1024/JPEG q70 に落としてある（トークンとレイテンシのため）が、
        その画質で OCR すると読字が落ちてノイズ床が上がる。文字ゲートは**送信しない**ので
        解像度と可逆性を落とす理由が無い。実測（`tools/ocr_bench.py --controlled` 1600/PNG）:
        真ノイズ床 最大0.23% / 1行変更 2.89% ＝ 分離比 12.6倍。

        同じスレッド（`vlm-capture`）からしか呼ばれないので mss インスタンスを共有してよい。
        """
        import cv2

        bgr = (self._grab_fn or self._grab_mss)()
        h, w = bgr.shape[:2]
        if downscale_max and max(h, w) > downscale_max:
            scale = downscale_max / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", bgr)
        return buf.tobytes() if ok else b""

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
