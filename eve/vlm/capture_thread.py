"""CaptureThread — 専用 OS スレッド（§9.2 の (iii)・連続キャプチャ）。

スレッドは **dumb**: grab→pHash→変化ゲート だけ。loop 所有状態（ring/PredictionState）には
**触れない**。変化判定の済んだフレームを `loop.call_soon_threadsafe(deliver, frame, changed)` で
ループ上の `deliver`(=VlmWorker.on_frame) に渡す（§9.3 means-1・単一書込はループ側）。

A10: キャプチャ失敗（mss 初期化失敗・headless/Wayland 等）は捕捉してスレッドを**安全停止**
（システムは落とさない・視覚なしで継続・捏造しない）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .capture import ScreenCapture
from .change_detector import ChangeDetector

logger = logging.getLogger(__name__)


class CaptureThread:
    def __init__(
        self,
        *,
        capture: ScreenCapture,
        change_detector: ChangeDetector,
        deliver: Callable[..., None],  # loop 上の on_frame（call_soon_threadsafe 経由で呼ぶ）
        loop,
        target_fps: float = 2.0,
        schedule: Callable[..., None] | None = None,  # テスト注入用（既定 loop.call_soon_threadsafe）
    ) -> None:
        self._capture = capture
        self._detector = change_detector
        self._deliver = deliver
        self._loop = loop
        self._interval = 1.0 / target_fps if target_fps > 0 else 0.5
        self._schedule = schedule or (lambda *a: self._loop.call_soon_threadsafe(*a))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="vlm-capture", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self._capture.close()

    def _step(self) -> bool:
        """1反復: grab→gate→deliver。True=継続可 / False=キャプチャ不可で停止すべき（A10）。"""
        try:
            frame, phash = self._capture.capture_one()
        except Exception:
            logger.exception("画面キャプチャ失敗 → キャプチャ停止（視覚なしで継続・A10）")
            return False
        changed = self._detector.evaluate(phash)
        self._schedule(self._deliver, frame, changed)  # ループ上へ橋渡し（単一書込はループ側）
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            if not self._step():
                break  # A10: 安全停止
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - t0)))
