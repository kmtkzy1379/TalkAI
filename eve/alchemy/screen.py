"""イブのゲーム画面を画面認識（VLM）へ渡す口。

画面認識のキャプチャは専用スレッド（`vlm/capture_thread.py`）で動き、`ScreenCapture(grab_fn=...)` の
grab_fn から1枚ずつ受け取る。ゲーム中はこの grab_fn を「デスクトップ」から「イブのゲームのページ」に
差し替える（ユーザ指定: ユーザーのPC画面は認識させない）。

- 画面写し（Playwright）は loop 上でしか撮れないので、loop のタスクが撮って**最新の1枚だけ**を置き、
  キャプチャのスレッドはそれを読むだけ（書くのは loop・読むのはスレッド。参照の差し替え1回なので
  ロックは要らない＝`PIPELINE_DESIGN.md` §9.3 の「スレッドは loop の状態を書き換えない」を守る）。
- まだ1枚も無い時は真っ黒の画像を返す（＝画面認識は「取得できませんでした」の正直な印を置き、中身を作らない）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GameScreenFeed:
    def __init__(self, client, *, interval_sec: float = 0.5, size: tuple = (1024, 576)) -> None:
        self._client = client
        self._interval = interval_sec
        self._size = size
        self._latest: Optional[bytes] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="alchemy-screen")

    async def stop(self) -> None:
        t, self._task = self._task, None
        if t is not None:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._latest = None

    async def warm(self, timeout: float = 3.0) -> bool:
        """最初の1枚が撮れるまで待つ（差し替え直後に真っ黒を画面認識へ渡さないため）。"""
        shot = await self._client.screenshot()
        if shot:
            self._latest = shot
        return self._latest is not None

    async def _run(self) -> None:
        while True:
            try:
                shot = await self._client.screenshot()
                if shot:
                    self._latest = shot
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("🎮 ゲーム画面の写しに失敗（続行）", exc_info=True)
            await asyncio.sleep(self._interval)

    def grab(self):
        """キャプチャのスレッドから呼ばれる: 最新の1枚を BGR の配列で返す（無ければ黒）。"""
        import cv2
        import numpy as np

        data = self._latest
        if data:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        w, h = self._size
        return np.zeros((h, w, 3), dtype=np.uint8)
