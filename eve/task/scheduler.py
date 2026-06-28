"""ReconcileTimer — 周期スケジューラ（`SilenceMonitor` 先例・~1s tick）。

「時計を見る」役: due な Pending があり executor が idle なら `trigger()`。SilenceMonitor とは
独立の loop タスク＝競合なし（沈黙中も回る＝予約は会話が無くても発火する）。`when` 比較は store が
壁時計UTC で行い、本 timer の周期は loop の sleep（monotonic）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ReconcileTimer:
    def __init__(self, *, store, executor, tick_sec: float = 1.0) -> None:
        self._store = store
        self._executor = executor
        self._tick = tick_sec
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._tick)
            try:
                self.tick()
            except Exception:
                logger.exception("ReconcileTimer tick で例外（継続）")

    def tick(self) -> bool:
        """due があり executor idle なら trigger（トリガしたら True・テスト用）。"""
        if not self._executor.is_idle():
            return False  # single-flight（処理中は重ねない）
        if not self._store.due_now():
            return False
        self._executor.trigger()
        return True
