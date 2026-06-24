"""VlmWorker — single-flight 画面認識サイドカー（staleness/backpressure を構造的に解決）。

FeedbackWorker / SpeechDecider と同形（Event トリガ・idle gate・start/stop drain）。差分は **A1
自己再トリガ**（latest-wins には watermark が無いので、narrate 中に来た最後のフレームを取り残さない）
と **min-interval pacing**（A9・連続変化のコスト上限。トリガを捨てず deferred で律速）。

不変条件:
- single-flight: 長寿命タスク1本。narrate 中に何枚変化が来ても呼び出しは重ねない。終わったら
  **最新ウィンドウ**で次を1回（latest-window）。→ 遅延 ≤ 推論1回・古いバックログを溜めない。
- snapshot は await 前に value-copy（A2）。frame は frozen・b64 文字列で安価。
- A11: blank フレーム / visible=false は **VLM に中身を語らせず** 正直マーカを latest_vision に置く。
  surprise を上げず・発話も叩かない（無いものは語らない・捏造しない）。
- guard 付き発話トリガ（A5/Q4）: busy/ユーザ発話中/decider 非idle なら叩かない（割り込まない）。
"""
from __future__ import annotations

import asyncio
import logging
from difflib import SequenceMatcher
from typing import Awaitable, Callable, Optional

from ..clock import now_mono
from .narrator import NarrateFn
from .state import VisionState
from .types import Frame, VisionResult

logger = logging.getLogger(__name__)

BLANK_MARKER = "（今は画面を取得できませんでした：キャプチャ禁止/保護コンテンツの可能性）"


class VlmWorker:
    def __init__(
        self,
        *,
        vision_state: VisionState,
        prediction_state,
        narrate_fn: NarrateFn,
        speak_trigger: Optional[Callable[[], None]] = None,
        speak_guard: Optional[Callable[[], bool]] = None,
        frames_per_call: int = 4,
        min_interval_sec: float = 1.0,
        dedup_ratio: float = 0.85,
        now_fn: Callable[[], float] = now_mono,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._vs = vision_state
        self._pred = prediction_state
        self._narrate = narrate_fn
        self._speak_trigger = speak_trigger
        self._speak_guard = speak_guard or (lambda: True)
        self._k = frames_per_call
        self._min_interval = min_interval_sec
        self._dedup_ratio = dedup_ratio
        self._now = now_fn
        self._sleep = sleep_fn
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._last_call_ts = -1e9  # 初回は即時許可
        self._last_narration = ""

    # --- ライフサイクル ---
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self, drain_timeout: float = 2.0) -> None:
        self._stopping = True
        self._event.set()
        if not self._idle.is_set():
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def trigger(self) -> None:
        self._event.set()

    def is_idle(self) -> bool:
        return self._idle.is_set() and not self._event.is_set()

    # --- bridge 着地点（capture スレッド→loop の call_soon_threadsafe で呼ばれる・means-1）---
    def on_frame(self, frame: Frame, changed: bool) -> None:
        """全キャプチャを ring に積み（A8）、変化フレームなら worker を起こす。loop 上で実行。"""
        self._vs.add_frame(frame)
        if changed:
            self._event.set()

    # --- worker 本体 ---
    async def _run_forever(self) -> None:
        while not self._stopping:
            await self._event.wait()
            self._event.clear()
            if self._stopping:
                break
            # A9: min-interval pacing（トリガは捨てず deferred で律速）。
            wait = self._min_interval - (self._now() - self._last_call_ts)
            if wait > 0:
                try:
                    await self._sleep(wait)
                except asyncio.CancelledError:
                    raise
                if self._stopping:
                    break
            try:
                await self._process_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("VlmWorker 処理で例外（継続）")

    async def _process_once(self) -> None:
        frames = self._vs.snapshot(self._k)  # await 前に value-copy（A2）
        if not frames:
            return
        snap_id = frames[-1].frame_id
        # A11: 最新が blank → 視認不可。VLM に語らせず正直マーカ。surprise/発話に触れない。
        if frames[-1].blank:
            if self._vs.latest_vision != BLANK_MARKER:
                logger.info("👁 視認不可（画面を取得できず）")
            self._vs.latest_vision = BLANK_MARKER
            return
        self._last_call_ts = self._now()  # 呼び出し開始＝pacing 起点
        self._idle.clear()
        try:
            result = await self._narrate(frames)
        finally:
            self._idle.set()
        self._apply(result)
        # A1: narrate 中に新フレームが来ていたら自己再トリガ（min-interval で律速される）。
        ring = self._vs.ring
        if ring and ring[-1].frame_id != snap_id:
            self._event.set()

    def _apply(self, result: VisionResult) -> None:
        if not result.visible:
            # A11: 視認不可（黒/判読不能）→ 正直マーカ・surprise/発話に触れない。
            self._vs.latest_vision = BLANK_MARKER
            return
        if not result.narration:
            return  # 空（パース失敗等）→ no-op（落とさない）
        if self._is_dup(result.narration):
            self._vs.latest_vision = result.narration  # 最新性のため更新するが再トリガしない
            return
        logger.info("👁 %s", result.narration)  # 画面実況（実機で VLM 稼働を可視化）
        self._vs.latest_vision = result.narration
        self._last_narration = result.narration
        if result.surprise_diff is not None:
            self._pred.note_vlm_surprise(result.surprise_diff)  # 指標（数値ゲートにしない）
        if result.notable and self._speak_trigger is not None and self._speak_guard():
            self._speak_trigger()  # ガード通過時のみ発話判定を起こす（A5/Q4）

    def _is_dup(self, narration: str) -> bool:
        if not self._last_narration:
            return False
        ratio = SequenceMatcher(None, self._last_narration, narration).ratio()
        return ratio >= self._dedup_ratio
