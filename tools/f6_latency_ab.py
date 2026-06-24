r"""Flash 系の VLM レイテンシ/精度 A/B（同一フレームで比較）。

実機画面を2枚キャプチャし、各モデルに同じ multi-frame 画像を N 回送って lat とナレーションを出す。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\f6_latency_ab.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.vlm import parse_vision  # noqa: E402
from eve.vlm.capture import ScreenCapture  # noqa: E402
from eve.vlm.narrator import build_messages, _content_of  # noqa: E402

MODELS = ["gemini/gemini-3.5-flash", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite"]
N = 3


async def main() -> None:
    cap = ScreenCapture(monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
                        jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD)
    f1, _ = cap.capture_one()
    await asyncio.sleep(0.4)
    f2, _ = cap.capture_one()
    cap.close()
    messages = build_messages([f1, f2])

    for model in MODELS:
        reg = ModelRegistry(overrides={"vlm_leaf": model})
        lats, sample = [], ""
        for i in range(N):
            t = time.monotonic()
            try:
                resp = await reg.complete("vlm_leaf", messages)
                lats.append(time.monotonic() - t)
                if i == 0:
                    sample = parse_vision(_content_of(resp)).narration
            except Exception as e:
                print(f"  {model}: FAIL {type(e).__name__}: {str(e)[:80]}")
                break
        if lats:
            print(f"  {model:34s} lat avg={sum(lats)/len(lats):.1f}s min={min(lats):.1f}s max={max(lats):.1f}s")
            print(f"      例: {sample[:140]}")


if __name__ == "__main__":
    asyncio.run(main())
