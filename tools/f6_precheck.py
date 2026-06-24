r"""F6 実機テストの前提チェック（本番ハーネスの前に潰す）。

1. mss が実画面を撮れるか（shape/std＝黒くないか）。
2. VOICEVOX が生きているか（127.0.0.1:50021）。
3. 必須鍵の有無（値は出さない・bool のみ）。
4. **未検証の核**: 実 Gemini(vlm_leaf) へ multi-frame 画像送信が通るか（litellm image_url→inline_data）。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\f6_precheck.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.vlm import parse_vision  # noqa: E402
from eve.vlm.capture import ScreenCapture  # noqa: E402
from eve.vlm.narrator import build_messages  # noqa: E402


async def main() -> None:
    print("=== 1. mss キャプチャ ===")
    try:
        cap = ScreenCapture(monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
                            jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD)
        frame, phash = cap.capture_one()
        cap.close()
        print(f"  OK frame_id={frame.frame_id} blank={frame.blank} b64_len={len(frame.jpeg_b64)} phash={phash:#018x}")
    except Exception as e:
        print(f"  FAIL mss: {type(e).__name__}: {e}")
        frame = None

    print("=== 2. VOICEVOX ===")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{Config.VOICEVOX_URL}/version", timeout=aiohttp.ClientTimeout(total=3)) as r:
                print(f"  OK version={await r.text()}")
    except Exception as e:
        print(f"  FAIL voicevox: {type(e).__name__}: {e}")

    print("=== 3. 鍵（bool のみ・値は出さない）===")
    print(f"  OPENAI={bool(Config.OPENAI_API_KEY)} GEMINI={bool(Config.GEMINI_API_KEY)} GROQ={bool(Config.GROQ_API_KEY)}")

    print("=== 4. 実 Gemini(vlm_leaf) へ画像送信（litellm image 形式の検証）===")
    reg = ModelRegistry()
    print(f"  vlm_leaf -> {reg.resolve('vlm_leaf')}")
    if frame is None:
        print("  SKIP（フレーム無し）")
        return
    try:
        # 同じフレームを2枚渡して multi-frame 経路も試す
        messages = build_messages([frame, frame])
        resp = await reg.complete("vlm_leaf", messages)
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = str(resp)[:300]
        print("  --- raw content ---")
        print("  " + (content or "(空)").replace("\n", "\n  "))
        vr = parse_vision(content)
        print(f"  --- parsed --- visible={vr.visible} notable={vr.notable} surprise={vr.surprise_diff}")
        print(f"  narration={vr.narration!r}")
        print("  OK 画像送信が通った")
    except Exception as e:
        print(f"  FAIL gemini image: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
