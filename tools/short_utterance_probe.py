"""短い返事と物音の「声の長さ」をマイクで測る（API 不要・`MIN_SPEECH_SEC` を決めるため）。

最短発話ゲート（`eve/audio_input.py`・ユーザ 2026-09-29「ノイズカットは修正」）は**声の長さ**で判定する。
合成音の実測では「はい」「え？」「ね」「へー」が 0.096秒（確認窓ちょうど）まで短くなり、長さだけでは
短い返事と物音を分けられない可能性がある。本番と同じマイク・VAD・判定で、自分の声と物音を並べて見る。

使い方（TalkAI-live のフォルダで・イヤホン推奨）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\short_utterance_probe.py --sec 90

1語ずつ間を空けて「うん」「はい」「え？」「ね」「へー」などを言い、そのあと咳・机を叩く・キーボードなどの
物音を出す。区間ごとに「声の長さ・最大振幅」と、閾値ごとに通るか（○＝通る／×＝捨てる）を表示する。
**音声はどこにも送らず、保存もしない**。`--sec` 秒で自動で止まる（Ctrl+C でも止まる）。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # .env（VAD・音量ゲートの設定）を本番と同じに読む

from eve.audio_input import AudioInput  # noqa: E402
from eve.config import Config  # noqa: E402

THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)


def _peak(pcm: bytes) -> float:
    import numpy as np

    if not pcm:
        return 0.0
    return float(np.abs(np.frombuffer(pcm, np.int16)).max()) / 32768.0


async def main(sec: float) -> None:
    Config.MIN_SPEECH_SEC = 0.0  # 全部の区間を出す（閾値ごとの判定は表示の側で示す）
    state = {"peak": 0.0, "n": 0}

    def seg_start(pcm: bytes) -> None:
        state["peak"] = _peak(pcm)

    def seg_audio(pcm: bytes) -> None:
        state["peak"] = max(state["peak"], _peak(pcm))

    def seg_end(_accepted: bool, speech_sec: float) -> None:
        state["n"] += 1
        loud = state["peak"] >= Config.MIC_MIN_PEAK
        marks = "  ".join(f"{t:.2f}:{'○' if (loud and speech_sec >= t) else '×'}" for t in THRESHOLDS)
        note = "" if loud else f"（音量ゲート MIC_MIN_PEAK={Config.MIC_MIN_PEAK} で捨てる）"
        print(f"#{state['n']:02d} 声の長さ {speech_sec:.3f}秒  最大振幅 {state['peak']:.3f}{note}  |  {marks}",
              flush=True)

    ai = AudioInput(on_segment_start=seg_start, on_segment_audio=seg_audio, on_segment_end=seg_end)
    ai.start()
    print(f"計測中（{sec:.0f}秒で止まる）。1語ずつ間を空けて話す → そのあと物音を出す。"
          f"本番の設定: VAD_THRESHOLD={Config.VAD_THRESHOLD} VAD_ONSET_SEC={Config.VAD_ONSET_SEC} "
          f"SILENCE_LIMIT={Config.SILENCE_LIMIT} MIC_MIN_PEAK={Config.MIC_MIN_PEAK}", flush=True)
    print("列の見方: 0.20:○ ＝ MIN_SPEECH_SEC=0.20 にしてもこの区間は通る", flush=True)
    try:
        await asyncio.sleep(sec)
    finally:
        ai.stop()
        print(f"終了（区間 {state['n']} 個）", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sec", type=float, default=90.0, help="計測する秒数（既定 90）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(main(args.sec))
    except KeyboardInterrupt:
        pass
