#!/usr/bin/env python3
"""長い版のフィラー（ユーザ 2026-09-28「1フィラー、1〜1.5秒くらいが最適かも」）の試聴ファイルを作る（API・GPU 不要）。

語ごとに番号つきの1ファイル: 今の短い版（bank/eve_calm）→ 0.8秒 → 長い版 1.0秒 → 0.8秒 → 長い版 1.5秒。
鳴らす時と同じく前後の無音を刈った音で並べる（`eve/filler/bank.py` の読み込みと同じ `fa.trim`）。

実行（EVE の venv）:
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\long_filler_audition.py
出力: listen/long/NN_<語>.wav と README.txt（git 管理外）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.filler import audio as fa  # noqa: E402

SHORT, LONG = REPO / "bank" / "eve_calm", REPO / "bank" / "eve_calm_long"
OUT = REPO / "listen" / "long"
GAP = 0.8


def clips(bank: Path) -> dict:
    """(語, 長さ) → 1本目のファイル（前後の無音を刈った音）。"""
    man = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    out: dict = {}
    for c in man["clips"]:
        if "independent" not in (c.get("role") or []):
            continue
        key = (c["text"], float(c["requested_sec"]))
        if key in out:
            continue
        x, sr = sf.read(str(bank / c["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        out[key] = (fa.trim(x, sr), sr)
    return out


def main() -> int:
    short, long_ = clips(SHORT), clips(LONG)
    OUT.mkdir(parents=True, exist_ok=True)
    words = sorted({w for (w, _s) in long_}, key=lambda w: ["あ", "あー", "え", "えー", "んー", "あの", "その",
                                                              "う", "うー", "お"].index(w) if w in
                   ["あ", "あー", "え", "えー", "んー", "あの", "その", "う", "うー", "お"] else 99)
    lines = ["長い版のフィラー（1語 1〜1.5秒）の試聴。各ファイル: 今の短い版 → 長い版 1.0秒 → 長い版 1.5秒（間 0.8秒）", ""]
    for n, w in enumerate(words, 1):
        parts, desc = [], []
        s = next(((x, sr) for (ww, _sec), (x, sr) in short.items() if ww == w), None)
        for label, item in (("今", s), ("長い1.0", long_.get((w, 1.0))), ("長い1.5", long_.get((w, 1.5)))):
            if item is None:
                continue
            x, sr = item
            if parts:
                parts.append(np.zeros(int(sr * GAP), np.float32))
            parts.append(x)
            desc.append(f"{label} {len(x) / sr:.2f}秒")
        sr = (s or long_[(w, 1.0)])[1]
        name = f"{n:02d}_{w}.wav"
        sf.write(str(OUT / name), np.concatenate(parts), sr, subtype="PCM_16")
        lines.append(f"{name}: " + " → ".join(desc))
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n出力: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
