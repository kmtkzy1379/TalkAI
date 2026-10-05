#!/usr/bin/env python3
"""「語＋っと・っとね」をまとめて合成した音（例 えーっと）の試聴ファイルを作る（API・GPU 不要）。

ユーザ 2026-09-29「少し不自然に感じる。そもそもこの数ならば普通に繋げてすべて作成してもいいかも」→ 付けてよい語 × っと・っとね を
1つの音で合成した（`tools/build_filler_bank.py --combos`）。鳴らす時と同じく前後の無音を刈った音を、組ごとに番号つきで書き出す。
比べる時は、前に 0.4秒の間で繋いだ試聴（listen/terminal_gap/ の同じ番号）と聴き比べる。

出力: listen/terminal_combo/NN_<組>.wav（普通の声・1本目）と README.txt（git 管理外）
実行（EVE の venv）:
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\combo_filler_audition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.config import Config  # noqa: E402
from eve.filler import audio as fa  # noqa: E402

OUT = REPO / "listen" / "terminal_combo"
ORDER = ["aa", "ee", "nn", "a", "e", "u", "uu", "o"]  # listen/terminal_gap/ と同じ並び（よく出る順）
TERMS = ["tto", "ttone"]


def main() -> int:
    bank = REPO / "bank" / Config.FILLER_VOICE
    man = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    first: dict = {}
    for c in man["clips"]:
        if "combo_terminal" not in (c.get("role") or []):
            continue
        key = (c.get("word"), c.get("terminal"))
        if key not in first:
            first[key] = c
    if not first:
        print(f"bank/{Config.FILLER_VOICE} に組が無い（tools/build_filler_bank.py --combos で作る）")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("[0-9][0-9]_*.wav"):  # 組を作り直した時に古い番号のファイルを残さない（このフォルダはこの道具の出力だけ）
        old.unlink()
    lines = [f"「語＋っと・っとね」をまとめて合成した音（バンク {Config.FILLER_VOICE}・普通の声・1本目・前後の無音を刈った音）",
             "並びは よく出る語の順（お は付けない・う／うー は うんっと／うーんっと＝ユーザ 2026-09-29）", ""]
    n = 0
    for w in ORDER:
        for t in TERMS:
            c = first.get((w, t))
            if c is None:
                continue
            n += 1
            x, sr = sf.read(str(bank / c["file"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            y = fa.trim(x, sr)
            name = f"{n:02d}_{c['text']}.wav"
            sf.write(str(OUT / name), y, sr, subtype="PCM_16")
            lines.append(f"{name}: {len(y) / sr:.2f}秒（合成した文字「{c.get('synth_text', '')}」）")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n出力: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
