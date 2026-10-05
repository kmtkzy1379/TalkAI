#!/usr/bin/env python3
"""感情ごとのフィラーの声の試聴（S5・番号つきの別ファイル・API 不要・合成しない＝バンクの音をそのまま並べる）。

ユーザ 2026-09-26「フィラーも感情載せるでいいと思う。変だったら変えればいいだけだし」。
`bank/<FILLER_VOICE>_emo_<感情>/`（`tools/build_emotion_banks.py` で作ったもの）を、語ごとに
「普通の声 → その感情の声」で並べる。1ファイル = 1感情。README.txt に番号と並び。

実行（EVE の venv）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\emotion_filler_audition.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.config import Config  # noqa: E402
from eve.emotion import BANK_SUFFIX, load_voice_emoji  # noqa: E402
from eve.filler import audio as fa  # noqa: E402
from eve.filler.bank import FillerBank, FillerTable  # noqa: E402
from eve.response.tts_irodori import resolve_repo_path  # noqa: E402

OUT = REPO / "listen" / "emotion_filler"


def main() -> int:
    table = FillerTable.from_yaml(Path(resolve_repo_path(Config.FILLER_TABLE)))
    emoji = load_voice_emoji(Path(resolve_repo_path(Config.EMOTION_TABLE)))
    base = REPO / "bank" / Config.FILLER_VOICE
    dirs = {k: base.parent / (base.name + BANK_SUFFIX[k]) for k in emoji}
    bank = FillerBank.load(table, base, emotion_dirs=dirs, emotion_emoji=emoji)
    if not bank.by_emotion:
        print("感情のフィラーのバンクが無い（tools/build_emotion_banks.py で作る）")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    sr = bank.sr
    order = list(bank.table.weights)  # 語＋特殊フィラーの組（バンクに無い組は bank.table から外れている）
    lines = ["感情ごとのフィラーの声（2026-09-26・bank/%s_emo_*）" % Config.FILLER_VOICE, "",
             "1ファイル = 1感情。語ごとに「普通の声 →（0.5秒）→ その感情の声」を並べ、語と語の間は1.2秒。",
             "語の並び: " + " / ".join(table.texts.get(w, w) for w in order), ""]
    for num, (emo, pool) in enumerate(bank.by_emotion.items(), 1):
        seq = []
        for w in order:
            if w not in pool or w not in bank.clips:
                continue
            seq += [bank.clips[w][0].samples, fa.silence(0.5, sr), pool[w][0].samples, fa.silence(1.2, sr)]
        f = f"{num:02d}_{emo}.wav"
        (OUT / f).write_bytes(fa.to_wav(np.concatenate(seq), sr))
        lines.append(f"  {num:02d}  {emo} {emoji.get(emo, '')}  {f}")
        print(f"{num:02d} {emo} {emoji.get(emo, '')}")
    lines += ["", "変だと思った感情は、番号で教えてください（その感情だけ絵文字を変えるか、普通の声に戻す）。"]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("出力:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
