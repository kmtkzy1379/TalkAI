#!/usr/bin/env python3
"""S2 の試聴ファイルを作る（API・GPU 不要。できたバンク `bank/<voice>/` を並べ替えるだけ）。

出力 `listen/s2/`（git 管理外）:
  bank_all.wav        … 応答非依存フィラー10語＋っと/っとね を、語ごとに3本ずつ（声が揃っているか）
  stretch_<語>.wav    … 伸ばし棒の試作: 元の語 →（1秒）→ 「ー」を継いだ語、を3本ぶん（途切れて聞こえないか）
  gap_<秒>.wav        … つなぎ「、」の間を 0.1〜0.5秒 で変えた組み合わせ（どの間が自然か）
  README.txt          … 並び順と、決めてほしいこと

**本番のフィラー再生と同じ処理（`eve/filler/audio.py` の trim / stretch）を通す**ので、
ここで聴いた音がそのまま本番で鳴る（間の長さだけは本番で再生装置を開く数十msが足される）。

実行（EVE の venv）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\filler_audition.py
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GAPS = (0.1, 0.2, 0.3, 0.4, 0.5)
# 間の聴き比べに使う組み合わせ（すべて表の規則どおり: っと・っとね は あの・その の後ろに付けない）
COMBOS = [("a", "ee"), ("aa", "ano"), ("e", "ttone"), ("nn", "sono"), ("ano", "ee"), ("ee", "a", "tto")]
BETWEEN = 1.5  # 組み合わせ同士の間（秒）


def save(path: Path, x, sr: int) -> float:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return len(x) / sr


def main() -> int:
    import numpy as np
    import yaml

    from eve.filler import audio as fa

    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="eve")
    ap.add_argument("--out", type=Path, default=REPO / "listen" / "s2")
    ap.add_argument("--compare", default="", help="聴き比べる元のバンク（例 eve）。語ごとに 元→新 を並べた bank_ab.wav を作る")
    ap.add_argument("--gaps", default="0.1,0.2,0.3,0.4,0.5", help="つなぎの間の試聴ファイルを作る秒（カンマ区切り）")
    ap.add_argument("--laugh", action="store_true",
                    help="笑いのバンク（bank/<voice>_laugh）の試聴も作る: laugh_words.wav・laugh_tone.wav")
    args = ap.parse_args()

    table = yaml.safe_load((REPO / "fillers.yaml").read_text(encoding="utf-8"))
    bank_dir = REPO / "bank" / args.voice
    man = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    clips: dict = {}
    sr = 48000
    for c in man["clips"]:
        x, sr = fa.read_wav((bank_dir / c["file"]).read_bytes())
        clips.setdefault(c["id"], []).append((c, x))
    for v in clips.values():
        v.sort(key=lambda cx: cx[0]["file"])
    entries = {e["id"]: e for e in table["entries"]}
    words = [e for e in table["entries"] if "independent" in (e.get("role") or [])]
    terms = [e for e in table["entries"] if "terminal" in (e.get("role") or [])]
    missing = [e["id"] for e in words + terms if e["id"] not in clips]
    if missing:
        print("バンクに無い語:", missing, "（tools/build_filler_bank.py で作る）")
        return 1
    st = table.get("stretch") or {}
    add_sec = float(st.get("add_sec") or 0.4)
    lines = ["フィラーの試聴（S2）— 生成 " + str(bank_dir), ""]

    # 1) 全語
    seq = []
    order = sorted(words, key=lambda e: -float(e.get("weight") or 0)) + terms
    names = []
    for e in order:
        for _, x in clips[e["id"]]:
            seq += [fa.trim(x, sr), fa.silence(0.8, sr)]
        seq.append(fa.silence(BETWEEN - 0.8, sr))
        names.append(f"{e['text']}×{len(clips[e['id']])}")
    d = save(args.out / "bank_all.wav", np.concatenate(seq), sr)
    lines += [f"■ bank_all.wav（{d:.0f}秒）: 声が揃っているか",
              "  並び（重みの大きい順・各3本・本番と同じく前後の無音を刈ったもの）: " + " / ".join(names), ""]

    # 2) 伸ばし棒
    lines += [f"■ stretch_<語>.wav: 伸ばし棒の試作（今鳴った語に「ー」を約{add_sec:.1f}秒継ぐ）",
              "  並び: 元の語 →（1秒）→ 継いだ語 →（1.5秒）… を3本ぶん。**途切れて聞こえないか**を聴く",
              "  （継ぎ目は波形の0点・30ms のクロスフェード・音量は切り口に合わせる・",
              "    素材3本の中から継ぎ目で声の高さが一番近い所を選ぶ）"]
    for e in words:
        v = e.get("stretch")
        src = clips.get(f"stretch_{v}")
        if not v or not src:
            continue
        seq = []
        ok = 0
        for i, (_, x) in enumerate(clips[e["id"]]):
            # 本番と同じく素材は全本から選ばせる（継ぎ目の高さが一番近い所を使う）
            s = fa.stretch(x, [cx[1] for cx in src], sr, add_sec=add_sec)
            if s is None:
                continue
            ok += 1
            seq += [fa.trim(x, sr), fa.silence(1.0, sr), s, fa.silence(BETWEEN, sr)]
        if ok:
            d = save(args.out / f"stretch_{e['id']}.wav", np.concatenate(seq), sr)
            lines.append(f"  stretch_{e['id']}.wav: {e['text']} → {e['text']}ー（{ok}本・{d:.0f}秒）")
    lines.append("")

    # 3) つなぎの間
    combo_txt = ["、".join(entries[i]["text"] for i in c) for c in COMBOS]
    lines += ["■ gap_<秒>.wav: つなぎ「、」の間の聴き比べ（0.1 / 0.2 / 0.3 / 0.4 / 0.5 秒）",
              "  並び（各ファイル共通・1.5秒ずつ空ける）: " + " / ".join(combo_txt),
              "  **どの間が自然か**を聴く（本番では再生装置を開く数十msが足される）", ""]
    for g in [float(x) for x in args.gaps.split(",") if x.strip()]:
        seq = []
        for k, c in enumerate(COMBOS):
            pieces = [fa.trim(clips[i][k % len(clips[i])][1], sr) for i in c]
            seq += [fa.join(pieces, sr, g), fa.silence(BETWEEN, sr)]
        save(args.out / f"gap_{g:.1f}.wav", np.concatenate(seq), sr)
    if args.compare:
        old_dir = REPO / "bank" / args.compare
        oman = json.loads((old_dir / "manifest.json").read_text(encoding="utf-8"))
        old: dict = {}
        for c in oman["clips"]:
            x, _ = fa.read_wav((old_dir / c["file"]).read_bytes())
            old.setdefault(c["id"], []).append((c, x))
        seq = []
        names = []
        for e in order:
            if e["id"] not in old:
                continue
            for _, x in sorted(old[e["id"]], key=lambda cx: cx[0]["file"]):
                seq += [fa.trim(x, sr), fa.silence(0.6, sr)]
            seq.append(fa.silence(1.0, sr))
            for _, x in clips[e["id"]]:
                seq += [fa.trim(x, sr), fa.silence(0.6, sr)]
            seq.append(fa.silence(2.0, sr))
            names.append(e["text"])
        d = save(args.out / "bank_ab.wav", np.concatenate(seq), sr)
        lines += [f"■ bank_ab.wav（{d:.0f}秒）: 元のバンク（{args.compare}）→ 新しいバンク（{args.voice}）の聴き比べ",
                  "  並び: 語ごとに 元3本 →（1秒）→ 新3本 →（2秒）… 順番: " + " / ".join(names), ""]
    if args.laugh:
        ldir = REPO / "bank" / (args.voice + str((table.get("laugh") or {}).get("bank_suffix") or "_laugh"))
        lman = json.loads((ldir / "manifest.json").read_text(encoding="utf-8"))
        lcl: dict = {}
        for c in lman["clips"]:
            x, _ = fa.read_wav((ldir / c["file"]).read_bytes())
            lcl.setdefault(c["id"], []).append((c, x))
        laugh_ids = [e["id"] for e in table["entries"] if "笑い" in (e.get("labels") or [])]
        seq, names = [], []
        for lid in laugh_ids:
            if lid not in lcl:
                continue
            for _, x in lcl[lid]:
                seq += [fa.trim(x, sr), fa.silence(0.8, sr)]
            seq.append(fa.silence(1.2, sr))
            names.append(entries[lid]["text"])
        d = save(args.out / "laugh_words.wav", np.concatenate(seq), sr)
        lines += [f"■ laugh_words.wav（{d:.0f}秒）: 笑いの語（表の B の「笑い」・🤭 の声色）を各3本",
                  "  その場で笑う時（確率 0.9 以上）はこの中から等確率で1本。順番: " + " / ".join(names), ""]
        seq, names = [], []
        for e in order:
            if e["id"] not in lcl or e["id"] not in clips:
                continue
            seq += [fa.trim(clips[e["id"]][0][1], sr), fa.silence(0.8, sr)]  # 普通の声
            for _, x in lcl[e["id"]]:
                seq += [fa.trim(x, sr), fa.silence(0.6, sr)]           # 笑いの声色
            seq.append(fa.silence(1.5, sr))
            names.append(e["text"])
        if seq:
            d = save(args.out / "laugh_tone.wav", np.concatenate(seq), sr)
            lines += [f"■ laugh_tone.wav（{d:.0f}秒）: 応答非依存フィラーの笑いの声色（確率 0.75 以上の時に使う）",
                      "  並び: 語ごとに 普通の声1本 →（0.8秒）→ 笑いの声色3本。順番: " + " / ".join(names), ""]
    gap_now = (table.get("timing") or {}).get("join_gap_sec")
    adopted = (table.get("stretch") or {}).get("adopted")
    lines += ["決めてほしいこと:",
              "  1. 声が揃っているか・落ち着いて聞こえるか（bank_all.wav" + ("・bank_ab.wav" if args.compare else "") + "）",
              "  2. 伸ばし棒を使うか（stretch_*.wav）→ fillers.yaml の stretch.adopted を true / false"
              + (f"（今: {adopted}）" if adopted is not None else "（未決）"),
              "  3. つなぎの間（gap_*.wav）→ fillers.yaml の timing.join_gap_sec"
              + (f"（決定済み: {gap_now}秒）" if gap_now is not None else "（未決）")]
    if args.laugh:
        lines.append("  4. 笑いの語と笑いの声色（laugh_words.wav・laugh_tone.wav）")
    (args.out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n出力:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
