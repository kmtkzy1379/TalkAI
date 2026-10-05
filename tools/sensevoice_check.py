#!/usr/bin/env python3
"""SenseVoice で音声から感情を取り、入れた感情と一致するかを確かめる（SenseVoice の venv で実行）。

役割（docs/FILLER_DESIGN.md「感情分析の役割分担」）:
  ユーザの感情 = SenseVoice（音声）/ イブの感情 = FeedbackLLM（テキスト）/ Qwen はフィラー選択のみ。
ここでは SenseVoice を**検証用の耳**として使い、Irodori で入れた感情が音声に乗っているかを見る。

2つのモード:
  --clips DIR    tools/emotion_probe_synth.py が作ったクリップ。入れた感情 × 聞き取った感情の対応表
  --wav PATH --transcript PATH
                 5分会話（tools/conversation_nucc.py の出力）。発話ごとに切り出して判定。
                 合成時に発話の間へ**ちょうど 0 の無音**を入れているので、そこで区切れば
                 発話単位に正確に戻せる（区切り数が台本の行数と一致するかを検査する）。

SenseVoice の出力は "<|ja|><|NEUTRAL|><|Speech|><|woitn|>本文" の形。
感情タグ・音響イベントタグ・認識テキストに分けて扱う。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TAG = re.compile(r"<\|([^|]+)\|>")
EMO_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}


def load_model(device: str):
    from funasr import AutoModel
    last = None
    for kw in ({"model": "FunAudioLLM/SenseVoiceSmall", "hub": "hf"},
               {"model": "iic/SenseVoiceSmall"}):
        try:
            return AutoModel(device=device, disable_update=True, **kw)
        except Exception as e:  # 取得元を順に試す
            last = e
    raise SystemExit(f"SenseVoice を読み込めない: {last}")


def to16k(d: np.ndarray, sr: int) -> np.ndarray:
    import torch
    import torchaudio.functional as F
    if d.ndim > 1:
        d = d.mean(axis=1)
    t = torch.from_numpy(d.astype("float32"))
    if sr != 16000:
        t = F.resample(t, sr, 16000)
    return t.numpy()


def recognize(model, audio16k: np.ndarray) -> dict:
    out = model.generate(input=audio16k, cache={}, language="ja", use_itn=False, batch_size_s=60)
    raw = out[0]["text"] if out else ""
    tags = TAG.findall(raw)
    emo = next((t for t in tags if t in EMO_TAGS), "?")
    events = [t for t in tags if t not in EMO_TAGS and t not in ("ja", "woitn", "withitn")]
    return {"emotion": emo, "events": events, "text": TAG.sub("", raw).strip(), "raw": raw}


def split_on_zero_gaps(d: np.ndarray, sr: int, min_gap: float = 0.04):
    """ちょうど 0 が min_gap 秒以上続く箇所で区切る（合成時に入れた無音）。"""
    z = (d == 0.0)
    segs, start, i, n = [], None, 0, len(d)
    need = int(sr * min_gap)
    while i < n:
        if z[i]:
            j = i
            while j < n and z[j]:
                j += 1
            if j - i >= need:
                if start is not None:
                    segs.append((start, i))
                    start = None
                i = j
                continue
            i = j
        else:
            if start is None:
                start = i
            i += 1
    if start is not None:
        segs.append((start, n))
    return segs


def mode_clips(model, clip_dir: Path) -> None:
    man = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, walls = [], []
    for it in man["items"]:
        d, sr = sf.read(clip_dir / it["file"], dtype="float32")
        a = to16k(d, sr)
        t = time.perf_counter()
        r = recognize(model, a)
        walls.append((time.perf_counter() - t, len(a) / 16000))
        rows.append({**it, **r})
    (clip_dir / "sensevoice.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    conds = list(dict.fromkeys(r["injected"] for r in rows))
    emos = sorted({r["emotion"] for r in rows})
    for kind in ("文", "フィラー"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        print(f"\n=== {kind}（{len(sub)} クリップ）: 入れた感情 → SenseVoice が聞き取った感情 ===")
        print(f"{'入れた':<5}" + "".join(f"{e:>10}" for e in emos) + f"{'笑い声検出':>12}")
        for c in conds:
            cs = [r for r in sub if r["injected"] == c]
            cnt = collections.Counter(r["emotion"] for r in cs)
            laugh = sum(1 for r in cs if "Laughter" in r["events"])
            print(f"{c:<5}" + "".join(f"{cnt.get(e, 0):>10}" for e in emos) + f"{laugh:>9}/{len(cs)}")
    tot_wall = sum(w for w, _ in walls)
    tot_audio = sum(a for _, a in walls)
    print(f"\nSenseVoice 処理時間: 音声 {tot_audio:.1f}秒 を {tot_wall:.2f}秒（CPU）"
          f" / 1クリップ中央値 {sorted(w for w, _ in walls)[len(walls)//2]*1000:.0f}ms")


def mode_wav(model, wav: Path, transcript: Path) -> None:
    d, sr = sf.read(wav, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    segs = split_on_zero_gaps(d, sr)
    lines = [ln for ln in transcript.read_text(encoding="utf-8").splitlines()
             if re.match(r"^\d\d:\d\d  ", ln)]
    print(f"{wav.name}: 区切り {len(segs)} / 台本 {len(lines)} 行"
          + ("（一致）" if len(segs) == len(lines) else "（**不一致**：対応づけは先頭から順に行う）"))
    rows = []
    for k, (s, e) in enumerate(segs):
        r = recognize(model, to16k(d[s:e], sr))
        ln = lines[k] if k < len(lines) else ""
        m = re.search(r"→ (\S+) (\d+)% ", ln)
        injected = m.group(1) if m else "なし"
        rows.append({"t": s / sr, "line": ln, "injected": injected, **r})
    out = wav.with_suffix(".sensevoice.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    cnt = collections.Counter(r["emotion"] for r in rows)
    laugh = sum(1 for r in rows if "Laughter" in r["events"])
    print(f"  聞き取った感情: " + " / ".join(f"{k} {v}" for k, v in cnt.most_common())
          + f"  / 笑い声 {laugh}")
    inj = [r for r in rows if r["injected"] != "なし"]
    if inj:
        print(f"  感情を入れた発話 {len(inj)} 件の判定:")
        for r in inj:
            print(f"    {r['line'][:40]:<42} 入れた={r['injected']:<3} → {r['emotion']} {r['events']}")
    print(f"  詳細: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path)
    ap.add_argument("--wav", type=Path, nargs="*")
    ap.add_argument("--transcript", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    t = time.perf_counter()
    model = load_model(args.device)
    print(f"SenseVoice 読み込み {time.perf_counter() - t:.1f}s（{args.device}）")
    if args.clips:
        mode_clips(model, args.clips)
    for w in args.wav or []:
        mode_wav(model, w, args.transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
