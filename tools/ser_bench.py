#!/usr/bin/env python3
"""音声感情分析ライブラリの比較（SenseVoice の venv・CPU で実行）。

目的（ユーザ指定）: 喜怒哀楽＋笑い を音声から分類できるライブラリを見つける。
  音声認識は EVE の既存（gpt-4o-transcribe）を使うので、ここでは感情と笑いだけを見る。

比べるもの:
  e2v_large  emotion2vec+ large（約3億）  9分類
  e2v_base   emotion2vec+ base（約9千万）  9分類
  sensevoice SenseVoice Small（基準。これまでの実測で「不明」ばかり）
  ast        AST（AudioSet 527クラス）… 笑いのクラスの確率だけ使う

2つの検証:
  --jvnv DIR   JVNV（プロ話者が6感情を演じた日本語音声・CC BY-SA 4.0）。**正解つき**で正解率を出す
  --clip WAV   実際の日常会話。正解は無いので、発話ごとの判定を並べてユーザが聴いて確かめる

感情の対応: happy→喜 / angry→怒 / sad→哀 / neutral→中立 / それ以外→その他。
**「楽」に当たる分類はどのモデルにも無い**（喜びとの区別は穏やかさ＝覚醒度の違いで、分類ラベルに無い）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

JVNV_EMO = {"anger": "怒", "disgust": "嫌悪", "fear": "恐れ", "happy": "喜",
            "happiness": "喜", "sad": "哀", "sadness": "哀", "surprise": "驚き"}
TO_JA = {"happy": "喜", "angry": "怒", "sad": "哀", "neutral": "中立",
         "HAPPY": "喜", "ANGRY": "怒", "SAD": "哀", "NEUTRAL": "中立",
         "disgusted": "嫌悪", "DISGUSTED": "嫌悪", "fearful": "恐れ", "FEARFUL": "恐れ",
         "surprised": "驚き", "SURPRISED": "驚き"}
LAUGH_CLASSES = ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter")


def load16k(path: Path) -> np.ndarray:
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="float32").copy()


class Models:
    def __init__(self, names):
        from funasr import AutoModel
        self.m = {}
        self.load_sec = {}
        for n in names:
            t = time.perf_counter()
            if n == "e2v_large":
                self.m[n] = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)
            elif n == "e2v_base":
                self.m[n] = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True)
            elif n == "sensevoice":
                from sensevoice_check import load_model
                self.m[n] = load_model("cpu")
            elif n == "ast":
                from transformers import pipeline
                self.m[n] = pipeline("audio-classification",
                                     model="MIT/ast-finetuned-audioset-10-10-0.4593", top_k=None, device=-1)
            self.load_sec[n] = time.perf_counter() - t

    def emotion(self, name, a):
        """(日本語ラベル, 確率, 生ラベル) を返す。"""
        m = self.m[name]
        if name.startswith("e2v"):
            r = m.generate(a, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
            labs = [l.split("/")[-1] for l in r["labels"]]
            k = int(np.argmax(r["scores"]))
            return TO_JA.get(labs[k], "その他"), float(r["scores"][k]), labs[k]
        if name == "sensevoice":
            from sensevoice_check import recognize
            r = recognize(m, a)
            return TO_JA.get(r["emotion"], "その他"), float("nan"), r["emotion"]
        raise KeyError(name)

    def laugh(self, a) -> float:
        out = self.m["ast"]({"raw": a, "sampling_rate": 16000})
        return max((o["score"] for o in out if o["label"] in LAUGH_CLASSES), default=0.0)


def run_jvnv(models: Models, root: Path, per_emotion: int, seed: int):
    files = []
    for p in root.rglob("*.wav"):
        parts = [x.lower() for x in p.parts]
        emo = next((JVNV_EMO[x] for x in parts if x in JVNV_EMO), None)
        if emo is None:
            m = re.search(r"_(anger|disgust|fear|happy|sad|surprise)", p.stem.lower())
            emo = JVNV_EMO[m.group(1)] if m else None
        if emo:
            files.append((emo, p))
    by = collections.defaultdict(list)
    for e, p in files:
        by[e].append(p)
    rng = random.Random(seed)
    picked = [(e, p) for e in sorted(by) for p in rng.sample(by[e], min(per_emotion, len(by[e])))]
    print(f"JVNV: 全 {len(files)} ファイル / 感情ごと {per_emotion} 件を抽出 → {len(picked)} 件")

    names = [n for n in models.m if n != "ast"]
    res = {n: collections.Counter() for n in names}
    walls = collections.defaultdict(list)
    laughs = collections.defaultdict(list)
    for gold, p in picked:
        a = load16k(p)
        for n in names:
            t = time.perf_counter()
            pred, _, _ = models.emotion(n, a)
            walls[n].append(time.perf_counter() - t)
            res[n][(gold, pred)] += 1
        if "ast" in models.m:
            # 言葉にならない声（nv_label）の区間だけを切り出して笑いの確率を見る。
            # ラベルは種類を区別しないが、喜びの発話ならほぼ笑い声、悲しみならすすり泣きのはず。
            lab = root / "nv_label" / p.parent.parent.parent.name / (p.stem + ".txt")
            if lab.exists():
                for ln in lab.read_text(encoding="utf-8").splitlines():
                    f = ln.split()
                    if len(f) >= 2:
                        s0, s1 = int(float(f[0]) * 16000), int(float(f[1]) * 16000)
                        if s1 - s0 > 1600:
                            laughs[gold].append(models.laugh(a[s0:s1]))

    golds = sorted(by)
    preds = ["喜", "怒", "哀", "中立", "嫌悪", "恐れ", "驚き", "その他"]
    for n in names:
        c = res[n]
        hit = sum(c[(g, g)] for g in golds)
        tot = sum(c.values())
        print(f"\n=== {n}  正解率 {hit}/{tot} ({hit/tot*100:.0f}%)  1件中央値 "
              f"{sorted(walls[n])[len(walls[n])//2]*1000:.0f}ms ===")
        print(f"{'正解＼判定':<6}" + "".join(f"{p:>5}" for p in preds))
        for g in golds:
            print(f"{g:<6}" + "".join(f"{c[(g, p)]:>5}" for p in preds)
                  + f"   ← 正答 {c[(g, g)]}/{sum(c[(g, p)] for p in preds)}")
        # 喜怒哀だけの見方（ユーザが使う3感情）
        for g in ("喜", "怒", "哀"):
            tp = c[(g, g)]
            fp = sum(c[(x, g)] for x in golds if x != g)
            fn = sum(c[(g, p)] for p in preds if p != g)
            pr = tp / (tp + fp) if tp + fp else 0
            rc = tp / (tp + fn) if tp + fn else 0
            print(f"    {g}: 適合率 {pr*100:.0f}% / 再現率 {rc*100:.0f}%")
    if laughs:
        print("\n=== AST 笑いの確率（正解の感情ごとの中央値・JVNV は笑い声を含む発話がある）===")
        for g in golds:
            v = sorted(laughs[g])
            print(f"  {g:<4} 中央値 {v[len(v)//2]:.3f} / 0.3以上 {sum(x >= 0.3 for x in v)}/{len(v)}")


def run_clip(models: Models, wav: Path, out: Path):
    from youtube_clip import vad_segments, fmt
    a = load16k(wav)
    segs = vad_segments(a)
    names = [n for n in models.m if n != "ast"]
    lines = [f"{wav}", "発話ごとの判定（正解は無いので聴いて確かめる）。笑い=AST の笑いクラスの確率",
             f"{'時刻':<15}" + "".join(f"{n:<16}" for n in names) + "笑い", ""]
    cnt = {n: collections.Counter() for n in names}
    laugh_n = 0
    for s, e in segs:
        x = a[s:e]
        cols = []
        for n in names:
            ja, p, raw = models.emotion(n, x)
            cnt[n][ja] += 1
            cols.append(f"{ja}{'' if p != p else f'({p*100:.0f}%)'}")
        lp = models.laugh(x) if "ast" in models.m else 0.0
        laugh_n += lp >= 0.3
        lines.append(f"{fmt(s/16000)}–{fmt(e/16000)}  " + "".join(f"{c:<16}" for c in cols)
                     + (f"{lp:.2f}" + (" ★笑" if lp >= 0.3 else "")))
    head = ["", "集計:"] + [f"  {n}: " + " / ".join(f"{k} {v}" for k, v in cnt[n].most_common()) for n in names]
    head.append(f"  笑い（AST 0.3以上）: {laugh_n}/{len(segs)}")
    out.write_text("\n".join(lines + head), encoding="utf-8")
    print("\n".join(head))
    print(f"詳細: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jvnv", type=Path)
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--models", default="e2v_large,e2v_base,sensevoice,ast")
    ap.add_argument("--per-emotion", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    models = Models(args.models.split(","))
    print("読み込み秒:", {k: round(v, 1) for k, v in models.load_sec.items()})
    if args.jvnv:
        run_jvnv(models, args.jvnv, args.per_emotion, args.seed)
    if args.clip:
        run_clip(models, args.clip, args.clip.with_name("感情判定_比較.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
