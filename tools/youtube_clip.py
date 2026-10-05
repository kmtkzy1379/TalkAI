#!/usr/bin/env python3
"""YouTube の日常会話を5分切り出し、音声＋時刻つき文字起こしにする（SenseVoice の venv で実行）。

**取得はユーザの判断による**（YouTube の規約はダウンロードを認めていない旨は伝達済み）。
出力はリポジトリの外に置き、手元での分析だけに使う。公開・再配布しない。

出力（動画ごと）: C:/Users/user/Desktop/corpus/youtube/<id>/
  会話_5分.wav      … 聴く用（元の音質）
  transcript.txt   … 発話ごとの 開始–終了時刻 / SenseVoice の感情・音響イベント / 文字起こし
  gaps.txt         … 発話と発話の間の長さ（実測）
  *.ja.vtt         … YouTube の字幕（あれば）
区切りは EVE と同じ Silero VAD。**話者の区別はしていない**（同じ人の言い淀みと
話者交代の間が混ざっている）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensevoice_check import EMO_TAGS, TAG, load_model  # noqa: E402

SILERO = Path(os.path.expanduser("~")) / ".cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit"


def download(url: str, out: Path, start: float, dur: float) -> tuple[Path, dict]:
    import yt_dlp
    from yt_dlp.utils import download_range_func
    out.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out / "src.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, start + dur)]),
        "force_keyframes_at_cuts": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": ["ja"], "subtitlesformat": "vtt",
        "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
    wav = out / "src.wav"
    final = out / "会話_5分.wav"
    if final.exists():
        final.unlink()
    wav.rename(final)
    meta = {k: info.get(k) for k in ("id", "title", "channel", "duration", "webpage_url")}
    meta["clip"] = [start, start + dur]
    (out / "info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return final, meta


def to16k_mono(path: Path) -> np.ndarray:
    # ffmpeg で 16kHz モノラルに（SenseVoice と Silero の入力形式）
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="float32").copy()


def vad_segments(a: np.ndarray, thr=0.5, min_sil=0.30, min_len=0.20, max_len=15.0, pad=0.10):
    import torch
    m = torch.jit.load(str(SILERO)).eval()
    sr, win = 16000, 512
    probs = []
    with torch.inference_mode():
        for i in range(0, len(a) - win, win):
            probs.append(m(torch.from_numpy(a[i:i + win]), sr).item())
    segs, s, sil = [], None, 0
    for k, p in enumerate(probs):
        t = k * win
        if p > thr:
            if s is None:
                s = t
            sil = 0
        elif s is not None:
            sil += win
            if sil >= min_sil * sr:
                segs.append((s, t - sil + win))
                s, sil = None, 0
    if s is not None:
        segs.append((s, len(probs) * win))
    out = []
    for s, e in segs:  # 長すぎる区間は分割・短すぎる区間は捨てる・前後に余白
        while e - s > max_len * sr:
            out.append((s, s + int(max_len * sr)))
            s += int(max_len * sr)
        if e - s >= min_len * sr:
            out.append((max(0, s - int(pad * sr)), min(len(a), e + int(pad * sr))))
    return out


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/youtube"))
    ap.add_argument("--start", type=float, default=60.0, help="切り出し開始（秒）。冒頭の挨拶を避ける")
    ap.add_argument("--dur", type=float, default=300.0)
    args = ap.parse_args()

    model = load_model("cpu")
    for url in args.urls:
        vid = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        vid = vid.group(1) if vid else url[-11:]
        out = args.out / vid
        wav, meta = download(url, out, args.start, args.dur)
        a = to16k_mono(wav)
        segs = vad_segments(a)
        t0 = time.perf_counter()
        rows = []
        for s, e in segs:
            r = model.generate(input=a[s:e], cache={}, language="ja", use_itn=True, batch_size_s=60)
            raw = r[0]["text"] if r else ""
            tags = TAG.findall(raw)
            emo = next((x for x in tags if x in EMO_TAGS), "?")
            ev = [x for x in tags if x not in EMO_TAGS and x not in ("ja", "woitn", "withitn")]
            rows.append({"start": s / 16000, "end": e / 16000, "emotion": emo, "events": ev,
                         "text": TAG.sub("", raw).strip()})
        took = time.perf_counter() - t0

        emo_cnt: dict[str, int] = {}
        for r in rows:
            emo_cnt[r["emotion"]] = emo_cnt.get(r["emotion"], 0) + 1
        laugh = sum(1 for r in rows if "Laughter" in r["events"])
        gaps = [rows[i + 1]["start"] - rows[i]["end"] for i in range(len(rows) - 1)]
        gaps_pos = sorted(g for g in gaps if g > 0)

        lines = [
            f"{meta['title']}",
            f"{meta['channel']} / {meta['webpage_url']}",
            f"切り出し: {fmt(args.start)} から {args.dur/60:.0f}分（この台本の時刻は切り出しの先頭が 00:00）",
            f"発話区間 {len(rows)} / SenseVoice の感情: "
            + " / ".join(f"{k} {v}" for k, v in sorted(emo_cnt.items(), key=lambda x: -x[1]))
            + f" / 笑い声 {laugh}",
            "※ 話者の区別はしていない。文字起こしは SenseVoice の自動認識（誤りを含む）。",
            "",
        ]
        for r in rows:
            tag = f"[{r['emotion']}]" + "".join(f"[{e}]" for e in r["events"] if e != "Speech")
            lines.append(f"{fmt(r['start'])}–{fmt(r['end'])}  {tag:<22} {r['text']}")
        (out / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")

        if gaps_pos:
            q = lambda p: gaps_pos[min(len(gaps_pos) - 1, int(len(gaps_pos) * p))]
            gl = [f"発話間の無音（区切りは無音0.3秒以上。0.3秒未満の間は1つの発話に含まれる）",
                  f"件数 {len(gaps_pos)} / 中央値 {statistics.median(gaps_pos):.2f}秒 / "
                  f"10%点 {q(0.1):.2f} / 90%点 {q(0.9):.2f} / 最大 {gaps_pos[-1]:.2f}"]
            (out / "gaps.txt").write_text("\n".join(gl), encoding="utf-8")
        print(f"[{vid}] {meta['title'][:40]}  区間{len(rows)}  SenseVoice {took:.1f}s  "
              f"感情 {emo_cnt}  笑い {laugh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
