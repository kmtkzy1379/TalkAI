#!/usr/bin/env python3
"""感情注釈の聞き取り検証用に、同じ文を感情ごとに合成する（Irodori の venv で実行）。

目的: 「Irodori の絵文字注釈で入れた感情が、音声として本当に聞き取れるか」を
SenseVoice（tools/sensevoice_check.py --clips）で確かめるための材料を作る。

内容が中立な文を使うのは、文の意味ではなく **注釈だけ** が感情の手がかりになるようにするため。
感情と絵文字の対応は docs/FILLER_DESIGN.md の仮置き（喜😊 怒😠 哀😭 楽😌 笑🤭）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# 内容が中立な文（感情の手がかりを注釈だけにする）
SENTENCES = [
    "明日は十時に駅前で待ち合わせね。",
    "さっきコンビニに寄ってきたよ。",
    "この本、三章まで読んだところ。",
    "午後から雨が降るみたい。",
    "電車は五分遅れてるって。",
    "次の会議は水曜日だよ。",
    "冷蔵庫に牛乳がまだあったよ。",
    "その資料は机の上に置いておいた。",
]
# ユーザの表のフィラー（短い語でも感情が乗るか）
FILLERS = ["うん", "へー", "あー", "ふーん"]
CONDITIONS = {"なし": "", "喜": "😊", "怒": "😠", "哀": "😭", "楽": "😌", "笑": "🤭"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "probe_emotion")
    ap.add_argument("--irodori-repo", type=Path, default=REPO.parent / "irodori-bench" / "Irodori-TTS")
    ap.add_argument("--voice", default="cand_04")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, str(args.irodori_repo))
    from irodori_tts.inference_runtime import (
        RuntimeKey, SamplingRequest, default_runtime_device,
        download_hf_checkpoint, get_cached_runtime, save_wav,
    )
    rt, _ = get_cached_runtime(RuntimeKey(
        checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
        model_device=default_runtime_device(), model_precision="bf16", codec_device="cuda"))
    ref = str(REPO / "voice_candidates" / f"{args.voice}.wav")
    args.out.mkdir(parents=True, exist_ok=True)

    items = []
    for kind, texts in (("文", SENTENCES), ("フィラー", FILLERS)):
        for ti, text in enumerate(texts):
            for cond, emoji in CONDITIONS.items():
                r = rt.synthesize(SamplingRequest(
                    text=text + emoji, ref_wavs=[ref], no_ref=False,
                    num_steps=args.steps, t_schedule_mode="sway", sway_coeff=-1.0, seed=0))
                name = f"{'s' if kind == '文' else 'f'}{ti:02d}_{cond}.wav"
                save_wav(args.out / name, r.audio, r.sample_rate)
                items.append({"file": name, "kind": kind, "text": text,
                              "injected": cond, "emoji": emoji,
                              "sec": round(r.audio.shape[-1] / r.sample_rate, 2)})
    (args.out / "manifest.json").write_text(
        json.dumps({"voice": args.voice, "items": items}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"{len(items)} クリップ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
