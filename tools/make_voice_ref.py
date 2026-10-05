#!/usr/bin/env python3
"""声の候補を作って選ぶ（バンク生成の前工程）。

なぜ必要か:
  `no_ref=True` で合成すると **seed ごとに話者が変わる**（実測: 同じ語で基本周波数が
  197Hz〜400Hz とばらついた）。バンクの声を揃えるには話者を固定する必要があり、
  Irodori では `ref_wavs`（参照音声によるクローン）がその手段。

使い方:
  1) 候補を生成する
       python tools/make_voice_ref.py --seeds 12
  2) voice_candidates/ を聴いて気に入った seed を選ぶ
  3) その wav を fillers.yaml の voices.<名前>.ref_wavs に書く
       voices:
         eve:
           ref_wavs: ["voice_candidates/cand_07.wav"]
  4) バンクを作り直す
       python tools/build_filler_bank.py --voice eve

参照音声は長いほどクローンが安定する（公式は30秒以上を推奨・短いクリップ複数も可）。
--extend を付けると、選んだ声で追加の参照音声を作って複数クリップにできる。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IRODORI = REPO.parent / "irodori-bench" / "Irodori-TTS"

# 声を判断しやすい文（母音・子音・語尾が一通り出る長さ）。参照音声としても使う。
PROBE = (
    "こんにちは、イブだよ。今日はどんな一日だった？"
    "わたしは画面を見ながら、いろんなことを考えてる。"
    "ねえ、よかったら少し話を聞かせてほしいな。"
)
# --extend 用。選んだ声で追加の参照を作る（クローンの安定度を上げる）
EXTRA = [
    "そうなんだ、それはちょっと大変だったね。無理してない？",
    "へえ、面白いね。もう少し詳しく教えてくれる？",
    "うん、わかった。じゃあそれで進めてみようか。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12, help="生成する候補数")
    ap.add_argument("--out", type=Path, default=REPO / "voice_candidates")
    ap.add_argument("--irodori-repo", type=Path, default=DEFAULT_IRODORI)
    ap.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--caption", default=None,
                    help="声の方向づけ（例: 明るく高めの、距離の近い女性の声）")
    # 候補作りは1回きりなので品質優先で steps を多めに取る（バンク生成とは別）
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--extend", type=Path, default=None,
                    help="この wav を参照に、追加の参照音声を作る（クローン安定化）")
    args = ap.parse_args()

    if str(args.irodori_repo) not in sys.path:
        sys.path.insert(0, str(args.irodori_repo))
    from irodori_tts.inference_runtime import (
        RuntimeKey, SamplingRequest, default_runtime_device,
        download_hf_checkpoint, get_cached_runtime, save_wav,
    )

    runtime, _ = get_cached_runtime(RuntimeKey(
        checkpoint=download_hf_checkpoint(args.checkpoint),
        model_device=default_runtime_device(),
        model_precision="bf16", codec_device="cuda",
    ))
    args.out.mkdir(parents=True, exist_ok=True)

    def synth(text: str, *, seed: int, ref: list[str] | None) -> tuple:
        req = SamplingRequest(
            text=text, no_ref=ref is None, ref_wavs=ref,
            caption=args.caption, num_steps=args.num_steps,
            t_schedule_mode="sway", sway_coeff=-1.0, seed=seed,
        )
        t = time.perf_counter()
        return runtime.synthesize(req), time.perf_counter() - t

    if args.extend:
        # 選んだ声を参照に、追加の参照音声を作る（合計を長くしてクローンを安定させる）
        ref = [str(args.extend)]
        print(f"参照: {args.extend}")
        for i, text in enumerate(EXTRA):
            res, w = synth(text, seed=100 + i, ref=ref)
            p = args.out / f"{args.extend.stem}_ext{i}.wav"
            save_wav(p, res.audio, res.sample_rate)
            print(f"  {p.name}  {res.audio.shape[-1]/res.sample_rate:.1f}s  ({w*1000:.0f}ms)")
        print("\nfillers.yaml の ref_wavs に、元の wav と ext を並べて書くと安定する。")
        return 0

    print(f"候補 {args.seeds} 個を生成（steps={args.num_steps}"
          f"{' / caption あり' if args.caption else ''}）\n")
    print(f"  {'file':<16} {'長さ':>6} {'基本周波数':>10}")
    import numpy as np
    for seed in range(args.seeds):
        res, w = synth(PROBE, seed=seed, ref=None)
        p = args.out / f"cand_{seed:02d}.wav"
        save_wav(p, res.audio, res.sample_rate)
        d = res.audio.detach().cpu().numpy().astype("float32").ravel()
        sr = res.sample_rate
        # 自己相関で基本周波数を概算（声の高さの目安。判断は必ず耳で）
        x = d[: int(sr * 0.8)] - d[: int(sr * 0.8)].mean()
        a = np.correlate(x, x, "full")[len(x) - 1:]
        lo, hi = sr // 400, sr // 80
        f0 = sr / (lo + int(np.argmax(a[lo:hi]))) if hi < len(a) else 0.0
        print(f"  {p.name:<16} {len(d)/sr:>5.1f}s {f0:>9.0f}Hz")

    print(f"\n出力: {args.out}")
    print("聴いて選んだら fillers.yaml の voices に書く:")
    print("  voices:\n    eve:\n      ref_wavs: [\"voice_candidates/cand_07.wav\"]")
    print("その後: python tools/build_filler_bank.py --voice eve --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
