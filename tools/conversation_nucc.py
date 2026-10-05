#!/usr/bin/env python3
"""名大会話コーパスの1つの会話を、途中で切らずに約5分つなげて読み上げる。

出力（リポジトリ外）:
  会話_A_感情なし.wav … 感情注釈なし
  会話_B_感情あり.wav … 各発話の感情を推定して Irodori の絵文字注釈を付けたもの
  transcript.txt     … 時刻つきの台本（話者・声・発話・推定した感情）
A と B は同じ発話を同じ声・同じ seed・同じ長さで合成しているので、時刻が揃う。

**限界（聴く前に読む）**
  - コーパスは文字起こしのみで時刻が無い。間の長さ・相槌の重なり方は推定値。
    割り込み相槌（（うん））は重ねずに順番に鳴らす。
  - 感情は文字から推定（Qwen3-4B・補正なし）。本番予定の音声からの推定（SenseVoice）ではない。
  - ＜笑い＞だけの発話は、ユーザ指定の笑いの表（あはは/ふふ/ふふふ/はは）から選んで読み上げる。
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_nucc import load  # noqa: E402
from listen_nucc import (  # noqa: E402
    BAD, BC_GAP, EMOTION_EMOJI, SAME_GAP, TURN_GAP, VOICES_F, VOICES_M, VOICES_X,
    EmotionClassifier, clean, segments, speech_chars, trim,
)

# ＜笑い＞だけの発話に使う（ユーザ指定の笑いの表から）
LAUGHS = ["あはは", "ふふ", "ふふふ", "はは"]


def pick_window(convs, min_chars: int):
    """伏せ字・録音中断を含まない連続区間から、5分ぶんの会話を1つ選ぶ。

    優先: 話者が2人 > 笑いと「うん」系が多い。
    """
    best = None
    for fname, turns in convs:
        run: list[int] = []

        def consider(run):
            nonlocal best
            head, chars = [], 0
            for k in run:
                head.append(k)
                chars += speech_chars(turns[k]["raw"])
                if chars >= min_chars:
                    break
            if chars < min_chars:
                return
            spk = {turns[k]["spk"] for k in head}
            lively = sum(turns[k]["raw"].count("＜笑") for k in head) \
                + sum(len(re.findall("うん", turns[k]["raw"])) for k in head)
            score = (len(spk) == 2, lively)
            if best is None or score > best[0]:
                best = (score, fname, turns, head)

        for k, t in enumerate(turns):
            if BAD.search(t["raw"]):
                consider(run)
                run = []
            else:
                run.append(k)
        consider(run)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/nucc/nucc"))
    ap.add_argument("--out", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/nucc_conv"))
    ap.add_argument("--irodori-repo", type=Path, default=REPO.parent / "irodori-bench" / "Irodori-TTS")
    ap.add_argument("--emotion-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--target-sec", type=float, default=300.0)
    ap.add_argument("--min-chars", type=int, default=1600, help="区間選びの目安（5分に足りる文字数）")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    convs = load(args.corpus)
    best = pick_window(convs, args.min_chars)
    if best is None:
        raise SystemExit("条件を満たす区間が見つからない")
    (two, lively), fname, turns, head = best
    print(f"会話: {fname}  ターン {head[0]}〜{head[-1]}  話者2人={two}  笑い+うん={lively}")

    # 発話の並び（ターン番号つき）。＜笑い＞だけの発話は笑いの表から読みを決めておく
    items = []
    for k in head:
        for s, x, is_bc in segments(turns, k):
            text, laughed = clean(x)
            laugh_only = False
            if not text and laughed:
                text, laugh_only = rng.choice(LAUGHS), True
            if text:
                items.append({"turn": k, "spk": s, "text": text, "bc": is_bc,
                              "laughed": laughed, "laugh_only": laugh_only})

    # 声（会話全体で固定）
    pools = {"F": list(VOICES_F), "M": list(VOICES_M), "X": list(VOICES_X)}
    vmap = {}
    for it in items:
        s = it["spk"]
        if s not in vmap:
            key = s[0] if s[0] in ("F", "M") else "X"
            src = pools[key] or pools["X"] or VOICES_X
            vmap[s] = src.pop(0) if src else VOICES_X[len(vmap) % len(VOICES_X)]
    print("声:", vmap)

    if str(args.irodori_repo) not in sys.path:
        sys.path.insert(0, str(args.irodori_repo))
    from irodori_tts.inference_runtime import (
        RuntimeKey, SamplingRequest, default_runtime_device,
        download_hf_checkpoint, get_cached_runtime,
    )
    rt, _ = get_cached_runtime(RuntimeKey(
        checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
        model_device=default_runtime_device(), model_precision="bf16", codec_device="cuda"))
    vdir = REPO / "voice_candidates"

    def synth(it, extra_tag: str, seconds):
        tag = "🤭" if it["laughed"] else ""
        if extra_tag and extra_tag not in tag:
            tag += extra_tag
        r = rt.synthesize(SamplingRequest(
            text=it["text"] + tag, ref_wavs=[str(vdir / f"{vmap[it['spk']]}.wav")], no_ref=False,
            seconds=seconds, num_steps=args.steps, t_schedule_mode="sway", sway_coeff=-1.0, seed=0))
        raw = r.audio.detach().cpu().numpy().astype("float32").ravel()
        d = trim(raw, r.sample_rate)
        d = d / (np.sqrt(np.mean(d ** 2)) + 1e-9) * 0.08
        return d, r.sample_rate, len(raw) / r.sample_rate

    def gap_before(prev, it):
        if prev is None:
            return 0.0
        if it["bc"] or prev["bc"]:
            return BC_GAP
        return SAME_GAP if it["spk"] == prev["spk"] else TURN_GAP

    # ---- A: 感情なし。5分を超えたら、そのターンを言い切ったところで止める ----
    audio_a, sr, total, prev, used = [], 48000, 0.0, None, []
    for it in items:
        if total >= args.target_sec and prev is not None and it["turn"] != prev["turn"]:
            break
        d, sr, raw_sec = synth(it, "", None)
        g = gap_before(prev, it)
        if g:
            audio_a.append(np.zeros(int(sr * g), dtype="float32"))
            total += g
        it["t"] = total
        it["len"] = raw_sec
        audio_a.append(d)
        total += len(d) / sr
        used.append(it)
        prev = it
        if len(used) % 20 == 0:
            print(f"  A: {len(used)} 発話 / {total:.0f}秒")
    wav_a = np.clip(np.concatenate(audio_a), -1, 1)
    print(f"A 完了: {len(used)} 発話 / {len(wav_a)/sr:.1f}秒")

    # ---- 感情推定（直前の発話を文脈に）----
    emo = EmotionClassifier(args.emotion_model)
    prev_text = ""
    for it in used:
        it["emo"], it["p"] = emo(prev_text, it["text"])
        prev_text = it["text"]

    # ---- B: 感情あり（各発話を A と同じ長さで合成するので時刻が揃う）----
    audio_b, prev = [], None
    for it in used:
        d, _, _ = synth(it, EMOTION_EMOJI[it["emo"]], it["len"])
        g = gap_before(prev, it)
        if g:
            audio_b.append(np.zeros(int(sr * g), dtype="float32"))
        audio_b.append(d)
        prev = it
    wav_b = np.clip(np.concatenate(audio_b), -1, 1)
    print(f"B 完了: {len(wav_b)/sr:.1f}秒")

    args.out.mkdir(parents=True, exist_ok=True)
    sf.write(args.out / "会話_A_感情なし.wav", wav_a, sr)
    sf.write(args.out / "会話_B_感情あり.wav", wav_b, sr)

    counts = {}
    for it in used:
        counts[it["emo"]] = counts.get(it["emo"], 0) + 1
    lines = [
        f"名大会話コーパス {fname}（ターン {head[0]}〜{used[-1]['turn']}）を約5分つなげて読み上げ",
        f"A 感情なし {len(wav_a)/sr:.0f}秒 / B 感情あり {len(wav_b)/sr:.0f}秒",
        "声: " + " / ".join(f"{s}={v}" for s, v in vmap.items()),
        "推定した感情の内訳: " + " / ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])),
        "絵文字（仮置き）: " + " / ".join(f"{k}={v or 'なし'}" for k, v in EMOTION_EMOJI.items()),
        "※ 時刻は A 基準（B は各発話の長さを A に揃えているのでほぼ同じ位置）",
        "※ 〔 〕は割り込み相槌（重ねずに順番に再生）。《笑》はコーパスで笑っていた発話。",
        "※ 間の長さはコーパスに無いので推定値。感情は文字から推定（Qwen3-4B）。",
        "",
    ]
    for it in used:
        m, s = divmod(int(it["t"]), 60)
        body = f"〔{it['text']}〕" if it["bc"] else it["text"]
        mark = "《笑》" if it["laughed"] else ""
        note = "（＜笑い＞のみの発話を笑いの表から読み上げ）" if it["laugh_only"] else ""
        emo_s = "" if it["emo"] == "中立" else f"    → {it['emo']} {it['p']*100:.0f}% {EMOTION_EMOJI[it['emo']]}"
        lines.append(f"{m:02d}:{s:02d}  {it['spk']}  {body}{mark}{note}{emo_s}")
    (args.out / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
