#!/usr/bin/env python3
"""名大会話コーパスの「うん」系の使われ方を、話者ごとに別の声で読み上げて聴く。

目的: うん系がどう使われているか・会話が人間らしいか・感情推定が合っているかを
**ユーザが耳で**確かめる。

各サンプルを2本作る（抜き出し・声・seed はまったく同じ。違いは感情注釈だけ）:
  NN_<種類>_A_素.wav   … 感情注釈なし（コーパスの＜笑い＞だけ 🤭 にする）
  NN_<種類>_B_感情.wav … 各発話の感情を推定し、Irodori の絵文字注釈を付ける

**限界（聴く前に読む）**
  - コーパスは文字起こしのみで**時刻が無い**。ターン間の間・相槌の重なり方は推定値。
    割り込み相槌（（うん））は重ねずに順番に鳴らす。打った人は「直前に話していた別の人」とみなす。
    → 評価できるのは「言葉のやりとり」であって「間の取り方」ではない。
  - **元音声が無いので、感情は文字から推定している**（Qwen3-4B）。
    実運用で予定している音声からの感情推定（SenseVoice）とは入力が違う。
  - 感情→絵文字の対応は**聴き比べ用の仮置き**（EMOTION_EMOJI）。公式の定義に基づく。

**出力はリポジトリの外**（コーパスは CC BY-NC-ND 4.0、TalkAI は MIT で公開）。
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

TURN_GAP = 0.20   # 別の人に交代するときの間（秒）※推定値
SAME_GAP = 0.08   # 同じ人が続けるときの間
BC_GAP = 0.06     # 割り込み相槌の前後の間

# 声の割り当て（voice_candidates の候補から）。F=女性 / M=男性 / それ以外は混在。
# 候補の声の高さは自己相関の概算で、性別の当て方がずれている可能性がある。
VOICES_F = ["cand_01", "cand_05", "cand_07", "cand_03"]
VOICES_M = ["cand_09", "cand_08", "cand_06", "cand_10"]
VOICES_X = ["cand_11", "cand_02"]

# 感情（ユーザ決定: 喜怒哀楽＋笑い）→ 絵文字。**聴き比べ用の仮置き**。
# 公式の定義（EMOJI_ANNOTATIONS.md）:
#   😊 Cheerfully, gladly / 😠 Angry, displeased, sulking / 😭 Sobbing, crying, sadly
#   😌 Relieved, contentedly / 🤭 Chuckle, giggle, suppressed laugh
# 😆(Joyfully) はユーザの耳で「驚き・焦り」に聴こえたので喜には使っていない。
EMOTIONS = ["喜", "怒", "哀", "楽", "笑", "中立"]
EMOTION_EMOJI = {"喜": "😊", "怒": "😠", "哀": "😭", "楽": "😌", "笑": "🤭", "中立": ""}
LETTERS = ["A", "B", "C", "D", "E", "F"]
# few-shot の例文（コーパス以外の自作文。テストへの漏れを避ける）
EMO_SHOTS = [
    ("やった、合格したよ！", "A"),
    ("なんで勝手に使うの、ほんとやめてよ", "B"),
    ("ずっと飼ってた猫が、昨日死んじゃって", "C"),
    ("温泉入って、のんびりしてきたんだ", "D"),
    ("寝ぼけて靴下はいたままお風呂入ってた", "E"),
    ("明日は十時集合ね", "F"),
]

UN = re.compile(r"^(うん[、,]?)+$")
UUN = re.compile(r"^うー+ん$")
BAD = re.compile(r"＊|＜(?!笑)")  # 伏せ字・録音中断などを含む箇所は使わない
PAREN_SPLIT = re.compile(r"（([^（）]{1,20})）")


def segments(turns, i):
    """ターン i を [(speaker, text, is_bc)] に分解する（割り込み相槌を取り出す）。"""
    t = turns[i]
    listener = next((turns[j]["spk"] for j in range(i - 1, -1, -1)
                     if turns[j]["spk"] != t["spk"]), None)
    if listener is None:
        listener = next((turns[j]["spk"] for j in range(i + 1, len(turns))
                         if turns[j]["spk"] != t["spk"]), "Ｘ")
    out, pos = [], 0
    for m in PAREN_SPLIT.finditer(t["raw"]):
        if m.start() > pos:
            out.append((t["spk"], t["raw"][pos:m.start()], False))
        out.append((listener, m.group(1), True))
        pos = m.end()
    if pos < len(t["raw"]):
        out.append((t["spk"], t["raw"][pos:], False))
    return [(s, x.strip("　 "), b) for s, x, b in out if x.strip("　 ")]


def speech_chars(text: str) -> int:
    return len(re.sub(r"＜[^＞]*＞|[、。？！\s　]", "", text))


def category(segs):
    for s, x, b in segs:
        if b and UN.match(x.strip("、。")):
            return "相槌のうん" if len(re.findall("うん", x)) == 1 else "相槌のうんうん"
        if b and UUN.match(x.strip("、。")):
            return "相槌のうーん"
    for s, x, b in segs:
        if not b and re.match(r"^うー+ん", x):
            return "ターン冒頭のうーん"
        if not b and re.match(r"^うん", x):
            return "ターン冒頭のうん"
    return None


def clean(text: str) -> tuple[str, bool]:
    """読み上げ用に整える。笑いの記号は外し、笑ったかどうかを返す。"""
    laughed = "＜笑" in text
    return re.sub(r"＜[^＞]*＞", "", text).strip("　 "), laughed


def trim(d, sr, db=-40.0):
    """前後の無音を削る（間はこちらで入れるため）。"""
    if len(d) == 0:
        return d
    thr = np.max(np.abs(d)) * (10 ** (db / 20))
    idx = np.where(np.abs(d) > thr)[0]
    if len(idx) == 0:
        return d
    pad = int(sr * 0.02)
    return d[max(0, idx[0] - pad): idx[-1] + pad]


class EmotionClassifier:
    """文字から感情を推定する（Qwen3-4B・記号ラベル・few-shot・補正なし）。

    bench_quality.py で確かめた手順と同じ（カテゴリ語をそのまま比べると
    トークン分割の差で壊れるので、1トークンの記号 A〜F で比べる）。
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16).to("cuda").eval()
        self.ids = []
        for l in LETTERS:
            e = self.tok.encode(l, add_special_tokens=False)
            assert len(e) == 1, f"ラベル {l} が1トークンでない"
            self.ids.append(e[0])
        self.system = (
            "会話の中の1つの発話について、話し手の感情を1つ選ぶ。記号1文字で答える。\n"
            + "\n".join(f"{l}: {e}" for l, e in zip(LETTERS, EMOTIONS))
            + "\n直前の発話は文脈として参考にする。はっきりしなければ F（中立）。"
        )
        self.base = self._logits("", "、")

    def _prompt(self, ctx: str, utt: str) -> str:
        msgs = [{"role": "system", "content": self.system}]
        for u, lab in EMO_SHOTS:
            msgs.append({"role": "user", "content": f"直前: （なし）\n発話: {u}\n\n記号:"})
            msgs.append({"role": "assistant", "content": lab})
        msgs.append({"role": "user", "content": f"直前: {ctx or '（なし）'}\n発話: {utt}\n\n記号:"})
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _logits(self, ctx: str, utt: str):
        with self.torch.inference_mode():
            enc = self.tok(self._prompt(ctx, utt), return_tensors="pt").to("cuda")
            lg = self.model(**enc).logits[0, -1]
            return self.torch.stack([lg[i] for i in self.ids]).float()

    def __call__(self, ctx: str, utt: str) -> tuple[str, float]:
        # **補正（事前バイアスの差し引き）はしない**。実測 2026-09-23:
        # 中身の無い発話「、」での生 logits が 中立=44.2 / 他=21〜27 で、差し引くと
        # 中立がほぼ選ばれなくなった（「明日は十時に駅前で。」→ 楽64%、「うん。」→ 笑64%）。
        # 補正なしなら 中立100% / 中立100% で正しい。フィラー選択（bench_quality.py）では
        # 補正が効いたが、感情推定は「中立」が頻繁に正解になるので逆効果。
        v = self._logits(ctx, utt)
        p = self.torch.softmax(v, dim=0)
        k = int(p.argmax())
        return EMOTIONS[k], float(p[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/nucc/nucc"))
    ap.add_argument("--out", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/nucc_listen"))
    ap.add_argument("--irodori-repo", type=Path, default=REPO.parent / "irodori-bench" / "Irodori-TTS")
    ap.add_argument("--emotion-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=34)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    # 長さは文字数では見積もれない（実測: 27文字で7.9秒）。合成した実長で採否を決める
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=5.0)
    ap.add_argument("--max-tries", type=int, default=15, help="カテゴリごとの試行上限")
    args = ap.parse_args()
    random.seed(args.seed)

    # ---- 抜き出し候補を集める（1〜3ターンの窓）----
    convs = load(args.corpus)
    pool: dict[str, list] = {}
    for fname, turns in convs:
        for i in range(1, len(turns)):
            for span in (1, 2, 3):
                if i + span > len(turns):
                    break
                segs = [s for k in range(i, i + span) for s in segments(turns, k)]
                if any(BAD.search(x) for _, x, _ in segs):
                    break
                n = sum(speech_chars(x) for _, x, _ in segs)
                if n > args.max_chars:
                    break
                if n < args.min_chars or len({s for s, _, _ in segs}) < 2:
                    continue
                cat = category(segs)
                if cat:
                    pool.setdefault(cat, []).append((fname, i, segs))
                break
    for v in pool.values():
        random.shuffle(v)
    print("候補数:", {k: len(v) for k, v in pool.items()})

    # ---- モデル ----
    if str(args.irodori_repo) not in sys.path:
        sys.path.insert(0, str(args.irodori_repo))
    from irodori_tts.inference_runtime import (
        RuntimeKey, SamplingRequest, default_runtime_device,
        download_hf_checkpoint, get_cached_runtime,
    )
    rt, _ = get_cached_runtime(RuntimeKey(
        checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small"),
        model_device=default_runtime_device(), model_precision="bf16", codec_device="cuda"))
    emo = EmotionClassifier(args.emotion_model)
    vdir = REPO / "voice_candidates"
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.wav"):
        old.unlink()

    def assign_voices(segs):
        pools = {"F": list(VOICES_F), "M": list(VOICES_M)}
        vmap = {}
        for s, _, _ in segs:
            if s in vmap:
                continue
            key = s[0] if s[0] in pools else "X"
            src = pools.get(key) or VOICES_X
            vmap[s] = src.pop(0) if src else VOICES_X[len(vmap) % len(VOICES_X)]
        return vmap

    def render(segs, vmap, emos=None, lengths=None):
        """emos=None なら素の版。emos[k]=(感情, 確率) なら絵文字を付ける。

        lengths を渡すと各発話をその長さ（秒）で合成する。絵文字を付けると Irodori が
        同じ文を長く合成する（実測: 3.2〜5.0秒 → 5.9〜8.2秒）ので、B版は A版と長さを揃える。
        """
        audio, sr, prev = [], 48000, None
        seg_secs = [None] * len(segs)
        for k, (s, x, is_bc) in enumerate(segs):
            text, laughed = clean(x)
            if not text:
                continue
            tag = "🤭" if laughed else ""          # コーパスの＜笑い＞は両方の版に残す
            if emos is not None:
                e = EMOTION_EMOJI[emos[k][0]]
                if e and e not in tag:
                    tag += e
            r = rt.synthesize(SamplingRequest(
                text=text + tag, ref_wavs=[str(vdir / f"{vmap[s]}.wav")], no_ref=False,
                seconds=(lengths[k] if lengths and lengths[k] else None),
                num_steps=args.steps, t_schedule_mode="sway", sway_coeff=-1.0, seed=0))
            sr = r.sample_rate
            raw = r.audio.detach().cpu().numpy().astype("float32").ravel()
            seg_secs[k] = len(raw) / sr
            d = trim(raw, sr)
            d = d / (np.sqrt(np.mean(d ** 2)) + 1e-9) * 0.08  # 音量を揃える
            if prev is not None:
                gap = BC_GAP if (is_bc or prev[1]) else (SAME_GAP if s == prev[0] else TURN_GAP)
                audio.append(np.zeros(int(sr * gap), dtype="float32"))
            audio.append(d)
            prev = (s, is_bc)
        wav = np.clip(np.concatenate(audio), -1, 1) if audio else np.zeros(1, dtype="float32")
        return wav, sr, seg_secs

    lines = [
        "名大会話コーパス『うん』系の聴き取りサンプル",
        "※ 各番号に A_素（感情注釈なし）と B_感情（推定した感情の絵文字つき）の2本。",
        "   抜き出し・声・seed・各発話の長さは同じで、違いは感情注釈だけ。",
        "※ 間の長さ・相槌の重なり方はコーパスに無いので推定値。〔 〕は割り込み相槌（重ねずに順番に再生）。",
        "※ 感情は文字から推定（元音声が無いため・Qwen3-4B・補正なし）。中立なら B も注釈なし。",
        "   絵文字の割り当ては仮置き:",
        "   " + " / ".join(f"{k}={v or 'なし'}" for k, v in EMOTION_EMOJI.items()),
        "",
    ]
    n = 0
    rejected: dict[str, list] = {}
    for cat in sorted(pool):
        got = tries = 0
        for fname, i, segs in pool[cat]:
            if got >= args.per_category or tries >= args.max_tries:
                break
            tries += 1
            vmap = assign_voices(segs)
            wav_a, sr, lens = render(segs, vmap)
            dur_a = len(wav_a) / sr
            if not (args.min_sec <= dur_a <= args.max_sec):
                rejected.setdefault(cat, []).append(round(dur_a, 1))
                continue
            # 感情推定（直前の発話を文脈に）
            emos, prev_text = [], ""
            for s, x, _ in segs:
                t, _ = clean(x)
                emos.append(emo(prev_text, t) if t else ("中立", 1.0))
                prev_text = t or prev_text
            wav_b, _, _ = render(segs, vmap, emos, lens)
            dur_b = len(wav_b) / sr
            got += 1
            n += 1
            sf.write(args.out / f"{n:02d}_{cat}_A_素.wav", wav_a, sr)
            sf.write(args.out / f"{n:02d}_{cat}_B_感情.wav", wav_b, sr)
            lines.append(f"[{n:02d}] {cat}  素 {dur_a:.1f}秒 / 感情 {dur_b:.1f}秒  ({fname})")
            for (s, x, is_bc), (e, p) in zip(segs, emos):
                t, laughed = clean(x)
                body = f"〔{t}〕" if is_bc else t
                lau = " ＋コーパス上の笑い" if laughed else ""
                em = EMOTION_EMOJI[e]
                lines.append(f"  {s}（{vmap[s]}）: {body or '＜笑い＞'}"
                             f"    → 推定 {e}（{p*100:.0f}%）{em}{lau}")
            lines.append("")
            print(f"  [{n:02d}] {cat}  素 {dur_a:.1f}s / 感情 {dur_b:.1f}s  (試行{tries})")
    print("長さで不採用:", rejected)
    (args.out / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
