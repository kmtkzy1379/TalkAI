#!/usr/bin/env python3
"""声が崩れる（発音・読み方がおかしい）のは何のせいかを測る（API 不要・ローカルの Irodori と SenseVoice だけ）。

ユーザ 2026-09-26「同じ文章でも音声おかしくなってる部分が多数ある（発音？読み方？）」への調べ。
同じ文を4通りに合成し、SenseVoice（ローカルの音声認識）で聞き取った文字と元の文を比べる（文字の誤り率＝CER）:
  A 文ごと・絵文字なし（S7 より前の本番）
  B 「、」「。」で区切る・絵文字なし（S7）
  C 区切る＋絵文字・長さそのまま
  D 区切る＋絵文字・長さを絵文字なしにそろえる（今の本番＝match_plain）
seed は 0〜3（本番は 0 固定＝同じ文はいつも同じ音）。
SenseVoice は漢字とかなの書き分けがずれることがある（「今日」と「きょう」等）ので、見るのは**条件どうしの差**。

出力: listen/quality/（クリップ・manifest.json・sensevoice.json・README.txt＝崩れが大きい順の一覧）
（本番と同じ `IrodoriTTS.generate` を通す＝節の終わりの「、」を落とす修正 2026-09-26 込み）
実行（EVE の venv・中で SenseVoice の venv を呼ぶ）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\tone_quality_check.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.config import Config  # noqa: E402
from eve.emotion import load_voice_emoji  # noqa: E402
from eve.filler import audio as fa  # noqa: E402
from eve.response.splitter import JapaneseSentenceSplitter  # noqa: E402
from eve.response.tts_irodori import IrodoriServer, IrodoriTTS, resolve_repo_path  # noqa: E402

OUT = REPO / "listen" / "quality"
SENSEVOICE_PY = Path(r"C:/Users/user/Desktop/irodori-bench/sensevoice/.venv/Scripts/python.exe")
SEEDS = (0, 1, 2, 3)

# 試聴（listen/emotion）と同じ文（感情つき）
AUDITION = {
    "喜": ["え、ほんとに？それすごくうれしい！", "やった、うまくいったんだね。"],
    "怒": ["それはさすがにひどいと思う。", "もう、また同じこと言ってる。"],
    "哀": ["そっか、それはつらかったね。", "なんだか、さみしくなっちゃうな。"],
    "楽": ["今日はのんびりできていいね。", "うん、ゆっくり話そうか。"],
    "笑い": ["それはさすがに笑っちゃうよ。", "ずっと手に持ってたんだね。"],
}
# イブの実際の応答（eve-v2 の会話履歴から・節の区切りの影響を普通の文で見る。絵文字は 楽 のもの）
REAL = [
    "今日はあったかいものにするなら、スープとかお鍋みたいなのもいいかもね。",
    "簡易電卓、画面に出したよ。まずは足し算とかで動くか見てみて。",
    "うん、了解。じゃあこの件はいったんなしにしておくね。",
    "うん、いるね。来てくれてうれしいよ。",
    "電子レンジってさ、仕組みを知ると急にちょっと魔法道具っぽく見えるよね。",
    "前に聞き取れなかったネガシットのこと、まだ少しだけ気になってるよ。",
]


def clauses(text: str) -> list[str]:
    sp = JapaneseSentenceSplitter()
    return sp.feed(text) + sp.flush()


def norm(s: str) -> str:
    """比べる形: 記号・空白・絵文字を落とし、カタカナをひらがなに。"""
    s = re.sub(r"[\s、。！？!?,.・「」『』（）()…〜~]", "", s or "")
    s = "".join(ch for ch in s if not (0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF
                                       or ch == "\ufe0f"))
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in s)


def cer(ref: str, hyp: str) -> float:
    a, b = norm(ref), norm(hyp)
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


async def synth(tts: IrodoriTTS, text: str, seed: int, match: bool) -> np.ndarray:
    """本番と同じ経路（`IrodoriTTS.generate`＝節の終わりの「、」を落とす・match_plain）で、seed と長さそろえだけ変える。"""
    tts.seed, tts.match_emoji_length = seed, match
    wav = await tts.generate(text)
    if wav is None:
        raise RuntimeError(f"合成に失敗: {text}")
    x, _ = fa.read_wav(wav)
    return x


async def make(tts: IrodoriTTS, text: str, emoji: str, cond: str, seed: int) -> np.ndarray:
    sr, gap = 48000, Config.SPEECH_GAP_SEC
    if cond == "A":
        return await synth(tts, text, seed, False)
    pieces = []
    for c in clauses(text):
        t = c + (emoji if cond in ("C", "D") else "")
        pieces.append(await synth(tts, t, seed, cond == "D"))
    seq = []
    for i, x in enumerate(pieces):
        seq.append(x)
        if i < len(pieces) - 1:
            seq.append(fa.silence(gap, sr))
    return np.concatenate(seq)


async def synthesize_all() -> list:
    table = load_voice_emoji(Path(resolve_repo_path(Config.EMOTION_TABLE)))
    cases = [(p, emo, table.get(emo, ""), "試聴") for emo, ps in AUDITION.items() for p in ps]
    cases += [(p, "楽", table.get("楽", ""), "実際の応答") for p in REAL]
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    tts = IrodoriTTS(IrodoriServer())
    items = []
    try:
        await tts.warmup()
        n = 0
        for text, emo, emoji, kind in cases:
            conds = ["A", "B", "C", "D"] if kind == "試聴" else ["A", "B", "D"]
            for cond in conds:
                for seed in SEEDS:
                    n += 1
                    f = f"{n:04d}.wav"
                    x = await make(tts, text, emoji, cond, seed)
                    (OUT / f).write_bytes(fa.to_wav(x, 48000))
                    items.append({"file": f, "kind": "文", "injected": cond, "cond": cond, "seed": seed,
                                  "text": text, "emotion": emo, "emoji": emoji if cond in ("C", "D") else "",
                                  "set": kind})
            print(f"{kind} {emo} {emoji}: {text}")
    finally:
        await tts.close()
    (OUT / "manifest.json").write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


def main() -> int:
    items = asyncio.run(synthesize_all())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([str(SENSEVOICE_PY), str(REPO / "tools" / "sensevoice_check.py"), "--clips", str(OUT)],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return r.returncode
    rows = {x["file"]: x for x in json.loads((OUT / "sensevoice.json").read_text(encoding="utf-8"))}
    for it in items:
        it["heard"] = rows.get(it["file"], {}).get("text", "")
        it["cer"] = cer(it["text"], it["heard"])
    names = {"A": "文ごと・絵文字なし", "B": "区切る・絵文字なし", "C": "区切る＋絵文字・長さそのまま",
             "D": "区切る＋絵文字・長さそろえ（本番）"}
    lines = ["声の崩れの調べ（2026-09-26・Irodori steps %d・SenseVoice で聞き取って元の文と比べた文字の誤り率 CER）" % Config.IRODORI_STEPS,
             "SenseVoice は漢字とかなの書き分けでも誤りに数えるので、見るのは条件どうしの差。", ""]
    for kind in ("試聴", "実際の応答"):
        lines.append(f"■ {kind}の文")
        for cond in ("A", "B", "C", "D"):
            v = [it["cer"] for it in items if it["set"] == kind and it["cond"] == cond]
            if not v:
                continue
            v0 = [it["cer"] for it in items if it["set"] == kind and it["cond"] == cond and it["seed"] == 0]
            bad = sum(1 for x in v if x >= 0.2)
            lines.append(f"  {cond} {names[cond]:<22} 平均 {statistics.mean(v):.3f} / seed0 {statistics.mean(v0):.3f}"
                         f" / 崩れ（CER 0.2 以上） {bad}/{len(v)}")
        lines.append("")
    # 感情ごと（試聴の文・D と B の差）
    lines.append("■ 絵文字ごと（試聴の文・seed 0〜3 の平均）: 区切る・なし(B) → 区切る＋絵文字・長さそろえ(D)")
    for emo in AUDITION:
        b = [it["cer"] for it in items if it["set"] == "試聴" and it["emotion"] == emo and it["cond"] == "B"]
        d = [it["cer"] for it in items if it["set"] == "試聴" and it["emotion"] == emo and it["cond"] == "D"]
        e = next((it["emoji"] for it in items if it["emotion"] == emo and it["cond"] == "D"), "")
        lines.append(f"  {emo} {e}: {statistics.mean(b):.3f} → {statistics.mean(d):.3f}")
    lines += ["", "■ 崩れが大きい順（上位20・ファイル名で聴ける）"]
    for it in sorted(items, key=lambda x: -x["cer"])[:20]:
        lines.append(f"  {it['file']} {it['cond']} seed{it['seed']} CER {it['cer']:.2f}  {it['text']}{it['emoji']}"
                     f"\n        聞き取り: {it['heard']}")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "result.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:22]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
