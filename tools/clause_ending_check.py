#!/usr/bin/env python3
"""「、」で終わる短い節を1つだけ合成すると、Irodori が続きをでっち上げる（存在しない音を足す）問題の調べ（API 不要）。

ユーザ 2026-09-26「同じ文章でも音声おかしくなってる部分が多数ある（発音？読み方？）」→ 試聴の聞き取り（SenseVoice）で
「え、」「もう、」「うん、」「そっか、」の直後に「そん姉ちゃんも」「クレチ恥なよ」等の**言っていない音**が入っていた。
「、」で終わる＝まだ続く文だと受け取り、長い時間（「そっか、」だけで約1.7秒）を予測して、その時間を作り話で埋めていると考え、
TTS に渡す文字の終わり方を変えて比べる:
  V0 そのまま（「そっか、」＝今の本番）/ V1 「、」を取る（「そっか」）/ V2 「、」を「。」に（「そっか。」）
各 seed 0〜3・絵文字なしと 😊（長さそろえ）。SenseVoice で聞き取り、元の語と比べる（CER）・長さ・作り話の数。
あわせて、試聴と同じ文を V0 / V1 でつないだ聴き比べ（listen/clause_ending/）を作る。

実行（EVE の venv・中で SenseVoice の venv を呼ぶ）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\clause_ending_check.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from eve.config import Config  # noqa: E402
from eve.filler import audio as fa  # noqa: E402
from eve.response.splitter import JapaneseSentenceSplitter  # noqa: E402
from eve.response.tts_irodori import IrodoriServer, IrodoriTTS  # noqa: E402
from tone_quality_check import AUDITION, SENSEVOICE_PY, cer  # noqa: E402

OUT = REPO / "listen" / "clause_ending"
SEEDS = (0, 1, 2, 3)
SHORT = ["え、", "もう、", "うん、", "そっか、", "なんだか、", "やった、", "おっけー、", "ごめん、", "そういえば、",
         "見た感じ、", "今日は、", "あのさ、"]
MEDIUM = ["前に雨の日に軽く遊べるSteamのゲームの話してたの、", "今日はあったかいものにするなら、",
          "画面がGitHubに切り替わったみたいだけど、", "電子レンジってさ、"]
VARIANTS = {"V0": lambda t: t, "V1": lambda t: t[:-1] if t.endswith("、") else t,
            "V2": lambda t: (t[:-1] + "。") if t.endswith("、") else t}
NAMES = {"V0": "そのまま（、付き）", "V1": "「、」を取る", "V2": "「、」→「。」"}


async def synth(tts: IrodoriTTS, text: str, seed: int) -> np.ndarray:
    s = await tts._get_session()
    async with s.post(f"{tts.server.url}/synthesize",
                      json={"text": text, "seed": seed, "num_steps": tts.steps, "match_plain": True}) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {text}")
        x, _ = fa.read_wav(await r.read())
        return x


def clauses(text: str) -> list[str]:
    sp = JapaneseSentenceSplitter()
    return sp.feed(text) + sp.flush()


async def run() -> list:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "clips").mkdir(parents=True)
    tts = IrodoriTTS(IrodoriServer())
    items = []
    sr, gap = 48000, Config.SPEECH_GAP_SEC
    try:
        await tts.warmup()
        n = 0
        for group, texts in (("短い節", SHORT), ("長めの節", MEDIUM)):
            for text in texts:
                for v, fn in VARIANTS.items():
                    for emoji in ("", "😊"):
                        for seed in SEEDS:
                            n += 1
                            x = await synth(tts, fn(text) + emoji, seed)
                            f = f"{n:04d}.wav"
                            (OUT / "clips" / f).write_bytes(fa.to_wav(x, sr))
                            items.append({"file": f, "kind": "文", "injected": v, "variant": v, "text": text,
                                          "group": group, "emoji": emoji, "seed": seed, "sec": len(x) / sr})
                print(f"{group}: {text}")
        # 聴き比べ: 試聴と同じ文を V0 → V1 でつなぐ（seed 0・絵文字なし）
        seq, lines = [], []
        for emo, phrases in AUDITION.items():
            for phrase in phrases:
                for v in ("V0", "V1"):
                    pieces = [await synth(tts, VARIANTS[v](c), 0) for c in clauses(phrase)]
                    for i, x in enumerate(pieces):
                        seq.append(x)
                        if i < len(pieces) - 1:
                            seq.append(fa.silence(gap, sr))
                    seq.append(fa.silence(1.0, sr))
                seq.append(fa.silence(1.5, sr))
                lines.append(f"  {phrase}")
        (OUT / "V0_vs_V1.wav").write_bytes(fa.to_wav(np.concatenate(seq), sr))
        (OUT / "listen_order.txt").write_text("文ごとに「そのまま（、付き）→ 1秒 →「、」を取る」:\n" + "\n".join(lines) + "\n",
                                             encoding="utf-8")
    finally:
        await tts.close()
    (OUT / "clips" / "manifest.json").write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    return items


def main() -> int:
    items = asyncio.run(run())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([str(SENSEVOICE_PY), str(REPO / "tools" / "sensevoice_check.py"), "--clips", str(OUT / "clips")],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return r.returncode
    heard = {x["file"]: x.get("text", "") for x in json.loads((OUT / "clips" / "sensevoice.json").read_text(encoding="utf-8"))}
    for it in items:
        it["heard"] = heard.get(it["file"], "")
        it["cer"] = cer(it["text"], it["heard"])
    lines = ["「、」で終わる節の終わり方の比べ（2026-09-26・Irodori steps %d・長さそろえ込み・SenseVoice で聞き取り）" % Config.IRODORI_STEPS,
             "作り話 = 聞き取った文字が元の語より 50% 以上多い/違う（CER 0.5 以上）", ""]
    for group in ("短い節", "長めの節"):
        for emoji in ("", "😊"):
            lines.append(f"■ {group}・{'絵文字なし' if not emoji else emoji}")
            for v in VARIANTS:
                sub = [it for it in items if it["group"] == group and it["emoji"] == emoji and it["variant"] == v]
                bad = sum(1 for it in sub if it["cer"] >= 0.5)
                lines.append(f"  {v} {NAMES[v]:<14} CER 平均 {statistics.mean(it['cer'] for it in sub):.2f}"
                             f" / 作り話 {bad}/{len(sub)} / 長さ 中央 {statistics.median(it['sec'] for it in sub):.2f}秒")
            lines.append("")
    lines.append("■ 作り話の例（V0・絵文字なし）")
    for it in [x for x in items if x["variant"] == "V0" and not x["emoji"] and x["cer"] >= 0.5][:15]:
        lines.append(f"  {it['text']} seed{it['seed']} → {it['heard']}")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "result.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
