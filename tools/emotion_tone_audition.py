#!/usr/bin/env python3
"""S5 声のトーンの聴き比べ（感情ごとに Irodori の絵文字の候補を**番号つきの別ファイル**で書き出す・API 不要）。

ユーザ 2026-09-26「番号がわからないから何とも言えない。すべての番号で書き出して」→ 候補ごとに1ファイル。
ファイル名の先頭の番号で選べる（例「哀は 10」）。README.txt に 番号 → 感情・絵文字 の一覧。

本番と同じ作り: 文を「、」「。」で区切り（S7）、節ごとに**後ろに絵文字を付けて**、**絵文字なしと同じ長さ**で合成し
（サーバの match_plain・ユーザ 2026-09-26「速度は揃えて感情載せる」）、区切りの間（SPEECH_GAP_SEC）でつなぐ。
1ファイル = その感情の2文（1秒あけて）。話者・steps・seed は .env と同じ。
manifest.json は `tools/sensevoice_check.py --clips listen/emotion` 用。

実行（EVE の venv）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\emotion_tone_audition.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.config import Config  # noqa: E402
from eve.filler import audio as fa  # noqa: E402
from eve.response.splitter import JapaneseSentenceSplitter  # noqa: E402
from eve.response.tts_irodori import IrodoriServer, IrodoriTTS  # noqa: E402

OUT = REPO / "listen" / "emotion"

# 候補は公式の効果（irodori_tts/gradio_emoji_palette.py）から感情に近いもの。(絵文字, 公式の効果の名前)
CANDIDATES = {
    "喜": [("😆", "喜び"), ("😊", "楽しげ")],
    "怒": [("😠", "怒り"), ("😒", "舌打ち"), ("🙄", "呆れ")],
    "哀": [("😭", "泣き声"), ("🥺", "震え声"), ("😟", "心配")],
    "楽": [("😌", "安堵"), ("😊", "楽しげ"), ("🫶", "優しく")],
    "笑い": [("🤭", "笑い"), ("😆", "喜び")],
}
# 感情に合う中身の文（イブの話し方に寄せた短い文・各2つ）
PHRASES = {
    "喜": ["え、ほんとに？それすごくうれしい！", "やった、うまくいったんだね。"],
    "怒": ["それはさすがにひどいと思う。", "もう、また同じこと言ってる。"],
    "哀": ["そっか、それはつらかったね。", "なんだか、さみしくなっちゃうな。"],
    "楽": ["今日はのんびりできていいね。", "うん、ゆっくり話そうか。"],
    "笑い": ["それはさすがに笑っちゃうよ。", "ずっと手に持ってたんだね。"],
}


def clauses(text: str) -> list[str]:
    sp = JapaneseSentenceSplitter()
    return sp.feed(text) + sp.flush()


async def synth(tts: IrodoriTTS, text: str) -> np.ndarray:
    """本番と同じ経路（`IrodoriTTS.generate`＝match_plain つき）。"""
    wav = await tts.generate(text)
    if wav is None:
        raise RuntimeError(f"合成に失敗: {text}")
    x, _ = fa.read_wav(wav)
    return x


async def main() -> int:
    # 前の書き出し（1本に並べたもの・通し番号の無いもの）は紛らわしいので消してから作る
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    gap = Config.SPEECH_GAP_SEC
    sr = 48000
    tts = IrodoriTTS(IrodoriServer())
    try:
        await tts.warmup()
        num, items, lines = 0, [], []
        for emo, cands in CANDIDATES.items():
            lines.append(f"■ {emo}（「{PHRASES[emo][0]}」「{PHRASES[emo][1]}」）")
            for emoji, name in [("", "なし")] + list(cands):
                num += 1
                seq = []
                for p_i, phrase in enumerate(PHRASES[emo]):
                    pieces = [await synth(tts, c + emoji) for c in clauses(phrase)]
                    for i, x in enumerate(pieces):
                        seq.append(x)
                        if i < len(pieces) - 1:
                            seq.append(fa.silence(gap, sr))
                    if p_i == 0:
                        seq.append(fa.silence(1.0, sr))
                f = f"{num:02d}_{emo}_{name}.wav"
                (OUT / f).write_bytes(fa.to_wav(np.concatenate(seq), sr))
                items.append({"file": f, "kind": "文", "injected": f"{emo}:{emoji or 'なし'}",
                              "number": num, "emotion": emo, "emoji": emoji})
                lines.append(f"  {num:2d}  {emoji or '  '}  {name}{'（絵文字なし）' if not emoji else ''}")
                print(f"{num:02d} {emo} {emoji or 'なし'}")
            lines.append("")
        (OUT / "manifest.json").write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
        head = [
            "声のトーン（S5）の聴き比べ（2026-09-26・Irodori・話者 cand_04・steps 6・seed 0）",
            "",
            "ファイル名の先頭の番号で選んでください（例「喜は 03・怒は 05・哀は 10・楽は 13・笑いは 17」）。",
            "1ファイル = その感情の2文（1秒あけて）。どれも本番と同じ作り:",
            "  「、」「。」で区切り、節ごとに後ろに絵文字を付けて、**絵文字なしと同じ長さ**で合成（速さをそろえた）、",
            "  区切りの間 %.2f秒でつなぐ。各感情の最初の番号が「絵文字なし」（比べる基準）。" % gap,
            "",
            "番号 絵文字 公式の効果",
        ]
        tail = ["決めたら emotion.yaml の voice_emoji に書く（なし＝null）。フィラーの声（感情ごとのバンク）も同じ絵文字で作る。"]
        (OUT / "README.txt").write_text("\n".join(head + lines + tail) + "\n", encoding="utf-8")
        print("出力:", OUT)
    finally:
        await tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
