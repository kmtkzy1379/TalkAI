#!/usr/bin/env python3
"""gpt-6-luna の感情判定を1件ずつ読める形に書き出す（API 不要・`luna_emotion.json` を並べ替えるだけ）。

ユーザ 2026-09-26「会話に対してどんな感情分類したのか確認したい」。
出力 `corpus/jvnv_eval/感情判定_luna.txt`（リポジトリ外・公開しない。名大会話コーパスは CC BY-NC-ND）。
並び: 相手が笑って返した会話（直前の会話つき）→ JVNV の6感情ごと。
各行に Luna の判定（話者の感情6・話者の感情5・イブが返す感情＋確信度）と、同じ素材の Jev / FeedbackLLM を並べる。
ラベル（喜怒哀楽笑い）は定義途中の現案。正解は JVNV の話者の6感情だけ（イブが返す感情に正解は無い）。

実行: & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\luna_emotion_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jvnv_eval import GOLD6, OUT, rd  # noqa: E402


def lab(d: dict, key: str) -> str:
    v = (d or {}).get(key) or {}
    if not v:
        return "-"
    return f"{v.get('label', '-')}({float(v.get('confidence') or 0):.2f})"


def main() -> int:
    items = [it for it in rd("items.json") if it["group"] in ("funny", "recog")]
    tx = rd("text.json")
    luna = json.loads((OUT / "luna_emotion.json").read_text(encoding="utf-8")).get("gpt-6-luna/none", {})
    L = ["gpt-6-luna の感情判定（reasoning_effort=none・FeedbackLLM と同じ判定プロンプト）",
         "各項目: 相手の発話（文字起こし） / Luna: 話者6・話者5・イブが返す感情（確信度＝自己申告）",
         "        比較: Jev（確率）・FeedbackLLM gpt-5.4（自己申告）",
         "ラベルは定義途中の現案。正解は JVNV の話者の6感情だけ。", ""]
    L.append("=" * 72)
    L.append("■ 相手が笑って返した会話（名大会話コーパス・直前の会話つき・40件）")
    L.append("  ＝人間の聞き手が実際に笑った所。イブがここで『笑い』を返せるか")
    L.append("=" * 72)
    for it in [x for x in items if x["group"] == "funny"]:
        r = tx.get(it["id"], {})
        lu = luna.get(it["id"], {})
        L.append("")
        for c in (it.get("context") or [])[-3:]:
            L.append(f"    {c}")
        L.append(f"  ▶ {r.get('stt', '')}")
        L.append(f"    Luna: イブ={lab(lu, 'eve5')}  話者={lab(lu, 'user5')}"
                 f" | Jev: イブ={lab(r.get('jev'), 'eve5')} | FB: イブ={lab(r.get('fb'), 'eve5')}")
    for gold in ["anger", "disgust", "fear", "happy", "sad", "surprise"]:
        grp = [x for x in items if x["group"] == "recog" and x["gold"] == gold]
        ok = sum(1 for x in grp if (luna.get(x["id"], {}).get("user6") or {}).get("label") == GOLD6[gold])
        L += ["", "=" * 72, f"■ JVNV・話者の感情＝{GOLD6[gold]}（{len(grp)}件・Luna の話者6感情の正解 {ok}/{len(grp)}）", "=" * 72]
        for it in grp:
            r = tx.get(it["id"], {})
            lu = luna.get(it["id"], {})
            hit = "○" if (lu.get("user6") or {}).get("label") == GOLD6[gold] else "×"
            L.append(f"  {r.get('stt', '')}")
            L.append(f"    Luna: 話者6={lab(lu, 'user6')}{hit} 話者5={lab(lu, 'user5')} イブ={lab(lu, 'eve5')}"
                     f" | Jev: イブ={lab(r.get('jev'), 'eve5')} | FB: イブ={lab(r.get('fb'), 'eve5')}")
    out = OUT / "感情判定_luna.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(L)} 行 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
