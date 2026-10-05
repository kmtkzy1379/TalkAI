#!/usr/bin/env python3
"""画面認識の結果の集計（API 不要）。

  python -m tools.prompt_eval.analyze_vlm
"""
from __future__ import annotations

import re
import statistics as st
from collections import defaultdict

from .common import OUT, dump, load

# 感想・感情・評価の語（事実だけ、の目安。多いほど主観が混ざっている）
SUBJECTIVE = re.compile(r"美しい|綺麗|きれい|かわいい|可愛い|楽しそう|悲しそう|怖そう|迫力|幻想的|印象的|素敵|素晴らしい|"
                        r"不気味|緊迫|感動|切ない|劇的|ドラマチック|雰囲気のある|魅力|見事")


def auc(pairs):
    pos = [s for l, s in pairs if l == 1]
    neg = [s for l, s in pairs if l == 0]
    if not pos or not neg:
        return None
    tot = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return tot / (len(pos) * len(neg))


def main() -> int:
    out = {}
    seq = load(OUT / "vlm_seq.json")
    pairs = defaultdict(list)
    print("[seq] 連続の場面（label: 0=予想どおりの続き / 1=予想外の変化）")
    for key, calls in sorted(seq.items()):
        name, var, rep = key.split("|")
        print(f"  {key}")
        for c in calls:
            print(f"     #{c['call']} label={c['label']} surprise={c['surprise']} notable={c['notable']}"
                  f" | 予想={str(c.get('prediction'))[:36]} | {c['narration'][:50]}")
            if c["label"] is not None and c["surprise"] is not None:
                pairs[var].append((c["label"], c["surprise"]))
    out["seq"] = {}
    for var, ps in sorted(pairs.items()):
        pos = [s for l, s in ps if l == 1]
        neg = [s for l, s in ps if l == 0]
        a = auc(ps)
        out["seq"][var] = {"auc": a, "mean_unexpected": st.mean(pos), "mean_expected": st.mean(neg),
                           "n_unexpected": len(pos), "n_expected": len(neg),
                           "sd_expected": st.pstdev(neg)}
        print(f"  {var}: AUC {a:.3f}（予想外 平均 {st.mean(pos):.1f} ／ 予想どおり 平均 {st.mean(neg):.1f}"
              f"・ばらつき {st.pstdev(neg):.1f}・n={len(pos)}/{len(neg)}）")

    narr = load(OUT / "vlm_narr.json")
    print("\n[narr] 実況の長さ・主観語")
    out["narr"] = {}
    for var in ("0", "U"):
        rs = [r for r in narr.values() if r["variant"] == var and not r.get("error")]
        ls = [len(r["narration"]) for r in rs]
        subj = [bool(SUBJECTIVE.search(r["narration"])) for r in rs]
        multi = [("\n" in (r.get("raw") or "").split("narration", 1)[-1].split("notable", 1)[0].strip()) for r in rs]
        out["narr"][var] = {"median_len": st.median(ls), "subjective": f"{sum(subj)}/{len(subj)}",
                            "narration_broken_lines": f"{sum(multi)}/{len(multi)}"}
        print(f"  {var}: 長さ中央 {st.median(ls):.0f}字 ／ 主観語あり {sum(subj)}/{len(subj)}"
              f" ／ 実況が複数行に割れた {sum(multi)}/{len(multi)}")
    dump(OUT / "vlm_metrics.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
