#!/usr/bin/env python3
"""AI の審査員に頼らない、言い方の数字（質問で返す・謝る・世話を焼く・長さ）。API 不要。

  python -m tools.prompt_eval.neuro_style            # 記録済みのイブの応答（B・Y・H の雑談）と Neuro-sama の元データ
Neuro-sama の元データは英語なので、英語の目印で数える（日本語訳の数字は neuro_eval が別に出す）。
"""
from __future__ import annotations

import csv
import re
import statistics as st
from collections import defaultdict

from .analyze_resp import spoken
from .common import OUT, load

NEURO_CSV = OUT / "neuro" / "Manually_dataset.csv"

# 日本語（イブ）の目印
JA = {
    "質問で終わる": re.compile(r"[？?]\s*$"),
    "謝る": re.compile(r"ごめん|すまん|申し訳|すみません"),
    # 世話を焼く・カウンセラー口調（AI臭さの定番）
    "気遣い・助言": re.compile(r"無理(しないで|せず|しなくて)|休んで|大丈夫[？?]|話したくなったら|いつでも|"
                          r"してみて|してね|した方がいい|したほうがいい|気をつけて|相談して"),
    "相手に説明を求める": re.compile(r"教えて|何があった|どうした|どんな|どこが|どの"),
}
# 英語（Neuro-sama）の目印
EN = {
    "質問で終わる": re.compile(r"\?\s*[\"')\]]*\s*$"),
    "謝る": re.compile(r"\bsorry\b|\bapolog", re.I),
    "気遣い・助言": re.compile(r"\btake care\b|\brest\b|\byou should\b|\btry to\b|\bmake sure\b|are you ok|\bokay\?|"
                          r"\bhere for you\b|\bany ?time\b", re.I),
    "相手に説明を求める": re.compile(r"\btell me\b|what happened|\bwhich\b.*\?|\bhow was\b|\bwhat kind\b", re.I),
}


def _rate(xs) -> str:
    xs = list(xs)
    return f"{sum(xs)}/{len(xs)}（{sum(xs) / len(xs):.0%}）" if xs else "-"


def neuro_rows() -> list[dict]:
    with open(NEURO_CSV, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("output") or "").strip()]


def main() -> int:
    from . import scenarios as SCN

    rows = neuro_rows()
    outs = [r["output"].strip() for r in rows]
    print(f"Neuro-sama（英語・{len(outs)}件・長さの中央 {st.median(len(o.split()) for o in outs)}語）")
    for k, p in EN.items():
        print(f"  {k}: {_rate(bool(p.search(o)) for o in outs)}")

    res = load(OUT / "resp.json")
    scs = {s["id"]: s for s in SCN.ALL}
    groups = defaultdict(list)
    for r in res.values():
        sc = scs.get(r["id"])
        if not sc:
            continue
        if r["item"] in ("B", "Y"):
            groups[f"B+Y {r['variant']}"].append(spoken(r))
        elif r["item"] == "H" and sc.get("group") in ("chat", "known"):
            groups[f"H雑談 {r['variant']}"].append(spoken(r))
    print("\nイブ（日本語・記録済み）")
    for g in sorted(groups):
        ts = groups[g]
        print(f"  {g:10s}（{len(ts)}件・長さの中央 {st.median(len(t) for t in ts):.0f}字）: "
              + " ／ ".join(f"{k} {_rate(bool(p.search(t)) for t in ts)}" for k, p in JA.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
