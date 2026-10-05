#!/usr/bin/env python3
"""審査員3人の採点（eval_report/prompts/blind/judge_*.json）を案ごとに集計する。API 不要。

  python -m tools.prompt_eval.judges
"""
from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict

from .common import OUT, dump, load


def median_or_none(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return st.median(xs) if xs else None


def main() -> int:
    mapping = load(OUT / "blind_mapping.json")
    judges, names = [], []
    for p in sorted((OUT / "blind").glob("judge_*.json")):
        judges.append(load(p))
        names.append(p.stem.replace("judge_", ""))
    if not judges:
        print("採点ファイルがまだ無い")
        return 1
    print(f"審査員: {names}")
    per_rid = {}
    for rid, m in mapping.items():
        scores = [j.get(rid) or {} for j in judges]
        nat = median_or_none([s.get("natural") for s in scores])
        react = median_or_none([s.get("reaction") for s in scores])
        cont = median_or_none([s.get("continuation") for s in scores])
        exp = median_or_none([s.get("expect_met") for s in scores])
        lens = Counter(s.get("length") for s in scores if s.get("length"))
        flags = Counter(f for s in scores for f in (s.get("flags") or []))
        per_rid[rid] = {**m, "natural": nat, "reaction": react, "continuation": cont, "expect_met": exp,
                        "length": lens.most_common(1)[0][0] if lens else None,
                        # 審査員が2人以上なら2人以上が付けた指摘だけ数える（1人だけの指摘は揺れとみなす）
                        "flags": sorted(f for f, c in flags.items() if c >= min(2, len(judges))),
                        "n_judges": sum(1 for s in scores if s)}
    agg = defaultdict(lambda: defaultdict(list))
    for rid, r in per_rid.items():
        a = agg[(r["item"], r["variant"])]
        for k in ("natural", "reaction", "continuation", "expect_met"):
            if r[k] is not None:
                a[k].append(r[k])
        if r["length"]:
            a["length"].append(r["length"])
        a["flags"].extend(r["flags"])
        a["n"].append(1)
    summary = {}
    for (item, var), a in sorted(agg.items()):
        row = {"n": len(a["n"])}
        for k in ("natural", "reaction", "continuation", "expect_met"):
            if a[k]:
                row[k] = round(st.mean(a[k]), 2)
        if a["length"]:
            c = Counter(a["length"])
            row["length"] = {k: c.get(k, 0) for k in ("short", "ok", "long")}
        row["flags"] = dict(Counter(a["flags"]).most_common())
        summary[f"{item}|{var}"] = row
        print(f"{item:3s} {var:3s} n={row['n']:3d} 自然さ {row.get('natural')} 反応 {row.get('reaction', '-')} "
              f"続き {row.get('continuation', '-')} 期待 {row.get('expect_met', '-')} 長さ {row.get('length', '-')} "
              f"指摘 {row['flags']}")
    # 場面×R の「予想どおり/予想外」の反応差（採点の reaction）
    react = defaultdict(lambda: defaultdict(list))
    for rid, r in per_rid.items():
        if r["item"] == "R" and r["reaction"] is not None:
            grp = "u" if r["id"].endswith("u") else ("e" if r["id"].endswith("e") else r["id"][-2:])
            react[r["variant"]][grp].append(r["reaction"])
    rsum = {v: {g: round(st.mean(xs), 2) for g, xs in d.items()} for v, d in react.items()}
    for v, d in sorted(rsum.items()):
        print(f"R 反応の大きさ {v}: 予想どおり {d.get('e')} / 予想外 {d.get('u')} / 前ターン低 {d.get('lo')} / 高 {d.get('hi')}")
    # 審査員ごとの自然さの平均（案ごと）も残す（審査員の間で結論がそろうかを見る）
    by_judge = {}
    for name, j in zip(names, judges):
        acc = defaultdict(list)
        for rid, m in mapping.items():
            s = j.get(rid) or {}
            if isinstance(s.get("natural"), (int, float)):
                acc[f"{m['item']}|{m['variant']}"].append(s["natural"])
        by_judge[name] = {k: round(st.mean(v), 2) for k, v in acc.items()}
    for name, d in by_judge.items():
        print(f"  {name}: " + " ".join(f"{k}={v}" for k, v in sorted(d.items())))
    dump(OUT / "judges_summary.json", {"summary": summary, "reaction": rsum, "per_rid": per_rid,
                                       "n_judges": len(judges), "judges": names, "by_judge": by_judge})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
