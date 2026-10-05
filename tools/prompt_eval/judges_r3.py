#!/usr/bin/env python3
"""3回目（性格）の採点の集計。API 不要。  python -m tools.prompt_eval.judges_r3"""
from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict

from .common import OUT, dump, load

KEYS = ("natural", "humor", "distance", "ai", "expect_met")


def main() -> int:
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "blind_r3"
    mapping = load(OUT / f"{name}_mapping.json")
    judges = {p.stem.replace("judge_", ""): load(p) for p in sorted((OUT / name).glob("judge_*.json"))}
    per = {}
    for rid, m in mapping.items():
        row = dict(m)
        for k in KEYS:
            xs = [j.get(rid, {}).get(k) for j in judges.values()]
            xs = [x for x in xs if isinstance(x, (int, float))]
            row[k] = st.median(xs) if xs else None
        fl = Counter(f for j in judges.values() for f in (j.get(rid, {}).get("flags") or []))
        row["flags"] = sorted(f for f, c in fl.items() if c >= 2)
        per[rid] = row
    agg = defaultdict(lambda: defaultdict(list))
    flags = defaultdict(Counter)
    scene = defaultdict(lambda: defaultdict(list))
    for r in per.values():
        for k in KEYS:
            if r[k] is not None:
                agg[r["variant"]][k].append(r[k])
        flags[r["variant"]].update(r["flags"])
        if r["natural"] is not None:
            scene[r["id"]][r["variant"]].append((r["natural"], r["humor"], r["distance"], r["ai"]))
    out = {"variant": {}, "flags": {}, "by_judge": {}, "scene": {}}
    print(f"審査員: {list(judges)}")
    for v in sorted(agg, key=lambda x: ("0", "N1", "N2", "N3").index(x) if x in ("0", "N1", "N2", "N3") else 9):
        a = agg[v]
        out["variant"][v] = {k: round(st.mean(a[k]), 2) for k in KEYS if a[k]}
        out["flags"][v] = dict(flags[v].most_common())
        print(f"  {v:3s} 自然さ {out['variant'][v].get('natural')} ／ ユーモア {out['variant'][v].get('humor')} ／ "
              f"距離感 {out['variant'][v].get('distance')} ／ AI臭さ {out['variant'][v].get('ai')} ／ 期待 {out['variant'][v].get('expect_met')}"
              f" ／ 指摘 {out['flags'][v]}")
    for jname, j in judges.items():  # name（シート名）を上書きしない＝下の保存先がずれない
        acc = defaultdict(lambda: defaultdict(list))
        for rid, m in mapping.items():
            s = j.get(rid) or {}
            for k in ("natural", "humor", "distance", "ai"):
                if isinstance(s.get(k), (int, float)):
                    acc[m["variant"]][k].append(s[k])
        out["by_judge"][jname] = {v: {k: round(st.mean(x), 2) for k, x in d.items()} for v, d in acc.items()}
        print(f"  [{jname}] " + " ／ ".join(
            f"{v}: 自然{d.get('natural')} 笑{d.get('humor')} 距{d.get('distance')} AI{d.get('ai')}"
            for v, d in sorted(out['by_judge'][jname].items())))
    for sid, d in sorted(scene.items()):
        out["scene"][sid] = {v: [round(st.mean(x[i] for x in xs), 2) for i in range(4)] for v, xs in d.items()}
        print(f"  {sid}: " + " ／ ".join(f"{v} 自{s[0]} 笑{s[1]} 距{s[2]} AI{s[3]}" for v, s in sorted(out['scene'][sid].items())))
    dump(OUT / f"judges_{name.replace('blind_', '')}_summary.json", {**out, "per_rid": per})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
