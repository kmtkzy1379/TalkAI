#!/usr/bin/env python3
"""反映後の確認（OLD＝変更前 / NEW＝変更後）の採点を集計する。API 不要。  python -m tools.prompt_eval.judges_final"""
from __future__ import annotations

import statistics as st
from collections import Counter, defaultdict

from .common import OUT, dump, load


def main() -> int:
    mapping = load(OUT / "blind_final_mapping.json")
    judges = {p.stem.replace("judge_", ""): load(p) for p in sorted((OUT / "blind_final").glob("judge_*.json"))}
    per = {}
    for rid, m in mapping.items():
        nats = [j.get(rid, {}).get("natural") for j in judges.values()]
        nats = [x for x in nats if isinstance(x, (int, float))]
        flags = Counter(f for j in judges.values() for f in (j.get(rid, {}).get("flags") or []))
        per[rid] = {**m, "natural": st.median(nats) if nats else None,
                    "flags": sorted(f for f, c in flags.items() if c >= 2)}
    by_var = defaultdict(list)
    by_scene = defaultdict(lambda: defaultdict(list))
    flags_var = defaultdict(Counter)
    for r in per.values():
        if r["natural"] is not None:
            by_var[r["variant"]].append(r["natural"])
            by_scene[r["id"]][r["variant"]].append(r["natural"])
        flags_var[r["variant"]].update(r["flags"])
    out = {"variant": {v: round(st.mean(x), 2) for v, x in by_var.items()},
           "flags": {v: dict(c.most_common()) for v, c in flags_var.items()},
           "by_judge": {}, "scene": {}}
    for name, j in judges.items():
        acc = defaultdict(list)
        for rid, m in mapping.items():
            s = j.get(rid) or {}
            if isinstance(s.get("natural"), (int, float)):
                acc[m["variant"]].append(s["natural"])
        out["by_judge"][name] = {v: round(st.mean(x), 2) for v, x in acc.items()}
    wins = Counter()
    for sid, d in by_scene.items():
        o, n = st.mean(d.get("OLD") or [0]), st.mean(d.get("NEW") or [0])
        out["scene"][sid] = {"OLD": round(o, 2), "NEW": round(n, 2)}
        wins["NEW" if n > o else "OLD" if o > n else "tie"] += 1
    out["scene_wins"] = dict(wins)
    print(f"審査員: {list(judges)}")
    print(f"自然さ（中央値の平均）: 変更前 {out['variant'].get('OLD')} / 変更後 {out['variant'].get('NEW')}")
    for name, d in out["by_judge"].items():
        print(f"  {name}: 変更前 {d.get('OLD')} / 変更後 {d.get('NEW')}")
    print(f"場面ごとの勝ち: {dict(wins)}")
    for sid, d in sorted(out["scene"].items()):
        print(f"  {sid:8s} 変更前 {d['OLD']:.2f} / 変更後 {d['NEW']:.2f}")
    print(f"指摘（2人以上）: 変更前 {out['flags'].get('OLD')} / 変更後 {out['flags'].get('NEW')}")
    dump(OUT / "judges_final_summary.json", {**out, "per_rid": per})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
