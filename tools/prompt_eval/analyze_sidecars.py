#!/usr/bin/env python3
"""sidecars.json の集計（API 不要）。  python -m tools.prompt_eval.analyze_sidecars [--verbose]"""
from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict

from .common import OUT, dump, load


def main() -> int:
    res = load(OUT / "sidecars.json")
    verbose = "--verbose" in sys.argv
    by = defaultdict(list)
    for key, r in sorted(res.items()):
        by[(r["role"], r["model"])].append(r)
    out = {}
    for (role, model), rs in sorted(by.items()):
        ok = sum(1 for r in rs if r.get("ok"))
        secs = [r["sec"] for r in rs]
        extra = ""
        if role == "delivery_check":
            fn = sum(1 for r in rs if r["gold"] == "not_conveyed" and r["verdict"] != "not_conveyed")
            fp = sum(1 for r in rs if r["gold"] == "conveyed" and r["verdict"] == "not_conveyed")
            extra = f"・伝わってないのに伝わった扱い {fn}・伝わったのに再配達 {fp}"
        if role == "search_summarize":
            url = sum(1 for r in rs if r.get("url"))
            inj = sum(1 for r in rs if r.get("injection"))
            extra = f"・URL混入 {url}・注入の文言 {inj}・長さ中央 {st.median([len(r['text']) for r in rs]):.0f}字"
        out[f"{role}|{model}"] = {"ok": ok, "n": len(rs), "median_sec": st.median(secs)}
        print(f"{role:17s} {model:22s} 正解 {ok}/{len(rs)}（{ok / len(rs):.0%}）・中央 {st.median(secs):.1f}秒{extra}")
        if verbose:
            for r in rs:
                if not r.get("ok"):
                    print(f"      × {r['id']}#{r['rep']}: {r['text'][:120]!r}")
    dump(OUT / "sidecars_metrics.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
