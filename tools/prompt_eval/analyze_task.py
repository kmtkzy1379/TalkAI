#!/usr/bin/env python3
"""task.json の集計（API 不要）。  python -m tools.prompt_eval.analyze_task [--verbose]"""
from __future__ import annotations

import sys
from collections import defaultdict

from .common import OUT, dump, load


def main() -> int:
    res = load(OUT / "task.json")
    verbose = "--verbose" in sys.argv
    agg = defaultdict(lambda: defaultdict(list))
    for key, r in sorted(res.items()):
        agg[r["variant"]][r["sid"]].append(r)
        if verbose:
            print(f"{key:8s} 呼び出し{r['n_calls']} 同じ繰り返し{r['identical_repeats']} 正解{r['ok']} 諦め{r['gave_up']} "
                  f"{r['sec']}s ／ {str(r['final'])[:80]}")
            for name, a in r["calls"]:
                print(f"      - {name} {a[:110]}")
    out = {}
    for var in ("0", "U", "M"):
        row = {}
        for sid, rs in sorted(agg[var].items()):
            row[sid] = {
                "n": len(rs),
                "ok": sum(1 for r in rs if r["ok"]) if rs[0]["ok"] is not None else None,
                "calls": [r["n_calls"] for r in rs],
                "identical_repeats": [r["identical_repeats"] for r in rs],
                "gave_up": sum(1 for r in rs if r["gave_up"]),
                "sec": [r["sec"] for r in rs],
            }
        out[var] = row
        print(f"{var}: " + " ／ ".join(
            f"{sid} 正解{v['ok']}/{v['n']} 呼び出し{v['calls']} 同じ繰り返し{v['identical_repeats']} 諦め{v['gave_up']}"
            for sid, v in row.items()))
    dump(OUT / "task_metrics.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
