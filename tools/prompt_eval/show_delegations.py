#!/usr/bin/env python3
"""委譲した回だけ、言った一言と goal を並べる（読む用・API 不要）。  python -m tools.prompt_eval.show_delegations N,X,C"""
from __future__ import annotations

import json
import sys

from .analyze_resp import delegated, spoken
from .common import OUT, load


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    items = sys.argv[1].split(",")
    for sc in SCN.ALL:
        if sc["item"] not in items:
            continue
        for key in sorted(k for k in res if res[k]["id"] == sc["id"]):
            r = res[key]
            if not delegated(r):
                continue
            goals = []
            for c in r["tools"]:
                if c["name"] == "delegate_task":
                    try:
                        a = json.loads(c["args"] or "{}")
                    except ValueError:
                        a = {}
                    goals.append(f"{a.get('goal', '')}（when={a.get('when_seconds')}）")
            print(f"{key:12s} {sc['text'][:22]:22s} | {spoken(r)[:44]} | goal: {' / '.join(goals)[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
