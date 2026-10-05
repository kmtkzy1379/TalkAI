#!/usr/bin/env python3
"""resp.json を場面ごと・変種ごとに並べて表示する（読む用・API 不要）。

  python -m tools.prompt_eval.show P2           # 項目
  python -m tools.prompt_eval.show K2 QK-U1     # 項目＋場面 id
"""
from __future__ import annotations

import sys

from .analyze_resp import spoken
from .common import OUT, load


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    emo_path = OUT / "emotions.json"
    emo = load(emo_path) if emo_path.exists() else {}
    items = sys.argv[1].split(",")
    ids = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    for sc in SCN.ALL:
        if sc["item"] not in items or (ids and sc["id"] not in ids):
            continue
        e = emo.get(sc["id"]) if sc.get("emotion") == "auto" else sc.get("emotion")
        print(f"\n### {sc['id']} [{sc.get('group', '')}] {sc['text']}"
              + (f"  (感情={e})" if e else "") + (f"  (つなぎ={sc['filler']})" if sc.get("filler") else ""))
        for v in ("0", "OLD", "U", "M", "M2", "M3", "T", "T+", "T1", "T2", "T3", "N1", "N2", "N3", "NEW"):
            for rep in range(3):
                r = res.get(f"{sc['id']}|{v}|{rep}")
                if not r:
                    continue
                t = spoken(r)
                tools = [c["name"] for c in r["tools"]]
                print(f"  {v:2s}{rep} ({len(t):3d}) {t}" + (f"  [tool:{','.join(tools)}]" if tools else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
