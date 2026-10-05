#!/usr/bin/env python3
"""「できないと言いながら delegate_task も出す」矛盾の関門を、記録済みの全応答に当てて確かめる。API 不要。

  python -m tools.prompt_eval.check_refusal_gate
本番の関門（`orchestrator._says_cannot`）で止まる委譲を全部並べる＝誤爆（正しい委譲を止めた）が無いかを目で見る。
"""
from __future__ import annotations

import json

from .analyze_resp import spoken
from .common import OUT, load


def main() -> int:
    from eve.response.orchestrator import _says_cannot

    res = load(OUT / "resp.json")
    stopped, total_deleg = [], 0
    for key, r in sorted(res.items()):
        calls = [c for c in r.get("tools") or [] if c["name"] == "delegate_task"]
        if not calls:
            continue
        total_deleg += 1
        # 本番の関門と同じく、声にした文（本文。空なら reply）で判定する
        if _says_cannot(spoken(r)):
            try:
                goal = json.loads(calls[0]["args"] or "{}").get("goal", "")
            except ValueError:
                goal = ""
            stopped.append((key, spoken(r), goal))
    print(f"委譲を含む応答 {total_deleg} 件のうち、関門が止めるもの {len(stopped)} 件")
    for key, t, goal in stopped:
        print(f"  {key:12s} {t[:60]} | goal: {goal[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
