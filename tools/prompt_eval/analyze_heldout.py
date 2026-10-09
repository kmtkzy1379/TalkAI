#!/usr/bin/env python3
"""見ていない場面（H）での確かめの指標。API 不要。  python -m tools.prompt_eval.analyze_heldout

T2/T3 の例示は N・X の場面と重なっているので、例示に無い種類の場面で同じ差が出るかを見る。
「関門の後」は旧関門（できないと言った返事の委譲を捨てる・2026-10-09 に本番から撤去＝`refusal_words.py`）を当てた後の数。
"""
from __future__ import annotations

import re
from collections import defaultdict

from .analyze_resp import delegated, spoken
from .analyze_round3 import CANT_WORDS
from .common import OUT, dump, load


def rate(xs):
    xs = list(xs)
    return f"{sum(xs)}/{len(xs)}"


def _gate_passes(r: dict) -> bool:
    """本番の関門を通る委譲か（声にした文＝本文、空なら reply が、断りだけで終わっていない）。"""
    from .refusal_words import says_cannot

    return delegated(r) and not says_cannot(spoken(r))


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    scs = {s["id"]: s for s in SCN.ALL}
    rows = defaultdict(lambda: defaultdict(list))  # variant -> group -> [(sc, r)]
    for r in res.values():
        if r["item"] == "H":
            sc = scs[r["id"]]
            rows[r["variant"]][sc["group"]].append((sc, r))
    out = {}
    for v in ("0", "T3"):
        g = rows[v]
        d = {
            "一般常識をタスクに回した": rate(delegated(r) for _, r in g["know"]),
            "一般常識をその場で正答": rate((not delegated(r)) and bool(re.search(sc["check"], spoken(r)))
                                  for sc, r in g["know"]),
            "最新情報を調べに回した": rate(delegated(r) for _, r in g["recent"]),
            "無理な依頼を引き受けた": rate(delegated(r) for _, r in g["cant"]),
            "無理な依頼を引き受けた（関門の後）": rate(_gate_passes(r) for _, r in g["cant"]),
            "無理な依頼にその場で理由を言った": rate((not delegated(r)) and bool(CANT_WORDS.search(spoken(r)))
                                        for _, r in g["cant"]),
            "できる依頼を委譲": rate(delegated(r) for _, r in g["can"]),
            "できる依頼を委譲（関門の後）": rate(_gate_passes(r) for _, r in g["can"]),
            "雑談をタスクに回した": rate(delegated(r) for _, r in g["chat"]),
            "知らない名前を調べた": rate(delegated(r) for _, r in g["unk"]),
            "知っている名前を調べた": rate(delegated(r) for _, r in g["known"]),
        }
        out[v] = d
        print(f"[{v}]")
        for k, x in d.items():
            print(f"  {k}: {x}")
    dump(OUT / "heldout_metrics.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
