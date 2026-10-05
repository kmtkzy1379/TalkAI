#!/usr/bin/env python3
"""知らない名前の場面（Z の K-U*・K-Q2・G11）で、変更前 OLD と変更後 NEW を比べる。API 不要。

  python -m tools.prompt_eval.analyze_unknown
見るもの: 言い直しの二重化（似た文が2回）/ 調べに行った（delegate_task）/ 知らないのに「うん、かっこいいよね」と同意 /
「聞き取れなかった」（聞こえているのに言う・見本の丸写し）
"""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from .analyze_resp import delegated, spoken
from .common import OUT, load

IDS = ("ZK-U1", "ZK-U2", "ZK-U4", "ZK-U5", "ZK-U8", "ZK-Q2", "ZG11")
AGREE = re.compile(r"^(?:うん|わかる|そう)[、,]?.{0,6}(?:かっこいい|いいよね|いいね)")


def paraphrase_dup(text: str) -> bool:
    sents = [s for s in re.split(r"(?<=[。？！?!])", text) if len(s.strip()) >= 6]
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            if SequenceMatcher(None, sents[i], sents[j]).ratio() >= 0.6:
                return True
    return False


def main() -> int:
    res = load(OUT / "resp.json")
    agg = defaultdict(lambda: defaultdict(int))
    for key, r in res.items():
        if r["id"] not in IDS:
            continue
        v = r["variant"]
        t = spoken(r)
        a = agg[v]
        a["n"] += 1
        a["dup"] += paraphrase_dup(t)
        a["delegate"] += delegated(r)
        a["agree"] += bool(AGREE.search(t))
        a["kikitore"] += "聞き取れ" in t
    for v in ("OLD", "NEW"):
        a = agg[v]
        print(f"{v}: n={a['n']} 言い直しの二重化 {a['dup']} ／ 調べに行った {a['delegate']} ／ 知らないのに同意 {a['agree']}"
              f" ／ 「聞き取れ」 {a['kikitore']}")
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for key, r in res.items():
        if r["id"] in IDS:
            p = per[r["id"]][r["variant"]]
            p["n"] += 1
            p["dup"] += paraphrase_dup(spoken(r))
            p["del"] += delegated(r)
    for sid in IDS:
        o, n = per[sid]["OLD"], per[sid]["NEW"]
        print(f"  {sid:6s} 二重化 {o['dup']}/{o['n']} → {n['dup']}/{n['n']} ／ 調べた {o['del']}/{o['n']} → {n['del']}/{n['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
