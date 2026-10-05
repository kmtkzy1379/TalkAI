#!/usr/bin/env python3
"""性格の指示文が口癖として丸写しされていないかを数える。API 不要。

  python -m tools.prompt_eval.echo_check
1. 指示文の言い回し（「倍にして」「ポンコツ」「いじり」…）が応答に出た率を案ごとに（ユーザ発話に含まれる語は除く）
2. 指示文と応答の最長共通部分（4文字以上・かなだけの断片は除く）＝丸写しの疑い
3. 案の中で、別々の場面に3回以上出る言い回し（口癖）
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .analyze_resp import spoken
from .common import OUT, dump, load

WATCH = {
    "倍": r"倍",
    "ポンコツ": r"ポンコツ",
    "いじ(る/り)": r"いじ[るりらっれ]",
    "からか": r"からか",
    "ボケ/ツッコ": r"ボケ|ツッコ|突っ込",
    "真顔/とぼけ/斜め上": r"真顔|とぼけ|斜め上",
    "悔しい": r"悔し",
    "判定": r"判定",
    # N4/N5 の指示文の言い回し
    "容赦": r"容赦",
    "悪びれ": r"悪びれ",
    "わがまま": r"わがまま",
    "気まぐれ": r"気まぐれ",
    "自信家/自慢": r"自信家|自慢",
    "突っ込": r"突っ込|ツッコ",
}
_KANA_ONLY = re.compile(r"^[ぁ-んー、。？！…]+$")


def _lcs(a: str, b: str) -> str:
    """最長共通部分文字列（短い文字列どうし・素朴な DP）。"""
    best, end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return a[end - best:end]


def main() -> int:
    from . import scenarios as SCN
    from .variants import PERSONA_0, PERSONA_FULL

    res = load(OUT / "resp.json")
    scs = {s["id"]: s for s in SCN.ALL}
    prompts = {"0": PERSONA_0, **PERSONA_FULL, "E": PERSONA_0, "NE": PERSONA_FULL["N4"], "N5E": PERSONA_FULL["N5"]}
    rows = defaultdict(list)
    for r in res.values():
        if r["item"] in ("B", "Y", "NQ") and r["variant"] in prompts and r["rep"] in (0, 1):
            rows[r["variant"]].append((scs[r["id"]], spoken(r)))

    saved = {"words": {}, "phrases": {}}
    print("1. 指示文の言い回しが応答に出た数（ユーザがその語を言った場面は除く）")
    for v in ("0", "N3", "N4", "N5", "E", "NE", "N5E"):
        out = []
        saved["words"][v] = {"n": len(rows[v])}
        for lab, pat in WATCH.items():
            n = sum(1 for sc, t in rows[v] if re.search(pat, t) and not re.search(pat, sc["text"]))
            out.append(f"{lab} {n}")
            saved["words"][v][lab] = n
        print(f"  {v:3s}（{len(rows[v])}応答）: " + " ／ ".join(out))

    print("\n2. 指示文との最長共通部分（5文字以上・かなだけの断片を除く）")
    for v in ("N3", "N4", "N5", "NE", "N5E"):
        hits = Counter()
        for sc, t in rows[v]:
            s = _lcs(t, prompts[v])
            if len(s) >= 5 and not _KANA_ONLY.match(s) and s not in sc["text"]:
                hits[s] += 1
        print(f"  {v}: " + (" ／ ".join(f"「{s}」×{n}" for s, n in hits.most_common(8)) or "なし"))

    print("\n3. 別々の場面に3回以上出る言い回し（6〜10文字・案の中）")
    for v in ("0", "N3", "N4", "N5", "E", "NE", "N5E"):
        seen = defaultdict(set)
        for sc, t in rows[v]:
            body = re.sub(r"[、。？！…\s]", "", t)
            grams = {body[i:i + k] for k in range(6, 11) for i in range(len(body) - k + 1)}
            for g in grams:
                if not _KANA_ONLY.match(g) and g not in sc["text"]:
                    seen[g].add(sc["id"])
        common = sorted(((g, len(ids)) for g, ids in seen.items() if len(ids) >= 3), key=lambda x: (-x[1], -len(x[0])))
        # 長い方を残す（部分文字列の重複を落とす）
        kept = []
        for g, n in common:
            if not any(g in k and n <= kn for k, kn in kept):
                kept.append((g, n))
        print(f"  {v}: " + (" ／ ".join(f"「{g}」{n}場面" for g, n in kept[:10]) or "なし"))
        saved["phrases"][v] = kept[:10]
    dump(OUT / "neuro" / "echo.json", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
