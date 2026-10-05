#!/usr/bin/env python3
"""応答の二重化の関門（`orchestrator.handle._emit`）を、記録済みの実応答（resp.json）に当てて確かめる。API 不要。

  python -m tools.prompt_eval.check_repeat_gate
本番と同じ分割（JapaneseSentenceSplitter）→ sanitize → 関門 の順で流し、切った位置と切らなかった二重化を出す。
"""
from __future__ import annotations

import re

from .common import OUT, dump, load


def dup_detector(t: str, n: int = 18) -> bool:
    """評価用の二重化検出（18文字以上の同じ部分文字列が離れて2回）。関門とは別の物差し。"""
    t2 = re.sub(r"\s", "", t)
    seen = {}
    for i in range(0, len(t2) - n + 1):
        s = t2[i:i + n]
        if s in seen and i - seen[s] >= n:
            return True
        seen.setdefault(s, i)
    return False


def main() -> int:
    from eve.response.orchestrator import REPEAT_MIN_CHARS, _repeat_key
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech

    res = load(OUT / "resp.json")
    cut, missed, false_cut, rows = 0, 0, 0, []
    total_dup = 0
    for key, r in sorted(res.items()):
        text = r.get("text") or ""
        sp = JapaneseSentenceSplitter()
        clauses = sp.feed(text) + sp.flush()
        said, out, stopped_at = set(), [], None
        for c in clauses:
            clean = sanitize_for_speech(c)
            if not clean:
                continue
            k = _repeat_key(clean)
            if len(k) >= REPEAT_MIN_CHARS and k in said:
                stopped_at = clean
                break
            said.add(k)
            out.append(clean)
        is_dup = dup_detector(text)
        total_dup += is_dup
        if stopped_at is not None:
            if is_dup:
                cut += 1
            else:
                false_cut += 1
            rows.append({"key": key, "dup": is_dup, "stopped_at": stopped_at, "spoken": "".join(out),
                         "original": text})
        elif is_dup:
            missed += 1
            rows.append({"key": key, "dup": True, "stopped_at": None, "spoken": "".join(out), "original": text})
    print(f"応答 {len(res)}・二重化 {total_dup}（評価用の物差し）")
    print(f"  関門が切った: 二重化 {cut} ／ 二重化ではないのに切った {false_cut} ／ 切れなかった二重化 {missed}")
    for row in rows:
        mark = "✂" if row["stopped_at"] else "・"
        print(f"{mark} {row['key']:16s} 二重化={row['dup']} 切った節={row['stopped_at']!r}")
        print(f"     残った発話: {row['spoken'][:140]}")
    dump(OUT / "repeat_gate.json", {"cut": cut, "false_cut": false_cut, "missed": missed, "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
