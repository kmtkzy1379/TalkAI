#!/usr/bin/env python3
"""反映後の確認（Z: 変更前 OLD / 変更後 NEW）用の匿名シート。声になる文は、それぞれの本番の後処理を通す:
NEW＝今の sanitize（絵文字・特殊トークンも落とす）＋二重化の関門 / OLD＝変更前の sanitize（記号・かぎかっこだけ）。

  python -m tools.prompt_eval.blind_final   → eval_report/prompts/blind_final/sheet.json と blind_final_mapping.json
"""
from __future__ import annotations

import random
import re

from .analyze_resp import spoken
from .blind import RUBRIC, context_of
from .common import OUT, dump, load

_OLD_LEAD = re.compile(r"^\s*(?:[-*・]|\d+[.)])\s+")
_OLD_MD = re.compile(r"[*`#]")
_OLD_QUOTES = re.compile(r"[「」『』]")


def old_sanitize(t: str) -> str:
    return _OLD_QUOTES.sub("", _OLD_MD.sub("", _OLD_LEAD.sub("", t))).strip()


def postprocess(text: str, variant: str) -> str:
    from eve.response.orchestrator import REPEAT_MIN_CHARS, _repeat_key
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech

    sp = JapaneseSentenceSplitter()
    out, said = [], set()
    for c in sp.feed(text) + sp.flush():
        if variant == "OLD":
            clean = old_sanitize(c)
            if clean:
                out.append(clean)
            continue
        clean = sanitize_for_speech(c)
        if not clean:
            continue
        k = _repeat_key(clean)
        if len(k) >= REPEAT_MIN_CHARS and k in said:
            break
        said.add(k)
        out.append(clean)
    return "".join(out)


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    rng = random.Random(20261001)
    sheet, mapping, n = [], {}, 0
    for sc in SCN.BY_ITEM["Z"]:
        rows = []
        for rep in range(3):
            for var in ("OLD", "NEW"):
                r = res.get(f"{sc['id']}|{var}|{rep}")
                if not r:
                    continue
                t = postprocess(spoken(r), var)
                if any(c["name"] == "delegate_task" for c in r.get("tools") or []):
                    t += "　〔※この後、調べ物・作業のタスクを開始〕"
                rows.append((f"{sc['id']}|{var}|{rep}", var, t))
        rng.shuffle(rows)
        responses = []
        for key, var, t in rows:
            n += 1
            rid = f"Z{n:04d}"
            mapping[rid] = {"key": key, "item": "Z", "id": sc["id"], "variant": var, "text": t}
            responses.append({"rid": rid, "text": t})
        sheet.append({"scene": sc["id"], "item": "G", "context": context_of(sc, {}),
                      "score_items": RUBRIC["common"], "responses": responses})
    rng.shuffle(sheet)
    d = OUT / "blind_final"
    d.mkdir(exist_ok=True)
    dump(d / "sheet.json", {"note": "イブ（AI VTuber・音声で会話する）の応答の採点用。どの案の出力かは伏せてある。",
                            "scenes": sheet})
    dump(OUT / "blind_final_mapping.json", mapping)
    print(f"場面 {len(sheet)}・応答 {n} → {d / 'sheet.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
