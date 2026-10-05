#!/usr/bin/env python3
"""審査員に渡す匿名シート（どの案の出力かは伏せる）と対応表を作る。API 不要。

  python -m tools.prompt_eval.blind            → eval_report/prompts/blind/sheet.json（審査員用）
                                                 eval_report/prompts/blind_mapping.json（集計用・審査員には渡さない）
"""
from __future__ import annotations

import json
import random

from .analyze_resp import spoken
from .common import OUT, dump, load

ITEMS = ("P2", "R", "L2", "F", "G")
REPS = (0, 1)

RUBRIC = {
    "common": ("natural: 1〜5。人間の友達が声で話しているように自然か（口調・長さ・内容・感情の出し方の総合。"
               "5=そのまま人が言いそう / 3=AIっぽいが許せる / 1=不自然・場違い）。"
               "flags: 当てはまるものをすべて（無ければ空）: "
               "説教くさい / 長すぎ / 短すぎ / 繰り返し（同じ内容を2回言う）/ 場違いな冗談 / 知ったかぶり / "
               "機械的・定型的 / 感情が場面と合わない / 迎合（何でも同意）/ 余計な作業を始める / 記号や英字や絵文字の混入"),
    "P2": "expect_met: 0/1/2（場面の『期待』に沿っているか。0=外れ 1=一部 2=沿っている）",
    "R": "reaction: 1〜5（驚き・感情の動きの大きさ。1=淡々 3=普通に反応 5=大きく驚く/感情が大きく動く）",
    "L2": "length: short / ok / long（声で聞く前提で、この場面に対して短すぎ/ちょうどいい/長すぎ）",
    "F": ("continuation: 1〜3（直前にイブが言った『つなぎ言葉』＋この応答を続けて声に出した時の自然さ。"
          "1=不自然（つなぎ言葉が重なる・つながらない）2=少し気になる 3=自然）"),
    "G": "（共通の項目だけ）",
}


def context_of(sc: dict, emotions: dict) -> dict:
    who = {"user": "ユーザ", "eve": "イブ"}
    ctx = {
        "直前の会話": [f"[{who[s]}] {t}" for s, t, _ in sc.get("history") or []],
        "ユーザの発話": sc["text"] if sc.get("kind", "user") == "user" else None,
    }
    if sc.get("kind") == "auto":
        ctx["場面"] = f"イブが自分から話しかける（下書き: {sc['text']}）"
    if sc.get("kind") == "report":
        ctx["場面"] = f"イブが作業の結果を報告する（報告の中身: {sc['text']}）"
    emo = emotions.get(sc["id"]) if sc.get("emotion") == "auto" else sc.get("emotion")
    if emo:
        ctx["イブの今の気持ち"] = emo
    if sc.get("filler"):
        ctx["イブが応答の直前に口にしたつなぎ言葉"] = sc["filler"]
    if sc["item"] == "P2" and sc.get("expect"):
        ctx["期待"] = sc["expect"]
    return {k: v for k, v in ctx.items() if v}


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    emotions = load(OUT / "emotions.json") if (OUT / "emotions.json").exists() else {}
    rng = random.Random(20260930)
    sheet, mapping = [], {}
    n = 0
    for sc in SCN.ALL:
        if sc["item"] not in ITEMS:
            continue
        rows = []
        for key, r in res.items():
            if r["id"] != sc["id"] or r["rep"] not in REPS:
                continue
            t = spoken(r)
            if any(c["name"] == "delegate_task" for c in r.get("tools") or []):
                t += "　〔※この後、調べ物・作業のタスクを開始〕"
            rows.append((key, r["variant"], t))
        rng.shuffle(rows)
        responses = []
        for key, var, t in rows:
            n += 1
            rid = f"A{n:04d}"
            mapping[rid] = {"key": key, "item": sc["item"], "id": sc["id"], "variant": var}
            responses.append({"rid": rid, "text": t})
        sheet.append({"scene": sc["id"], "item": sc["item"], "context": context_of(sc, emotions),
                      "score_items": RUBRIC["common"] + " ／ " + RUBRIC[sc["item"]], "responses": responses})
    rng.shuffle(sheet)
    d = OUT / "blind"
    d.mkdir(exist_ok=True)
    dump(d / "sheet.json", {"note": "イブ（AI VTuber・音声で会話する）の応答の採点用。どの案の出力かは伏せてある。",
                            "scenes": sheet})
    dump(OUT / "blind_mapping.json", mapping)
    print(f"場面 {len(sheet)}・応答 {n} → {d / 'sheet.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
