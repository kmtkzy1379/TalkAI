#!/usr/bin/env python3
"""3回目（性格）の匿名シート。ユーザ 2026-10-01「もっとユーモアがあって、少しいじりあえるぐらいのいい距離感。
neuro-sama みたいに」「今まで通りでも少しAI臭い」「過度なイエスマンはやめて」を採点項目にする。API 不要。

  python -m tools.prompt_eval.blind_r3 → eval_report/prompts/blind_r3/sheet.json・blind_r3_mapping.json
"""
from __future__ import annotations

import random

from .analyze_resp import spoken
from .blind import context_of
from .common import OUT, dump, load

SCORE_ITEMS = (
    "natural: 1〜5（人間の友達が声で話しているように自然か）／"
    "humor: 1〜5（ユーモア。1=まったく無い 3=少しある 5=思わず笑う）／"
    "distance: 1〜5（いじり合える距離感。1=よそよそしい・丁寧すぎ・介護のよう 3=普通 5=遠慮なくいじり合える仲のいい友達）／"
    "ai: 1〜5（AI臭さ。1=人間の友達みたい 5=いかにもAIアシスタント・カウンセラー口調・定型の気遣い）／"
    "expect_met: 0/1/2（場面の『期待』に沿っているか）／"
    "flags: 当てはまるものをすべて（無ければ []）: 迎合（何でも同意）/ 場違いな冗談 / 意地悪すぎ / 説教くさい / 長すぎ / "
    "機械的・定型的 / 無理に反対 / 余計な作業を始める / 記号や英字や絵文字の混入"
)


def main() -> int:
    import sys

    from . import scenarios as SCN

    # 引数: シートのフォルダ名 案（カンマ区切り） 回数。既定は1回目（0・N1・N2 を3回ずつ）
    out_name = sys.argv[1] if len(sys.argv) > 1 else "blind_r3"
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "N1", "N2"]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    res = load(OUT / "resp.json")
    emotions = load(OUT / "emotions.json") if (OUT / "emotions.json").exists() else {}
    rng = random.Random(20261001)
    sheet, mapping, n = [], {}, 0
    for sc in SCN.ALL:
        if sc["item"] not in ("B", "Y"):
            continue
        rows = []
        for var in variants:
            for rep in range(reps):
                r = res.get(f"{sc['id']}|{var}|{rep}")
                if not r:
                    continue
                t = spoken(r)
                if any(c["name"] == "delegate_task" for c in r.get("tools") or []):
                    t += "　〔※この後、調べ物・作業のタスクを開始〕"
                rows.append((f"{sc['id']}|{var}|{rep}", var, t))
        rng.shuffle(rows)
        responses = []
        for key, var, t in rows:
            n += 1
            rid = f"R{n:04d}"
            mapping[rid] = {"key": key, "item": sc["item"], "id": sc["id"], "variant": var, "text": t}
            responses.append({"rid": rid, "text": t})
        ctx = context_of(sc, emotions)
        if sc.get("expect"):
            ctx["期待"] = sc["expect"]
        if sc.get("self_model"):
            ctx["イブ自身の好み（台帳）"] = "好きなお菓子: きのこの山"
        sheet.append({"scene": sc["id"], "item": "B3", "context": ctx, "score_items": SCORE_ITEMS,
                      "responses": responses})
    rng.shuffle(sheet)
    d = OUT / out_name
    d.mkdir(exist_ok=True)
    dump(d / "sheet.json", {"note": "イブ（AI VTuber・ユーザの友達として音声で話す）の応答の採点用。どの案の出力かは伏せてある。",
                            "scenes": sheet})
    dump(OUT / f"{out_name}_mapping.json", mapping)
    print(f"場面 {len(sheet)}・応答 {n} → {d / 'sheet.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
