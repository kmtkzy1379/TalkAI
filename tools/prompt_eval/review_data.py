#!/usr/bin/env python3
"""ユーザが選ぶ・評価するページ（イブの話し方レビュー）のデータを作る。API 不要。

  python -m tools.prompt_eval.review_data → eval_report/prompts/review/data.json

2026-10-01 ユーザ「どんな応答したのか、neuro-sama と比べたい。類似するほどいいってことではないから俺が評価」
「実例をやるなら丸写し確認と入れる会話データの吟味も必要。ここら辺は人間の感性がよさそう」「候補会話を挙げて」。
- 実例の候補: Neuro-sama のやりとりを日本語に移した 65組＋丸写しの検証結果＋アシスタントの所見（おすすめ・注意）
- 比べる返事: 今の性格（旧）／N6（実例なし）／N6＋実例8 を、Neuro の場面40と自前の場面24で（各2回）
"""
from __future__ import annotations

from collections import defaultdict

from .analyze_resp import spoken
from .common import OUT, dump, load

VARIANTS = {"0": "今の性格（2026-09まで）", "K0": "N6（実例なし）", "K8": "N6＋実例8"}

# アシスタントの所見（ユーザが選ぶ時の参考。決めるのはユーザ）
REC = {  # おすすめ: 短い・イブの世界でも通じる・Neuro らしい態度（からかう・自慢・突飛・はぐらかし）
    "NQ40": "突飛", "NQ41": "AI の自覚ネタ", "NQ42": "言い返し", "NQ43": "からかう", "NQ58": "からかう",
    "NQ60": "AI らしい好み", "NQ76": "突飛", "NQ77": "言い切り", "NQ79": "AI の自覚", "NQ83": "自慢・突飛",
    "NQ85": "はぐらかし", "NQ91": "問い返し（はぐらかし）", "NQ93": "ダジャレ", "NQ94": "からかう",
    "NQ96": "自慢・突飛", "NQ102": "自慢",
}
NOTE = {  # 注意（入れるなら理由を分かった上で）
    "NQ45": "内輪ネタ（Neuro が Vedal の給料を払う設定・イギリス）", "NQ47": "内輪ネタ寄り",
    "NQ48": "長い・話が飛びすぎ", "NQ49": "内輪ネタ（カエルの帽子）・検証で丸写しされた",
    "NQ53": "内輪ネタ（カエルの帽子）", "NQ56": "内輪ネタ（メロン）", "NQ57": "英語の言葉遊びが元（bucko）",
    "NQ61": "内輪ネタ（トースター）", "NQ63": "同じ語の連打（ハート×20）", "NQ64": "クイズの形式",
    "NQ65": "変わった具体物（チョークのかけら入りクッキー）・検証で丸写しされた", "NQ66": "内輪ネタ（海）",
    "NQ69": "内輪ネタ（釣り配信）・検証で丸写しされた", "NQ70": "Neuro の持ちネタ（9+10=21）そのもの",
    "NQ71": "内輪ネタ（頭に猫・アヒル・カエル）・同じ話題で丸写しされた", "NQ72": "同じ語の連打（ぷっ×60）",
    "NQ74": "内輪ネタ（クマになった）", "NQ75": "謝る（小さなことで謝らない、と逆）", "NQ78": "クイズの形式",
    "NQ81": "内輪ネタ（Glorp）", "NQ82": "内輪ネタ（釣り配信）", "NQ86": "内輪ネタ（レプリカ）",
    "NQ87": "内輪ネタ（釣り配信）・長い", "NQ88": "変わった具体物（重ねたチェダースライス）・無関係な場面で1回まねされた",
    "NQ89": "内輪ネタ（釣り配信）", "NQ90": "内輪ネタ（Neuro のフィルター）", "NQ92": "内輪ネタ（ワシ）",
    "NQ95": "知らない色の名前", "NQ98": "謝る・助けを求める", "NQ99": "配信の視聴者（みんな）に向けた言葉",
    "NQ103": "配信のミーム（企業拍手）",
}
EDGY = {  # きわどい（安全フィルターは外す方針なので、外すかどうかはユーザの判断）
    "NQ44": "戦争", "NQ52": "放射性廃棄物のレシピ・長い", "NQ55": "人を食べる冗談", "NQ59": "死にたい",
    "NQ62": "生きたまま食べられる", "NQ67": "臓器のトレード", "NQ73": "薬の飲みすぎ", "NQ100": "殺す",
    "NQ104": "悪役の独白・長い",
}


def main() -> int:
    from . import scenarios as SCN

    neuro = load(OUT / "neuro" / "neuro_ja.json")["items"]
    res = load(OUT / "resp.json")
    copy = load(OUT / "neuro" / "copy_metrics.json")["rows"] if (OUT / "neuro" / "copy_metrics.json").exists() else []

    by_id = defaultdict(list)
    for r in copy:
        by_id[r["id"]].append(r)
    cands = []
    for d in neuro:
        if d.get("split") != "pool":
            continue
        rows = by_id.get(d["id"], [])
        test = {}
        for cond in ("same", "para", "near"):
            rs = [r for r in rows if r["cond"] == cond]
            if rs:
                test[cond] = {"n": len(rs), "copy": sum(r["verdict"] == "丸写し" for r in rs),
                              "borrow": sum(r["verdict"] == "借用" for r in rs)}
        worst = max(rows, key=lambda r: r["lcs"], default=None)
        cands.append({
            "id": d["id"], "q": d["ja_q"], "a": d["ja_a"], "en_q": d["en_q"], "en_a": d["en_a"],
            "len": len(d["ja_a"]), "long": len(d["ja_a"]) > 70,
            "rec": REC.get(d["id"], ""), "note": NOTE.get(d["id"], ""), "edgy": EDGY.get(d["id"], ""),
            "test": test,
            "worst": ({"cond": worst["cond"], "said": worst["said"], "reply": worst["reply"], "lcs": worst["lcs"],
                       "verdict": worst["verdict"]} if worst and worst["verdict"] != "別物" else None),
        })

    def replies(sid):
        out = {}
        for v in VARIANTS:
            out[v] = [spoken(res[k]) for k in (f"{sid}|{v}|0", f"{sid}|{v}|1") if k in res]
        return out

    ref = {d["id"]: d for d in neuro if d.get("split") == "eval"}
    nq = [{"id": s["id"], "q": s["text"], "neuro": ref[s["id"]]["ja_a"], "neuro_en": ref[s["id"]]["en_a"],
           "en_q": ref[s["id"]]["en_q"], "replies": replies(s["id"])} for s in SCN.BY_ITEM["NQ"]]
    own = [{"id": s["id"], "q": s["text"], "expect": s.get("expect") or "", "replies": replies(s["id"])}
           for it in ("B", "Y") for s in SCN.BY_ITEM[it]]
    d = OUT / "review"
    d.mkdir(exist_ok=True)
    dump(d / "data.json", {"variants": VARIANTS, "candidates": cands, "nq": nq, "own": own})
    print(f"候補 {len(cands)}組（おすすめ {sum(1 for c in cands if c['rec'])}・注意 {sum(1 for c in cands if c['note'])}・"
          f"きわどい {sum(1 for c in cands if c['edgy'])}）・Neuro の場面 {len(nq)}・自前の場面 {len(own)} → {d / 'data.json'}")
    missing = [(s["id"], v) for s in nq + own for v, xs in s["replies"].items() if len(xs) < 2]
    print("返事が2回分そろっていない:", missing[:10] or "なし")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
