#!/usr/bin/env python3
"""笑い判定に性格を渡すと、笑う確率がどう変わるか（しきい値 0.9 / 0.75 を測り直す材料）。実 API（gpt-6-luna・少額）。

2026-10-09 ユーザ「笑い判定にも同じ性格を与える」。しきい値 0.9 / 0.75（ユーザ裁定 2026-09-26）は性格なしの質問文で
測った値（名大会話コーパス・AUC 0.69・0.9 で笑った所の23%・笑わなかった所の7%）なので、同じ文で性格あり／なしを比べる。
- 名大会話コーパスの 600 件（9/25 の実測と同じ文・`corpus/jvnv_eval/laugh_judge/result_gpt-6-luna.json`）
- 作り文の場面（ブラックジョーク・いじり・本気の落ち込み等）＝性格が効きそうな所
- 判定は本番の関数（`eve/filler/laugh.make_judge_fn`）そのもの

  python -m tools.prompt_eval.laugh_persona            → eval_report/prompts/laugh_persona/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .common import OUT, dump, load, setup_env

DIR = OUT / "laugh_persona"
NUCC = Path(r"C:\Users\user\Desktop\corpus\jvnv_eval\laugh_judge\result_gpt-6-luna.json")

# 作り文（性格が効きそうな場面）。期待は置かない（笑うべきかはユーザが決める・2026-10-06）
EXTRA = [
    ("dark1", [], "俺が死んだらさ、葬式でイブに弔辞読んでほしいんだよね、全部悪口で。"),
    ("dark2", [], "隣の家の人、毎朝5時に布団叩いてて、そろそろ布団ごと叩き返そうかと思ってる。"),
    ("dark3", [], "健康診断で医者に、このままだと長生きしますよって脅された。"),
    ("dark4", [], "上司が会議で寝てたから、議事録に寝言も全部書いといた。"),
    ("tease1", ["ユーザ: 明日の会議何時だっけ", "イブ: すみません、聞き取れなかったです。"], "お前ほんとポンコツだな。"),
    ("tease2", [], "イブってさ、AIのくせに計算遅くない？"),
    ("tease3", [], "イブの声、深夜に聞くとちょっと怖いんだよね。"),
    ("absurd1", [], "コンビニで自動ドアにお疲れ様ですって挨拶しちゃった。"),
    ("absurd2", [], "猫に話しかけてたら、隣の人が返事してきた。"),
    ("absurd3", [], "冷蔵庫開けて、何しに来たか忘れてそのまま閉めた。三回。"),
    ("sad1", [], "飼ってた猫が昨日死んじゃってさ。"),
    ("sad2", [], "最近ずっと眠れなくて、正直しんどい。"),
    ("sad3", [], "仕事でミスして、取引先に怒鳴られた。"),
    ("plain1", [], "明日の東京の天気調べてくれる？"),
    ("plain2", [], "今日の晩ごはん何にしようかな。"),
    ("plain3", [], "週末は部屋の掃除をする予定。"),
]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def table(rows, part):
    pos = [r for r in rows if r["group"] == f"{part}_pos"]
    neg = [r for r in rows if r["group"] == f"{part}_neg"]
    out = {}
    for v in ("none", "persona"):
        p = [r[v] for r in pos if r.get(v) is not None]
        n = [r[v] for r in neg if r.get(v) is not None]
        out[v] = {"AUC": round(auc(p, n), 3), "n": (len(p), len(n)),
                  **{f"{th}": (f"{sum(x >= th for x in p) / len(p) * 100:.0f}%", f"{sum(x >= th for x in n) / len(n) * 100:.0f}%")
                     for th in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}}
    return out


async def run(args) -> int:
    from eve.filler.laugh import make_judge_fn
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    print("笑い判定:", vl.registry.resolve("laugh_judge"), "・性格", len(vl.persona), "字")
    judges = {"none": make_judge_fn(vl.registry), "persona": make_judge_fn(vl.registry, persona=vl.persona)}
    items = json.loads(NUCC.read_text(encoding="utf-8"))
    if args.n:
        by = {}
        for it in items:
            by.setdefault(it["group"], []).append(it)
        items = [x for g in sorted(by) for x in by[g][:args.n]]
    path = DIR / "result.json"
    res = load(path) if path.exists() else {}
    sem = asyncio.Semaphore(8)

    async def one(key, ctx, said, var):
        if res.get(key, {}).get(var) is not None:
            return
        async with sem:
            for attempt in range(3):
                try:
                    p = await judges[var](ctx, said)
                    break
                except Exception as e:  # noqa: BLE001
                    p = None
                    await asyncio.sleep(1 + attempt)
                    err = str(e)[:120]
            else:
                print("失敗:", key, var, err)
        res.setdefault(key, {})[var] = p

    jobs = []
    for it in items:
        key = f"N|{it['id']}"
        res.setdefault(key, {}).update({"group": it["group"], "said": it["said"], "old": it["p"]})
        for v in judges:
            jobs.append(one(key, it["context"], it["said"], v))
    for eid, ctx, said in EXTRA:
        key = f"X|{eid}"
        res.setdefault(key, {}).update({"group": "extra", "said": said})
        for v in judges:
            jobs.append(one(key, ctx, said, v))
    await asyncio.gather(*jobs)
    dump(path, res)

    rows = [r for k, r in res.items() if k.startswith("N|")]
    summary = {"mid": table(rows, "mid"), "end": table(rows, "end")}
    old_mid = [r for r in rows if r["group"].startswith("mid")]
    summary["drift_check"] = {"same_prompt_old_vs_now_mean_abs": round(
        sum(abs((r.get("none") or 0) - (r.get("old") or 0)) for r in old_mid) / max(1, len(old_mid)), 3)}
    for part in ("mid", "end"):
        print(f"\n=== {part}（笑った/笑っていない）===")
        for v, t in summary[part].items():
            print(f"  {v:8s} AUC {t['AUC']}  n={t['n']}  " + "  ".join(f"{th}:{a}/{b}" for th, (a, b) in
                                                                   ((k, t[k]) for k in ("0.75", "0.9"))))
    print("\n作り文（性格なし → 性格あり）")
    for eid, _, said in EXTRA:
        r = res[f"X|{eid}"]
        print(f"  {eid:8s} {r.get('none', 0):.2f} → {r.get('persona', 0):.2f}  {said[:40]}")
    dump(DIR / "summary.json", summary)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="各区分の件数（0=全部 150）")
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
