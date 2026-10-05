#!/usr/bin/env python3
"""実例の除外を意味の近さ（Ruri）で判定できるかの確かめ。API 不要（ローカルの Ruri）。

文字2-gram では言い換えを 50件中 3〜19件しか捕まえられなかった（copy_test）。Ruri の余弦で:
- 言い換え（同じ場面）: 実例の話しかけとの類似度 → しきい値以上なら捕まえた
- 普通の話しかけ（評価の場面すべて）: 実例の候補と似すぎて巻き込まれる数（多いと実例が減る）

  python -m tools.prompt_eval.exclude_ruri → eval_report/prompts/neuro/exclude_ruri.json
"""
from __future__ import annotations

import asyncio
import statistics as st

from .common import OUT, dump, load, setup_env


async def run() -> int:
    from eve.memory.embed.ruri import RuriEmbedder

    from . import scenarios as SCN
    from .variants import exemplar_pool

    pool = exemplar_pool()
    paras = load(OUT / "neuro" / "paraphrases.json")
    normal = sorted({s["text"] for s in SCN.ALL if s.get("kind", "user") == "user" and s["item"] != "CP"})
    emb = RuriEmbedder()
    # 話しかけどうしを比べる（どちらも同じ「検索クエリ」側で埋める＝対称）
    qv = await emb.embed_query_batch([d["ja_q"] for d in pool]) if hasattr(emb, "embed_query_batch") else \
        [await emb.embed_query(d["ja_q"]) for d in pool]
    pv = {d["id"]: await emb.embed_query(paras[d["id"]]) for d in pool if d["id"] in paras}
    nv = [await emb.embed_query(t) for t in normal]

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))

    para_sim = {d["id"]: cos(pv[d["id"]], qv[i]) for i, d in enumerate(pool) if d["id"] in pv}
    # 言い換えが、本来の実例より別の実例に近くなっていないか
    rank_ok = sum(1 for i, d in enumerate(pool) if d["id"] in pv and
                  max(range(len(pool)), key=lambda j: cos(pv[d["id"]], qv[j])) == i)
    normal_max = [max(cos(v, q) for q in qv) for v in nv]
    out = {"para_sim": para_sim, "normal_n": len(normal), "rank_top1": rank_ok, "thresholds": {}}
    xs = sorted(para_sim.values())
    print(f"言い換えと元の話しかけの類似度: 最小 {xs[0]:.3f} ／ 中央 {st.median(xs):.3f} ／ 最大 {xs[-1]:.3f}"
          f"（{len(xs)}件）・一番近い実例が元の実例だった {rank_ok}/{len(xs)}")
    ys = sorted(normal_max)
    print(f"普通の話しかけ {len(normal)}個の、実例の候補への最大類似度: 中央 {st.median(ys):.3f} ／ 90% {ys[int(len(ys) * .9)]:.3f} ／ 最大 {ys[-1]:.3f}")
    for t in (0.85, 0.87, 0.88, 0.90, 0.92):
        caught = sum(1 for s in para_sim.values() if s >= t)
        hit = [(normal[i], m) for i, m in enumerate(normal_max) if m >= t]
        excl = [sum(1 for q in qv if cos(v, q) >= t) for v in nv]
        out["thresholds"][str(t)] = {"para_caught": caught, "para_n": len(para_sim), "normal_hit": len(hit),
                                     "mean_excluded": round(st.mean(excl), 3)}
        print(f"  しきい値 {t}: 言い換えを捕まえた {caught}/{len(para_sim)} ／ 普通の話しかけで何か外れた {len(hit)}/{len(normal)}"
              f"（1回あたり平均 {st.mean(excl):.2f}個外れる）")
        for u, m in hit[:6]:
            j = max(range(len(pool)), key=lambda k: cos(nv[normal.index(u)], qv[k]))
            print(f"      {m:.3f} 「{u[:30]}」 ≒ 実例「{pool[j]['ja_q'][:30]}」")
    dump(OUT / "neuro" / "exclude_ruri.json", out)
    return 0


def main() -> int:
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
