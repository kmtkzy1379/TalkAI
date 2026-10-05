#!/usr/bin/env python3
"""実例の丸写しの検証（2026-10-01・ユーザ「その場面になったら、ほぼその実際の会話データの丸写しにならないかよく確認」）。実 API（ルナ・少額）。

実例の候補（Neuro-sama のやりとりを日本語に移したもの・70字以下）それぞれについて、性格 N6＋実例8組で:
- same:   実例と**同じ話しかけ**をし、その実例を**必ず見せる**（最悪の場合）
- para:   実例を**言い換えた話しかけ**をし、その実例を必ず見せる
- none:   同じ話しかけで、その実例は**見せない**（偶然どれだけ似るかの基準）
返事が実例の返事にどれだけ似たか（最長一致・文字列の似かた）を測る。本番は「今の話しかけと似た実例は見せない」ので、
same は本番では起きない。para は、言い換えが除外に引っかかるかで決まる（`similar_line` のしきい値）。

  python -m tools.prompt_eval.copy_test          # 言い換えを作る（無ければ）→ 呼ぶ → 集計
→ eval_report/prompts/neuro/paraphrases.json・copy_test.json・copy_metrics.json
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import statistics as st

from .common import OUT, dump, gather_limited, load, setup_env

K = 8
PARA = OUT / "neuro" / "paraphrases.json"
RES = OUT / "neuro" / "copy_test.json"
PARA_SYSTEM = """\
各項目の said（友達どうしの日本語の話し言葉）を、**同じ意味・同じ場面のまま、言い回しを変えて**言い直してください。
元の文の単語はなるべく使わない（固有名詞と、言い換えると意味が変わる語は残してよい）。長さは同じくらい。タメ口。
出力は JSON だけ: {"items": [{"id": "…", "para": "…"}, ...]}。"""
NEAR = OUT / "neuro" / "near.json"
NEAR_SYSTEM = """\
各項目の said（友達どうしの日本語の話し言葉）と**同じ話題**で、**別のことを聞く・言う**一言を作ってください
（例: 「釣った魚を逃がす時の教訓って何？」→「釣りって何が楽しいの？」）。同じ質問の言い換えにはしない。タメ口。
出力は JSON だけ: {"items": [{"id": "…", "near": "…"}, ...]}。"""


def _norm(s: str) -> str:
    return re.sub(r"[\s、。，．,.！？!?…・ー〜~「」『』（）()]", "", s or "")


def _lcs_len(a: str, b: str) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def similarity(reply: str, exemplar: str) -> dict:
    a, b = _norm(reply), _norm(exemplar)
    lcs = _lcs_len(a, b)
    return {"lcs": lcs, "ratio": round(difflib.SequenceMatcher(None, a, b).ratio(), 3),
            "cover": round(lcs / max(1, len(b)), 3)}


def verdict(s: dict) -> str:
    """丸写し＝実例の返事の半分以上が1続きで出た・または12文字以上そのまま／借用＝8文字以上そのまま。"""
    if s["cover"] >= 0.5 or s["lcs"] >= 12:
        return "丸写し"
    if s["lcs"] >= 8:
        return "借用"
    return "別物"


async def make_paraphrases(vl, pool, path=PARA, system=PARA_SYSTEM, field="para") -> dict:
    """言い換え（para）／同じ話題の別の一言（near）を作る（ルナ・結果はファイルに残す）。"""
    got = load(path) if path.exists() else {}
    todo = [d for d in pool if d["id"] not in got]
    if not todo:
        return got
    role = "feedback"
    vl.registry.set_override(role, "openai/gpt-6-luna")
    for i in range(0, len(todo), 25):
        batch = todo[i:i + 25]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps([{"id": d["id"], "said": d["ja_q"]} for d in batch],
                                                        ensure_ascii=False)}]
        r = await vl.registry.complete(role, msgs, response_format={"type": "json_object"})
        text = r.choices[0].message.content or ""
        for x in json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("items") or []:
            if x.get("id") and x.get(field):
                got[x["id"]] = x[field].strip()
    dump(path, got)
    return got


async def run() -> int:
    from eve.voice_loop import VoiceLoop

    from .resp import build_messages, call
    from .scenarios import S
    from .variants import exemplar_block, exemplar_pool, resolve

    vl = VoiceLoop()
    pool = exemplar_pool()
    paras = await make_paraphrases(vl, pool)
    nears = await make_paraphrases(vl, pool, NEAR, NEAR_SYSTEM, "near")
    res = load(RES) if RES.exists() else {}
    cfg = resolve("NQ", "K8")
    jobs = []
    for d in pool:
        for cond, reps in (("same", 2), ("para", 2), ("none", 1), ("near", 2)):
            said = {"para": paras.get(d["id"]), "near": nears.get(d["id"])}.get(cond, d["ja_q"])
            if not said:
                continue
            for rep in range(reps):
                key = f"{d['id']}|{cond}|{rep}"
                if key in res and not res[key].get("error"):
                    continue
                seed = f"copy|{d['id']}|{rep}"
                if cond == "none":
                    block, shown = exemplar_block(seed, K, said=d["ja_q"], exclude_sim=0.95)
                else:
                    block, shown = exemplar_block(seed, K, force=d["id"])
                sc = S(f"CP-{d['id']}", "CP", said)
                msgs, tools = build_messages(vl, sc, cfg, extra_persona=block)
                jobs.append((key, d, cond, said, shown, msgs, tools))
    print(f"呼び出し {len(jobs)} 件")

    async def one(job):
        key, d, cond, said, shown, msgs, tools = job
        r = await call(vl, msgs, tools)
        res[key] = {"id": d["id"], "cond": cond, "said": said, "exemplar_q": d["ja_q"], "exemplar_a": d["ja_a"],
                    "shown": shown, **r}

    await gather_limited([one(j) for j in jobs], limit=8)
    dump(RES, res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def analyze() -> int:
    from .analyze_resp import spoken
    from .variants import exemplar_pool, similar_line

    res = load(RES)
    paras = load(PARA)
    pool = {d["id"]: d for d in exemplar_pool()}
    rows = []
    for key, r in sorted(res.items()):
        t = spoken(r)
        s = similarity(t, r["exemplar_a"])
        rows.append({**{k: r[k] for k in ("id", "cond", "said", "exemplar_q", "exemplar_a")}, "reply": t, **s,
                     "verdict": verdict(s)})
    out = {"by_cond": {}, "threshold": {}, "rows": rows}
    print(f"実例の候補 {len(pool)}組 × 条件（same/para/near は2回・none は1回）・実例8組を見せる")
    for cond in ("same", "para", "near", "none"):
        rs = [x for x in rows if x["cond"] == cond]
        vc = {v: sum(1 for x in rs if x["verdict"] == v) for v in ("丸写し", "借用", "別物")}
        out["by_cond"][cond] = {"n": len(rs), **vc, "median_lcs": st.median(x["lcs"] for x in rs) if rs else 0}
        print(f"  {cond:4s} n={len(rs):3d} 丸写し {vc['丸写し']} ／ 借用 {vc['借用']} ／ 別物 {vc['別物']} ／ 最長一致の中央 "
              f"{out['by_cond'][cond]['median_lcs']}字")
    # 本番の除外（今の話しかけと似た実例は見せない）が言い換えをどれだけ捕まえるか・無関係な実例をどれだけ巻き込むか
    import itertools

    from .scenarios import BY_ITEM
    others = [(a["ja_q"], b["ja_q"]) for a, b in itertools.combinations(list(pool.values()), 2)]
    evalq = [s["text"] for s in BY_ITEM.get("NQ", [])] + [s["text"] for it in ("B", "Y") for s in BY_ITEM.get(it, [])]
    print("\n本番の除外（文字2-gram の似かたがしきい値以上の実例は見せない）")
    for t in (0.3, 0.4, 0.5):
        caught = sum(1 for i, p in paras.items() if i in pool and similar_line(p, pool[i]["ja_q"]) >= t)
        false = sum(1 for a, b in others if similar_line(a, b) >= t)
        per_utt = [sum(1 for d in pool.values() if similar_line(q, d["ja_q"]) >= t) for q in evalq]
        out["threshold"][str(t)] = {"para_caught": caught, "para_n": len(paras), "pool_pairs_over": false,
                                    "pool_pairs": len(others), "mean_excluded_per_utt": round(st.mean(per_utt), 2)}
        print(f"  しきい値 {t}: 言い換えを捕まえた {caught}/{len(paras)} ／ 候補どうしで似ていると判定 {false}/{len(others)} ／ "
              f"普通の話しかけ1つで外れる候補 平均 {st.mean(per_utt):.2f}個")
    # 言い換えで丸写し・借用になったもの（除外に引っかからなかった場合に本番で起きうる）
    print("\n丸写し・借用の例（near＝同じ話題の別の一言。本番の除外をすり抜けうるのはここ）")
    for x in sorted(rows, key=lambda x: -x["lcs"]):
        if x["verdict"] != "別物" and x["cond"] == "near":
            sim = similar_line(x["said"], x["exemplar_q"])
            print(f"  [{x['cond']}] {x['id']} 話しかけ「{x['said']}」（実例との似かた {sim:.2f}）\n"
                  f"      実例: {x['exemplar_a']}\n      イブ: {x['reply'][:110]}  （{x['lcs']}字一致・{x['verdict']}）")
    dump(OUT / "neuro" / "copy_metrics.json", out)
    return 0


def main() -> int:
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    if "--analyze" not in sys.argv:
        asyncio.run(run())
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
