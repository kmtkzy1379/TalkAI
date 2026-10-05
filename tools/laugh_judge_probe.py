#!/usr/bin/env python3
"""Jev の笑い判定を名大会話コーパスで実測する（eve-v2 の .venv で実行）。

ユーザ裁定 2026-09-25「笑いの出し方」:
  (a) 話の途中で声に出して笑う … 確率 ≥ 0.85（仮）
  (b) 話し終えた後にフィラー・応答を笑いにする … 発話中に ≥ 0.70（仮）
正解（人間が実際に笑った位置）:
  (a) 話の途中: ターン内に埋め込まれた聞き手の笑い「（＜笑い＞）」の直前の文
  (b) 話し終わり: 次のターンが相手の ＜笑…＞ で始まる、そのターンの最後の文
  笑っていない: 同じ条件で、直後に聞き手の笑いが無い文
Jev には、直前の会話（最大10ターン）＋ 現在のターンのその文までを渡し、Noul で「笑うか」を聞く。
名大会話コーパスは CC BY-NC-ND。素材と結果はリポジトリの外（corpus/jvnv_eval/laugh_judge/）。
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_nucc import load  # noqa: E402
from jvnv_eval import NUCC, OUT, load_env, nucc_clean  # noqa: E402

OUTDIR = OUT / "laugh_judge"
SENT_END = re.compile(r"(?<=[。？！?!])")
PAREN = re.compile(r"（[^）]*）")
BAD = re.compile(r"[＊%％]")  # 伏せ字・注記を含む文は使わない

QUESTION = ("相手の話のこの時点で、聞き手のイブが思わず笑う場面である"
            "（面白い・微笑ましい・呆れて笑う・気まずさを笑ってごまかす など。"
            "相手がつらい話・悲しい話をしているなど、笑うと不適切な場面では笑わない）")


def samples():
    """(a)/(b) × 笑った/笑っていない の候補を作る。"""
    out = {"mid_pos": [], "mid_neg": [], "end_pos": [], "end_neg": []}
    for fname, turns in load(NUCC):
        for k, t in enumerate(turns):
            raw = t["raw"]
            ctx = [f"{x['spk']}: {nucc_clean(x['raw'])}" for x in turns[max(0, k - 10):k] if nucc_clean(x["raw"])]
            # 文の区切りと、その直後の聞き手の反応を並べる
            pieces, pos = [], 0
            for m in PAREN.finditer(raw):
                pieces.append((raw[pos:m.start()], m.group(0)))
                pos = m.end()
            pieces.append((raw[pos:], None))
            said = ""
            for seg, paren in pieces:
                sents = [s for s in SENT_END.split(seg) if s]
                for i, s in enumerate(sents):
                    said += s
                    cur = nucc_clean(said)
                    if len(cur) < 6 or BAD.search(s):
                        continue
                    last_in_seg = i == len(sents) - 1
                    if not re.search(r"[。？！?!]$", s) and not last_in_seg:
                        continue
                    item = {"id": f"{fname[:-4]}_{k}_{len(said)}", "context": ctx, "spk": t["spk"], "said": cur}
                    if last_in_seg and paren is not None:
                        (out["mid_pos"] if "笑" in paren else out["mid_neg"]).append(item)
                    elif last_in_seg and paren is None and k + 1 < len(turns) and turns[k + 1]["spk"] != t["spk"]:
                        nxt = turns[k + 1]["raw"]
                        (out["end_pos"] if nxt.startswith("＜笑") else out["end_neg"]).append(item)
                    elif re.search(r"[。？！?!]$", s):
                        out["mid_neg"].append(item)  # 文の途中の区切りで聞き手の反応が無い
    return out


def ask(key, it):
    body = {"model": "typesafe/jev-1.13",
            "state": {"直前の会話": it["context"], "相手（" + it["spk"] + "）が今話している途中の内容": it["said"]},
            "questions": {"laugh": {"type": "noul", "instructions": QUESTION}}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/alpha/decisions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.perf_counter()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["answers"]["laugh"]["noul"], time.perf_counter() - t, (d.get("usage") or {}).get("cost")
        except Exception as e:  # 一時失敗は数回だけ再試行
            err = e
            time.sleep(1 + attempt)
    raise err


def ask_llm(client, model, it):
    """LLM に yes/no の1語で答えさせ、logprobs から「笑う」確率を出す（reasoning_effort=none で約0.6秒）。"""
    msgs = [{"role": "system", "content": "yes か no の1語だけで答える。"},
            {"role": "user", "content": "直前の会話:\n" + "\n".join(it["context"])
             + f"\n\n相手（{it['spk']}）が今話している途中の内容:\n{it['said']}\n\n"
             + f"判定: {QUESTION}。そうなら yes、違うなら no。"}]
    t = time.perf_counter()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, logprobs=True, top_logprobs=5,
                                               reasoning_effort="none", max_completion_tokens=5)
            break
        except Exception as e:  # 一時失敗は数回だけ再試行
            err = e
            time.sleep(1 + attempt)
    else:
        raise err
    sec = time.perf_counter() - t
    import math
    top = r.choices[0].logprobs.content[0].top_logprobs
    py = sum(math.exp(x.logprob) for x in top if x.token.strip().lower() in ("yes", "はい"))
    pn = sum(math.exp(x.logprob) for x in top if x.token.strip().lower() in ("no", "いいえ"))
    return (py / (py + pn) if py + pn else 0.0), sec, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="各区分の件数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default="jev", help="jev / OpenAI のモデル名（例 gpt-5.4-mini）")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    env = load_env()
    if args.judge == "jev":
        key = env["OPENROUTER_API_KEY"]
        judge = lambda it: ask(key, it)  # noqa: E731
    else:
        from openai import OpenAI
        client = OpenAI(api_key=env["OPENAI_API_KEY"])
        judge = lambda it: ask_llm(client, args.judge, it)  # noqa: E731
    tag = "" if args.judge == "jev" else f"_{args.judge}"
    rng = random.Random(args.seed)
    pool = samples()
    print("候補: " + " / ".join(f"{k} {len(v)}" for k, v in pool.items()))
    picked = [(g, it) for g, v in pool.items() for it in rng.sample(v, min(args.n, len(v)))]
    # 判定器を比べるときは Jev と**同じ文**を使う
    prev = OUTDIR / "result.json"
    if args.judge != "jev" and prev.exists():
        picked = [(r["group"], {k: r[k] for k in ("id", "context", "spk", "said")})
                  for r in json.loads(prev.read_text(encoding="utf-8"))]

    res = []
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(judge, it): (g, it) for g, it in picked}
        for f in cf.as_completed(futs):
            g, it = futs[f]
            try:
                p, sec, cost = f.result()
            except Exception as e:
                print("  失敗:", e)
                continue
            res.append({**it, "group": g, "p": p, "sec": sec, "cost": cost})
    (OUTDIR / f"result{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["Jev の笑い判定（名大会話コーパス・正解＝人間の聞き手が実際に笑った位置）", f"質問: {QUESTION}", ""]
    for part, name in (("mid", "(a) 話の途中"), ("end", "(b) 話し終わり")):
        pos = sorted(r["p"] for r in res if r["group"] == f"{part}_pos")
        neg = sorted(r["p"] for r in res if r["group"] == f"{part}_neg")
        if not pos or not neg:
            continue
        # AUC（笑った文の方が確率が高い組の割合）
        auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
        L.append(f"=== {name}: 笑った {len(pos)} / 笑っていない {len(neg)}  AUC {auc:.2f} ===")
        L.append(f"  確率の中央値: 笑った {pos[len(pos)//2]:.2f} / 笑っていない {neg[len(neg)//2]:.2f}")
        L.append("  閾値    笑った文を拾えた率   笑っていない文で笑ってしまう率   （同数で混ぜたときの）笑いの当たり率")
        for th in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
            tp = sum(x >= th for x in pos)
            fp = sum(x >= th for x in neg)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            L.append(f"  {th:.2f}   {tp/len(pos)*100:5.0f}% ({tp:>3})        {fp/len(neg)*100:5.0f}% ({fp:>3})"
                     f"                   {prec*100:5.0f}%")
        L.append("")
    secs = sorted(r["sec"] for r in res)
    cost = sum(r["cost"] or 0 for r in res)
    L.append(f"所要時間の中央値 {secs[len(secs)//2]:.2f}秒（8並列で計測） / 費用 ${cost:.4f} / {len(res)} 件")
    L.append("\n--- 確率が高かった文（上位）と、笑ったのに低かった文 ---")
    for g in ("mid_pos", "mid_neg", "end_pos", "end_neg"):
        rs = sorted((r for r in res if r["group"] == g), key=lambda r: -r["p"])
        L.append(f"\n[{g}] 上位5")
        for r in rs[:5]:
            L.append(f"  {r['p']:.2f}  …{r['said'][-60:]}")
        if g.endswith("pos"):
            L.append(f"[{g}] 下位5")
            for r in rs[-5:]:
                L.append(f"  {r['p']:.2f}  …{r['said'][-60:]}")
    L.insert(0, f"判定器: {args.judge}")
    (OUTDIR / f"笑い判定_結果{tag}.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
