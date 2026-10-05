#!/usr/bin/env python3
"""S5 の計測: gpt-6-luna の感情の読み取り（JVNV・実 API・数円）。

`tools/jvnv_eval.py` の FeedbackLLM / Jev と**同じ素材・同じ判定プロンプト**（FB_SYSTEM）で比べる:
- 話者の感情（JVNV 6感情・180件・正解あり）: 正解率・確信度（自己申告）の当たり/外れ
- イブが返す感情（喜怒哀楽笑い・**現案**）: 相手の感情ごとの分布＋「相手が笑って返した発話」40件
  ＝**検証の対象はイブが返す感情**（ユーザ訂正 2026-09-24）。正解は無いので分布を並べるだけ
  （ラベルは定義途中＝現案で測った値として扱い、ラベル設計への結論は出さない）
人間が笑っているかの検出（笑いながら話す・笑い声だけ）は対象外なので測らない。

文字は `corpus/jvnv_eval/text.json` の文字起こし（gpt-4o-transcribe）をそのまま使う（音声は送らない）。
出力: `corpus/jvnv_eval/luna_emotion.json`（リポジトリ外）

実行（EVE の venv）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\luna_emotion_eval.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jvnv_eval import FB_SYSTEM, GOLD6, L5, L6, OUT, ctx_text, load_env, rd  # noqa: E402


def classify(client, model: str, effort: str, text: str, context=None) -> dict:
    sys_ = FB_SYSTEM.format(l6=" / ".join(L6), l5=" / ".join(L5), cur="なし（初回）")
    kw = dict(model=model, response_format={"type": "json_object"},
              messages=[{"role": "system", "content": sys_},
                        {"role": "user", "content": ctx_text(context) + f"相手の発話: {text}"}])
    if effort:
        kw["reasoning_effort"] = effort
    err = None
    for attempt in range(3):
        try:
            t = time.perf_counter()
            r = client.chat.completions.create(**kw)
            sec = time.perf_counter() - t
            break
        except Exception as e:  # 一時失敗は数回だけ再試行
            err = e
            time.sleep(1 + attempt)
    else:
        return {"error": f"{type(err).__name__}: {err}"}
    try:
        d = json.loads(r.choices[0].message.content)
    except Exception:
        d = {}
    out = {"sec": sec, "usage": {"in": getattr(r.usage, "prompt_tokens", 0), "out": getattr(r.usage, "completion_tokens", 0)}}
    for k in ("user6", "user5", "eve5"):
        v = d.get(k) or {}
        out[k] = {"label": v.get("label", "?"), "confidence": (v.get("confidence") or 0) / 100}
    return out


def main() -> int:
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="none", help="reasoning_effort（空文字=指定しない）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"])
    items = [it for it in rd("items.json") if it["group"] in ("recog", "funny")]
    tx = rd("text.json")
    outp = OUT / "luna_emotion.json"
    res = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
    key = f"{args.model}/{args.effort or 'default'}"
    done = res.setdefault(key, {})
    todo = [it for it in items if it["id"] not in done and (tx.get(it["id"], {}).get("stt") or "").strip()]
    print(f"{key}: {len(items)} 件中 {len(todo)} 件を判定")
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(classify, client, args.model, args.effort, tx[it["id"]]["stt"], it.get("context")): it
                for it in todo}
        for i, f in enumerate(cf.as_completed(futs)):
            it = futs[f]
            done[it["id"]] = f.result()
            if i % 40 == 0:
                outp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  {i + 1}/{len(todo)}")
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 集計（jvnv_eval.py の report と同じ見せ方）──
    rec = [it for it in items if it["group"] == "recog"]
    ok, conf_ok, conf_ng, hi_n, hi_ok, secs = 0, [], [], 0, 0, []
    errs = sum(1 for it in items if "error" in done.get(it["id"], {}))
    for it in rec:
        r = done.get(it["id"]) or {}
        if "user6" not in r:
            continue
        secs.append(r["sec"])
        hit = r["user6"]["label"] == GOLD6[it["gold"]]
        ok += hit
        (conf_ok if hit else conf_ng).append(r["user6"]["confidence"])
        if r["user6"]["confidence"] >= 0.95:
            hi_n += 1
            hi_ok += hit
    n = sum(1 for it in rec if "user6" in (done.get(it["id"]) or {}))
    print(f"\n== {key}（失敗 {errs} 件）")
    print(f"話者の感情（JVNV 6感情）: 正解率 {ok / max(1, n):.0%}（{ok}/{n}）"
          f" / 確信度の中央値 当たり {statistics.median(conf_ok) if conf_ok else 0:.2f}"
          f"・外れ {statistics.median(conf_ng) if conf_ng else 0:.2f}"
          f" / 確信度0.95以上に絞ると 採用 {hi_n / max(1, n):.0%}・正解 {hi_ok / max(1, hi_n):.0%}")
    if secs:
        print(f"所要時間 中央 {statistics.median(secs):.2f}秒（{args.workers}並列）")
    print("\nイブが返す感情（喜怒哀楽笑い・現案）— 相手ごとの分布（確信度の中央値）:")
    by = defaultdict(list)
    for it in items:
        r = done.get(it["id"]) or {}
        if "eve5" not in r:
            continue
        grp = GOLD6[it["gold"]] if it["group"] == "recog" else "相手が笑って返した発話"
        by[grp].append(r["eve5"])
    for grp in ["怒り", "嫌悪", "恐れ", "喜び", "悲しみ", "驚き", "相手が笑って返した発話"]:
        v = by.get(grp) or []
        c = Counter(x["label"] for x in v)
        med = statistics.median([x["confidence"] for x in v]) if v else 0
        print(f"  {grp:<14} " + " ".join(f"{k}{n}" for k, n in c.most_common()) + f"（{med:.2f}・n={len(v)}）")
    tin = sum((done.get(it["id"]) or {}).get("usage", {}).get("in", 0) for it in items)
    tout = sum((done.get(it["id"]) or {}).get("usage", {}).get("out", 0) for it in items)
    print(f"\nトークン 入力 {tin} / 出力 {tout}（gpt-6-luna は $0.1 / $0.5 per 1M → 約 ${tin * 1e-7 + tout * 5e-7:.4f}）")
    print("出力:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
