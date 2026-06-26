"""RAG 検索精度の一次診断（実 rag_memory.jsonl + 実 ruri 埋め込み・API不要）。

ユーザ実機ログの「記憶の引っ掛かりが弱い」を再現/反証する。
- ストア統計（件数・prediction_diff 有無・summary 長・タグ頻度）。
- 実機セッションのクエリで search_debug を回し、何が retrieve され何が floor 除外されたかを
  relevance スコア付きで出す（チーズケーキ記憶が浮上するか／floor/baseline が高すぎないか）。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\rag_diag.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402
from eve.memory.long_term import RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402


def _short(s, n=42):
    s = (s or "").replace("\n", " / ")
    return s if len(s) <= n else s[:n] + "…"


async def main() -> None:
    store = RagStore(make_embedder())
    await store.warmup()
    await store.initialize()
    chunks = list(store._chunks)  # 診断目的の直接参照
    print(f"=== ストア統計 ===")
    print(f"件数: {len(chunks)}  floor={Config.RAG_RELEVANCE_FLOOR} baseline={Config.RAG_REL_BASELINE} "
          f"w_rel={Config.RAG_W_REL} top_k={Config.RAG_TOP_K}")
    have_pd = sum(1 for c in chunks if isinstance(c.get("prediction_diff"), (int, float)))
    slens = [len((c.get("summary") or "")) for c in chunks]
    tlens = [len((c.get("text") or "")) for c in chunks]
    print(f"prediction_diff 有: {have_pd}/{len(chunks)}  "
          f"summary長 avg={sum(slens)//max(1,len(slens))} max={max(slens or [0])}  "
          f"text長 avg={sum(tlens)//max(1,len(tlens))}")
    tags = Counter()
    for c in chunks:
        for t in (c.get("topic_tags") or []):
            tags[t] += 1
    print("頻出タグ:", ", ".join(f"{k}×{v}" for k, v in tags.most_common(20)))
    print()
    # 食べ物/チーズケーキ系の記憶が実在するか（summary/tags/text に含むもの）
    print("=== 食べ物/甘いもの系の記憶（summary 抜粋） ===")
    for i, c in enumerate(chunks):
        blob = (c.get("summary", "") or "") + " " + " ".join(c.get("topic_tags") or []) + " " + (c.get("text", "") or "")
        if any(w in blob for w in ["チーズ", "ケーキ", "甘い", "スイーツ", "おやつ", "食べ", "ご飯", "晩"]):
            print(f"  [{i}] sum={_short(c.get('summary'))}  tags={c.get('topic_tags')}  pd={c.get('prediction_diff')}")
    print()

    queries = [
        "甘いもの おすすめ",
        "前にどんな食べ物調べてた",
        "チーズケーキ",
        "おやつ何食べよう",
        "スターデューバレー ゲーム おすすめ",
        "前に食べた晩ご飯",
    ]
    for q in queries:
        dbg = await store.search_debug(q, k=Config.RAG_TOP_K)
        sel = dbg["selected"]
        info = dbg["info"]
        print(f"=== Q: {q!r}  (total={info['total']} floor除外={info['floor_excluded']} 選出={len(sel)}) ===")
        for s in sel:
            print(f"  ✓ rel={s['relevance']:.3f} cos={s['cos']:.3f} base={s['base']:.3f}  {_short(s['chunk'].get('summary') or s['chunk'].get('text'))}")
        # floor 直下で惜しく落ちた上位（baseline/floor が厳しすぎないかの判断材料）
        pool = info.get("pool") or []
        if not sel and not pool:
            # pool は floor 通過後なので、floor で全滅した時の最大 cos を別途出す
            qv = await store._embedder.embed_query(q)
            import numpy as np
            qn = np.asarray(qv, float); qn = qn/(np.linalg.norm(qn) or 1)
            best = []
            for c in chunks:
                e = np.asarray(c["embedding"], float); e = e/(np.linalg.norm(e) or 1)
                cos = float(e@qn)
                rel = max(0.0, (cos - Config.RAG_REL_BASELINE)/max(1e-9, 1-Config.RAG_REL_BASELINE))
                best.append((rel, cos, c))
            best.sort(key=lambda x: -x[0])
            print("   floor で全滅。惜しかった上位3:")
            for rel, cos, c in best[:3]:
                print(f"     rel={rel:.3f} cos={cos:.3f}  {_short(c.get('summary') or c.get('text'))}")
        print()
    await store.shutdown()


asyncio.run(main())
