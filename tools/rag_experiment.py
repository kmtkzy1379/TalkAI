"""F3.5 RAG 実験ハーネス（Tier-2・実埋め込み）。

仮データ（FeedbackLLM 出力を模した会話記憶 ~16件）を seed し、代表クエリを流して
**どういう抽出になったか**（クエリ→候補スコア→フロア除外→top-1/MMR選出）を整形表示する。
ユーザがこのログを見て精度を判断し、Ruri/API・パラメータ(floor/λ/weights/k)を確定する。

使い方:
  $env:PYTHONIOENCODING="utf-8"
  python tools\rag_experiment.py --backend ruri    # ローカル日本語（初回はモデルDL）
  python tools\rag_experiment.py --backend openai  # API（要 OPENAI_API_KEY）

A/B は同じクエリで backend を変えて2回流し、抽出結果を見比べる。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402
from eve.memory import RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


# 仮データ: FeedbackLLM の出力を模した会話記憶（text=表示, search=埋め込み対象, emotions, importance, 経過日数）。
SEED = [
    ("夏にスイカを食べた話で盛り上がった。種飛ばし対決が楽しかった", "夏にスイカを食べて種飛ばしで遊んだ楽しい思い出", "楽しい・懐かしい", 0.6, 30),
    ("夏祭りの花火を一緒に見た。大きな音と光に感動した", "夏祭りで花火を見て感動した夜の思い出", "感動", 0.6, 28),
    ("暑い日にかき氷を食べた。いちご味がお気に入り", "暑い日にいちご味のかき氷を食べて涼んだ", "満足・涼しい", 0.4, 5),
    ("海に泳ぎに行った夏の話。日焼けして痛かったけど楽しかった", "夏に海水浴へ行って泳いだ。日焼けした", "爽快・楽しい", 0.5, 20),
    ("夏にプールで一日遊んだ。水が気持ちよかった", "夏のプールで遊んで気持ちよかった", "爽快", 0.5, 12),
    ("二郎系ラーメンの話で熱くなった。こってり背脂が好き", "二郎系のこってりラーメンが好きという話", "満足・情熱", 0.4, 3),
    ("仕事の締切に追われて徹夜した。かなりストレスだった", "仕事の締切に追われ徹夜してストレスを感じた", "ストレス・疲労", 0.7, 2),
    ("試験勉強が大変で不安だと話していた", "試験勉強が大変で不安を感じている", "不安", 0.5, 7),
    ("好きなバンドのライブに行った。最高に盛り上がった", "大好きなバンドのライブで盛り上がり興奮した", "興奮・幸福", 0.7, 10),
    ("京都へ旅行して紅葉を見た。とても綺麗で癒された", "京都旅行で紅葉を見て癒された秋の思い出", "癒し・感動", 0.5, 60),
    ("寒い冬に鍋を囲んで温まった話", "寒い冬にみんなで鍋を食べて温まった", "安心・温かい", 0.4, 90),
    ("飼っている猫がすごく可愛いと嬉しそうに話した", "飼い猫がとても可愛くて愛おしい", "癒し・愛しい", 0.6, 4),
    ("新作ゲームを徹夜でクリアした。夢中になった", "新作ゲームに夢中で徹夜してクリアした", "夢中・達成感", 0.5, 6),
    ("朝のコーヒーが一日の楽しみだと話した", "朝に飲むコーヒーが好きで落ち着く", "落ち着く", 0.3, 1),
    ("誕生日を祝ってもらってとても嬉しかった", "誕生日を皆に祝ってもらえて幸せだった", "幸福・感謝", 0.8, 45),
    ("雨が続いて散歩に行けず退屈だとこぼした", "雨続きで外に出られず退屈している", "退屈", 0.3, 8),
]

QUERIES = [
    "夏って感じだね、暑いなあ",          # 夏クラスタ（スイカ/花火/海/プール/かき氷）が連想されるか
    "お腹すいた、何か食べたいな",          # food クラスタ（ラーメン/かき氷/鍋/コーヒー）
    "最近ちょっと疲れててストレス溜まってる",  # ネガ（仕事/勉強）
    "音楽が聴きたい気分",                # 音楽ライブ
    "どこか旅行に行きたいなあ",            # 旅行/海
]


async def seed(store: RagStore) -> None:
    for text, search, emo, imp, ago in SEED:
        await store.add_chunk(
            text=text, search_text=search, emotions=emo,
            importance=imp, timestamp=_days_ago(ago),
        )


def _fmt(s: dict) -> str:
    c = s["chunk"]
    return (
        f"    rel={s['relevance']:.3f} imp={s['importance']:.2f} "
        f"rec={s['recency']:.3f} base={s['base']:.3f} | {c['text']}  "
        f"〔感情:{c.get('emotions')}〕"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=Config.EMBED_BACKEND, choices=["ruri", "openai"])
    ap.add_argument("--k", type=int, default=Config.RAG_TOP_K)
    ap.add_argument("--pool", action="store_true", help="候補プール（フロア通過の全件）も表示")
    args = ap.parse_args()

    # 実験用の使い捨てファイル（毎回まっさらに）
    rag_file = os.path.join(tempfile.gettempdir(), f"eve_rag_experiment_{args.backend}.jsonl")
    if os.path.exists(rag_file):
        os.remove(rag_file)

    print(f"=== RAG 実験  backend={args.backend}  k={args.k} ===")
    print(f"weights: REL={Config.RAG_W_REL} IMP={Config.RAG_W_IMP} REC={Config.RAG_W_REC}"
          f" | floor={Config.RAG_RELEVANCE_FLOOR} λ={Config.RAG_MMR_LAMBDA} τ={Config.RAG_RECENCY_TAU}s")
    embedder = make_embedder(args.backend)
    print("埋め込みモデル準備中（ruri 初回はDLあり）...")
    await embedder.warmup()
    store = RagStore(embedder, rag_file=rag_file)
    await store.initialize()
    await seed(store)
    print(f"seed 完了: {len(store)} 件\n")

    for q in QUERIES:
        dbg = await store.search_debug(q, k=args.k)
        info = dbg["info"]
        print(f"▶ クエリ: 「{q}」  (フロア除外 {info['floor_excluded']}/{info['total']}件)")
        if not dbg["selected"]:
            print("    （関連記憶なし）")
        for rank, s in enumerate(dbg["selected"], 1):
            tag = "★top1" if rank == 1 else f" #{rank}"
            print(f"  {tag}{_fmt(s)}")
        if args.pool:
            print("    --- 候補プール(フロア通過・base降順) ---")
            for s in info["pool"]:
                print(f"   {_fmt(s)}")
        print()

    await store.shutdown()
    print("完了。ruri vs openai を見比べ、floor/λ/weights を調整してください。")


if __name__ == "__main__":
    asyncio.run(main())
