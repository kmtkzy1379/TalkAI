"""F3.5 RAG 実験ハーネス（Tier-2・実埋め込み）。

仮データ（FeedbackLLM 出力を模した会話記憶 ~28件）を seed し、代表クエリを流して
**どういう抽出になったか**（クエリ→候補スコア→フロア除外→top-1/MMR選出）を表示する。
ユーザがこのログを見て精度を判断し、Ruri/API・パラメータを確定する。

  python tools\rag_experiment.py --backend ruri            # 単一設定の詳細ログ
  python tools\rag_experiment.py --backend ruri --sweep    # 重みプリセットの比較探索
  python tools\rag_experiment.py --backend ruri --pool     # 候補プールも表示

--sweep は chunk を1度だけ埋め込み、スコアリング属性だけ差し替えて各プリセットを比較する
（埋め込みは再計算しない＝速い）。会話の自然さに直結するので weights/baseline/floor を探索する。
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


# 仮データ: 会話記憶（text=表示&埋め込み対象, emotions, importance, 経過日数）。多様な話題・感情・重要度。
SEED = [
    # 夏クラスタ
    ("夏にスイカを食べて種飛ばし対決で盛り上がった", "楽しい・懐かしい", 0.6, 30),
    ("夏祭りで花火を見て大きな光と音に感動した", "感動", 0.6, 28),
    ("暑い日にいちご味のかき氷を食べて涼んだ", "満足・涼しい", 0.4, 5),
    ("夏に海へ泳ぎに行って日焼けした", "爽快・楽しい", 0.5, 20),
    ("夏にプールで一日中遊んで気持ちよかった", "爽快", 0.5, 12),
    ("縁側で風鈴の音を聞きながらのんびり過ごした夏の午後", "安らぎ", 0.4, 25),
    # 食べ物
    ("二郎系のこってりラーメンを食べて満足した", "満足・情熱", 0.4, 3),
    ("回転寿司でお腹いっぱい食べて幸せだった", "満足・幸福", 0.5, 9),
    ("辛いカレーを作って汗をかきながら食べた", "満足", 0.4, 14),
    ("寒い冬に鍋を囲んでみんなで温まった", "安心・温かい", 0.4, 70),
    ("朝のコーヒーが一日の楽しみで落ち着く", "落ち着く", 0.3, 1),
    ("甘いケーキを食べて幸せな気分になった", "幸福", 0.4, 11),
    # ネガティブ
    ("仕事の締切に追われて徹夜してストレスだった", "ストレス・疲労", 0.7, 2),
    ("試験勉強が大変で不安を感じていた", "不安", 0.5, 7),
    ("満員電車に押し込まれてぐったり疲れた", "疲労・不快", 0.5, 4),
    ("風邪をひいて寝込んで辛かった", "辛い・しんどい", 0.5, 40),
    # 趣味
    ("好きなバンドのライブで盛り上がって興奮した", "興奮・幸福", 0.7, 10),
    ("新作ゲームに夢中で徹夜してクリアした", "夢中・達成感", 0.5, 6),
    ("面白い小説を一気に読み終えて満足した", "満足・没頭", 0.4, 18),
    ("映画館で感動的な映画を見て泣いた", "感動", 0.6, 22),
    ("カメラを持って散歩しながら風景を撮った", "楽しい・穏やか", 0.4, 33),
    # 人・ポジティブ
    ("誕生日をみんなに祝ってもらって本当に嬉しかった", "幸福・感謝", 0.8, 45),
    ("飼っている猫がとても可愛くて癒される", "癒し・愛しい", 0.6, 4),
    ("久しぶりに友達と再会して語り合った", "嬉しい・懐かしい", 0.6, 15),
    # 季節（夏以外）
    ("京都へ旅行して紅葉を見て癒された秋の思い出", "癒し・感動", 0.5, 60),
    ("冬に雪だるまを作ってはしゃいだ", "楽しい", 0.4, 120),
    ("春に満開の桜を見てお花見をした", "穏やか・幸福", 0.5, 100),
    # 日常
    ("雨が続いて外に出られず退屈していた", "退屈", 0.3, 8),
]

QUERIES = [
    "夏って感じだね、暑いなあ",          # 夏クラスタの連想
    "お腹すいた、何か食べたいな",          # 食べ物クラスタ
    "最近疲れててストレス溜まってる",       # ネガティブ
    "音楽が聴きたい気分",                # 趣味-音楽
    "どこか旅行に行きたいなあ",            # 旅行
    "猫って癒されるよね",                # ペット
    "寒くなってきたね、もう冬かな",         # 冬（夏を引かない判別）
    "暇でつまらないなあ",                # 退屈（趣味/退屈）
]

# 重み探索プリセット: (名前, baseline, floor, w_rel, w_imp, w_rec)
PRESETS = [
    ("旧:補正なし",      0.00, 0.30, 0.50, 0.35, 0.15),  # before（importance支配・floor効かず）
    ("baseline補正",     0.75, 0.10, 0.60, 0.25, 0.15),  # 現 default
    ("imp最小",          0.75, 0.10, 0.72, 0.15, 0.13),
    ("floor高+rel純",    0.78, 0.15, 0.70, 0.20, 0.10),
    ("imp少し残す",      0.75, 0.10, 0.55, 0.32, 0.13),
]


def _apply(store: RagStore, p: tuple) -> None:
    _, store.rel_baseline, store.rel_floor, store.w_rel, store.w_imp, store.w_rec = p


def _short(text: str, n: int = 16) -> str:
    return text[:n] + ("…" if len(text) > n else "")


def _fmt_detail(s: dict) -> str:
    c = s["chunk"]
    return (
        f"    rel={s['relevance']:.3f}(cos={s['cos']:.3f}) imp={s['importance']:.2f} "
        f"rec={s['recency']:.3f} base={s['base']:.3f} | {c['text']}  〔{c.get('emotions')}〕"
    )


async def seed(store: RagStore) -> None:
    for text, emo, imp, ago in SEED:
        await store.add_chunk(text=text, emotions=emo, importance=imp, timestamp=_days_ago(ago))


async def run_detail(store: RagStore, k: int, pool: bool) -> None:
    for q in QUERIES:
        dbg = await store.search_debug(q, k=k)
        info = dbg["info"]
        print(f"▶ 「{q}」  (フロア除外 {info['floor_excluded']}/{info['total']})")
        if not dbg["selected"]:
            print("    （関連記憶なし）")
        for rank, s in enumerate(dbg["selected"], 1):
            tag = "★top1" if rank == 1 else f" #{rank}"
            print(f"  {tag}{_fmt_detail(s)}")
        if pool:
            print("    --- 候補プール(フロア通過・base降順) ---")
            for s in info["pool"]:
                print(f"   {_fmt_detail(s)}")
        print()


async def run_sweep(store: RagStore, k: int) -> None:
    print("各クエリで、重みプリセットごとに「注入される記憶」を比較（先頭=top1）:\n")
    for q in QUERIES:
        print(f"▶ 「{q}」")
        for p in PRESETS:
            _apply(store, p)
            dbg = await store.search_debug(q, k=k)
            picks = " / ".join(_short(s["chunk"]["text"]) for s in dbg["selected"]) or "（なし）"
            print(f"   [{p[0]:<12}] {picks}")
        print()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=Config.EMBED_BACKEND, choices=["ruri", "openai"])
    ap.add_argument("--k", type=int, default=Config.RAG_TOP_K)
    ap.add_argument("--sweep", action="store_true", help="重みプリセットの比較探索")
    ap.add_argument("--pool", action="store_true", help="候補プールも表示（詳細モード）")
    args = ap.parse_args()

    rag_file = os.path.join(tempfile.gettempdir(), f"eve_rag_experiment_{args.backend}.jsonl")
    if os.path.exists(rag_file):
        os.remove(rag_file)

    print(f"=== RAG 実験  backend={args.backend}  k={args.k}  mode={'sweep' if args.sweep else 'detail'} ===")
    embedder = make_embedder(args.backend)
    print("埋め込みモデル準備中（ruri 初回はDLあり）...")
    await embedder.warmup()
    store = RagStore(embedder, rag_file=rag_file)
    await store.initialize()
    await seed(store)
    print(f"seed 完了: {len(store)} 件")
    if not args.sweep:
        print(f"params: REL={store.w_rel} IMP={store.w_imp} REC={store.w_rec}"
              f" | baseline={store.rel_baseline} floor={store.rel_floor} λ={store.mmr_lambda}")
    print()

    if args.sweep:
        await run_sweep(store, args.k)
    else:
        await run_detail(store, args.k, args.pool)

    await store.shutdown()
    print("完了。")


if __name__ == "__main__":
    asyncio.run(main())
