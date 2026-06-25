"""F3.5 長期記憶（連想RAG）の決定論テスト（API/モデル不要）。

fake 埋め込み（キーワード軸ベクトル）を注入し、memory-stream ランキング / 関連度フロア /
top-1保証 / MMR多様化(近重複の排除) / 件数上限 / random / 永続化往復 / ロケット鉛筆 /
ResponseOrchestrator への注入（「過去の記憶」ブロック）を検証する。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f3_5_rag.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.memory import RagStore  # noqa: E402
from eve.memory.embed import Embedder  # noqa: E402
from eve.pipeline import AudioPlayQueue, Stimulus, StimulusKind  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402

_passed = 0
_failed = 0
_tmpdir = tempfile.mkdtemp(prefix="eve_f35_")
_counter = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


def _tmp() -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"rag{_counter}.jsonl")


class FakeEmbedder(Embedder):
    """キーワード軸の決定論ベクトル。各軸キーワードが含まれれば 1、なければ 0。"""

    AXES = ["夏", "スイカ", "ラーメン", "仕事", "旅行", "音楽"]

    def __init__(self) -> None:
        self.dim = len(self.AXES)

    def _vec(self, text: str) -> list[float]:
        v = [1.0 if ax in text else 0.0 for ax in self.AXES]
        if not any(v):
            v = [0.001] * len(self.AXES)  # zero-norm 回避
        return v

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _store(rag_file: str | None = None, **kw) -> RagStore:
    """fake 埋め込みは生コサイン運用＝異方性補正なし(rel_baseline=0)でテストする。"""
    s = RagStore(FakeEmbedder(), rag_file=rag_file or _tmp(), **kw)
    s.rel_baseline = 0.0
    return s


async def _seed(store: RagStore, items: list[tuple[str, str, float]]) -> None:
    """(display_text, search_text, importance) のリストを記憶に追加。"""
    for text, search, imp in items:
        await store.add_chunk(text=text, search_text=search, importance=imp)


async def t_rocket_pencil() -> bool:
    store = _store(max_chunks=5)
    for i in range(8):
        await store.add_chunk(text=f"記憶{i}", search_text="夏", importance=0.5)
    return len(store) == 5


async def t_floor_excludes_unrelated() -> bool:
    store = _store(top_k=5)
    await _seed(store, [
        ("スイカは夏に最高", "夏 スイカ", 0.5),
        ("仕事が忙しい", "仕事", 0.5),  # 夏スイカ クエリと無関係 → フロアで除外
    ])
    res = await store.search("夏 スイカ", k=5)
    texts = [c.text for c in res]
    return "仕事が忙しい" not in texts and "スイカは夏に最高" in texts


async def t_top1_most_relevant() -> bool:
    store = _store(top_k=3)
    await _seed(store, [
        ("夏スイカ完全一致", "夏 スイカ", 0.4),  # 最類似(cosine=1.0)だが importance 低
        ("夏の旅行", "夏 旅行", 0.9),            # importance 高いが relevance 低
        ("夏とラーメン", "夏 ラーメン", 0.9),
    ])
    res = await store.search("夏 スイカ", k=3)
    # top-1 は relevance 最大（importance に負けない）＝最類似が必ず先頭に入る
    return len(res) >= 1 and res[0].text == "夏スイカ完全一致"


async def t_dedup_diversity() -> bool:
    store = _store(top_k=2)
    await _seed(store, [
        ("スイカ最高", "夏 スイカ", 0.5),    # A
        ("スイカ大好き", "夏 スイカ", 0.5),  # A' = A と同一ベクトル → hard-cut で片方排除
        ("夏は旅行", "夏 旅行", 0.5),        # C = 多様（A から離れている）
    ])
    res = await store.search("夏 スイカ", k=2)
    texts = [c.text for c in res]
    dup_count = sum(1 for t in texts if t in ("スイカ最高", "スイカ大好き"))
    # 近重複は1件だけ・残り枠は多様な C が埋める（ユーザ懸念「同じ内容ばかり」を解消）
    return len(res) == 2 and dup_count == 1 and "夏は旅行" in texts


async def t_count_cap() -> bool:
    store = _store(top_k=2)
    await _seed(store, [
        ("夏1", "夏 スイカ", 0.5), ("夏2", "夏 旅行", 0.5),
        ("夏3", "夏 音楽", 0.5), ("夏4", "夏 ラーメン", 0.5),
    ])
    res = await store.search("夏", k=2)
    return len(res) <= 2


async def t_random() -> bool:
    store = _store()
    await _seed(store, [("a", "夏", 0.5), ("b", "スイカ", 0.5), ("c", "旅行", 0.5)])
    res = store.random(2)
    return len(res) == 2 and all(c.as_topic_seed for c in res)


def t_topic_importance() -> bool:
    # 予測差(surprise)が大きいほど話題重要度が高い・pd 無しは importance フォールバック
    hi = RagStore._topic_importance({"prediction_diff": 90})
    lo = RagStore._topic_importance({"prediction_diff": 5})
    none = RagStore._topic_importance({"importance": 0.5})
    return hi > lo and abs(hi - 0.9) < 0.01 and abs(none - 0.5) < 0.01


async def t_topic_candidates() -> bool:
    # 自律発話の話題の種: k 件・as_topic_seed=True・空ストアは []・relevance 不要(同期)
    store = _store()
    for i in range(8):
        await store.add_chunk(text=f"記憶{i}", search_text="夏", prediction_diff=(90 if i == 0 else 5))
    res = store.topic_candidates(2)
    empty = _store().topic_candidates(2)
    return len(res) == 2 and all(c.as_topic_seed for c in res) and empty == []


async def t_persistence_roundtrip() -> bool:
    path = _tmp()
    s1 = _store(rag_file=path)
    await s1.initialize()
    await _seed(s1, [("スイカの思い出", "夏 スイカ", 0.6)])
    await s1.shutdown()

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    s2 = _store(rag_file=path)
    await s2.initialize()
    res = await s2.search("夏 スイカ", k=3)
    await s2.shutdown()
    return (
        '"embedding"' in raw and '"timestamp"' in raw
        and len(s2) == 1
        and any(c.text == "スイカの思い出" for c in res)
    )


async def t_orch_injects_rag() -> bool:
    store = _store()
    await store.initialize()
    await _seed(store, [("スイカは夏に食べると最高だった", "夏 スイカ", 0.6)])
    seen: list[str] = []

    async def play_fn(a):
        pass

    audio = AudioPlayQueue(play_fn=play_fn)

    async def stream_fn(messages):
        seen.append("\n".join(m["content"] for m in messages))  # 文脈は system 等に分散→全結合で検査
        yield "うん。"

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn, rag_store=store)
    worker = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "夏 スイカ"))
    worker.cancel()
    await store.shutdown()
    return bool(seen) and "スイカは夏に食べると最高だった" in seen[0] and "過去の記憶" in seen[0]


async def main() -> None:
    check("ロケット鉛筆: max_chunks で上限", await t_rocket_pencil())
    check("関連度フロア: 無関係を除外", await t_floor_excludes_unrelated())
    check("top-1保証: 最類似が必ず先頭", await t_top1_most_relevant())
    check("MMR: 近重複を排除し多様な記憶を入れる", await t_dedup_diversity())
    check("件数上限: top_k 以下", await t_count_cap())
    check("random: k件・話題の種フラグ", await t_random())
    check("topic 重要度=予測差由来", t_topic_importance())
    check("topic_candidates: k件・話題の種・空安全", await t_topic_candidates())
    check("JSONL 永続化往復（embedding込み・復元）", await t_persistence_roundtrip())
    check("配線: 応答に「過去の記憶」が注入される", await t_orch_injects_rag())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
