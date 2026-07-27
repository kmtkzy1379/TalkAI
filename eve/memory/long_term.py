"""RagStore — 長期記憶（連想RAG）。

人間的な連想想起: 今の会話に関連する過去の記憶（＋その時の感情・話した内容）を 2〜3 件引っ張る。
ConversationCache と同型: `deque(maxlen)` ロケット鉛筆 + 非同期 write-queue で JSONL 永続化。
埋め込みは Embedder 注入で backend 非依存（ruri/openai を A/B 可能、テストは fake を注入）。
~500 件なら numpy 全件コサインが数ms → ベクタDB不要（DBなしのフラットファイル方針と一致）。

検索 = Generative Agents の memory-stream（v1 rag.py の実証式を踏襲）:
  base = W_rel*relevance(cosine) + W_imp*importance + W_rec*recency(exp減衰)
  ① 関連度フロア未満を除外（無関係混入で会話破綻させない）
  ② base 降順で候補プール → ③ hard-cut で重複除外
  ④ 最類似(relevance最大)を必ず #1（top-1保証）→ ⑤ 残りは MMR で多様化 → top_k
recency は弱め（短期記憶が直近を既にカバー＝高recencyは重複想起の元）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..clock import now_iso
from ..config import Config
from ..context_assembler import RagChunk
from .embed import Embedder, make_embedder

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# 話題の種として使えない「中身の無い要約」（会話の区切り/相談なし/反応なし 等）。
# FeedbackLLM は毎ターン要約を書くので、無言や締めの往復もチャンク化される。それ自体は記憶として
# 正しいが、**話題の種**にすると「もう話すことは無い」を毎回材料として渡すことになる。
_META_SUMMARY = re.compile(
    r"特に(の|)(相談|予定|話題|用件|すること)(は|も)(出て|)(い|)な"
    r"|新しい(発言|話題|情報)(は|も)(無|な)"
    r"|会話を(いったん|)(区切|終|締)"
    r"|(返答|反応|返事)(や[^、。]{0,8})?(は|が|も)(まだ|)(無|な|して)"
    r"|やり取りは(無|な)"
    r"|短く(了承|区切)"
)


def _is_meta_summary(text: Optional[str]) -> bool:
    return bool(_META_SUMMARY.search((text or "").splitlines()[0] if text else ""))


def _ts_epoch(ts_iso: str, fallback: float) -> float:
    try:
        return datetime.fromisoformat(ts_iso).timestamp()
    except (ValueError, TypeError):
        return fallback


class RagStore:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        rag_file: Optional[str] = None,
        max_chunks: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        # embedder 注入（None なら Config の backend で生成）。テストは fake を渡す。
        self._embedder = embedder if embedder is not None else make_embedder()
        path = rag_file or Config.RAG_FILE
        self.rag_path = Path(path)
        if not self.rag_path.is_absolute():
            self.rag_path = _REPO_ROOT / self.rag_path
        self.max_chunks = max_chunks if max_chunks is not None else Config.RAG_MAX_CHUNKS
        self.top_k = top_k if top_k is not None else Config.RAG_TOP_K
        # スコアリング・パラメータ（Config 既定だが属性として差替え可＝重み sweep / テスト用）。
        # rel_baseline: 埋め込みの異方性補正。cos をこの値基準で [0,1] に再スケール
        # （rel' = max(0,(cos-baseline)/(1-baseline))）。Ruri は ~0.75、生コサインなら 0.0。
        self.rel_baseline = Config.RAG_REL_BASELINE
        self.rel_floor = Config.RAG_RELEVANCE_FLOOR
        self.w_rel = Config.RAG_W_REL
        self.w_imp = Config.RAG_W_IMP
        self.w_rec = Config.RAG_W_REC
        self.recency_tau = Config.RAG_RECENCY_TAU
        self.mmr_lambda = Config.RAG_MMR_LAMBDA
        self.dup_hardcut = Config.RAG_DUP_HARDCUT
        self._chunks: "deque[dict]" = deque(maxlen=self.max_chunks)  # ロケット鉛筆
        self._write_queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
        self._write_task: Optional[asyncio.Task] = None
        # J-2 ②-5: 自律発話の関連枠に使った直前クエリ（同じなら関連枠を使わない＝同じ記憶の固着防止）
        self._last_auto_query: Optional[str] = None

    # --- ライフサイクル ---------------------------------------------------

    async def initialize(self) -> None:
        await self._load()
        self._write_task = asyncio.create_task(self._write_worker())

    async def _load(self) -> None:
        if not self.rag_path.exists():
            return

        def _read() -> list[dict]:
            out: list[dict] = []
            with open(self.rag_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return out

        try:
            records = await asyncio.to_thread(_read)
        except Exception:
            logger.exception("RAG ログ読み込み失敗（空で続行）: %s", self.rag_path)
            return

        skipped = 0
        for r in records:
            emb = r.get("embedding")
            if not emb or not r.get("text"):
                continue
            # backend 切替で次元が変わった永続データは混ぜない（別空間のため）
            if self._embedder.dim and len(emb) != self._embedder.dim:
                skipped += 1
                continue
            self._chunks.append(r)
        logger.info("RAG 記憶を復元: %d 件（次元不一致でスキップ %d）", len(self._chunks), skipped)

    async def shutdown(self) -> None:
        if self._write_task is not None:
            await self._write_queue.put(None)
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
            self._write_task = None

    async def warmup(self) -> None:
        await self._embedder.warmup()

    def latest_timestamp(self) -> Optional[str]:
        """保持中チャンクの最大 timestamp(iso)。None=チャンク無し。

        FeedbackLLM の **watermark 起動復元**用（前回セッションで feedback 済みの会話位置を
        永続 RAG の timestamp から復元し、それより新しい復元会話を起動時 catch-up する）。
        """
        best: Optional[str] = None
        for r in self._chunks:
            ts = r.get("timestamp")
            if isinstance(ts, str) and (best is None or ts > best):
                best = ts
        return best

    # --- 書き込み ---------------------------------------------------------

    async def add_chunk(
        self,
        *,
        text: str,
        summary: Optional[str] = None,
        emotions=None,
        next_prediction: Optional[str] = None,
        prediction_diff=None,
        reason: Optional[str] = None,
        importance: float = 0.5,
        topic_tags: Optional[list[str]] = None,
        search_text: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """1 記憶チャンクを追加（埋め込み→メモリ→永続化キュー）。

        **圧縮埋め込み / 展開注入**（ユーザ設計 2026-06-20）— 土台はこの schema で実装済:
        - 埋め込み（＝検索キー）は **要約 + タグ** だけに行う（`search_text` 省略時 `summary`＋`topic_tags`）。
          FeedbackLLM の感情/予測/誤差まで埋め込むとユーザ発話への関連度が薄まるため検索キーは話題に絞る。
        - 注入される表示文は `text`（選出時の「解凍」＝感情/予測等を含む完全版）。schema は全フィールドを保持。
        - feedback 実装時は `add_chunk(text=<完全版レンダリング>, summary=<要約>, topic_tags=<tags>,
          emotions=..., next_prediction=..., prediction_diff=..., reason=...)` を呼ぶだけ（新規機構は不要）。
        """
        text = (text or "").strip()
        if not text:
            return
        tags = topic_tags or []
        # 検索キー = 要約(or search_text) + タグ のみ。完全版(text)は埋め込まない（圧縮埋め込み）。
        target = (search_text or summary or text).strip()
        if tags:
            target = f"{target} | {' '.join(tags)}"
        embs = await self._embedder.embed_documents([target])
        if not embs or not embs[0]:
            logger.warning("RAG 埋め込み取得できず（このチャンクをスキップ）: %.20s", text)
            return
        record = {
            "type": "feedback_response",
            "text": text,
            "summary": summary,
            "emotions": emotions,
            "next_prediction": next_prediction,
            "prediction_diff": prediction_diff,
            "reason": reason,
            "importance": float(importance),
            "topic_tags": tags,
            "embedding": embs[0],
            "timestamp": timestamp or now_iso(),
        }
        self._chunks.append(record)
        try:
            self._write_queue.put_nowait(record)
        except Exception:
            logger.exception("RAG 書き込みキュー投入失敗（メモリ層は記録済み）")

    # --- 検索 -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _to_chunk(record: dict, as_topic_seed: bool) -> RagChunk:
        return RagChunk(
            text=record.get("text", ""),
            iso=record.get("timestamp", now_iso()),
            as_topic_seed=as_topic_seed,
            summary=record.get("summary", ""),  # 話題の種はこの1行だけを使う（decider）
        )

    async def _retrieve(
        self, query: str, k: int, *, exclude_after: Optional[float] = None
    ) -> tuple[list[dict], dict]:
        """memory-stream + フロア + top-1保証 + MMR の本体。

        選出された scored dict のリストと、デバッグ情報（フロア除外・候補プール）を返す。
        `search`（本番）と `search_debug`（実験ログ）の両方がここを使う＝ロジック一本化。

        exclude_after: この epoch 秒**以降**に書かれたチャンクを候補から外す（自律発話専用）。
        既定 None＝除外なし＝ユーザ発話への通常検索は一切影響を受けない（既定値で構造保証）。
        """
        info: dict = {"floor_excluded": 0, "recent_excluded": 0,
                      "total": len(self._chunks), "pool": []}
        if k <= 0 or not self._chunks:
            return [], info
        q = np.asarray(await self._embedder.embed_query(query), dtype=float)
        qn = q / (np.linalg.norm(q) or 1.0)

        chunks = list(self._chunks)
        embs = np.asarray([c["embedding"] for c in chunks], dtype=float)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        unit = embs / np.clip(norms, 1e-12, None)  # 行ごとに正規化
        rel = unit @ qn  # cosine（正規化済みなので内積）

        now = time.time()
        denom = max(1e-9, 1.0 - self.rel_baseline)
        scored: list[dict] = []
        for i, c in enumerate(chunks):
            ts = _ts_epoch(c.get("timestamp", ""), now)
            if exclude_after is not None and ts >= exclude_after:
                # 「今まさに見えている会話」から生まれた記憶を、記憶として二重に想起しない。
                # floor_excluded とは別カウント（無関係で落ちたのか新しすぎて落ちたのかを区別）。
                info["recent_excluded"] += 1
                continue
            cos = float(rel[i])
            # 異方性補正: 高ベースラインに圧縮された cos を [0,1] に広げて relevance を実効化。
            relevance = max(0.0, (cos - self.rel_baseline) / denom)
            if relevance < self.rel_floor:
                info["floor_excluded"] += 1  # ① 無関係を排除（破綻防止）
                continue
            recency = math.exp(-(now - ts) / self.recency_tau)
            importance = float(c.get("importance", 0.5))
            base = self.w_rel * relevance + self.w_imp * importance + self.w_rec * recency
            scored.append({
                "relevance": relevance, "cos": cos, "recency": recency,
                "importance": importance, "base": base, "unit": unit[i], "chunk": c,
            })
        if not scored:
            return [], info

        # ② base 降順で候補プール
        scored.sort(key=lambda s: -s["base"])
        pool = scored[: max(k * 4, 20)]
        info["pool"] = pool

        # ③ hard-cut: ほぼ同一(cosine>閾値)の重複を除去（query非依存の chunk 間類似）
        deduped: list[dict] = []
        for s in pool:
            if any(float(s["unit"] @ d["unit"]) > self.dup_hardcut for d in deduped):
                continue
            deduped.append(s)

        # ④ top-1 保証: 最も relevance の高いものを必ず先頭に
        top1 = max(deduped, key=lambda s: s["relevance"])
        selected = [top1]
        remaining = [s for s in deduped if s is not top1]

        # ⑤ MMR: λ*base - (1-λ)*max_sim_to_selected で多様化
        lam = self.mmr_lambda
        while len(selected) < k and remaining:
            best, best_mmr = None, -float("inf")
            for s in remaining:
                max_sim = max(float(s["unit"] @ t["unit"]) for t in selected)
                mmr = lam * s["base"] - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr, best = mmr, s
            selected.append(best)
            remaining.remove(best)

        return selected, info

    async def search(
        self, query: str, k: Optional[int] = None, *, exclude_after: Optional[float] = None
    ) -> list[RagChunk]:
        """memory-stream + フロア + top-1保証 + MMR で関連記憶を返す。

        exclude_after は自律発話専用（既定 None＝ユーザ発話への通常検索は不変）。
        """
        k = k if k is not None else self.top_k
        selected, _ = await self._retrieve(query, k, exclude_after=exclude_after)
        return [self._to_chunk(s["chunk"], as_topic_seed=False) for s in selected]

    async def search_debug(self, query: str, k: Optional[int] = None) -> dict:
        """実験用: 選出記憶＋各スコア（relevance/recency/importance/base）と
        フロア除外件数・候補プールを返す（どういう抽出になったかの可視化）。"""
        k = k if k is not None else self.top_k
        selected, info = await self._retrieve(query, k)
        return {"query": query, "selected": selected, "info": info}

    def _topic_pool(self) -> list[dict]:
        """話題の種に使える記憶だけを残す（メタ要約を除外）。

        「会話を区切った」「特に相談や予定は出ていない」等の**中身の無い要約**は、話題として
        振れないうえ「もう話すことは無い」という含意を毎回注入して沈黙側に効く。実測
        (2026-07-26): コーパス189件中8件(4%)のメタ要約が、関連枠の**26/78回(33%)**を占拠して
        いた。ユーザ発話への通常検索(`search`)からは外さない（何が起きたかの記憶としては正しい）。
        """
        pool = [c for c in self._chunks if not _is_meta_summary(c.get("summary") or c.get("text"))]
        return pool or list(self._chunks)  # 全部メタなら諦めて全件（種ゼロにはしない）

    def random(self, k: int = 2) -> list[RagChunk]:
        """無言時の「話題の種」用ランダム取得（埋め込み不要・同期）。"""
        chunks = self._topic_pool()
        if not chunks:
            return []
        picked = random.sample(chunks, min(k, len(chunks)))
        return [self._to_chunk(c, as_topic_seed=True) for c in picked]

    @staticmethod
    def _topic_importance(c: dict) -> float:
        """話題提起用の重要度。予測差(surprise)が大きい記憶ほど印象的＝重要とみなす（0-1）。
        importance フィールドは現状一様(0.5)なので prediction_diff を主信号に使う（user 検索は不変）。"""
        pd = c.get("prediction_diff")
        if isinstance(pd, (int, float)):
            return min(1.0, abs(float(pd)) / 100.0)
        return float(c.get("importance", 0.5))

    def topic_candidates(self, k: int = 2, *, jitter: Optional[float] = None) -> list[RagChunk]:
        """無言時の自律発話用「話題の種」。**重要度を優遇しつつ少しバラけさせて新しい切り口**を出す
        （埋め込み不要・同期）。ユーザ会話の `search`（関連度重視）とは別系統＝relevance を使わず、
        重要度(予測差) + ランダム jitter で並べる（random の「過去に逸れない/新しい話題」狙いを保ちつつ
        重要な記憶を少し優遇）。"""
        chunks = self._topic_pool()
        if not chunks:
            return []
        jit = jitter if jitter is not None else Config.RAG_TOPIC_JITTER
        scored = sorted(chunks, key=lambda c: self._topic_importance(c) + jit * random.random(), reverse=True)
        pool = scored[: max(k * 3, 6)]
        random.shuffle(pool)  # プール内で更にバラけさせる（同じ重要記憶ばかりにしない）
        return [self._to_chunk(c, as_topic_seed=True) for c in pool[:k]]

    async def autonomous_memories(
        self, query: Optional[str], k: int = 3, *, context_since_iso: Optional[str] = None
    ) -> list[RagChunk]:
        """自律発話用: **関連記憶(search)** ＋ **完全ランダム1件** ＋ **重要度(topic_candidates)** を混ぜる。
        - 関連: 「今の画面↔過去の話」を結ぶ（例: 甘いもの検索→前のチーズケーキの話）。
        - ランダム1件: 関連度/重要度だけだと**毎回同じ記憶が選ばれて単調**になるため、関連度に依らない
          1件を必ず混ぜて新しい切り口・意外性を出す（ユーザ案。沈黙時はコンテキストに余裕がある）。
        - 重要度: 残り枠を予測差の大きい印象的な記憶で埋める。

        **記憶の自己参照を断つ（J-2 ②-3）**: 関連枠のクエリは直近ユーザ発話なので、FeedbackLLM が
        その会話から数十秒前に書いたチャンクが relevance 最大で必ず先頭に来る（実測 2026-07-26:
        関連枠 78回中53回が5分以内のチャンク・中央値2.2分・最短1.8秒・同一チャンクが27回連続）。
        「今まさに会話として見えているもの」を「過去の記憶」として再提示しても新しい話題にならない
        ので候補から外す。除外は**この関数の中だけ**（`search` 既定は不変＝応答経路に影響しない）。
        context_since_iso: 今 recent_turns として注入されている会話の最古 iso（前回セッションの
        復元分を含む）。時刻窓(RAG_AUTO_EXCLUDE_SEC)だけでは、9日前に中断した会話がそのまま
        注入されている場合を捕まえられない（実測 round0）。

        **クエリ不変なら関連枠を使わない（J-2 ②-5）**: 放置中はクエリ（画面+直近ユーザ発話）が
        変化しないので、relevance top-1 は毎回**同じ1件**になる（実測: 80回の呼び出しで関連枠の
        unique は8件・最頻26回）。同じ記憶を関連として出し続けると判定LLMがその話題に固着するため、
        前回と同じクエリでは関連枠を空けてランダム/重要度に回す。ユーザが何か言えばクエリが変わり、
        関連想起はその場で復活する。
        """
        out: list[RagChunk] = []
        seen: set[str] = set()

        def _add(c: RagChunk) -> None:
            if c.text not in seen and len(out) < k:
                out.append(c)
                seen.add(c.text)

        q = (query or "").strip()
        repeated = bool(q) and q == self._last_auto_query
        self._last_auto_query = q or None
        if q and not repeated:
            cutoff = time.time() - Config.RAG_AUTO_EXCLUDE_SEC
            if context_since_iso:
                cutoff = min(cutoff, _ts_epoch(context_since_iso, cutoff))
            for c in await self.search(query, max(1, k // 3), exclude_after=cutoff):
                if not _is_meta_summary(c.summary or c.text):  # メタ要約は種にしない
                    _add(c)
        for c in self.random(2):  # 完全ランダム1件（関連度エコーチェンバーを破る）
            if len(out) >= k:
                break
            if c.text not in seen:
                _add(c)
                break  # ランダムは1件だけ
        for c in self.topic_candidates(k):  # 残りを重要度+jitter で埋める
            if len(out) >= k:
                break
            _add(c)
        return out

    # --- 背景書き込み -----------------------------------------------------

    async def _write_worker(self) -> None:
        while True:
            try:
                record = await self._write_queue.get()
                if record is None:
                    self._write_queue.task_done()
                    break
                try:
                    await asyncio.to_thread(self._append_line, record)
                finally:
                    self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("RAG 追記失敗（継続）")

    def _append_line(self, record: dict) -> None:
        with open(self.rag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
