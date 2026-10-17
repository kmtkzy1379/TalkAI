"""X で話題を仕入れるサイドカー（起動中は6時間ごと・1回3件・起動時に前回からの分を1回取り戻す）。

## 1回の流れ（ユーザ設計 2026-10-17）
1. 材料を集める: 日本のトレンド（X）・ユーザのタイムライン（本人のログインがある時）・最近の会話の要約・前に「もっと知りたい」と
   思ったこと・イブの好みと性格・最近仕入れた話題・今の記憶に多い話題
2. 情報収集AI（`x_curate` 役）が候補を選ぶ（`curate.py`）
3. 候補ごとに X を検索して投稿を読み、中立に要約する（`search_summarize` 役・ツールなし＝外部の文を読むのはここだけ）
4. イブ（応答と同じ `response` 役＋性格＋好みの控え）が感想をつける（`impression.py`）
5. 「Xで見た」記憶として**別ファイル**に保存する（会話の記憶とは `RagStore.attach_side` で思い出す時だけ合わせる）

## 止まっている間（ユーザ決定 2026-10-17「まずPCだけ」）
起動中は6時間ごと。起動した時に前回から6時間以上たっていれば、1回だけ取り戻す（何回分も溜めて取らない）。
検索は直近7日まで遡れるので、止まっていた間に話題になったことは拾える。拾えないのはその時のトレンド順位だけ。

## 守り
- タイムラインの投稿は情報収集AI に「ユーザが何に興味があるか」の材料として渡すだけ。記憶にも記録（x_log）にも本文を残さない
  （友達や鍵アカウントの投稿が配信で出る道を作らない・ユーザ決定 2026-10-17）
- 1日の費用の上限（`X_DAILY_BUDGET_USD`・UTC の日ごと）。超えたらその日は取らない（費用は X API の分だけを数える）
- 同じ話題（形をそろえて同じもの）を `DEDUP_DAYS` 日以内にもう一度取らない
- 失敗（ネット・鍵・回数の上限）は記録して `retry_sec` 後にやり直す。会話の邪魔をしない（全部 await で、loop を塞がない）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

from ..config import Config
from ..memory.long_term import KIND_X
from .client import PRICE_POST, WOEID_JAPAN, Post, Trend, XApiError, XRateLimited, iso_z
from .curate import Pick, build_curate_messages, normalize_topic, parse_picks
from .digest import Digest, content_of, digest_posts
from .impression import Impression, eve_impression, importance_of, rate_interest, render_memory

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEDUP_DAYS = 3.0          # 同じ話題をこの日数のうちにもう一度取らない
STOP_STATUSES = (401, 402, 403)  # 鍵が使えない・残高が無い＝この回の続きを叩いても同じ理由で失敗する
# 話題名がこれ以上近ければ「同じ話題の言い換え」とみなす（記憶の検索と同じ Ruri の余弦）。1回の中（普段の3件）と、
# 最初の取り込みのようにまとめて増やす時の全体で、似た話題は取らずに次の候補にする（ユーザ 2026-10-17「同じような話題が重複すると
# 記憶がその話題に占領される」）。実測（最初の100件の話題名）: 0.92 で 17組（「CS初得点」「CS初得点の反応」…）、0.90 だと
# 「深海フェス2026」と「お笑いの日2026」のような別の話題まで同じ組に入った
SIMILAR_TOPIC = 0.92
HISTORY_MAX = 200         # 状態ファイルに残す「最近仕入れた話題」の件数
MIN_POSTS = 3             # 投稿がこれより少ない話題は使わない（何の話か読めない）
TREND_WINDOW_SEC = 86400  # トレンドの検索は直近1日だけ（7日前の同じ語の投稿を混ぜない）
RECENT_CONVO_SEC = 3 * 86400  # 「最近の会話」として渡す範囲


@dataclass
class XSettings:
    interval_sec: float = 6 * 3600
    per_run: int = 3
    candidates: int = 5          # 情報収集AI に出させる候補の数（上から使い、投稿が見つからなければ次）
    posts_per_topic: int = 10    # X の検索は1回10件から
    timeline_posts: int = 20     # 0 ならタイムラインを使わない
    trends_max: int = 30
    woeid: int = WOEID_JAPAN
    start_delay_sec: float = 120.0   # 起動直後の会話の邪魔をしない
    daily_budget_usd: float = 1.5    # 0 なら上限なし
    retry_sec: float = 1800.0        # 失敗した後にやり直すまで

    @classmethod
    def from_config(cls) -> "XSettings":
        return cls(
            interval_sec=max(0.0, Config.X_INTERVAL_HOURS) * 3600, per_run=max(1, Config.X_PER_RUN),
            candidates=Config.X_PER_RUN + 2, posts_per_topic=Config.X_POSTS_PER_TOPIC,
            timeline_posts=max(0, Config.X_TIMELINE_POSTS), trends_max=Config.X_TRENDS_MAX, woeid=Config.X_WOEID,
            start_delay_sec=Config.X_START_DELAY_SEC, daily_budget_usd=Config.X_DAILY_BUDGET_USD,
        )


@dataclass
class Materials:
    """1回ぶんの材料のうち X から取る物（最初の取り込みでは何回分かで使い回す＝トレンドとタイムラインを取り直さない）。"""

    trends: list[Trend] = field(default_factory=list)
    timeline: list[Post] = field(default_factory=list)


def _resolve(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _epoch(iso: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except (TypeError, ValueError):
        return None


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def _nearest(vec, pool) -> float:
    """pool の中でいちばん近い余弦（vec か pool が無ければ 0）。"""
    if vec is None or not pool:
        return 0.0
    na = sum(x * x for x in vec) ** 0.5
    best = 0.0
    for v in pool:
        nb = sum(y * y for y in v) ** 0.5
        if na and nb:
            best = max(best, sum(x * y for x, y in zip(vec, v)) / (na * nb))
    return best


class XCollector:
    """loop 単一所有のサイドカー（start/stop は FeedbackWorker・SelfModelWorker と同形）。"""

    def __init__(self, *, client, registry, store, memory=None, persona: str = "",
                 self_model_fn: Optional[Callable[[], str]] = None,
                 self_model_pick: Optional[Callable[[str], Awaitable[str]]] = None,
                 on_impression: Optional[Callable[[str], None]] = None, embedder=None,
                 settings: Optional[XSettings] = None,
                 state_path: Optional[str] = None, log_path: Optional[str] = None,
                 response_kwargs: Optional[dict] = None,
                 now_fn: Callable[[], float] = time.time, sleep_fn=asyncio.sleep) -> None:
        self._client = client
        self._registry = registry
        self._store = store            # X で見たこと（別ファイルの RagStore）
        self._memory = memory          # 会話の記憶（材料を読むだけ・書かない）
        self._persona = persona
        # イブの好みの控え。情報収集AI には全部（「イブの好み」の話題を選ぶ材料）、感想には話題に近い項目だけ
        # （`memory/self_model.SelfRecall`・応答と同じ）。全部を感想に渡すと、試した3件の感想が3件とも好きな物に触れた
        # （2026-10-17・関係の無いセキュリティのニュースにも）＝配信で直した、同じ好物ばかり話す件と同じ形
        self._self_model_fn = self_model_fn
        self._self_model_pick = self_model_pick
        # 記憶に入れた感想（イブ自身の言葉）を渡す先（好みの台帳の係・`SelfModelWorker.note_text`）
        self._on_impression = on_impression
        # 話題名の近さを測る埋め込み（記憶と同じ物）。None なら似た話題の除外はしない（文字が同じ話題の除外だけ）
        self._embedder = embedder
        self.s = settings or XSettings()
        self._state_path = _resolve(state_path)
        self._log_path = _resolve(log_path)
        self._response_kwargs = dict(response_kwargs or {})
        self._now = now_fn
        self._sleep = sleep_fn
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self.state: dict = self._load_state()

    # --- 状態（前回いつ取ったか・最近仕入れた話題・日ごとの費用）-----------------

    def _load_state(self) -> dict:
        st: dict = {}
        try:
            if self._state_path is not None and self._state_path.exists():
                st = json.loads(self._state_path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            logger.exception("🐦 X の状態ファイルを読めなかった（初めからとして続行）: %s", self._state_path)
            st = {}
        st.setdefault("last_run", None)
        st.setdefault("last_attempt", None)
        st.setdefault("history", [])
        st.setdefault("spend", {})
        return st

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        try:
            spend = self.state.get("spend") or {}
            self.state["spend"] = dict(sorted(spend.items())[-14:])  # 2週間ぶんだけ残す
            self.state["history"] = list(self.state.get("history") or [])[:HISTORY_MAX]
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError:
            logger.exception("🐦 X の状態ファイルを保存できなかった（続行）: %s", self._state_path)

    def spent_today(self, now: Optional[float] = None) -> float:
        return float((self.state.get("spend") or {}).get(_day(now if now is not None else self._now()), 0.0))

    def note_spend(self, usd: float) -> None:
        """X の費用を今日の分に足す（`run_once` の外で X を叩いた時＝最初の取り込みの材料集め）。"""
        if usd <= 0:
            return
        day = _day(self._now())
        spend = self.state.setdefault("spend", {})
        spend[day] = round(float(spend.get(day, 0.0)) + usd, 6)

    def over_budget(self, extra: float = 0.0) -> bool:
        b = self.s.daily_budget_usd
        return b > 0 and self.spent_today() + extra > b + 1e-9

    def seconds_until_due(self, now: Optional[float] = None) -> float:
        """次に取りに行くまでの秒（0 以下＝今）。前回の成功から interval、失敗した直後は retry_sec 待つ。"""
        if self.s.interval_sec <= 0:
            return float("inf")  # 定期の取得はしない（最初の取り込みの道具だけで使う）
        now = self._now() if now is None else now
        last = _epoch(self.state.get("last_run"))
        due = (last + self.s.interval_sec) if last is not None else now
        tried = _epoch(self.state.get("last_attempt"))
        if tried is not None and (last is None or tried > last):
            due = max(due, tried + self.s.retry_sec)  # 失敗した直後に何度も叩かない
        return due - now

    @property
    def has_timeline(self) -> bool:
        """本人のタイムラインを材料に使うか（ログイン済みで件数が 1 以上）。"""
        return self.s.timeline_posts > 0 and self._client.has_timeline

    @property
    def client(self):
        """X のクライアント（費用・回数の合計を読む用）。"""
        return self._client

    # --- ライフサイクル -----------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_forever(self) -> None:
        await self._sleep(self.s.start_delay_sec)
        while not self._stopping:
            wait = self.seconds_until_due()
            if wait > 0:
                # 長く眠らない（PC のスリープ明けでも壁時計で測り直す）
                await self._sleep(min(wait, 600.0))
                continue
            if self.over_budget():
                logger.info("🐦 X: 今日の上限 $%.2f に達しているので取らない（使った分 $%.3f）",
                            self.s.daily_budget_usd, self.spent_today())
                self.state["last_attempt"] = _iso(self._now())
                self._save_state()
                continue
            last = _epoch(self.state.get("last_run"))
            reason = ("初回" if last is None else
                      f"前回から{(self._now() - last) / 3600:.1f}時間")
            try:
                await self.run_once(reason=reason)
            except asyncio.CancelledError:
                raise
            except Exception:  # サイドカーの失敗で会話を止めない
                logger.exception("🐦 X の仕入れで例外（%d分後にやり直す）", int(self.s.retry_sec // 60))
                self.state["last_attempt"] = _iso(self._now())
                self._save_state()

    # --- 1回ぶん -----------------------------------------------------------------

    async def gather(self, *, timeline_posts: Optional[int] = None) -> Materials:
        """X から取る材料（トレンド・タイムライン）。片方が失敗しても、もう片方で続ける。

        鍵が使えない・残高が無い（401/402/403）時だけは XApiError をそのまま上げる（この先の検索も全部同じ理由で失敗する）。
        """
        m = Materials()
        try:
            m.trends = await self._client.trends(self.s.woeid, self.s.trends_max)
        except XApiError as e:
            if e.status in STOP_STATUSES:
                raise
            logger.warning("🐦 X のトレンドを取れなかった（%s）", e)
        n = self.s.timeline_posts if timeline_posts is None else timeline_posts
        if n > 0 and self._client.has_timeline:
            try:
                m.timeline = await self._client.home_timeline(max_results=n)
            except XApiError as e:
                logger.warning("🐦 X のタイムラインを取れなかった（%s）", e)
        return m

    def _curious(self, k: int = 8) -> list[tuple[str, str, str]]:
        rows = [r for r in self._store.records() if r.get("curious")]
        rows.sort(key=lambda r: str(r.get("timestamp", "")))
        return [(str(r.get("timestamp", "")), str(r.get("topic") or r.get("summary") or ""), str(r["curious"]))
                for r in rows[-k:]]

    async def topic_vec(self, topic: str) -> Optional[list[float]]:
        """話題名の埋め込み（測れなければ None＝似た話題の除外を飛ばす）。"""
        if self._embedder is None or not (topic or "").strip():
            return None
        try:
            return await self._embedder.embed_query(topic)
        except Exception:
            logger.exception("🐦 話題名の近さを測れなかった（この話題は似た話題の除外を飛ばす）: %s", topic)
            return None

    def _recent_topics(self, now: float) -> set[str]:
        cutoff = now - DEDUP_DAYS * 86400
        return {normalize_topic(h.get("topic", "")) for h in self.state.get("history") or []
                if (_epoch(h.get("at")) or 0) >= cutoff}

    async def curate(self, materials: Materials, *, count: int, candidates: int, bootstrap: bool) -> list[Pick]:
        now = self._now()
        now_s = _iso(now)
        recent, older, bias = [], [], []
        if self._memory is not None:
            since = _iso(now - RECENT_CONVO_SEC)
            recent = self._memory.recent_summaries(since, 12)
            older = self._memory.sample_summaries(4, before_iso=since)
            bias = self._memory.tag_counts(12)
        self_model = ""
        if self._self_model_fn is not None:
            try:
                self_model = self._self_model_fn() or ""
            except Exception:
                logger.exception("🐦 イブの好みの控えを読めなかった（無しで続行）")
        messages = build_curate_messages(
            trends=materials.trends, timeline=materials.timeline, recent=recent, older=older,
            curious=self._curious(), history=self.state.get("history") or [], bias=bias, self_model=self_model,
            persona=self._persona, count=count, candidates=candidates, bootstrap=bootstrap, now=now_s)
        try:
            resp = await self._registry.complete("x_curate", messages)
        except Exception:
            logger.exception("🐦 情報収集AI の呼び出しに失敗")
            return []
        picks = parse_picks(content_of(resp), bootstrap=bootstrap)
        if not picks:
            logger.warning("🐦 情報収集AI の候補を読めなかった: %.80s", content_of(resp))
        return picks

    async def run_once(self, *, reason: str = "", count: Optional[int] = None, candidates: Optional[int] = None,
                       bootstrap: bool = False, materials: Optional[Materials] = None,
                       max_usd: Optional[float] = None, daily: bool = True,
                       similar_pool: Optional[list] = None) -> dict:
        """1回仕入れる。返り値は記録（x_log に書く物と同じ）。

        max_usd はこの回の X の費用の上限。daily=False は1日の上限を見ず・数えもしない（最初の取り込みは
        自分の上限 max_usd で止める。数えると、その日の定期の仕入れが全部止まる）。
        similar_pool は「似た話題の除外」に使う、もう取った話題名の埋め込みの並び（入れた話題はここに足す）。
        None ならこの回の中だけで比べる（普段の3件）。まとめて増やす時は呼び手が同じ並びを毎回渡す。
        """
        pool = similar_pool if similar_pool is not None else []
        count = count or self.s.per_run
        candidates = max(count, candidates or self.s.candidates)
        start = self._now()
        spent0, requests0 = self._client.spent_usd, self._client.requests
        self.state["last_attempt"] = _iso(start)
        logger.info("🐦 X: 仕入れ開始（%s・%d件）", reason or "手動", count)
        if materials is None:
            try:
                materials = await self.gather()
            except XApiError as e:
                # 鍵・残高の問題は AI を呼ぶ前に打ち切る（情報収集AI の分も無駄になる）。retry_sec 後にやり直す
                logger.error("🐦 X API が使えない（%s）。X_BEARER_TOKEN と X の開発者コンソールの残高を確かめてください", e)
                self._save_state()
                record = {"at": _iso(start), "reason": reason, "error": str(e), "stored": 0,
                          "x_usd": 0.0, "x_requests": self._client.requests - requests0}
                self._log(record)
                return record
        picks = await self.curate(materials, count=count, candidates=candidates, bootstrap=bootstrap)
        seen = self._recent_topics(start)
        items: list[dict] = []
        stored = 0
        noted = 0.0  # 今日の分に足し済みの X の費用（話題ごとに足して保存する＝途中で止まっても数え漏れない）

        def settle() -> float:
            nonlocal noted
            spent = self._client.spent_usd - spent0
            if daily:
                self.note_spend(spent - noted)
                noted = spent
            return spent

        for p in picks:
            if stored >= count:
                break
            spent = self._client.spent_usd - spent0
            est = PRICE_POST * self.s.posts_per_topic
            if (daily and self.over_budget(spent - noted + est)) or (max_usd is not None and spent + est > max_usd):
                items.append({"topic": p.topic, "skip": "費用の上限"})
                logger.info("🐦 X: 費用の上限で打ち切り（この回 $%.3f）", spent)
                break
            key = normalize_topic(p.topic)
            if key in seen:
                items.append({"topic": p.topic, "category": p.category, "skip": f"{DEDUP_DAYS:g}日以内に取った話題"})
                continue
            # 似た話題（言い換え）は X を叩く前に外して次の候補へ（費用をかけない）
            vec = await self.topic_vec(p.topic)
            near = _nearest(vec, pool)
            if near >= SIMILAR_TOPIC:
                items.append({"topic": p.topic, "category": p.category,
                              "skip": f"取った話題と似ている（{near:.2f}）"})
                continue
            item = await self._one(p)
            items.append(item)
            settle()
            if item.get("stored"):
                stored += 1
                seen.add(key)
                if vec is not None:
                    pool.append(vec)
                self.state["last_run"] = _iso(self._now())
                self._save_state()  # 1件ごとに残す（止めた時に、入れた話題を「最近仕入れた話題」から失わない）
            elif item.get("stop"):
                break
        spent = settle()
        self._save_state()
        record = {
            "at": _iso(start), "reason": reason, "bootstrap": bootstrap,
            "trends": [t.name for t in materials.trends], "timeline_posts": len(materials.timeline),
            "picks": [vars(p) for p in picks], "items": items, "stored": stored,
            "x_usd": round(spent, 4), "x_requests": self._client.requests - requests0,
            "sec": round(self._now() - start, 1),
        }
        self._log(record)
        logger.info("🐦 X: %d件を記憶に入れた（候補 %d・X の費用 $%.3f・今日 $%.3f）",
                    stored, len(picks), spent, self.spent_today())
        return record

    async def _one(self, p: Pick) -> dict:
        """1つの話題: 検索 → 中立の要約 → イブの感想 → 記憶。記録用の dict を返す（stored=True で入った）。"""
        item: dict = {"category": p.category, "topic": p.topic, "query": p.query, "why": p.why}
        start_time = iso_z(self._now() - TREND_WINDOW_SEC) if p.is_trend else None
        try:
            posts = await self._client.search_recent(p.query, max_results=self.s.posts_per_topic,
                                                     start_time=start_time)
        except XRateLimited as e:
            logger.warning("🐦 X の回数の上限（この回はここまで）: %s", e)
            item.update(skip="回数の上限", stop=True)
            return item
        except XApiError as e:
            logger.warning("🐦 X の検索に失敗「%s」: %s", p.query, e)
            item["skip"] = f"検索の失敗 {e.status}"
            if e.status in STOP_STATUSES:
                item["stop"] = True  # 鍵が使えない・残高が無い＝次の話題も同じ
            return item
        item["posts"] = [{"text": x.text[:280], "reactions": x.reactions} for x in posts]
        if len(posts) < MIN_POSTS:
            item["skip"] = f"投稿が少ない（{len(posts)}件）"
            return item
        digest = await digest_posts(self._registry, topic=p.topic, query=p.query, posts=posts)
        if digest is None:
            item["skip"] = "要約できない・関連なし"
            return item
        item["digest"] = vars(digest)
        self_model = ""
        if self._self_model_pick is not None:
            try:
                self_model = await self._self_model_pick(p.topic) or ""
            except Exception:
                logger.exception("🐦 イブの好みの控えを選べなかった（無しで続行）")
        imp = await eve_impression(self._registry, digest=digest, topic=p.topic, query=p.query,
                                   category=p.category, why=p.why, posts=len(posts), persona=self._persona,
                                   self_model=self_model, response_kwargs=self._response_kwargs)
        if imp is None:
            item["skip"] = "感想の形が崩れた"
            return item
        # 興味の点は好みを見せずに別に付ける（好きな物の話題ばかりが話題の種の上に来ないため・ユーザ 2026-10-17）
        interest = await rate_interest(self._registry, digest=digest, persona=self._persona,
                                       response_kwargs=self._response_kwargs)
        imp.interest = interest if interest is not None else 50
        item["impression"] = vars(imp)
        if not await self._remember(p, digest, imp, len(posts)):
            item["skip"] = "記憶に書けなかった"
            return item
        item["stored"] = True
        if self._on_impression is not None:
            # 好みの台帳の係に、イブ自身の言葉（要約と率直な感想）を預ける（ユーザ 2026-10-17「情報収集中でも更新はしていい」）。
            # ネタ（冗談）は渡さない＝冗談で言った物を好みとして書かない（最初の取り込みで、冗談で言った曲が好きな音楽として台帳に入った・2026-10-17）
            try:
                self._on_impression(f"{p.topic}: {imp.summary} {imp.feeling_text}")
            except Exception:
                logger.exception("🐦 好みの係に感想を渡せなかった（続行）")
        self.state.setdefault("history", []).insert(0, {
            "at": _iso(self._now()), "category": p.category, "topic": p.topic, "query": p.query})
        logger.info("🐦 Xで見た［%s］%s → %s", p.category, p.topic, imp.summary)
        return item

    async def _remember(self, p: Pick, digest: Digest, imp: Impression, n_posts: int) -> bool:
        before = len(self._store)
        try:
            await self._store.add_chunk(
                text=render_memory(imp, digest),
                summary=imp.summary,
                emotions=imp.mood or None,
                importance=importance_of(imp.interest),
                topic_tags=digest.tags or [p.topic],
                # 検索の鍵（埋め込む文）は要約＋見た内容＋タグ。「Xで見た」を頭に付けるのは「最近Xで何見た？」で引けるように
                search_text=f"Xで見た {imp.summary} {digest.topic}",
                kind=KIND_X,
                extra={"category": p.category, "topic": p.topic, "query": p.query, "why": p.why,
                       "curious": imp.curious or None, "interest": imp.interest, "posts": n_posts},
            )
        except Exception:
            logger.exception("🐦 Xで見たことを記憶に書けなかった: %s", p.topic)
            return False
        return len(self._store) > before

    def _log(self, record: dict) -> None:
        """1回ぶんの記録（手元だけ・*.jsonl は .gitignore 済み）。タイムラインの本文は入れない（件数だけ）。"""
        if self._log_path is None:
            return
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("🐦 X の記録を書けなかった（続行）: %s", self._log_path)


def response_kwargs_from_config() -> dict:
    """応答と同じ reasoning_effort（`voice_loop` の応答と同じ渡し方）。イブの感想は応答と同じ役で作るため。"""
    effort = (Config.RESPONSE_REASONING_EFFORT or "").strip()
    return {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}


def build_collector(*, registry, store, memory=None, persona: str = "", self_model_fn=None,
                    self_model_pick=None, on_impression=None, embedder=None, request_fn=None,
                    state_path: Optional[str] = None, log_path: Optional[str] = None) -> Optional[XCollector]:
    """`.env` から作る（VoiceLoop・最初の取り込みの道具が共通で使う）。Bearer Token が無ければ None。

    self_model_fn は好みの控えの全部（情報収集AI 用）、self_model_pick は話題に近い項目だけ（感想用・`SelfRecall.block_for`）。
    on_impression は記憶に入れた感想を好みの係に渡す口（`SelfModelWorker.note_text`）。
    embedder は似た話題の除外で話題名の近さを測る物（記憶と同じ埋め込みを渡す）。
    state_path・log_path は試しの時だけ別の場所にする（既定は X_STATE_FILE・X_LOG_FILE）。
    """
    from .auth import UserAuth
    from .client import XClient

    if not Config.X_BEARER_TOKEN:
        return None
    user = None
    if Config.X_TIMELINE_POSTS > 0 and Config.X_CLIENT_ID:
        user = UserAuth(Config.X_TOKEN_FILE, client_id=Config.X_CLIENT_ID, client_secret=Config.X_CLIENT_SECRET)
        if not user.ready():
            logger.info("🐦 X のタイムラインはログインしていないので使わない（tools/x_login.py でログインできる）")
            user = None
    client = XClient(Config.X_BEARER_TOKEN, user_auth=user, request_fn=request_fn)
    return XCollector(client=client, registry=registry, store=store, memory=memory, persona=persona,
                      self_model_fn=self_model_fn, self_model_pick=self_model_pick, on_impression=on_impression,
                      embedder=embedder, settings=XSettings.from_config(),
                      state_path=state_path or Config.X_STATE_FILE, log_path=log_path or Config.X_LOG_FILE,
                      response_kwargs=response_kwargs_from_config())
