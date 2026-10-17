"""X（旧 Twitter）公式 API の読み取りクライアント（読むだけ・投稿はしない）。

## 使う口と単価（2026-10 時点・従量課金のみ・https://docs.x.com/x-api/getting-started/pricing）
| 口 | 認証 | 単価 |
|---|---|---|
| 日本のトレンド `GET /2/trends/by/woeid/:id` | アプリの Bearer Token | 1回 $0.01（最大50件・投稿数つき） |
| 投稿の検索 `GET /2/tweets/search/recent` | Bearer Token | 返ってきた投稿1件 $0.005（1回10〜100件・直近7日まで） |
| 本人のホームタイムライン `GET /2/users/:id/timelines/reverse_chronological` | 本人のログイン（OAuth 2.0） | 1件 $0.005 |

**投稿者の情報（expansions=author_id）は取らない**: ユーザー情報は1件 $0.01 かかり、話題を知るのには要らない。
費用は返ってきた件数から数える（X は同じ日（UTC）に同じ投稿を読み直しても二重には課金しないが、ここでは
読んだ分を全部数える＝1日の上限の見張りは多めに見積もる側）。

失敗は `XApiError`（429 は `XRateLimited`）。呼び手（`collector.py`）がその話題だけ諦める。
HTTP は差し替えられる（`request_fn`・テストはネットに出ない）。既定は aiohttp。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

API_BASE = "https://api.x.com/2"
WOEID_JAPAN = 23424856

PRICE_POST = 0.005    # 投稿の読み取り 1件
PRICE_TRENDS = 0.01   # トレンド 1回
PRICE_USER = 0.01     # ユーザー情報 1件（ログインした本人の ID を最初に1回だけ引く）

# 検索に毎回足す条件。返信は会話の断片で何の話か読めず、リポストは同じ本文の重複になる
SEARCH_FILTER = ("lang:ja", "-is:retweet", "-is:reply")
_OR = re.compile(r"(?:^|\s)OR(?:\s|$)")

# (method, url, *, headers, params, data, timeout) -> (status, json の dict, レスポンスヘッダ)
RequestFn = Callable[..., Awaitable[tuple[int, dict, dict]]]


@dataclass
class Trend:
    name: str
    count: Optional[int] = None  # 投稿数（X が出さない時は None）


@dataclass
class Post:
    id: str
    text: str
    created_at: str = ""
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    quotes: int = 0

    @property
    def reactions(self) -> int:
        return self.likes + self.reposts + self.replies + self.quotes


class XApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"X API {status}: {message}")
        self.status = status


class XRateLimited(XApiError):
    def __init__(self, reset_epoch: Optional[float]) -> None:
        super().__init__(429, "回数の上限（しばらく待つ）")
        self.reset_epoch = reset_epoch


def build_query(words: str) -> str:
    """検索語に日本語・リポストと返信を除く条件を足す（純関数）。

    OR を含む語は括弧でくくる（X の検索は空白＝AND を OR より先に結ぶので、くくらないと条件が最後の語にしか掛からない）。
    """
    q = " ".join((words or "").split())
    if not q:
        return ""
    if _OR.search(q) and not (q.startswith("(") and q.endswith(")")):
        q = f"({q})"
    have = set(q.split())
    return " ".join([q] + [f for f in SEARCH_FILTER if f not in have])[:512]


def iso_z(epoch: float) -> str:
    """X の start_time 形式（秒まで・Z）。"""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _int(v) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def _float_or_none(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_posts(body: dict) -> list[Post]:
    """`data` の投稿一覧 → Post（型の崩れた行は捨てる・raise しない）。"""
    out: list[Post] = []
    for d in (body or {}).get("data") or []:
        if not isinstance(d, dict):
            continue
        text = str(d.get("text") or "").strip()
        if not text:
            continue
        m = d.get("public_metrics") if isinstance(d.get("public_metrics"), dict) else {}
        out.append(Post(
            id=str(d.get("id") or ""), text=text, created_at=str(d.get("created_at") or ""),
            likes=_int(m.get("like_count")), reposts=_int(m.get("retweet_count")),
            replies=_int(m.get("reply_count")), quotes=_int(m.get("quote_count")),
        ))
    return out


def parse_trends(body: dict) -> list[Trend]:
    out: list[Trend] = []
    for d in (body or {}).get("data") or []:
        if not isinstance(d, dict):
            continue
        name = str(d.get("trend_name") or "").strip()
        if name:
            c = d.get("tweet_count")
            out.append(Trend(name, _int(c) if c is not None else None))
    return out


def error_text(body: dict) -> str:
    for key in ("detail", "title", "error_description", "error", "_text"):
        v = (body or {}).get(key)
        if v:
            return str(v)[:200]
    errs = (body or {}).get("errors")
    if isinstance(errs, list) and errs and isinstance(errs[0], dict):
        return str(errs[0].get("message") or errs[0].get("detail") or errs[0])[:200]
    return "（理由不明）"


async def aiohttp_request(method: str, url: str, *, headers=None, params=None, data=None,
                          timeout: float = 15.0) -> tuple[int, dict, dict]:
    """既定の HTTP（aiohttp）。JSON でない返事は `_text` に入れて返す（raise はネットワークの失敗だけ）。
    レスポンスヘッダの名前は小文字にそろえて返す。"""
    import aiohttp  # 遅延 import（決定論テストはここを通らない）

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
        async with s.request(method, url, headers=headers, params=params, data=data) as r:
            try:
                body = await r.json(content_type=None)
            except Exception:
                body = {"_text": (await r.text())[:300]}
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return r.status, body if isinstance(body, dict) else {"_body": body}, hdrs


class XClient:
    """読むだけのクライアント。`spent_usd`・`requests`・`posts_read` は作ってからの合計（呼び手が回ごとの差を取る）。"""

    def __init__(self, bearer_token: str, *, user_auth=None, request_fn: Optional[RequestFn] = None,
                 timeout_sec: float = 15.0) -> None:
        self._bearer = bearer_token
        self._user = user_auth  # `auth.UserAuth`（本人のログイン）か None（タイムラインは使わない）
        self._request = request_fn or aiohttp_request
        self._timeout = timeout_sec
        self.spent_usd = 0.0
        self.requests = 0
        self.posts_read = 0

    @property
    def has_timeline(self) -> bool:
        return self._user is not None and self._user.ready()

    async def _get(self, path: str, params: dict, *, user: bool = False) -> dict:
        token = await self._user.access_token(self._request) if user else self._bearer
        status, body, headers = await self._request(
            "GET", f"{API_BASE}{path}", headers={"Authorization": f"Bearer {token}"},
            params=params, timeout=self._timeout)
        self.requests += 1
        if status == 429:
            raise XRateLimited(_float_or_none((headers or {}).get("x-rate-limit-reset")))
        if status >= 400:
            raise XApiError(status, error_text(body))
        return body or {}

    def _spend(self, usd: float, *, kind: str, items: int) -> None:
        self.spent_usd += usd
        try:
            from ..usage import meter

            m = meter()
            if m is not None:
                m.record_cost(role="x_api", model="x-api", cost_usd=usd, extra={"kind": kind, "items": items})
        except Exception:  # 計測のために本体を落とさない
            logger.debug("X API の使用量の記録に失敗（無視）", exc_info=True)

    async def trends(self, woeid: int = WOEID_JAPAN, max_trends: int = 30) -> list[Trend]:
        body = await self._get(f"/trends/by/woeid/{int(woeid)}", {
            "max_trends": max(1, min(50, int(max_trends))), "trend.fields": "trend_name,tweet_count"})
        self._spend(PRICE_TRENDS, kind="trends", items=1)
        return parse_trends(body)

    async def search_recent(self, words: str, *, max_results: int = 10,
                            start_time: Optional[str] = None) -> list[Post]:
        """関連度の高い順（`sort_order=relevancy`）。start_time は直近7日以内（X の決まり）。"""
        q = build_query(words)
        if not q:
            return []
        params = {"query": q, "max_results": max(10, min(100, int(max_results))),
                  "sort_order": "relevancy", "tweet.fields": "created_at,public_metrics"}
        if start_time:
            params["start_time"] = start_time
        posts = parse_posts(await self._get("/tweets/search/recent", params))
        self.posts_read += len(posts)
        self._spend(PRICE_POST * len(posts), kind="search", items=len(posts))
        return posts

    async def home_timeline(self, *, max_results: int = 20) -> list[Post]:
        """本人のホームタイムラインの新しい順（リポストと返信を除く）。ログインしていなければ空。"""
        if not self.has_timeline:
            return []
        params = {"max_results": max(1, min(100, int(max_results))), "exclude": "retweets,replies",
                  "tweet.fields": "created_at,public_metrics"}
        posts = parse_posts(await self._get(
            f"/users/{self._user.user_id}/timelines/reverse_chronological", params, user=True))
        self.posts_read += len(posts)
        self._spend(PRICE_POST * len(posts), kind="timeline", items=len(posts))
        return posts
