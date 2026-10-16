"""YouTube Data API（API キーだけで使える範囲）— 配信中の動画を見つける・ライブチャットを読む。

## 割り当て（quota）
YouTube Data API の割り当ては既定で1日 10,000 単位。重さは呼び出しごとに違う（公式の表・2026-10-15 確認）:
videos / playlistItems / channels = 1、liveChatMessages.list = 5、search = 100。

**配信中の動画はチャンネルのアップロード一覧（uploads プレイリスト）の先頭に載る**（実測 2026-10-15: 配信開始
2〜24分の5件すべてが先頭に載っていた）。そこで search（100）は使わず、uploads → videos の2単位で探す。
search は uploads で見つからない時の予備（間隔を空けて数回だけ）。
限定公開・非公開の配信はどちらでも見つからない（API キーで見えるのは公開の物だけ）＝URL を指定してもらう。

## チャットの読み方
pageToken をつないで「前回から新しく届いた分だけ」を受け取る（同じコメントを読み直さない）。
**最初の1回は配信の過去のコメントが返る**（実測: 73件）。イブが配信を始めるより前に書かれた分は読まずに
「ここから」を作る。始めた後に書かれた分（配信を探している間に届いたスーパーチャット等）は渡す。
返ってくる `pollingIntervalMillis`（実測 約4.8秒）より短い間隔では読まない（rateLimitExceeded になる）。

`displayName` は「@ハンドル」の形（実測: @user-kvz8hg62hd / @なるとかまぼこ）。先頭の @ は外して渡す。
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

API = "https://www.googleapis.com/youtube/v3"

# 1回の呼び出しの重さ（割り当ての単位）。数えるのは観測用（ログに「この配信で約 N 単位」を出す）
COST = {"videos": 1, "playlistItems": 1, "channels": 1, "search": 100, "liveChat/messages": 5}

# コメントの本文の上限（YouTube のチャットは最大200字）。名前も長すぎるものは切る
TEXT_MAX = 200
NAME_MAX = 30

# fetch(url, params) -> (HTTP status, JSON)。テストは偽物を渡す（ネットに出ない）
FetchFn = Callable[[str, dict], Awaitable[tuple[int, dict]]]


class YouTubeError(Exception):
    """API がエラーを返した（reason は YouTube の errors[0].reason・通信の失敗は "network"）。"""

    def __init__(self, reason: str, status: int = 0, message: str = "") -> None:
        super().__init__(f"{reason} (HTTP {status}) {message}".strip())
        self.reason = reason
        self.status = status


# チャットが終わった・読めない（再試行しても無駄）
CHAT_GONE = frozenset({"liveChatEnded", "liveChatDisabled", "liveChatNotFound", "forbidden"})
# 割り当てを使い切った（その日はもう読めない）
QUOTA = frozenset({"quotaExceeded", "dailyLimitExceeded"})


@dataclass(frozen=True)
class LiveVideo:
    video_id: str
    title: str = ""
    channel_title: str = ""
    state: str = "none"          # live | upcoming | none（snippet.liveBroadcastContent）
    chat_id: str = ""            # activeLiveChatId（配信中・予定の時だけ）
    started_iso: str = ""        # actualStartTime
    viewers: Optional[int] = None  # concurrentViewers（非公開設定なら無い）


@dataclass(frozen=True)
class ChatMessage:
    """チャットの1件（イブに渡す形に整えたもの）。"""

    id: str
    kind: str                     # text / superchat / supersticker / member / milestone / gift_members / gift
                                  # 以下は会話に渡さない知らせ: deleted / banned / ended
    author: str = ""
    author_id: str = ""
    text: str = ""
    amount: str = ""              # 「¥500」等（お金を添えたものだけ）
    color: str = ""               # スーパーチャットのチャット欄での色（`superchat_color`・分からなければ ""）
    owner: bool = False           # チャンネルの持ち主
    moderator: bool = False
    member: bool = False          # メンバー（isChatSponsor）
    published_iso: str = ""
    target_id: str = ""           # deleted: 消されたコメントの id / banned: その人のチャンネル id
    detail: str = ""              # メンバーのランク名・継続月数・ギフトの人数など

    @property
    def is_conversation(self) -> bool:
        return self.kind not in ("deleted", "banned", "ended")

    @property
    def auto_name(self) -> bool:
        """YouTube が自動で付けたハンドル（「user-」＋英数字）か。読み上げにくい。"""
        return bool(_AUTO_NAME.fullmatch(self.author))


_AUTO_NAME = re.compile(r"user-[a-z0-9]{6,}")
_SPACES = re.compile(r"\s+")


def _clean(text: str, limit: int) -> str:
    t = _SPACES.sub(" ", (text or "").replace("​", "")).strip()
    return t[:limit]


def clean_name(display_name: str) -> str:
    """表示名の先頭の @ を外す（ハンドルは「@名前」で届く）。"""
    return _clean((display_name or "").lstrip("@＠"), NAME_MAX)


def _amount(details: dict) -> str:
    """金額の表示（hl=ja の amountDisplayString を優先。無ければ micros から作る）。"""
    s = (details.get("amountDisplayString") or "").strip()
    if s:
        return s
    try:
        micros = int(details.get("amountMicros") or 0)
    except (TypeError, ValueError):
        micros = 0
    cur = (details.get("currency") or "").strip()
    if micros <= 0:
        return cur
    v = micros / 1_000_000
    num = f"{int(v)}" if v == int(v) else f"{v:.2f}"
    return {"JPY": f"¥{num}", "USD": f"${num}"}.get(cur, f"{num} {cur}".strip())


# スーパーチャットの色（チャット欄でのそのコメントの色・安い順）。円は金額の段階から（YouTube のヘルプに表が無く、
# 第三者の一覧で確認 2026-10-15: 100/200/500/1,000/2,000/5,000/10,000円〜）、ほかの通貨は API の tier
# （「いちばん安い段階が tier 1」・公式の説明）から。スーパーステッカーには付けない（色の扱いを確かめていない）
SC_COLORS = ("青", "水色", "緑", "黄色", "オレンジ", "マゼンタ", "赤")
_JPY_STEPS = (100, 200, 500, 1000, 2000, 5000, 10000)


def superchat_color(details: dict) -> str:
    try:
        micros = int(details.get("amountMicros") or 0)
    except (TypeError, ValueError):
        micros = 0
    if str(details.get("currency") or "").upper() == "JPY" and micros > 0:
        yen = micros / 1_000_000
        idx = [i for i, step in enumerate(_JPY_STEPS) if yen >= step]
        return SC_COLORS[idx[-1]] if idx else ""
    try:
        tier = int(details.get("tier") or 0)
    except (TypeError, ValueError):
        tier = 0
    return SC_COLORS[min(tier, len(SC_COLORS)) - 1] if tier >= 1 else ""


def parse_message(item: dict) -> Optional[ChatMessage]:
    """API の liveChatMessage 1件 → ChatMessage（使わない種類は None・raise しない）。"""
    try:
        sn = item.get("snippet") or {}
        au = item.get("authorDetails") or {}
        typ = sn.get("type") or ""
        base = dict(
            id=str(item.get("id") or ""),
            author=clean_name(au.get("displayName") or ""),
            author_id=str(au.get("channelId") or sn.get("authorChannelId") or ""),
            owner=bool(au.get("isChatOwner")), moderator=bool(au.get("isChatModerator")),
            member=bool(au.get("isChatSponsor")), published_iso=str(sn.get("publishedAt") or ""),
        )
        if typ == "textMessageEvent":
            text = (sn.get("textMessageDetails") or {}).get("messageText") or sn.get("displayMessage") or ""
            return ChatMessage(kind="text", text=_clean(text, TEXT_MAX), **base)
        if typ == "superChatEvent":
            d = sn.get("superChatDetails") or {}
            return ChatMessage(kind="superchat", text=_clean(d.get("userComment") or "", TEXT_MAX),
                               amount=_amount(d), color=superchat_color(d), **base)
        if typ == "superStickerEvent":
            d = sn.get("superStickerDetails") or {}
            alt = (d.get("superStickerMetadata") or {}).get("altText") or ""
            return ChatMessage(kind="supersticker", text=_clean(alt, TEXT_MAX), amount=_amount(d), **base)
        if typ == "newSponsorEvent":
            d = sn.get("newSponsorDetails") or {}
            detail = "ランクアップ" if d.get("isUpgrade") else _clean(d.get("memberLevelName") or "", 30)
            return ChatMessage(kind="member", detail=detail, **base)
        if typ == "memberMilestoneChatEvent":
            d = sn.get("memberMilestoneChatDetails") or {}
            months = d.get("memberMonth")
            return ChatMessage(kind="milestone", text=_clean(d.get("userComment") or "", TEXT_MAX),
                               detail=f"{months}か月" if months else "", **base)
        if typ == "membershipGiftingEvent":
            d = sn.get("membershipGiftingDetails") or {}
            n = d.get("giftMembershipsCount")
            return ChatMessage(kind="gift_members", detail=f"{n}人" if n else "", **base)
        if typ == "giftEvent":
            d = sn.get("giftDetails") or {}
            return ChatMessage(kind="gift", text=_clean(d.get("altText") or d.get("giftName") or "", 60), **base)
        if typ == "messageDeletedEvent":
            target = (sn.get("messageDeletedDetails") or {}).get("deletedMessageId") or ""
            return ChatMessage(kind="deleted", target_id=str(target), **base)
        if typ == "tombstone":
            # 消されたメッセージは tombstone として同じ id で届く（公式の resource の説明）
            return ChatMessage(kind="deleted", target_id=base["id"], **base)
        if typ == "userBannedEvent":
            banned = ((sn.get("userBannedDetails") or {}).get("bannedUserDetails") or {}).get("channelId") or ""
            return ChatMessage(kind="banned", target_id=str(banned), **base)
        if typ == "chatEndedEvent":
            return ChatMessage(kind="ended", **base)
    except Exception:
        logger.exception("チャットのメッセージを読めなかった（飛ばす）")
    return None


def parse_video(item: dict) -> LiveVideo:
    sn = item.get("snippet") or {}
    ls = item.get("liveStreamingDetails") or {}
    try:
        viewers = int(ls["concurrentViewers"]) if ls.get("concurrentViewers") is not None else None
    except (TypeError, ValueError):
        viewers = None
    return LiveVideo(
        video_id=str(item.get("id") or ""), title=_clean(sn.get("title") or "", 100),
        channel_title=_clean(sn.get("channelTitle") or "", 60),
        state=str(sn.get("liveBroadcastContent") or "none"), chat_id=str(ls.get("activeLiveChatId") or ""),
        started_iso=str(ls.get("actualStartTime") or ""), viewers=viewers,
    )


_VIDEO_ID = re.compile(r"(?:v=|/live/|youtu\.be/|/video/|/shorts/|^)([A-Za-z0-9_-]{11})(?:[?&#/]|$)")


def video_id_from(text: str) -> str:
    """URL か動画 ID から動画 ID を取り出す（取り出せなければ ""）。"""
    t = (text or "").strip()
    m = _VIDEO_ID.search(t)
    return m.group(1) if m else ""


def uploads_playlist(channel_id: str) -> str:
    """チャンネル ID（UC…）→ アップロード一覧のプレイリスト ID（UU…）。形が違えば ""。"""
    c = (channel_id or "").strip()
    return "UU" + c[2:] if c.startswith("UC") and len(c) == 24 else ""


async def _aiohttp_fetch(session_holder: dict, url: str, params: dict) -> tuple[int, dict]:
    import aiohttp

    s = session_holder.get("s")
    if s is None or s.closed:
        s = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        session_holder["s"] = s
    async with s.get(url, params=params) as r:
        try:
            data = await r.json(content_type=None)
        except Exception:
            data = {}
        return r.status, data if isinstance(data, dict) else {}


class YouTubeClient:
    """API キーで読む部分だけ。**キーをログ・例外に出さない**。"""

    def __init__(self, api_key: str, *, fetch: Optional[FetchFn] = None, hl: str = "ja") -> None:
        self._key = api_key
        self._session: dict = {}
        self._fetch = fetch or (lambda url, params: _aiohttp_fetch(self._session, url, params))
        self.hl = hl
        self.units = 0  # この起動で使った割り当て（おおよそ・成功も失敗も数える＝YouTube も数える）

    async def get(self, path: str, **params) -> dict:
        self.units += COST.get(path, 1)
        q = {k: v for k, v in params.items() if v is not None}
        q["key"] = self._key
        try:
            status, data = await self._fetch(f"{API}/{path}", q)
        except Exception as e:  # 通信の失敗（キーを含む URL を例外文に載せない）
            raise YouTubeError("network", 0, type(e).__name__) from None
        if status != 200:
            err = (data.get("error") or {}) if isinstance(data, dict) else {}
            errs = err.get("errors") or [{}]
            reason = (errs[0] or {}).get("reason") or err.get("status") or "error"
            raise YouTubeError(str(reason), status, str(err.get("message") or "")[:120])
        return data

    async def videos(self, ids: list[str]) -> list[LiveVideo]:
        ids = [i for i in ids if i][:50]
        if not ids:
            return []
        d = await self.get("videos", part="snippet,liveStreamingDetails", id=",".join(ids))
        return [parse_video(it) for it in d.get("items") or []]

    async def video(self, video_id: str) -> Optional[LiveVideo]:
        vs = await self.videos([video_id])
        return vs[0] if vs else None

    async def channel_title(self, channel_id: str) -> str:
        if not channel_id:
            return ""
        d = await self.get("channels", part="snippet", id=channel_id)
        items = d.get("items") or []
        return _clean(((items[0].get("snippet") or {}).get("title") or ""), 60) if items else ""

    async def find_live(self, channel_id: str) -> tuple[Optional[LiveVideo], Optional[LiveVideo]]:
        """(配信中の動画, 予定のままの動画) を uploads の先頭から探す（2単位）。"""
        pl = uploads_playlist(channel_id)
        if not pl:
            return None, None
        d = await self.get("playlistItems", part="contentDetails", playlistId=pl, maxResults=5)
        ids = [((it.get("contentDetails") or {}).get("videoId") or "") for it in d.get("items") or []]
        return _pick_live(await self.videos(ids))

    async def search_live(self, channel_id: str) -> tuple[Optional[LiveVideo], Optional[LiveVideo]]:
        """予備: search で探す（100単位・uploads に載るのが遅れた時用）。"""
        d = await self.get("search", part="id", channelId=channel_id, eventType="live", type="video",
                           maxResults=3)
        ids = [((it.get("id") or {}).get("videoId") or "") for it in d.get("items") or []]
        return _pick_live(await self.videos(ids))

    async def close(self) -> None:
        s = self._session.get("s")
        if s is not None and not s.closed:
            await s.close()


def _pick_live(videos: list[LiveVideo]) -> tuple[Optional[LiveVideo], Optional[LiveVideo]]:
    live = [v for v in videos if v.state == "live" and v.chat_id]
    live.sort(key=lambda v: v.started_iso, reverse=True)  # 新しく始まった方
    upcoming = [v for v in videos if v.state == "upcoming"]
    return (live[0] if live else None), (upcoming[0] if upcoming else None)


def iso_time(s: str) -> Optional[datetime]:
    """ISO-8601（YouTube の publishedAt・`clock.now_iso`）→ datetime。時差の無いものは UTC とみなす。読めなければ None。"""
    if not s:
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


@dataclass
class ChatReader:
    """1つのライブチャットを pageToken でつないで読む（loop 上・同時に1回だけ呼ぶ）。"""

    client: YouTubeClient
    chat_id: str
    min_interval: float = 6.0
    since: str = ""               # 最初の1回で、これ（ISO）より前に書かれた分は読み捨てる（空なら最初の1回は全部）
    token: str = ""
    interval: float = 6.0         # 次に読むまで待つ秒（YouTube の指定と min_interval の大きい方）
    primed: bool = False          # 最初の1回（過去のコメント）を読み捨てたか
    ended: bool = False
    _seen: set = field(default_factory=set)
    _order: deque = field(default_factory=deque)

    async def poll(self) -> list[ChatMessage]:
        """新しく届いた分（古い→新しい順）。最初の1回は `since` より前に書かれた分を読み捨てる。"""
        d = await self.client.get("liveChat/messages", liveChatId=self.chat_id,
                                  part="snippet,authorDetails", maxResults=2000,
                                  pageToken=self.token or None, hl=self.client.hl)
        self.token = d.get("nextPageToken") or self.token
        try:
            wait = float(d.get("pollingIntervalMillis") or 0) / 1000.0
        except (TypeError, ValueError):
            wait = 0.0
        self.interval = max(self.min_interval, wait)
        if d.get("offlineAt"):
            self.ended = True
        msgs = [m for m in (parse_message(it) for it in d.get("items") or []) if m is not None]
        if any(m.kind == "ended" for m in msgs):
            self.ended = True
        if not self.primed:
            self.primed = True
            self._remember(m.id for m in msgs)
            t0 = iso_time(self.since)
            if t0 is None:
                return []
            return [m for m in msgs if (t := iso_time(m.published_iso)) is not None and t >= t0]
        fresh = [m for m in msgs if m.id and m.id not in self._seen]
        self._remember(m.id for m in fresh)
        return fresh

    def _remember(self, ids) -> None:
        for i in ids:
            if i and i not in self._seen:
                self._seen.add(i)
                self._order.append(i)
        while len(self._order) > 4000:  # 上限（配信が長くても膨らまない）
            self._seen.discard(self._order.popleft())
