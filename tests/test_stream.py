"""Tier-1: YouTube 配信モード（`eve/stream/`・API もネットも OBS も使わない）。

守ること（ユーザ指定 2026-10-15）:
- コメントは**いちばん新しい1件**を、イブの手が空いた時にだけ渡す（割り込まない）。スーパーチャットは先に全部
- 配信の音: PC の音は OBS の設定のまま・OBS のマイクは切って始め、イブのマイクのボタンに合わせる・イブの声は1回だけ
- 配信中は 画面認識 OFF（ロック）・自律発話 ON・タスク ON（＋マイク OFF）。終わったら配信前に戻す
- YouTube 用の文脈（今配信している・コメントの形・誰が何回目か）を応答と発話判定に**事実として**渡す
- 配信前の私的な会話を配信の文脈に持ち込まない（会話の区切り・区切り前の内省を入れない）
- API キー・OBS のパスワードをログや例外に出さない
OBS とのやりとりは、手元に立てた偽の obs-websocket サーバーで本物のプロトコルを通す。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ControlPlane は Config（=.env）を読む。.env の中身で結果が変わらないよう、eve を import する前に固定する
os.environ["VLM_ENABLED"] = "1"
os.environ["TASK_ENABLED"] = "1"
os.environ["AUTOSPEECH_ENABLED"] = "0"
os.environ["CALLFUNCTION_ENABLED"] = "1"

from eve.stream import youtube as yt  # noqa: E402
from eve.stream.manager import (  # noqa: E402
    MSG_NO_KEY, StreamError, StreamManager, StreamSettings, ViewerRegistry, badge, format_comment)
from eve.stream.youtube import ChatMessage, ChatReader, LiveVideo, YouTubeClient, YouTubeError  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def _item(mid, typ, name="@山田", text="", ch="UCabc", **extra):
    sn = {"type": typ, "publishedAt": "2026-10-15T05:00:00.000000+00:00", "displayMessage": text}
    if typ == "textMessageEvent":
        sn["textMessageDetails"] = {"messageText": text}
    sn.update(extra)
    return {"id": mid, "snippet": sn,
            "authorDetails": {"channelId": ch, "displayName": name, "isChatOwner": False,
                              "isChatModerator": False, "isChatSponsor": False}}


# --------------------------------------------------------------------------
# [1] チャットのメッセージを読む
# --------------------------------------------------------------------------

def t_parse() -> None:
    print("\n[1] YouTube のメッセージ → イブに渡す形")
    m = yt.parse_message(_item("a", "textMessageEvent", text="こんにちは！\n初見です"))
    check("普通のコメント", m and m.kind == "text" and m.text == "こんにちは！ 初見です", f"{m}")
    check("**名前の先頭の @ を外す**（ハンドルは「@名前」で届く・実測）", m.author == "山田", m.author)
    m = yt.parse_message(_item("b", "superChatEvent", superChatDetails={
        "amountMicros": "500000000", "currency": "JPY", "amountDisplayString": "￥500", "userComment": "応援してます"}))
    check("スーパーチャット（金額は表示用の文字を使う）", m.kind == "superchat" and m.amount == "￥500"
          and m.text == "応援してます", f"{m}")
    m = yt.parse_message(_item("b2", "superChatEvent", superChatDetails={"amountMicros": "1500000", "currency": "USD"}))
    check("表示用の文字が無ければ micros から作る", m.amount == "$1.50" and m.text == "", f"{m.amount}")
    got = {yen: yt.superchat_color({"amountMicros": str(yen * 1_000_000), "currency": "JPY"})
           for yen in (99, 100, 199, 200, 499, 500, 999, 1000, 1999, 2000, 4999, 5000, 9999, 10000, 50000)}
    check("**スーパーチャットの色（円は金額の段階から）**",
          got == {99: "", 100: "青", 199: "青", 200: "水色", 499: "水色", 500: "緑", 999: "緑", 1000: "黄色",
                  1999: "黄色", 2000: "オレンジ", 4999: "オレンジ", 5000: "マゼンタ", 9999: "マゼンタ", 10000: "赤",
                  50000: "赤"}, f"{got}")
    check("ほかの通貨は API の tier（1＝いちばん安い）から・分からなければ空",
          yt.superchat_color({"amountMicros": "5000000", "currency": "USD", "tier": 3}) == "緑"
          and yt.superchat_color({"currency": "USD", "tier": 9}) == "赤"
          and yt.superchat_color({"amountMicros": "5000000", "currency": "USD"}) == "")
    m = yt.parse_message(_item("b3", "superChatEvent", superChatDetails={
        "amountMicros": "1000000000", "currency": "JPY", "amountDisplayString": "￥1,000", "userComment": "応援"}))
    check("スーパーチャットに色が付く", m.color == "黄色", m.color)
    m = yt.parse_message(_item("c", "superStickerEvent", superStickerDetails={
        "superStickerMetadata": {"altText": "ありがとうのうさぎ"}, "amountDisplayString": "￥200"}))
    check("スーパーステッカー", m.kind == "supersticker" and m.text == "ありがとうのうさぎ" and m.amount == "￥200")
    m = yt.parse_message(_item("d", "newSponsorEvent", newSponsorDetails={"memberLevelName": "見習い"}))
    check("メンバー加入", m.kind == "member" and m.detail == "見習い")
    m = yt.parse_message(_item("e", "memberMilestoneChatEvent", memberMilestoneChatDetails={
        "memberMonth": 3, "userComment": "3か月！"}))
    check("メンバー継続", m.kind == "milestone" and m.detail == "3か月" and m.text == "3か月！")
    m = yt.parse_message(_item("f", "membershipGiftingEvent", membershipGiftingDetails={"giftMembershipsCount": 5}))
    check("メンバーシップのギフト", m.kind == "gift_members" and m.detail == "5人")
    m = yt.parse_message(_item("g", "messageDeletedEvent", messageDeletedDetails={"deletedMessageId": "a"}))
    check("消されたコメントの知らせ", m.kind == "deleted" and m.target_id == "a" and not m.is_conversation)
    m = yt.parse_message(_item("a", "tombstone"))
    check("tombstone（消されたコメント・同じ id）", m.kind == "deleted" and m.target_id == "a")
    m = yt.parse_message(_item("h", "userBannedEvent",
                               userBannedDetails={"bannedUserDetails": {"channelId": "UCbad"}}))
    check("ブロックされた人の知らせ", m.kind == "banned" and m.target_id == "UCbad")
    check("チャットの終わり", yt.parse_message(_item("i", "chatEndedEvent")).kind == "ended")
    check("知らない種類は None（落ちない）", yt.parse_message(_item("j", "pollEvent")) is None)
    check("壊れた形でも落ちない", yt.parse_message({"snippet": None}) is None)
    m = yt.parse_message(_item("k", "textMessageEvent", name="@user-kvz8hg62hd", text="w"))
    check("自動で付いた名前を見分ける", m.auto_name, m.author)
    m = yt.parse_message(_item("l", "textMessageEvent", text="あ" * 500))
    check("本文は 200字で切る", len(m.text) == 200)


def t_ids() -> None:
    print("\n[2] URL から動画 ID / チャンネル → uploads")
    vid = "wiYUKHr9-kA"
    for u in (vid, f"https://www.youtube.com/watch?v={vid}", f"https://youtu.be/{vid}?si=x",
              f"https://www.youtube.com/live/{vid}?feature=share", f"https://studio.youtube.com/video/{vid}/livestreaming",
              f"https://www.youtube.com/live_chat?is_popout=1&v={vid}"):
        check(f"取り出せる: {u[:48]}", yt.video_id_from(u) == vid, yt.video_id_from(u))
    check("取り出せない物は空", yt.video_id_from("https://www.youtube.com/@channel") == "")
    check("チャンネル ID → uploads", yt.uploads_playlist("UCtestchannel0000000000A") == "UUtestchannel0000000000A")
    check("形が違えば空", yt.uploads_playlist("abc") == "")


class _FakeApi:
    """YouTube Data API の偽物（path ごとの返事を順に返す・呼ばれた引数を控える）。"""

    def __init__(self, routes: dict) -> None:
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, url: str, params: dict):
        path = url.split("/youtube/v3/", 1)[1]
        self.calls.append((path, dict(params)))
        q = self.routes.get(path) or [(200, {})]
        return q.pop(0) if len(q) > 1 else q[0]


def t_client() -> None:
    print("\n[3] YouTube の読み取り（偽の API）")
    key = "SECRET-KEY-123"
    live = {"id": "LIVE1", "snippet": {"title": "雑談", "liveBroadcastContent": "live", "channelTitle": "イブ"},
            "liveStreamingDetails": {"activeLiveChatId": "CHAT1", "actualStartTime": "2026-10-15T05:00:00Z",
                                     "concurrentViewers": "12"}}
    old = {"id": "OLD1", "snippet": {"title": "前の", "liveBroadcastContent": "none"}, "liveStreamingDetails": {}}
    api = _FakeApi({
        "playlistItems": [(200, {"items": [{"contentDetails": {"videoId": "LIVE1"}},
                                           {"contentDetails": {"videoId": "OLD1"}}]})],
        "videos": [(200, {"items": [live, old]})],
    })
    c = YouTubeClient(key, fetch=api)

    async def run():
        found, upcoming = await c.find_live("UCtestchannel0000000000A")
        check("uploads の先頭から配信中の動画を見つける", found and found.video_id == "LIVE1"
              and found.chat_id == "CHAT1" and found.viewers == 12, f"{found}")
        check("uploads → videos の2単位（search の100を使わない）", c.units == 2 and
              [p for p, _ in api.calls] == ["playlistItems", "videos"], f"{c.units} {api.calls}")
        check("キーは params で渡す", api.calls[0][1].get("key") == key)
        bad = YouTubeClient(key, fetch=_FakeApi({"videos": [(403, {"error": {
            "errors": [{"reason": "quotaExceeded"}], "message": "quota"}})]}))
        try:
            await bad.video("X")
            check("エラーは YouTubeError", False)
        except YouTubeError as e:
            check("エラーの理由を取り出す", e.reason == "quotaExceeded", e.reason)
            check("**例外文にキーが出ない**", key not in str(e), str(e))

        async def boom(url, params):
            raise ConnectionError(f"failed {url}?key={params.get('key')}")

        net = YouTubeClient(key, fetch=boom)
        try:
            await net.video("X")
        except YouTubeError as e:
            check("通信の失敗は network（**キー入りの URL を例外文に載せない**）",
                  e.reason == "network" and key not in str(e), str(e))

    asyncio.run(run())


def t_reader() -> None:
    print("\n[4] チャットを pageToken でつないで読む")
    page1 = {"items": [_item("h1", "textMessageEvent", text="過去1"), _item("h2", "textMessageEvent", text="過去2")],
             "nextPageToken": "T1", "pollingIntervalMillis": 4800}
    page2 = {"items": [_item("h2", "textMessageEvent", text="過去2"), _item("n1", "textMessageEvent", text="新1")],
             "nextPageToken": "T2", "pollingIntervalMillis": 9000}
    page3 = {"items": [_item("n2", "textMessageEvent", text="新2")], "nextPageToken": "T3",
             "pollingIntervalMillis": 1000, "offlineAt": "2026-10-15T06:00:00Z"}
    api = _FakeApi({"liveChat/messages": [(200, page1), (200, page2), (200, page3)]})
    c = YouTubeClient("k", fetch=api)
    r = ChatReader(c, "CHAT1", min_interval=6.0)

    async def run():
        first = await r.poll()
        check("**最初の1回（過去のコメント）は読まない**", first == [] and r.primed, f"{first}")
        check("間隔は YouTube の指定と最短の大きい方", r.interval == 6.0, f"{r.interval}")
        got = await r.poll()
        check("2回目は新しい分だけ（読み捨てた過去分は出さない）", [m.id for m in got] == ["n1"], f"{[m.id for m in got]}")
        check("pageToken をつなぐ", api.calls[1][1].get("pageToken") == "T1", f"{api.calls[1][1]}")
        check("YouTube の指定が長ければそちら", r.interval == 9.0)
        await r.poll()
        check("offlineAt が来たら終わり", r.ended)
        check("1回 5単位で数える", c.units == 15, f"{c.units}")
        hist = {"items": [_item("o1", "textMessageEvent", text="始める前", publishedAt="2026-10-15T04:59:58.5+00:00"),
                          _item("s1", "superChatEvent", publishedAt="2026-10-15T05:00:30.25Z",
                                superChatDetails={"amountDisplayString": "￥500", "userComment": "探している間"}),
                          _item("o2", "textMessageEvent", text="時刻なし", publishedAt="")],
                "nextPageToken": "H1"}
        r2 = ChatReader(YouTubeClient("k", fetch=_FakeApi({"liveChat/messages": [(200, hist)]})), "C2",
                        since="2026-10-15T05:00:00.000000+00:00")
        got = await r2.poll()
        check("**最初の1回でも、配信を始めた後に書かれた分は渡す**（探している間のスーパーチャットを落とさない）",
              [m.id for m in got] == ["s1"], f"{[m.id for m in got]}")

    asyncio.run(run())


# --------------------------------------------------------------------------
# [5] イブに渡す形
# --------------------------------------------------------------------------

def t_format() -> None:
    print("\n[5] コメントの1行（会話の記録にもこの形で残る）")
    m = ChatMessage(id="1", kind="text", author="山田", text="こんにちは")
    check("普通のコメント", format_comment(m) == "（チャット）山田: こんにちは", format_comment(m))
    m = ChatMessage(id="2", kind="superchat", author="佐藤", text="がんばって", amount="￥500")
    check("スーパーチャット", format_comment(m) == "（スーパーチャット ￥500）佐藤: がんばって", format_comment(m))
    m = ChatMessage(id="2b", kind="superchat", author="佐藤", text="がんばって", amount="￥1,000", color="黄色")
    check("**イブに渡す1行にはスパチャの色を付ける**", format_comment(m) == "（スーパーチャット ￥1,000・黄色）佐藤: がんばって",
          format_comment(m))
    check("配信の画面のカードの印には色を付けない", badge(m) == "スーパーチャット ￥1,000", badge(m))
    import eve.stream.manager as _mgr

    _mgr.SC_COLOR = False
    check("色を渡さない設定なら付けない（A/B の口）", format_comment(m) == "（スーパーチャット ￥1,000）佐藤: がんばって")
    _mgr.SC_COLOR = True
    m = ChatMessage(id="3", kind="member", author="鈴木", detail="見習い")
    check("メンバー加入（本文なし）", format_comment(m) == "（メンバー加入・見習い）鈴木", format_comment(m))
    m = ChatMessage(id="4", kind="text", author="kmt", text="テスト", owner=True)
    check("チャンネルの持ち主の印", "（チャンネルの持ち主）" in format_comment(m), format_comment(m))
    m = ChatMessage(id="5", kind="text", author="mod", text="x", moderator=True, member=True)
    check("モデレーター・メンバーの印", "（モデレーター・メンバー）" in format_comment(m), format_comment(m))
    m = ChatMessage(id="6", kind="text", author="user-kvz8hg62hd", text="w")
    check("自動で付いた名前の印", "（自動で付いた名前）" in format_comment(m), format_comment(m))
    check("普通のコメントに印は付けない", badge(ChatMessage(id="7", kind="text")) == "")


# --------------------------------------------------------------------------
# [6] 配信の本体（コメントの受け渡し）
# --------------------------------------------------------------------------

class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _manager(tmp: Path, *, free=True, submits=None, api=None, obs_factory=None, audio="pc", clock=None):
    state = {"free": free}
    submits = submits if submits is not None else []
    settings = StreamSettings(api_key="k", channel_id="UCtestchannel0000000000A", poll_sec=6.0, gap_sec=0.0,
                              audio=audio, viewers_file=str(tmp / "viewers.json"))
    clock = clock or _Clock()
    m = StreamManager(settings, is_free=lambda: state["free"], submit=lambda t, k: submits.append((t, k)),
                      youtube=YouTubeClient("k", fetch=api or _FakeApi({})), obs_factory=obs_factory,
                      now_fn=clock, wall_fn=lambda: "2026-10-15T05:00:00+00:00", runtime_dir=tmp / "rt")
    return m, state, submits, clock


def _msg(i, author="山田", text="", kind="text", aid=None):
    return ChatMessage(id=f"m{i}", kind=kind, author=author, author_id=aid or f"UC{author}", text=text or f"コメント{i}",
                       published_iso="2026-10-15T05:00:00+00:00")


def t_pick_latest() -> None:
    print("\n[6] いちばん新しい1件だけを渡す・消された/ブロックされた分は渡さない")
    tmp = Path(tempfile.mkdtemp())
    m, state, submits, clock = _manager(tmp)
    m.active = True
    m.started_iso = "2026-10-15T05:00:00+00:00"
    for i, who in enumerate(["山田", "佐藤", "鈴木"], 1):
        m.take(_msg(i, who))
    got = m.pick()
    check("**最新の1件**を選ぶ", got and got.id == "m3", f"{got}")
    check("それより前の件数を数える", m.skipped == 2, f"{m.skipped}")
    check("選んだら未読は空", m.unread == [])
    m.take(_msg(4, "山田"))
    m.take(ChatMessage(id="d", kind="deleted", target_id="m4"))
    check("**消されたコメントは渡さない**", m.pick() is None)
    m.take(_msg(5, "荒らし", aid="UCbad"))
    m.take(_msg(6, "佐藤"))
    m.take(ChatMessage(id="b", kind="banned", target_id="UCbad"))
    got = m.pick()
    check("**ブロックされた人のコメントは渡さない**", got and got.id == "m6" and m.skipped == 0, f"{got} {m.skipped}")
    m.feed(_msg(7, "山田", "やっほー"))
    check("渡す時はユーザ発話として（重複キーつき）",
          submits[-1] == ("（チャット）山田: やっほー", "yt:m7"), f"{submits[-1]}")
    check("同じ人の回数を数える（この配信で何回目か）", m.answering_info.get("count") == 2, f"{m.answering_info}")

    print("\n[6b] スーパーチャットは先に・届いた順に全部（ユーザ指定「スパチャは優先」）")
    m.take(_msg(10, "山田"))
    m.take(_msg(11, "佐藤", "応援してます", kind="superchat"))
    m.take(_msg(12, "鈴木"))
    m.take(_msg(13, "田中", "", kind="supersticker"))
    m.take(_msg(14, "高橋"))
    order = []
    while (got := m.pick()) is not None:
        order.append((got.id, m.skipped))
    check("**スーパーチャット・ステッカーを届いた順に先に読み、そのあと最新の1件**",
          [i for i, _ in order] == ["m11", "m13", "m14"], f"{order}")
    check("スーパーチャットを読む時は「読まなかった」を数えない・最後の1件で残りを数える",
          [k for _, k in order] == [0, 0, 2], f"{order}")
    for i in range(80):
        m.take(_msg(100 + i, f"人{i}"))
        if i == 3:
            m.take(_msg(99, "早い人", "最初の方のスパチャ", kind="superchat"))
    got = m.pick()
    check("**未読が溢れてもスーパーチャットは捨てない**", got and got.id == "m99", f"{got}")
    m.take(_msg(200, "荒らし", "x", kind="superchat", aid="UCbad2"))
    m.take(ChatMessage(id="b2", kind="banned", target_id="UCbad2"))
    m.take(_msg(201, "佐藤", "消すね", kind="superchat"))
    m.take(ChatMessage(id="d2", kind="deleted", target_id="m201"))
    check("消された・ブロックされた人のスーパーチャットは読まない", m.paid == [], f"{m.paid}")


def t_feed_loop() -> None:
    print("\n[7] 手が空いた時だけ読む・渡す（割り込まない）")
    tmp = Path(tempfile.mkdtemp())
    pages = [(200, {"items": [], "nextPageToken": "A", "pollingIntervalMillis": 100}),
             (200, {"items": [_item("x1", "textMessageEvent", text="一つ目"),
                              _item("x2", "textMessageEvent", name="@佐藤", text="二つ目", ch="UCsato")],
                    "nextPageToken": "B", "pollingIntervalMillis": 100})]
    api = _FakeApi({"liveChat/messages": pages})
    import time as _time

    m, state, submits, clock = _manager(tmp, free=False, api=api, clock=_time.monotonic)  # 実時間で回す

    async def run():
        m.active = True
        m.started_mono = clock()
        m._overlay_q = asyncio.Queue()
        m.reader = ChatReader(m._yt, "CHAT1", min_interval=0.0)
        await m.reader.poll()  # 過去分を読み捨てる
        m.chat_state = "live"
        m._next_poll = 0.0
        task = asyncio.create_task(m._feed_loop())
        await asyncio.sleep(0.6)
        check("**イブが話している間は読まない**（割り当てを使わない）", len(api.calls) == 1, f"{len(api.calls)}")
        check("話している間は渡さない", submits == [])
        state["free"] = True
        for _ in range(20):
            await asyncio.sleep(0.1)
            if submits:
                break
        check("手が空いたら読んで、最新の1件を渡す", submits and submits[0][0] == "（チャット）佐藤: 二つ目", f"{submits}")
        check("読まなかった件数", m.skipped == 1, f"{m.skipped}")
        state["free"] = False
        await asyncio.sleep(0.3)
        state["free"] = True
        await asyncio.sleep(0.9)
        check("返事が終わったら「返事中のコメント」を外す", m.answering is None)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def t_poll_errors() -> None:
    print("\n[8] 読めない時（上限・チャットの終わり）")
    tmp = Path(tempfile.mkdtemp())
    ended: list = []

    async def run():
        api = _FakeApi({"liveChat/messages": [(200, {"items": []}),
                                              (403, {"error": {"errors": [{"reason": "quotaExceeded"}]}})]})
        m, _s, _sub, _c = _manager(tmp, api=api)
        m.active = True
        m.reader = ChatReader(m._yt, "C", min_interval=0)
        await m.reader.poll()
        m.chat_state = "live"
        await m.poll_once()
        check("上限に達したら読むのをやめる（配信は続ける）", m.chat_state == "quota" and m.reader is None)
        api2 = _FakeApi({"liveChat/messages": [(200, {"items": []}),
                                               (403, {"error": {"errors": [{"reason": "liveChatEnded"}]}})]})
        m2, _s, _sub, _c = _manager(tmp, api=api2)
        m2.on_end = ended.append
        m2.active = True
        m2.reader = ChatReader(m2._yt, "C", min_interval=0)
        await m2.reader.poll()
        m2.chat_state = "live"
        await m2.poll_once()
        check("チャットが終わったら配信の終わりを知らせる", ended and m2.chat_state == "ended", f"{ended}")
        m2.on_end = ended.append
        m2._end("もう一度")
        check("終わりの知らせは1回だけ", len(ended) == 1, f"{ended}")

    asyncio.run(run())


def t_registry() -> None:
    print("\n[9] 視聴者を配信をまたいで覚える")
    tmp = Path(tempfile.mkdtemp())
    r = ViewerRegistry(tmp / "v.json")
    a = r.note("UC1", "山田", "S1", "2026-10-10T00:00:00+00:00")
    check("初めての人", a["streams_before"] == 0)
    r.note("UC1", "山田", "S1", "2026-10-10T00:10:00+00:00")
    r.save()
    r2 = ViewerRegistry(tmp / "v.json")
    b = r2.note("UC1", "山田", "S2", "2026-10-15T00:00:00+00:00")
    check("次の配信で「前にも来ていた」", b["streams_before"] == 1 and b["last_iso"] == "2026-10-10T00:10:00+00:00", f"{b}")
    c = r2.note("UC1", "山田", "S2", "2026-10-15T00:05:00+00:00")
    check("同じ配信の2回目でも前回の日時は変わらない", c["streams_before"] == 1
          and c["last_iso"] == "2026-10-10T00:10:00+00:00", f"{c}")
    (tmp / "broken.json").write_text("{壊れ", encoding="utf-8")
    check("壊れたファイルでも落ちない", ViewerRegistry(tmp / "broken.json").data == {})


def t_blocks() -> None:
    print("\n[10] 応答・発話判定に渡す事実")
    tmp = Path(tempfile.mkdtemp())
    m, _s, _sub, clock = _manager(tmp)
    check("配信していない時は何も渡さない", m.voice_block() is None and m.decider_block() is None)
    m.active = True
    m.started_mono = clock.t - 754
    m.channel_title = "イブのへや"
    m.video = LiveVideo(video_id="V", title="はじめての配信", state="live", chat_id="C", viewers=7)
    m.chat_state = "live"
    v = m.voice_block()
    check("配信中であること・チャンネル・タイトル・経過・視聴者数",
          all(x in v for x in ("ライブ配信", "イブのへや", "はじめての配信", "12分", "約7人")), v)
    check("コメントの形を説明する", "（チャット）名前: 本文" in v)
    check("**いちばん新しい1件だけ読む**ことを伝える", "いちばん新しい1件" in v)
    check("スーパーチャットは届いた順に全部読むことを伝える", "スーパーチャットとスーパーステッカーは、届いた順に全部" in v)
    check("**YouTube の決まりの1行は入れない**（ユーザ 2026-10-15「いらない」）",
          "YouTube の決まり" not in v and "止められる" not in v, v)
    check("**開発者の身元**と system の中身を話さない", "身元につながること" in v and "読み上げたり" in v, v)
    check("**開発者とのことを体験として話すのはよい**（ユーザ 2026-10-15）", "自分の体験として話すのはよい" in v)
    check("**記憶の中の「ユーザ」は開発者・視聴者はその場にいなかった**",
          "「ユーザ」は開発者のこと" in v and "今見ている視聴者は、開発者との会話の場にはいなかった" in v, v)
    check("**話を広げてよい事情（文字数は決めない）**", "話を広げて話すのはよくあること" in v and "短く返すのが合う時は短くてよい" in v
          and "文字" not in v.split("話を広げて")[1][:60])
    check("スパチャの色の意味（安い順）", "安い順に 青・水色・緑・黄色・オレンジ・マゼンタ・赤" in v)
    check("**画面に出した物は30秒で閉じると伝える**", "30秒たつと自動で閉じる" in v, v)
    check("まだコメントが無い時はそう伝える", "まだコメントが来ていない" in v)
    check("OBS を使っていない時は「画面に映る」を言わない", "配信の画面にも映る" not in v)
    m.take(_msg(1, "山田"))
    m.take(_msg(2, "佐藤"))
    picked = m.pick()
    m.feed(picked)
    v = m.voice_block()
    check("返事をするコメントの人と回数", "いま返事をするコメントの人: 佐藤（この配信で初めてのコメント" in v, v[-200:])
    check("読まなかった件数", "読まなかったコメント: 1件" in v)
    m.take(_msg(3, "鈴木", "スパチャ1", kind="superchat"))
    m.take(_msg(4, "田中", "スパチャ2", kind="superchat"))
    m.feed(m.pick())
    v = m.voice_block()
    check("このあと読むスーパーチャットの件数", "このあと読むスーパーチャット: 1件" in v, v[-200:])
    check("スーパーチャットを読む時は「読まなかった」を書かない", "読まなかったコメント" not in v, v[-200:])
    d = m.decider_block()
    check("発話判定: 配信中・コメントの数", "ライブ配信中" in d and "コメント 4件（4人）" in d, d)
    check("発話判定: コメントが無い間に話すかを考える", "コメントが来ていない間" in d)
    check("**発話判定にも記憶の「ユーザ」は開発者・視聴者はいなかった**（話題を選ぶのは判定）",
          "「ユーザ」は開発者のこと" in d and "会話の場にはいなかった" in d, d)
    check("発話判定にも開発者の身元の決まり（体験として話すのはよい）", "身元につながること" in d and "体験として話すのはよい" in d)
    check("**発話判定: 視聴者はいつでも書き込める・無音が流れる（事実）**",
          "視聴者の番を奪わない" in d and "動きの無い画面と無音" in d, d)
    check("発話判定に「何分に一回」のような決まりを入れない", "分に一回" not in d and "分おき" not in d)
    check("**発話判定: 配信では無音が長く感じられる（事実）**", "無音が長く感じられる" in d and "間が短く" not in d, d)
    from eve.speech.decider import build_decide_messages as _bdm

    pm = _bdm(surprise=20, silence_seconds=60, recent_turns=[], topic_seeds=[], prior_items=["[1分前・発話済] 話題"],
              stream=d)[1]["content"]
    check("配信中も「直前に話したばかりなら間を空ける」の決まりは残す（事実を足す方にした）", "間を空ける" in pm)
    m.chat_state = "waiting"
    check("チャットがまだ読めていない時", "まだ読めていない" in m.voice_block())
    for bad in ("例えば", "例:", "「こんにちは」"):
        check(f"例文・決まり文句を入れない（{bad}）", bad not in m.voice_block() + m.decider_block())
    from eve.stream.manager import viewer_note

    sent: list = []
    m._overlay_q = asyncio.Queue()
    m.obs = object()  # 字幕へ送る経路だけ通す
    m._set_chat("error", "どの配信のチャットを読むか分からない（TARGET_CHANNEL_ID か配信の URL が要る）")
    while not m._overlay_q.empty():
        sent.append(m._overlay_q.get_nowait())
    m.obs = None
    check("**配信の画面には設定の名前や原因を出さない**（視聴者が見る）",
          sent and all("TARGET" not in str(d.get("note")) and "API" not in str(d.get("note")) for d in sent)
          and all(viewer_note(s) for s in ("waiting", "ended", "error", "quota", "off")), f"{sent}")


def t_repeat() -> None:
    print("\n[10b] この配信でイブの発言に何度も出ている言葉（ユーザ 2026-10-17「異様にプリンに触れてる」）")
    import eve.stream.manager as mgr
    from eve.clock import Stamp
    from eve.context_assembler import Turn
    from eve.stream.manager import REPEAT_RECENT, repeat_fact, repeated_words

    w = repeated_words(["プリンを買います。プリン最高", "プリン城を建てます", "今日もプリンの話", "開発者が来た", "開発者が負けた"],
                       ["（チャット）山田: こんばんは"])
    check("1つの発言で何回出ても1回と数え、3回の発言に出た語だけ挙げる", w == [("プリン", 3)], f"{w}")
    check("視聴者・持ち主が書いた語は挙げない（イブが持ち出した語だけ）",
          repeated_words(["プリンだ", "プリンです", "プリンかも"], ["（チャット）山田: プリン好き？"]) == [])
    many = ["と".join(("アア", "イイ", "ウウ", "エエ", "オオ", "カカ", "キキ"))] * 3
    check("挙げるのは多くて5つ", len(repeated_words(many, [])) == 5, f"{repeated_words(many, [])}")
    f = repeat_fact([("プリン", 6), ("開発者", 4)], "あなた")
    check("事実の文: 見出し・語と回数・視聴者は言っていない・繰り返しに聞こえる",
          f.startswith("# あなたの最近の話\n") and "プリン（6回の発言に出た）・開発者（4回の発言に出た）" in f
          and "視聴者のコメントには出ていない言葉" in f and "繰り返しに聞こえる" in f, f)
    check("**どうするかは書かない**（言わない・避ける等の決まりを入れない）",
          all(x not in f for x in ("言わない", "避け", "控え", "しない", "例")), f)
    check("何度も出ている語が無ければ何も足さない", repeat_fact([], "あなた") == "")

    tmp = Path(tempfile.mkdtemp())
    m, _s, _sub, clock = _manager(tmp)
    m.active = True
    m.started_mono = clock.t - 600
    m.chat_state = "live"
    st = Stamp.now()
    turns = [Turn("user", "（チャット）山田: こんばんは", st)] + [Turn("eve", f"プリンの話その{i}。", st) for i in range(3)]
    check("会話を差していない時は足さない", "最近の話" not in m.voice_block() + m.decider_block())
    m.turns_fn = lambda: turns
    v, d = m.voice_block(), m.decider_block()
    check("**応答: 何度も出ている言葉を事実で渡す**（配信の事実の後・別の見出し）",
          "# あなたの最近の話" in v and "プリン（3回の発言に出た）" in v and v.index("# 配信") < v.index("# あなたの最近の話"), v[-300:])
    check("**発話判定: イブの発言として渡す**（判定はイブではない）",
          "# イブの最近の話" in d and "この配信のイブの発言" in d and "あなたの発言" not in d, d[-300:])
    m.feed(_msg(9, "佐藤", "プリンちょうだい"))
    check("**今返事をするコメントの語は挙げない**（返事の後で会話に記録されるので、まだ数えられていない）",
          "最近の話" not in m.voice_block(), m.voice_block()[-200:])
    m.answering = None
    old = [Turn("eve", "カレーの話。", st) for _ in range(3)] + [Turn("eve", f"雑談{i}。", st) for i in range(REPEAT_RECENT)]
    m.turns_fn = lambda: old
    check(f"数えるのはイブの直近{REPEAT_RECENT}件の発言だけ（長い配信でよく使う語ばかりにならない）",
          "カレー" not in (m.voice_block() or ""))
    m.turns_fn = lambda: turns
    mgr.REPEAT_FACT = False
    try:
        check("REPEAT_FACT=False なら足さない（測る口）", "最近の話" not in m.voice_block() + m.decider_block())
    finally:
        mgr.REPEAT_FACT = True

    def boom():
        raise RuntimeError("x")

    m.turns_fn = boom
    check("会話の取得に失敗しても配信の事実は出る（何度も出ている言葉だけ渡さない）",
          "# 配信" in m.voice_block() and "最近の話" not in m.voice_block())
    from eve.memory.conversation_cache import ConversationCache

    import time as _t

    cache = ConversationCache(history_file=str(tmp / "h.jsonl"))
    cache.add_turn("user", "配信の前の話")
    _t.sleep(0.002)
    cache.mark_boundary()
    _t.sleep(0.002)
    cache.add_turn("eve", "配信の話")
    check("会話の記録: 区切りより後だけを返す", [t.text for t in cache.since_boundary()] == ["配信の話"],
          f"{[t.text for t in cache.since_boundary()]}")


# --------------------------------------------------------------------------
# [11] 応答の組み立て・発話判定・会話の区切り
# --------------------------------------------------------------------------

def t_context() -> None:
    print("\n[11] 応答の system と自発発話の指示")
    from eve.context_assembler import ROLE_ANCHOR, ContextAssembler

    ca = ContextAssembler(system_prompt="スタイル")
    plain = ca.assemble(user_text="やあ")
    check("配信していない時は何も足さない", "# 配信" not in plain[0]["content"])
    msgs = ca.assemble(user_text="（チャット）山田: やあ", stream="# 配信（今 YouTube でライブ配信中）\n事実")
    sys_text = msgs[0]["content"]
    check("配信の事実は役割の説明の直後", sys_text.index(ROLE_ANCHOR) < sys_text.index("# 配信"))
    auto = ca.assemble(autonomous_content="メモ", recent_turns=[], stream="# 配信\nx")
    check("配信中の自発発話は「見ている人みんなに向けて」", "配信を見ている人みんなに向けて" in auto[-1]["content"])
    check("**配信中の自発発話は「一言だけ」を外す**", "一言だけ" not in auto[-1]["content"], auto[-1]["content"])
    ca2 = ContextAssembler(system_prompt="スタイル（短く）", stream_system_prompt="スタイル（配信）")
    check("**配信中は配信用のスタイル（「短く」なし）を使う**",
          ca2.assemble(user_text="x", stream="# 配信\nx")[0]["content"].startswith("スタイル（配信）")
          and ca2.assemble(user_text="x")[0]["content"].startswith("スタイル（短く）"))
    from eve.persona import persona_for_stream
    from eve.response.style import SPEECH_STYLE, SPEECH_STYLE_STREAM

    check("**配信中の性格の文は「短く」の指示だけを外す**",
          persona_for_stream("- 短く、元気に話す。\n- よく笑う。") == "- 元気に話す。\n- よく笑う。"
          and persona_for_stream("- よく笑う。") == "- よく笑う。")
    co = ca.assemble(user_text="x", stream="# 配信\nx", carry_over="# 配信の直前に開発者と話していたこと\n- y")
    check("持ち越しの要約は配信の事実の後に入る", co[0]["content"].index("# 配信") < co[0]["content"].index("# 配信の直前"))
    check("持ち越しが無ければ何も足さない", "直前に開発者と" not in ca.assemble(user_text="x")[0]["content"])
    import eve.context_assembler as ca_mod
    from eve.clock import now_iso
    from eve.context_assembler import MEMORY_INTERNAL_LINES, RagChunk, memory_for_stream
    from eve.feedback.feedback_llm import render_chunk_text
    from eve.feedback.parser import FeedbackResult

    note = render_chunk_text(FeedbackResult(summary="しろくまさんが今日は何をするか聞いた。", emotions="楽しい",
                                            user_emotion="不明", next_prediction="プリン城の話を続けそう",
                                            prediction_diff=70, reason="前回はプリン城を予測した"))
    check("外す行の名前が、記憶を書く側（render_chunk_text）と同じ（ずれると外れない）",
          [ln for ln in note.splitlines() if ln.startswith(MEMORY_INTERNAL_LINES)]
          == ["予測差: 70", "理由: 前回はプリン城を予測した"], note)
    check("**外すのは予測差と理由だけ**（要約・感情・次の予測は残す・ユーザ「次の行動指標まで渡して」）",
          memory_for_stream(note) == "しろくまさんが今日は何をするか聞いた。\n感情: イブ=楽しい / ユーザ=不明\n"
                                     "次の予測: プリン城の話を続けそう", memory_for_stream(note))
    rag = [RagChunk(text=note, iso=now_iso())]
    live = ca.assemble(user_text="x", rag_chunks=rag, stream="# 配信\nx")[0]["content"]
    talk = ca.assemble(user_text="x", rag_chunks=rag)[0]["content"]
    check("**配信中は記憶から予測差と理由の行を外して見せる**",
          "しろくまさんが今日は" in live and "感情: イブ=楽しい" in live and "次の予測: プリン城" in live
          and "予測差" not in live and "理由:" not in live, live[-300:])
    check("普段の会話は今までどおり全文（測ったのは配信だけ）", "次の予測: プリン城" in talk and "予測差: 70" in talk)
    ca_mod.STREAM_MEMORY_SLIM = False
    try:
        check("STREAM_MEMORY_SLIM=False なら配信中も全文（測る口）",
              "予測差: 70" in ca.assemble(user_text="x", rag_chunks=rag, stream="# 配信\nx")[0]["content"])
    finally:
        ca_mod.STREAM_MEMORY_SLIM = True

    check("配信用のスタイルは「短く」だけを外した物", "短く" in SPEECH_STYLE and "短く" not in SPEECH_STYLE_STREAM
          and SPEECH_STYLE_STREAM.replace("・話し言葉で自然に話す。", "・話し言葉で短く自然に答える。") == SPEECH_STYLE)
    auto0 = ca.assemble(autonomous_content="メモ", recent_turns=[])
    check("配信していない時の自発発話の指示は今までどおり（一言だけ）", "一言だけ話しかけてください" in auto0[-1]["content"]
          and "相手は今こちらに注意を向けていない" in auto0[-1]["content"]
          and "問いかける時は文末に「？」を付ける。\n話しかける中身" in auto0[-1]["content"], auto0[-1]["content"])


def t_decider() -> None:
    print("\n[12] 発話判定に配信の事実を渡す")
    from eve.speech.decider import SpeechDecision, build_decide_messages, should_speak

    msgs = build_decide_messages(surprise=20, silence_seconds=10, recent_turns=[], topic_seeds=[],
                                 stream="# 配信（今 YouTube でライブ配信中）\nチャット: x")
    check("判定の入力に入る", "# 配信（今 YouTube でライブ配信中）" in msgs[1]["content"])
    check("配信していない時は入らない",
          "# 配信" not in build_decide_messages(surprise=20, silence_seconds=10, recent_turns=[],
                                               topic_seeds=[])[1]["content"])

    async def old_style(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        return SpeechDecision(True, "r", "c")

    async def run():
        d = await should_speak(surprise=20, silence_seconds=5, recent_turns=[], topic_seeds=[],
                               decide_fn=old_style)
        check("配信していない時は stream を渡さない（受け取らない判定も壊れない）", d.speak)

    asyncio.run(run())


def t_boundary() -> None:
    print("\n[13] 配信の前後で会話を区切る（私的な会話を配信に持ち込まない）")
    from eve.feedback.prediction_state import PredictionState
    from eve.memory.conversation_cache import ConversationCache
    from eve.pipeline.audio_play_queue import AudioPlayQueue
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.orchestrator import ResponseOrchestrator

    tmp = Path(tempfile.mkdtemp())
    cache = ConversationCache(history_file=str(tmp / "h.jsonl"), recent_count=6)
    cache.add_turn("user", "実は明日引っ越すんだ")
    cache.add_turn("eve", "そうなんですね")
    check("区切りが無ければ今までどおり", len(cache.recent_for_injection()) == 2)
    pred = PredictionState()
    pred.last_feedback = "ユーザは明日引っ越す"
    pred.watermark = cache.recent()[-1].stamp.iso
    import time as _t

    _t.sleep(0.002)
    cache.mark_boundary()
    check("**区切りより前の会話は注入しない**", cache.recent_for_injection() == [], f"{cache.recent_for_injection()}")
    _t.sleep(0.002)
    cache.add_turn("user", "（チャット）山田: こんばんは")
    inj = cache.recent_for_injection()
    check("区切りの後の会話は入る", [t.text for t in inj] == ["（チャット）山田: こんばんは"], f"{inj}")
    check("区切りより前の内省（直近フィードバック）を区切り前と判定", cache.before_boundary(pred.watermark))

    async def stream_fn(messages, **k):
        if False:
            yield ""

    orch = ResponseOrchestrator(AudioPlayQueue(play_fn=None), stream_fn, lambda s: None,
                                conversation_cache=cache, prediction_state=pred)
    msgs = orch._build_messages(Stimulus(StimulusKind.USER_UTTERANCE, "（チャット）山田: こんばんは"))
    check("**区切り前の内省を応答に入れない**", "引っ越す" not in msgs[0]["content"], msgs[0]["content"][-300:])
    pred.watermark = inj[-1].stamp.iso
    pred.last_feedback = "視聴者の山田があいさつした"
    msgs = orch._build_messages(Stimulus(StimulusKind.USER_UTTERANCE, "（チャット）山田: こんばんは"))
    check("区切りの後の内省は入る", "視聴者の山田" in msgs[0]["content"])


def t_audio_text() -> None:
    print("\n[14] 字幕のために節の文を知らせる（再生は止めない）")
    from eve.pipeline.audio_play_queue import AudioPlayQueue

    played: list = []

    async def play(audio):
        played.append(audio)

    async def run():
        q = AudioPlayQueue(play_fn=play)
        seen: list = []
        q.add_text_listener(lambda t, s: seen.append((t, s)))
        q.add_text_listener(lambda t, s: 1 / 0)  # 壊れた受け手がいても再生は続く
        w = asyncio.create_task(q.play_worker())
        g = q.current_generation()
        q.enqueue(g, q.reserve_seq(g), b"a", text="こんにちは。")
        q.enqueue(g, q.reserve_seq(g), None, text="（番兵）")
        q.enqueue(g, q.reserve_seq(g), b"b", text="元気？")
        await asyncio.sleep(0.1)
        w.cancel()
        check("鳴り始めと終わりに文が届く",
              seen == [("こんにちは。", True), ("こんにちは。", False), ("元気？", True), ("元気？", False)], f"{seen}")
        check("受け手が例外を出しても再生は続く", played == [b"a", b"b"], f"{played}")

    asyncio.run(run())


def t_voiceloop_hooks() -> None:
    print("\n[15] VoiceLoop: 手が空いているか・コメントの入れ方（割り込まない）")
    from types import SimpleNamespace

    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.pipeline.stimulus_queue import StimulusQueue
    from eve.voice_loop import VoiceLoop

    q = StimulusQueue()
    ss = SimpleNamespace(user_speaking=False, stt_pending=False, user_activity_seq=0)
    runner = SimpleNamespace(is_paused=lambda: False, is_busy=lambda: False)
    audio = SimpleNamespace(is_quiet=lambda: True, is_playing=lambda: False)
    fake = SimpleNamespace(runner=runner, audio=audio, speech_state=ss, queue=q, _filler_active=lambda: False)
    check("静かなら空いている", VoiceLoop._free_for_comment(fake))
    runner.is_busy = lambda: True
    check("応答中は空いていない", not VoiceLoop._free_for_comment(fake))
    runner.is_busy = lambda: False
    audio.is_playing = lambda: True
    check("鳴っている間は空いていない", not VoiceLoop._free_for_comment(fake))
    audio.is_playing = lambda: False
    ss.user_speaking = True
    check("持ち主が話している間は空いていない", not VoiceLoop._free_for_comment(fake))
    ss.user_speaking = False
    runner.is_paused = lambda: True
    check("停止中は渡さない", not VoiceLoop._free_for_comment(fake))
    runner.is_paused = lambda: False

    async def run():
        await q.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "下書き"))
        check("自発発話の下書きだけが待っているなら空いている（コメントが勝つ）", VoiceLoop._free_for_comment(fake))
        await q.put(Stimulus(StimulusKind.CALLFUNCTION_RESULT, "報告"))
        check("機能の報告が待っていれば空いていない", not VoiceLoop._free_for_comment(fake))
        q2 = StimulusQueue()
        await q2.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "下書き"))
        marks = []
        ss2 = SimpleNamespace(mark_user_utterance=lambda: marks.append(1))
        barged = []
        fake2 = SimpleNamespace(speech_state=ss2, queue=q2, _ui_tasks=set(), _barge_in=lambda *a: barged.append(1))
        VoiceLoop.submit_stream_comment(fake2, "（チャット）山田: やあ", "yt:1")
        await asyncio.sleep(0)
        snap = q2.snapshot()
        check("コメントはユーザ発話として入る", [s.kind for s in snap] == [StimulusKind.USER_UTTERANCE]
              and snap[0].payload == "（チャット）山田: やあ", f"{snap}")
        check("待っていた自発発話の下書きは捨てる", all(s.kind != StimulusKind.AUTONOMOUS_SPEECH for s in snap))
        check("沈黙時計を戻す（判定中の自発発話を捨てさせる）", marks == [1])
        check("**割り込まない**（_barge_in を呼ばない）", barged == [])

        from eve.speech.monitor import SpeechState

        st = SpeechState()
        q3 = StimulusQueue()
        await q3.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "配信前に決めた話題"))
        st.evals_since_activity = 3
        seq = st.user_activity_seq
        fake3 = SimpleNamespace(speech_state=st, queue=q3)
        VoiceLoop._stream_live(fake3)
        check("**配信が始まったら、それまでに決めた自発発話を捨てる**（判定中は seq・待っている分はキュー）",
              st.user_activity_seq == seq + 1 and q3.snapshot() == [] and st.evals_since_activity == 0,
              f"{st.user_activity_seq} {q3.snapshot()}")
        await q3.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "視聴者に向けた話題"))
        bounds = []
        fake4 = SimpleNamespace(speech_state=st, queue=q3, cache=SimpleNamespace(mark_boundary=lambda: bounds.append(1)),
                               stream=None, _carry=None)
        VoiceLoop._stream_session_end(fake4)
        check("配信が終わったら区切りを入れ、配信中に決めた自発発話も捨てる",
              bounds == [1] and q3.snapshot() == [] and st.user_activity_seq == seq + 2)

    asyncio.run(run())


# --------------------------------------------------------------------------
# [16] ControlPlane の「配信」
# --------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.started = 0
        self.stopped = 0
        self.chat_state = "waiting"
        self.active = False
        self.obs = None
        self.mic_calls: list = []

    def available(self):
        return True, ""

    @property
    def mic_live(self):
        return bool(self.mic_calls and self.mic_calls[-1])

    async def start(self):
        self.started += 1
        self.autospeech_at_start = getattr(getattr(self, "vl", None), "speech_decider", None)
        self.autospeech_at_start = (self.autospeech_at_start.started if self.autospeech_at_start is not None
                                    else None)
        if self.fail is not None:
            raise self.fail
        self.active, self.obs = True, object()

    async def stop(self):
        self.stopped += 1
        self.active, self.obs = False, None

    async def set_mic(self, on):
        self.mic_calls.append(on)


def t_control() -> None:
    print("\n[16] 配信ボタン: 機能をまとめて切り替え、終わったら戻す")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_control_plane import _FakeVL, _settle  # 既存の偽の VoiceLoop を使う（同じ面を触る）

    from eve.control import ControlPlane

    async def scenario(fail=None):
        vl = _FakeVL()
        vl.stream = _FakeStream(fail)
        vl.stream.vl = vl
        seen: list = []
        cp = ControlPlane(vl, quiesce_sec=0.2, on_state=lambda k, s, on: seen.append((k, s, on)))
        cp.start()
        return vl, cp, seen

    async def run():
        vl, cp, seen = await scenario()
        check("最初は配信していない", cp.snapshot()["stream"] == ("off", False), f"{cp.snapshot()['stream']}")
        before = {k: cp.is_on(k) for k in ("vision", "mic", "autospeech", "task")}
        check("（前提）自律発話は起動時 OFF・画面認識 ON", before["autospeech"] is False and before["vision"] is True)
        cp.request("stream", True)
        await _settle(cp)
        check("配信が始まる", cp.is_on("stream") and vl.stream.started == 1)
        check("**画面認識 OFF**", not cp.is_on("vision") and vl.vision_stopped == 1)
        check("**自律発話 ON**", cp.is_on("autospeech") and vl.speech_decider.started >= 1)
        check("**自律発話は配信が始まってから入れる**（配信の事実なしに最初の話題を決めない）",
              vl.stream.autospeech_at_start == 0, f"{vl.stream.autospeech_at_start}")
        check("**タスク ON**", cp.is_on("task"))
        check("マイク OFF（入力は視聴者のチャット）", not cp.is_on("mic") and vl.input.running is False)
        check("画面認識はロック（押せない）", cp.snapshot()["vision"][0] == "disabled", f"{cp.snapshot()['vision']}")
        seen.clear()
        cp.request("vision", True)
        await _settle(cp)
        check("**配信中は画面認識を入れられない**", not cp.is_on("vision") and seen[-1] == ("vision", "disabled", False),
              f"{seen}")
        off = cp.features_off()
        check("今オフの機能は配信中の言い方で伝える", any("配信中なので" in t for t in off), f"{off}")
        check("配信の始めのマイク OFF では OBS に触らない（OBS のマイクは配信の準備で切る）", vl.stream.mic_calls == [])
        cp.request("mic", True)
        await _settle(cp)
        check("**配信中にマイクを入れると OBS のマイクも入る**（イブが聞く＋配信に乗る）",
              cp.is_on("mic") and vl.input.running and vl.stream.mic_calls == [True], f"{vl.stream.mic_calls}")
        cp.request("mic", False)
        await _settle(cp)
        check("**切ると OBS のマイクも切る**", not cp.is_on("mic") and vl.stream.mic_calls == [True, False],
              f"{vl.stream.mic_calls}")
        cp.request("stream", False)
        await _settle(cp, timeout=8.0)
        check("配信が終わる", not cp.is_on("stream") and vl.stream.stopped == 1)
        check("配信の後にマイクを戻しても OBS には触らない（OBS は配信前の状態に manager が戻す）",
              vl.stream.mic_calls == [True, False], f"{vl.stream.mic_calls}")
        after = {k: cp.is_on(k) for k in ("vision", "mic", "autospeech", "task")}
        check("**機能は配信前の状態に戻る**", after == before, f"{before} → {after}")
        check("画面認識のロックが外れる", cp.snapshot()["vision"][0] == "on")
        await cp.stop()

        vl, cp, seen = await scenario(fail=StreamError(MSG_NO_KEY))
        notes: list = []
        cp.on_note = notes.append
        cp.request("stream", True)
        await _settle(cp, timeout=8.0)
        check("始められなければ配信は OFF のまま", not cp.is_on("stream") and seen[-1][0] == "stream"
              and seen[-1][1] == "off", f"{seen[-3:]}")
        check("**失敗したら機能を配信前に戻す**", cp.is_on("vision") and cp.is_on("mic")
              and not cp.is_on("autospeech"), f"{[(k, cp.is_on(k)) for k in ('vision', 'mic', 'autospeech')]}")
        check("理由をそのまま伝える（OBS で何をすればよいか）", any("ストリームキー" in n for n in notes), f"{notes}")
        await cp.stop()

        vl, cp, seen = await scenario()
        vl.runner.paused = True
        cp._on["pause"] = False
        cp.request("stream", True)
        await _settle(cp)
        check("停止中に配信を始めたら再開する", cp.is_on("pause") and vl.runner.paused is False)
        await cp.stop()

        vl = _FakeVL()
        vl.stream = None
        cp = ControlPlane(vl, quiesce_sec=0.2)
        check("配信モードが無い構成では押せない", cp.snapshot()["stream"][0] == "disabled")

    asyncio.run(run())


# --------------------------------------------------------------------------
# [17] OBS（偽の obs-websocket サーバーで本物のプロトコルを通す）
# --------------------------------------------------------------------------

def t_obs_protocol() -> None:
    print("\n[17] OBS の WebSocket（認証・要求と返事・イベント・切断）")
    from aiohttp import WSMsgType, web

    from eve.stream.obs import ObsClient, ObsError, auth_string

    password, salt, challenge = "pass-xyz", "SALT", "CHAL"
    log: list = []

    async def handler(request):
        ws = web.WebSocketResponse(protocols=("obswebsocket.json",))
        await ws.prepare(request)
        await ws.send_str(json.dumps({"op": 0, "d": {"obsStudioVersion": "32.2.2", "obsWebSocketVersion": "5.7.4",
                                                     "rpcVersion": 1,
                                                     "authentication": {"challenge": challenge, "salt": salt}}}))
        ident = json.loads((await ws.receive()).data)
        log.append(("identify", ident))
        if ident["d"].get("authentication") != auth_string(password, salt, challenge):
            await ws.close(code=4009)
            return ws
        await ws.send_str(json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                break
            d = json.loads(msg.data)["d"]
            log.append(("req", d))
            rt = d["requestType"]
            if rt == "GetVersion":
                res = {"result": True, "code": 100}, {"obsVersion": "32.2.2"}
                await ws.send_str(json.dumps({"op": 5, "d": {"eventType": "StreamStateChanged", "eventIntent": 64,
                                                             "eventData": {"outputActive": True,
                                                                           "outputState": "OBS_WEBSOCKET_OUTPUT_STARTED"}}}))
            elif rt == "Bye":
                await ws.close()
                break
            else:
                res = {"result": False, "code": 600, "comment": "No source was found"}, None
            out = {"op": 7, "d": {"requestType": rt, "requestId": d["requestId"], "requestStatus": res[0]}}
            if res[1] is not None:
                out["d"]["responseData"] = res[1]
            await ws.send_str(json.dumps(out))
        return ws

    async def run():
        app = web.Application()
        app.router.add_get("/", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        events: list = []
        closed: list = []
        c = ObsClient(on_event=lambda k, d: events.append((k, d)), on_close=lambda: closed.append(1))
        hello = await c.connect(url, password)
        check("つないで認証できる", c.connected and hello.get("obsWebSocketVersion") == "5.7.4")
        check("配信の状態のイベントだけ受け取る（Outputs）", log[0][1]["d"]["eventSubscriptions"] == 64)
        check("**パスワードそのものは送らない**", password not in json.dumps(log[0][1]))
        v = await c.call("GetVersion")
        check("要求と返事を対応づける", v.get("obsVersion") == "32.2.2")
        await asyncio.sleep(0.05)
        check("イベントが届く", events and events[0][0] == "StreamStateChanged", f"{events}")
        try:
            await c.call("GetSourceScreenshot", {"sourceName": "x"})
            check("断られたら ObsError", False)
        except ObsError as e:
            check("断られたら ObsError（理由と code つき）", e.code == 600 and "No source" in str(e), f"{e}")
        await c.close()
        check("自分から閉じた時は切断の知らせを出さない", closed == [])
        c2 = ObsClient(on_close=lambda: closed.append(2))
        await c2.connect(url, password)
        try:
            await c2.call("Bye", timeout=2.0)
        except ObsError:
            pass
        await asyncio.sleep(0.1)
        check("OBS 側から切れたら知らせる（配信を止める判断に使う）", closed == [2] and not c2.connected, f"{closed}")
        await c2.close()
        bad = ObsClient()
        try:
            await bad.connect(url, "wrong", timeout=2.0)
            check("パスワードが違えば失敗", False)
        except ObsError as e:
            check("パスワードが違えば失敗（パスワードを例外に出さない）", "wrong" not in str(e), str(e))
        await runner.cleanup()

    asyncio.run(run())


def t_obs_misc() -> None:
    print("\n[18] OBS の設定ファイル・色・窓の指定")
    from eve.stream.obs import obs_color, read_ws_config, window_spec

    tmp = Path(tempfile.mkdtemp())
    p = tmp / "config.json"
    p.write_text(json.dumps({"server_enabled": False, "server_port": 4460, "server_password": "pw",
                             "auth_required": True}), encoding="utf-8")
    cfg = read_ws_config(p)
    check("OBS の WebSocket の設定を読む", cfg == {"enabled": False, "port": 4460, "password": "pw", "auth": True}, f"{cfg}")
    check("無ければ空", read_ws_config(tmp / "none.json") == {})
    check("色は 0xAABBGGRR", obs_color("#112233") == 0xFF332211, hex(obs_color("#112233")))
    check("窓の指定（: と # を OBS の書式で置き換える）",
          window_spec("a:b#c", "Qt6QWindowIcon", "python.exe") == "a#3Ab#22c:Qt6QWindowIcon:python.exe",
          window_spec("a:b#c", "Qt6QWindowIcon", "python.exe"))


def t_crop() -> None:
    print("\n[19] Live2D の切り出し（枠の縦横比・DPI）")
    from eve.stream.scene import live2d_crop

    geom = {"win": (1000, 800), "view": (10, 40, 600, 600)}
    l, t, r, b = live2d_crop(geom, 1500, 1200, 1600, 800)  # 映像は1.5倍（DPI 150%）・枠は 2:1
    w = 1500 - l - r
    h = 1200 - t - b
    check("切り出しは枠と同じ縦横比", abs(w / h - 2.0) < 0.01, f"{l, t, r, b} → {w}x{h}")
    check("縦長の表示は上下を削る（左右は表示の端）", l == 15 and r == 1500 - 15 - 900, f"{l, r}")
    from eve.stream.scene import LIVE2D_TOP_BIAS

    vt = t - 40 * 1.5                  # Live2D の表示の上端から削った分（窓の上の部分を除く）
    vb = b - (800 - 640) * 1.5         # Live2D の表示の下端から削った分（窓の下の部分を除く）
    check("縦に削る分を上と下に決めた割合で分ける（天井を減らし机を少し残す）",
          abs(vt / (vt + vb) - LIVE2D_TOP_BIAS) < 0.01, f"t={t} b={b} → 表示の上 {vt} / 下 {vb}")
    l, t, r, b = live2d_crop({"win": (1000, 800), "view": (0, 0, 1000, 200)}, 1000, 800, 800, 400)
    check("横長の表示は左右を中央で削る", t == 0 and abs(l - r) <= 1 and l > 0, f"{l, t, r, b}")
    check("大きさが分からなければ削らない", live2d_crop({"win": (0, 0), "view": (0, 0, 0, 0)}, 0, 0, 1, 1) == (0, 0, 0, 0))


class _FakeObs:
    """シーンを作る要求を記録する偽の OBS（返事は最小限）。"""

    def __init__(self, *, scenes=(), inputs=(), items=(), kinds=None) -> None:
        self.scenes = list(scenes)
        self.inputs = dict(inputs)
        self.items = dict(items)
        self.kinds = kinds if kinds is not None else ["color_source_v3", "browser_source", "window_capture",
                                                      "wasapi_process_output_capture"]
        self.calls: list = []
        self.next_id = 100
        self.css: dict = {}

    async def call(self, rt, data=None, timeout=10.0):
        self.calls.append((rt, data))
        if rt == "GetVideoSettings":
            return {"baseWidth": 1920, "baseHeight": 1080}
        if rt == "GetInputKindList":
            return {"inputKinds": self.kinds}
        if rt == "GetSceneList":
            return {"scenes": [{"sceneName": s} for s in self.scenes]}
        if rt == "GetInputList":
            return {"inputs": [{"inputName": n, "inputKind": k} for n, k in self.inputs.items()]}
        if rt == "GetSceneItemList":
            return {"sceneItems": [{"sourceName": n, "sceneItemId": i} for n, i in self.items.items()]}
        if rt in ("CreateInput", "CreateSceneItem"):
            self.next_id += 1
            return {"sceneItemId": self.next_id}
        if rt == "GetSceneItemTransform":
            return {"sceneItemTransform": {"sourceWidth": 1500, "sourceHeight": 1200}}
        if rt == "GetInputSettings":
            return {"inputSettings": {"css": self.css.get(data["inputName"], "")}}
        if rt == "SetInputSettings" and "css" in (data.get("inputSettings") or {}):
            self.css[data["inputName"]] = data["inputSettings"]["css"]
        return {}

    def names(self, rt):
        return [d.get("inputName") or d.get("sourceName") for r, d in self.calls if r == rt]


def t_scene() -> None:
    print("\n[20] OBS のシーンを用意する（無ければ作る・あれば配置を触らない）")
    from eve.stream.scene import (SRC_CHAT, SRC_LIVE2D, SRC_OVERLAY, SRC_TASK, SRC_VOICE, SceneBuilder, WindowInfo,
                                  chat_url)

    tmp = Path(tempfile.mkdtemp())
    main = WindowInfo("イブ — Live2D（リップシンク接続）", "Qt6QWindowIcon", "python.exe")
    task = WindowInfo("イブのタスクウィンドウ", "Qt6QWindowIcon", "python.exe")

    async def run():
        obs = _FakeObs()
        sb = SceneBuilder(obs, scene="イブ配信", overlay_path=tmp / "o" / "overlay.html")
        await sb.ensure(main=main, task=task)
        check("シーンを作る", ("CreateScene", {"sceneName": "イブ配信"}) in obs.calls)
        created = obs.names("CreateInput")
        check("ソースを全部作る（背景・Live2D・作業画面・チャット・字幕・イブの声）",
              set(created) >= {SRC_LIVE2D, SRC_TASK, SRC_CHAT, SRC_OVERLAY, SRC_VOICE}, f"{created}")
        win = next(d for r, d in obs.calls if r == "CreateInput" and d["inputName"] == SRC_LIVE2D)["inputSettings"]
        check("Live2D はタイトルが一致する窓だけを映す", win["priority"] == 1 and win["client_area"] is True
              and win["window"].startswith("イブ — Live2D"), f"{win}")
        chat = next(d for r, d in obs.calls if r == "CreateInput" and d["inputName"] == SRC_CHAT)["inputSettings"]
        check("チャットは見つかるまで空", chat["url"] == "about:blank" and "yt-live-chat" in chat["css"])
        check("配置を決める", len([1 for r, _ in obs.calls if r == "SetSceneItemTransform"]) >= 5)
        html = (tmp / "o" / "overlay.html").read_text(encoding="utf-8")
        check("字幕の HTML に配置が埋まる", '"live2d": [28, 28, 1308, 782]' in html and "/*LAYOUT*/null" not in html)
        check("字幕の HTML は視聴者の文を textContent で入れる（innerHTML を使わない）",
              "innerHTML" not in html and "textContent" in html)
        show = html[html.index("function showComment"):html.index("function hideComment")]
        check("**次のコメントが来たら、前の返事の「消す予約」を取り消す**（続けて読んだ2件目のカードが消えない）",
              "clearTimeout(doneTimer)" in show and "doneTimer = setTimeout" in html, show[:200])
        ok = await sb.update_crop({"win": (1000, 800), "view": (10, 40, 600, 600)})
        check("Live2D を切り出す", ok and any(r == "SetSceneItemTransform" and "cropTop" in d["sceneItemTransform"]
                                            for r, d in obs.calls[-2:]))

        # 2回目: シーンもソースもある → 配置は触らず、中身（窓・URL）だけ入れ直す
        have = {n: i for n, i in zip((SRC_LIVE2D, SRC_TASK, SRC_CHAT, SRC_OVERLAY, SRC_VOICE), range(1, 6))}
        have["イブ 背景"] = 9
        obs2 = _FakeObs(scenes=["イブ配信"], inputs={n: "x" for n in have}, items=have)
        sb2 = SceneBuilder(obs2, scene="イブ配信", overlay_path=tmp / "o" / "overlay.html")
        await sb2.ensure(main=main, task=task, video_id="VID12345678")
        check("2回目はシーンを作らない", not any(r == "CreateScene" for r, _ in obs2.calls))
        check("**2回目は配置を変えない**（OBS で動かした配置を壊さない）",
              not any(r == "SetSceneItemTransform" for r, _ in obs2.calls))
        chat_set = next(d for r, d in obs2.calls if r == "SetInputSettings" and d["inputName"] == SRC_CHAT)
        check("**2回目は CSS を上書きしない**（URL だけ）", chat_set["inputSettings"] == {"url": chat_url("VID12345678")},
              f"{chat_set}")
        check("2回目は字幕の HTML を OBS に読み直させる（同じファイルだと読み直さない）",
              ("PressInputPropertiesButton", {"inputName": SRC_OVERLAY, "propertyName": "refreshnocache"}) in obs2.calls)
        from eve.stream.scene import _sha, chat_css

        check("作った時に CSS の控えを残す", (tmp / "o" / "chat_css.sha256").read_text().strip() == _sha(chat_css()))
        old_css = "/* 前の版 */ #author-name { color: purple; }"
        (tmp / "o" / "chat_css.sha256").write_text(_sha(old_css))
        obs4 = _FakeObs(scenes=["イブ配信"], inputs={n: "x" for n in have}, items=have)
        obs4.css[SRC_CHAT] = old_css
        await SceneBuilder(obs4, scene="イブ配信", overlay_path=tmp / "o" / "overlay.html").ensure(main=main, task=task)
        check("**前回イブが入れたままの CSS は新しい版に入れ替える**", obs4.css[SRC_CHAT] == chat_css())
        obs5 = _FakeObs(scenes=["イブ配信"], inputs={n: "x" for n in have}, items=have)
        obs5.css[SRC_CHAT] = old_css + "\n/* ユーザが足した */ body { font-size: 30px; }"
        (tmp / "o" / "chat_css.sha256").write_text(_sha(old_css))
        await SceneBuilder(obs5, scene="イブ配信", overlay_path=tmp / "o" / "overlay.html").ensure(main=main, task=task)
        check("**ユーザが OBS で書き換えた CSS は触らない**", "ユーザが足した" in obs5.css[SRC_CHAT])
        (tmp / "o" / "chat_css.sha256").unlink()
        obs6 = _FakeObs(scenes=["イブ配信"], inputs={n: "x" for n in have}, items=have)
        obs6.css[SRC_CHAT] = old_css
        await SceneBuilder(obs6, scene="イブ配信", overlay_path=tmp / "o" / "overlay.html").ensure(main=main, task=task)
        check("控えが無ければ（誰が入れたか分からない）触らない", obs6.css[SRC_CHAT] == old_css)
        obs3 = _FakeObs(kinds=["color_source_v3", "browser_source", "window_capture"])
        sb3 = SceneBuilder(obs3, scene="イブ配信", overlay_path=tmp / "o3.html")
        await sb3.ensure(main=main, task=None)
        check("アプリの音を取り込めない OBS ではイブの声のソースを作らない（OBS の音声に任せる）",
              not sb3.voice_ok and SRC_VOICE not in obs3.names("CreateInput"))

    asyncio.run(run())


def t_manager_obs_errors() -> None:
    print("\n[21] OBS を始められない時の案内（何をすればよいかを言う）")
    from eve.stream import manager as mgr

    tmp = Path(tempfile.mkdtemp())

    class _Obs:
        def __init__(self, **k) -> None:
            self.connected = True

        async def connect(self, url, password, timeout=3.0, subscriptions=0):
            from eve.stream.obs import ObsError

            subs.append(subscriptions)
            raise ObsError("つながらない")

        async def close(self):
            pass

    orig = (mgr.read_ws_config, mgr.obs_running, mgr.find_obs_exe, mgr.launch_obs)
    launched: list = []
    subs: list = []
    try:
        mgr.read_ws_config = lambda: {"enabled": False, "port": 4455, "password": "pw", "auth": True}
        mgr.obs_running = lambda: False
        mgr.find_obs_exe = lambda explicit="": Path("obs64.exe")
        mgr.launch_obs = lambda exe: launched.append(exe)
        m, _s, _sub, _c = _manager(tmp, obs_factory=_Obs)
        caps: list = []
        m.ui = lambda ev, d: caps.append((ev, d))

        async def run():
            try:
                await m.start()
                check("WebSocket が無効なら始めない", False)
            except StreamError as e:
                check("WebSocket が無効なら、有効にする手順を言う", "WebSocket サーバーを有効にする" in str(e), str(e))
            check("設定できるよう OBS は開いておく", launched == [Path("obs64.exe")], f"{launched}")
            check("配信の状態と音声のミュートの知らせを受け取る（Outputs + Inputs）", subs and subs[0] == (1 << 6) | (1 << 3),
                  f"{subs}")
            check("失敗したらキャプチャ除外を戻してもらう", caps[-1] == ("capture", {"restore": True}), f"{caps}")
            check("失敗したら配信中にならない", not m.active and m.voice_block() is None)

        asyncio.run(run())
    finally:
        mgr.read_ws_config, mgr.obs_running, mgr.find_obs_exe, mgr.launch_obs = orig



class _AudioObs:
    """音声の要求に答える偽の OBS。ミュートを切り替えたら、本物と同じく InputMuteStateChanged を返す。"""

    def __init__(self, mgr, muted: dict, special: dict) -> None:
        self.mgr = mgr
        self.muted = dict(muted)
        self.special = special
        self.calls: list = []
        self.connected = True

    async def call(self, rt, data=None, timeout=10.0):
        self.calls.append((rt, data))
        if rt == "GetSpecialInputs":
            return dict(self.special)
        if rt == "GetInputMute":
            return {"inputMuted": self.muted[data["inputName"]]}
        if rt == "SetInputMute":
            self.muted[data["inputName"]] = data["inputMuted"]
            self.mgr._on_obs_event("InputMuteStateChanged", {"inputName": data["inputName"],
                                                             "inputMuted": data["inputMuted"]})
        return {}

    def user_sets(self, name, muted):
        """ユーザが OBS の画面でミュートを切り替えた。"""
        self.muted[name] = muted
        self.mgr._on_obs_event("InputMuteStateChanged", {"inputName": name, "inputMuted": muted})


def t_live_hook() -> None:
    print("\n[21a] 配信が始まった時の知らせ（ここから先の判定に配信の事実が入る）")
    tmp = Path(tempfile.mkdtemp())
    m, _s, _sub, _c = _manager(tmp, api=_FakeApi({"channels": [(200, {"items": [{"snippet": {"title": "AIイブ"}}]})]}))
    seen: list = []
    m._on_live = lambda: seen.append((m.active, m.voice_block() is not None))
    m.configure(use_obs=False)

    async def run():
        await m.start()
        check("**配信中になってから知らせる**（その時には配信の事実を渡せる）", seen == [(True, True)], f"{seen}")
        check("チャンネル名を読む", m.channel_title == "AIイブ")
        await m.stop()

    asyncio.run(run())


def t_carry() -> None:
    print("\n[21e] 配信の始め・終わりで、直前に話していたことの要約だけを持ち越す（会話そのものは入れない）")
    from types import SimpleNamespace

    from eve.memory.long_term import RagStore
    from eve.voice_loop import CARRY_POST_SEC, VoiceLoop

    tmp = Path(tempfile.mkdtemp())
    rag = RagStore(embedder=SimpleNamespace(), rag_file=str(tmp / "rag.jsonl"))
    rag._chunks = [
        {"timestamp": "2026-10-16T09:00:00+00:00", "summary": "ずっと前の話"},
        {"timestamp": "2026-10-16T09:50:00+00:00", "summary": "ユーザが今日アルケミーで5連敗したと話した"},
        {"timestamp": "2026-10-16T09:55:00+00:00", "summary": "ユーザが配信を始めると言った"},
        {"timestamp": "2026-10-16T10:10:00+00:00", "summary": "チャットのみかんさんが挨拶した"},
        {"timestamp": "2026-10-16T10:12:00+00:00", "summary": ""},
    ]
    got = rag.summaries_between("2026-10-16T09:30:00+00:00", "2026-10-16T10:00:00+00:00")
    check("**会話の時刻で区切りの手前の要約を古い順に拾う**（範囲外・空は入れない）",
          got == ["ユーザが今日アルケミーで5連敗したと話した", "ユーザが配信を始めると言った"], f"{got}")
    check("新しい方から k 件", rag.summaries_between("2026-10-16T00:00:00+00:00", "2026-10-16T23:00:00+00:00", k=1)
          == ["チャットのみかんさんが挨拶した"])
    stream = SimpleNamespace(active=True)
    fake = SimpleNamespace(_carry={"kind": "pre", "since": "2026-10-16T09:30:00+00:00",
                                   "until": "2026-10-16T10:00:00+00:00"}, stream=stream, rag=rag)
    text = VoiceLoop._carry_text(fake)
    check("**配信中は、配信の直前に開発者と話していたことの要約を渡す**（視聴者は知らない）",
          text.startswith("# 配信の直前に開発者と話していたこと") and "視聴者はこの会話を知らない" in text
          and "- ユーザが今日アルケミーで5連敗したと話した" in text, text)
    stream.active = False
    check("配信が終わったら、配信の前の要約は渡さない", VoiceLoop._carry_text(fake) is None)
    import time as _time

    fake._carry = {"kind": "post", "since": "2026-10-16T10:00:00+00:00", "until": "2026-10-16T10:30:00+00:00",
                   "note": "配信は約30分・コメント 1件（1人）。", "expires": _time.monotonic() + 60}
    text = VoiceLoop._carry_text(fake)
    check("**配信の後は、さっきまでの配信の要約を渡す**（配信どうだった？に答えられる）",
          text.startswith("# さっきまでの配信") and "みかんさん" in text and text.endswith("コメント 1件（1人）。"), text)
    stream.active = True
    check("次の配信が始まったら、前の配信の要約は渡さない", VoiceLoop._carry_text(fake) is None)
    stream.active = False
    fake._carry["expires"] = _time.monotonic() - 1
    check("配信の後しばらく（30分）たったら渡さない", VoiceLoop._carry_text(fake) is None and fake._carry is None
          and CARRY_POST_SEC == 1800.0)

    async def run():
        marks: list = []
        st = SimpleNamespace(started_iso="2026-10-16T10:00:00+00:00", started_mono=1.0, total=3,
                             people={"a": {}, "b": {}}, elapsed_sec=lambda: 1250.0)
        f2 = SimpleNamespace(stream=st, cache=SimpleNamespace(mark_boundary=lambda: marks.append("b")),
                             speech_state=SimpleNamespace(mark_session_start=lambda: marks.append("s")),
                             queue=SimpleNamespace(discard_kind=lambda k: 0), _carry=None)
        VoiceLoop._stream_session_end(f2)
        check("配信の終わりに、配信の要約を持ち越す準備をする（時間・コメント数も）",
              f2._carry["kind"] == "post" and f2._carry["since"] == st.started_iso
              and f2._carry["note"] == "配信は約21分・コメント 3件（2人）。" and marks[0] == "b", f"{f2._carry}")
        f3 = SimpleNamespace(feedback_worker=SimpleNamespace(trigger=lambda: None, is_idle=lambda: True),
                             cache=SimpleNamespace(turns_since=lambda w: [], mark_boundary=lambda: marks.append("b2")),
                             prediction=SimpleNamespace(watermark=None), _carry=None)
        await VoiceLoop._stream_session_start(f3, wait_sec=2.0)
        check("配信の始めに、手前30分の会話の要約を持ち越す準備をしてから区切る",
              f3._carry["kind"] == "pre" and f3._carry["since"] < f3._carry["until"] and marks[-1] == "b2", f"{f3._carry}")

    asyncio.run(run())


def t_cadence() -> None:
    print("\n[21f] 配信中は、自分から話すかを考え直す間隔の上限を短くする（話すかどうかはイブが決める）")
    from types import SimpleNamespace

    from eve.config import Config
    from eve.speech.monitor import SilenceMonitor, SpeechState

    clock = _Clock()
    st = SpeechState(now_fn=clock)
    st.evals_since_activity = 6          # 6回続けて話さなかった（普段の間隔は 5秒×2^6＝320 → 上限 300秒）
    st.last_eval_mono = clock.t
    triggered: list = []
    dec = SimpleNamespace(is_idle=lambda: True, trigger=lambda src: triggered.append(src))
    mon = SilenceMonitor(state=st, decider=dec, is_busy_fn=lambda: False, threshold_sec=5.0)
    clock.t += 40
    check("普段は40秒たっても考え直さない（間隔は倍々で最大5分）", mon.tick() is False and triggered == [])
    streaming = {"on": True}
    mon.max_interval_fn = lambda: 30.0 if streaming["on"] else None
    check("**配信中は上限（30秒）たてば考え直す**", mon.tick() is True and triggered == ["silence"])
    st.last_eval_mono = clock.t
    clock.t += 40
    streaming["on"] = False
    check("配信が終わったら普段の上限に戻る", mon.tick() is False)
    check("上限の既定は30秒", Config.STREAM_SPEECH_MAX_INTERVAL_SEC == 30.0)


class _Win:
    def __init__(self) -> None:
        self.closed = False

    def poll(self):
        return 0 if self.closed else None

    def terminate(self):
        self.closed = True


def t_taskwin() -> None:
    print("\n[21d] 配信中に画面に出した窓は決めた秒数で閉じ、何も出ていない間は OBS の作業画面を隠す")
    from types import SimpleNamespace

    import time as _time

    from eve.stream.scene import SRC_TASK

    tmp = Path(tempfile.mkdtemp())
    m, _s, _sub, _c = _manager(tmp, clock=_time.monotonic)
    m.s.taskwin_sec = 0.6
    calls: list = []

    class _O:
        connected = True

        async def call(self, rt, data=None, timeout=10.0):
            calls.append((rt, data))
            return {}

    m.obs = _O()
    m.scene = SimpleNamespace(items={SRC_TASK: 7})
    wins: list = []
    m.viewers_fn = lambda: [(w, n) for w, n in wins if w.poll() is None]

    async def run():
        m.active = True
        task = asyncio.create_task(m._taskwin_loop())
        await asyncio.sleep(0.3)
        check("何も出していない間は作業画面を隠す",
              calls and calls[-1][1]["sceneItemEnabled"] is False and calls[-1][1]["sceneItemId"] == 7, f"{calls}")
        w = _Win()
        wins.append((w, "calc.html"))
        await asyncio.sleep(0.4)
        check("出したら作業画面を出す", calls[-1][1]["sceneItemEnabled"] is True and not w.closed, f"{calls[-1]}")
        await asyncio.sleep(1.0)
        check("**決めた秒数たったら閉じる**", w.closed)
        await asyncio.sleep(0.3)
        check("閉じたら作業画面を隠す", calls[-1][1]["sceneItemEnabled"] is False, f"{calls[-1]}")
        w2 = _Win()
        wins.append((w2, "memo.txt"))
        await asyncio.sleep(0.3)
        w2.terminate()  # イブが先に閉じた（close_window）
        await asyncio.sleep(0.3)
        check("イブが先に閉じても作業画面を隠す", calls[-1][1]["sceneItemEnabled"] is False)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        n = len(calls)
        await m._set_task_visible(True)
        check("配信の後は作業画面を OBS の配置のまま（出す）に戻せる", len(calls) == n + 1 and calls[-1][1]["sceneItemEnabled"])

    asyncio.run(run())


def t_audio() -> None:
    print("\n[21b] 配信の音（PC の音はそのまま・マイクは切って始めてボタンに合わせる・イブの声は1回だけ）")
    from types import SimpleNamespace

    from eve.stream.scene import SRC_VOICE

    tmp = Path(tempfile.mkdtemp())
    special = {"desktop1": "デスクトップ音声", "desktop2": None, "mic1": "マイク", "mic2": None}

    async def settle():
        for _ in range(5):
            await asyncio.sleep(0)

    async def run():
        m, _s, _sub, _c = _manager(tmp)
        obs = _AudioObs(m, {"デスクトップ音声": False, "マイク": False, SRC_VOICE: False}, special)
        m.obs, m.scene = obs, SimpleNamespace(voice_ok=True)
        m.active = True
        await m._setup_audio()
        await settle()
        check("**PC の音（デスクトップ音声）はそのまま**（BGM・ゲームの音を流す）", obs.muted["デスクトップ音声"] is False)
        check("**OBS のマイクは切って始める**", obs.muted["マイク"] is True and not m.mic_live)
        check("**イブの声はデスクトップ音声に入っているので「イブ 声」は切る**（二重にしない）", obs.muted[SRC_VOICE] is True)
        v = m.voice_block()
        check("持ち主の声は配信に出ていない（事実）", "声も打った文字も配信には出ていない" in v, v)
        await m.set_mic(True)
        check("**ボタンでマイクを入れると OBS のマイクも入る**", obs.muted["マイク"] is False and m.mic_live)
        check("UI に「マイクが配信に乗っている」を出せる", m.snapshot()["mic"] is True)
        check("持ち主の声が配信に乗っている（事実）", "マイクの声は配信にも乗っていて" in m.voice_block())
        await m.set_mic(False)
        check("切ると OBS のマイクも切る", obs.muted["マイク"] is True and not m.mic_live)
        obs.user_sets("マイク", False)
        check("OBS の画面でマイクを入れても、配信に乗っていると分かる", m.mic_live)
        obs.user_sets("デスクトップ音声", True)
        await settle()
        check("**OBS でデスクトップ音声を切ったら「イブ 声」を入れる**（イブの声を消さない）", obs.muted[SRC_VOICE] is False)
        obs.user_sets("デスクトップ音声", False)
        await settle()
        check("デスクトップ音声を戻したら「イブ 声」を切る", obs.muted[SRC_VOICE] is True)
        m._closing = True
        await m._restore_mutes()
        check("**終わったら OBS のマイクを配信前の状態に戻す**・PC の音には触らない",
              obs.muted["マイク"] is False and obs.muted["デスクトップ音声"] is False)

        m2, _s, _sub, _c = _manager(tmp, audio="eve")
        obs2 = _AudioObs(m2, {"デスクトップ音声": False, "マイク": True, SRC_VOICE: True}, special)
        m2.obs, m2.scene = obs2, SimpleNamespace(voice_ok=True)
        m2.active = True
        await m2._setup_audio()
        await settle()
        check("eve: デスクトップ音声を切り、「イブ 声」だけを流す", obs2.muted["デスクトップ音声"] is True
              and obs2.muted[SRC_VOICE] is False)
        m2._closing = True
        await m2._restore_mutes()
        check("eve: 終わったらデスクトップ音声を戻す・もともと切っていたマイクは切ったまま",
              obs2.muted["デスクトップ音声"] is False and obs2.muted["マイク"] is True)

        m3, _s, _sub, _c = _manager(tmp, audio="eve")
        obs3 = _AudioObs(m3, {"デスクトップ音声": False, "マイク": False}, special)
        m3.obs, m3.scene = obs3, SimpleNamespace(voice_ok=False)
        await m3._setup_audio()
        check("アプリの音を取り込めない OBS では、eve でもデスクトップ音声を切らない（イブの声が消える）",
              obs3.muted["デスクトップ音声"] is False and not any(d and d.get("inputName") == SRC_VOICE
                                                              for _r, d in obs3.calls))

    asyncio.run(run())


def t_connect_history() -> None:
    print("\n[21c] 配信を探している間に届いたスーパーチャットを落とさない")
    tmp = Path(tempfile.mkdtemp())
    hist = {"items": [_item("o1", "textMessageEvent", text="前の配信の", publishedAt="2026-10-15T04:00:00Z"),
                      _item("t1", "textMessageEvent", name="@佐藤", text="こんばんは", ch="UCsato",
                            publishedAt="2026-10-15T05:00:20Z"),
                      _item("s1", "superChatEvent", name="@鈴木", ch="UCsuzu", publishedAt="2026-10-15T05:00:40Z",
                            superChatDetails={"amountDisplayString": "￥1,000", "userComment": "初スパチャ"})],
            "nextPageToken": "N"}
    m, _s, _sub, _c = _manager(tmp, api=_FakeApi({"liveChat/messages": [(200, hist)]}))
    m.started_iso = "2026-10-15T05:00:00+00:00"
    m.active = True

    async def run():
        await m._connect_chat(LiveVideo(video_id="V", title="t", state="live", chat_id="C"))
        check("始めた後に書かれた分は受け取る（前の分は読まない）",
              [x.id for x in m.paid] == ["s1"] and [x.id for x in m.unread] == ["t1"] and m.total == 2,
              f"{m.paid} {m.unread}")
        check("最初に読むのはスーパーチャット", m.pick().id == "s1")

    asyncio.run(run())


def t_ui_contract() -> None:
    print("\n[22] UI: 配信ボタン・確認・キャプチャ除外")
    import inspect

    from eve.ui import menu_bar, theme
    from eve.ui import app as app_mod

    items = {k: (on, off, init) for k, _l, on, off, init in menu_bar.ITEMS}
    check("メニューに配信がある（初期は OFF）", "stream" in items and items["stream"][2] is False)
    check("配信中は赤（live）", "e5484d" in theme.menu_button_qss("live"))
    src = inspect.getsource(app_mod)
    check("始める時は確認する（StreamDialog）", "StreamDialog(self)" in src and "dlg.exec()" in src)
    check("終える時も確認する", "QMessageBox.question" in src)
    check("押した瞬間に配信の表示を変えない（request_control を出すだけ）",
          'self.thread_.request_control("stream", True)' in src)
    check("値は configure_stream で先に渡す（call_soon_threadsafe の順番どおり）",
          src.index("self.thread_.configure_stream(") < src.index('self.thread_.request_control("stream", True)'))
    check("キャプチャ除外は配信前の値を覚えて戻す", "_capture_before" in src)
    check("配信中は設定を触らせない", "self.menu.settings.setEnabled(not busy)" in src)


def t_ui_window() -> None:
    print("\n[23] 本物の MainWindow（画面は開かない）: 確認 → 値を渡す → 要求・キャプチャ除外・設定のロック・表示")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    from eve.ui import app as app_mod

    QApplication.instance() or QApplication([])
    win = app_mod.MainWindow()
    calls: list = []

    class _Thread:
        def configure_stream(self, **kw):
            calls.append(("configure", kw))

        def request_control(self, key, on):
            calls.append(("request", key, on))

        def push_stream_geometry(self, geom):
            calls.append(("geom", geom))

    win.thread_ = _Thread()
    win._ready, win._speaking = True, False

    class _Dlg:
        result = QDialog.Accepted

        def __init__(self, parent=None):
            from types import SimpleNamespace

            self.url = SimpleNamespace(text=lambda: " https://youtu.be/wiYUKHr9-kA ")
            self.chat_only = SimpleNamespace(isChecked=lambda: False)

        def exec(self):
            return _Dlg.result

    orig_dlg, orig_q = app_mod.StreamDialog, QMessageBox.question
    try:
        app_mod.StreamDialog = _Dlg
        _Dlg.result = QDialog.Rejected
        win._menu_requested("stream", True)
        check("確認で「やめる」なら何も要求しない", calls == [], f"{calls}")
        _Dlg.result = QDialog.Accepted
        win._menu_requested("stream", True)
        check("確認したら、値を渡してから配信を要求する", [c[0] for c in calls] == ["configure", "request"]
              and calls[1] == ("request", "stream", True), f"{calls}")
        kw = calls[0][1]
        check("配信の URL・OBS を使うか・窓・Live2D の位置を渡す",
              kw["target"] == "https://youtu.be/wiYUKHr9-kA" and kw["use_obs"] is True
              and kw["main_window"].title == win.windowTitle()
              and kw["task_window"].title == "イブのタスクウィンドウ"
              and set(kw["geometry"]) == {"win", "view"}, f"{kw}")
        win.hide_cb.setChecked(True)
        win.hide_task_cb.setChecked(True)
        win._on_stream_event("capture", {"hide": False})
        check("配信を始める時はキャプチャ除外を外す（OBS に映す）",
              not win.hide_cb.isChecked() and not win.hide_task_cb.isChecked())
        win._on_control_state("stream", "on", True)
        check("配信中は設定を押せない", not win.menu.settings.isEnabled())
        check("配信ボタンは「● 配信中」", win.menu.buttons["stream"].text() == "● 配信中",
              win.menu.buttons["stream"].text())
        win._on_status({"mic_on": False, "stream": {"active": True, "chat": "live", "comments": 3, "viewers": 5,
                                                    "elapsed": 65}})
        check("配信の様子を出す", win.stream_line.text() == "● 配信中 1分・チャット接続中・コメント 3・視聴 5",
              win.stream_line.text())
        win._on_status({"mic_on": True, "stream": {"active": True, "chat": "live", "comments": 3, "elapsed": 65,
                                                   "mic": True}})
        check("マイクが配信に乗っている時はそう出す", win.stream_line.text().endswith("マイクが配信に乗っている"),
              win.stream_line.text())
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QResizeEvent

        win.resizeEvent(QResizeEvent(QSize(1200, 800), QSize(1280, 860)))
        check("配信中に大きさが変わったら Live2D の位置を送り直す（少し待ってから）", win._geom_timer.isActive())
        win._on_stream_event("capture", {"restore": True})
        check("配信が終わったらキャプチャ除外を配信前に戻す", win.hide_cb.isChecked() and win.hide_task_cb.isChecked())
        win._on_control_state("stream", "off", False)
        check("終わったら設定を押せる・表示を消す", win.menu.settings.isEnabled() and win.stream_line.text() == "")
        calls.clear()
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
        win._menu_requested("stream", False)
        check("終える確認で「いいえ」なら止めない", calls == [], f"{calls}")
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
        win._menu_requested("stream", False)
        check("終える確認で「はい」なら止める", calls == [("request", "stream", False)], f"{calls}")
    finally:
        app_mod.StreamDialog = orig_dlg
        QMessageBox.question = orig_q
        win.log_bridge.uninstall()


if __name__ == "__main__":
    t_parse()
    t_ids()
    t_client()
    t_reader()
    t_format()
    t_pick_latest()
    t_feed_loop()
    t_poll_errors()
    t_registry()
    t_blocks()
    t_repeat()
    t_context()
    t_decider()
    t_boundary()
    t_audio_text()
    t_voiceloop_hooks()
    t_control()
    t_obs_protocol()
    t_obs_misc()
    t_crop()
    t_scene()
    t_manager_obs_errors()
    t_live_hook()
    t_carry()
    t_cadence()
    t_taskwin()
    t_audio()
    t_connect_history()
    t_ui_contract()
    t_ui_window()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
