"""AlchemyManager — イブが Alchemy を遊ぶ流れ全体（準備・部屋・対戦・終わり）を持つ（loop 所有・単一書込）。

## 状態（ユーザ指定「普段は入れずに、画面や会話からAlchemyをやりそうになりそうならプレイできるように準備」）
- idle: 何もしない。**ルールは会話に入れない**（`voice_block()` が None）。
- prepared: ルールを会話に入れる。入口は2つ: 応答AI が `alchemy_prepare`/`alchemy_join` を呼んだ（会話）／
  画面認識が「Alchemy の画面が映っている」と答えた（画面）。しばらく何も無ければ idle に戻る（TTL）。
- room: 合言葉（既定「AIバトル」）の部屋で準備OKにして待っている。**画面認識はイブのゲーム画面だけを見る**。
- playing: 対戦中。手は判断役（`brain.py`）とコード（`autoplay.py`）が締切までに必ず打つ。

部屋に入る（＝ゲームを始めうる）のは `alchemy_join` だけ＝会話でユーザが遊ぶと言った時だけ。画面から勝手に
始めない（ユーザの確認点「普段の会話中に誤認してゲームを始めないか」）。

## 会話との関係
- 応答AI・発話判定には `voice_block()` / `decider_block()` で**サーバーの数から作った事実**だけを渡す。
- 大きく動いた時（点が入った・3連鎖・勝ち負け・始まり・相手が部屋に来た）だけ `react(文)` で発話判定を起こす。
  話すかどうか・何を言うかは発話判定と応答AI が決める（ここは文を言わせない）。
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ..clock import now_mono
from . import autoplay, events, sim
from .brain import Brain, Decision, steps_text, strategy_text
from .client import AlchemyClient, ClientError
from .rules import DEFAULT_RULES, PHASE_JA, SKILL_JA, counts_text, mat, recipe_text, rules_brief
from .screen import GameScreenFeed

logger = logging.getLogger(__name__)

IDLE, PREPARED, ROOM, PLAYING = "idle", "prepared", "room", "playing"

# 締切の何秒前までに打ち終えるか（判断役が遅くてもここでコードの手に切り替える）。
# 錬成は1手に数通（出力の選択・素材の配置・おわり）送るので長めに取る
_MARGIN = {"select": 6.0, "skill": 6.0, "brew": 15.0}
_HISTORY = 10          # 会話に見せる最近の出来事の数（ラウンドの出来事ごと）
_REACT_FRESH_SEC = 60  # 発話判定に「直前の出来事」として見せる長さ


@dataclass
class AlchemySettings:
    url: str = "https://alchemy-game.fly.dev/"
    keyword: str = "AIバトル"
    name: str = "イブ"
    mode: str = "action"
    headless: bool = True
    channel: str = "chrome"
    prepare_ttl_sec: float = 900.0
    reasoning_effort: str = "low"
    max_think_sec: float = 20.0
    nudge_sec: float = 300.0   # 部屋で1人で待ってこの秒数たったら、来るかを確かめる（0 で無効）
    giveup_sec: float = 30.0   # 確かめた後この秒数返事が無ければ、やめると伝えて部屋から抜ける
    max_rewaits: int = 2       # 確かめて返事があったのに来ない（待ち直し）がこの回数たまったら、次は確かめずに抜ける
    all_left_grace_sec: float = 10.0  # イブ以外が全員抜けてからイブも抜けるまで（ページの読み込み直し等のつなぎ直しを待つ）

    @classmethod
    def from_config(cls) -> "AlchemySettings":
        from ..config import Config

        return cls(
            url=Config.ALCHEMY_URL, keyword=Config.ALCHEMY_KEYWORD, name=Config.ALCHEMY_PLAYER_NAME,
            mode=Config.ALCHEMY_PLAY_MODE, headless=not Config.ALCHEMY_SHOW_WINDOW,
            channel=Config.ALCHEMY_BROWSER_CHANNEL, prepare_ttl_sec=Config.ALCHEMY_PREPARE_TTL_SEC,
            reasoning_effort=Config.ALCHEMY_REASONING_EFFORT, max_think_sec=Config.ALCHEMY_MAX_THINK_SEC,
            nudge_sec=Config.ALCHEMY_NUDGE_SEC, giveup_sec=Config.ALCHEMY_GIVEUP_SEC,
            max_rewaits=Config.ALCHEMY_MAX_REWAITS,
        )


ReportFn = Callable[[str, str, bool], None]          # (機能名, 結果の文, 成功か) → 機能の報告として応答AIへ
ReactFn = Callable[[str], None]                       # (出来事の文) → 発話判定を起こす
SayFn = Callable[[str, str], None]                    # (何について話すか, なぜ) → イブが自分から話す（言い回しは応答AI）
ScreenHook = Callable[[Optional[GameScreenFeed]], Awaitable[None]]  # 画面認識の見る先（None＝デスクトップに戻す）


class AlchemyManager:
    def __init__(self, registry, *, settings: Optional[AlchemySettings] = None,
                 react: Optional[ReactFn] = None, report: Optional[ReportFn] = None,
                 screen_hook: Optional[ScreenHook] = None, client_factory=None,
                 now_fn: Callable[[], float] = now_mono, say: Optional[SayFn] = None,
                 user_seq: Optional[Callable[[], int]] = None, busy: Optional[Callable[[], bool]] = None) -> None:
        self.s = settings or AlchemySettings.from_config()
        self._say = say
        self._user_seq = user_seq or (lambda: 0)   # ユーザが話すたびに増える数（催促の後に返事があったか）
        self._busy_fn = busy or (lambda: False)     # イブが話している・応答中か
        self._wait_since: Optional[float] = None    # 部屋で1人で待ち始めた（待ち直した）時刻
        self._room_since: Optional[float] = None    # 部屋で1人で待ち始めた最初の時刻（待ち直しても戻さない）
        self._rewaits = 0                           # 確かめた後に返事があって待ち直した回数
        self._nudges = 0                            # 来るかを確かめた回数
        self._nudged_at: Optional[float] = None     # 来るかを確かめた時刻
        self._nudge_seq = 0
        self._quiet_from = 0.0
        self._wait_task: Optional[asyncio.Task] = None
        self._seats: dict = {}                      # 対戦中のほかの席の様子（in / left / auto / bot）
        self._all_left_since: Optional[float] = None
        self.brain = Brain(registry, self.s.mode, reasoning_effort=self.s.reasoning_effort,
                           max_wait_sec=self.s.max_think_sec)
        self._react = react
        self._report = report
        self._screen_hook = screen_hook
        self._client_factory = client_factory or self._default_client
        self._now = now_fn
        self.state = IDLE
        self.client: Optional[AlchemyClient] = None
        self.feed: Optional[GameScreenFeed] = None
        self._screen_on = False
        self._screen_lock = asyncio.Lock()  # 部屋に入った時と対戦の始まりが重なっても切り替えは1回
        self.rules_d: dict = dict(DEFAULT_RULES)
        self.rules = sim.Rules.from_catalog(DEFAULT_RULES)
        self.recipes: dict = {}
        self.seat: Optional[int] = None
        self.view: Optional[dict] = None
        self._view_mono = 0.0
        self.members: list = []
        self.room_playing = False
        self.history: deque = deque(maxlen=_HISTORY)
        self.last_event: Optional[tuple] = None   # (文, mono) 発話判定に見せる直前の大きな出来事
        self.last_result: Optional[str] = None    # 直前の対戦の結果（対戦後の会話用）
        self.moves: list = []                     # このラウンドのイブの手（会話用）
        self.intent = ""
        self.stats: dict = {}
        self._touched = 0.0
        self._acted: set = set()
        self._act_task: Optional[asyncio.Task] = None
        self._busy: Optional[asyncio.Task] = None  # 部屋に入る/出る（同時に1つ）
        self._tasks: set = set()
        self._waiters: list = []
        self._start_pending = False
        self._ttl_task: Optional[asyncio.Task] = None
        self._game_t0 = 0.0
        # 準備OKのつもりか（部屋に入った〜対戦が始まるまで）。通信が切れてつなぎ直すと、ページが部屋に入り直すが
        # サーバーは準備OKを外すので、これを見て付け直す
        self._want_ready = False
        self._welcomed = 0  # サーバーとつながった回数（2回目以降＝つなぎ直し）
        # 対戦が終わってから部屋が閉じるまで（サーバーは終わりの演出の分だけ部屋を対戦中のまま持ち、閉じる時に全員の
        # 準備OKを外す）。この間に準備OKにしても消えるので、もう一回の時は閉じる（部屋の知らせが来る）まで待つ
        self._room_finishing = False

    # ------------------------------------------------------------------ 起動・終了
    def start(self) -> None:
        if self._ttl_task is None:
            self._ttl_task = asyncio.create_task(self._ttl_loop(), name="alchemy-ttl")
        if self._wait_task is None:
            self._wait_task = asyncio.create_task(self._wait_loop(), name="alchemy-wait")
        logger.info("🎮 Alchemy 稼働（%s・合言葉「%s」・遊ばせ方 %s・判断役 %s）", self.s.url, self.s.keyword,
                    self.brain.mode, self.brain._role)

    async def stop(self) -> None:
        for t in [self._ttl_task, self._wait_task, self._act_task, self._busy, *self._tasks]:
            if t is not None and not t.done():
                t.cancel()
        self._ttl_task = None
        self._wait_task = None
        if self.client is not None and self.client.alive:
            try:
                await self.client.leave()
            except Exception:
                pass
        await self._close_client()

    def _default_client(self) -> AlchemyClient:
        return AlchemyClient(self.s.url, self.s.name, on_message=self.on_message, on_lost=self._on_lost,
                             headless=self.s.headless, channel=self.s.channel)

    def _on_lost(self, why: str) -> None:
        """部屋・対戦から外れた（ページが落ちた／つなぎ直したら対戦が無かった）。外れたことを機能の報告で伝える。"""
        if self.state in (ROOM, PLAYING):
            self._spawn(self._lost(why), "alchemy-lost")

    async def _lost(self, why: str) -> None:
        was = self.state
        if was not in (ROOM, PLAYING):
            return
        await self._set_screen(False)
        await self._close_client()
        self.members = []
        self.view = None
        self._want_ready = False
        self.state = PREPARED
        self.touch()
        text = (f"（{why}ため、Alchemy の{'対戦' if was == PLAYING else '部屋'}から外れた。"
                "もう一度遊ぶなら部屋に入り直す）")
        self.history.append(text)
        logger.warning("🎮 %s", text)
        self._report_result("alchemy", text, False)

    def _spawn(self, coro, name: str) -> asyncio.Task:
        t = asyncio.create_task(coro, name=name)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    # ------------------------------------------------------------------ 状態の公開
    @property
    def active(self) -> bool:
        """ルールを会話に入れているか（idle 以外）。"""
        return self.state != IDLE

    def owns_screen(self) -> bool:
        """画面認識がイブのゲーム画面を見ているか（＝ユーザの画面は見ていない）。"""
        return self._screen_on

    def touch(self) -> None:
        self._touched = self._now()

    # ------------------------------------------------------------------ 入口（応答AI の機能・画面認識）
    def prepare(self, source: str) -> str:
        """ルールを会話に入れる（部屋にはまだ入らない）。source は観測用（tool / screen / join）。

        返す文はログ用（`alchemy_prepare` は結果を応答AI に報告しない＝この文をイブに言い直させない）。
        """
        self.touch()
        if self.state == IDLE:
            self.state = PREPARED
            logger.info("🎮 Alchemy の準備（ルールを会話に入れる・きっかけ: %s）", source)
        return "Alchemy のルールを会話に入れた（部屋には入っていない）"

    def note_seen_on_screen(self) -> None:
        """画面認識が「ユーザの画面に Alchemy が映っている」と答えた（準備だけ・始めない）。"""
        if self._screen_on:
            return  # 自分のゲーム画面を見ている間は数えない
        if self.state == IDLE:
            self.prepare("screen")
        else:
            self.touch()

    def tool_prepare(self, args: dict) -> str:
        return self.prepare("tool")

    def tool_join(self, args: dict) -> str:
        """部屋に入る（または対戦後にもう一度準備OKにする）。結果は後で機能の報告として届く。"""
        if self.state == PLAYING:
            return "（もう対戦中なので、新しく部屋に入る必要はない）"
        if self._busy is not None and not self._busy.done():
            return "（いま部屋の出入りの途中）"
        self.prepare("join")
        self._busy = self._spawn(self._join(), "alchemy-join")
        return "Alchemy の部屋に入る準備を始めた"

    def tool_leave(self, args: dict) -> str:
        if self.state == IDLE and self.client is None:
            return "（今は Alchemy をしていない）"
        if self.state == PREPARED and self.client is None:
            self._go_idle("やめると言われた")
            return "Alchemy の準備をやめた"
        if self._busy is not None and not self._busy.done():
            self._busy.cancel()
        self._busy = self._spawn(self._leave(), "alchemy-leave")
        return "Alchemy をやめる"

    # ------------------------------------------------------------------ 部屋に入る・出る
    async def _open_client(self) -> None:
        self.client = self._client_factory()
        await self.client.start()

    async def _close_client(self) -> None:
        if self.feed is not None:
            await self.feed.stop()
        c, self.client = self.client, None
        if c is not None:
            await c.close()

    async def _join(self) -> None:
        kw = self.s.keyword
        try:
            if self.client is None or not self.client.alive:
                await self._open_client()
            # もう一回の時: 前の対戦の部屋が閉じるまで待つ（終わりの演出の間に準備OKにしても、閉じる時に外される・
            # 演出の途中で画面を部屋に戻すとゲームの画面がエラーを出す）
            t_wait = self._now()
            while self._room_finishing and self._now() - t_wait < 25.0:
                await asyncio.sleep(0.2)
            await self.client.back_to_room()  # 対戦の結果画面にいたら部屋の画面へ（もう一回遊ぶ時）
            if not self._me_in_room():
                waiter = self._expect(lambda m: (m.get("type") == "keyword" and m.get("keyword"))
                                      or (m.get("type") == "error" and m.get("for") == "keyword.join"))
                await self.client.join_keyword(kw)
                msg = await self._await(waiter, 8.0)
                if msg is None:
                    raise ClientError("合言葉の部屋の返事が来なかった")
                if msg.get("type") == "error":
                    raise ClientError(msg.get("message") or "合言葉の部屋に入れなかった")
            # 自分が準備OKになった知らせ（か、すぐ始まった）を待つ。部屋に入り直した時の知らせ（まだ準備中）で
            # 進むと「準備中」と報告してしまう
            waiter = self._expect(lambda m: m.get("type") == "match"
                                  or (m.get("type") == "keyword"
                                      and any(x.get("me") and x.get("ready") for x in m.get("members") or []))
                                  or (m.get("type") == "error" and m.get("for") == "keyword.ready"))
            await self.client.set_ready(True)
            msg = await self._await(waiter, 5.0)
            if msg is not None and msg.get("type") == "error":
                raise ClientError(msg.get("message") or "準備OKにできなかった")
            self._want_ready = self.state != PLAYING
            await self._enter_room()
            # 入れた時は報告しない（ユーザ 2026-10-11「入室退出の報告はなくしていい」・受付の一言と重なっていた）。
            # 部屋の様子は `# Alchemy` に入る。ユーザが来なければ待ちの見張り（`_wait_tick`）が催促する
            logger.info("🎮 合言葉「%s」の部屋で準備OKにした（%s・部屋のメンバー: %s）", kw,
                        "すぐに対戦が始まった" if self.state == PLAYING else "ほかの人を待つ", self._members_text())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001（ブラウザ・通信の失敗は報告して続ける）
            why = str(e) if isinstance(e, ClientError) else f"{type(e).__name__}"
            logger.warning("🎮 部屋に入れなかった: %s", why)
            if self.state not in (ROOM, PLAYING):
                await self._close_client()
            self._report_result("alchemy_join", f"（Alchemy の部屋に入れなかった: {why}）", False)

    async def _leave(self) -> None:
        was_playing = self.state == PLAYING
        if self._act_task is not None and not self._act_task.done():
            self._act_task.cancel()  # 打っている途中の手は打たない（抜けた後に送っても「参加していません」になる）
        try:
            if self.client is not None and self.client.alive:
                await self.client.leave()
                await asyncio.sleep(0.3)  # 退出がサーバーに届く間
        except Exception:
            logger.debug("🎮 退出の送信に失敗（閉じて終える）", exc_info=True)
        await self._set_screen(False)
        await self._close_client()
        self.members = []
        self.view = None
        self._want_ready = False
        self.state = PREPARED  # 対戦の話を続けられるよう、ルールと結果はしばらく残す（TTL で消える）
        self.touch()
        logger.info("🎮 Alchemy をやめた（%s）", "対戦の途中で抜けた" if was_playing else "部屋を出た")

    async def _enter_room(self) -> None:
        self._reset_wait()  # 待ちの数え（確かめた回数・待ち直し）は部屋に入るたびに 0 から
        if self.state not in (ROOM, PLAYING):
            self.state = ROOM
        self.touch()
        await self._set_screen(True)

    async def _set_screen(self, on: bool) -> None:
        """画面認識の見る先を切り替える（イブのゲーム画面 ⇔ デスクトップ）。"""
        async with self._screen_lock:
            await self._set_screen_locked(on)

    async def _set_screen_locked(self, on: bool) -> None:
        if on == self._screen_on:
            return
        if on:
            if self.client is None:
                return
            self.feed = GameScreenFeed(self.client)
            await self.feed.warm()
            self.feed.start()
        self._screen_on = on
        if self._screen_hook is not None:
            try:
                await self._screen_hook(self.feed if on else None)
            except Exception:
                logger.exception("🎮 画面認識の切り替えに失敗（続行）")
        if not on and self.feed is not None:
            await self.feed.stop()
            self.feed = None
        logger.info("🎮 画面認識の見る先: %s", "イブのゲーム画面（ユーザの画面は見ない）" if on else "デスクトップ")

    def _go_idle(self, why: str) -> None:
        self.state = IDLE
        self.last_result = None
        self.history.clear()
        logger.info("🎮 Alchemy の準備を解いた（ルールを会話から外す・%s）", why)

    def check_ttl(self) -> bool:
        """準備（ルールを会話に入れた状態）の期限切れなら idle に戻す（戻したら True）。部屋にいる間は戻さない。"""
        if (self.state == PREPARED and self.client is None
                and self._now() - self._touched > self.s.prepare_ttl_sec):
            self._go_idle(f"{int(self.s.prepare_ttl_sec // 60)}分 Alchemy の話が無かった")
            return True
        return False

    async def _ttl_loop(self) -> None:
        while True:
            await asyncio.sleep(15.0)
            try:
                self.check_ttl()
            except Exception:
                logger.exception("🎮 TTL の確認で例外（続行）")

    # ------------------------------------------------------------------ 部屋で待つ（催促と見切り）
    # ユーザ 2026-10-11「5分以上待ってもユーザーが入室しなければ催促して、返事がない（30秒）・やらない反応を示したら
    # やめる意思を示して退出」。決めるのは「いつ」だけで、何をどう言うかは応答AI（言い回しの例は渡さない）。
    # 「やらない」と返された時は応答AI が alchemy_leave を呼ぶ（ユーザ発話への返事なので機能が使える）。
    # 返事が無い時はイブが自分から話す番で機能を使えないので、抜けるのはここ（コード）がやる
    def _alone(self) -> bool:
        return self.state == ROOM and self.client is not None and not any(not m.get("me") for m in self.members)

    def waiting_sec(self) -> Optional[float]:
        """部屋で1人で待っている秒数（待っていなければ None）。"""
        if not self._alone() or self._wait_since is None:
            return None
        return self._now() - self._wait_since

    def _wait_tick(self) -> None:
        now = self._now()
        if not self._alone() or self._busy_leave():
            self._reset_wait()
            return
        if self._wait_since is None:
            self._wait_since = now
        if self._room_since is None:
            self._room_since = now
        if self._busy_fn():
            self._quiet_from = now  # 話している間は数えない（催促を言い終わってから返事を待つ）
        if self._nudged_at is None:
            if now - self._wait_since >= self.s.nudge_sec and not self._busy_fn():
                total = max(1, int((now - self._room_since) // 60))
                if self._rewaits >= self.s.max_rewaits:
                    # 確かめて返事はあったのに来ない、を上限まで繰り返した → もう確かめずに抜ける
                    # （ユーザ 2026-10-11「2回以上待ち直したらさすがに抜けていい」）
                    logger.info("🎮 %d回待ち直しても来ない（計%d分）→ やめると伝えて部屋から抜ける", self._rewaits, total)
                    self.history.append(f"イブは部屋で計{total}分待ったがユーザが来なかったので、部屋から抜けた")
                    self._say_now(f"部屋で待ち始めてから計{total}分、何度か確かめたのにユーザが部屋に来ないので、今日の Alchemy は"
                                  "やめて部屋から抜けることを相手に伝える（もう部屋から抜けている）", "何度待っても来ない")
                    self._give_up()
                    return
                self._nudged_at = self._quiet_from = now
                self._nudge_seq = self._user_seq()
                self._nudges += 1
                again = f"（来るかを確かめるのは{self._nudges}回目）" if self._nudges > 1 else ""
                text = (f"合言葉「{self.s.keyword}」の部屋でイブだけが計{total}分待っていて、ユーザがまだ部屋に入ってこない{again}。"
                        "遊ぶのか、やめるのかを相手に確かめる")
                self.history.append(f"イブが部屋で計{total}分待っても、ユーザが部屋に入ってこなかった")
                logger.info("🎮 部屋で計%d分待った（ユーザが来ない）→ 遊ぶかやめるかを確かめる（%d回目）", total, self._nudges)
                self._say_now(text, "部屋で待っているのに、ユーザが来ない")
            return
        if self._user_seq() != self._nudge_seq:
            # 返事があった: どうするか（やめる・今行く・関係ない話）は応答AI が決めた（やめるなら alchemy_leave）。
            # 抜けずに待ち直す。言われた中身はコードでは見分けない（語の一覧で判定しない）ので、回数で区切る
            self._rewaits += 1
            logger.info("🎮 確かめた後にユーザが話した → 待ち直す（%d回目・上限 %d）", self._rewaits, self.s.max_rewaits)
            self._wait_since = now
            self._nudged_at = None
            return
        if now - self._quiet_from >= self.s.giveup_sec:
            sec = int(self.s.giveup_sec)
            logger.info("🎮 確かめてから%d秒返事が無い → やめると伝えて部屋から抜ける", sec)
            self.history.append(f"来るか確かめてから{sec}秒返事が無かったので、イブは部屋から抜けた")
            self._say_now(f"部屋に来るか確かめてから{sec}秒たっても返事が無いので、今日の Alchemy はやめて"
                          "部屋から抜けることを相手に伝える（もう部屋から抜けている）", "返事が無い")
            self._give_up()

    def _reset_wait(self) -> None:
        self._wait_since = self._nudged_at = self._room_since = None
        self._rewaits = self._nudges = 0

    def _give_up(self) -> None:
        self._wait_since = self._nudged_at = None
        self._busy = self._spawn(self._leave(), "alchemy-giveup")

    def _busy_leave(self) -> bool:
        return self._busy is not None and not self._busy.done()

    def _say_now(self, content: str, reason: str) -> None:
        if self._say is None:
            return
        try:
            self._say(content, reason)
        except Exception:
            logger.exception("🎮 自分から話す合図で例外（無視）")

    async def _wait_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                if self.s.nudge_sec > 0:
                    self._wait_tick()
                self._left_tick()
            except Exception:
                logger.exception("🎮 待ちの見張りで例外（続行）")

    def _report_result(self, name: str, content: str, ok: bool) -> None:
        if self._report is None:
            return
        try:
            self._report(name, content, ok)
        except Exception:
            logger.exception("🎮 機能の報告に失敗")

    def _fire_react(self, text: str) -> None:
        self.last_event = (text, self._now())
        if self._react is None:
            return
        try:
            self._react(text)
        except Exception:
            logger.exception("🎮 反応の合図で例外（無視）")

    # ------------------------------------------------------------------ 受信
    def _expect(self, pred) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((pred, fut))
        return fut

    async def _await(self, fut: asyncio.Future, timeout: float) -> Optional[dict]:
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._waiters = [(p, f) for p, f in self._waiters if f is not fut]
            if not fut.done():
                fut.cancel()

    def on_message(self, msg: dict) -> None:
        """ページが受けた通信（loop 上・同期）。重い処理はタスクへ。"""
        t = msg.get("type")
        try:
            if t == "welcome":
                self._welcomed += 1
                if self._welcomed > 1 and self.state == PLAYING and not msg.get("resume"):
                    # つなぎ直したら対戦が無かった（サーバーの再起動・長い切断で部屋が閉じた）
                    self._on_lost("通信が切れている間に対戦が終わっていた")
            elif t == "keyword":
                self._on_keyword(msg)
            elif t == "match":
                self._on_match(msg)
            elif t == "joined":
                self._on_joined(msg)
            elif t == "state":
                self._on_state(msg.get("state") or {})
            elif t == "events":
                for ev in msg.get("events") or []:
                    self._on_event(ev)
            elif t == "error":
                logger.info("🎮 サーバー: %s（%s）", msg.get("message"), msg.get("for"))
            elif t == "kicked":
                logger.warning("🎮 サーバーから切断された: %s", msg.get("reason"))
        except Exception:
            logger.exception("🎮 受信の処理で例外（続行）: %s", t)
        for pred, fut in list(self._waiters):
            if not fut.done():
                try:
                    if pred(msg):
                        fut.set_result(msg)
                except Exception:
                    pass

    def _me_in_room(self) -> bool:
        return any(m.get("me") for m in self.members)

    def _members_text(self) -> str:
        return "・".join(
            f"{'イブ' if m.get('me') else m.get('name', '?')}（{'準備OK' if m.get('ready') else '準備中'}）"
            for m in self.members) or "（まだ誰もいない）"

    def _on_keyword(self, msg: dict) -> None:
        before = {m.get("name") for m in self.members if not m.get("me")}
        self.members = list(msg.get("members") or []) if msg.get("keyword") else []
        self.room_playing = bool(msg.get("playing"))
        if not self.room_playing:
            self._room_finishing = False  # 前の対戦の部屋が閉じた（準備OKが外された後の知らせ）
        me = next((m for m in self.members if m.get("me")), None)
        if (self.state == ROOM and self._want_ready and me is not None and not me.get("ready")
                and not self.room_playing and self.client is not None
                and (self._busy is None or self._busy.done())):
            # 準備OKのつもりなのに外れている（つなぎ直しでページが部屋に入り直した等）→ 付け直す
            logger.info("🎮 準備OKが外れていたので付け直す")
            self._spawn(self.client.set_ready(True), "alchemy-reready")
        if self.state == ROOM:
            joined = [m.get("name") for m in self.members if not m.get("me") and m.get("name") not in before]
            for name in joined:
                text = f"{name}が合言葉「{self.s.keyword}」の部屋に入ってきた（今のメンバー: {self._members_text()}）"
                self.history.append(text)
                self._fire_react(text)

    def _on_match(self, msg: dict) -> None:
        self.state = PLAYING
        self._want_ready = False  # 次の対戦の準備OKは、もう一回と言われた時（alchemy_join）だけ
        self.touch()
        self._acted.clear()
        self.moves = []
        self.intent = ""
        self.brain.strategy = sim.NO_STRATEGY
        self.history.clear()
        self.stats = {}
        self._start_pending = True
        self._game_t0 = self._now()
        self._seats = {}
        self._all_left_since = None
        if not self._screen_on:
            self._spawn(self._set_screen(True), "alchemy-screen-on")

    def _on_joined(self, msg: dict) -> None:
        cat = msg.get("catalog") or {}
        if cat.get("rules"):
            self.rules_d = dict(cat["rules"])
            self.rules, self.recipes = sim.load_catalog(cat)
        self.seat = msg.get("seat")

    def _on_state(self, view: dict) -> None:
        if not view.get("players"):
            return
        self.view = view
        self._view_mono = self._now()
        if self._start_pending and self.state == PLAYING:
            self._start_pending = False
            others = [p.get("name", "?") for p in view["players"] if p.get("seat") != (view.get("me") or {}).get("seat")]
            text = f"Alchemy の対戦が始まった（相手: {'・'.join(others)}・{len(view['players'])}人対戦）"
            self.history.append(text)
            self._fire_react(text)
        if self.state == PLAYING:
            self._track_seats(view)
        if view.get("phase") == "end":
            return
        self._schedule_act()

    # ------------------------------------------------------------------ 誰が参加しているか（抜けた・戻った）
    # ユーザ 2026-10-12「誰かが抜けたら情報を渡す、EVE以外全員抜けてBotになったらEVEも抜ける。誰が参加してるのか
    # 認識させる」。以前はユーザが黙って抜けてもイブは気づかず、代わりの自動操作と打ち続けて勝ち、戻ったユーザに
    # 「部屋で待たせるなんて」と言った（実測）。「やめよう」に抜けそこねた時も、ユーザが自分の画面で抜ければ拾える
    @staticmethod
    def _seat_state(p: dict) -> str:
        if p.get("bot"):
            return "bot"
        if p.get("connected") is False:
            return "left"
        if p.get("autopilot"):
            return "auto"
        return "in"

    _SEAT_JA = {"in": "参加中", "left": "抜けた（自動操作が代わりに打っている）",
                "auto": "時間切れが続いて自動操作中", "bot": "Bot"}

    def _track_seats(self, view: dict) -> None:
        me_seat = (view.get("me") or {}).get("seat", self.seat)
        for p in view.get("players") or []:
            seat = p.get("seat")
            if seat == me_seat:
                continue
            new, old = self._seat_state(p), self._seats.get(seat)
            self._seats[seat] = new
            if old is None or old == new or new == "bot":
                continue
            name = p.get("name", "?")
            if new == "left":
                text = f"{name}が対戦から抜けた（接続が切れて、自動操作が代わりに打っている）"
            elif old == "left":
                text = f"{name}が対戦に戻ってきた"
            elif new == "auto":
                text = f"{name}は時間切れが続いて、自動操作が代わりに打っている"
            else:
                text = f"{name}が自分で打つのに戻った"
            self.history.append(text)
            logger.info("🎮 %s", text)
            self._fire_react(text)
        humans = [s for s, st in self._seats.items() if st != "bot"]
        if humans and all(self._seats[s] == "left" for s in humans):
            if self._all_left_since is None:
                self._all_left_since = self._now()
        else:
            self._all_left_since = None

    def _left_tick(self) -> None:
        """イブ以外の人が全員抜けて自動操作になったら、少し待って（つなぎ直しの分）イブも抜ける。"""
        if self.state != PLAYING or self._all_left_since is None or self._busy_leave():
            return
        if self._now() - self._all_left_since < self.s.all_left_grace_sec:
            return
        self._all_left_since = None
        logger.info("🎮 イブ以外が全員対戦から抜けた → イブも抜ける")
        self.history.append("イブ以外が全員対戦から抜けたので、イブも対戦から抜けた")
        self._say_now("一緒に対戦していた人が全員抜けて自動操作に代わったので、イブも対戦から抜けた（もう抜けている）。"
                      "そのことを相手に伝える", "相手が全員抜けた")
        self._busy = self._spawn(self._leave(), "alchemy-all-left")

    def _participants_text(self) -> str:
        v = self.view or {}
        me_seat = (v.get("me") or {}).get("seat", self.seat)
        return "・".join(
            "イブ（自分）" if p.get("seat") == me_seat
            else f"{p.get('name', '?')}（{self._SEAT_JA[self._seat_state(p)]}）"
            for p in v.get("players") or [])

    def _on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        players = (self.view or {}).get("players") or []
        me = self.seat
        if kind == "reveal":
            self.history.append(events.reveal_text(ev, players, me, self.recipes))
        elif kind == "skills":
            text = events.skills_text(ev, players, me)
            if text:
                self.history.append(text)
        elif kind == "resolve":
            text, big, why = events.resolve_summary(ev, players, me, self.rules.win_points)
            self.history.append(text)
            logger.info("🎮 %s", text)
            if big:
                self._fire_react(text)
        elif kind == "round":
            self.moves = []
        elif kind == "end":
            text = events.end_text(ev, players, me)
            self.history.append(text)
            self.last_result = text
            self.state = ROOM  # 合言葉の部屋に戻る（もう一回は alchemy_join で準備OK）
            self._room_finishing = True  # 演出が終わって部屋が閉じる知らせ（keyword）まで
            logger.info("🎮 %s（%.0f秒）", text, self._now() - self._game_t0)
            self._log_stats()
            self._fire_react(text)

    # ------------------------------------------------------------------ 手を打つ
    def _needed(self, view: dict) -> Optional[str]:
        me = view.get("me") or {}
        phase = view.get("phase")
        if phase == "select" and me.get("hand") and not me.get("selection"):
            return "select"
        if phase == "skill" and any(int(v) > 0 for v in (me.get("skill_pending") or {}).values()):
            return "skill"
        if phase == "brew" and me.get("brew") and not me["brew"].get("done"):
            return "brew"
        return None

    def _schedule_act(self) -> None:
        if self.state != PLAYING or self.view is None or not self.recipes:
            return
        if self._act_task is not None and not self._act_task.done():
            return
        kind = self._needed(self.view)
        if kind is None:
            return
        key = (self.view.get("round"), kind)
        if key in self._acted:
            return
        self._acted.add(key)
        self._act_task = asyncio.create_task(self._act(kind, dict(self.view)), name=f"alchemy-{kind}")
        self._act_task.add_done_callback(lambda _t: self._after_act())

    def _after_act(self) -> None:
        # 打っている間に次の段階の state が来ていたら、ここで拾う
        try:
            self._schedule_act()
        except Exception:
            logger.exception("🎮 次の手の予定で例外")

    def _budget(self, kind: str, view: dict) -> float:
        deadline, server_now = view.get("deadline"), view.get("server_time")
        if deadline and server_now:
            remaining = float(deadline) - float(server_now) - (self._now() - self._view_mono)
        else:
            remaining = float(self.rules.timers.get(kind, 45))
        return max(1.0, remaining - _MARGIN.get(kind, 6.0))

    async def _act(self, kind: str, view: dict) -> None:
        budget = self._budget(kind, view)
        recent = list(self.history)
        t0 = self._now()
        try:
            if kind == "select":
                d = await self.brain.decide_select(view, self.rules, self.rules_d, self.recipes,
                                                   budget=budget, recent=recent)
                ok = await self._send_select(d.payload)
                if not ok:
                    fb = autoplay.choose_select(view, self.rules, self.recipes, self.brain.strategy)
                    if fb != d.payload:
                        d.source += "→コードで打ち直し"
                        ok = await self._send_select(fb)
                        d.payload = fb
                self._record(d, view, ok, self._now() - t0)
            elif kind == "skill":
                d = await self.brain.decide_skills(view, self.rules, self.rules_d, self.recipes,
                                                   budget=budget, recent=recent)
                ok = await self._send_skills(d.payload)
                self._record(d, view, ok, self._now() - t0)
            elif kind == "brew":
                d = await self.brain.decide_brew(view, self.rules, self.rules_d, self.recipes,
                                                 budget=budget, recent=recent)
                ok = await self._send_brew(d.payload, view)
                self._record(d, view, ok, self._now() - t0)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("🎮 手を打つ途中で例外（締切でサーバーが自動で進める）: %s", kind)

    async def _send(self, msg: dict, done, timeout: float = 4.0) -> bool:
        """1通送り、done(state) が真の state か、その操作へのエラーが来るまで待つ。"""
        if self.client is None or not self.client.alive:
            return False
        kind = msg.get("type")
        fut = self._expect(lambda m: (m.get("type") == "state" and done(m.get("state") or {}))
                           or (m.get("type") == "error" and m.get("for") == kind))
        if not await self.client.send(msg):
            self._waiters = [(p, f) for p, f in self._waiters if f is not fut]
            return False
        got = await self._await(fut, timeout)
        if got is None:
            logger.info("🎮 %s の返事が来なかった", kind)
            return False
        if got.get("type") == "error":
            logger.info("🎮 %s が通らなかった: %s", kind, got.get("message"))
            return False
        return True

    async def _send_select(self, picks: list) -> bool:
        if not picks:
            return False
        cards = [{"card": c, "rotated": bool(r)} for c, r in picks]
        return await self._send({"type": "play.cards", "cards": cards},
                                lambda st: bool((st.get("me") or {}).get("selection")) or st.get("phase") != "select")

    async def _send_skills(self, uses: list) -> bool:
        ok = True
        for u in uses:
            left_before = self._pending_total()
            sent = await self._send({"type": "skill.use", "skill": u["skill"], "materials": list(u["materials"])},
                                    lambda st, n=left_before: _pending_total(st) < n or st.get("phase") != "skill")
            ok = ok and sent
        if self._pending_total() > 0 and (self.view or {}).get("phase") == "skill":
            await self._send({"type": "skill.skip"}, lambda st: _pending_total(st) == 0 or st.get("phase") != "skill")
        return ok

    def _pending_total(self) -> int:
        return _pending_total(self.view or {})

    async def _send_brew(self, steps: list, view: dict) -> bool:
        sides = autoplay.tableau_sides(view, self.recipes)
        ok = True
        for st in steps:
            if (self.view or {}).get("phase") != "brew":
                return False
            ins, outs = sides[st.index]
            choice = {str(k): m for k, m in enumerate(st.outputs) if len(outs[k]) > 1}
            made = (lambda s, i=st.index: i in (((s.get("me") or {}).get("brew") or {}).get("used") or [])
                    or s.get("phase") != "brew")
            done_step = False
            if choice:
                await self._send({"type": "brew.choose", "index": st.index, "outputs": choice},
                                 lambda s, i=st.index: str(i) in (((s.get("me") or {}).get("brew") or {}).get("pending") or {})
                                 or s.get("phase") != "brew")
            if ins:
                for slot, (m, src) in enumerate(st.plan):
                    last = slot == len(st.plan) - 1
                    cond = made if last else (
                        lambda s, i=st.index, k=slot: str(k) in ((((s.get("me") or {}).get("brew") or {})
                                                                  .get("pending") or {}).get(str(i)) or {}).get("slots", {})
                        or s.get("phase") != "brew")
                    if not await self._send({"type": "brew.place", "index": st.index, "slot": slot,
                                             "material": m, "source": src}, cond):
                        break
                else:
                    done_step = True
            if not done_step:
                # 置き方が通らなかった（見積もりと実際の箱のずれ）→ サーバーのおまかせで作る
                msg = {"type": "brew.auto", "index": st.index}
                if choice:
                    msg["outputs"] = choice
                if not await self._send(msg, made):
                    ok = False
                    logger.info("🎮 《%s》は作れなかった（飛ばす）", self._tableau_name(view, st.index))
        await self._send({"type": "brew.done"},
                         lambda s: bool(((s.get("me") or {}).get("brew") or {}).get("done")) or s.get("phase") != "brew")
        return ok

    def _tableau_name(self, view: dict, i: int) -> str:
        t = (autoplay.me_player(view).get("tableau") or [])
        if 0 <= i < len(t) and t[i]["card"] in self.recipes:
            return self.recipes[t[i]["card"]].name
        return f"{i + 1}枚目"

    # ------------------------------------------------------------------ 記録（会話と評価のため）
    def _record(self, d: Decision, view: dict, ok: bool, total_sec: float) -> None:
        names = [self._tableau_name(view, i) for i in range(len(autoplay.me_player(view).get("tableau") or []))]
        if d.kind == "select":
            what = "＋".join(f"《{self.recipes[c].name}》" + ("（逆向き）" if r else "") for c, r in d.payload)
            move = f"{what}を置いた"
        elif d.kind == "skill":
            move = "・".join(f"{SKILL_JA.get(u['skill'], u['skill'])}（{'・'.join(mat(m) for m in u['materials']) or 'なし'}）"
                            for u in d.payload) or "スキルは使わなかった"
        else:
            sides = autoplay.tableau_sides(view, self.recipes)
            move = "錬成: " + steps_text(d.payload, sides, names) if d.payload else "錬成: 作れるレシピが無かった"
        self.moves.append(move)
        if self.brain.mode == "strategy":
            st = self.brain.strategy
            self.intent = f"作戦 {strategy_text(st)}" + (f"（狙い: {st.note}）" if st.note else "")
        elif d.reason:
            self.intent = d.reason
        s = self.stats.setdefault(d.kind, {"n": 0, "llm": 0, "fallback": 0, "think": [], "total": [], "fail": 0})
        s["n"] += 1
        s["llm"] += d.source.startswith("llm")
        s["fallback"] += d.source.startswith("fallback")
        s["fail"] += not ok
        if d.latency:
            s["think"].append(d.latency)
        s["total"].append(total_sec)
        logger.info("🎮 R%s %s: %s ｜ %s（考える %.1f秒・全体 %.1f秒）%s%s", view.get("round"),
                    PHASE_JA.get(d.kind, "作戦"), move, d.source, d.latency, total_sec,
                    f" 狙い: {d.reason}" if d.reason else "", "" if ok else " ⚠ 一部通らず")

    def _log_stats(self) -> None:
        parts = []
        for kind, s in self.stats.items():
            think = f"考える中央 {statistics.median(s['think']):.1f}秒・最大 {max(s['think']):.1f}秒" if s["think"] else "考える 0秒"
            parts.append(f"{PHASE_JA.get(kind, kind)} {s['n']}回（LLM {s['llm']}・コード代打 {s['fallback']}・"
                         f"通らず {s['fail']}・{think}・全体最大 {max(s['total']):.1f}秒）")
        if parts:
            logger.info("🎮 判断の記録（遊ばせ方 %s）: %s", self.brain.mode, " / ".join(parts))

    # ------------------------------------------------------------------ 会話に渡す事実
    def voice_block(self) -> Optional[str]:
        """応答AI の system に入れる `# Alchemy`（idle なら None＝ルールを入れない）。"""
        if self.state == IDLE:
            return None
        brief = rules_brief(self.rules_d)
        if self.state == PREPARED:
            lines = ["# Alchemy（ユーザが作ったゲーム・今は遊んでいない）", brief]
            if self.last_result:
                lines.append(f"さっきの対戦: {self.last_result}")
            lines.append(f"一緒に遊ぶ時は、合言葉「{self.s.keyword}」の部屋で対戦する。遊ぶかどうかはユーザが決める"
                         "（こちらから何度も誘わない）。")
            return "\n".join(lines)
        if self.state == ROOM:
            lines = [f"# Alchemy（合言葉「{self.s.keyword}」の部屋で、対戦が始まるのを待っている）",
                     f"部屋のメンバー: {self._members_text()}。全員が準備OKを押すと始まる。"]
            if self._wait_text():
                lines.append(self._wait_text())
            if self.last_result:
                lines.append(f"さっきの対戦: {self.last_result}（もう一回遊ぶかはユーザ次第）")
            lines += ["ルール: " + brief, _FACTS_NOTE]
            return "\n".join(lines)
        return "\n".join(["# Alchemy（今ユーザたちと対戦中・イブも1人のプレイヤーとして自分で打っている）",
                          "ルール: " + brief, self._play_facts(full=True), _FACTS_NOTE])

    def decider_block(self) -> Optional[str]:
        """発話判定に入れる短い事実（部屋・対戦中だけ・見出しと扱いの注記つき）。"""
        if self.state not in (ROOM, PLAYING):
            return None
        if self.state == PLAYING and self.view:
            lines = ["# 今イブが遊んでいるゲーム（Alchemy・ユーザと対戦中）", self._play_facts(full=False)]
        else:
            lines = [f"# Alchemy（合言葉「{self.s.keyword}」の部屋で対戦を待っている）",
                     f"部屋のメンバー: {self._members_text()}"]
            if self._wait_text():
                lines.append(self._wait_text())
            if self.last_result:
                lines.append(f"さっきの対戦: {self.last_result}")
        if self.last_event and self._now() - self.last_event[1] <= _REACT_FRESH_SEC:
            lines.append(f"直前の大きな出来事（{int(self._now() - self.last_event[1])}秒前）: {self.last_event[0]}")
        # 反応の頻度（ユーザ指定「毎行動うるさく反応するのはうざいから、ゲームが大きく動いた瞬間などを話しかけられると良い」）。
        # 大きく動いた時だけ発話判定を起こしている（`events.py`）。話すかどうかはここで判断する
        lines.append("（ゲームが大きく動いた直後（点が入った・勝ち負けが決まった・対戦が始まった・相手が部屋に来た）は、"
                     "ひと言反応してよい。毎回の手番や細かい動きにはいちいち反応しない。"
                     "相手がゲームと関係ない話をしていたら、そちらを優先する）")
        return "\n".join(lines)

    def _wait_text(self) -> Optional[str]:
        sec = self.waiting_sec()
        if sec is None:
            return None
        total = self._now() - (self._room_since or self._wait_since)
        asked = f"・イブは来るかを{self._nudges}回確かめた" if self._nudges else ""
        return f"イブだけが部屋で待って計{int(total // 60)}分{int(total % 60)}秒（ユーザはまだ部屋に入っていない{asked}）"

    def screen_note(self) -> Optional[str]:
        """画面認識への断り（イブのゲーム画面を見ている間だけ）。"""
        if not self._screen_on:
            return None
        return ("この画像はユーザのPC画面ではなく、イブ（AI・あなたが目の役をしている相手）が自分で遊んでいる"
                "ブラウザゲーム「Alchemy」の画面（イブの席から見た画面）。手札・素材箱・工房は「イブの」もので、"
                "ユーザの操作として書かない。")

    def _play_facts(self, full: bool) -> str:
        v = self.view or {}
        players = v.get("players") or []
        me_seat = (v.get("me") or {}).get("seat", self.seat)
        n = len(players)
        lines = [f"いま: {v.get('round', '?')}ラウンド目（最大{self.rules.max_rounds}）・"
                 f"{PHASE_JA.get(v.get('phase', ''), v.get('phase', ''))}の段階"]
        lines.append("参加者: " + self._participants_text())
        lines.append("得点: " + " / ".join(
            f"{'イブ' if p.get('seat') == me_seat else p.get('name', '?')} {p.get('points', 0)}点" for p in players)
            + f"（{self.rules.win_points}点で勝ち）")
        if full and me_seat is not None and me_seat < n:
            me = players[me_seat]
            tgt = players[(me_seat - 1) % n] if n >= 2 else {}
            lines.append("イブの素材箱: " + counts_text(me.get("box") or {}, self.rules.materials,
                                                    capacity=self.rules.capacity))
            if tgt:
                lines.append(f"イブの送り先 {tgt.get('name', '?')} の素材箱: "
                             + counts_text(tgt.get("box") or {}, self.rules.materials, capacity=self.rules.capacity))
            if n > 2:
                src = players[(me_seat + 1) % n]
                lines.append(f"イブに送ってくるのは {src.get('name', '?')}")
            hand = (v.get("me") or {}).get("hand") or []
            if hand:
                lines.append("イブの手札（イブにしか見えない）: " + " / ".join(
                    recipe_text(self._cat(c), number=False, emblems=False) for c in hand if c in self.recipes))
            if self.moves:
                lines.append("このラウンドのイブの手: " + " / ".join(self.moves))
            if self.intent:
                lines.append(f"イブの狙い: {self.intent}")
        if self.history:
            lines.append("最近の出来事:\n" + "\n".join(f"- {h}" for h in list(self.history)[-(6 if full else 3):]))
        return "\n".join(lines)

    def _cat(self, card: str) -> dict:
        r = self.recipes[card]
        return {"id": r.id, "name": r.name, "number": r.number, "rotatable": r.rotatable, "skills": list(r.skills),
                "inputs": [list(s) for s in r.inputs], "outputs": [list(s) for s in r.outputs]}


_FACTS_NOTE = ("（ここに書いた数・出来事はゲームのサーバーから届いた正確な記録。画面の実況は見た目の補足で、"
               "数や出来事が食い違ったらこの記録を信じる。書いていないことは作らない。"
               "手番ごとの細かい動きをいちいち実況しない。相手がゲームと関係ない話をしたら、そのまま普通に会話する）")


def _pending_total(view: dict) -> int:
    return sum(int(v) for v in ((view.get("me") or {}).get("skill_pending") or {}).values())
