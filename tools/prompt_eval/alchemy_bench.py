#!/usr/bin/env python3
"""遊ばせ方（action / strategy）の速さと強さを、同じ相手と何試合か打たせて比べる（実 API＝判断役だけ課金）。

ユーザ 2026-10-11「どっちのほうが速度早くて賢い？」への計測。
- イブ: 本番の `AlchemyManager`（判断役・コードの手・締切の守り方は本番のまま・.env の GAME_MODEL と考える深さ）。
  ブラウザの代わりにゲームの通信を直接つなぐ（手の決め方は画面を使わないので同じ）
- 相手: 判断役を呼ばない `AlchemyManager`（＝コードの手だけ。錬成の見積もりはゲーム付属の Bot 以上を突き合わせ済み）。
  すぐ打つ（人の考える時間は入れない＝ラウンドの長さはイブの考える時間と演出で決まる）
- 基準 code: イブもコードの手だけで同じ相手と打つ（判断役の有無だけが違う）
- 数える: 勝ち負け・点・ラウンド数・1試合の時間／1手の考える時間・コードで打ち直した数／
  「えらぶ」「錬成」でコードの見積もりのいちばん良い手と同じ手を選んだ割合と、違う手の時の見積もりの差
  （見積もり＝その手の直後の得点の見込み。先の読みは入らないので、差がある＝悪手、とは限らない）
ゲームのサーバーは手元で動かしておく（FlaskRondo・読むだけ）。

  python -m tools.prompt_eval.alchemy_bench --url http://127.0.0.1:8765/ --games 8 --modes action,strategy,code
      → eval_report/prompts/alchemy_bench/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from collections import Counter

from .common import OUT, setup_env

DIR = OUT / "alchemy_bench"


class WsClient:
    """`AlchemyClient` の口だけ真似る（ブラウザなし・ゲームの通信だけ）。"""

    def __init__(self, url: str, name: str, on_message) -> None:
        self.ws_url = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"
        self.name = name
        self._on_message = on_message
        self._session = None
        self._ws = None
        self._reader = None
        self.closed = False
        self.ws_open = False

    @property
    def alive(self) -> bool:
        return not self.closed and self._ws is not None and not self._ws.closed

    async def start(self, timeout: float = 15.0) -> None:
        import aiohttp

        self._session = aiohttp.ClientSession()
        self._ws = await asyncio.wait_for(self._session.ws_connect(self.ws_url, heartbeat=20), timeout)
        self.ws_open = True
        self._reader = asyncio.create_task(self._read())
        await self.send({"type": "hello", "name": self.name})

    async def _read(self) -> None:
        import aiohttp

        try:
            async for m in self._ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    d = json.loads(m.data)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    try:
                        self._on_message(d)
                    except Exception:
                        logging.getLogger("bench").exception("受信の処理で例外")
        finally:
            self.ws_open = False

    async def send(self, msg: dict) -> bool:
        if not self.alive:
            return False
        await self._ws.send_json(msg)
        return True

    async def join_keyword(self, kw: str) -> bool:
        return await self.send({"type": "keyword.join", "keyword": kw})

    async def set_ready(self, ready: bool = True) -> bool:
        return await self.send({"type": "keyword.ready", "ready": bool(ready)})

    async def back_to_room(self) -> bool:
        return False

    async def leave(self) -> None:
        await self.send({"type": "room.leave"})
        await self.send({"type": "keyword.leave"})

    async def screenshot(self, quality: int = 85):
        return None

    async def close(self) -> None:
        self.closed = True
        try:
            if self._ws is not None:
                await self._ws.close()
            if self._session is not None:
                await self._session.close()
        finally:
            if self._reader is not None:
                self._reader.cancel()


class NoLLM:
    """判断役を呼ばない（呼ばれたらすぐ失敗＝コードの手で打つ）。"""

    async def complete(self, role, messages, **kw):
        raise RuntimeError("コードの手だけ")


def _uses_key(uses: list) -> Counter:
    return Counter((u["skill"], tuple(sorted(u["materials"]))) for u in uses)


def _measure(d, view, rules, recipes) -> dict:
    """コードの見積もり（作戦なし）でいちばん良い手と比べる。"""
    from eve.alchemy import autoplay, sim

    row: dict = {}
    if d.kind == "select":
        opts = autoplay.select_options(view, rules, recipes, sim.NO_STRATEGY)
        if opts:
            chosen = next((o for o in opts if [tuple(p) for p in o.picks] == [tuple(p) for p in d.payload]), None)
            row["same_as_code"] = chosen is not None and [tuple(p) for p in chosen.picks] == [tuple(p) for p in opts[0].picks]
            row["gap"] = round(opts[0].value - chosen.value, 3) if chosen is not None else None
            row["n_cards"] = len(d.payload)
            row["code_n_cards"] = len(opts[0].picks)
    elif d.kind == "brew":
        start = autoplay.brew_start(view)
        if start is not None:
            target = autoplay.brew_target(view)
            sides = autoplay.tableau_sides(view, recipes)
            used = set(((view.get("me") or {}).get("brew") or {}).get("used") or [])
            base = sim.score(start, Counter(start.box), target, rules)
            best = sim.best_plan(rules, sides, used, start, target)
            best_sc = sim.outcome(rules, start, target, best).score if best else base
            ch_sc = sim.outcome(rules, start, target, d.payload).score if d.payload else base
            row["gap"] = round(best_sc - ch_sc, 3)
            row["same_as_code"] = abs(best_sc - ch_sc) < 1e-6
    elif d.kind == "skill":
        code = autoplay.choose_skills(view, rules, recipes, sim.NO_STRATEGY)
        row["same_as_code"] = _uses_key(code) == _uses_key(d.payload)
    return row


async def one_game(idx: int, mode: str, url: str, registry, sem: asyncio.Semaphore, limit: float, settings_kw: dict,
                   out: list) -> None:
    from eve.alchemy import AlchemyManager, AlchemySettings

    async with sem:
        kw = f"bench{int(time.time()) % 100000}{mode[:2]}{idx}"
        rec: dict = {"mode": mode, "game": idx, "keyword": kw, "decisions": [], "round_t": {}, "end": None}
        t0 = time.monotonic()
        eve_mode = "strategy" if mode == "code" else mode
        eve = AlchemyManager(NoLLM() if mode == "code" else registry,
                             settings=AlchemySettings(url=url, keyword=kw, name="イブ", mode=eve_mode, **settings_kw),
                             client_factory=lambda: WsClient(url, "イブ", eve.on_message))
        opp = AlchemyManager(NoLLM(), settings=AlchemySettings(url=url, keyword=kw, name="あいて", mode="strategy"),
                             client_factory=lambda: WsClient(url, "あいて", opp.on_message))
        orig_msg = eve.on_message

        def on_msg(m: dict) -> None:
            orig_msg(m)
            if m.get("type") == "state":
                r = (m.get("state") or {}).get("round")
                if r is not None and r not in rec["round_t"]:
                    rec["round_t"][r] = round(time.monotonic() - t0, 2)
            elif m.get("type") == "events":
                for ev in m.get("events") or []:
                    if ev.get("type") == "end":
                        rec["end"] = ev

        eve.on_message = on_msg  # client_factory の lambda は呼ぶ時に eve.on_message を読む
        orig_record = eve._record

        def record(d, view, ok, total):
            orig_record(d, view, ok, total)
            row = {"round": view.get("round"), "kind": d.kind, "source": d.source, "think": round(d.latency, 2),
                   "total": round(total, 2), "ok": ok, "rank": (d.detail or {}).get("rank"), "reason": d.reason}
            try:
                row.update(_measure(d, view, eve.rules, eve.recipes))
            except Exception as e:  # noqa: BLE001
                row["measure_error"] = f"{type(e).__name__}: {e}"
            rec["decisions"].append(row)

        eve._record = record
        try:
            eve.tool_join({})
            while eve.state not in ("room", "playing") and time.monotonic() - t0 < 30:
                await asyncio.sleep(0.2)
            opp.tool_join({})
            while rec["end"] is None and time.monotonic() - t0 < limit:
                await asyncio.sleep(0.5)
        finally:
            rec["seat"] = eve.seat
            rec["sec"] = round(time.monotonic() - t0, 1)
            rec["stats"] = {k: {kk: vv for kk, vv in v.items()} for k, v in eve.stats.items()}
            rec["strategy_notes"] = None
            for m in (eve, opp):
                try:
                    m.tool_leave({})
                except Exception:
                    pass
            await asyncio.sleep(1.0)
            for m in (eve, opp):
                try:
                    await m.stop()
                except Exception:
                    pass
        end = rec["end"] or {}
        rank = {r.get("seat"): r for r in end.get("ranking") or []}
        me = rank.get(rec["seat"], {})
        other = next((r for s, r in rank.items() if s != rec["seat"]), {})
        rec["result"] = ("win" if rec["seat"] in (end.get("winners") or []) and len(end.get("winners") or []) == 1
                         else "draw" if rec["seat"] in (end.get("winners") or []) else "lose" if end else "timeout")
        rec["points"] = (me.get("points"), other.get("points"))
        rec["rounds"] = end.get("round")
        out.append(rec)
        print(f"[{mode:8s} #{idx}] {rec['result']:7s} {rec['points'][0]}-{rec['points'][1]} "
              f"{rec['rounds']}R {rec['sec']:.0f}秒・手 {len(rec['decisions'])}", flush=True)


def summarize(games: list) -> dict:
    by: dict = {}
    for g in games:
        s = by.setdefault(g["mode"], {"games": 0, "win": 0, "draw": 0, "lose": 0, "timeout": 0, "pts_me": [],
                                      "pts_opp": [], "rounds": [], "sec": [], "sec_per_round": [],
                                      "think": {}, "fallback": 0, "decisions": 0, "same": {}, "gap": {}})
        s["games"] += 1
        s[g["result"]] += 1
        if g["points"][0] is not None:
            s["pts_me"].append(g["points"][0])
            s["pts_opp"].append(g["points"][1])
        if g["rounds"]:
            s["rounds"].append(g["rounds"])
            s["sec_per_round"].append(g["sec"] / g["rounds"])
        s["sec"].append(g["sec"])
        for d in g["decisions"]:
            s["decisions"] += 1
            s["fallback"] += d["source"].startswith("fallback")
            if d["think"]:
                s["think"].setdefault(d["kind"], []).append(d["think"])
            if "same_as_code" in d:
                s["same"].setdefault(d["kind"], []).append(bool(d["same_as_code"]))
            if d.get("gap") is not None:
                s["gap"].setdefault(d["kind"], []).append(d["gap"])

    def med(xs):
        return round(statistics.median(xs), 2) if xs else None

    out = {}
    for mode, s in by.items():
        out[mode] = {
            "試合": s["games"], "勝ち": s["win"], "引き分け": s["draw"], "負け": s["lose"], "打ち切り": s["timeout"],
            "イブの点(平均)": round(statistics.mean(s["pts_me"]), 2) if s["pts_me"] else None,
            "相手の点(平均)": round(statistics.mean(s["pts_opp"]), 2) if s["pts_opp"] else None,
            "ラウンド数(中央)": med(s["rounds"]), "1試合の秒(中央)": med(s["sec"]),
            "1ラウンドの秒(中央)": med(s["sec_per_round"]),
            "考える秒(中央/最大)": {k: [med(v), round(max(v), 1)] for k, v in s["think"].items()},
            "手の数": s["decisions"], "コードで打ち直し": s["fallback"],
            "コードの最良と同じ手": {k: f"{sum(v)}/{len(v)}" for k, v in s["same"].items()},
            "違う手の見積もりの差(平均)": {k: round(statistics.mean([x for x in v if x > 1e-6] or [0]), 3)
                                     for k, v in s["gap"].items()},
            "見積もりで損な手(差>0.5)": {k: sum(1 for x in v if x > 0.5) for k, v in s["gap"].items()},
        }
    return out


async def run(args) -> int:
    import eve.config  # noqa: F401（.env を読む）
    from eve.config import Config
    from eve.model_registry import ModelRegistry
    from eve.usage import UsageMeter, install

    meter = UsageMeter(DIR / "usage.jsonl", tag="alchemy_bench")
    install(meter)
    registry = ModelRegistry()
    print("判断役:", registry.resolve("game"), "考える深さ", Config.ALCHEMY_REASONING_EFFORT,
          "・最大", Config.ALCHEMY_MAX_THINK_SEC, "秒")
    settings_kw = {"headless": True, "reasoning_effort": Config.ALCHEMY_REASONING_EFFORT,
                   "max_think_sec": Config.ALCHEMY_MAX_THINK_SEC}
    sem = asyncio.Semaphore(args.parallel)
    games: list = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    jobs = [one_game(i, m, args.url, registry, sem, args.limit, settings_kw, games)
            for i in range(args.games) for m in modes]
    await asyncio.gather(*jobs)
    summary = summarize(games)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "games.json").write_text(json.dumps(games, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("使用量:\n" + meter.format_summary())
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("eve.alchemy.brain").setLevel(logging.ERROR)  # 相手（コードの手だけ）の「呼び出し失敗」を出さない
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--modes", default="action,strategy,code")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--limit", type=float, default=900.0, help="1試合の打ち切り秒")
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
