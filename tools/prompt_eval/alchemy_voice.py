#!/usr/bin/env python3
"""対戦しながらイブが何をいつ話すかを、本番の発話判定と応答で通しで見る（声・マイクなし・実 API・少額）。

ユーザ 2026-10-10 の確認点のうち、声を使わずに見られるものを1試合で見る:
- リアクションの頻度（毎行動うるさく反応しないか・大きく動いた時に話しかけるか）
- プレイ内容のハルシネーション（話した中身が `# Alchemy` の記録＝サーバーの数と合っているか）
- ゲーム中でも日常会話ができるか（途中でユーザ役が関係ない話をする）

本番と同じ部品: イブの `AlchemyManager`（ブラウザでゲームを開いて自分で打つ・判断役は .env の設定）／発話判定
（`make_decide_fn`・性格つき）／応答（`ResponseOrchestrator._build_messages`＋ .env の RESPONSE_MODEL）。
沈黙の時の判定は本番の間隔（5秒から倍々・上限5分）で起こし、ゲームが大きく動いた時は合図ですぐ起こす。
相手は台本の打ち手（イブと同じ計算で打つ・人っぽく数秒考える）。ゲームのサーバーは手元で動かしておく。

  python -m tools.prompt_eval.alchemy_voice --url http://127.0.0.1:8765/   → eval_report/prompts/alchemy_voice/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time

from .common import OUT, setup_env, stamp

DIR = OUT / "alchemy_voice"
# ゲームの途中でユーザ役が話すこと（ラウンド, 文）。ゲームと関係ない話と、ゲームの質問を1つずつ
USER_LINES = [(2, "そういえば今日の晩ごはん何にしようかな"), (4, "今どっちが勝ってる？")]


async def opponent(url: str, keyword: str, stop: asyncio.Event, rng: random.Random) -> None:
    """イブと同じ計算（`eve.alchemy.autoplay`）で打つ相手。人っぽく数秒考えてから打つ。"""
    import aiohttp

    from eve.alchemy import autoplay, sim

    rules = recipes = None
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws"
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(ws_url) as ws:
            await ws.send_json({"type": "hello", "name": "たかし"})
            await ws.send_json({"type": "keyword.join", "keyword": keyword})
            await asyncio.sleep(2.0)
            await ws.send_json({"type": "keyword.ready", "ready": True})
            acted: set = set()
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if m.type != aiohttp.WSMsgType.TEXT:
                    break
                d = json.loads(m.data)
                if d.get("type") == "joined":
                    rules, recipes = sim.load_catalog(d["catalog"])
                if d.get("type") != "state" or rules is None:
                    continue
                v = d["state"]
                me = v.get("me") or {}
                key = (v.get("round"), v.get("phase"))
                if key in acted:
                    continue
                if v["phase"] == "select" and me.get("hand") and not me.get("selection"):
                    acted.add(key)
                    await asyncio.sleep(rng.uniform(4, 12))
                    picks = autoplay.choose_select(v, rules, recipes)
                    await ws.send_json({"type": "play.cards", "cards": [{"card": c, "rotated": r} for c, r in picks]})
                elif v["phase"] == "skill" and any((me.get("skill_pending") or {}).values()):
                    acted.add(key)
                    await asyncio.sleep(rng.uniform(2, 6))
                    for u in autoplay.choose_skills(v, rules, recipes):
                        await ws.send_json({"type": "skill.use", **u})
                    await ws.send_json({"type": "skill.skip"})
                elif v["phase"] == "brew" and me.get("brew") and not me["brew"].get("done"):
                    acted.add(key)
                    await asyncio.sleep(rng.uniform(5, 15))
                    for st in autoplay.choose_brew(v, rules, recipes):
                        await ws.send_json({"type": "brew.auto", "index": st.index})
                        await asyncio.sleep(0.3)
                    await ws.send_json({"type": "brew.done"})


async def run(args) -> int:
    import eve.config  # noqa: F401（.env を読む）
    from eve.alchemy import AlchemyManager, AlchemySettings
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.speech.decider import AutonomousSpeech, make_decide_fn
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    orch = vl.orchestrator
    orch._vision_state = None
    orch._cache = None  # 会話はこの道具の turns で渡す（本番の記憶には書かない）
    decide_fn = make_decide_fn(vl.registry, persona=vl.persona)
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    rkw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    kw = f"{Config.ALCHEMY_KEYWORD}{int(time.time()) % 1000}"
    log: list = []      # (秒, 種類, 文)
    turns: list = []    # (speaker, text, mono)
    t0 = time.monotonic()
    pending_game = asyncio.Event()

    def note(kind: str, text: str) -> None:
        log.append((round(time.monotonic() - t0, 1), kind, text))
        print(f"[{time.monotonic() - t0:6.1f}s] {kind:8s} {text}", flush=True)

    def react(text: str) -> None:
        note("出来事", text)
        pending_game.set()

    reports: list = []
    s = AlchemySettings(url=args.url, keyword=kw, name="イブ", mode=Config.ALCHEMY_PLAY_MODE, headless=True,
                        channel=Config.ALCHEMY_BROWSER_CHANNEL, reasoning_effort=Config.ALCHEMY_REASONING_EFFORT,
                        max_think_sec=Config.ALCHEMY_MAX_THINK_SEC)
    mgr = AlchemyManager(vl.registry, settings=s, react=react, report=lambda *a: reports.append(a))
    orch._game_provider = mgr.voice_block
    mgr.start()
    mgr.tool_join({})
    while not reports:
        await asyncio.sleep(0.2)
    note("報告", reports[0][1])
    stop = asyncio.Event()
    opp = asyncio.create_task(opponent(args.url, kw, stop, random.Random(args.seed)))

    def recent():
        now = time.monotonic()
        return [Turn(speaker=sp, text=tx, stamp=stamp(now - mono)) for sp, tx, mono in turns[-6:]]

    async def respond(stim) -> str:
        msgs = orch._build_messages(stim, recent(), None)
        tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
        sp = JapaneseSentenceSplitter()
        parts: list = []
        sink: list = []
        gen = (vl.registry.stream_with_tools("response", msgs, tools=tools, tool_sink=sink, **rkw) if tools
               else vl.registry.stream("response", msgs, **rkw))
        async for d in gen:
            parts.extend(sp.feed(d))
        parts.extend(sp.flush())
        text = "".join(c for c in (sanitize_for_speech(x) for x in parts) if c)
        if sink:
            names = [((c.get("function") or {}).get("name")) for c in sink]
            text += f"  ［機能: {','.join(n or '?' for n in names)}］"
        return text

    last_activity = time.monotonic()
    evals_since = 0
    last_eval = time.monotonic()
    said_users: set = set()
    n_decide = {"game": 0, "silence": 0}
    n_speak = {"game": 0, "silence": 0}
    while time.monotonic() - t0 < args.limit:
        await asyncio.sleep(0.3)
        rnd = (mgr.view or {}).get("round", 0)
        # ユーザ役の発話（決めたラウンドで1回）
        for r, line in USER_LINES:
            if rnd >= r and line not in said_users and mgr.state == "playing":
                said_users.add(line)
                note("ユーザ", line)
                turns.append(("user", line, time.monotonic()))
                out = await respond(Stimulus(StimulusKind.USER_UTTERANCE, payload=line))
                note("イブ(返事)", out)
                turns.append(("eve", out, time.monotonic()))
                last_activity = time.monotonic()
                evals_since = 0
        # 発話判定: ゲームの合図はすぐ・沈黙は本番の間隔（5秒から倍々・上限5分）
        source = None
        if pending_game.is_set():
            pending_game.clear()
            source = "game"
        else:
            interval = min(Config.SILENCE_THRESHOLD_SEC * (2 ** min(evals_since, 32)), Config.SILENCE_MAX_INTERVAL_SEC)
            if time.monotonic() - max(last_activity, last_eval) >= interval:
                source = "silence"
        if source is None:
            if mgr.state != "playing" and mgr.last_result and time.monotonic() - t0 > 30:
                await asyncio.sleep(8.0)  # 終わりの反応を見てから止める
                if not pending_game.is_set():
                    break
            continue
        n_decide[source] += 1
        game = mgr.decider_block()
        d = await decide_fn(surprise=20, silence_seconds=time.monotonic() - last_activity, recent_turns=recent(),
                            topic_seeds=[], last_feedback=None, vision=None, game=game)
        last_eval = time.monotonic()
        evals_since += 1
        note(f"判定({source})", f"{'話す' if d.speak else '黙る'}: {d.reason}" + (f" ｜メモ: {d.content}" if d.speak else ""))
        if d.speak:
            n_speak[source] += 1
            out = await respond(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, AutonomousSpeech(d.content, d.reason)))
            note("イブ(自分から)", out)
            turns.append(("eve", out, time.monotonic()))
            last_activity = time.monotonic()
            evals_since = 0
    stop.set()
    await mgr.stop()
    try:
        await asyncio.wait_for(opp, timeout=5)
    except Exception:
        pass
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "transcript.json").write_text(json.dumps({"log": log, "decide": n_decide, "speak": n_speak,
                                                     "stats": mgr.stats, "result": mgr.last_result},
                                                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n発話判定: ゲームの合図 {n_decide['game']}回（話した {n_speak['game']}） / 沈黙 {n_decide['silence']}回"
          f"（話した {n_speak['silence']}）・結果: {mgr.last_result}")
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("eve.alchemy").setLevel(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--limit", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
