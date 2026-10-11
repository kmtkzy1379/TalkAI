r"""イブの Alchemy プレイヤーだけを動かして、相手（台本の打ち手）と1試合通す（声・マイクは使わない）。

本番と同じ `AlchemyManager`（ブラウザでゲームを開き、合言葉の部屋に入って打つ）を使う。会話の代わりに、
大きく動いた時の合図（react）と、応答AI に渡る `# Alchemy` の中身をそのまま表示する。

  # ゲームのサーバーを手元で動かしておく（FlaskRondo のフォルダで）
  .\.venv\Scripts\python.exe -m uvicorn server.app:app --port 8765

  $env:PYTHONIOENCODING="utf-8"
  & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\alchemy_play.py --url http://127.0.0.1:8765/ --llm none
  & ... tools\alchemy_play.py --url http://127.0.0.1:8765/ --llm real --mode action      # 実 API（判断役）

--llm none は判断役を呼ばずにコードの手だけで打つ（API 不要・通信と速さの確認用）。
--llm real は .env の GAME_MODEL（既定 gpt-6-luna）を呼ぶ＝**課金あり**。
相手は「手札の先頭を置く・スキルは使わない・作れるレシピを順に作る」だけの台本（--opponent idle なら何もしない）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _NoLLM:
    """判断役を呼ばない（呼ばれたら失敗を返す＝コードの手で打つ）。"""

    async def complete(self, role, messages, **kw):
        raise RuntimeError("LLM なしの試験")


async def opponent(url: str, keyword: str, name: str, style: str, stop: asyncio.Event, log: list) -> None:
    import aiohttp

    ws_url = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(ws_url) as ws:
            await ws.send_json({"type": "hello", "name": name})
            await ws.send_json({"type": "keyword.join", "keyword": keyword})
            await asyncio.sleep(0.5)
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
                if d.get("type") == "events":
                    for ev in d.get("events") or []:
                        if ev.get("type") == "end":
                            log.append(("end", ev))
                if d.get("type") != "state" or style == "idle":
                    continue
                st = d["state"]
                me = st.get("me") or {}
                key = (st.get("round"), st.get("phase"))
                if st["phase"] == "select" and me.get("hand") and not me.get("selection") and key not in acted:
                    acted.add(key)
                    await asyncio.sleep(1.0)
                    await ws.send_json({"type": "play.cards", "cards": [{"card": me["hand"][0]}]})
                elif st["phase"] == "skill" and any((me.get("skill_pending") or {}).values()) and key not in acted:
                    acted.add(key)
                    await ws.send_json({"type": "skill.skip"})
                elif st["phase"] == "brew" and me.get("brew") and not me["brew"]["done"] and key not in acted:
                    acted.add(key)
                    await asyncio.sleep(1.0)
                    for i, a in (me["brew"].get("analysis") or {}).items():
                        if a.get("status") == "now":
                            await ws.send_json({"type": "brew.auto", "index": int(i)})
                            await asyncio.sleep(0.2)
                    await ws.send_json({"type": "brew.done"})


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--keyword", default="")
    ap.add_argument("--mode", default="strategy", choices=["strategy", "action"])
    ap.add_argument("--llm", default="none", choices=["none", "real"])
    ap.add_argument("--opponent", default="simple", choices=["simple", "idle"])
    ap.add_argument("--show", action="store_true", help="イブのゲーム画面を窓で出す")
    ap.add_argument("--limit", type=float, default=900.0, help="打ち切りの秒数")
    ap.add_argument("--shots", default="", help="イブの画面を保存するフォルダ（--shot-every 秒ごと）")
    ap.add_argument("--shot-every", type=float, default=10.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "LiteLLM", "litellm", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from eve.alchemy import AlchemyManager, AlchemySettings
    from eve.config import Config

    registry = _NoLLM()
    if args.llm == "real":
        from eve.model_registry import ModelRegistry

        registry = ModelRegistry()
    kw = args.keyword or f"{Config.ALCHEMY_KEYWORD}{int(time.time()) % 1000}"  # 手元の試験は毎回別の部屋
    settings = AlchemySettings(url=args.url, keyword=kw, name="イブ", mode=args.mode, headless=not args.show,
                               channel=Config.ALCHEMY_BROWSER_CHANNEL, reasoning_effort=Config.ALCHEMY_REASONING_EFFORT,
                               max_think_sec=Config.ALCHEMY_MAX_THINK_SEC)
    reacts: list = []
    reports: list = []
    t0 = time.monotonic()

    def react(text: str) -> None:
        reacts.append((round(time.monotonic() - t0, 1), text))
        print(f"\n💬 [反応の合図 {time.monotonic() - t0:.1f}s] {text}\n", flush=True)

    def report(name: str, content: str, ok: bool) -> None:
        reports.append((name, content, ok))
        print(f"\n📣 [機能の報告] {name}: {content}\n", flush=True)

    async def screen_hook(feed) -> None:
        print(f"🖥 画面認識の見る先: {'イブのゲーム画面' if feed else 'デスクトップ'}", flush=True)

    mgr = AlchemyManager(registry, settings=settings, react=react, report=report, screen_hook=screen_hook)
    mgr.start()
    print("idle の時の # Alchemy:", mgr.voice_block())
    print(mgr.tool_join({}))
    stop = asyncio.Event()
    log: list = []
    while not reports and time.monotonic() - t0 < 30:
        await asyncio.sleep(0.2)
    opp = asyncio.create_task(opponent(args.url, kw, "テスト相手", args.opponent, stop, log))
    shot_i = 0
    next_shot = time.monotonic()
    printed_block = False
    while time.monotonic() - t0 < args.limit:
        await asyncio.sleep(0.5)
        if args.shots and mgr.client is not None and time.monotonic() >= next_shot:
            os.makedirs(args.shots, exist_ok=True)
            img = await mgr.client.screenshot()
            if img:
                with open(os.path.join(args.shots, f"eve_{shot_i:03d}.jpg"), "wb") as f:
                    f.write(img)
                shot_i += 1
            next_shot = time.monotonic() + args.shot_every
        if not printed_block and mgr.state == "playing" and mgr.view and mgr.view.get("round", 1) >= 3:
            printed_block = True
            print("\n===== 対戦中の # Alchemy（応答AI へ）=====\n" + (mgr.voice_block() or ""))
            print("\n===== 対戦中の発話判定へ =====\n" + (mgr.decider_block() or "") + "\n")
        if log and any(k == "end" for k, _ in log):
            await asyncio.sleep(2.0)
            break
    dur = time.monotonic() - t0
    print("\n===== 結果 =====")
    print(f"かかった時間 {dur:.0f}秒 / 状態 {mgr.state} / 反応の合図 {len(reacts)}回")
    print(f"最後の結果: {mgr.last_result}")
    print("終了後の # Alchemy:\n" + (mgr.voice_block() or ""))
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else [round(x, 2) for x in vv])
                          for kk, vv in v.items()} for k, v in mgr.stats.items()}, ensure_ascii=False))
    print(mgr.tool_leave({}))
    await asyncio.sleep(1.5)
    stop.set()
    await asyncio.wait_for(opp, timeout=5)
    await mgr.stop()
    print("やめた後の状態:", mgr.state, "/ # Alchemy あり" if mgr.voice_block() else "/ # Alchemy なし")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
