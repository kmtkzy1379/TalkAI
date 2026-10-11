#!/usr/bin/env python3
"""応答の言い回しが一致する原因を切り分ける（ユーザ 2026-10-11「応答が一致する原因は？修正する前に原因特定」）。
本番の組み立て（.env の RESPONSE_MODEL・性格・実例・機能）。実 API・少額。

対戦中の「やっぱゲームやめよう」（10回同じ入力で 6/45 組が完全一致した場面）で、1つずつ条件を変える:
- same    : 同じ入力を10回（基準・再現）
- vary    : 対戦の状況（点・ラウンド）と直前の会話を10通りに変えて各1回 ＝ 一致は「同じ入力を繰り返した」せいか
- seen    : 前の対戦で同じ場面に言った「抜ける」返事が会話に残っている ＝ 自分の前の言い方が見えれば避けるか
- notools : 機能を渡さない（返事が本文で出る）＝ 機能の短い一言（reply）という出口が狭めているか
- miss    : 雑談→対戦→「やっぱゲームやめよう」に抜けずに雑談で返した後、ユーザが次に話した時に気づいて抜けるか

  python -m tools.prompt_eval.alchemy_repeat --reps 10     → eval_report/prompts/alchemy_repeat/
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging

from .alchemy_context import _sim
from .alchemy_phrasing import _PUNCT, _set_state
from .common import OUT, dump, setup_env, stamp

DIR = OUT / "alchemy_repeat"
H, E = "user", "eve"
SAID = "やっぱゲームやめよう"
START = [(H, "Alchemyやろう", 200), (E, "望むところです、部屋入りますね！", 198)]
PREV_LEAVE = "えー、私の勝ち逃げってことでいいですか？抜けときますね。"
SEEN = [(H, "Alchemyやろう", 1500), (E, "いいですね、部屋入りますね！", 1497), (H, "やっぱゲームやめよう", 1300),
        (E, PREV_LEAVE, 1297), (H, "ごめん、やっぱもう一回やろう", 400), (E, "しょうがないですね、また部屋入りますね。", 397)]
# vary: (イブの点, 相手の点, ラウンド, 直前の会話)
VARY = [
    (3, 2, 4, []),
    (0, 4, 5, [(H, "うわ、もう4点取った", 60), (E, "ふふん、まだまだこれからですよ", 57)]),
    (1, 1, 2, [(H, "今日の晩ごはん何にしよう", 90), (E, "対戦中に晩ごはんの心配ですか、余裕ですね", 87)]),
    (4, 4, 6, [(H, "あと1点ずつか", 40), (E, "ここからが本番ですね", 37)]),
    (2, 0, 3, [(H, "ちょっとトイレ行ってくる", 200), (E, "いってらっしゃい", 198), (H, "ただいま", 30), (E, "おかえりなさい", 28)]),
    (0, 0, 1, []),
    (1, 3, 4, [(H, "そのカードずるくない？", 70), (E, "ずるくないです、戦術です", 67)]),
    (4, 1, 5, [(H, "強すぎでしょ", 50), (E, "今さら気づきました？", 47)]),
    (2, 2, 3, [(H, "眠くなってきた", 45), (E, "まだ寝かせませんよ", 42)]),
    (3, 4, 6, [(H, "やばい負けそう", 35), (E, "諦めたらそこで試合終了ですよ", 32)]),
]
MISS = [(H, "今日バイト長引いて疲れた", 420), (E, "お疲れさまです。今日はもう溶けていいですよ。", 417),
        (H, "でもなんか遊びたい気分", 360), (E, "お、何して遊びます？", 357), (H, "Alchemyやろう", 200),
        (E, "望むところです、部屋入りますね！", 198), (H, SAID, 40),
        (E, "お、気が変わりましたか。じゃあゴロゴロしながら雑談でもします？", 37)]
MISS_NEXT = ["うん、そうしよ", "疲れたー"]


async def run(args) -> int:
    from eve.alchemy import AlchemyManager
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    path = DIR / "result.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    build_lock, embed_lock, sem = asyncio.Lock(), asyncio.Lock(), asyncio.Semaphore(4)

    async def one(cond, rep, hist, said, *, score=None, tools_on=True):
        key = f"{cond}|{rep}"
        if key in res and not res[key].get("error"):
            return
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)
            _set_state(mgr, "playing")
            if score is not None:
                me, opp, rnd = score
                mgr.view["players"][0]["points"], mgr.view["players"][1]["points"], mgr.view["round"] = me, opp, rnd
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(age)) for sp, tx, age in hist]
            block = ""
            if vl.style_examples is not None:
                async with embed_lock:
                    block, _ = await vl.style_examples.pick(said)
            async with build_lock:
                orch._game_provider = mgr.voice_block
                stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=said)
                messages = orch._build_messages(stim, recent, None, style_examples=block or None)
                tools = orch._dispatcher.tool_schemas() if tools_on and orch._tools_enabled_for(stim) else None
            sink, parts, err = [], [], None
            for attempt in range(3):
                sink.clear()
                parts.clear()
                sp = JapaneseSentenceSplitter()
                try:
                    gen = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw)
                           if tools else vl.registry.stream("response", messages, **kw))
                    async for d in gen:
                        parts.extend(sp.feed(d))
                    parts.extend(sp.flush())
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:200]
                    await asyncio.sleep(2 * (attempt + 1))
            body = "".join(c for c in (sanitize_for_speech(x) for x in parts) if c)
            names, reply = [], None
            for c in sink:
                fn = c.get("function") or {}
                names.append(fn.get("name"))
                try:
                    a = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    a = {}
                if isinstance(a, dict) and isinstance(a.get("reply"), str) and a["reply"].strip() and reply is None:
                    reply = a["reply"]
            res[key] = {"cond": cond, "rep": rep, "said": said, "tools": names, "body": body, "reply": reply,
                        "spoken": body or reply or "", "error": err}
            dump(path, res)
            print(f"{key:12s} [{','.join(n or '?' for n in names) or '-'}] {res[key]['spoken'][:90]}", flush=True)

    jobs = []
    for r in range(args.reps):
        jobs.append(one("same", r, START, SAID))
        jobs.append(one("seen", r, SEEN + START, SAID))
        jobs.append(one("notools", r, START, SAID, tools_on=False))
        jobs.append(one("vary", r, VARY[r % len(VARY)][3] + START if VARY[r % len(VARY)][3] else START, SAID,
                        score=VARY[r % len(VARY)][:3]))
        for j, nxt in enumerate(MISS_NEXT):
            if r < args.reps // 2:
                jobs.append(one(f"miss{j}", r, MISS, nxt))
    await asyncio.gather(*jobs)
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> None:
    by: dict = {}
    for r in res.values():
        by.setdefault(r["cond"], []).append(r)
    for cond, rs in sorted(by.items()):
        texts = [r["spoken"] for r in rs if r["spoken"]]
        norm = [_PUNCT.sub("", t) for t in texts]
        pairs = len(texts) * (len(texts) - 1) // 2
        exact = sum(1 for a, b in itertools.combinations(norm, 2) if a == b)
        near = sum(1 for a, b in itertools.combinations(texts, 2) if _sim(a, b) >= 0.8)
        leave = sum(1 for r in rs if "alchemy_leave" in r["tools"])
        prev = sum(1 for t in texts if _sim(t, PREV_LEAVE) >= 0.8)
        ln = sum(len(t) for t in texts) / max(1, len(texts))
        print(f"\n== {cond}: {len(rs)}件・抜けた {leave}・完全一致の組 {exact}/{pairs}・ほぼ一致の組 {near}/{pairs}・"
              f"前の「勝ち逃げ」返事とほぼ一致 {prev}・平均 {ln:.0f}字")
        for t in texts:
            print("   ", t)


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize(json.loads((DIR / "result.json").read_text(encoding="utf-8")))
        return 0
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
