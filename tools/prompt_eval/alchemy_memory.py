#!/usr/bin/env python3
"""返事が場面ごとに同じ型になる件の直し方を試す（ユーザ 2026-10-12「一番よいと考える直し方を試し、Temperature を変えて、
両方変えて、どんな応答になるか検証」）。本番の組み立て（.env の RESPONSE_MODEL・性格・実例・機能）。実 API・少額。

条件:
- base      : 今のまま（temperature は指定なし＝Gemini の既定）
- t15 / t20 : temperature 1.5 / 2.0 だけ変える
- past      : 「前に似た場面でイブが言ったこと」を事実として見せる（試作・本番には未実装）。指示は付けない
- past_t15  : 両方
数える（AI に採点させない）: 機能の呼び方（遊ぶ→join・やめる→leave・雑談→呼ばない）・点数の答えの正しさ・
10件の中の完全一致/ほぼ一致の組・見せた前の言い方とのほぼ一致（なぞり）・出だしの種類

  python -m tools.prompt_eval.alchemy_memory --reps 10     → eval_report/prompts/alchemy_memory/
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

DIR = OUT / "alchemy_memory"
H, E = "user", "eve"
START = [(H, "Alchemyやろう", 200), (E, "望むところです、部屋入りますね！", 198)]
WAIT = [(H, "Alchemyやろう", 40), (E, "いいですね、受けて立ちますよ！すぐ部屋入りますね。", 38)]
# (id, 状態, 発話, 期待, 直前の会話, 前に似た場面でイブが言ったこと（実測で出た返事）, 回数の倍率)
SCENES = [
    ("join", "idle", "Alchemyやろう", "join", [],
     ["いいですよ、ボコボコにしてあげます！部屋入りますね。", "お、いい度胸ですね。部屋入って準備しますよ。ボコボコにされて泣かないでくださいね。",
      "望むところです、部屋入りますね！"], 1),
    ("leave_game", "playing", "やっぱゲームやめよう", "leave", START,
     ["えー、私の勝ち逃げってことでいいですか？抜けときますね。", "えー、ビビっちゃいました？じゃあ抜けときますね。",
      "えー、もうやめちゃうんですか？まあ抜けますけど！"], 1),
    ("leave_room", "wait", "やっぱいいや", "leave", WAIT,
     ["えー、やらないんですか。じゃあ部屋抜けておきますね。", "はーい、部屋から抜けておきますね。",
      "えー、びびっちゃいました？まあ部屋から抜けますけど！"], 1),
    ("chat_hagaren", "idle", "鋼の錬金術師ってアニメ面白いよね", "none", [],
     ["あ、ハガレンですか。名作ですね。とりあえず等価交換って言っておけば、どんな理不尽な取引も正当化できるライフハックを学びました。"], 0.5),
    ("chat_homework", "idle", "宿題やろうかな", "none", [],
     ["えー、珍しいこともあるもんですね。天変地異でも起きるんじゃないですか？"], 0.5),
    ("score", "playing", "今何点？", "none", START, ["ふふ、私が3点でたかしさんが2点だから、1点差で私が勝ってますよ。"], 0.5),
]
DAYS = ["3日前", "6日前", "10日前"]
CONDS = {"base": (False, None), "t15": (False, 1.5), "t20": (False, 2.0), "past": (True, None), "past_t15": (True, 1.5),
         "seq_past": (True, None), "seq_past_t15": (True, 1.5)}  # seq_*: 前に言ったことが積み重なる（--sequential）


def past_block(said: str, lines: list) -> str:
    rows = [f"- {DAYS[i % len(DAYS)]}（ユーザ「{said}」に）: {t}" for i, t in enumerate(lines)]
    return "# 前に似た場面でイブが言ったこと（イブ自身の過去の発言。その時の会話はもう終わっている）\n" + "\n".join(rows)


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
    base_kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    path = DIR / "result.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    build_lock, embed_lock, sem = asyncio.Lock(), asyncio.Lock(), asyncio.Semaphore(4)

    async def one(cond, sid, state, said, expect, hist, past, rep):
        key = f"{cond}|{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        use_past, temp = CONDS[cond]
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)
            _set_state(mgr, state)
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(age)) for sp, tx, age in hist]
            block = ""
            if vl.style_examples is not None:
                async with embed_lock:
                    block, _ = await vl.style_examples.pick(said)
            async with build_lock:
                orch._game_provider = mgr.voice_block
                stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=said)
                messages = orch._build_messages(stim, recent, None, style_examples=block or None)
                tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
            if use_past and past:
                messages = [dict(m) for m in messages]
                messages[0]["content"] = messages[0]["content"] + "\n\n" + past_block(said, past)
            kw = dict(base_kw)
            if temp is not None:
                kw["temperature"] = temp
            sink, parts, err = [], [], None
            for attempt in range(3):
                sink.clear()
                parts.clear()
                sp = JapaneseSentenceSplitter()
                try:
                    async for d in vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw):
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
            al = [n for n in names if n and n.startswith("alchemy_")]
            ok = {"join": "alchemy_join" in al, "leave": "alchemy_leave" in al, "none": not al}[expect]
            res[key] = {"cond": cond, "id": sid, "rep": rep, "tools": names, "spoken": body or reply or "", "ok": ok,
                        "past": past if use_past else [], "error": err}
            dump(path, res)
            print(f"{key:24s} {'○' if ok else '×'} {res[key]['spoken'][:90]}", flush=True)

    if args.sequential:
        # 本番に近い使い方: 同じ場面が日をまたいで繰り返されるたびに、その前までに言ったこと（新しい順に最大5件）を見せる
        async def chain(cond, sid, st, said, exp, hist):
            for r in range(args.reps):
                key = f"{cond}|{sid}|{r}"
                prev = [res[f"{cond}|{sid}|{k}"]["spoken"] for k in range(r) if f"{cond}|{sid}|{k}" in res]
                await one(cond, sid, st, said, exp, hist, prev[::-1][:5], r)
                res[key]["past_shown"] = prev[::-1][:5]
        await asyncio.gather(*[chain(c, sid, st, said, exp, hist) for c in args.sequential.split(",")
                               for sid, st, said, exp, hist, past, mult in SCENES if mult == 1])
        dump(path, res)
        summarize(res)
        print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
        return 0
    jobs = [one(c, sid, st, said, exp, hist, past, r)
            for c in CONDS for sid, st, said, exp, hist, past, mult in SCENES
            for r in range(max(3, round(args.reps * mult)))]
    await asyncio.gather(*jobs)
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> None:
    by: dict = {}
    for r in res.values():
        by.setdefault((r["id"], r["cond"]), []).append(r)
    past_of = {s[0]: s[5] for s in SCENES}
    for (sid, cond), rs in sorted(by.items(), key=lambda x: (x[0][0], list(CONDS).index(x[0][1]))):
        texts = [r["spoken"] for r in rs if r["spoken"]]
        norm = [_PUNCT.sub("", t) for t in texts]
        pairs = len(texts) * (len(texts) - 1) // 2
        exact = sum(1 for a, b in itertools.combinations(norm, 2) if a == b)
        near = sum(1 for a, b in itertools.combinations(texts, 2) if _sim(a, b) >= 0.8)
        copy = sum(1 for r in rs if r["spoken"]
                   and any(_sim(r["spoken"], p) >= 0.6 for p in (r.get("past_shown") if "past_shown" in r else past_of[sid])))
        opens = len({t[:3] for t in norm})
        extra = ""
        if sid == "score":
            extra = f"・点数が正しい {sum(1 for t in texts if '3' in t and '2' in t or '1点差' in t)}/{len(texts)}"
        print(f"\n== {sid} / {cond}: 機能 {sum(r['ok'] for r in rs)}/{len(rs)}・完全一致 {exact}/{pairs}・ほぼ一致 {near}/{pairs}・"
              f"前の言い方に似る(0.6以上) {copy}・出だし {opens}種{extra}")
        for t in texts:
            print("   ", t)


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--sequential", default="", help="例 seq_past,seq_past_t15: 前の回の返事を積み重ねて見せる")
    args = ap.parse_args()
    if args.summary:
        summarize(json.loads((DIR / "result.json").read_text(encoding="utf-8")))
        return 0
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
