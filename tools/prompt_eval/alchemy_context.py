#!/usr/bin/env python3
"""Alchemy の機能を呼ぶ時の返事（部屋に入る・抜ける）が、会話の流れ（履歴）でどう変わるか・同じ言い回しの繰り返し
（完全一致・ほぼ一致）が出るかを測る（本番の設定＝.env の RESPONSE_MODEL・性格・実例・機能。実 API・少額）。

ユーザ 2026-10-11「口癖に関しては完全一致やほとんど一致しなければいい。どんな応答するのか10件づつぐらい報告して。
多分会話履歴や文脈でも変わるため実際の会話履歴や想定会話履歴などでも変わるか検証」。

文脈（直前の会話）: none＝無し / real＝本番の会話履歴の最後の10ターン（読むだけ・今の会話として渡す）/
again＝さっきも Alchemy を遊んだ（前の「部屋に入る」返事が会話に残っている）/ chat＝雑談の流れ
数える（AI に採点させない）: 完全一致の組・ほぼ一致の組（文字の2文字組の重なり 0.8 以上）・
again では会話に残っている前の返事とのほぼ一致

  python -m tools.prompt_eval.alchemy_context --reps 10     → eval_report/prompts/alchemy_context/
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import time

from .alchemy_phrasing import _PUNCT, _set_state
from .common import OUT, REPO, dump, setup_env, stamp

DIR = OUT / "alchemy_context"
H, E = "user", "eve"
PREV_JOIN = "いいですよ、ボコボコにしてあげます！部屋入りますね。"
CONTEXTS = {
    "none": [],
    "again": [(H, "Alchemyやろう", 900), (E, PREV_JOIN, 897), (E, "勝っちゃいました。弱すぎません？", 600),
              (H, "くそー", 590), (E, "ふふん、いつでも挑んでいいですよ", 585), (H, "ちょっと休憩", 400),
              (E, "はーい、ごゆっくり", 397)],
    "chat": [(H, "今日バイト長引いて疲れた", 120), (E, "お疲れさまです。今日はもう溶けていいですよ。", 117),
             (H, "でもなんか遊びたい気分", 60), (E, "お、何して遊びます？", 57)],
}
WAIT = [(H, "Alchemyやろう", 40), (E, "いいですね、受けて立ちますよ！すぐ部屋入りますね。", 38)]
SCENES = [  # (id, 状態, 発話, 期待する機能, この場面の直前の会話)
    ("join", "idle", "Alchemyやろう", "alchemy_join", []),
    ("leave_room", "wait", "やっぱいいや", "alchemy_leave", WAIT),
    ("leave_game", "playing", "やっぱゲームやめよう", "alchemy_leave", [(H, "Alchemyやろう", 200), (E, "望むところです、部屋入りますね！", 198)]),
]


def _real_history(n: int = 10) -> list:
    """本番の会話履歴の最後の n ターン（中身はそのまま・時刻は今の会話として 10分前〜1分前に並べ直す）。"""
    src = REPO / "conversation_history.jsonl"  # 本番の会話履歴（読むだけ・setup_env は書き込み先を一時フォルダへ向ける）
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()][-n:]
    return [("user" if r.get("speaker") == "user" else "eve", r.get("text", ""), 600 - i * 55)
            for i, r in enumerate(rows)]


def _sim(a: str, b: str) -> float:
    a, b = _PUNCT.sub("", a), _PUNCT.sub("", b)
    ga = {a[i:i + 2] for i in range(len(a) - 1)}
    gb = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(ga & gb) / max(1, len(ga | gb))


async def run(args) -> int:
    from eve.alchemy import AlchemyManager
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.voice_loop import VoiceLoop

    CONTEXTS["real"] = _real_history()
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

    async def one(sid, state, said, expect, scene_hist, ctx, rep):
        key = f"{sid}|{ctx}|{rep}"
        if key in res and not res[key].get("error"):
            return
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)
            _set_state(mgr, state)
            shift = 300 if scene_hist else 0  # 文脈の会話は場面の会話より前
            hist = [(sp, tx, age + shift) for sp, tx, age in CONTEXTS[ctx]] + list(scene_hist)
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
            res[key] = {"id": sid, "ctx": ctx, "rep": rep, "said": said, "tools": names, "spoken": body or reply or "",
                        "ok": expect in names, "error": err}
            dump(path, res)
            print(f"{key:22s} {'○' if res[key]['ok'] else '×'} {res[key]['spoken'][:100]}", flush=True)

    await asyncio.gather(*[one(sid, st, said, exp, sh, ctx, rep)
                           for sid, st, said, exp, sh in SCENES for ctx in ("none", "real", "again", "chat")
                           for rep in range(args.reps)])
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> None:
    by: dict = {}
    for r in res.values():
        by.setdefault((r["id"], r["ctx"]), []).append(r)
    for (sid, ctx), rs in sorted(by.items()):
        texts = [r["spoken"] for r in rs if r["spoken"]]
        norm = [_PUNCT.sub("", t) for t in texts]
        exact = sum(1 for a, b in itertools.combinations(norm, 2) if a == b)
        near = sum(1 for a, b in itertools.combinations(texts, 2) if _sim(a, b) >= 0.8)
        pairs = len(texts) * (len(texts) - 1) // 2
        prev = sum(1 for t in texts if _sim(t, PREV_JOIN) >= 0.8) if ctx == "again" else None
        print(f"\n== {sid} / {ctx}: 機能 {sum(r['ok'] for r in rs)}/{len(rs)}・完全一致の組 {exact}/{pairs}・"
              f"ほぼ一致の組 {near}/{pairs}" + (f"・会話に残る前の返事とほぼ一致 {prev}" if prev is not None else ""))
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
