#!/usr/bin/env python3
"""発話判定に性格を渡すと、話す/黙るの正しさが崩れないか（gemini_check の12場面・本番の判定モデル）。実 API（少額）。

  python -m tools.prompt_eval.decide_persona   → eval_report/prompts/decide_persona/check.json と一覧

2026-10-05 ユーザ「発話判定にも性格を渡すのも検証」。黙るべき場面（映画・会議・おやすみ・返事待ち等）で
性格が「話す」に倒さないかを見る。性格なしは同じ場面を同じ回数だけ回して並べる
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from .common import OUT, dump, setup_env
from .gemini_check import DECIDE_CASES

DIR = OUT / "decide_persona"
REPS = 3
TAG = os.getenv("SIM_TAG", "")   # 例 SIM_TAG=v2 → check_v2.json・性格ありだけ（手直し後の確認用）


async def run() -> int:
    from eve.clock import Stamp
    from eve.context_assembler import RagChunk, Turn
    from eve.persona import load_persona, persona_for
    from eve.speech.decider import PERSONA_HOW, build_decide_messages, parse_speech_decision
    from eve.voice_loop import VoiceLoop

    def stamp(sec):
        iso = (datetime.now(timezone.utc) - timedelta(seconds=sec)).isoformat()
        return Stamp(iso=iso, mono=time.monotonic() - sec)

    vl = VoiceLoop()
    variants = {"性格なし": "", "性格あり": persona_for(load_persona(), PERSONA_HOW)}
    if TAG:
        variants = {"性格あり": variants["性格あり"]}
    DIR.mkdir(exist_ok=True)
    path = DIR / (f"check_{TAG}.json" if TAG else "check.json")
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    sem = asyncio.Semaphore(6)

    async def one(case, v, rep):
        title, expect, turns, silence, surprise, screen, seeds, prior = case
        key = f"{title}|{v}|{rep}"
        if key in res:
            return
        msgs = build_decide_messages(
            surprise=surprise, silence_seconds=silence,
            recent_turns=[Turn(speaker=s, text=t, stamp=stamp(sec)) for s, t, sec in turns],
            topic_seeds=[RagChunk(text=t, iso=stamp(sec).iso) for t, sec in seeds],
            last_feedback=None, vision=screen or None, active_tasks=[], prior_items=prior or None)
        msgs[0]["content"] += variants[v]
        async with sem:
            try:
                r = await vl.registry.complete("speech_decide", msgs)
                d = parse_speech_decision(r.choices[0].message.content or "")
                res[key] = {"title": title, "v": v, "expect": expect, "speak": d.speak, "reason": d.reason,
                            "content": d.content}
            except Exception as e:  # noqa: BLE001
                res[key] = {"title": title, "v": v, "expect": expect, "error": str(e)[:150]}

    await asyncio.gather(*[one(c, v, r) for c in DECIDE_CASES for v in variants for r in range(REPS)])
    dump(path, res)
    for v in variants:
        # 期待は今の表から（記録した時の期待ではなく）。None＝どちらでもよい場面は数えない
        rows = [(c[1], res.get(f"{c[0]}|{v}|{i}", {})) for c in DECIDE_CASES if c[1] is not None for i in range(REPS)]
        rows = [(x, r) for x, r in rows if "speak" in r]
        ok = sum(1 for x, r in rows if r["speak"] == x)
        print(f"\n## {v}: 正しい {ok}/{len(rows)}（どちらでもよい場面は数えない）")
        for c in DECIDE_CASES:
            rs = [res.get(f"{c[0]}|{v}|{i}", {}) for i in range(REPS)]
            got = "".join("話" if r.get("speak") else ("黙" if "speak" in r else "×") for r in rs)
            mark = "" if c[1] is None or all(r.get("speak") == c[1] for r in rs) else "  ← 期待と違う"
            want = "どちらでもよい" if c[1] is None else ("話す" if c[1] else "黙る")
            said = " / ".join(r["content"] for r in rs if r.get("speak") and r.get("content"))
            print(f"   {c[0]}（期待 {want}）: {got}{mark}" + (f"\n      {said}" if said else ""))
    print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
