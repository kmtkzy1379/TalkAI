#!/usr/bin/env python3
"""フィードバックと感情判定が、応答の性格（persona.md）と食い違わないかの確かめ。実 API（少額）。

  python -m tools.prompt_eval.feedback_persona   → eval_report/prompts/feedback_persona/check.json と一覧

2026-10-05 ユーザ「フィードバックにもペルソナ反映しないとちぐはぐになる可能性あるから、フィードバックが応答ペルソナに沿ってるか確認」
- 材料: Gemini＋性格 N8 の実際のやりとり（prod_check の B・Y／model_trial の1往復）
- フィードバック（本番の FEEDBACK_SYSTEM・Sonnet 5.5・感情つき）と感情判定（本番の JUDGE_SYSTEM・ルナ）を、
  今のまま（性格なし）と、system の最後に性格を足した版で出し比べる
"""
from __future__ import annotations

import asyncio
import json
import logging

from .common import OUT, dump, load, setup_env

DIR = OUT / "feedback_persona"


def persona_block() -> str:
    from eve.persona import load_persona

    body = load_persona().split("\n", 1)[1]
    return ("\n\n# イブの性格（感情や次の予測は、この性格のイブとして考える。"
            "ただし summary は事実だけ。イブの冗談や作り話は、冗談として書く）\n" + body)


def spans():
    pc = load(OUT / "prod_check" / "check.json")
    mt = load(OUT / "model_trial" / "trial.json")
    out = []
    for k, r in pc.items():
        if r["item"] in ("B", "Y"):
            out.append((k, r["said"], r["spoken"]))
    for said, r in mt["single"].items():
        g = r.get("gemini") or {}
        if g.get("reply"):
            out.append((f"MT:{said[:8]}", said, g["reply"]))
    return out


async def run() -> int:
    from eve.clock import Stamp
    from eve.context_assembler import Turn
    from eve.emotion import make_judge_fn
    from eve.emotion import JUDGE_SYSTEM  # noqa: F401  （文面は本番のものを使う）
    from eve.feedback.parser import parse_feedback
    from eve.feedback.prompts import build_messages
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    pb = persona_block()
    DIR.mkdir(exist_ok=True)
    path = DIR / "check.json"
    res = load(path) if path.exists() else {}
    sem = asyncio.Semaphore(4)

    judge = make_judge_fn(vl.registry)

    async def judge_with(said, with_persona):
        if not with_persona:
            return await judge([], said, None)
        # 本番の判定関数の system に性格を足す（同じ呼び方・同じ解析）
        import eve.emotion as em
        orig = em.JUDGE_SYSTEM
        em.JUDGE_SYSTEM = orig + pb.replace("{", "{{").replace("}", "}}")
        try:
            return await make_judge_fn(vl.registry)([], said, None)
        finally:
            em.JUDGE_SYSTEM = orig

    async def one(key, said, reply):
        if key in res:
            return
        async with sem:
            st = Stamp.now()
            turns = [Turn(speaker="user", text=said, stamp=st), Turn(speaker="eve", text=reply, stamp=st)]
            row = {"said": said, "reply": reply}
            for tag, extra in (("now", ""), ("persona", pb)):
                msgs = build_messages(turns, None, with_emotion=True, current_emotion=None)
                msgs[0]["content"] += extra
                try:
                    r = await vl.registry.complete("feedback", msgs)
                    text = r.choices[0].message.content or ""
                    fb = parse_feedback(text)
                    row[f"fb_{tag}"] = {"values": fb.emotion_values, "emotion": fb.emotions, "summary": fb.summary,
                                        "pred": fb.next_prediction}
                except Exception as e:  # noqa: BLE001
                    row[f"fb_{tag}"] = {"error": str(e)[:150]}
            # 感情判定は判定関数を差し替える都合で順番に
            res[key] = row

    items = spans()
    await asyncio.gather(*[one(k, s, r) for k, s, r in items])
    for k, s, _ in items:
        if "judge_now" not in res[k]:
            res[k]["judge_now"] = await judge_with(s, False)
            res[k]["judge_persona"] = await judge_with(s, True)
    dump(path, res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    for k, r in res.items():
        a, b = r.get("fb_now", {}), r.get("fb_persona", {})
        print(f"\n## {r['said'][:30]}\n   イブ: {r['reply'][:90]}")
        print(f"   感情判定 今={r.get('judge_now')} 性格あり={r.get('judge_persona')}")
        print(f"   FB 今   [{a.get('label')}] {a.get('emotion','')[:30]} ｜ 要約: {a.get('summary','')[:80]}")
        print(f"   FB 性格 [{b.get('label')}] {b.get('emotion','')[:30]} ｜ 要約: {b.get('summary','')[:80]}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    setup_env({"EMOTION_ENABLED": "1"})
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
