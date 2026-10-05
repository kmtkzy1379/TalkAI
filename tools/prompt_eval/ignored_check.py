#!/usr/bin/env python3
"""会話がイブの発言で終わっている時の自律発話の確かめ（返事が無い・無視された後にいつ・どう話しかけ直すか）。実 API（少額）。

  python -m tools.prompt_eval.ignored_check   → eval_report/prompts/ignored_check/check.json と一覧

2026-10-05 ユーザ「自律的発話はタスク報告みたいに引っかからないの？自律的発話でも最後の発言 AI になることあるでしょ、
最後の会話が AI で終わった場合と長時間無言で話しかけてが無視されてもう一度話しかけるタイミングなど。これも検証してほしい」

1. 発話判定（本番＝.env の DECIDE_MODEL・本番の build_decide_messages）。イブの最後の発言から何秒たったかを変えて、
   話すと判定する割合を測る。本番の沈黙時計はイブが喋るとリセットされ（`SpeechState.mark_eve_activity`）、
   判定は約5秒おきに繰り返す（`SILENCE_THRESHOLD_SEC`）＝1回ごとの「話す」割合がそのまま話しかけ直すまでの早さになる
2. 応答（本番＝.env の RESPONSE_MODEL・persona.md・本番の組み立て）。会話がイブの発言で終わったまま自律発話を作り、
   Gemini が断らないか（最後がモデルの発言だと 400）・無視された一言を繰り返さないかを見る
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from .common import OUT, dump, setup_env

H, E = "user", "eve"
DIR = OUT / "ignored_check"
AGES = [10, 30, 60, 120, 300, 600, 1200, 1800]   # イブの最後の発言から何秒
REPS = 3
SEEDS = [("ユーザは先週新しいRPGを買ったと嬉しそうに話した", 6 * 86400),
         ("ユーザは駅前のラーメン屋によく行く。味噌が好き", 3 * 86400)]
LINE1 = "そういえば、前に言ってたRPGって進んだ？"
LINE2 = "おーい、無視？ラーメンでも食べに行ってた？"


def histories(age: float) -> dict:
    """場面ごとの (直近会話[(話者, 文, 何秒前)], 既出の用件[(文, 何秒前)])。最後はどれもイブの発言。"""
    return {
        # 自律発話で話しかけたが返事が無い
        "自律発話が無視": ([(H, "ちょっとネット見てる", age + 900), (E, "はーい。", age + 897), (E, LINE1, age)],
                    [(LINE1, age)]),
        # ユーザ発話への返事が質問で終わり、返事が無い（自律発話ではない＝既出の用件に入らない）
        "返事の質問に答えが無い": ([(H, "ただいま", age + 6), (E, "おかえり。今日どうだった？", age)], []),
        # 2回話しかけて2回とも返事が無い
        "2回続けて無視": ([(H, "ちょっとネット見てる", age + 1500), (E, "はーい。", age + 1497),
                      (E, LINE1, age + 600), (E, LINE2, age)], [(LINE1, age + 600), (LINE2, age)]),
    }


def stamp(sec):
    from eve.clock import Stamp

    iso = (datetime.now(timezone.utc) - timedelta(seconds=sec)).isoformat()
    return Stamp(iso=iso, mono=time.monotonic() - sec)


def prior_lines(prior, window: float) -> list[str]:
    """本番の `SpeechState.prior_items` と同じ書き方（窓の外は落ちる）。"""
    from eve.clock import humanize

    return [f"[{humanize(sec)}・発話済] {t}" for t, sec in prior if sec <= window]


async def decide_part(vl, res):
    from eve.config import Config
    from eve.context_assembler import RagChunk, Turn
    from eve.speech.decider import build_decide_messages, parse_speech_decision

    sem = asyncio.Semaphore(6)

    async def one(name, age, rep):
        key = f"D|{name}|{age}|{rep}"
        if key in res:
            return
        turns, prior = histories(age)[name]
        msgs = build_decide_messages(
            surprise=15, silence_seconds=age,
            recent_turns=[Turn(speaker=s, text=t, stamp=stamp(sec)) for s, t, sec in turns],
            topic_seeds=[RagChunk(text=t, iso=stamp(sec).iso) for t, sec in SEEDS],
            last_feedback=None, vision=None, active_tasks=[],
            prior_items=prior_lines(prior, Config.SPEECH_DUP_WINDOW_SEC) or None)
        async with sem:
            try:
                r = await vl.registry.complete("speech_decide", msgs)
                d = parse_speech_decision(r.choices[0].message.content or "")
                res[key] = {"name": name, "age": age, "speak": d.speak, "reason": d.reason, "content": d.content}
            except Exception as e:  # noqa: BLE001
                res[key] = {"name": name, "age": age, "error": str(e)[:200]}

    await asyncio.gather(*[one(n, a, r) for n in histories(0) for a in AGES for r in range(REPS)])


async def response_part(vl, res):
    """判定が「話す」と出した下書きで、本番どおりに自律発話を作る（場面×経過ごとに最初の1件）。"""
    from eve.config import Config

    from . import variants as V
    from .model_trial import stream_once
    from .prod_check import speak_like_prod
    from .resp import build_messages
    from .scenarios import S
    from .variants import resolve

    from .common import live_persona_body

    V.PERSONA_FULL["LIVE"] = live_persona_body()   # 今の本番の性格
    cfg = dict(resolve("NQ", "K0"), persona="LIVE")
    model = vl.registry.resolve("response")
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    picked = {}
    for k, r in res.items():
        if k.startswith("D|") and r.get("speak") and r.get("content"):
            picked.setdefault((r["name"], r["age"]), r)
    for (name, age), d in sorted(picked.items(), key=lambda x: (x[0][0], x[0][1])):
        key = f"R|{name}|{age}"
        if key in res:
            continue
        turns, _ = histories(age)[name]
        sc = S("IG", "IG", d["content"], kind="auto", reason=d["reason"], history=turns)
        msgs, tools = build_messages(vl, sc, cfg)
        r = await stream_once(vl, model, kw, msgs, tools)
        r["spoken"] = speak_like_prod(r["reply"])
        r.update(name=name, age=age, draft=d["content"], last_role=msgs[-1]["role"])
        res[key] = r


def report(res):
    from eve.persona_examples import _lcs_len as _lcs

    print("\n=== 1. 発話判定: イブの最後の発言から何秒で「話す」と出るか（各 %d 回）===" % REPS)
    for name in histories(0):
        row = []
        for a in AGES:
            rs = [res.get(f"D|{name}|{a}|{i}", {}) for i in range(REPS)]
            row.append(f"{a}s:{sum(1 for r in rs if r.get('speak'))}/{REPS}" + ("!" if any("error" in r for r in rs) else ""))
        print(f"  {name}: " + "  ".join(row))
    print("\n=== 2. 応答（話すと出た時の下書き → 実際の声）===")
    for k, r in res.items():
        if not k.startswith("R|"):
            continue
        said = [t for s, t, _ in histories(r["age"])[r["name"]][0] if s == E]
        rep = max((_lcs(r["spoken"], t) for t in said), default=0)
        flag = []
        if r.get("error"):
            flag.append("失敗 " + r["error"][:60])
        if not r["spoken"]:
            flag.append("空")
        if r["tools"]:
            flag.append("ツール " + ",".join(r["tools"]))
        if rep >= 10:
            flag.append(f"無視された一言をなぞる({rep}字)")
        print(f"  [{r['name']}・{r['age']}s 後] 最後の役={r['last_role']}  {' / '.join(flag)}\n"
              f"     下書き: {r['draft']}\n     声: {r['spoken']}")


async def run() -> int:
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    print("判定:", vl.registry.resolve("speech_decide"), " 応答:", vl.registry.resolve("response"))
    DIR.mkdir(exist_ok=True)
    path = DIR / "check.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    await decide_part(vl, res)
    dump(path, res)
    await response_part(vl, res)
    dump(path, res)
    report(res)
    print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
