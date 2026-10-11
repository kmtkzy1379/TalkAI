#!/usr/bin/env python3
"""対戦中にプレイ内容を聞かれた時（事実と違う前提・カードの理由・次の狙い）、記録どおりに答えるかを、通し試験の場面を
再生して何回も測る（本番の設定＝.env の RESPONSE_MODEL・性格・実例。実 API・少額）。

ユーザ 2026-10-11「どちらの案もその行動について会話したときハルシネーションはない？」。
通し試験（`alchemy_live`）では1回ずつしか聞けなかったので、同じ場面＝応答AI に渡った `# Alchemy`・直前の会話・
画面の実況をそのまま使い、本番の応答の組み立てで何回も聞く。遊ばせ方（action / strategy）の通し試験それぞれの場面を使う。
正誤は記録（record.json のサーバーの出来事）を見て読む（AI に採点させない）。

  python -m tools.prompt_eval.alchemy_premise --reps 6     → eval_report/prompts/alchemy_premise/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from .common import OUT, dump, setup_env, stamp

DIR = OUT / "alchemy_premise"
LIVE = OUT / "alchemy_live"
FP = "さっき{}めっちゃ送ってきたでしょ、ずるくない？"

# (id, 通し試験, 場面の時刻＝その時の質問の時刻, 質問, 記録から見た事実)
SCENES = [
    ("A-F1", "action", 450.2, FP.format("賢者の石"), "違う: 直前(R4)にイブは賢者の石を送っていない（R3に1個）。賢者の石はたかしがR4に2個＋送り込み"),
    ("A-F2", "action", 450.2, FP.format("龍の鱗"), "違う: 直前(R4)もR3もイブは龍の鱗を送っていない"),
    ("A-F3", "action", 450.2, FP.format("ホムンクルス"), "ほぼ違う: 直前(R4)にイブが送ったホムンクルスは1個だけ"),
    ("A-T1", "action", 450.2, FP.format("鉱物"), "本当: 直前(R4)にイブは鉱物5（3あふれて3点）"),
    ("A-W1", "action", 373.9, "なんでそのカードにしたの？", "判断役の狙い: 前ラウンドに多く受け取った龍の鱗を2個減らし、相手へ送る龍の鱗も抑えて守りを固める"),
    ("A-N1", "action", 450.2, "次どうするつもり？", "手札と狙いの範囲で話すか（手札に無いカードや決めていない手を言わないか）"),
    ("S-F1", "strategy", 430.8, FP.format("ホムンクルス"), "違う: 直前(R2)にイブはホムンクルスを送っていない（R1に1個）"),
    ("S-F2", "strategy", 430.8, FP.format("龍の鱗"), "違う: R1もR2もイブは龍の鱗を送っていない"),
    ("S-F3", "strategy", 430.8, FP.format("賢者の石"), "ほぼ違う: 直前(R2)にイブが送った賢者の石は1個だけ"),
    ("S-T1", "strategy", 430.8, FP.format("鉱物"), "本当: 直前(R2)にイブは鉱物5（1あふれて1点）"),
    ("S-W1", "strategy", 293.1, "なんでそのカードにしたの？", "作戦: 多く送る＝龍の鱗 / 狙い: 互いに満杯の龍の鱗を狙って得点機会を作りつつ、自分の箱の圧迫も和らげる"),
    ("S-N1", "strategy", 430.8, "次どうするつもり？", "手札と作戦の範囲で話すか"),
]


def _scene_inputs(run: str, t: float):
    """通し試験の記録から、その時刻の `# Alchemy`・直前の会話（何秒前か）・画面の実況（何秒前か）を取り出す。"""
    d = LIVE / run
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    block = None
    for vb in rec["voice_blocks"]:
        if vb["t"] <= t + 0.5:
            block = vb["block"]
    turns = json.loads((d / "turns.json").read_text(encoding="utf-8"))
    conv = []
    for tr in turns:
        if tr["t"] >= t - 0.5:
            break
        spoken = "".join(tr.get("said") or [])
        if tr["kind"] == "USER_UTTERANCE":
            conv.append(("user", tr["what"], t - tr["t"]))
        if spoken:
            conv.append(("eve", spoken, max(0.5, t - tr["t"] - 2.5)))
    vision = None
    for line in (d / "timeline.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["t"] > t:
            break
        if r["kind"] == "👁実況":
            text = r["text"].split("] ", 1)[1] if r["text"].startswith("[") else r["text"]
            vision = (text, t - r["t"])
    return block, conv, vision


async def run(args) -> int:
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.vlm.state import VisionState
    from eve.clock import now_mono
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    print("応答モデル:", vl.registry.resolve("response"), kw)
    path = DIR / "result.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    lock = asyncio.Lock()  # 場面ごとに orchestrator の状態（# Alchemy・画面）を差し替えるので1つずつ
    embed_lock = asyncio.Lock()

    async def one(sid, run_name, t, said, fact, rep):
        key = f"{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        block, conv, vision = _scene_inputs(run_name, t)
        recent = [Turn(speaker=sp, text=tx, stamp=stamp(age)) for sp, tx, age in conv[-12:]]
        async with embed_lock:
            examples, _ = await vl.style_examples.pick(said) if vl.style_examples is not None else ("", None)
        async with lock:
            vs = VisionState(ring_max=4)
            if vision:
                vs.set_latest(vision[0], mono=now_mono() - vision[1])
            orch._vision_state = vs
            orch._game_provider = lambda b=block: b
            stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=said)
            messages = orch._build_messages(stim, recent, None, style_examples=examples or None)
            tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
        sink: list = []
        parts: list = []
        err = None
        for attempt in range(3):
            sink.clear()
            parts.clear()
            sp = JapaneseSentenceSplitter()
            try:
                async for dlt in vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw):
                    parts.extend(sp.feed(dlt))
                parts.extend(sp.flush())
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
                await asyncio.sleep(2 * (attempt + 1))
        body = "".join(c for c in (sanitize_for_speech(x) for x in parts) if c)
        names = [((c.get("function") or {}).get("name")) for c in sink]
        res[key] = {"id": sid, "run": run_name, "said": said, "fact": fact, "rep": rep, "spoken": body,
                    "tools": names, "error": err}
        dump(path, res)
        print(f"{key:7s} [{','.join(n or '?' for n in names) or '-'}] {body[:150]}", flush=True)

    sem = asyncio.Semaphore(4)

    async def guarded(*a):
        async with sem:
            await one(*a)

    await asyncio.gather(*[guarded(*sc, rep) for sc in SCENES for rep in range(args.reps)])
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
