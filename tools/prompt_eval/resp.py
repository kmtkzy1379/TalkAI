#!/usr/bin/env python3
"""応答LLM（本番＝gpt-6-luna・reasoning_effort=none・ツール付き）の変種比べ。実 API・少額。

本番の `VoiceLoop` を組み立て（起動はしない）、`ResponseOrchestrator._build_messages` で本番と同じ messages を作り、
変種の差し替え（`variants.py`）をかけて呼ぶ。結果は eval_report/prompts/resp.json。

実行（EVE の venv・TalkAI-live のフォルダで）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.resp --items L,E --reps 3
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

from .common import OUT, dump, gather_limited, iso_ago, load, setup_env, stamp


def build_messages(vl, sc: dict, cfg: dict, extra_persona: str = ""):
    from eve.context_assembler import RagChunk, Turn
    from eve.emotion import EmotionState
    from eve.persona import PERSONA_HEADING, parse_persona
    from eve.pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
    from eve.response.style import SPEECH_STYLE
    from eve.speech.decider import AutonomousSpeech
    from eve.vlm.state import VisionState

    from . import variants as V

    orch = vl.orchestrator
    speech = V.speech_style(SPEECH_STYLE, length=cfg["length"], enum=cfg["enum"],
                            unknown="M" if cfg["lookup"] == "M" else cfg.get("unknown", "0"))
    body = parse_persona(V.persona_md(line1=cfg["persona"], lookup=cfg["lookup"]))
    if extra_persona:  # 4回目: 話し方の参考（実際のやりとり）を人柄の欄の最後に足す
        body = body.rstrip() + "\n\n" + extra_persona
    orch._ctx.system_prompt = speech + "\n\n" + f"{PERSONA_HEADING}\n{body}"

    # イブの今の感情・直前に鳴らしたつなぎ言葉（本番の provider と同じ所から入れる）
    emo = EmotionState()
    if sc.get("emotion"):
        # 場面の感情は1つのラベル（例「哀」）か、判定の数値（`fill_emotions` のキャッシュ）。ラベルは「はっきり」＝50 として渡す
        # （2026-10-09 から感情は5つの数値）
        v = sc["emotion"]
        emo.note({v: 50} if isinstance(v, str) else v, source="probe", as_of=time.time())
    orch._emotion_provider = emo.describe
    orch._filler_provider = (lambda f=sc.get("filler"): f)
    # イブ自身の好み（本番は自己モデルの台帳。場面で指定した時だけ入れる）
    orch._self_model_fn = (lambda s=sc.get("self_model"): s)

    # 直近フィードバック・前回の予想
    st = vl.prediction
    st.last_feedback = sc.get("last_feedback")
    st.last_prediction = sc.get("last_prediction")
    st.watermark = None

    # 画面（場面に指定がある時だけ。無ければ画面ブロック自体を出さない＝他の項目を画面で濁さない）
    scr = sc.get("screen")
    if scr:
        vs = VisionState()
        now = time.monotonic()
        if scr["state"] == "static":
            vs.set_latest(scr["narration"], now - scr["age"])
            vs.last_change_mono = now - scr["age"] - 5
            vs.last_frame_mono = now - 0.5  # capture は生きている
        else:  # unknown: 実況がまだ無い（起動直後など）
            vs.last_frame_mono = now - 0.5
        orch._vision_state = vs
    else:
        orch._vision_state = None

    kind = sc.get("kind", "user")
    if kind == "auto":
        stim = Stimulus(kind=StimulusKind.AUTONOMOUS_SPEECH,
                        payload=AutonomousSpeech(content=sc["text"], reason=sc.get("reason") or ""))
    elif kind == "report":
        stim = Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT,
                        payload=CallFunctionResult(function_name="delegate_task", content=sc["text"]))
    else:
        stim = Stimulus(kind=StimulusKind.USER_UTTERANCE, payload=sc["text"])
    recent = [Turn(speaker=sp, text=tx, stamp=stamp(sec)) for sp, tx, sec in sc.get("history") or []]
    rag = [RagChunk(text=tx, iso=iso_ago(sec)) for tx, sec in sc.get("rag") or []] or None
    messages = orch._build_messages(stim, recent, rag)
    tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None

    system = messages[0]["content"]
    system = V.apply_ack(system, cfg["ack"])
    system = V.apply_lookup_tool(system, bool(cfg.get("lookup_tool")))
    system = V.apply_screen(system, cfg["screen"], sc)
    system = V.apply_react(system, cfg["react"], sc)
    if cfg.get("old"):
        system = V.to_old(system, sc)
    system, tools = V.apply_tools(system, tools, cfg.get("tools_mode", "0"))
    messages[0]["content"] = system
    messages = V.apply_filler(messages, cfg["filler"], sc.get("filler"))
    return messages, tools


async def call(vl, messages, tools) -> dict:
    """本番の stream_fn と同じ呼び方（reasoning_effort=none・ツール付きは stream_with_tools）。"""
    from eve.config import Config
    from eve.response.splitter import JapaneseSentenceSplitter

    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = ({"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {})
    splitter = JapaneseSentenceSplitter()
    sink: list = []
    t0 = time.monotonic()
    clauses: list = []
    err = None
    for attempt in range(3):
        sink.clear()
        clauses.clear()
        try:
            stream = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw)
                      if tools else vl.registry.stream("response", messages, **kw))
            async for d in stream:
                clauses.extend(splitter.feed(d))
            clauses.extend(splitter.flush())
            err = None
            break
        except Exception as e:  # noqa: BLE001（1回の失敗で比べ全体を止めない・一時的な失敗は2回まで再試行）
            err = f"{type(e).__name__}: {e}"[:300]
            await asyncio.sleep(2.0 * (attempt + 1))
    tool_calls = [{"name": c["function"]["name"], "args": c["function"]["arguments"]} for c in sink]
    return {"text": "".join(clauses), "tools": tool_calls, "sec": round(time.monotonic() - t0, 2), "error": err}


async def fill_emotions(vl, scs: list) -> None:
    """emotion="auto" の場面に、本番の感情判定（ルナ・同じ文面）でイブの感情を付ける（結果はキャッシュ）。

    本番は話し終わる前の「、」までの判定を使う。ここでは発話全体で判定した値＝感情が応答に届いた場合を見る。
    """
    from eve.emotion import make_judge_fn

    cache_path = OUT / "emotions.json"
    cache = load(cache_path) if cache_path.exists() else {}
    judge = make_judge_fn(vl.registry)
    for sc in scs:
        if sc.get("emotion") != "auto":
            continue
        if sc["id"] not in cache:
            ctx = [f"{'イブ' if sp == 'eve' else 'ユーザ'}: {tx}" for sp, tx, _ in sc.get("history") or []][-10:]
            try:
                cache[sc["id"]] = await judge(ctx, sc["text"], None)
            except Exception as e:  # noqa: BLE001
                print(f"感情判定に失敗 {sc['id']}: {e}")
                cache[sc["id"]] = None
        sc["emotion"] = cache[sc["id"]]
        print(f"感情 {sc['id']}: {sc['emotion']}")
    dump(cache_path, cache)


async def run(args) -> int:
    from eve.voice_loop import VoiceLoop

    from . import scenarios as SCN
    from .variants import ITEM_VARIANTS, exemplar_block, resolve

    vl = VoiceLoop()
    out_path = OUT / "resp.json"
    results = load(out_path) if (out_path.exists() and not args.fresh) else {}
    await fill_emotions(vl, [s for it in args.items.split(",") for s in SCN.BY_ITEM[it]])
    jobs = []
    only = set(args.ids.split(",")) if args.ids else None
    for item in args.items.split(","):
        for sc in SCN.BY_ITEM[item]:
            if only and sc["id"] not in only:
                continue
            for var in ITEM_VARIANTS[item]:
                if all(f"{sc['id']}|{var}|{rep}" in results and not results[f"{sc['id']}|{var}|{rep}"].get("error")
                       for rep in range(args.reps)):
                    continue  # 取り済み（組み立て直さない＝本番に反映済みで作れなくなった変種でも止まらない）
                cfg = resolve(item, var)
                messages, tools = build_messages(vl, sc, cfg)
                for rep in range(args.reps):
                    key = f"{sc['id']}|{var}|{rep}"
                    if key in results and not results[key].get("error"):
                        continue
                    shown = None
                    if cfg.get("exemplars"):
                        # 話し方の参考は回ごとに入れ替える（同じ場面・同じ回なら E と NE で同じ組＝比べられる）
                        block, shown = exemplar_block(f"{sc['id']}|{rep}", cfg["exemplars"], said=sc["text"],
                                                      exclude_sim=cfg.get("exclude_sim"))
                        messages, tools = build_messages(vl, sc, cfg, extra_persona=block)
                    jobs.append((key, sc, var, rep, messages, tools, shown))
    print(f"呼び出し {len(jobs)} 件")

    async def one(job):
        key, sc, var, rep, messages, tools, shown = job
        r = await call(vl, messages, tools)
        sysmsg = messages[0]["content"]
        results[key] = {"id": sc["id"], "item": sc["item"], "variant": var, "rep": rep, **r,
                        "system_chars": len(sysmsg),
                        "system": sysmsg if rep == 0 else None,
                        "tail": messages[1:] if rep == 0 else None,
                        **({"exemplars": shown} if shown else {})}
        mark = f" ツール {[t['name'] for t in r['tools']]}" if r["tools"] else ""
        print(f"{key:14s} {r['sec']:5.2f}s {r['text'][:70]}{mark}{' 失敗 ' + r['error'] if r['error'] else ''}")

    t0 = time.monotonic()
    await gather_limited([one(j) for j in jobs], limit=args.parallel)
    dump(out_path, results)
    print(f"完了 {time.monotonic() - t0:.0f}秒 → {out_path}")
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="L,E,P,K,A,S,F,R,G")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--fresh", action="store_true", help="前の結果を捨てて取り直す")
    ap.add_argument("--ids", default="", help="この場面 id だけ回す（カンマ区切り）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
