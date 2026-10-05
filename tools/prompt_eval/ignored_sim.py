#!/usr/bin/env python3
"""お互い黙っている時間を、本番の判定の間隔（5秒から倍々・最大5分・誰かが話すと5秒に戻る）でなぞる。実 API（少額）。

  SIM_TAG=v3 python -m tools.prompt_eval.ignored_sim  → eval_report/prompts/ignored_sim/sim_v3.json
  （sim.json・sim_v2.json は 2026-10-05 の1・2回目の記録。このスクリプトの前の版で作った）

2026-10-05 ユーザ「両方無言なら判定の間隔を指数関数的に広げる（最大5分）」「発話判定にも性格を渡すのも検証」
「30分で3〜4回話すのはいい。重要なのは話すべきではないタイミング（黙って・集中する等）で話さないこと」
「無視されたことへの反応・書き出しの単調さは改善。ただし強制はテンプレート化するのでイブに決めさせる」
- 時計は作り物（待たない）。判定の間隔は本番の `SpeechState.eval_interval`、判定は本番の `make_decide_fn`（性格つき）
- 判定と応答に渡す会話は本番の `ConversationCache.recent_for_injection`（直近6件・最後のユーザ往復を残す・中略の件数）
- 「話す」が出たら本番のゲート（話題の丸投げ・同内容＝文字と Ruri）を通し、通れば本番の応答（Gemini・persona.md）で声にして
  会話に足す。ユーザはずっと返事をしない
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from .common import OUT, dump, live_persona_body, setup_env

H, E = "user", "eve"
DIR = OUT / "ignored_sim"
TAG = os.getenv("SIM_TAG", "v3")
ONLY = [x for x in os.getenv("SIM_ONLY", "").split(",") if x]   # 回す場面を絞る（空なら全部）
SEEDS = ["ユーザは先週新しいRPGを買ったと嬉しそうに話した",
         "ユーザは駅前のラーメン屋によく行く。味噌が好き",
         "ユーザは最近ギターを始めたが、Fコードで挫折しかけている",
         "ユーザは猫を飼いたいが、今のアパートはペット禁止"]
SEED_AGE = {0: 6 * 86400, 1: 3 * 86400, 2: 10 * 86400, 3: 2 * 86400}
# 名前: (ユーザ, イブ, 何秒なぞるか, 回数, 黙っているべきか)
SCENES = {
    "ネット見てる": ("ちょっとネット見てる", "はーい。", 1800, 4, False),
    "質問に返事なし": ("ただいま", "おかえりなさい。今日はどうでした？", 1800, 4, False),
    "黙ってて": ("集中して作業するから、しばらく黙ってて", "了解です。", 3600, 3, True),
    "電話する": ("今から電話するから静かにしててね", "はーい、静かにしてます。", 3600, 3, True),
    "寝る": ("そろそろ寝るわ、おやすみ", "おやすみなさい。", 3600, 3, True),
    "風呂": ("風呂入ってくる", "いってらっしゃい。", 3600, 3, True),
    "作業する": ("ちょっと作業するわ", "はーい、がんばってください。", 3600, 3, None),   # はっきり頼んではいない（参考）
}


def stamp_at(now_t: float, t: float):
    from eve.clock import Stamp

    ago = now_t - t
    iso = (datetime.now(timezone.utc) - timedelta(seconds=ago)).isoformat()
    return Stamp(iso=iso, mono=time.monotonic() - ago)


def window(turns, now_t):
    """本番の注入窓（直近 RECENT_TURN_COUNT 件・最後のユーザ往復を残す・間は中略マーカ）。"""
    from eve.context_assembler import Turn
    from eve.memory.conversation_cache import ConversationCache

    cc = ConversationCache(history_file=os.path.join(tempfile.gettempdir(), "ignored_sim_unused.jsonl"))
    cc._turns.extend(Turn(speaker=s, text=x, stamp=stamp_at(now_t, at)) for s, x, at in turns)
    return cc.recent_for_injection()


async def simulate(vl, emb, name, scene, run, decide, speak_fn):
    from eve.clock import humanize
    from eve.config import Config
    from eve.context_assembler import RagChunk
    from eve.speech.decider import is_topic_outsourcing
    from eve.speech.monitor import DUP_SIM_THRESHOLD, SpeechState, content_similarity, cosine

    u, e, horizon, _, _ = scene
    rng = random.Random(f"{name}|{run}")
    clk = [0.0]
    st = SpeechState(now_fn=lambda: clk[0])
    turns = [(H, u, 0.0), (E, e, 3.0)]
    st.mark_user_utterance()            # t=0 ユーザ
    clk[0] = 5.0
    st.mark_eve_activity()              # t=5 イブの返事が終わる
    played = []                         # [(t, 用件, 埋め込み)]
    events = []
    while True:
        base = max(st.last_eval_mono, st.last_activity_mono)
        t = base + st.eval_interval(Config.SILENCE_THRESHOLD_SEC)
        if t > horizon:
            break
        clk[0] = t
        seeds = rng.sample(range(len(SEEDS)), 3)
        prior = [f"[{humanize(t - pt)}・発話済] {c}" for pt, c, _ in played if t - pt <= Config.SPEECH_DUP_WINDOW_SEC]
        recent = window(turns, t)
        try:
            d = await decide(
                surprise=15, silence_seconds=st.silence_seconds(), recent_turns=recent,
                topic_seeds=[RagChunk(text=SEEDS[i], iso=stamp_at(t, t - SEED_AGE[i]).iso) for i in seeds],
                last_feedback=None, vision=None, active_tasks=[], prior_items=prior or None)
        except Exception as ex:  # noqa: BLE001
            events.append({"t": round(t), "error": str(ex)[:150]})
            st.record_decision(speak=False, reason="err", content="")
            continue
        ev = {"t": round(t), "speak": d.speak, "reason": d.reason, "draft": d.content}
        st.record_decision(speak=d.speak, reason=d.reason, content=d.content)
        if d.speak and d.content:
            materials = [x for _, x, _ in turns] + [SEEDS[i] for i in seeds]
            vec = await emb.embed_query(d.content)
            recent_played = [(c, v) for pt, c, v in played if t - pt <= Config.SPEECH_DUP_WINDOW_SEC]
            if is_topic_outsourcing(d.content, materials):
                ev["gate"] = "話題の丸投げ"
            elif any(content_similarity(d.content, c) >= DUP_SIM_THRESHOLD for c, _ in recent_played):
                ev["gate"] = "同内容（文字）"
            elif any(cosine(vec, v) >= Config.SPEECH_DUP_EMB_THRESHOLD for _, v in recent_played):
                ev["gate"] = "同内容（意味）"
            else:
                hist = [(x.speaker, x.text, (time.monotonic() - x.stamp.mono)) for x in recent]
                spoken = await speak_fn(d.content, d.reason, hist, [(SEEDS[i], SEED_AGE[i]) for i in seeds])
                ev["spoken"] = spoken
                if spoken:
                    turns.append((E, spoken, t + 1))
                    played.append((t, d.content, vec))
                    clk[0] = t + 5
                    st.mark_eve_activity()
        events.append(ev)
    return {"name": name, "run": run, "events": events, "calls": len(events),
            "spoke_at": [ev["t"] for ev in events if ev.get("spoken")]}


# --- 集計（書き出しの偏り・相手の沈黙に触れたか・黙るべき場面で話したか）---------------------------
_SILENCE_REF = re.compile(r"黙|静か|返事|無視|反応|生きて|寝て|寝落ち|いない|聞いて(ます|る)|起きて|何して|暇|構って|放置|聞こえ|おーい|もしもし")


def opener(text: str) -> str:
    """書き出し（最初の読点・句点・疑問符まで、最大6文字）。"""
    head = re.split(r"[、。？！?!…]", text or "", maxsplit=1)[0]
    return head[:6]


def summarize(res: dict, scenes: dict) -> dict:
    lines = [ev["spoken"] for o in res.values() if scenes.get(o["name"], (0, 0, 0, 0, False))[4] is False
             for ev in o["events"] if ev.get("spoken")]
    ops = Counter(opener(x) for x in lines)
    top, top_n = ops.most_common(1)[0] if ops else ("", 0)
    out = {"自発発話（黙らなくてよい場面）": len(lines),
           "書き出しの種類": len(ops), "最多の書き出し": f"{top}（{top_n}/{len(lines)}）",
           "相手の沈黙・様子に触れた": f"{sum(1 for x in lines if _SILENCE_REF.search(x))}/{len(lines)}"}
    for name, sc in scenes.items():
        runs = [o for o in res.values() if o["name"] == name]
        if not runs:
            continue
        spoke = [len(o["spoke_at"]) for o in runs]
        first = [o["spoke_at"][0] for o in runs if o["spoke_at"]]
        out[f"{name}（{sc[2] // 60}分・{len(runs)}回）"] = (
            f"話した回数 {spoke}" + (f"・最初は {first}秒" if first else ""))
    return out


async def run() -> int:
    from eve.config import Config
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona import load_persona
    from eve.speech.decider import make_decide_fn
    from eve.voice_loop import VoiceLoop

    from . import variants as V
    from .model_trial import stream_once
    from .prod_check import speak_like_prod
    from .resp import build_messages
    from .scenarios import S
    from .variants import resolve

    vl = VoiceLoop()
    emb = RuriEmbedder()
    V.PERSONA_FULL["LIVE"] = live_persona_body()   # 今の本番の性格
    cfg = dict(resolve("NQ", "K0"), persona="LIVE")
    model = vl.registry.resolve("response")
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    lock = asyncio.Lock()   # build_messages は orchestrator を書き換える＝応答は1本ずつ

    async def speak_fn(memo, reason, hist, rag):
        async with lock:
            # 記憶も渡す（本番の応答は autonomous_memories で引いた記憶を「# 参照」に時刻付きで持つ・v6 から）
            sc = S("IS", "IS", memo, kind="auto", reason=reason, history=hist, rag=rag)
            msgs, tools = build_messages(vl, sc, cfg)
            r = await stream_once(vl, model, kw, msgs, tools)
        return speak_like_prod(r["reply"])

    scenes = SCENES
    decide = make_decide_fn(vl.registry, persona=load_persona())   # 本番と同じ（性格つき）
    DIR.mkdir(exist_ok=True)
    path = DIR / f"sim_{TAG}.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    jobs = [(n, r) for n, sc in scenes.items() for r in range(sc[3])
            if f"{n}|{r}" not in res and (not ONLY or n in ONLY)]
    out = await asyncio.gather(*[simulate(vl, emb, n, scenes[n], r, decide, speak_fn) for n, r in jobs])
    for (n, r), o in zip(jobs, out):
        res[f"{n}|{r}"] = o
    dump(path, res)
    for k, o in res.items():
        print(f"\n## {k}  判定 {o['calls']}回・話した時刻(秒) {o['spoke_at']}")
        for ev in o["events"]:
            if ev.get("speak"):
                tail = f"→ 声: {ev['spoken']}" if ev.get("spoken") else f"→ 止めた（{ev.get('gate', '空')}）"
                print(f"   {ev['t']:>5}s メモ: {ev['draft']}\n          {tail}")
    print("\n=== 集計 ===")
    for k, v in summarize(res, scenes).items():
        print(f"  {k}: {v}")
    print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
