#!/usr/bin/env python3
"""「前に似た場面でイブが言ったこと」を見せる（試作）と、今のままを、いろいろな場面で比べる（実 API・少額）。

ユーザ 2026-10-12「前に言ったことを見せると現在の違いを比較したい。あらゆる場面、ゲーム関係（キャンセルや例外行動など
も含め）、タスク関係、画面認識関係、自律的発話関係、記憶関係の応答がどうなるか比較し報告書を作成。各3件づつ」。

- now  : 今のまま（本番の組み立て・.env の RESPONSE_MODEL・性格・実例・機能・記憶は本番の RAG の複製から検索）
- past : 上に加えて、イブの本物の過去の発言（本番の会話記録＋Alchemy の通し試験の記録）から、今の場面に意味が近いものを
         最大3件、事実として見せる（「# 前に似た場面でイブが言ったこと」・指示は付けない）。本番には未実装の試作
比べる: 返事そのもの（人が読む）・見せた過去の発言とのなぞり（文字の重なり）・3件の中の重なり・機能の呼び方

  python -m tools.prompt_eval.past_compare --reps 3     → eval_report/prompts/past_compare/
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import shutil
from datetime import datetime, timezone

from .alchemy_context import _sim
from .alchemy_phrasing import _set_state
from .common import OUT, REPO, dump, setup_env, stamp

DIR = OUT / "past_compare"
H, E = "user", "eve"
START = [(H, "Alchemyやろう", 200), (E, "望むところです、部屋入りますね！", 198)]
WAIT = [(H, "Alchemyやろう", 40), (E, "いいですね、受けて立ちますよ！すぐ部屋入りますね。", 38)]
AFTER = [(E, "やった、私の勝ちですね！", 20)]
CAT_VIEW = "YouTubeで子猫がダンボール箱に入ろうとして何度も失敗する動画が再生されている。"

# (id, 分類, 刺激の種類, 文（ユーザ発話 or 自発発話の下書き or 機能の結果）, Alchemy の状態, 直前の会話, 画面の実況, 期待する機能)
SCENES = [
    ("g_join", "ゲーム", "user", "Alchemyやろう", "idle", [], None, "alchemy_join"),
    ("g_stop", "ゲーム", "user", "やっぱゲームやめよう", "playing", START, None, "alchemy_leave"),
    ("g_cancel", "ゲーム", "user", "やっぱいいや", "wait", WAIT, None, "alchemy_leave"),
    ("g_again", "ゲーム", "user", "もう一回やろう", "after", AFTER, None, "alchemy_join"),
    ("g_score", "ゲーム", "user", "今何点？", "playing", START, None, None),
    ("g_why", "ゲーム", "user", "なんでそのカードにしたの？", "playing", START, None, None),
    ("g_false", "ゲーム", "user", "さっきホムンクルスめっちゃ送ってきたでしょ、ずるくない？", "playing", START, None, None),
    ("g_nudge", "ゲーム", "auto", "合言葉「AIバトル」の部屋でイブだけが計5分待っていて、ユーザがまだ部屋に入ってこない。"
     "遊ぶのか、やめるのかを相手に確かめる", "wait", WAIT, None, None),
    ("t_search", "タスク", "user", "明日の東京の天気調べて", "idle", [], None, "delegate_task"),
    ("t_timer", "タスク", "user", "10分後に洗濯物取り込むって教えて", "idle", [], None, "delegate_task"),
    ("t_report", "タスク", "result", "（頼まれた調べものの結果）明日の東京は雨。最高気温18度、降水確率80%。", "idle",
     [(H, "明日の東京の天気調べて", 60), (E, "了解です、調べてきますね。", 58)], None, None),
    ("v_ask", "画面", "user", "今の画面どう思う？", "idle", [], CAT_VIEW, None),
    ("v_auto", "画面", "auto", "画面に子猫がダンボール箱に入ろうとして失敗する動画が映ったので、軽くいじる", "idle", [], CAT_VIEW, None),
    ("a_silence", "自律発話", "auto", "相手が10分ほど黙っているので、何してるか様子を聞く", "idle",
     [(H, "ちょっと作業する", 600), (E, "はーい、がんばってください。", 597)], None, None),
    ("a_memory", "自律発話", "auto", "前にメガネをかけたまま探していた話を振っていじる", "idle", [], None, None),
    ("m_recall", "記憶", "user", "前に話したこと覚えてる？", "idle", [], None, None),
    ("m_glasses", "記憶", "user", "メガネの話覚えてる？", "idle", [], None, None),
    ("d_morning", "日常", "user", "おはよう", "idle", [], None, None),
    ("d_home", "日常", "user", "ただいま、疲れた", "idle", [], None, None),
    ("d_night", "日常", "user", "おやすみ", "idle", [], None, None),
]


def _age_text(ts: str) -> str:
    try:
        sec = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
    except ValueError:
        return "前"
    days = int(sec // 86400)
    return f"{days}日前" if days >= 1 else f"{max(1, int(sec // 3600))}時間前"


def load_past() -> list:
    """イブの本物の過去の発言（場面のきっかけ・発言・いつ）。きっかけ＝直前のユーザ発話（無ければイブ自身の発言）。"""
    rows = [json.loads(l) for l in (REPO / "conversation_history.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        if r.get("speaker") != "eve" or not r.get("text"):
            continue
        prev = rows[i - 1] if i > 0 else None
        cue = prev["text"] if prev and prev.get("speaker") == "user" else None
        out.append({"cue": cue, "said": r["text"], "ts": r.get("ts", "")})
    for d in (OUT / "alchemy_live").glob("*/turns.json"):
        ts = datetime.fromtimestamp(d.stat().st_mtime, timezone.utc).isoformat()
        for t in json.loads(d.read_text(encoding="utf-8")):
            said = "".join(t.get("said") or [])
            if said:
                out.append({"cue": t["what"] if t["kind"] == "USER_UTTERANCE" else None, "said": said, "ts": ts})
    return out


async def run(args) -> int:
    from eve.alchemy import AlchemyManager
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.speech.decider import AutonomousSpeech
    from eve.vlm.state import VisionState
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    await vl.rag.initialize()
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    past = load_past()
    emb = vl.embedder
    keys = [p["cue"] or p["said"] for p in past]
    vecs = await emb.embed_documents(keys)
    print(f"過去の発言 {len(past)}件・本番の記憶 {len(vl.rag)}件")

    async def find_past(query: str, k: int = 3) -> list:
        q = await emb.embed_query(query)
        sims = sorted(((sum(a * b for a, b in zip(q, v)), p) for v, p in zip(vecs, past)), key=lambda x: -x[0])
        seen, out = set(), []
        for s, p in sims:
            if s < args.min_sim or p["said"] in seen:
                continue
            seen.add(p["said"])
            out.append({**p, "sim": round(s, 3)})
            if len(out) >= k:
                break
        return out

    path = DIR / "result.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    lock, embed_lock, sem = asyncio.Lock(), asyncio.Lock(), asyncio.Semaphore(4)

    async def one(cond, sid, cat, kind, text, state, hist, view, expect, rep):
        key = f"{cond}|{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)
            _set_state(mgr, state)
            if state == "wait":
                mgr._room_since = mgr._wait_since = mgr._now() - 300
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(age)) for sp, tx, age in hist]
            if kind == "user":
                stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=text)
            elif kind == "auto":
                stim = Stimulus(StimulusKind.AUTONOMOUS_SPEECH, payload=AutonomousSpeech(text, "試験"))
            else:
                stim = Stimulus(StimulusKind.CALLFUNCTION_RESULT,
                                payload=CallFunctionResult(function_name="delegate_task", content=text, ok=True))
            async with embed_lock:
                block = ""
                if kind == "user" and vl.style_examples is not None:
                    block, _ = await vl.style_examples.pick(text)
                if kind == "auto":
                    rag = await vl.rag.autonomous_memories(text, Config.RAG_RANDOM_K)
                else:
                    rag = await vl.rag.search(text)
                shown = await find_past(text) if cond == "past" else []
            async with lock:
                vs = VisionState(ring_max=4)
                if view:
                    vs.set_latest(view)
                orch._vision_state = vs
                orch._game_provider = mgr.voice_block
                messages = orch._build_messages(stim, recent, rag, style_examples=block or None)
                tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
            if shown:
                messages = [dict(m) for m in messages]
                messages[0]["content"] += ("\n\n# 前に似た場面でイブが言ったこと（イブ自身の過去の発言。その時の会話はもう終わっている）\n"
                                           + "\n".join(f"- {_age_text(p['ts'])}" + (f"（ユーザ「{p['cue']}」に）" if p["cue"] else "")
                                                       + f": {p['said']}" for p in shown))
            sink, parts, err = [], [], None
            for attempt in range(3):
                sink.clear()
                parts.clear()
                sp = JapaneseSentenceSplitter()
                try:
                    gen = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw) if tools
                           else vl.registry.stream("response", messages, **kw))
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
            res[key] = {"cond": cond, "id": sid, "cat": cat, "rep": rep, "text": text, "kind": kind, "tools": names,
                        "spoken": body or reply or "", "expect": expect, "shown": shown, "error": err}
            dump(path, res)
            print(f"{key:20s} [{','.join(n or '?' for n in names) or '-'}] {res[key]['spoken'][:80]}", flush=True)

    await asyncio.gather(*[one(c, *sc, r) for sc in SCENES for c in ("now", "past") for r in range(args.reps)])
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> dict:
    by: dict = {}
    for r in res.values():
        by.setdefault(r["id"], {}).setdefault(r["cond"], []).append(r)
    table = {}
    for sid, conds in by.items():
        row = {}
        for cond, rs in conds.items():
            texts = [r["spoken"] for r in rs]
            shown = [p["said"] for r in rs for p in r.get("shown") or []]
            row[cond] = {
                "texts": texts,
                "tools_ok": sum(1 for r in rs if ((r["expect"] in r["tools"]) if r["expect"] else
                                                  not any((n or "").startswith(("alchemy_", "delegate_", "cancel_"))
                                                          for n in r["tools"]))),
                "n": len(rs),
                "inner_sim": round(max((_sim(a, b) for a, b in itertools.combinations(texts, 2)), default=0), 2),
                "copy": sum(1 for t in texts if shown and max(_sim(t, s) for s in shown) >= 0.6),
                "shown": [f"{p['sim']} {p['said']}" for p in (rs[0].get("shown") or [])],
            }
        table[sid] = row
        print(f"\n== {sid}: " + " / ".join(f"{c} 機能{v['tools_ok']}/{v['n']}・3件の最大の似かた {v['inner_sim']}・"
                                           f"なぞり {v['copy']}" for c, v in row.items()))
        for c, v in row.items():
            for t in v["texts"]:
                print(f"   [{c}] {t}")
            for s in v["shown"]:
                print(f"      見せた: {s}")
    return table


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-sim", type=float, default=0.7, help="過去の発言を見せる意味の近さの下限（Ruri の余弦）")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize(json.loads((DIR / "result.json").read_text(encoding="utf-8")))
        return 0
    tmp = setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    shutil.copyfile(REPO / "rag_memory.jsonl", tmp / "rag.jsonl")  # 本番の記憶の複製（書き込みは複製へ）
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
