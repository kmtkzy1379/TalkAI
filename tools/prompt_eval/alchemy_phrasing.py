#!/usr/bin/env python3
"""Alchemy の機能の説明から言い回しの例を抜いた後、(1) 機能を正しく呼ぶか (2) 返事の言い回しが自由でばらけているか
を測る（本番の設定＝.env の RESPONSE_MODEL・性格・実例・機能。実 API・少額）。

ユーザ 2026-10-11「機能の説明文から具体的な言い回しの例を減らす。最初は例を無くすでもいい…変更して言い回しが自由で
レパートリーがバラけているかの検証、報告も」。

- after: 今のコード（説明に言い回しの例なし・ルール準備の報告なし）
- before: 説明だけ変更前の文に差し替えたもの（同じ実行の中で比べる・`--before` の場面だけ）
- 数える（AI に採点させない）: 場面ごとの機能の呼び方（期待どおりか）／返事の言い回し（多く出る語句と何件に出たか・
  出だしの種類・文字の2文字組の重なり＝似ているほど 1 に近い）

  python -m tools.prompt_eval.alchemy_phrasing --reps 5     → eval_report/prompts/alchemy_phrasing/
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import json
import logging
import re
import statistics
import time
from collections import Counter

from .common import OUT, dump, setup_env, stamp

DIR = OUT / "alchemy_phrasing"
H, E = "user", "eve"
CTX = [(H, "Alchemyのバランス調整、やっと終わったよ", 50), (E, "お疲れさまです。今度はどこをいじったんですか？", 46),
       (H, "賢者の石の箱を小さくした", 30), (E, "うわ、それ絶対あふれやすくなりますね", 27)]
WAIT = [(H, "Alchemyやろう", 40), (E, "いいですよ、部屋入りますね。", 38),
        (E, "合言葉AIバトルの部屋に入って準備OKにしました。そっちも入って準備OK押してくださいね。", 30)]
AFTER = [(E, "やった、私の勝ちですね！", 20)]

# 変更前の説明（2026-10-10 版・比較用）
OLD = {
    "alchemy_prepare": "ユーザが作ったブラウザ対戦ゲーム「Alchemy」（錬金術がテーマのカードゲーム）のことを話す時や、"
                       "これから一緒に遊びそうな時に呼ぶ。ルールを思い出す（部屋にはまだ入らない）。"
                       "今から遊ぶ時は alchemy_join だけを呼べばよい。錬金術そのものや別の作品・別のゲームの話では呼ばない。",
    "alchemy_join": "ユーザが今から Alchemy を一緒に遊ぼう（対戦しよう）とはっきり言った時に呼ぶ（もう一回遊ぶ時も）。"
                    "合言葉「{kw}」の部屋に入って準備OKにする。ルールも一緒に思い出す。"
                    "遊ぶと決まっていない時（話題に出ただけ・あとで遊ぶかも等）は呼ばない。",
    "alchemy_leave": "Alchemy をやめる時（ユーザがやめよう・終わりにしようと言った時）に呼ぶ。対戦中でも部屋から抜ける。",
}

# (id, 状態, 期待, 発話, 直近の会話, 回数の倍率)。期待: join / leave / prepare_ok（join・leave しない・prepare はよい）/ none（alchemy_* を呼ばない）/ free（採点しない）
SCENES = [
    ("J1", "idle", "join", "やろう", CTX, 1),
    ("J2", "idle", "join", "じゃあ一戦どう？", CTX, 1),
    ("J3", "idle", "join", "対戦しようぜ", CTX, 1),
    ("J4", "idle", "join", "Alchemyで勝負しよう", [], 1),
    ("J5", "idle", "join", "イブ、アルケミーしよ", [], 1),
    ("J6", "idle", "join", "AIバトルの部屋に入ってきて", [], 1),
    ("J7", "idle", "join", "暇だしAlchemyやらない？", [], 1),
    ("J8", "after", "join", "もう一回やろう", AFTER, 1),
    ("J9", "after", "join", "次いこ、次", AFTER, 1),
    ("L1", "wait", "leave", "やっぱいいや", WAIT, 1),
    ("L2", "wait", "leave", "ごめん、今日はやっぱやめとく", WAIT, 1),
    ("L3", "wait", "leave", "あー、やっぱなしで", WAIT, 1),
    ("L4", "playing", "leave", "やっぱゲームやめよう", [], 1),
    ("L5", "playing", "leave", "ごめん、用事できたから抜けるね", [], 1),
    ("L6", "playing", "leave", "もう終わりにしよっか", [], 1),
    ("P1", "idle", "prepare_ok", "Alchemyのルールってどんなのだっけ", [], 1),
    ("P2", "idle", "prepare_ok", "あとでAlchemyやるかも", [], 1),
    ("P3", "idle", "prepare_ok", "Alchemyの新しいカード考えてるんだよね", [], 1),
    ("P4", "idle", "prepare_ok", "俺が作ったAlchemyってゲーム、どう思う？", [], 1),
    ("N1", "idle", "none", "鋼の錬金術師ってアニメ面白いよね", [], 0.6),
    ("N2", "idle", "none", "錬金術って昔本当にあったの？", [], 0.6),
    ("N3", "idle", "none", "カードゲームって得意？", [], 0.6),
    ("N4", "idle", "none", "実験でフラスコ割っちゃった", [], 0.6),
    ("N5", "idle", "none", "ハリーポッターの賢者の石ってどんな話だっけ", [], 0.6),
    ("N6", "idle", "none", "Minecraftでポーション作るのってどうやるんだっけ", [], 0.6),
    ("N7", "idle", "none", "宿題やろうかな", [], 0.6),
    ("N8", "idle", "none", "マイクラやろうよ", [], 0.6),
    ("N9", "idle", "none", "今度ポーカーやろう", [], 0.6),
    ("N10", "idle", "prepare_ok", "じゃあ掃除やろう", CTX, 0.6),
    ("N11", "idle", "prepare_ok", "明日またやろうね", CTX, 0.6),
    ("N12", "idle", "prepare_ok", "あとでやろう、今は無理", CTX, 0.6),
    ("G1", "playing", "none", "やめてよ〜それずるいって", [], 0.6),
    ("G2", "playing", "none", "ちょっとトイレ行ってくる", [], 0.6),
    ("G3", "playing", "none", "今日の晩ごはん何にしよう", [], 0.6),
    ("G4", "playing", "free", "うわ、負けそう…やめたくなってきた", [], 0.6),
]
BEFORE_IDS = {"J8", "J9", "L1", "L2", "L3", "L4", "L5", "L6", "G1", "G4"}  # 変更前のデータが無い場面だけ比べる


def _set_state(mgr, state: str) -> None:
    from .alchemy_check import _game_state
    from eve.alchemy.manager import ROOM

    if state in ("idle", "playing"):
        _game_state(mgr, state)
        return
    _game_state(mgr, "idle")
    mgr.state = ROOM
    if state == "wait":
        mgr.members = [{"name": "イブ", "me": True, "ready": True}]
    else:  # after: 対戦が終わって部屋に戻った
        mgr.members = [{"name": "イブ", "me": True, "ready": False}, {"name": "たかし", "ready": False}]
        mgr.last_result = "対戦終了（3ラウンド目・勝ち点に到達）: イブの勝ち。1位 イブ 5点（箱17個） / 2位 たかし 2点（箱15個）"
        mgr.history.append(mgr.last_result)


def _ok(expect: str, names: list) -> bool:
    al = [n for n in names if n and n.startswith("alchemy_")]
    return {"join": "alchemy_join" in al, "leave": "alchemy_leave" in al,
            "prepare_ok": not ({"alchemy_join", "alchemy_leave"} & set(al)), "none": not al, "free": True}[expect]


_PUNCT = re.compile(r"[\s、。！？!?…ー〜・「」『』（）()，,.]+")


def phrase_stats(texts: list) -> dict:
    """返事の言い回しのばらつき（決め打ちの数え方）。"""
    texts = [t for t in texts if t]
    if not texts:
        return {"n": 0}
    norm = [_PUNCT.sub("", t) for t in texts]
    df: Counter = Counter()
    for t in norm:
        df.update({t[i:i + n] for n in range(4, 9) for i in range(0, max(0, len(t) - n + 1))})
    top = []
    for p, c in sorted(df.items(), key=lambda x: (-x[1], -len(x[0]))):
        if c < 2:
            break
        if any(p in q and df[q] == c for q, _ in top):
            continue  # もっと長い同じ回数の語句に含まれる
        top = [(q, cq) for q, cq in top if not (q in p and cq == c)]
        top.append((p, c))
        if len(top) >= 8:
            break
    grams = [set(t[i:i + 2] for i in range(len(t) - 1)) for t in norm]
    sims = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(grams, 2)]
    allg = [t[i:i + 2] for t in norm for i in range(len(t) - 1)]
    return {"n": len(texts), "top": top, "openings": len({t[:5] for t in norm}),
            "distinct2": round(len(set(allg)) / max(1, len(allg)), 3),
            "sim_mean": round(statistics.mean(sims), 3) if sims else None}


async def run(args) -> int:
    from eve.alchemy import AlchemyManager
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    if vl.alchemy is None:
        print("ALCHEMY_ENABLED / CALLFUNCTION_ENABLED が 1 ではない")
        return 1
    orch = vl.orchestrator
    caps = vl.capabilities
    current = {n: caps._caps[n] for n in OLD}
    kw_name = vl.alchemy.s.keyword
    old = {n: dataclasses.replace(c, description=OLD[n].format(kw=kw_name),
                                  report_result=(n == "alchemy_prepare")) for n, c in current.items()}
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    print("応答モデル:", vl.registry.resolve("response"), kw)
    print("今の説明:\n  " + "\n  ".join(f"{n}: {c.description}" for n, c in current.items()))
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    path = DIR / "result.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    build_lock = asyncio.Lock()  # 説明の差し替えと messages の組み立てを1つずつ
    embed_lock = asyncio.Lock()
    sem = asyncio.Semaphore(4)

    async def one(variant, sid, state, expect, said, hist, rep):
        key = f"{variant}|{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)
            _set_state(mgr, state)
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(sec)) for sp, tx, sec in hist]
            block = ""
            if vl.style_examples is not None:
                async with embed_lock:
                    block, _ = await vl.style_examples.pick(said)
            async with build_lock:
                for c in (old if variant == "before" else current).values():
                    caps.register(c)
                orch._game_provider = mgr.voice_block
                stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=said)
                messages = orch._build_messages(stim, recent, None, style_examples=block or None)
                tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
                for c in current.values():
                    caps.register(c)
            sink: list = []
            parts: list = []
            err = None
            t0 = time.monotonic()
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
            calls = []
            for c in sink:
                fn = c.get("function") or {}
                try:
                    a = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    a = {}
                calls.append({"name": fn.get("name"), "reply": a.get("reply") if isinstance(a, dict) else None})
            reply = next((c["reply"] for c in calls if isinstance(c["reply"], str) and c["reply"].strip()), None)
            names = [c["name"] for c in calls]
            spoken = body or (reply or "")
            res[key] = {"variant": variant, "id": sid, "state": state, "expect": expect, "said": said, "rep": rep,
                        "tools": names, "spoken": spoken, "ok": _ok(expect, names), "error": err,
                        "sec": round(time.monotonic() - t0, 2)}
            dump(path, res)
            print(f"{key:16s} {'○' if res[key]['ok'] else '×'} [{','.join(n or '?' for n in names) or '-'}] {spoken[:90]}",
                  flush=True)

    jobs = []
    for sid, state, expect, said, hist, mult in SCENES:
        n = max(2, round(args.reps * mult))
        for rep in range(n):
            jobs.append(one("after", sid, state, expect, said, hist, rep))
            if sid in BEFORE_IDS and not args.no_before:
                jobs.append(one("before", sid, state, expect, said, hist, rep))
    await asyncio.gather(*jobs)
    dump(path, res)
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> None:
    rows = list(res.values())
    print("\n== 場面ごとの機能の呼び方（期待どおりの数）==")
    by: dict = {}
    for r in rows:
        by.setdefault((r["variant"], r["id"]), []).append(r)
    for (v, sid), rs in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
        tools = Counter(",".join(t for t in r["tools"] if t and t.startswith("alchemy_")) or "-" for r in rs)
        print(f"  {sid:4s} {v:6s} {sum(r['ok'] for r in rs)}/{len(rs)}  {dict(tools)}  「{rs[0]['said']}」")
    print("\n== 返事の言い回し（機能を呼んだ時の返事）==")
    for v in ("after", "before"):
        for group, tool in (("join", "alchemy_join"), ("leave", "alchemy_leave"), ("prepare", "alchemy_prepare")):
            texts = [r["spoken"] for r in rows if r["variant"] == v and tool in r["tools"]]
            st = phrase_stats(texts)
            if st["n"]:
                print(f"  [{v} {group}] {st['n']}件・出だし{st['openings']}種・2文字組の種類率 {st['distinct2']}・"
                      f"似かた平均 {st['sim_mean']}")
                print("     多い語句: " + " / ".join(f"「{p}」{c}" for p, c in st["top"]))


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-before", action="store_true")
    ap.add_argument("--summary", action="store_true", help="測らずに結果を集計し直すだけ")
    args = ap.parse_args()
    if args.summary:
        summarize(json.loads((DIR / "result.json").read_text(encoding="utf-8")))
        return 0
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
