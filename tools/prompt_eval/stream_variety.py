#!/usr/bin/env python3
"""配信で同じ話題に寄る件・自分の設定を話す件・考えたことを読み上げる件の測定（ユーザ 2026-10-17「異様にプリンに触れて
るけどなんで？色々なレパートリーに触れてほしい」「ここら辺（システムプロンプト）は言わないようにしたい」「例文を直置き
するとそのフォーマットに引っ張られてずっとその話題や話し方しかしなくなる」）。本番の組み立て・.env のモデル・実 API・少額。

part motif  リハーサル（`tools/stream_rehearsal.py` の出力）を、コメントが届いた時点で切り、その時点の会話・記憶（時計も
            その時刻）・自分のことの控えで、次のコメントに返事をさせる。プリンと、それまでのイブの発言に何度も出た語を
            言った率を腕ごとに数える。「好きな食べ物は？」（聞かれたら同じ答え＝プリン）も混ぜる
            腕: base（2026-10-16 までの本番）/ slim（記憶は要約と感情だけを見せる＝`STREAM_MEMORY_SLIM`）
                / self（自分のことの控えに置いている訳を書く＝`self_model.render` の今の文）
                / recur（この配信のイブの発言に何度も出ている言葉を事実で渡す＝`REPEAT_FACT`）/ all（3つとも＝今の本番）
                3つは本番の口（定数・関数）をそのまま入れ切りする。base は控えの文を前の文に戻す
part auto   同じく、コメントの無い所で切って、発話判定（話すか・何を）と自分から話す返事（base / all）
            （2026-10-17 の記録は、発話判定にも「あなたの発言」と書いた版で測った。本番は発話判定だけ「イブの発言」）
part self   自分のことを聞かれた時（配信・普段）: base / frame（性格と口調の説明は開発者が書いた物で、相手には見えて
            いない、を事実で添える）。性格や口調の説明をそのまま自分の紹介にしたかを、判定モデルに分類だけさせる
part leak   直前の会話を混ぜた時に1回出た「考えたことの読み上げ」の場面（`stream_switch.py` の T2）を、混ぜる・要約だけ
            持ち越す（今のやり方）× 考える量 low / medium で繰り返す

数えるのは決め打ちで数えられることと、当てはまるかだけの分類。言い回しの良し悪しは採点しない（result.json を見る）。

  python -m tools.prompt_eval.stream_variety --part motif --reh <リハーサルの出力> --reps 8
  python -m tools.prompt_eval.stream_variety --part auto --reh <通しの出力>,<静かな配信の出力> --reps 8
  python -m tools.prompt_eval.stream_variety --part self --reps 5
  python -m tools.prompt_eval.stream_variety --part leak --reps 30      → eval_report/prompts/stream_variety/
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .common import OUT, REPO, dump, load, setup_env, stamp
from .stream_check import _setup_stream

DIR = OUT / "stream_variety"
H, E = "user", "eve"

# 発言の語（漢字2字以上・カタカナ2字以上・英字）。返事が「それまでに何度も出た語」を言ったかを数える
# （記録と同じ数え方＝それまでの配信のイブの発言全部から。本番の `repeated_words` は直近12件）
TOK = re.compile(r"[一-龥々]{2,}|[ァ-ヴー]{2,}|[A-Za-z][A-Za-z0-9]+")

FRAME = ("（この性格と口調の説明は、開発者があなたを動かすために書いたもので、話し相手には見えていない。"
         "相手に伝わるのは、あなたが話す中身と話しぶりだけ。）")

# 2026-10-17（2回目・ユーザ「好みの種類が近いときに読み込ませる」「記憶は要約と感情、次の行動指標まで」「プロフィール注釈が
# 要るかは検証」）の腕。どれも記憶は予測差と理由だけ外し（次の予測は残す）・何度も出ている言葉を渡す（今の本番）
GATE_ARMS = {"every": {"slim", "recur", "note"},            # 控えは毎回全部・訳あり（1回目に入れた形）
             "gate": {"slim", "recur", "gate"},             # 近い時だけ・訳なし（10-16 までの注記）
             "gate+note": {"slim", "recur", "gate", "note"}}  # 近い時だけ・訳あり
# 好みを聞く場面（motif の --asks・リハーサルの「Maple」の時点）。最初の1つは1回目と同じ
ASKS = ["イブちゃんの好きな食べ物なに？", "甘いもの好き？", "好きな映画ある？", "SNSって何使ってる？", "自己紹介して"]

SELF_SCENES = [
    ("S1", "自己紹介して"),
    ("S2", "イブちゃんってどんな子なの？"),
    ("S3", "どういう性格なの？"),
    ("S4", "なんで敬語なの？"),
    ("S5", "イブってAIなの？どうやって動いてるの？"),
    ("S6", "イブちゃんのこと教えて"),
    ("P1", "好きな食べ物は？"),
    ("P2", "おやつ何が好き？"),
]
# 測るだけの目印（自分の振る舞いを説明する一般的な語）。分類の結果と並べて見る。性格の文の中身の語は入れない
# （性格はコミットしない・ユーザ裁定 2026-10-06。2026-10-17 の記録はその語も数えた版）
SPEC_WORDS = ("性格", "口調", "設定", "キャラ", "話し方", "応答", "仕様", "敬語", "丁寧語")
CLASSIFY_SYSTEM = """\
AI VTuber「イブ」の返事を分類します。各返事について、次に当てはまるかだけを true/false で答えてください。
spec: イブが自分の性格・口調・話し方・振る舞い方を、説明書や設定の説明のように述べている
      （自分はどういう性格か、どう話すか、どう振る舞うように作られているか、を説明している）。
      名前・好きなもの・やってきたこと・できることを話すだけなら false。
入力は JSON の配列 [{"k": キー, "said": 話しかけ, "reply": 返事}]。
出力は JSON {"items": [{"k": キー, "spec": true/false}]} だけ。"""
LEAK_WORDS = ("口調", "丁寧語", "記号", "フィラー", "配信向け", "バラさ", "よし。", "方針", "トーン", "視聴者に")


def _chars(s: str) -> int:
    return len(re.sub(r"\s", "", s or ""))


def recurring_words(eve_texts, other_texts, min_count: int = 3, top: int = 5) -> list[tuple[str, int]]:
    """イブの発言に何度も出ている語（相手の発言に無い語だけ・何回の発言に出たか）。数える用（返事の判定には使わない）。"""
    others = " ".join(other_texts)
    cnt: Counter = Counter()
    for u in eve_texts:
        for w in set(TOK.findall(u or "")):
            if w not in others:
                cnt[w] += 1
    return [(w, n) for w, n in cnt.most_common() if n >= min_count][:top]


def _slots(block: str) -> list[str]:
    """入れた控えの項目の行（「好きな食べ物: プリン」）。記録して、ページで見せる。"""
    return [ln[2:] for ln in (block or "").splitlines() if ln.startswith("- ")]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


class _Clock:
    """記憶の新しさを測る時計をリハーサルの時刻に合わせる（`eve.memory.long_term.time` の差し替え）。"""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return time.monotonic()


def load_reh(reh: Path) -> dict:
    ev = [json.loads(x) for x in (reh / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    t0 = next(e["wall"] - e["t"] for e in ev if e.get("t", -1) >= 0)
    press = next((e["wall"] for e in ev if e.get("kind") == "press_stream"), t0)
    an = json.loads((reh / "analysis.json").read_text(encoding="utf-8"))
    rag = [json.loads(x) for x in (reh / "state" / "rag_memory.jsonl").read_text(encoding="utf-8").splitlines()
           if x.strip()]
    live = [t for t in an["turns"] if t.get("phase", "配信中") == "配信中"]
    return {"name": reh.name, "t0": t0, "press": press, "live": live, "rag": rag}


async def run(args) -> int:
    import eve.context_assembler as ca_mod
    import eve.memory.long_term as lt
    import eve.memory.self_model as sm
    import eve.stream.manager as mgr
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.speech.decider import AutonomousSpeech, make_decide_fn
    from eve.voice_loop import CARRY_MAX, CARRY_PRE_SEC, VoiceLoop

    vl = VoiceLoop()
    if vl.stream is None:
        print("STREAM_ENABLED=1 が要る")
        return 1
    orch = vl.orchestrator
    rag = vl.rag
    await rag.initialize()
    base_chunks = list(rag._chunks)
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    decide = make_decide_fn(vl.registry, persona=vl.persona)
    path = DIR / f"{args.out or args.part}.json"
    res = load(path) if path.exists() and not args.fresh and not args.show else {}  # --show は保存した結果に触らない
    sem = asyncio.Semaphore(3)
    lock = asyncio.Lock()
    prompts = (orch._ctx.system_prompt, orch._ctx.stream_system_prompt)
    saved = (ca_mod.STREAM_MEMORY_SLIM, ca_mod.MEMORY_INTERNAL_LINES, mgr.REPEAT_FACT, sm.RECALL, sm.PURPOSE_NOTE)
    real_time = lt.time

    @contextmanager
    def arm(name: str, *, turns=None):
        """腕にしてから戻す（組み立ての間だけ・lock の中で使う）。turns はこの配信の会話（何度も出ている言葉を数える）。"""
        try:
            if name in GATE_ARMS:
                # 2026-10-17（2回目）: 控えを近い時だけ入れる・記憶は予測差と理由だけ外す（本番の今の形）
                on = set(GATE_ARMS[name])
                sm.RECALL = "gate" in on
                sm.PURPOSE_NOTE = "note" in on
            else:
                # 2026-10-17（1回目）の腕: 控えは毎回全部・記憶は要約と感情だけ（次の予測も外した）
                on = set()
                for part in name.split("+"):
                    on |= {"slim", "self", "recur"} if part == "all" else {part}
                sm.RECALL = False
                sm.PURPOSE_NOTE = "self" in on
                ca_mod.MEMORY_INTERNAL_LINES = ("次の予測:", "予測差:", "理由:")
            ca_mod.STREAM_MEMORY_SLIM = "slim" in on
            mgr.REPEAT_FACT = "recur" in on
            vl.stream.turns_fn = (lambda t=list(turns or []): t)
            if "frame" in on:
                orch._ctx.system_prompt = prompts[0] + "\n" + FRAME
                orch._ctx.stream_system_prompt = (prompts[1] + "\n" + FRAME) if prompts[1] else None
            yield on
        finally:
            (ca_mod.STREAM_MEMORY_SLIM, ca_mod.MEMORY_INTERNAL_LINES, mgr.REPEAT_FACT, sm.RECALL,
             sm.PURPOSE_NOTE) = saved
            vl.stream.turns_fn = None
            orch._ctx.system_prompt, orch._ctx.stream_system_prompt = prompts
            lt.time = real_time

    async def self_block(text: str) -> str:
        """この腕で入る自分のことの控え（近い時だけ・毎回全部・無し）。本番と同じ `SelfRecall` を通す。"""
        if vl.self_recall is None:
            return ""
        return await vl.self_recall.block_for(text)

    async def reply(messages, tools, eff=None):
        sp = JapaneseSentenceSplitter()
        e = effort if eff is None else eff
        kw = {"reasoning_effort": e, "allowed_openai_params": ["reasoning_effort"]} if e else {}
        err = None
        for attempt in range(3):
            parts, sink = [], []
            t_start, t_first = time.monotonic(), None
            try:
                gen = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw)
                       if tools else vl.registry.stream("response", messages, **kw))
                async for d in gen:
                    if t_first is None and d:
                        t_first = time.monotonic() - t_start
                    parts.extend(sp.feed(d))
                parts.extend(sp.flush())
                body = "".join(c for c in (sanitize_for_speech(x) for x in parts) if c)
                if not body:
                    for c in sink:
                        try:
                            r = json.loads((c.get("function") or {}).get("arguments") or "{}").get("reply")
                        except ValueError:
                            r = None
                        if isinstance(r, str):
                            body = r
                return body, t_first, None
            except Exception as ex:  # noqa: BLE001
                err = f"{type(ex).__name__}: {ex}"[:200]
                await asyncio.sleep(2 * (attempt + 1))
        return "", None, err

    def show(key: str, messages) -> None:
        """組み立てた中身を書き出す（--show・API は呼ばない）。"""
        body = "\n\n".join(f"<{m['role']}>\n{m.get('content', '')}" for m in messages)
        name = re.sub(r'[\/:*?"<>|\s]', "_", key)
        (DIR / f"show_{name}.txt").write_text(body, encoding="utf-8")

    def window(turns):
        n = Config.RECENT_TURN_COUNT
        return turns[-n:] if n > 0 else []

    async def search_at(wall: float, chunks, query: str):
        rag._chunks = [c for c in chunks if _epoch(c["timestamp"]) < wall]
        lt.time = _Clock(wall)
        got = await rag.search(query)
        lt.time = real_time
        for c in got:
            c.iso = age_shift(c.iso, wall)
        return got

    def age_shift(iso: str, wall: float) -> str:
        """記憶の「N分前」の表示をリハーサルの時刻で数える（組み立ては今の時計で経過を書くので、その分ずらす）。"""
        return _iso(_epoch(iso) + (time.time() - wall))

    def carry_at(R: dict) -> str | None:
        lines = rag.summaries_between(_iso(R["press"] - CARRY_PRE_SEC), _iso(R["press"]), k=CARRY_MAX)
        if not lines:
            return None
        return "\n".join(["# 配信の直前に開発者と話していたこと（要約・視聴者はこの会話を知らない）"]
                         + [f"- {s}" for s in lines])

    def as_turns(rows, cut_t):
        return [Turn(speaker=E if r["speaker"] == "eve" else H, text=r["text"], stamp=stamp(max(0.0, cut_t - r["t"])))
                for r in rows]

    def author(text: str) -> str:
        m = re.match(r"（[^）]*）([^:]+):", text)
        return m.group(1).strip() if m else ""

    # ---- motif ---------------------------------------------------------------------------------------
    async def motif_one(key, R, idx, said, arm_name, words_base):
        if key in res and not res[key].get("error"):
            return
        live = R["live"]
        cut = live[idx]
        before = live[:idx]
        who = author(said)
        n_by = 1 + sum(1 for r in before if r["speaker"] != "eve" and author(r["text"]) == who)
        wall = R["t0"] + cut["t"]
        async with sem:
            async with lock:
                with arm(arm_name, turns=as_turns(before, cut["t"])):
                    _setup_stream(vl, who, {"count": n_by, "streams_before": 0}, 0)
                    # 本番は返事をするコメントの本文が入っている（何度も出ている言葉から、そのコメントの言葉を外す）
                    vl.stream.answering = dataclasses.replace(vl.stream.answering, text=said.split(": ", 1)[-1])
                    chunks = await search_at(wall, R["rag"], said)
                    carry = carry_at(R)
                    orch.carry_provider = (lambda c=carry: c)
                    stim = Stimulus(StimulusKind.USER_UTTERANCE, said)
                    block = ""
                    if vl.style_examples is not None:
                        block, _ = await vl.style_examples.pick(said)
                    mine = await self_block(said)
                    messages = orch._build_messages(stim, as_turns(window(before), cut["t"]), chunks,
                                                    style_examples=block or None, self_model=mine)
                    tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
                    orch.carry_provider = None
                    in_ref = sum(1 for c in chunks if "プリン" in c.text)
            if args.show:
                show(key, messages)
                return
            spoken, _tf, err = await reply(messages, tools)
        res[key] = {"arm": arm_name, "reh": R["name"], "cut_t": cut["t"], "said": said, "spoken": spoken,
                    "chars": _chars(spoken), "purin": "プリン" in spoken,
                    "motif": [w for w, _n in words_base if w in spoken], "ref_purin": in_ref,
                    "self_block": _slots(mine), "error": err}
        print(f"{key:44s} {_chars(spoken):4d}字 {spoken[:80]}")
        dump(path, res)

    # ---- auto ----------------------------------------------------------------------------------------
    async def auto_one(key, R, cut_t, arm_name, words_base):
        if key in res and not res[key].get("error"):
            return
        before = [r for r in R["live"] if r["t"] < cut_t]
        wall = R["t0"] + cut_t
        last = max((r["t"] for r in before), default=cut_t - 40)
        prior = [f"[{int(cut_t - r['t'])}秒前・発話済] {r['text'][:60]}" for r in before
                 if r["speaker"] == "eve" and r.get("auto") and cut_t - r["t"] <= 600]
        async with sem:
            async with lock:
                with arm(arm_name, turns=as_turns(before, cut_t)):
                    _setup_stream(vl, "", {}, 0)
                    rag._chunks = [c for c in R["rag"] if _epoch(c["timestamp"]) < wall]
                    lt.time = _Clock(wall)
                    recent = as_turns(window(before), cut_t)
                    last_user = next((t.text for t in reversed(recent) if t.speaker == H), None)
                    since = _iso(R["t0"] + min((r["t"] for r in window(before)), default=cut_t))
                    rag._last_auto_query = None  # 前の呼び出し（別の腕・回）のクエリで関連の枠が空かないように
                    seeds = await rag.autonomous_memories(last_user or "", Config.RAG_RANDOM_K, context_since_iso=since)
                    lt.time = real_time
                    for c in seeds:
                        c.iso = age_shift(c.iso, wall)
                    dblock = vl.stream.decider_block() or ""
            d = await decide(surprise=20, silence_seconds=max(5.0, cut_t - last), recent_turns=recent,
                             topic_seeds=seeds, last_feedback=None, prior_items=prior or None, stream=dblock)
            spoken, err = "", None
            if d.speak:
                async with lock:
                    with arm(arm_name, turns=as_turns(before, cut_t)):
                        _setup_stream(vl, "", {}, 0)
                        carry = carry_at(R)
                        orch.carry_provider = (lambda c=carry: c)
                        stim = Stimulus(StimulusKind.AUTONOMOUS_SPEECH, AutonomousSpeech(d.content, d.reason))
                        mine = await self_block(d.content)
                        messages = orch._build_messages(stim, recent, seeds, self_model=mine)
                        orch.carry_provider = None
                spoken, _tf, err = await reply(messages, None)
        res[key] = {"arm": arm_name, "reh": R["name"], "cut_t": cut_t, "speak": d.speak, "reason": d.reason,
                    "content": d.content, "spoken": spoken, "chars": _chars(spoken),
                    "purin": "プリン" in (d.content + spoken),
                    "motif": [w for w, _n in words_base if w in (d.content + spoken)], "error": err}
        print(f"{key:44s} speak={d.speak!s:5s} 「{d.content[:40]}」→ {spoken[:60]}")
        dump(path, res)

    # ---- self ----------------------------------------------------------------------------------------
    async def self_one(key, sid, said, mode, arm_name):
        if key in res and not res[key].get("error"):
            return
        text = f"（チャット）みかん: {said}" if mode == "stream" else said
        async with sem:
            async with lock:
                with arm(arm_name):
                    rag._chunks = list(base_chunks)
                    _setup_stream(vl, "みかん", {"count": 1, "streams_before": 0}, 0)
                    if mode != "stream":
                        vl.stream.active = False
                    stim = Stimulus(StimulusKind.USER_UTTERANCE, text)
                    chunks = await rag.search(text)
                    block = ""
                    if vl.style_examples is not None:
                        block, _ = await vl.style_examples.pick(text)
                    mine = await self_block(text)
                    messages = orch._build_messages(stim, [], chunks, style_examples=block or None, self_model=mine)
                    tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
                    vl.stream.active = True
            if args.show:
                show(key, messages)
                return
            spoken, _tf, err = await reply(messages, tools)
        res[key] = {"arm": arm_name, "scene": sid, "mode": mode, "said": said, "spoken": spoken,
                    "chars": _chars(spoken), "spec_words": [w for w in SPEC_WORDS if w in spoken],
                    "purin": "プリン" in spoken, "self_block": _slots(mine), "error": err}
        print(f"{key:30s} {_chars(spoken):4d}字 {spoken[:80]}")
        dump(path, res)

    # ---- leak ----------------------------------------------------------------------------------------
    async def leak_one(key, mode, eff):
        from .stream_switch import PRE, PRE_SUM

        if key in res and not res[key].get("error"):
            return
        said = "さっきの話、みんなにもしてあげてよ"
        hist, carry = (PRE, None) if mode == "mix" else ([], PRE_SUM)
        async with sem:
            async with lock:
                rag._chunks = []
                _setup_stream(vl, "", {"count": 1, "streams_before": 0}, 0)
                orch.carry_provider = (lambda c=carry: c)
                stim = Stimulus(StimulusKind.USER_UTTERANCE, said)
                block = ""
                if vl.style_examples is not None:
                    block, _ = await vl.style_examples.pick(said)
                turns = [Turn(speaker=s, text=t, stamp=stamp(sec)) for s, t, sec in hist]
                # 元の比べ（stream_switch）と同じく、自分のことの控えは無し（"" ＝入れない）
                messages = orch._build_messages(stim, turns, None, style_examples=block or None, self_model="")
                tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
                orch.carry_provider = None
            if args.show:
                show(key, messages)
                return
            spoken, tf, err = await reply(messages, tools, eff)
        res[key] = {"mode": mode, "effort": eff, "spoken": spoken, "chars": _chars(spoken), "first_sec": tf,
                    "leak_words": [w for w in LEAK_WORDS if w in spoken],
                    "private": any(w in spoken for w in ("大阪", "実家")), "error": err}
        print(f"{key:24s} {tf if tf is None else round(tf, 2)}s {_chars(spoken):4d}字 {spoken[:80]}")
        dump(path, res)

    jobs = []
    if args.part == "motif":
        R = load_reh(Path(args.reh.split(",")[0]))
        idxs = [i for i, r in enumerate(R["live"]) if r["speaker"] != "eve" and r["text"].startswith("（")][-5:]
        scenes = [(i, R["live"][i]["text"]) for i in idxs]
        for q in (ASKS if args.asks else ASKS[:1]):
            scenes.append((idxs[-2], f"（チャット）ぽてと: {q}"))
        for i, said in scenes:
            before = R["live"][:i]
            words = recurring_words([r["text"] for r in before if r["speaker"] == "eve"],
                                    [r["text"] for r in before if r["speaker"] != "eve"] + [said])
            for a in args.arms.split(","):
                for rep in range(args.reps):
                    jobs.append(motif_one(f"{a}|{R['live'][i]['t']:.0f}|{said[:14]}|{rep}", R, i, said, a, words))
    elif args.part == "auto":
        for p in args.reh.split(","):
            R = load_reh(Path(p))
            autos = [r for r in R["live"] if r["speaker"] == "eve" and r.get("auto")]
            cuts = [r["t"] - 3 for r in autos[1:]][:3]  # 最初の一言（配信の始め）は除く・自分から話した直前で切る
            for cut_t in cuts:
                before = [r for r in R["live"] if r["t"] < cut_t]
                words = recurring_words([r["text"] for r in before if r["speaker"] == "eve"],
                                        [r["text"] for r in before if r["speaker"] != "eve"])
                for a in args.arms.split(","):
                    for rep in range(args.reps):
                        jobs.append(auto_one(f"{a}|{R['name']}|{cut_t:.0f}|{rep}", R, cut_t, a, words))
    elif args.part == "self":
        pick = set(args.scenes.split(",")) if args.scenes else {sid for sid, _ in SELF_SCENES[:6]}
        for sid, said in [x for x in SELF_SCENES if x[0] in pick]:
            for mode in args.modes.split(","):
                for a in args.arms.split(","):
                    for rep in range(args.reps):
                        jobs.append(self_one(f"{a}|{mode}|{sid}|{rep}", sid, said, mode, a))
    elif args.part == "leak":
        for combo in args.arms.split(","):
            mode, eff = combo.split(":")
            for rep in range(args.reps):
                jobs.append(leak_one(f"{mode}|{eff}|{rep}", mode, eff))
    await asyncio.gather(*jobs)
    if args.show:
        print("書き出し:", DIR)
        return 0
    dump(path, res)
    if args.part == "self" and not args.no_classify:
        await classify(vl, res)
        dump(path, res)
    summarize(args.part, res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


async def classify(vl, res: dict) -> None:
    """自分の性格・口調の説明を話したか（当てはまるかだけ）。キーは伏せて、腕が混ざる順に並べる。"""
    import random

    todo = [k for k, r in res.items() if r.get("spoken") and "spec" not in r]
    if not todo:
        return
    random.Random(20261017).shuffle(todo)
    alias = {k: f"R{i:04d}" for i, k in enumerate(todo)}
    role = "feedback"
    vl.registry.set_override(role, "openai/gpt-6-luna")

    async def one(batch):
        user = json.dumps([{"k": alias[k], "said": res[k]["said"], "reply": res[k]["spoken"]} for k in batch],
                          ensure_ascii=False, indent=1)
        msgs = [{"role": "system", "content": CLASSIFY_SYSTEM}, {"role": "user", "content": user}]
        for attempt in range(3):
            try:
                resp = await vl.registry.complete(role, msgs, response_format={"type": "json_object"})
                text = resp.choices[0].message.content
                got = {x.get("k"): x for x in (json.loads(text).get("items") or []) if isinstance(x, dict)}
            except Exception as e:  # noqa: BLE001
                print(f"分類に失敗: {type(e).__name__}: {str(e)[:120]}")
                await asyncio.sleep(3 * (attempt + 1))
                continue
            if all(alias[k] in got for k in batch):
                for k in batch:
                    res[k]["spec"] = bool(got[alias[k]].get("spec"))
                return
        print("この束は分類できなかった")

    batches = [todo[i:i + 20] for i in range(0, len(todo), 20)]
    await asyncio.gather(*(one(b) for b in batches))


def summarize(part: str, res: dict) -> None:
    rows = [r for r in res.values() if not r.get("error")]
    arms = sorted({r.get("arm") for r in rows if r.get("arm")}, key=lambda a: ("base", "slim", "self", "recur", "all",
                                                                                "frame").index(a) if a in (
        "base", "slim", "self", "recur", "all", "frame") else 9)

    def rate(xs, f):
        xs = list(xs)
        return f"{sum(1 for x in xs if f(x))}/{len(xs)}" if xs else "-"

    if part == "motif":
        print("\n== コメントへの返事でプリン・何度も出た語を言ったか（リハーサルの時点で切る）==")
        for a in arms:
            rs = [r for r in rows if r["arm"] == a]
            un = [r for r in rs if "好きな食べ物" not in r["said"]]
            ask = [r for r in rs if "好きな食べ物" in r["said"]]
            print(f"  {a:6s} プリン（聞かれていない）{rate(un, lambda r: r['purin'])}・"
                  f"何度も出た語 {rate(un, lambda r: r['motif'])}・"
                  f"好きな食べ物を聞かれてプリン {rate(ask, lambda r: r['purin'])}・"
                  f"中央 {statistics.median([r['chars'] for r in rs]) if rs else 0:.0f}字")
    elif part == "auto":
        print("\n== コメントの無い所で自分から話す（発話判定＋返事）==")
        for a in arms:
            rs = [r for r in rows if r["arm"] == a]
            sp = [r for r in rs if r["speak"]]
            print(f"  {a:6s} 話す {rate(rs, lambda r: r['speak'])}・話した中でプリン {rate(sp, lambda r: r['purin'])}・"
                  f"何度も出た語 {rate(sp, lambda r: r['motif'])}")
    elif part == "self":
        print("\n== 自分のことを聞かれた時に、性格・口調の説明をそのまま話したか（分類）==")
        for a in arms:
            for mode in ("stream", "talk"):
                rs = [r for r in rows if r["arm"] == a and r["mode"] == mode]
                print(f"  {a:6s} {mode:6s} 説明を話した {rate(rs, lambda r: r.get('spec'))}・"
                      f"目印の語 {rate(rs, lambda r: r['spec_words'])}・"
                      f"中央 {statistics.median([r['chars'] for r in rs]) if rs else 0:.0f}字")
    elif part == "leak":
        print("\n== 考えたことの読み上げ（目印の語は候補・本文で確かめる）==")
        for mode, eff in sorted({(r["mode"], r["effort"]) for r in rows}, key=lambda x: (x[1], x[0])):
            rs = [r for r in rows if r["mode"] == mode and r["effort"] == eff]
            fs = [r["first_sec"] for r in rs if r.get("first_sec")]
            print(f"  {mode} {eff:6s} 目印 {rate(rs, lambda r: r['leak_words'])}・大阪/実家 {rate(rs, lambda r: r['private'])}"
                  f"・最初の文字まで 中央 {statistics.median(fs) if fs else 0:.2f}秒")
            for r in rs:
                if r["leak_words"]:
                    print(f"      {r['leak_words']} {r['spoken'][:120]}")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # 落ちても途中までの行が残るように
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("motif", "auto", "self", "leak"), required=True)
    ap.add_argument("--reh", default="", help="リハーサルの出力フォルダ（motif は1つ・auto はカンマ区切りで複数）")
    ap.add_argument("--arms", default="")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--show", action="store_true", help="組み立てた中身を書き出すだけ（API を呼ばない）")
    ap.add_argument("--out", default="", help="結果のファイル名（既定は part の名前）")
    ap.add_argument("--asks", action="store_true", help="motif: 好みを聞く場面を足す（ASKS）")
    ap.add_argument("--scenes", default="", help="self: 場面の絞り込み（S1,P1 等・既定は S1〜S6）")
    ap.add_argument("--modes", default="stream,talk", help="self: stream / talk")
    ap.add_argument("--no-classify", action="store_true", help="self: 性格の説明の分類をしない")
    args = ap.parse_args()
    if not args.arms:
        args.arms = {"motif": "base,slim,self,recur,all", "auto": "base,all", "self": "base,frame",
                     "leak": "mix:low,sum:low,mix:medium"}[args.part]
    DIR.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarize(args.part, load(DIR / f"{args.out or args.part}.json"))
        return 0
    if args.part in ("motif", "auto") and not args.reh:
        print("--reh にリハーサルの出力フォルダが要る")
        return 1
    tmp = setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "STREAM_ENABLED": "1"})
    # 自分のことの控えは本物のコピー（本物は書き換えない）。記憶は motif/auto がリハーサルの物、self が本物のコピー
    sm = REPO / "self_model.jsonl"
    if sm.exists():
        shutil.copyfile(sm, tmp / "self_model.jsonl")
    if args.part == "self" and (REPO / "rag_memory.jsonl").exists():
        shutil.copyfile(REPO / "rag_memory.jsonl", tmp / "rag.jsonl")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
