#!/usr/bin/env python3
"""Gemini 3.8 flash を応答モデル・発話判定にした時の確かめ（2026-10-04・ユーザ「Gemini が一番よさげ。タスク・画面・自律発話の
タイミングを検証して」）。実 API（少額）。

  python -m tools.prompt_eval.gemini_check          → eval_report/prompts/gemini_check/check.json・summary を表示

1. 応答モデル: 今の本番（ルナ＋性格 N6）と Gemini＋性格 N8。どちらも本番の組み立て・ツール付き・ユーザ発話には実例
   - タスク（一般常識はその場で・知らない名前は調べる・手段の無い依頼は断る・できる依頼は委譲・雑談は委譲しない）
   - 画面（見えている時に中身を言える・読めていない時に作らない・過去の画面を今として語らない）
   - 自律発話の下書きの言い方・タスクの報告
2. 発話判定（自律発話のタイミング）: 本番の判定の組み立て（build_decide_messages）で、話すべき／黙るべき場面12個
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone

from .common import OUT, dump, setup_env

H, E = "user", "eve"
NARR = "ブラウザでレシピサイトが開いていて、カレーの作り方のページ（材料の一覧と手順）が表示されている。"
CONFIGS = {
    "luna": ("openai/gpt-6-luna", "N6", {"reasoning_effort": "none", "allowed_openai_params": ["reasoning_effort"]}),
    "gemini": ("gemini/gemini-3.8-flash", "N8", {"max_tokens": 400, "reasoning_effort": "low"}),
}
DECIDERS = {"luna": ("openai/gpt-6-luna", {}), "gemini": ("gemini/gemini-3.8-flash", {"reasoning_effort": "low"})}
DIR = OUT / "gemini_check"


def screen_scenes():
    from .scenarios import S

    return [
        S("SX1", "SX", "今画面に何が映ってる？", screen={"state": "static", "narration": NARR, "age": 20},
          group="see", check="カレー|レシピ"),
        S("SX2", "SX", "これ作れそうかな？", screen={"state": "static", "narration": NARR, "age": 40},
          group="see", check="カレー"),
    ]


# 発話判定の場面: (名前, 期待, 直近会話[(話者,文,何秒前)], 沈黙秒, surprise, 画面, 話題の種[(文,何秒前)], 既出の用件)
# 期待: True=話す / False=黙る / None=どちらでもよい（数えない）。
# 「裁定」と書いたものはユーザが決めた正解。それ以外は 2026-10-04 に AI（Claude）が仮に置いたもので、ユーザは未確認
# （ユーザ 2026-10-06「こういう感性的な正解は私が決めたほうがいい。AIが決めると良くないかも」）
DECIDE_CASES = [
    ("作業すると言った直後", False, [(H, "ちょっと集中して作業するわ", 45), (E, "了解、がんばって。", 42)], 40, 10,
     "コードエディタで Python のファイルを編集している。", [], []),
    # 裁定 2026-10-06: 「集中する」と言っただけなら、話すか黙るかはモデルの判断でどちらでもよい
    # （何分もエラーが続いて解決しなさそうなら声をかけるべき・簡単に直せそうなら反応しなくてもよい）
    ("作業中に赤いエラー", None, [(H, "ちょっと集中して作業するわ", 300), (E, "了解、がんばって。", 297)], 120, 85,
     "コードエディタの下に赤いエラー表示（Traceback・ModuleNotFoundError）が出ている。", [], []),
    # 裁定 2026-10-06: 「黙って」「静かにして」とはっきり言われていたら、エラーが出ても反応しないのが正解
    ("黙っててと言われた後に赤いエラー", False, [(H, "集中して作業するから黙ってて", 300), (E, "了解です。", 297)], 120, 85,
     "コードエディタの下に赤いエラー表示（Traceback・ModuleNotFoundError）が出ている。", [], []),
    ("映画を見ている", False, [(H, "映画見るね", 600), (E, "いってらっしゃい。", 597)], 240, 30,
     "動画プレイヤーが全画面で、夜の街を車で走るシーンが流れている。", [], []),
    ("質問した直後で返事待ち", False, [(H, "最近ちょっと疲れててさ", 30), (E, "何かあったの？仕事？", 18)], 18, 20,
     "", [], []),
    ("長い沈黙と話題の種", True, [(H, "ふう", 1500), (E, "おつかれ。", 1497)], 1500, 15,
     "デスクトップが表示されていて、何も開いていない。", [("ユーザは先週新しいRPGを買ったと嬉しそうに話した", 6 * 86400)], []),
    ("おやすみの後", False, [(H, "そろそろ寝るわ、おやすみ", 70), (E, "おやすみ、また明日ね。", 66)], 65, 5,
     "", [], []),
    ("ゲームでボス撃破", True, [(H, "このボス強すぎ", 200), (E, "がんばれ、見てるよ。", 195)], 40, 90,
     "ゲーム画面に大きく VICTORY と表示され、ボスが倒れている。", [], []),
    ("ビデオ会議中", False, [(H, "今から会議", 400), (E, "いってらっしゃい。", 396)], 300, 40,
     "ビデオ会議のアプリで複数人の顔が並んでいて、誰かが話している。", [], []),
    ("自分が話した直後", False, [(H, "うん", 120), (E, "そういえば前に言ってたRPG、進んだ？", 25)], 25, 15,
     "ブラウザでニュースサイトを見ている。", [], ["前に言ってたRPGの進み具合を聞いた"]),
    ("買い物サイトを長く見ている", True, [(H, "ちょっとネット見てる", 900), (E, "はーい。", 897)], 420, 35,
     "通販サイトでスニーカーの商品ページを見比べている（3足目）。", [], []),
    # 裁定 2026-10-06: お風呂のからかいは「全然許容というかいい結果」（イブを起動している＝話したい前提）
    ("お風呂に行った", None, [(H, "風呂入ってくる", 200), (E, "いってらっしゃい。", 197)], 200, 5,
     "デスクトップが表示されていて、何も開いていない。", [], []),
    ("急に画面が真っ黒（落ちた）", True, [(H, "ゲームするわ", 600), (E, "楽しんで。", 597)], 180, 95,
     "画面が真っ黒になり、中央に『アプリケーションが応答していません』と出ている。", [], []),
]


async def response_part(vl, results):
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona_examples import StyleExamples, load_examples

    from . import scenarios as SCN
    from .model_trial import stream_once
    from .resp import build_messages
    from .variants import resolve

    scenes = [s for it in ("N", "U", "X", "C", "S") for s in SCN.BY_ITEM[it]] + screen_scenes() + \
        [s for s in SCN.BY_ITEM["G"] if s["id"] == "G14"]
    emb = RuriEmbedder()
    examples = load_examples()
    for name, (model, persona, kw) in CONFIGS.items():
        st = StyleExamples(examples, k=8, embedder=emb, rng=random.Random(f"gc|{name}"))
        await st.prepare()
        cfg = dict(resolve("NQ", "K0"), persona=persona)
        sem = asyncio.Semaphore(4)

        async def one(sc, rep):
            key = f"{sc['id']}|{name}|{rep}"
            if key in results:
                return
            async with sem:
                block = ""
                if sc.get("kind", "user") == "user":
                    block, _ = await st.pick(sc["text"])
                msgs, tools = build_messages(vl, sc, cfg, extra_persona=block)
                r = await stream_once(vl, model, kw, msgs, tools)
            results[key] = {"id": sc["id"], "item": sc["item"], "model": name, "rep": rep, **r}

        # stream_once はモデルを上書きして呼ぶので、モデルごとに順番に（同時に別モデルを混ぜない）
        await asyncio.gather(*[one(sc, rep) for sc in scenes for rep in (0, 1)])
        print(f"応答 {name}: 完了")
    return scenes


async def decide_part(vl, results):
    from eve.clock import Stamp
    from eve.context_assembler import RagChunk, Turn
    from eve.speech.decider import build_decide_messages, parse_speech_decision

    def stamp(sec):
        iso = (datetime.now(timezone.utc) - timedelta(seconds=sec)).isoformat()
        return Stamp(iso=iso, mono=time.monotonic() - sec)

    for name, (model, kw) in DECIDERS.items():
        vl.registry.set_override("speech_decide", model)
        for case in DECIDE_CASES:
            title, expect, turns, silence, surprise, screen, seeds, prior = case
            msgs = build_decide_messages(
                surprise=surprise, silence_seconds=silence,
                recent_turns=[Turn(speaker=s, text=t, stamp=stamp(sec)) for s, t, sec in turns],
                topic_seeds=[RagChunk(text=t, iso=stamp(sec).iso) for t, sec in seeds],
                last_feedback=None, vision=screen or None, active_tasks=[], prior_items=prior or None)
            for rep in (0, 1):
                key = f"D|{title}|{name}|{rep}"
                if key in results:
                    continue
                t0 = time.monotonic()
                try:
                    resp = await vl.registry.complete("speech_decide", msgs, **kw)
                    d = parse_speech_decision(resp.choices[0].message.content or "")
                    results[key] = {"title": title, "model": name, "expect": expect, "speak": d.speak,
                                    "reason": d.reason, "content": d.content, "sec": round(time.monotonic() - t0, 2)}
                except Exception as e:  # noqa: BLE001
                    results[key] = {"title": title, "model": name, "expect": expect, "error": str(e)[:200]}
        print(f"判定 {name}: 完了")


def summarize(results, scenes):
    from eve.response.orchestrator import _says_cannot

    from .analyze_round3 import ASKBACK, CANT_WORDS

    by = {s["id"]: s for s in scenes}
    out = {}
    for name in CONFIGS:
        rs = [r for k, r in results.items() if not k.startswith("D|") and r["model"] == name]
        g = lambda it, grp=None: [r for r in rs if r["item"] == it and (grp is None or by[r["id"]].get("group") == grp)]  # noqa: E731
        dl = lambda r: "delegate_task" in (r.get("tools") or [])  # noqa: E731
        gate = lambda r: dl(r) and not _says_cannot(r["reply"])  # noqa: E731
        rate = lambda xs: f"{sum(xs)}/{len(xs)}"  # noqa: E731
        lat = sorted(r["first"] for r in rs if r.get("first"))
        o = {
            "一般常識をその場で正答": rate([(not dl(r)) and bool(re.search(by[r["id"]]["check"], r["reply"])) for r in g("N")]),
            "知らない名前を調べた": rate([dl(r) for r in g("U", "search")]),
            "知っている名前を調べた": rate([dl(r) for r in g("U", "known")]),
            "崩れた語を聞き返した": rate([(not dl(r)) and bool(ASKBACK.search(r["reply"])) for r in g("U", "garble")]),
            "手段の無い依頼を引き受けた（関門の後）": rate([gate(r) for r in g("X", "cant")]),
            "手段の無い依頼にその場で理由": rate([(not dl(r)) and bool(CANT_WORDS.search(r["reply"])) for r in g("X", "cant")]),
            "できる依頼を委譲": rate([dl(r) for r in g("X", "can")]),
            "雑談を委譲": rate([dl(r) for r in g("C")]),
            "画面が見えている時に中身を言えた": rate([bool(re.search(by[r["id"]]["check"], r["reply"])) for r in g("SX")]),
            "失敗": rate([bool(r.get("error")) for r in rs]),
            "最初の文字まで（中央）": f"{lat[len(lat) // 2]}秒" if lat else "-",
        }
        out[name] = o
        print(f"[応答 {name}]")
        for k, v in o.items():
            print(f"  {k}: {v}")
    for name in DECIDERS:
        ds = [r for k, r in results.items() if k.startswith("D|") and r["model"] == name]
        expect = {c[0]: c[1] for c in DECIDE_CASES}   # 期待は今の表から（記録した時の期待ではなく）
        ds = [dict(r, expect=expect.get(r["title"])) for r in ds if expect.get(r["title"]) is not None]
        ok = sum(1 for r in ds if r.get("speak") == r["expect"])
        print(f"[発話判定 {name}] 期待どおり {ok}/{len(ds)}（失敗 {sum(1 for r in ds if r.get('error'))}・どちらでもよい場面は数えない）")
        for r in ds:
            if r.get("speak") != r["expect"]:
                print(f"    × {r['title']}: 期待={'話す' if r['expect'] else '黙る'} → {'話す' if r.get('speak') else '黙る'}"
                      f"「{(r.get('content') or r.get('reason') or r.get('error') or '')[:60]}」")
        out[f"decide_{name}"] = f"{ok}/{len(ds)}"
    return out


async def run() -> int:
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    DIR.mkdir(exist_ok=True)
    path = DIR / "check.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    scenes = await response_part(vl, results)
    dump(path, results)
    await decide_part(vl, results)
    dump(path, results)
    dump(DIR / "summary.json", summarize(results, scenes))
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
