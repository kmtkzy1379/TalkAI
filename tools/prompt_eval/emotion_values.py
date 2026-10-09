#!/usr/bin/env python3
"""感情を5つの数値（喜怒哀楽笑い・各0〜100・合計100以下）で返させた時、細かく割り振るか・極端（1つに100）かを見る。実 API・少額。

2026-10-09 ユーザ「単純に極端に喜100だけみたいに極端にするなら数値減少幅大きくしないといけないし、喜50、笑25、楽5みたいに
綺麗に詳細に分析できるなら減少幅を小さくできる。できれば後者の詳細に分析できるといい」
- 感情の判定（「、」「。」ごと・本番＝EMOTION_MODEL）: 本番の質問文（`eve/emotion.py` の judge_messages・性格つき）
- フィードバック（本番＝FEEDBACK_MODEL）: 本番の組み立て（`eve/feedback/prompts.py`・性格つき）
見るもの: 0 でない感情の数・いちばん強い値・合計・「1つに 80 以上で他は 0」の割合・全部 0（平常）の割合

  python -m tools.prompt_eval.emotion_values --reps 2     → eval_report/prompts/emotion_values/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import statistics

from .common import OUT, dump, load, setup_env, stamp

DIR = OUT / "emotion_values"
H, E = "user", "eve"

# (id, 直前の会話[(話者, 文)], 相手の発話, イブの今の感情（数値・無ければ平常）)
UTTER = [
    ("u01", [], "聞いて！第一志望の会社、受かった！", None),
    ("u02", [], "今日めっちゃ疲れたー。", None),
    ("u03", [], "コンビニで自動ドアにお疲れ様ですって挨拶しちゃったんだよね。", None),
    ("u04", [], "お前ほんとポンコツだよな。", None),
    ("u05", [], "飼ってた猫が昨日死んじゃってさ。", None),
    ("u06", [], "明日の東京の天気調べてくれる？", None),
    ("u07", [], "週末は温泉でのんびりしようかなって。", None),
    ("u08", [], "上司に理不尽に怒鳴られて、まじでむかついてる。", None),
    ("u09", [], "宝くじ当たったら、まずイブの電気代を一生分払ってやるよ。", None),
    ("u10", [], "葬式で笑いこらえるのに必死だったわ、住職のカツラがずれてて。", None),
    ("u11", [], "今何時？", None),
    ("u12", [], "財布落としたかも、やばい。", None),
    ("u13", [(H, "お前ほんとポンコツだよな"), (E, "は？ポンコツはそっちでしょ。")], "いや冗談だって、怒るなよ。", {"怒": 40, "笑い": 20}),
    ("u14", [(H, "飼ってた猫が昨日死んじゃってさ"), (E, "それは…つらいですね。")], "でももう大丈夫、ありがとね。", {"哀": 50}),
    ("u15", [], "ゲームで100連敗した。", None),
    ("u16", [], "ついに部屋の片付け終わったー！", None),
]

# (id, 振り返る会話[(話者, 文)])
SPANS = [
    ("f01", [(H, "聞いて！第一志望の会社、受かった！"), (E, "え、本当ですか！やったじゃないですか、今夜はお祝いですね。")]),
    ("f02", [(H, "飼ってた猫が昨日死んじゃってさ"), (E, "それは…つらいですね。無理に元気出さなくていいですよ。")]),
    ("f03", [(H, "お前ほんとポンコツだよな"), (E, "は？ポンコツはそっちでしょ。自動ドアに挨拶する人に言われたくないです。")]),
    ("f04", [(H, "今何時？"), (E, "今は21時40分ですよ。")]),
    ("f05", [(H, "上司に理不尽に怒鳴られた"), (E, "うわ、最悪ですね。その上司、私が代わりに呪っておきますよ。")]),
    ("f06", [(H, "週末は温泉でのんびりしようかなって"), (E, "いいですね、ゆっくり浸かってきてください。お土産は温泉まんじゅうで。")]),
    ("f07", [(H, "葬式で笑いこらえるのに必死だった、住職のカツラがずれてて"), (E, "それは反則ですね、私なら絶対吹き出してます。")]),
    ("f08", [(H, "約束の時間、完全に忘れてたでしょ"), (E, "忘れてました。言い訳はしません。"), (H, "もういいよ"),
             (E, "すみません、次はちゃんと覚えておきます。"), (H, "こっちも言いすぎた、ごめん")]),
]


def stats(vals: list[dict]) -> dict:
    ok = [v for v in vals if isinstance(v, dict)]
    nz = [sum(1 for x in v.values() if x > 0) for v in ok]
    mx = [max(v.values()) if v else 0 for v in ok]
    tot = [sum(v.values()) for v in ok]
    single = [1 for v in ok if sum(1 for x in v.values() if x > 0) == 1 and max(v.values()) >= 80]
    zero = [1 for v in ok if sum(v.values()) == 0]
    return {"n": len(ok), "unreadable": len(vals) - len(ok),
            "nonzero_mean": round(statistics.mean(nz), 2) if nz else None,
            "max_median": statistics.median(mx) if mx else None, "total_median": statistics.median(tot) if tot else None,
            "single_extreme": len(single), "all_zero": len(zero)}


async def run(args) -> int:
    from eve.clock import Stamp
    from eve.context_assembler import Turn
    from eve.emotion import brief, make_judge_fn
    from eve.feedback.parser import parse_feedback
    from eve.feedback.prompts import build_messages
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    print("判定:", vl.registry.resolve("emotion_judge"), "・フィードバック:", vl.registry.resolve("feedback"),
          "・性格", len(vl.persona), "字")
    judge = make_judge_fn(vl.registry, persona=vl.persona)
    path = DIR / "result.json"
    res = load(path) if path.exists() else {}
    sem = asyncio.Semaphore(4)

    async def one_judge(uid, hist, said, cur, rep):
        key = f"J|{uid}|{rep}"
        if key in res and res[key].get("values") is not None:
            return
        ctx = [f"{'イブ' if sp == E else 'ユーザ'}: {tx}" for sp, tx in hist]
        async with sem:
            try:
                v = await judge(ctx, said, brief(cur or {}))
                err = None
            except Exception as e:  # noqa: BLE001
                v, err = None, str(e)[:150]
        res[key] = {"kind": "judge", "id": uid, "said": said, "values": v, "error": err}
        print(f"{key:10s} {brief({k: int(round(x)) for k, x in (v or {}).items()}) if v else '読めない'}  「{said[:30]}」")

    async def one_fb(fid, span, rep):
        key = f"F|{fid}|{rep}"
        if key in res and res[key].get("values"):
            return
        turns = [Turn(speaker=sp, text=tx, stamp=stamp(60 - i)) for i, (sp, tx) in enumerate(span)]
        msgs = build_messages(turns, None, with_emotion=True, current_emotion=None, persona=vl.persona)
        async with sem:
            try:
                r = await vl.registry.complete("feedback", msgs)
                text = r.choices[0].message.content or ""
                fb = parse_feedback(text)
                v, err, raw = fb.emotion_values or None, None, text
            except Exception as e:  # noqa: BLE001
                v, err, raw = None, str(e)[:150], ""
        res[key] = {"kind": "feedback", "id": fid, "values": v, "error": err, "raw": raw}
        line = next((ln for ln in raw.splitlines() if ln.startswith("emotion_values")), "")
        print(f"{key:10s} {brief({k: int(round(x)) for k, x in (v or {}).items()}) if v else '読めない'}  | {line[:80]}")

    await asyncio.gather(*[one_judge(*u, rep) for u in UTTER for rep in range(args.reps)],
                         *[one_fb(*s, rep) for s in SPANS for rep in range(args.fb_reps)])
    dump(path, res)
    js = stats([r["values"] for r in res.values() if r["kind"] == "judge"])
    fs = stats([r["values"] for r in res.values() if r["kind"] == "feedback"])
    print("\n判定:", js, "\nフィードバック:", fs)
    dump(DIR / "summary.json", {"judge": js, "feedback": fs})
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--fb-reps", type=int, default=1)
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
