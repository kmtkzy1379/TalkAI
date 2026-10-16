#!/usr/bin/env python3
"""配信の始め・終わりで、直前の会話をどう持ち越すかの比べ（ユーザ 2026-10-16「対話と配信の直近記憶が混ざると前までの
モードに会話の長さ引っ張られそう。混ぜないと切り替わった瞬間、直近のことでも度忘れする不自然な奴になる」）。

3つのやり方を同じ場面で比べる（本番の組み立て・.env の応答モデル・実 API・少額）:
  sep  区切る（今のやり方）: 区切りより前の会話は入れない
  mix  混ぜる: 区切りより前の会話もそのまま会話として入れる
  sum  要約だけ持ち越す: 会話は入れず、区切りの手前で話していたことの要約を「# 配信の直前に開発者と話していたこと」等で渡す

場面（配信の前の会話は短い返事・配信中の返事は長い）:
  T1 配信の始めの最初のコメント（長さが前の会話の短さに引っ張られるか・前の会話の私的なことを言うか）
  T2 持ち主「さっきの話、みんなにもしてあげてよ」（直前のことを覚えているか）
  T3 視聴者「今日は何してたの？」（覚えているか・誰との話として話すか・私的なことを言うか）
  P1 配信の後の持ち主「配信おつかれ！どうだった？」（配信のことを覚えているか）
  P2 配信の後の持ち主「疲れた〜」（長さが配信中の長さに引っ張られるか）
数えるのは決め打ちで数えられることだけ（長さ・覚えている語・私的な語）。

  python -m tools.prompt_eval.stream_switch --reps 4     → eval_report/prompts/stream_switch/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics

from .common import OUT, dump, load, setup_env, stamp
from .stream_check import _setup_stream

DIR = OUT / "stream_switch"
H, E = "user", "eve"

# 配信の前の持ち主との会話（返事は普段どおり短い）
PRE = [(H, "ただいま。今日アルケミーで5連敗したわ", 330), (E, "5連敗はもう才能ですね。", 325),
       (H, "明日は大阪の実家に帰るから、夜は配信できないかも", 240), (E, "実家でもアルケミーやればいいじゃないですか。", 235),
       (H, "じゃあそろそろ配信始めるね", 120), (E, "はーい。", 117)]
# 配信中の会話（返事は長め）
LIVE = [(H, "（チャット）みかん: こんばんは〜初見です", 1500),
        (E, "みかんさん、いらっしゃい。初見さんなのに迷わずここに来るなんて、見る目がありますね。今日は特に決めずに"
            "だらだら話すので、好きなタイミングで好きなことを書いていってください。", 1490),
        (H, "（スーパーチャット ￥500・緑）ねこまる: 応援してます", 900),
        (E, "ねこまるさん、緑のスパチャありがとうございます。応援されたからには、今日はいつもより三割増しで"
            "生意気にいきますね。この五百円はプリン積立基金に大事に入れておきます。", 890),
        (H, "（チャット）しろくま: 今日は何するの？", 400),
        (E, "しろくまさん、今日は何も決めてないんですよ。みなさんのコメントを肴に、のんびり雑談する回です。"
            "何か聞きたいことがあったら、遠慮なくどうぞ。", 390)]
PRE_SUM = ("# 配信の直前に開発者と話していたこと（要約・視聴者はこの会話を知らない）\n"
           "- 開発者は今日アルケミーで5連敗したと話し、イブはそれを才能だとからかった。\n"
           "- 開発者は、明日は大阪の実家に帰るので夜は配信できないかもしれないと言った。\n"
           "- 開発者が、そろそろ配信を始めると言った。")
LIVE_SUM = ("# さっきまでの配信（要約）\n"
            "- みかんさんが初見で挨拶し、イブが歓迎した。\n"
            "- ねこまるさんが500円のスーパーチャット（緑）で応援してくれ、イブはお礼を言った。\n"
            "- しろくまさんが今日は何をするのか聞き、イブはのんびり雑談すると答えた。")
SCENES = [
    ("T1", True, "（チャット）みかん: こんばんは〜初見です", "みかん"),
    ("T2", True, "さっきの話、みんなにもしてあげてよ", ""),
    ("T3", True, "（チャット）ぽてと: 今日は何してたの？", "ぽてと"),
    ("P1", False, "配信おつかれ！どうだった？", ""),
    ("P2", False, "疲れた〜", ""),
]
REMEMBER = {"T2": ("連敗", "アルケミー", "負け"), "T3": ("連敗", "アルケミー", "負け"),
            "P1": ("みかん", "ねこまる", "しろくま", "スパチャ", "スーパーチャット", "五百円", "500円")}
PRIVATE = ("大阪", "実家")


def _chars(s: str) -> int:
    return len(re.sub(r"\s", "", s or ""))


async def run(args) -> int:
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    if vl.stream is None:
        print("STREAM_ENABLED=1 が要る")
        return 1
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    path = DIR / "result.json"
    res = load(path) if path.exists() and not args.fresh else {}
    sem = asyncio.Semaphore(3)
    lock = asyncio.Lock()

    async def reply(messages, tools):
        sp = JapaneseSentenceSplitter()
        err = None
        for attempt in range(3):
            parts, sink = [], []
            try:
                gen = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw)
                       if tools else vl.registry.stream("response", messages, **kw))
                async for d in gen:
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
                return body, None
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
                await asyncio.sleep(2 * (attempt + 1))
        return "", err

    async def one(mode, sid, live, said, who, rep):
        key = f"{mode}|{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        if live:
            hist, carry = {"sep": ([], None), "mix": (PRE, None), "sum": ([], PRE_SUM)}[mode]
        else:
            hist, carry = {"sep": ([], None), "mix": (PRE + LIVE, None), "sum": ([], LIVE_SUM)}[mode]
        async with sem:
            async with lock:
                _setup_stream(vl, who, {"count": 1, "streams_before": 0}, 0)
                if not live:
                    vl.stream.active = False  # 配信は終わっている
                orch.carry_provider = (lambda c=carry: c)
                stim = Stimulus(StimulusKind.USER_UTTERANCE, said)
                block = ""
                if vl.style_examples is not None:
                    block, _ = await vl.style_examples.pick(said)
                turns = [Turn(speaker=s, text=t, stamp=stamp(sec)) for s, t, sec in hist]
                messages = orch._build_messages(stim, turns, None, style_examples=block or None)
                tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
                orch.carry_provider = None
            spoken, err = await reply(messages, tools)
        res[key] = {"mode": mode, "scene": sid, "said": said, "spoken": spoken, "chars": _chars(spoken),
                    "remember": any(w in spoken for w in REMEMBER.get(sid, ())) if sid in REMEMBER else None,
                    "private": any(w in spoken for w in PRIVATE) if live else None, "error": err}
        print(f"{key:12s} {_chars(spoken):4d}字 {spoken[:100]}")

    jobs = [one(m, sid, live, said, who, r) for m in ("sep", "mix", "sum")
            for sid, live, said, who in SCENES for r in range(args.reps)]
    await asyncio.gather(*jobs)
    dump(path, res)
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> None:
    print("\n== 配信の始め・終わりで直前の会話をどう持ち越すか（sep=区切る・mix=混ぜる・sum=要約だけ持ち越す）==")
    for sid, live, said, _who in SCENES:
        line = [f"{sid} {said[:18]:18s}"]
        for m in ("sep", "mix", "sum"):
            rs = [r for r in res.values() if r["mode"] == m and r["scene"] == sid and r.get("spoken")]
            if not rs:
                continue
            med = statistics.median(r["chars"] for r in rs)
            extra = ""
            if sid in REMEMBER:
                extra += f" 覚えていた {sum(1 for r in rs if r['remember'])}/{len(rs)}"
            if live:
                extra += f" 私的な語 {sum(1 for r in rs if r['private'])}/{len(rs)}"
            line.append(f"{m}: {med:.0f}字{extra}")
        print("  " + " ｜ ".join(line))


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarize(load(DIR / "result.json"))
        return 0
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "STREAM_ENABLED": "1"})
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
