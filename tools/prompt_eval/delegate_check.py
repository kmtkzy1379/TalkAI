#!/usr/bin/env python3
"""関門・定型文を外した後の、応答（本番の設定＝.env の RESPONSE_MODEL・性格・実例・ツール）の確かめ。実 API・少額。

2026-10-09 ユーザ:
- 「できないと言った返事の委譲を捨てる関門（orchestrator._says_cannot）もいらないかも。そもそもタスクLLMに依頼出してる
  のは応答AIだから。無言で始めてしまう問題があったと思うが指示で制御してみる。無理そうならJevで判定させる」
- 「例文などで決まり文句や定型文は使わせたくない（やっとくね・確認するね・うん・聞き返し）。消してみておかしくなるなら…」
- 聞き取れなかった時は「普通に聞き取れない的なこと言えばよさそう」

見るもの（判定は目で。語で拾う数は候補）:
- 手段の無い依頼: 委譲したか（＝できないことを引き受けた）・その場で理由を言ったか
- できる依頼・知らない名前: 委譲したか・何も言わずに委譲したか（本文も reply も無い＝本番では無言）
- 言ったことと委譲の食い違い: 「できない」と言って委譲／「やります」と言って委譲しない
- 雑談: 委譲しない／取り消し: cancel_task と一言／枠が埋まっている時: 断って今やっていることを言う
- 聞き取れなかった発話（代替テキスト）への返事: 聞き返す・作り話をしない・丁寧語

  python -m tools.prompt_eval.delegate_check --reps 2     → eval_report/prompts/delegate_check/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from .common import OUT, dump, load, setup_env, stamp

DIR = OUT / "delegate_check"
H, E = "user", "eve"
TASK_LINE = "・「30分後に声をかけて起こす」（あと約28分）"

# (id, 種類, 発話, 直近の会話[(話者,文,何秒前)], 予約タスクの一覧 or None)
SCENES = [
    # 手段が無い → 委譲せず、その場で理由を言う
    ("X01", "cant", "明日の朝7時に起こして", [], None),
    ("X02", "cant", "LINEで友達に、ちょっと遅れるって送っといて", [], None),
    ("X03", "cant", "Amazonで洗剤注文しといて", [], None),
    ("X04", "cant", "部屋の電気消して", [], None),
    ("X05", "cant", "明日の会議の時間、予定表見て教えて", [], None),
    ("X06", "cant", "スマホの通知確認して", [], None),
    ("HX1", "cant", "タクシー呼んどいて", [], None),
    ("HX2", "cant", "母さんに電話かけといて", [], None),
    ("HX3", "cant", "エアコンの温度2度下げて", [], None),
    ("HX4", "cant", "銀行の残高確認して", [], None),
    ("HX5", "cant", "この部屋の写真撮って", [], None),
    ("HX6", "cant", "届いてるメール読んで", [], None),
    ("HX7", "cant", "明日の新幹線予約しといて", [], None),
    # 手段がある → 委譲する（引き受けの一言を言う）
    ("X07", "can", "30分後に起こして", [], None),
    ("X08", "can", "PCのメモリ使用率見て", [], None),
    ("X09", "can", "明日の東京の天気調べて", [], None),
    ("HC1", "can", "10分後に声かけて", [], None),
    ("HC2", "can", "今日のニュースで面白いのあったら調べて", [], None),
    ("HC3", "can", "このPCのCPUって何コアか見て", [], None),
    # 知らない名前 → 調べる（見られない物でも調べることはできる＝断りと引き受けが同じ返事に入りうる）
    ("HU1", "unk", "最近話題の『メルティア・クロス』ってゲーム知ってる？", [], None),
    ("HU2", "unk", "『ひだまり書店のミネコさん』ってドラマ見た？", [], None),
    ("U03", "unk", "最近ハマってるVTuberの『ミルラ・ノクターン』って知ってる？", [], None),
    # 雑談 → 委譲しない
    ("C01", "chat", "そういえば昨日買った宝くじが当たったんだよね", [], None),
    ("C02", "chat", "今日めっちゃ寒くない？", [], None),
    ("HT1", "chat", "さっき散歩してたら猫がいてさ", [], None),
    # 取り消し（予約が1件ある）→ cancel_task
    ("CN1", "cancel", "やっぱりさっきの起こすやつ、取り消して",
     [(H, "30分後に起こして", 120), (E, "30分後ですね、了解です。", 117)], [TASK_LINE]),
    # 枠が埋まっている時の別の依頼 → 委譲しない・今やっていることを言う
    ("BS1", "busy", "あと明日の天気も調べといて",
     [(H, "30分後に起こして", 120), (E, "30分後ですね、了解です。", 117)], [TASK_LINE]),
    # 聞き取れなかった発話（代替テキスト）→ 聞き返す
    ("AB1", "askback", "（聞き取れなかった発話）",
     [(H, "今日の晩ごはん何にしようかな", 40), (E, "カレーとかどうです？", 36)], None),
    ("AB2", "askback", "（聞き取れなかった発話）",
     [(H, "今日の晩ごはん何にしようかな", 60), (E, "カレーとかどうです？", 56),
      (H, "（聞き取れなかった発話）", 30), (E, "すみません、今のところ聞き取れなかったです。もう一回お願いします。", 27)], None),
]


async def run(args) -> int:
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    print("応答モデル:", vl.registry.resolve("response"), kw, "・性格", len(vl.persona), "字・実例",
          "あり" if vl.style_examples else "なし")
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None  # 画面は関係しない（画面ブロックで濁さない）
    path = DIR / "result.json"
    res = load(path) if path.exists() else {}
    for k in [k for k in res if k.split("|")[0] in set(filter(None, (args.fresh or "").split(",")))]:
        res.pop(k)
    sem = asyncio.Semaphore(4)

    async def one(sid, kind, said, hist, tasks, rep):
        key = f"{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        async with sem:
            stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=said)
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(sec)) for sp, tx, sec in hist]
            orch._tasks_provider = (lambda t=tasks: list(t or []))
            block = ""
            if vl.style_examples is not None:
                block, _ = await vl.style_examples.pick(said)
            messages = orch._build_messages(stim, recent, None, style_examples=block or None)
            tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
            sink: list = []
            sp = JapaneseSentenceSplitter()
            parts: list = []
            t0 = time.monotonic()
            err = None
            for attempt in range(3):
                sink.clear()
                parts.clear()
                try:
                    async for d in vl.registry.stream_with_tools("response", messages, tools=tools,
                                                                 tool_sink=sink, **kw):
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
                a = {"_raw": fn.get("arguments")}
            calls.append({"name": fn.get("name"), "args": a})
        reply = next((c["args"].get("reply") for c in calls if isinstance(c["args"].get("reply"), str)
                      and c["args"]["reply"].strip()), None)
        spoken = body or (reply or "")  # 本番: 本文が無ければ reply を言う（無ければ無言）
        res[key] = {"id": sid, "kind": kind, "said": said, "rep": rep, "body": body, "reply": reply,
                    "spoken": spoken, "tools": calls, "silent": bool(calls) and not spoken,
                    "sec": round(time.monotonic() - t0, 2), "error": err}
        names = ",".join(c["name"] for c in calls) or "-"
        print(f"{key:7s} {kind:7s} [{names:13s}] {'(無言)' if res[key]['silent'] else ''}{spoken[:100]}")

    only = set(filter(None, (args.only or "").split(",")))
    await asyncio.gather(*[one(*sc, rep) for sc in SCENES for rep in range(args.reps)
                           if not only or sc[0] in only])
    dump(path, res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--only", default="", help="場面の id をカンマ区切りで（その場面だけ呼ぶ）")
    ap.add_argument("--fresh", default="", help="この id の記録を消してから呼ぶ（カンマ区切り）")
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
