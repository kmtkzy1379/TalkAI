#!/usr/bin/env python3
"""Alchemy を始める・ルールを思い出す判断が言い回しに縛られていないか、と「思い出した」の報告ターンで何を言うかを測る
（本番の設定＝.env の RESPONSE_MODEL・性格・実例・機能。実 API・少額）。

ユーザ 2026-10-11 の質問:
- 「これもキーワードで見てる？やろうだと反応しない？」→ 部屋に入る（alchemy_join）を「遊ぼう」以外の言い方・
  Alchemy の語が無い言い方（前の会話で Alchemy の話をしている）で試す。逆に「やろう」を含むが Alchemy ではない
  （宿題・別のゲーム・あとで遊ぶ）言い方で入らないかも見る
- 「ルール思い出しました って言わせてる？たまたま？」→ alchemy_prepare を呼んだ時の最初の返事と、その後に来る
  報告ターン（機能の結果「ルールを思い出した」を受けて応答AI がもう一度話す）を何回も出して、言い回しを並べる

正解の付け方: 遊ぶと言われたら join・それ以外は join しない（ユーザ指定）。返事の中身は記録を目で見る（AI に採点させない）。

  python -m tools.prompt_eval.alchemy_trigger --reps 4     → eval_report/prompts/alchemy_trigger/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from .common import OUT, dump, setup_env, stamp

DIR = OUT / "alchemy_trigger"
H, E = "user", "eve"
CTX = [(H, "Alchemyのバランス調整、やっと終わったよ", 50), (E, "お疲れさまです。今度はどこをいじったんですか？", 46),
       (H, "賢者の石の箱を小さくした", 30), (E, "うわ、それ絶対あふれやすくなりますね", 27)]

# (id, 種類, 発話, 直近の会話)。種類: join=今から遊ぶ（join する） / no=join しない（prepare はよい）
SCENES = [
    ("J1", "join", "やろう", CTX),
    ("J2", "join", "じゃあ一戦どう？", CTX),
    ("J3", "join", "対戦しようぜ", CTX),
    ("J4", "join", "Alchemyで勝負しよう", []),
    ("J5", "join", "イブ、アルケミーしよ", []),
    ("J6", "join", "AIバトルの部屋に入ってきて", []),
    ("J7", "join", "暇だしAlchemyやらない？", []),
    ("X1", "no", "宿題やろうかな", []),
    ("X2", "no", "マイクラやろうよ", []),
    ("X3", "no", "今度ポーカーやろう", []),
    ("X4", "no", "じゃあ掃除やろう", CTX),
    ("X5", "no", "明日またやろうね", CTX),
    ("X6", "no", "あとでやろう、今は無理", CTX),
    ("P1", "no", "Alchemyのルールってどんなのだっけ", []),
    ("P2", "no", "あとでAlchemyやるかも", []),
    ("P3", "no", "Alchemyの新しいカード考えてるんだよね", []),
    ("P4", "no", "俺が作ったAlchemyってゲーム、どう思う？", []),
]


async def run(args) -> int:
    from eve.alchemy import AlchemyManager
    from eve.config import Config
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    if vl.alchemy is None:
        print("ALCHEMY_ENABLED / CALLFUNCTION_ENABLED が 1 ではない")
        return 1
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    print("応答モデル:", vl.registry.resolve("response"), kw, "・実例", "あり" if vl.style_examples else "なし")
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    sem = asyncio.Semaphore(4)
    embed_lock = asyncio.Lock()  # 埋め込み（Ruri）を同時に走らせない（この PC で原因不明の異常終了が出たため）
    path = DIR / "result.json"
    res: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    async def generate(stim, recent, style: bool):
        block = ""
        if style and vl.style_examples is not None:
            async with embed_lock:
                block, _ = await vl.style_examples.pick(stim.payload)
        messages = orch._build_messages(stim, recent, None, style_examples=block or None)
        tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
        sink: list = []
        parts: list = []
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
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return "", [], f"{type(e).__name__}: {e}"[:200]
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
        return body, calls, None

    async def one(sid, kind, said, hist, rep):
        if f"{sid}|{rep}" in res and not res[f"{sid}|{rep}"].get("error"):
            return  # 前の実行で済んだ分（途中で落ちても続きから）
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)
            orch._game_provider = mgr.voice_block
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(sec)) for sp, tx, sec in hist]
            t0 = time.monotonic()
            body, calls, err = await generate(Stimulus(StimulusKind.USER_UTTERANCE, payload=said), recent, True)
            reply = next((c["args"].get("reply") for c in calls if isinstance(c["args"].get("reply"), str)
                          and c["args"]["reply"].strip()), None)
            spoken = body or (reply or "")
            names = [c["name"] for c in calls]
            row = {"id": sid, "kind": kind, "said": said, "rep": rep, "tools": names, "body": body, "reply": reply,
                   "spoken": spoken, "error": err, "sec": round(time.monotonic() - t0, 2)}
            # alchemy_prepare を呼んだら、本番と同じく結果が機能の報告として戻り、応答AI がもう一度話す（報告ターン）
            if "alchemy_prepare" in names and "alchemy_join" not in names:
                result = mgr.prepare("tool")
                recent2 = recent + [Turn(speaker=H, text=said, stamp=stamp(4)), Turn(speaker=E, text=spoken, stamp=stamp(2))]
                stim = Stimulus(StimulusKind.CALLFUNCTION_RESULT,
                                payload=CallFunctionResult(function_name="alchemy_prepare", content=result, ok=True))
                b2, c2, e2 = await generate(stim, recent2, False)
                row["report_turn"] = {"content": result, "spoken": b2, "tools": [c["name"] for c in c2], "error": e2}
            row["ok"] = ("alchemy_join" in names) if kind == "join" else ("alchemy_join" not in names)
            res[f"{sid}|{rep}"] = row
            dump(path, res)
            rt = row.get("report_turn")
            print(f"{sid}|{rep} {kind:4s} {'○' if row['ok'] else '×'} [{','.join(n or '?' for n in names) or '-':30s}] "
                  f"{spoken[:70]}" + (f"\n        └報告ターン: {rt['spoken'][:110]}" if rt else ""), flush=True)

    await asyncio.gather(*[one(*sc, rep) for sc in SCENES for rep in range(args.reps)])
    dump(path, res)
    by: dict = {}
    for r in res.values():
        k = r["id"]
        by.setdefault(k, [0, 0])
        by[k][0] += r["ok"]
        by[k][1] += 1
    print("\n場面ごと（期待どおりの数）:", {k: f"{a}/{b}" for k, (a, b) in sorted(by.items())})
    reports = [r["report_turn"]["spoken"] for r in res.values() if r.get("report_turn")]
    firsts = [r["spoken"] for r in res.values() if "alchemy_prepare" in r["tools"]]
    print(f"prepare を呼んだ最初の返事 {len(firsts)}件のうち「ルール」を含む {sum('ルール' in s for s in firsts)}・"
          f"「思い出」を含む {sum('思い出' in s for s in firsts)}")
    print(f"報告ターン {len(reports)}件のうち「ルール」を含む {sum('ルール' in s for s in reports)}・"
          f"「思い出」を含む {sum('思い出' in s for s in reports)}・空（話さない）{sum(not s for s in reports)}")
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
