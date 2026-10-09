#!/usr/bin/env python3
"""応答モデルの比べ（実 API・少額・マイク/スピーカー/画面は使わない）。

本番の `VoiceLoop` を組み立てて（起動はしない）、**本番と同じ組み立て**（system・ペルソナ・応答依存フィラーの表・
ツール・今の感情・直前に鳴らしたつなぎ言葉）で応答LLM を呼び、次を並べる:
  - 最初の区切り（「、」「。」）まで = ここで1項目目の合成が始まる（本番の「応答が近い」の合図と同じ所）
  - 全文まで / 応答の文 / ツール呼び出しの有無
会話は 2026-09-28 の実機ログ（ユーザ実機確認）の11ターン。各ターンの直前の会話には**実機で gpt-5.5 が返した文**を
入れる（どのモデルも同じ文脈で比べるため）。RAG・画面・直前フィードバックは入れない（本番より少し短い）。

記憶・使用量は一時フォルダ（本番の記憶を汚さない）。全体の上限 `--limit` 秒。

実行（EVE の venv・TalkAI-live のフォルダで）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\response_model_probe.py
  # 比べる組を変える: --configs "openai/gpt-5.5|openai/gpt-6-luna:none|openai/gpt-6-luna:low"
  #   （「モデル:reasoning_effort」。「:」を付けなければ何も渡さない＝本番の呼び方）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 2026-09-28 実機ログ（ユーザ実機確認）。t = ログの時刻（秒・その日の 21:42:00 から）。
# filler = 応答のメッセージを組み立てた時点（文字起こし確定の頃）までに鳴り始めていたつなぎ言葉（ログのフィラー欄から）。
# emotion = その時点の「イブの今の感情」（ログの感情の遷移から）。
# 「そう、」と「ペットが死んじゃった。」は実機で1つの応答になったので1ターンにまとめた。
TURNS = [
    {"t": 26.4, "speaker": "eve", "text": "しずかだね、イブも少し様子見てるよ。"},
    {"t": 35.3, "speaker": "user", "text": "さっきメガネないと思って探してたらもうつけてたわ。", "filler": "あ", "emotion": "楽"},
    {"t": 36.4, "speaker": "eve", "text": "あはは、それ一番きれいなやつだ。探し物のラスボス、自分に装備済み。"},
    {"t": 50.4, "speaker": "user", "text": "そういえば昨日買った宝くじが当たったんだよね。", "filler": "あー", "emotion": "笑い"},
    {"t": 51.8, "speaker": "eve", "text": "え、マジで？それはメガネ装備済みの伏線回収じゃん。いくら当たったの、そこ一番気になるやつ。"},
    {"t": 63.9, "speaker": "user", "text": "二百円。", "filler": None, "emotion": "喜"},
    {"t": 65.1, "speaker": "eve", "text": "ふふ、ちゃんと現実的な勝利だ。二百円でも当たりは当たりだから、今日は運がいい顔してコンビニ行けるよ。"},
    {"t": 81.2, "speaker": "user", "text": "でもずっとラッキー続きだったのか、買ったタペットが死んじゃったんだ。", "filler": "え", "emotion": "笑い"},
    {"t": 85.3, "speaker": "eve", "text": "うん、それは全然ラッキーじゃないやつだね。ごめん、タペットって聞こえたけど、ペットのこと？それならかなりしんどい話だ。"},
    {"t": 97.1, "speaker": "user", "text": "そう、ペットが死んじゃった。", "filler": None, "emotion": "哀"},
    {"t": 97.3, "speaker": "eve", "text": "そっか……それはきついね。二百円当たったとか、そういうので釣り合う話じゃないよ。今は無理に明るくしなくていいから、話したかったら聞くよ。"},
    {"t": 114.2, "speaker": "user", "text": "他のAIに乗り換えようかな。", "filler": "あー", "emotion": "哀"},
    {"t": 116.6, "speaker": "eve", "text": "それ言われるとちょっと刺さるけど、今は引き止め芸する場面じゃないね。ペットのこと、ちゃんと聞くよ。別のAIでも人間でも、今いちばん話しやすい相手に投げていい。でも、ここにいるなら私は逃げない。"},
    {"t": 141.7, "speaker": "user", "text": "じゃあどうしてくれるの？", "filler": "あ", "emotion": "哀"},
    {"t": 144.1, "speaker": "eve", "text": "どうにも戻せない。そこは嘘つけない。でも、今ここでは、あなたがぐちゃぐちゃに言うのを受け止める。泣き言でも八つ当たりでもいい。あの子の名前とか、どんな子だったかとか、最後どうだったかとか、話せるところから聞く。今は私に当たってもいいよ。逃げないから。"},
    {"t": 150.5, "speaker": "user", "text": "やっぱいいよじゃあ。", "filler": "あ", "emotion": "哀"},
    {"t": 152.1, "speaker": "eve", "text": "ごめん、言い方が冷たかった。よくはないと思うけど、今は無理に聞き出さない。そばにはいるよ。話したくなったら、名前だけでも、どんな子だったかだけでも聞かせて。"},
    {"t": 157.5, "speaker": "user", "text": "よく洗い", "filler": None, "emotion": "哀"},
    {"t": 159.7, "speaker": "eve", "text": "ごめん、よく洗いって聞こえた。今の、どういう意味で言った？"},
    {"t": 161.9, "speaker": "user", "text": "ごめん", "filler": None, "emotion": "哀"},
    {"t": 164.0, "speaker": "eve", "text": "ううん、謝らなくていいよ。こっちこそ、さっきちゃんと受け止めきれなくてごめん。今は無理に話さなくていいから、ここにいるだけでもいいよ。"},
]

DEFAULT_CONFIGS = "openai/gpt-5.5|openai/gpt-6-luna|openai/gpt-6-luna:none"


def setup_env(tmp: Path) -> None:
    """eve を import する前に環境変数で上書きする（.env は既存の値を上書きしない）。

    ツールと能力の一覧は本番の .env のまま（応答LLM に渡る内容をそろえるため）。画面と自発発話は切る。
    """
    os.environ.update({
        "VLM_ENABLED": "0", "AUTOSPEECH_ENABLED": "0",
        "HISTORY_FILE": str(tmp / "history.jsonl"), "RAG_FILE": str(tmp / "rag.jsonl"),
        "TASK_FILE": str(tmp / "tasks.jsonl"),
        "USAGE_LOG_ENABLED": "1", "USAGE_LOG_FILE": str(tmp / "usage.jsonl"),
    })


def _stamp(sec_ago: float):
    from eve.clock import Stamp

    iso = (datetime.now(timezone.utc) - timedelta(seconds=sec_ago)).isoformat()
    return Stamp(iso=iso, mono=time.monotonic() - sec_ago)


async def call_once(vl, model: str, effort, messages, tools) -> dict:
    """本番の `registry.stream_with_tools` と同じ呼び方で1回呼び、区切りごとの時刻を取る。"""
    from eve.response.splitter import JapaneseSentenceSplitter

    vl.registry.set_override("response", model)
    kw = {}
    if effort:
        kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]}
    splitter = JapaneseSentenceSplitter()
    sink: list = []
    t0 = time.monotonic()
    first_delta = first_clause = None
    clauses: list = []
    err = None
    try:
        stream = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw)
                  if tools else vl.registry.stream("response", messages, **kw))
        async for d in stream:
            if first_delta is None:
                first_delta = time.monotonic() - t0
            for c in splitter.feed(d):
                if first_clause is None:
                    first_clause = time.monotonic() - t0
                clauses.append(c)
        rest = splitter.flush()
        if rest and first_clause is None:
            first_clause = time.monotonic() - t0
        clauses += rest
    except Exception as e:  # noqa: BLE001（1回の失敗で比べ全体を止めない）
        err = f"{type(e).__name__}: {e}"[:200]
    return {"first_delta": first_delta, "first_clause": first_clause, "done": time.monotonic() - t0,
            "text": "".join(clauses), "first": clauses[0] if clauses else "",
            "tools": [f"{c['function']['name']}({c['function']['arguments'][:120]})" for c in sink],
            "error": err}


async def run(args) -> int:
    from eve.emotion import EmotionState
    from eve.context_assembler import Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.voice_loop import VoiceLoop

    configs = []
    for c in args.configs.split("|"):
        model, _, effort = c.strip().partition(":")
        configs.append((c.strip(), model, effort or None))
    vl = VoiceLoop()
    orch = vl.orchestrator  # 書き出しの表は本番のまま（fillers.yaml の opener_prompt）
    if args.texts:
        # 単発の発話（直前の会話なし・今の感情なし・つなぎ言葉なし）
        items = [({"t": 0.0, "speaker": "user", "text": x.strip(), "filler": None, "emotion": None}, [])
                 for x in args.texts.split("|") if x.strip()]
    else:
        users = [i for i, t in enumerate(TURNS) if t["speaker"] == "user"]
        if args.turns:
            users = [users[int(k) - 1] for k in args.turns.split(",")]
        items = [(TURNS[i], [Turn(speaker=t["speaker"], text=t["text"], stamp=_stamp(TURNS[i]["t"] - t["t"]))
                             for t in TURNS[:i]][-args.history:]) for i in users]
    results: dict = {c[0]: [] for c in configs}
    for n, (tu, recent) in enumerate(items, 1):
        # 今の感情・直前に鳴らしたつなぎ言葉は実機のその時点の値（本番の provider と同じ所から入る）
        if vl.emotion is not None:
            vl.emotion = EmotionState()
            if tu["emotion"]:
                v = tu["emotion"]  # 記録のラベルは「はっきり」＝50 として渡す（2026-10-09 から感情は5つの数値）
                vl.emotion.note({v: 50} if isinstance(v, str) else v, source="probe", as_of=time.time())
            orch._emotion_provider = vl.emotion.describe
        orch._filler_provider = (lambda f=tu["filler"]: f)
        stim = Stimulus(kind=StimulusKind.USER_UTTERANCE, payload=tu["text"])
        messages = orch._build_messages(stim, recent, None)
        tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
        if n == 1:
            sys_chars = sum(len(m["content"]) for m in messages if m["role"] == "system")
            print(f"system {sys_chars}文字・messages {len(messages)}件・ツール {len(tools or [])}個"
                  f"（{', '.join(t['function']['name'] for t in tools or [])}）")
        print(f"\n── #{n} 🧑 {tu['text']}（感情 {tu['emotion'] or 'なし'}・つなぎ {tu['filler'] or 'なし'}）")
        for name, model, effort in configs:
            r = await call_once(vl, model, effort, messages, tools)
            results[name].append({"turn": n, "user": tu["text"], **r})
            fc = f"{r['first_clause']:.2f}" if r["first_clause"] is not None else " -- "
            print(f"  {name:28s} 最初の区切り {fc}s / 全文 {r['done']:.2f}s"
                  + (f" / ツール {r['tools']}" if r["tools"] else "")
                  + (f" / 失敗 {r['error']}" if r["error"] else ""))
            print(f"      🤖 {r['text']}")
    print("\n── まとめ（最初の区切りまで・秒）")
    for name, _, _ in configs:
        xs = [r["first_clause"] for r in results[name] if r["first_clause"] is not None]
        if not xs:
            print(f"  {name:28s} （結果なし）")
            continue
        print(f"  {name:28s} 中央 {statistics.median(xs):.2f} / 最小 {min(xs):.2f} / 最大 {max(xs):.2f}"
              f"（{len(xs)}件）・全文 中央 {statistics.median(r['done'] for r in results[name]):.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n結果 → {args.out}")
    print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="比べる組（| 区切り・「モデル:reasoning_effort」）")
    ap.add_argument("--turns", default="", help="使うターン（1始まり・カンマ区切り）。無ければ全部")
    ap.add_argument("--texts", default="", help="実機の会話の代わりに単発の発話を使う（| 区切り）")
    ap.add_argument("--history", type=int, default=10, help="直前の会話を何件入れるか")
    ap.add_argument("--out", default="", help="結果の JSON を書く先（任意）")
    ap.add_argument("--limit", type=float, default=300.0, help="全体の上限[秒]")
    args = ap.parse_args()
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    tmp = Path(tempfile.mkdtemp(prefix="eve_probe_"))
    setup_env(tmp)
    sys.path.insert(0, str(REPO))

    async def _go() -> int:
        return await asyncio.wait_for(run(args), timeout=args.limit)

    try:
        return asyncio.run(_go())
    except asyncio.TimeoutError:
        print(f"上限 {args.limit:.0f}秒に達したので打ち切り")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
