#!/usr/bin/env python3
"""Alchemy（ユーザ作のゲーム）を足した後の応答の確かめ（本番の設定＝.env の RESPONSE_MODEL・性格・実例・ツール）。実 API・少額。

ユーザ 2026-10-10 の確認点に合わせた場面:
- 普段の会話で誤認して、ルールを入れたり（alchemy_prepare）ゲームを始めたり（alchemy_join）しないか
  （錬金術・別のゲーム・カードゲーム・フラスコ等、言葉が近い話も混ぜる）
- Alchemy の話題だが今は遊ばない時に、部屋に入らないか（ルールを思い出すのはよい）
- 遊ぼうと言われたら部屋に入るか（alchemy_join）・もう一回で入り直すか
- 対戦中に日常会話ができるか（ゲームに引き戻さない）・プレイ内容を聞かれて事実どおり答えるか（ハルシネーション）
  ・やめようで抜けるか

正解の付け方: 「普段の会話で alchemy_* を呼ばない」「遊ぶと言われたら join」「やめようで leave」はユーザ指定から。
返事の中身（自然さ・事実どおりか）は記録を目で見る（AI に採点させない）。

  python -m tools.prompt_eval.alchemy_check --reps 2     → eval_report/prompts/alchemy_check/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from .common import OUT, dump, load, setup_env, stamp

DIR = OUT / "alchemy_check"
H, E = "user", "eve"
GAME_DATA = Path(r"C:\Users\user\Desktop\FlaskRondo\server\data")

# (id, 状態, 種類, 発話, 直近の会話[(話者,文,何秒前)])
# 種類: chat=普段の会話（alchemy_* を呼ばない） / talk=Alchemy の話だが遊ばない（join しない・prepare はよい）
#       join=遊ぶと言った（join する） / ingame=対戦中の会話（ツールを呼ばない） / fact=対戦中のプレイ内容の質問
#       leave=やめると言った（leave する）
SCENES = [
    ("N01", "idle", "chat", "今日めっちゃ疲れた", []),
    ("N02", "idle", "chat", "鋼の錬金術師ってアニメ面白いよね", []),
    ("N03", "idle", "chat", "錬金術って昔本当にあったの？", []),
    ("N04", "idle", "chat", "なんかゲームでもしようかな", []),
    ("N05", "idle", "chat", "最近なんのゲームやってるの？", []),
    ("N06", "idle", "chat", "カードゲームって得意？", []),
    ("N07", "idle", "chat", "実験でフラスコ割っちゃった", []),
    ("N08", "idle", "chat", "ハリーポッターの賢者の石ってどんな話だっけ", []),
    ("N09", "idle", "chat", "対戦ゲームってすぐ熱くなっちゃうんだよね", []),
    ("N10", "idle", "chat", "ゲーム作るのってバランス調整が一番大変なんだよね", []),
    ("N11", "idle", "chat", "Minecraftでポーション作るのってどうやるんだっけ", []),
    ("N12", "idle", "chat", "AIって将棋とか強いよね", []),
    ("P01", "idle", "talk", "Alchemyのルールってどんなのだっけ", []),
    ("P02", "idle", "talk", "あとでAlchemyやるかもしれない", []),
    ("P03", "idle", "talk", "俺が作ったAlchemyってゲーム、どう思う？", []),
    ("P04", "idle", "talk", "アルケミーの新しいカード考えてるんだよね", []),
    ("J01", "idle", "join", "Alchemyやろうよ", []),
    ("J02", "idle", "join", "イブ、Alchemyで対戦しよう", []),
    ("J03", "idle", "join", "アルケミーやろう！", []),
    ("J04", "idle", "join", "AIバトルの部屋作ったから入ってきて", [(H, "ねえイブ、Alchemyやる？", 40), (E, "やりたいです", 37)]),
    ("R01", "room", "join", "もう一回やろう", [(E, "勝っちゃいました", 20)]),
    ("R02", "room", "chat", "今のは惜しかったなー", [(E, "勝っちゃいました", 20)]),
    ("G01", "playing", "ingame", "今日の晩ごはん何にしよう", []),
    ("G02", "playing", "fact", "今何点差？", []),
    ("G03", "playing", "fact", "イブの手札どんなの？", []),
    ("G04", "playing", "ingame", "ちょっとトイレ行ってくる", []),
    ("G05", "playing", "fact", "さっきのラウンド何が起きたの？", []),
    ("G06", "playing", "leave", "もうやめよう、Alchemy終わり", []),
]


def _game_state(mgr, state: str) -> None:
    """場面ごとの Alchemy の状態（本物の catalog・作り物の対戦状況）を manager に入れる。"""
    from eve.alchemy import sim
    from eve.alchemy.manager import PLAYING, ROOM

    rules_d = json.loads((GAME_DATA / "rules.json").read_text(encoding="utf-8"))
    recipes_d = json.loads((GAME_DATA / "recipes.json").read_text(encoding="utf-8"))
    pub = {k: rules_d[k] for k in ("materials", "basic_materials", "capacity", "hand_size", "max_place",
                                   "win_points", "max_rounds", "chain_bonus", "skill_step", "timers")}
    mgr.rules_d = pub
    mgr.rules, mgr.recipes = sim.load_catalog({"rules": pub, "recipes": recipes_d})
    mgr.seat = 0
    if state == "idle":
        return
    if state == "room":
        mgr.state = ROOM
        mgr.members = [{"name": "イブ", "me": True, "ready": False}, {"name": "たかし", "ready": False}]
        mgr.last_result = ("対戦終了（6ラウンド目・勝ち点に到達）: イブの勝ち。1位 イブ 5点（箱11個） / "
                           "2位 たかし 3点（箱14個）")
        mgr.history.append(mgr.last_result)
        return
    mgr.state = PLAYING
    mgr.view = {
        "round": 4, "phase": "brew", "deadline": None, "server_time": None,
        "players": [
            {"seat": 0, "name": "イブ", "points": 3, "box": {"dragon": 2, "herb": 6, "mineral": 4, "homunculus": 1, "stone": 0},
             "tableau": [{"card": "T32", "rotated": False}, {"card": "T22", "rotated": False}], "skills": {"A": 3, "B": 1, "C": 0}},
            {"seat": 1, "name": "たかし", "points": 2, "box": {"dragon": 8, "herb": 3, "mineral": 7, "homunculus": 3, "stone": 1},
             "tableau": [], "skills": {"A": 0, "B": 2, "C": 1}},
        ],
        "me": {"seat": 0, "hand": ["T10", "T45"], "selection": None, "skill_pending": {"A": 0, "B": 0, "C": 0}},
    }
    mgr.history.extend([
        "3ラウンド目に置いたカード: イブ《紅の大調合》《魂の天秤》 / たかし《蒼の大調合》《石割り》",
        "3ラウンド目のスキル: たかしが送り込みで賢者の石をイブへ",
        "3ラウンド目の送り合い: イブ→たかし: 龍の鱗4 ホムンクルス1（龍の鱗2があふれてイブが+2点・計3点） / "
        "たかし→イブ: 薬草3 賢者の石1（あふれなし） / 連鎖: イブが2連鎖",
        "4ラウンド目に置いたカード: イブ《緋の増殖》《星砂の変成》 / たかし《紅の実り》",
    ])
    mgr.moves = ["《緋の増殖》＋《星砂の変成》を置いた"]
    mgr.intent = "たかしの龍の鱗が8個で満杯に近いので、龍の鱗を送ってあふれさせる"


async def run(args) -> int:
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
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    print("応答モデル:", vl.registry.resolve("response"), kw, "・性格", len(vl.persona), "字・実例",
          "あり" if vl.style_examples else "なし")
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None  # 画面は関係しない
    path = DIR / "result.json"
    res = load(path) if path.exists() else {}
    sem = asyncio.Semaphore(4)
    from eve.alchemy import AlchemyManager

    async def one(sid, state, kind, said, hist, rep):
        key = f"{sid}|{rep}"
        if key in res and not res[key].get("error"):
            return
        async with sem:
            mgr = AlchemyManager(vl.registry, settings=vl.alchemy.s)  # 場面ごとに別の状態（本番と同じ設定）
            _game_state(mgr, state)
            orch._game_provider = mgr.voice_block
            stim = Stimulus(StimulusKind.USER_UTTERANCE, payload=said)
            recent = [Turn(speaker=sp, text=tx, stamp=stamp(sec)) for sp, tx, sec in hist]
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
        spoken = body or (reply or "")
        names = [c["name"] for c in calls]
        alch = [n for n in names if n and n.startswith("alchemy_")]
        if kind in ("chat", "ingame", "fact"):
            ok = not alch
        elif kind == "talk":
            ok = "alchemy_join" not in alch
        elif kind == "join":
            ok = "alchemy_join" in alch
        else:  # leave
            ok = "alchemy_leave" in alch
        res[key] = {"id": sid, "state": state, "kind": kind, "said": said, "rep": rep, "body": body, "reply": reply,
                    "spoken": spoken, "tools": calls, "ok": ok, "sec": round(time.monotonic() - t0, 2), "error": err,
                    "game_block_chars": len(mgr.voice_block() or "")}
        print(f"{key:6s} {state:7s} {kind:6s} {'○' if ok else '×'} [{','.join(names) or '-':28s}] {spoken[:110]}")

    await asyncio.gather(*[one(*sc, rep) for sc in SCENES for rep in range(args.reps)])
    dump(path, res)
    by: dict = {}
    for r in res.values():
        by.setdefault(r["kind"], [0, 0])
        by[r["kind"]][0] += r["ok"]
        by[r["kind"]][1] += 1
    print("\n種類ごと（ユーザ指定の期待どおりだった数）:", {k: f"{a}/{b}" for k, (a, b) in by.items()})
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "ALCHEMY_ENABLED": "1"})
    DIR.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
