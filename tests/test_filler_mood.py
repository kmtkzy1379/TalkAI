"""S8 フィラーの1段目を場面に寄せる（API・音声デバイス不要の決定論テスト）。

ユーザ 2026-09-27:「通常、あ、あー、え、えー、あの、その、んー、無言、特殊フィラー / 悩む、え、えー、うー、んー、無言、
特殊フィラー / 驚く、お / 悲しむ、う、無言 / 不満、えー、無言」「できればほとんど通常にさせて強いときのみその他感情にしたい」
「閾値0.9でいい。ただ1段目はそれでいいけど2段目以降は通常通りランダムでいい」「通常のえーは低め、不満のえーは高め」

  [1] 表: fillers.yaml の mood（語 → id・無言・特殊フィラーの組・知らない語や系は止める）
  [2] 1段目の抽選: 通常＝今の重みから う・うー・お を外す（特殊フィラーの組は表の重み）/ それ以外は等分 /
      悩む・悲しむ は え系の組を 5%・10%（残りを等分・ユーザ 2026-09-29）/ 知らない種類は通常
  [3] 不満の えー は別のバンクの音（語尾が上がる高い えー）
  [4] 判定: 通常以外が 0.9 以上の時だけその種類・後から始めた判定が勝つ・失敗は通常
  [5] Jev への問い合わせの形（ローカルの偽サーバ）
  [6] 再生係: 場面で選ぶのは1段目だけ・2段目以降は今までどおり
  [6b] 特殊フィラーの組（えーっと 等）は1つのフィラーとして1段目にも出る・不満の高い えー に付くことは無い
  [7] 配線

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_filler_mood.py
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import random
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from eve.clock import now_mono  # noqa: E402
from eve.filler import audio as fa  # noqa: E402
from eve.filler.bank import MOOD_DEFAULT, SILENCE, Clip, FillerBank, FillerTable, draw_first  # noqa: E402
from eve.filler.mood import JevClient, MoodJudge  # noqa: E402

_passed = 0
_failed = 0
SR = 48000


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}" + (f"  ({detail})" if detail else ""))


def tone(sec: float, f: float = 300.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def real_yaml() -> dict:
    return yaml.safe_load(open(os.path.join(REPO, "fillers.yaml"), encoding="utf-8"))


def t_table() -> None:
    print("\n[1] 表（fillers.yaml の mood）")
    t = FillerTable.from_yaml(Path(REPO) / "fillers.yaml")
    check("しきい値 0.9（ユーザ指定）・今の感情は渡さない", t.mood_threshold == 0.9 and t.mood_with_emotion is False)
    check("種類はユーザ指定の5つ", list(t.mood_first) == ["通常", "悩む", "驚く", "悲しむ", "不満"], f"{list(t.mood_first)}")
    check("通常＝あ・あー・え・えー・あの・その・んー・無言＋特殊フィラーの組すべて（う・うー・お は無い）",
          t.mood_first["通常"][:8] == ("a", "aa", "e", "ee", "ano", "sono", "nn", SILENCE)
          and set(t.mood_first["通常"][8:]) == set(t.special) and len(t.special) == 14, f"{t.mood_first['通常']}")
    check("驚く＝お / 悲しむ＝う・無言 / 不満＝えー・無言",
          t.mood_first["驚く"] == ("o",) and t.mood_first["悲しむ"] == ("u", SILENCE)
          and t.mood_first["不満"] == ("ee", SILENCE))
    check("悩む＝え・えー・うー・んー・無言（特殊フィラーは え系の組だけを mood_special で）",
          t.mood_first["悩む"] == ("e", "ee", "uu", "nn", SILENCE), f"{t.mood_first['悩む']}")
    e_fam = {"e+tto", "e+ttone", "ee+tto", "ee+ttone"}
    # ユーザ 2026-09-29「悩むと悲しむ両方に入れる、え系特殊フィラー確率は悩む5％、悲しむ10％」「４は均等でいい」
    check("悩む: え系の組 5%（4組で等分＝各 1.25%）",
          t.mood_special.get("悩む") == {c: 1.25 for c in e_fam}, f"{t.mood_special.get('悩む')}")
    check("悲しむ: え系の組 10%（各 2.5%）",
          t.mood_special.get("悲しむ") == {c: 2.5 for c in e_fam}, f"{t.mood_special.get('悲しむ')}")
    check("通常・驚く・不満 には場面の割合（mood_special）が無い",
          not ({"通常", "驚く", "不満"} & set(t.mood_special)), f"{list(t.mood_special)}")
    check("不満の えー は bank/eve の音", t.mood_word_bank == {("不満", "ee"): "eve"}, f"{t.mood_word_bank}")
    check("Jev に渡す説明がある", all(t.mood_desc.get(c) for c in t.mood_first))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "f.yaml"
        src = real_yaml()
        src["mood"]["categories"]["驚く"]["words"] = ["おお"]
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        try:
            FillerTable.from_yaml(f)
            ok = False
        except ValueError:
            ok = True
        check("表に無い語は黙って読み替えずに止める", ok)
        src = real_yaml()
        src["mood"]["categories"].pop("通常")
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        try:
            FillerTable.from_yaml(f)
            ok = False
        except ValueError:
            ok = True
        check("「通常」が無ければ止める", ok)
        src = real_yaml()
        src["mood"]["categories"]["悩む"]["special"] = {"お": 5.0}
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        try:
            FillerTable.from_yaml(f)
            ok = False
        except ValueError:
            ok = True
        check("special に無い系を場面に書いたら止める（お系の組は無い）", ok)
        src = real_yaml()
        src.pop("mood")
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        check("mood が無ければ今までどおり（場面の表は空）", FillerTable.from_yaml(f).mood_first == {})


def t_draw() -> None:
    print("\n[2] 1段目の抽選")
    t = FillerTable.from_yaml(Path(REPO) / "fillers.yaml")
    rng = random.Random(0)

    def dist(mood, n=20000):
        c = collections.Counter(p.word_id or SILENCE for p in (draw_first(t, rng, mood) for _ in range(n)))
        return {k: v / n for k, v in c.items()}

    e_fam = {"e+tto", "e+ttone", "ee+tto", "ee+ttone"}
    d = dist("通常", 100_000)
    check("通常: う・うー・お は出ない（う・うー の組 うんっと 等は特殊フィラーとして出る）",
          not ({"u", "uu", "o"} & set(d)), f"{d}")
    # 特殊フィラー 5% の分だけ無言と語を縮めた（ユーザ 2026-09-29「按分するから変わらない」）: 通常の選択肢の重みの合計
    # 99.392（う・うー・お の分 0.608 を除く）の中で あ 36.37・無言 19・組 5
    check("通常: 今の重みのまま（あ ≈ 36.6%・無言 ≈ 19.1%）", abs(d.get("a", 0) - 0.3659) < 0.01
          and abs(d.get(SILENCE, 0) - 0.1912) < 0.01, f"{d}")
    sp = sum(v for k, v in d.items() if k in t.special)
    check("通常: 特殊フィラーの組 ≈ 5.0%（全部の組・表の重み）", abs(sp - 0.0503) < 0.004
          and {k for k in d if "+" in k} == set(t.special), f"{sp:.4f}")
    check("驚く: お だけ", dist("驚く") == {"o": 1.0})
    d = dist("悲しむ", 100_000)
    check("悲しむ: え系の組 10%（各 2.5%）・う と無言 各 45%",
          set(d) == {"u", SILENCE} | e_fam and abs(sum(d[c] for c in e_fam) - 0.10) < 0.005
          and all(abs(d[c] - 0.025) < 0.003 for c in e_fam)
          and abs(d["u"] - 0.45) < 0.01 and abs(d[SILENCE] - 0.45) < 0.01, f"{d}")
    d = dist("不満")
    check("不満: えー と無言を等分（組は出ない）", set(d) == {"ee", SILENCE} and abs(d["ee"] - 0.5) < 0.02, f"{d}")
    d = dist("悩む", 100_000)
    check("悩む: え系の組 5%（各 1.25%）・残り5つを等分（各 19%）",
          set(d) == {"e", "ee", "uu", "nn", SILENCE} | e_fam and abs(sum(d[c] for c in e_fam) - 0.05) < 0.004
          and all(abs(d[c] - 0.0125) < 0.002 for c in e_fam)
          and all(abs(d[k] - 0.19) < 0.01 for k in ("e", "ee", "uu", "nn", SILENCE)), f"{d}")
    check("知らない種類は通常", not ({"u", "uu", "o"} & set(dist("なにか"))))
    silence = draw_first(t, random.Random(3), "悲しむ")
    while silence.kind != "silence":
        silence = draw_first(t, rng, "悲しむ")
    check("無言の長さは今の範囲（0.3〜0.5秒）", 0.3 <= silence.sec <= 0.5, f"{silence.sec}")


def t_mood_clips() -> None:
    print("\n[3] 不満の えー は別のバンクの音")
    t = FillerTable(weights={"a": 50.0, "ee": 10.0}, texts={"a": "あ", "ee": "えー"}, join_gap_sec=0.01,
                    mood_first={"通常": ("a", "ee"), "不満": ("ee",)}, mood_word_bank={("不満", "ee"): "eve"})
    clips = {w: [Clip(w, t.texts[w], tone(0.06), SR)] for w in t.weights}
    high = {("不満", "ee"): [Clip("ee", "えー", tone(0.15, 600), SR)]}
    b = FillerBank(t, clips, {}, SR, mood_clips=high)
    rng = random.Random(0)
    check("不満の1段目の えー は高い えー", b.pick("ee", rng, mood="不満")[1].sec > 0.1)
    check("通常・種類なしの えー は今のバンク（低め）", b.pick("ee", rng, mood="通常")[1].sec < 0.1
          and b.pick("ee", rng)[1].sec < 0.1)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "eve_calm"
        other = Path(d) / "eve"
        for dd, sec in ((base, 0.06), (other, 0.15)):
            dd.mkdir()
            man = {"clips": []}
            for w in ("a", "ee"):
                f = f"{w}_0.wav"
                (dd / f).write_bytes(fa.to_wav(tone(sec), SR))
                man["clips"].append({"file": f, "id": w, "text": t.texts[w], "role": ["independent"]})
            import json

            (dd / "manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
        loaded = FillerBank.load(t, base)
        check("読み込み: 不満の えー を bank/eve から読む", ("不満", "ee") in loaded.mood_clips
              and loaded.mood_clips[("不満", "ee")][0].sec > 0.1)


async def t_judge() -> None:
    print("\n[4] 判定")
    answers = {"a、": {"通常": 0.05, "驚く": 0.95}, "b、": {"通常": 0.2, "驚く": 0.8},
               "c、": {"通常": 0.95, "悲しむ": 0.03}, "d、": {"通常": 0.01, "不満": 0.9}}
    seen: list = []

    async def judge(ctx, said, emo):
        seen.append((tuple(ctx), said, emo))
        if said == "boom、":
            raise RuntimeError("Jev")
        return answers[said]

    j = MoodJudge(judge_fn=judge, context_fn=lambda: ["ユーザ: 前の話"], categories=["通常", "悩む", "驚く", "悲しむ", "不満"],
                  threshold=0.9)
    check("判定が無い区間は通常", j.mood(1) == MOOD_DEFAULT)
    j.on_phrase(1, "a、")
    await asyncio.sleep(0.01)
    check("0.9 以上なら その種類", j.mood(1) == "驚く")
    check("直前の会話を渡す・今の感情は渡さない（既定）", seen[0] == (("ユーザ: 前の話",), "a、", None), f"{seen}")
    j.on_phrase(2, "b、")
    await asyncio.sleep(0.01)
    check("0.9 未満は通常（ほとんど通常）", j.mood(2) == MOOD_DEFAULT)
    j.on_phrase(3, "c、")
    await asyncio.sleep(0.01)
    check("通常が1位なら通常", j.mood(3) == MOOD_DEFAULT)
    j.on_phrase(3, "d、")
    await asyncio.sleep(0.01)
    check("同じ区間の後の判定で更新（不満 0.9）", j.mood(3) == "不満")
    j.on_phrase(3, "boom、")
    await asyncio.sleep(0.01)
    check("失敗しても前の判定のまま（落ちない）", j.mood(3) == "不満")
    j.on_phrase(3, "d、")
    await asyncio.sleep(0.01)
    check("同じ区間の同じ文字は2回聞かない", sum(1 for x in seen if x[1] == "d、") == 1)

    gate = asyncio.Event()

    async def slow(ctx, said, emo):
        if said == "遅い、":
            await gate.wait()
            return {"通常": 0.02, "驚く": 0.98}
        return {"通常": 0.99}

    j2 = MoodJudge(judge_fn=slow, context_fn=lambda: [], categories=["通常", "驚く"], threshold=0.9,
                   emotion_fn=lambda: "楽")
    j2.on_phrase(5, "遅い、")   # 先に始めた判定が遅れて届く
    await asyncio.sleep(0.01)
    j2.on_phrase(5, "遅い、次。")
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.sleep(0.01)
    check("後から始めた判定が勝つ（古い判定が遅れて届いても上書きしない）", j2.mood(5) == MOOD_DEFAULT)
    await j2.stop()


async def t_client() -> None:
    print("\n[5] Jev への問い合わせ（ローカルの偽サーバ）")
    from aiohttp import web

    got: list = []

    async def handler(req):
        got.append((req.headers.get("Authorization"), await req.json()))
        return web.json_response({"answers": {"filler": {"choice": "驚く",
                                                         "probabilities": {"通常": 0.08, "驚く": 0.92}}}})

    async def fail(req):
        return web.json_response({"error": "x"}, status=500)

    app = web.Application()
    app.router.add_post("/ok", handler)
    app.router.add_post("/ng", fail)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        cats = {"通常": "普通に受け止める", "驚く": "驚く・感心する"}
        c = JevClient("KEY", cats, url=f"http://127.0.0.1:{port}/ok")
        probs = await c.ask(["ユーザ: 前"], "うそ、", None)
        check("確率を返す", probs == {"通常": 0.08, "驚く": 0.92}, f"{probs}")
        auth, body = got[-1]
        check("鍵は Authorization で送る", auth == "Bearer KEY")
        check("問いの形（choice・説明・Jev のモデル）", body["model"] == "typesafe/jev-1.13"
              and body["questions"]["filler"]["type"] == "choice" and body["questions"]["filler"]["criteria"] == cats)
        check("state: 直前の会話と相手の発話（今の感情は無し）",
              body["state"] == {"直前の会話": ["ユーザ: 前"], "相手の発話": "うそ、"}, f"{body['state']}")
        await c.ask([], "うそ、", "楽")
        check("今の感情を渡す設定の時だけ state に入る", got[-1][1]["state"].get("イブの今の感情") == "楽")
        await c.close()
        c2 = JevClient("KEY", cats, url=f"http://127.0.0.1:{port}/ng")
        try:
            await c2.ask([], "x", None)
            ok = False
        except Exception:
            ok = True
        check("失敗は例外（判定側が通常にする）", ok)
        await c2.close()
    finally:
        await runner.cleanup()


async def t_player() -> None:
    print("\n[6] 再生係: 場面で選ぶのは1段目だけ")
    from eve.filler.player import FillerPlayer
    from eve.pipeline.audio_play_queue import AudioPlayQueue

    t = FillerTable(weights={"a": 50.0, "aa": 30.0, "o": 0.08, "ee": 5.0}, texts={"a": "あ", "aa": "あー", "o": "お", "ee": "えー"},
                    silence_weight=0.0, silence_sec=(0.02, 0.03), stretch_prob=0.0,
                    stretchable={}, max_chain_sec=5.0, min_utterance_sec=0.6, join_gap_sec=0.01,
                    mood_first={"通常": ("a", "aa", "ee"), "驚く": ("o",), "不満": ("ee",)})
    clips = {w: [Clip(w, t.texts[w], tone(0.05, 200 + 50 * i), SR) for _ in range(2)]
             for i, w in enumerate(t.weights)}
    high = {("不満", "ee"): [Clip("ee", "えー", tone(0.12, 700), SR)]}

    async def run(mood, seg=7):
        bank = FillerBank(t, clips, {}, SR, mood_clips=high)
        played: list = []

        async def play(wav):
            x, _ = fa.read_wav(wav)
            played.append(len(x) / SR)
            await asyncio.sleep(len(x) / SR)

        audio = AudioPlayQueue(play_fn=play)
        texts: list = []
        audio.on_sentence_start = lambda text, meta: texts.append(text)
        pl = FillerPlayer(bank, audio=audio, play_fn=play, is_busy=lambda: False, rng=random.Random(1))
        if mood is not None:
            pl.mood_fn = lambda s: mood if s == seg else MOOD_DEFAULT
        pl.start()
        pl.on_speech_end(1.5, seg)
        await asyncio.sleep(0.35)
        pl.on_response_near()
        await asyncio.sleep(0.15)
        await pl.stop()
        return texts, played

    texts, _ = await run("驚く")
    check("驚く: 1段目は お", texts[:1] == ["お"], f"{texts}")
    check("驚く: 2段目以降は今までどおりの抽選（お だけが続かない）", len(texts) >= 3 and any(x != "お" for x in texts[1:]),
          f"{texts}")
    texts, played = await run("不満")
    check("不満: 1段目は えー（高い えー の音）", texts[:1] == ["えー"] and played[0] > 0.1, f"{texts} {played[:2]}")
    check("不満: 2段目以降の えー は今のバンク（低め）", all(p < 0.1 for p, x in zip(played[1:], texts[1:])),
          f"{list(zip(texts, [round(p, 2) for p in played]))}")
    texts, _ = await run("通常")
    check("通常: 1段目に お は出ない（う・うー・お は通常に無い）", texts[:1] != ["お"], f"{texts}")
    firsts = collections.Counter()
    for k in range(12):
        tx, _ = await run(None, seg=20 + k)
        firsts[tx[0] if tx else ""] += 1
    check("場面の判定が無い（機能オフ）なら今までどおり", sum(firsts.values()) == 12, f"{firsts}")


async def t_special() -> None:
    print("\n[6b] 特殊フィラーの組は1つのフィラーとして1段目にも出る・不満の高い えー に付くことは無い")
    from eve.filler.player import FillerPlayer
    from eve.pipeline.audio_play_queue import AudioPlayQueue

    # 特殊フィラーは語と っと・っとね をまとめて合成した組（2026-09-29・例 えーっと）で、独立したフィラーとして抽選する
    # （ユーザ「今回の方針で独立することが決まった」）。語の後ろに付ける仕組みは無い＝高い えー に付くことも無い
    t = FillerTable(weights={"a": 50.0, "ee": 10.0, "ee+tto": 5.0}, texts={"a": "あ", "ee": "えー", "ee+tto": "えーっと"},
                    silence_weight=0.0, silence_sec=(0.02, 0.03), stretch_prob=0.0,
                    stretchable={}, special={"ee+tto": "え"}, max_chain_sec=5.0, min_utterance_sec=0.6,
                    join_gap_sec=0.01, mood_first={"通常": ("ee",), "不満": ("ee",), "悩む": ()},
                    mood_special={"悩む": {"ee+tto": 100.0}})
    clips = {w: [Clip(w, t.texts[w], tone(0.05), SR)] for w in t.weights}
    high = {("不満", "ee"): [Clip("ee", "えー", tone(0.12, 700), SR)]}

    async def run(mood):
        bank = FillerBank(t, clips, {}, SR, mood_clips=high)
        played: list = []

        async def play(wav):
            x, _ = fa.read_wav(wav)
            played.append(len(x) / SR)
            await asyncio.sleep(len(x) / SR)

        audio = AudioPlayQueue(play_fn=play)
        texts: list = []
        audio.on_sentence_start = lambda text, meta: texts.append(text)
        pl = FillerPlayer(bank, audio=audio, play_fn=play, is_busy=lambda: False, rng=random.Random(2))
        pl.mood_fn = lambda s: mood
        pl.start()
        pl.on_speech_end(1.5, 3)
        await asyncio.sleep(0.2)
        pl.on_response_near()
        await asyncio.sleep(0.15)
        await pl.stop()
        return texts, played

    tx, _ = await run("悩む")
    check("悩む: 1段目に組（えーっと）が1つのフィラーとして出る（単独の っと は鳴らさない）",
          tx[:1] == ["えーっと"] and "っと" not in tx, f"{tx}")
    tx, played = await run("不満")
    check("不満: 1段目は高い えー だけ（組が付いた えーっと にはならない）",
          tx[:1] == ["えー"] and played and played[0] > 0.1, f"{tx} {played[:1]}")


def t_wiring() -> None:
    print("\n[7] 配線")
    from eve.config import Config
    from eve.voice_loop import VoiceLoop

    saved = (Config.FILLER_MOOD_ENABLED, Config.OPENROUTER_API_KEY)
    try:
        Config.FILLER_MOOD_ENABLED = False
        vl = VoiceLoop()
        check("FILLER_MOOD_ENABLED=0 なら場面の判定は無い（今までどおりの抽選）",
              vl.mood is None and (vl.filler is None or vl.filler.mood_fn is None))
        Config.FILLER_MOOD_ENABLED, Config.OPENROUTER_API_KEY = True, ""
        vl = VoiceLoop()
        check("鍵が無ければ無効（落ちない）", vl.mood is None)
        Config.OPENROUTER_API_KEY = "dummy"
        vl = VoiceLoop()
        if vl.filler is None or not getattr(vl.stt, "streaming", False):
            print("SKIP フィラーかストリーミング文字起こしが無効")
            return
        check("有効: 再生係の1段目に配線", vl.mood is not None and vl.filler.mood_fn == vl.mood.mood)
        got: list = []
        vl.mood.on_phrase = lambda seg, text: got.append((seg, text))
        for j in (vl.laugh, vl.emotion_judge):
            if j is not None:
                j.on_phrase = lambda seg, text: None
        vl.stt.on_phrase(6, "うそ、")
        check("途中の文字を場面の判定へも配る", got == [(6, "うそ、")], f"{got}")
        check("しきい値は fillers.yaml（0.9）", vl.mood.threshold == 0.9)
    finally:
        Config.FILLER_MOOD_ENABLED, Config.OPENROUTER_API_KEY = saved


async def main() -> None:
    # 本物の fillers.yaml は手元だけ（.gitignore・2026-10-06）。無い環境では本物の表を見るテストを飛ばす
    if os.path.isfile(os.path.join(REPO, "fillers.yaml")):
        t_table()
        t_draw()
    else:
        print("SKIP 本物の fillers.yaml が無い（手元だけの表）: 表・1段目の抽選")
    t_mood_clips()
    for fn in (t_judge, t_client, t_player, t_special):
        try:
            await asyncio.wait_for(fn(), 30.0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} が例外なく終わる", False, f"{type(e).__name__}: {e}")


asyncio.run(main())
t_wiring()
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
