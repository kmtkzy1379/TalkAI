"""S5 イブの今の感情（API 不要の決定論テスト）。

ユーザ裁定 2026-09-26:「、。ごとに感情を分析してその感情を現感情として保存、その感情によってirodoriの声トーンや
応答LLM,フィードバックLLMに入れたりして感情にそった応答をさせる。フィードバックでも感情分析してるから
フィードバックがきたらそれに更新。…（現感情を、。やフィードバックに入れれば大きく不自然な感情切り替わりは
少ないと予想）。感情ラベルは喜怒哀楽笑いでいい。」

2026-10-09（ユーザ）:「喜怒哀楽笑…に数値0〜100の数値をつけて起動終了時〜起動開始時までの時間で数値を減少させる。
…0が平常。50が明確にその感情を抱いてる。100がとても激しくその感情を抱いてる。最大合計数値100で割り振る。」

  [1] 今の感情: 喜怒哀楽笑いの5つの数値（各0〜100・合計100以下）・新しい情報が勝つ・判定した時点から時間で薄れる・保存と読み込み
  [2] 判定の質問文: 5つに数値を割り振らせる・今の感情を渡す
  [3] 「、」「。」ごとの判定: 更新・同じ文字は2回聞かない・遅れて届いた古い判定は捨てる・失敗は変えない
  [4] フィードバック: 今の感情を入力に添え、emotion_values で更新・判定の時点は振り返った会話の最後の発話（何日も前の会話は平常）
  [5] 応答LLM: 「# イブの今の気持ち」
  [6] 声のトーン: emotion.yaml の絵文字を応答の文に付ける（応答を組み立てた時点の感情で1ターン1つ・未決なら付けない）
  [8] フィラーの声: 感情ごとのバンク（絵文字が emotion.yaml と同じ時だけ）・チェーンを始めた時の感情で鳴らす
  [7] VoiceLoop の配線

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_emotion.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
logging.disable(logging.CRITICAL)

from eve import emotion as em  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}" + (f"  ({detail})" if detail else ""))


def t_state() -> None:
    print("\n[1] 今の感情（5つの数値・時間で薄れる）")
    now = [1_000_000.0]
    s = em.EmotionState(decay_per_hour=10, tone_min=30, now_fn=lambda: now[0])
    check("最初は平常（何も判定していない）", s.values() == {} and s.label is None and s.describe() is None
          and s.brief() == em.NEUTRAL)
    check("ラベルは喜怒哀楽笑いの5つ", list(em.LABELS) == ["喜", "怒", "哀", "楽", "笑い"])
    check("知らないラベルだけの判定は捨てる", s.note({"驚き": 80}, source="phrase", as_of=now[0]) is False
          and s.values() == {})
    check("1つのラベルの形（旧形式）は数値ではないので捨てる", s.note("哀", source="phrase", as_of=now[0]) is False)
    check("更新できる（細かく割り振った値をそのまま持つ）",
          s.note({"喜": 50, "笑い": 25, "楽": 5}, source="phrase", as_of=now[0])
          and s.values() == {"喜": 50, "笑い": 25, "楽": 5} and s.source == "phrase", f"{s.values()}")
    check("短い形（判定・フィードバックに渡す）は強い順", s.brief() == "喜 50・笑い 25・楽 5", s.brief())
    check("応答用の説明は数値と意味つき",
          s.describe().startswith("喜（嬉しい・喜んでいる）50・笑い（笑っている・可笑しがっている）25")
          and "0=平常" in s.describe() and "50=はっきり" in s.describe(), f"{s.describe()}")
    check("声に乗せるのはいちばん強い感情（しきい値以上）", s.dominant() == "喜" and s.label == "喜")
    check("合計が100を超えたら比率を保って100に縮める",
          em.normalize({"喜": 80, "笑い": 80}) == {"喜": 50.0, "怒": 0.0, "哀": 0.0, "楽": 0.0, "笑い": 50.0})
    check("各値は0〜100に収める・数字でない値は無視", em.normalize({"喜": -5, "哀": "x", "怒": 30})
          == {"喜": 0.0, "怒": 30.0, "哀": 0.0, "楽": 0.0, "笑い": 0.0})
    check("全部0は平常として有効（判定を捨てない）", em.normalize({"喜": 0}) is not None)
    # 時間で薄れる（1時間に10ずつ・0で止まる）
    now[0] += 2 * 3600
    check("2時間後: 20ずつ引いた値（弱い感情から消える）", s.values() == {"喜": 30, "笑い": 5}, f"{s.values()}")
    now[0] += 1 * 3600
    check("しきい値を下回ったら声には乗せない（普通の声）", s.values() == {"喜": 20} and s.dominant() is None,
          f"{s.values()}")
    now[0] += 10 * 3600
    check("十分たてば平常（応答に気持ちのブロックを出さない）", s.values() == {} and s.describe() is None
          and s.brief() == em.NEUTRAL)
    # 新しい情報が勝つ
    t0 = now[0]
    check("新しい時点の判定で更新", s.note({"哀": 60}, source="feedback", as_of=t0) and s.values() == {"哀": 60})
    check("古い時点の判定は上書きしない（新しい情報が勝つ）",
          s.note({"喜": 70}, source="feedback", as_of=t0 - 1) is False and s.values() == {"哀": 60})
    check("同じ時点なら後から来た方", s.note({"笑い": 40}, source="phrase", as_of=t0) and s.values() == {"笑い": 40})
    check("古い会話の判定は、その会話からの経過の分だけ薄れて入る（10日前→平常）",
          s.note({"哀": 80}, source="feedback", as_of=t0 + 1) and s.values(now=t0 + 1 + 240 * 3600) == {})

    # 保存と読み込み（起動をまたぐ）
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "emotion_state.json")
        clock = [5_000_000.0]
        a = em.EmotionState(path=path, decay_per_hour=10, now_fn=lambda: clock[0])
        a.note({"怒": 40, "笑い": 20}, source="phrase", as_of=clock[0])
        check("判定のたびに保存する", os.path.exists(path) and not os.path.exists(path + ".tmp"))
        clock[0] += 3 * 3600  # 3時間止めて起動し直した
        b = em.EmotionState(path=path, decay_per_hour=10, now_fn=lambda: clock[0])
        check("起動時に前回の感情を読み、止まっていた間の分だけ薄れている", b.values() == {"怒": 10}
              and b.source == "phrase", f"{b.values()}")
        check("読み込んだ後も、それより古い判定は上書きしない",
              b.note({"喜": 90}, source="feedback", as_of=clock[0] - 4 * 3600) is False)
        Path(path).write_text("{壊れ", encoding="utf-8")
        c = em.EmotionState(path=path, now_fn=lambda: clock[0])
        check("保存が壊れていても平常から始める（落ちない）", c.values() == {})
        check("保存先が無ければ平常から", em.EmotionState(path=os.path.join(d, "none.json")).values() == {})
    rel = em.EmotionState(path="emotion_state_relcheck_never_written.json")
    check("相対パスはリポジトリ基準", rel._path == Path(REPO) / "emotion_state_relcheck_never_written.json"
          and not rel._path.exists())


def t_prompt() -> None:
    print("\n[2] 判定の質問文（5つに数値を割り振る）")
    m = em.judge_messages(["ユーザ: 前の話"], "財布を落としちゃって。", "楽 30")
    sysmsg = m[0]["content"]
    check("5つの感情に 0〜100 で割り振らせる（数値の意味・合計100以下）",
          "喜 / 怒 / 哀 / 楽 / 笑い の5つに数値で割り振る" in sysmsg and em.SCALE in sysmsg and "合計は100以下" in sysmsg,
          sysmsg)
    check("出力形式は5つの数値", '"eve5": {"喜": 0, "怒": 0, "哀": 0, "楽": 0, "笑い": 0}' in sysmsg)
    check("今の感情を渡す", "イブの今の感情: 楽 30" in sysmsg)
    check("user は直前の会話＋相手の発話",
          m[1]["content"] == "直前の会話:\nユーザ: 前の話\n\n相手の発話: 財布を落としちゃって。")
    m0 = em.judge_messages([], "こんにちは。", None)
    check("今の感情が無ければ平常", f"イブの今の感情: {em.NEUTRAL}" in m0[0]["content"])
    check("会話が無ければ相手の発話だけ", m0[1]["content"] == "相手の発話: こんにちは。")
    check("eve5 の数値を読む", em.parse_eve5('{"eve5": {"喜": 40, "怒": 0, "哀": 0, "楽": 20, "笑い": 30}}')
          == {"喜": 40.0, "怒": 0.0, "哀": 0.0, "楽": 20.0, "笑い": 30.0})
    check("1つのラベルの形（旧形式）は読めない＝変えない",
          em.parse_eve5('{"eve5": {"label": "哀", "confidence": 80}}') is None)
    check("壊れた JSON は None", em.parse_eve5("eve5: 哀") is None and em.parse_eve5("") is None)


async def t_judge() -> None:
    print("\n[3] 「、」「。」ごとの判定")
    s = em.EmotionState()
    calls: list = []
    gates: dict = {}

    async def judge(ctx, said, current):
        calls.append((tuple(ctx), said, current))
        ev = gates.get(said)
        if ev is not None:
            await ev.wait()
        return {"えっとね、": {"楽": 40}, "財布をなくして、": {"哀": 60}, "やばい。": {"笑い": 70},
                "ぼろぼろ。": None}.get(said)

    j = em.EmotionJudge(state=s, judge_fn=judge, context_fn=lambda: ["ユーザ: 前の話"])
    j.on_phrase(1, "えっとね、")
    await asyncio.sleep(0.01)
    check("判定して今の感情を更新", s.values() == {"楽": 40} and s.source == "phrase", f"{s.values()}")
    check("直前の会話と今の感情（判定前の値）を渡す", calls[0] == (("ユーザ: 前の話",), "えっとね、", em.NEUTRAL),
          f"{calls}")
    j.on_phrase(1, "えっとね、")
    await asyncio.sleep(0.01)
    check("同じ区間の同じ文字は2回聞かない", len(calls) == 1)
    j.on_phrase(2, "財布をなくして、")
    await asyncio.sleep(0.01)
    check("次の判定には今の感情を渡す", calls[-1][2] == "楽 40" and s.values() == {"哀": 60},
          f"{calls[-1]} {s.values()}")
    # 先に始めた判定が遅れて届いても、後から始めた判定を上書きしない
    gates["やばい。"] = asyncio.Event()
    j.on_phrase(3, "やばい。")         # 先に始まる（遅れて届く）
    await asyncio.sleep(0.01)
    later = time.time() + 1
    s.note({"喜": 50}, source="phrase", as_of=later)  # 後から始めた判定が先に届いた
    gates["やばい。"].set()
    await asyncio.sleep(0.01)
    check("遅れて届いた古い判定は捨てる", s.values() == {"喜": 50}, f"{s.values()}")
    j.on_phrase(4, "ぼろぼろ。")
    await asyncio.sleep(0.01)
    check("読めない判定は今の感情のまま", s.values() == {"喜": 50})

    async def boom(ctx, said, current):
        raise RuntimeError("API")

    j2 = em.EmotionJudge(state=s, judge_fn=boom, context_fn=lambda: [])
    j2.on_phrase(5, "こわい。")
    await asyncio.sleep(0.01)
    check("判定の失敗は今の感情のまま（落ちない）", s.values() == {"喜": 50})

    hold = asyncio.Event()

    async def slow(ctx, said, current):
        await hold.wait()
        return {"楽": 30}

    j3 = em.EmotionJudge(state=em.EmotionState(), judge_fn=slow, context_fn=lambda: [], max_inflight=2)
    for k in range(4):
        j3.on_phrase(9, f"文{k}、")
    check("同時の判定は上限まで（混んでいれば見送る）", len(j3._tasks) == 2, f"{len(j3._tasks)}")
    hold.set()
    await asyncio.sleep(0.01)
    await j3.stop()

    # 本物の呼び出しの形（役・JSON・reasoning_effort=none）
    seen: dict = {}

    class Reg:
        async def complete(self, role, messages, **kw):
            seen.update(role=role, messages=messages, **kw)
            return {"choices": [{"message": {"content": '{"user6": {"label": "怒り", "confidence": 70}, '
                                                          '"eve5": {"喜": 0, "怒": 60, "哀": 0, "楽": 0, "笑い": 10}}'}}]}

    got = await em.make_judge_fn(Reg())(["ユーザ: x"], "ふざけないで。", "楽 30")
    check("判定の呼び出し: emotion_judge 役・JSON・reasoning_effort=none",
          seen.get("role") == "emotion_judge" and seen.get("response_format") == {"type": "json_object"}
          and seen.get("reasoning_effort") == "none"
          and "reasoning_effort" in (seen.get("allowed_openai_params") or []), f"{seen.keys()}")
    check("判定の呼び出し: eve5 の数値を返す", got == {"喜": 0.0, "怒": 60.0, "哀": 0.0, "楽": 0.0, "笑い": 10.0},
          f"{got}")

    # 2026-10-05 性格を感情判定にも渡す（無いと口の悪いいじりを「哀」と読み、応答の性格と食い違った）
    from eve.persona import PERSONA_HEADING
    persona = f"{PERSONA_HEADING}\n- 性格: テスト用の一行。{{波かっこ}}"
    await em.make_judge_fn(Reg(), persona=persona)(["ユーザ: x"], "ポンコツ。", None)
    sysmsg = seen["messages"][0]["content"]
    check("判定に性格を添える（見出しは判定用・応答用の見出しは使わない）",
          "# イブの性格（" in sysmsg and "テスト用の一行。{波かっこ}" in sysmsg and PERSONA_HEADING not in sysmsg, sysmsg[-120:])
    check("性格が無ければ従来どおり", "イブの性格" not in em.judge_messages([], "x", None)[0]["content"])


async def t_feedback() -> None:
    print("\n[4] フィードバック")
    from datetime import datetime, timedelta, timezone

    from eve.clock import Stamp
    from eve.context_assembler import Turn
    from eve.feedback.feedback_llm import FeedbackLLM
    from eve.feedback.parser import parse_feedback
    from eve.feedback.prompts import EMOTION_VALUES_RULE, FEEDBACK_SYSTEM, build_messages
    from eve.model_registry import ModelRegistry

    turns = [Turn("user", "財布なくした", Stamp.now()), Turn("eve", "えっ、大丈夫？", Stamp.now())]
    off = build_messages(turns, None)
    check("感情の機能が OFF なら従来と同じ文面", off[0]["content"] == FEEDBACK_SYSTEM
          and "イブの今の感情" not in off[1]["content"])
    on = build_messages(turns, None, with_emotion=True, current_emotion="哀 60")
    check("ON なら emotion_values（5つの数値）を聞く", on[0]["content"].endswith(EMOTION_VALUES_RULE)
          and "5つに数値で割り振る" in EMOTION_VALUES_RULE and em.SCALE in EMOTION_VALUES_RULE)
    check("ON なら今の感情を添える", on[1]["content"].endswith("# イブの今の感情\n哀 60"), on[1]["content"][-30:])
    on0 = build_messages(turns, None, with_emotion=True, current_emotion=None)
    check("今の感情が無ければ平常", on0[1]["content"].endswith(f"# イブの今の感情\n{em.NEUTRAL}"))

    r = parse_feedback("summary: 財布の話\nemotion: ほっとした\n"
                       "emotion_values: 喜=10, 怒=0, 哀=20, 楽=40, 笑い=0\nsurprise: 20")
    check("emotion_values を読む（emotion と混ざらない）",
          r.emotion_values == {"喜": 10.0, "怒": 0.0, "哀": 20.0, "楽": 40.0, "笑い": 0.0} and r.emotions == "ほっとした",
          f"{r.emotion_values!r} {r.emotions!r}")
    check("全角の区切り・笑の略・JSON 風でも読む",
          parse_feedback("emotion_values: 喜＝5、笑=30").emotion_values.get("笑い") == 30.0
          and parse_feedback('emotion_values: {"喜": 5, "笑い": 30}').emotion_values.get("喜") == 5.0)
    check("読めなければ空（感情を変えない）", parse_feedback("emotion_values: よくわからない").emotion_values == {}
          and parse_feedback("summary: x").emotion_values == {})

    out = "summary: ユーザは財布をなくした\nemotion: 心配\nemotion_values: 喜=0, 怒=0, 哀=60, 楽=0, 笑い=0\nsurprise: 30"
    captured: list = []

    async def fake(model, messages, **kw):
        captured.append(messages)
        return {"choices": [{"message": {"content": out}}]}

    st = em.EmotionState()
    st.note({"楽": 30}, source="phrase", as_of=time.time() - 60)
    fb = FeedbackLLM(ModelRegistry(completion_fn=fake), emotion_state=st)
    await fb.run(turns)
    check("フィードバックに今の感情を渡す", "# イブの今の感情\n楽 30" in captured[-1][1]["content"])
    check("フィードバックの emotion_values で更新", st.values() == {"哀": 60} and st.source == "feedback",
          f"{st.values()}")

    # 起動時の取りこぼし分（10日前の会話）の振り返り → 判定の時点は会話の時刻＝薄れて平常（2026-10-09 の実害）
    old = datetime.now(timezone.utc) - timedelta(days=10)
    old_turns = [Turn("user", "ごめん", Stamp(iso=(old - timedelta(seconds=5)).isoformat(), mono=0.0)),
                 Turn("eve", "こちらこそごめん", Stamp(iso=old.isoformat(), mono=0.0))]
    st_old = em.EmotionState()
    await FeedbackLLM(ModelRegistry(completion_fn=fake), emotion_state=st_old).run(old_turns)
    check("⭐10日前の会話の振り返りは「今の感情」にならない（その会話の時点で入り、薄れて平常）",
          st_old.values() == {} and abs(st_old.as_of - old.timestamp()) < 1 and st_old.describe() is None,
          f"{st_old.values()} {st_old.as_of}")

    # 振り返っている間にユーザの新しい発話で感情が変わった → 古い会話の振り返りでは上書きしない
    gate = asyncio.Event()

    async def slow(model, messages, **kw):
        await gate.wait()
        return {"choices": [{"message": {"content": out.replace("哀=60", "楽=60")}}]}

    st2 = em.EmotionState()
    fb2 = FeedbackLLM(ModelRegistry(completion_fn=slow), emotion_state=st2)
    task = asyncio.create_task(fb2.run(turns))
    await asyncio.sleep(0.01)
    st2.note({"怒": 50}, source="phrase", as_of=time.time())
    gate.set()
    await task
    check("振り返りの間に新しくなった感情は上書きしない", st2.values() == {"怒": 50}, f"{st2.values()}")

    fb3 = FeedbackLLM(ModelRegistry(completion_fn=fake))
    await fb3.run(turns)
    check("感情の機能が無ければ従来どおり（今の感情を入れない）",
          captured[-1][0]["content"] == FEEDBACK_SYSTEM and "イブの今の感情" not in captured[-1][1]["content"])

    # 2026-10-05 性格をフィードバックにも渡す（応答とちぐはぐにならないため・summary は事実だけ）
    from eve.persona import PERSONA_HEADING
    fb4 = FeedbackLLM(ModelRegistry(completion_fn=fake), persona=f"{PERSONA_HEADING}\n- 性格: テスト用の一行。")
    await fb4.run(turns)
    sysmsg = captured[-1][0]["content"]
    check("フィードバックに性格を添える（summary は事実だけ・冗談は冗談として）",
          sysmsg.startswith(FEEDBACK_SYSTEM) and "- 性格: テスト用の一行。" in sysmsg
          and "summary は事実だけ" in sysmsg and PERSONA_HEADING not in sysmsg, sysmsg[-120:])


def t_context() -> None:
    print("\n[5] 応答LLM")
    from eve.context_assembler import ContextAssembler

    ca = ContextAssembler(system_prompt="SYS")
    sys_on = ca.assemble(user_text="やあ", emotion="哀（悲しい・つらい）")[0]["content"]
    check("「# イブの今の気持ち」に今の感情を入れる",
          "# イブの今の気持ち\n哀（悲しい・つらい）。この気持ちに沿って話す" in sys_on, sys_on[-120:])
    check("自発発話にも入れる（イブの感情は刺激の種類に依らない）",
          "# イブの今の気持ち" in ca.assemble(autonomous_content="天気の話", emotion="楽（x）")[0]["content"])
    check("未判定なら入れない", "# イブの今の気持ち" not in ca.assemble(user_text="やあ")[0]["content"])


async def t_orchestrator() -> None:
    print("\n[5b] 応答の組み立てに今の感情が届く")
    from eve.pipeline.audio_play_queue import AudioPlayQueue
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.orchestrator import ResponseOrchestrator

    seen: list = []

    async def stream_fn(messages):
        seen.append(messages)
        yield "うん。"

    async def tts(s):
        return None

    async def play(a):
        return None

    audio = AudioPlayQueue(play_fn=play)
    st = em.EmotionState()
    st.note({"笑い": 60, "喜": 20}, source="phrase", as_of=time.time())
    orch = ResponseOrchestrator(audio, stream_fn, tts, emotion_provider=st.describe)
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "ねえ"))
    w.cancel()
    check("応答LLM の system に届く（数値つき）",
          "# イブの今の気持ち\n笑い（笑っている・可笑しがっている）60・喜（嬉しい・喜んでいる）20" in seen[0][0]["content"])


class _Cache:
    def __init__(self) -> None:
        self.turns: list = []

    def recent_for_injection(self):
        return []

    def add_turn(self, role, text):
        if text:
            self.turns.append((role, text))


async def _respond(tone_provider, tts_hook=None) -> tuple:
    from eve.pipeline.audio_play_queue import AudioPlayQueue
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.orchestrator import ResponseOrchestrator

    sent: list = []

    async def tts(s):
        sent.append(s)
        if tts_hook is not None:
            tts_hook(len(sent))
        return b"x"

    async def play(a):
        return None

    async def stream_fn(messages):
        yield "そっか、それはつらかったね。"

    audio = AudioPlayQueue(play_fn=play)
    cache = _Cache()
    orch = ResponseOrchestrator(audio, stream_fn, tts, conversation_cache=cache, tone_provider=tone_provider)
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "ねえ"))
    w.cancel()
    return sent, cache.turns


async def t_voice_tone() -> None:
    print("\n[6] 声のトーン（Irodori の絵文字・応答ごとに1つ）")
    st = em.EmotionState()
    st.note({"哀": 60}, source="phrase", as_of=time.time())
    table = {"哀": "😭", "楽": "😌"}
    reads: list = []

    def tone():
        reads.append(st.dominant())
        return table.get(st.dominant())

    def change(n):
        if n == 1:  # 1節目の合成中に、感情の判定が届いて「楽」になった
            st.note({"楽": 60}, source="phrase", as_of=time.time() + 1)

    sent, turns = await _respond(tone, change)
    check("TTS に渡す文字にだけ、今の感情の絵文字を付ける", sorted(sent) == sorted(["そっか、😭", "それはつらかったね。😭"]),
          f"{sent}")
    check("応答の途中で感情が変わっても、そのターンの声は1つ（組み立てた時点の感情・1回だけ読む）",
          reads == ["哀"], f"{reads}")
    check("記憶の文字には絵文字を付けない", ("eve", "そっか、それはつらかったね。") in turns, f"{turns}")
    sent, _ = await _respond(lambda: None)
    check("その感情の絵文字が未決なら付けない", sorted(sent) == sorted(["そっか、", "それはつらかったね。"]), f"{sent}")
    sent, _ = await _respond(None)
    check("声のトーンが無ければ今までと同じ", sorted(sent) == sorted(["そっか、", "それはつらかったね。"]), f"{sent}")

    def boom():
        raise RuntimeError("x")

    sent, _ = await _respond(boom)
    check("取得に失敗しても付けずに合成（落ちない）", sorted(sent) == sorted(["そっか、", "それはつらかったね。"]))

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "emotion.yaml"
        p.write_text("voice_emoji:\n  喜: null\n  哀: \"😭\"\n  驚き: \"😲\"\n", encoding="utf-8")
        check("表: 決まった感情だけ読む（知らない感情・null は無視）", em.load_voice_emoji(p) == {"哀": "😭"},
              f"{em.load_voice_emoji(p)}")
        p.write_text("voice_emoji: [", encoding="utf-8")
        check("表: 壊れていれば付けない", em.load_voice_emoji(p) == {})
        check("表: 無ければ付けない", em.load_voice_emoji(Path(d) / "none.yaml") == {})
    # 本物の表は中身（どの絵文字か）をユーザが決める＝テストは中身に依らず、読める形かだけ見る
    real = em.load_voice_emoji(Path(REPO) / "emotion.yaml")
    check("本物の表: 知っている感情だけ・絵文字は1〜2文字", set(real) <= set(em.LABELS)
          and all(isinstance(v, str) and 1 <= len(v) <= 3 for v in real.values()), f"{real}")


def t_wiring() -> None:
    print("\n[7] 配線")
    from eve.config import Config
    from eve.voice_loop import VoiceLoop

    saved = (Config.EMOTION_ENABLED, Config.EMOTION_TABLE, Config.EMOTION_STATE_FILE)
    tmp = tempfile.TemporaryDirectory()
    undecided = Path(tmp.name) / "emotion.yaml"
    undecided.write_text("voice_emoji:\n  喜: null\n", encoding="utf-8")
    Config.EMOTION_STATE_FILE = str(Path(tmp.name) / "emotion_state.json")  # 本物の保存を汚さない
    try:
        # 本物の emotion.yaml（ユーザが決める中身）に依らないよう、表は一時ファイルで与える
        Config.EMOTION_TABLE = str(undecided)
        Config.EMOTION_ENABLED = False
        vl = VoiceLoop()
        check("EMOTION_ENABLED=0 なら感情は無い（フィードバック・応答は従来どおり）",
              vl.emotion is None and vl.emotion_judge is None and vl.feedback._emotion is None
              and vl.orchestrator._emotion_provider is None)
        Config.EMOTION_ENABLED = True
        vl = VoiceLoop()
        check("EMOTION_ENABLED=1: 今の感情をフィードバックと応答に配線",
              vl.emotion is not None and vl.feedback._emotion is vl.emotion
              and vl.orchestrator._emotion_provider == vl.emotion.describe)
        check("EMOTION_ENABLED=1: 今の感情は保存先つき（起動をまたいで続く）",
              vl.emotion._path == Path(Config.EMOTION_STATE_FILE))
        from eve.persona import load_persona
        check("性格を応答・フィードバックで同じものにする（persona.md）",
              vl.persona == load_persona(os.getenv("PERSONA_FILE")) and vl.feedback._persona == vl.persona)
        streaming = bool(getattr(vl.stt, "streaming", False))
        check("ストリーミング文字起こしなら「、」「。」ごとの判定を作る",
              (vl.emotion_judge is not None) == streaming)
        if streaming:
            got: list = []
            vl.emotion_judge.on_phrase = lambda seg, text: got.append(("emotion", seg, text))
            if vl.laugh is not None:
                vl.laugh.on_phrase = lambda seg, text: got.append(("laugh", seg, text))
            vl.stt.on_phrase(4, "あのね、")
            check("途中の文字を感情の判定（と笑い判定）へ配る", ("emotion", 4, "あのね、") in got
                  and (vl.laugh is None or ("laugh", 4, "あのね、") in got), f"{got}")

            def boom(seg, text):
                raise RuntimeError("x")

            vl.emotion_judge.on_phrase = boom
            ok = True
            try:
                vl.stt.on_phrase(5, "えっと、")
            except Exception:
                ok = False
            check("1つの判定の例外で他を止めない・外へ出さない", ok)
        check("声のトーンは未決の間は付けない", vl._voice_tone is None and vl.orchestrator._tone_provider is None)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "emotion.yaml"
            p.write_text("voice_emoji:\n  哀: \"😭\"\n", encoding="utf-8")
            Config.EMOTION_TABLE = str(p)
            vl = VoiceLoop()
            if hasattr(vl.tts, "match_emoji_length"):
                check("声のトーン: 応答の組み立てに配線", vl.orchestrator._tone_provider is vl._voice_tone
                      and vl._voice_tone is not None)
                t0 = time.time()
                vl.emotion.note({"哀": 60}, source="phrase", as_of=t0)
                a = vl.orchestrator._tone_provider()
                vl.emotion.note({"楽": 60}, source="phrase", as_of=t0 + 1)
                b = vl.orchestrator._tone_provider()
                vl.emotion.note({"哀": 10, "喜": 5}, source="phrase", as_of=t0 + 2)
                c = vl.orchestrator._tone_provider()
                check("声のトーン: いちばん強い感情の絵文字（決まっていない感情・弱い感情は付けない）",
                      a == "😭" and b is None and c is None, f"{a} {b} {c}")
                check("長さは絵文字なしにそろえる（サーバへ match_plain）", vl.tts.match_emoji_length is True)
                vl.emotion.note({"楽": 60}, source="phrase", as_of=t0 + 3)
                check("フィラーの感情: 今のいちばん強い感情を読む", vl.filler is None or vl.filler._emotion_fn() == "楽")
            else:
                print("SKIP 声のトーン（TTS が Irodori でない）")
    finally:
        Config.EMOTION_ENABLED, Config.EMOTION_TABLE, Config.EMOTION_STATE_FILE = saved
        tmp.cleanup()


SR = 48000


def _tone_wave(sec: float, f: float = 300.0):
    import numpy as np

    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def t_filler_bank() -> None:
    print("\n[8] フィラーの声（感情ごとのバンク）")
    import json

    from eve.filler import audio as fa
    from eve.filler.bank import Clip, FillerBank, FillerTable, _load_emotion_bank

    table = FillerTable(weights={"a": 50.0, "aa": 30.0}, texts={"a": "あ", "aa": "あー"}, join_gap_sec=0.01)
    want = {"a", "aa"}
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "eve_calm_emo_sad"
        b.mkdir()
        clips = []
        for k in range(2):
            f = f"a_{k}.wav"
            (b / f).write_bytes(fa.to_wav(_tone_wave(0.09), SR))
            clips.append({"file": f, "id": "a", "text": "あ", "synth_text": "あ😭"})
        (b / "manifest.json").write_text(json.dumps({"clips": clips}, ensure_ascii=False), encoding="utf-8")
        pool = _load_emotion_bank(table, b, want, SR, "😭")
        check("感情のバンクを読む（絵文字が emotion.yaml と同じ）", list(pool) == ["a"] and len(pool["a"]) == 2,
              f"{ {k: len(v) for k, v in pool.items()} }")
        check("絵文字が違うバンクは使わない（作り直すまで普通の声）", _load_emotion_bank(table, b, want, SR, "😊") == {})
        check("レートが違うバンクは使わない", _load_emotion_bank(table, b, want, 24000, "😭") == {})
        check("バンクが無ければ普通の声", _load_emotion_bank(table, Path(d) / "none", want, SR, "😭") == {})

    neutral = {w: [Clip(w, table.texts[w], _tone_wave(0.06), SR) for _ in range(3)] for w in table.weights}
    sad = {"a": [Clip("a", "あ", _tone_wave(0.09), SR) for _ in range(2)]}
    bank = FillerBank(table, neutral, {}, SR, by_emotion={"哀": sad},
                      laugh_words={"a": [Clip("a", "あ", _tone_wave(0.12), SR)]})
    import random

    rng = random.Random(1)
    check("感情があればその感情の声から選ぶ", bank.pick("a", rng, emotion="哀")[1].sec > 0.08)
    check("その感情に無い語は普通の声", bank.pick("aa", rng, emotion="哀")[1].sec < 0.07)
    check("感情が無い・バンクが無い感情は普通の声", bank.pick("a", rng)[1].sec < 0.07
          and bank.pick("a", rng, emotion="喜")[1].sec < 0.07)
    check("笑いの声色（S6 の tone）が先", bank.pick("a", rng, laughing=True, emotion="哀")[1].sec > 0.11)


async def t_filler_chain() -> None:
    print("\n[8b] フィラーを鳴らし始めた時の感情で、そのチェーンを鳴らす")
    import random

    from eve.filler import audio as fa
    from eve.filler.bank import Clip, FillerBank, FillerTable
    from eve.filler.player import FillerPlayer
    from eve.pipeline.audio_play_queue import AudioPlayQueue

    table = FillerTable(weights={"a": 50.0, "aa": 30.0}, texts={"a": "あ", "aa": "あー"},
                        silence_weight=0.0, silence_sec=(0.02, 0.03), stretch_prob=0.5,
                        stretchable={"a": "a", "aa": "a"}, max_chain_sec=5.0, min_utterance_sec=0.6,
                        join_gap_sec=0.01, stretch_adopted=True)
    neutral = {w: [Clip(w, table.texts[w], _tone_wave(0.06), SR) for _ in range(3)] for w in table.weights}
    sad = {w: [Clip(w, table.texts[w], _tone_wave(0.09), SR) for _ in range(2)] for w in table.weights}
    stretched = {(w, k): Clip(w, table.texts[w] + "ー", _tone_wave(0.2), SR) for w in table.weights for k in range(3)}
    bank = FillerBank(table, neutral, stretched, SR, by_emotion={"哀": sad})
    played: list = []

    async def play(wav):
        x, sr = fa.read_wav(wav)
        played.append(len(x) / sr)  # 前の「、」の間（0.01秒）込み
        await asyncio.sleep(len(x) / sr)

    audio = AudioPlayQueue(play_fn=play)
    texts: list = []
    audio.on_sentence_start = lambda text, meta: texts.append(text)
    emo = {"v": "哀"}
    pl = FillerPlayer(bank, audio=audio, play_fn=play, is_busy=lambda: False, rng=random.Random(5),
                      emotion_fn=lambda: emo["v"])
    pl.start()
    pl.on_speech_end(1.5, 1)
    await asyncio.sleep(0.12)
    emo["v"] = "喜"  # チェーンの途中で感情が変わった
    await asyncio.sleep(0.25)
    pl.on_response_near()
    await asyncio.sleep(0.2)
    await pl.stop()
    check("鳴らし始めた時の感情の声で鳴らす（途中で変わってもそのチェーンは同じ声）",
          len(played) >= 2 and all(0.085 <= d <= 0.105 for d in played), f"{[round(d, 3) for d in played]}")
    check("感情の声の語は伸ばさない（伸ばし棒の素材が普通の声のため・継いだ語は 0.2秒）",
          not any(d > 0.15 for d in played), f"{texts}")


async def main() -> None:
    t_state()
    t_prompt()
    t_filler_bank()
    for fn in (t_judge, t_feedback, t_orchestrator, t_voice_tone, t_filler_chain):
        try:
            await asyncio.wait_for(fn(), 15.0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} が例外なく終わる", False, f"{type(e).__name__}: {e}")
    t_context()


asyncio.run(main())
t_wiring()
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
