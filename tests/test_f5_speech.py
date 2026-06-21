"""F5 発話判定 / 自発発話の決定論テスト（API 不要・fake LLM）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f5_speech.py
ハーネスは他 Tier-1 と同形。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.config import Config  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import Embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.pipeline.orchestrator import StubOrchestrator  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.feedback import PredictionState  # noqa: E402
from eve.speech import (  # noqa: E402
    AutonomousSpeech,
    SilenceMonitor,
    SpeechDecider,
    SpeechDecision,
    SpeechState,
    make_decide_fn,
    parse_speech_decision,
    should_speak,
)

_tmpdir = tempfile.mkdtemp(prefix="eve_f5_")
_counter = 0


def _tmp() -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"f5_{_counter}.jsonl")


class FakeEmbedder(Embedder):
    AXES = ["夏", "祭り", "仕事", "旅行"]

    def __init__(self) -> None:
        self.dim = len(self.AXES)

    def _vec(self, text: str):
        v = [1.0 if ax in text else 0.0 for ax in self.AXES]
        return v if any(v) else [0.001] * len(self.AXES)

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        return self._vec(text)


def _store() -> RagStore:
    s = RagStore(FakeEmbedder(), rag_file=_tmp())
    s.rel_baseline = 0.0
    return s


async def _noop_play(audio, should_stop=None) -> None:
    pass


async def _tts(s: str) -> bytes:
    return b"x"

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


async def _fixed_silence(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
    return SpeechDecision(False, "fixed-silence", "")


async def _fixed_speak(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
    return SpeechDecision(True, "fixed-speak", "やあ")


async def _surprise_driven(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
    # surprise を読む fake（＝surprise を加味する LLM の代理）。配線の死活検出に使う。
    return SpeechDecision(surprise >= 50, f"surprise={surprise}", "x")


def _reg(text: str) -> ModelRegistry:
    async def fake(model, messages, **kw):
        return {"choices": [{"message": {"content": text}}]}

    return ModelRegistry(completion_fn=fake)


async def _call(surprise, decide_fn, pending=False) -> SpeechDecision:
    return await should_speak(
        surprise=surprise, silence_seconds=10.0, recent_turns=[], topic_seeds=[],
        decide_fn=decide_fn, pending_obligation=pending,
    )


# ===== T2 death-detection（surprise が判定に効く配線である保証）=====
async def t_t2_surprise_wired() -> bool:
    """surprise を読む decider なら surprise を振ると判定が反転（surprise が決定に効く配線）。"""
    hi = await _call(80, _surprise_driven)
    lo = await _call(10, _surprise_driven)
    return hi.speak is True and lo.speak is False


def t_surprise_is_required() -> bool:
    p = inspect.signature(should_speak).parameters["surprise"]
    return p.default is inspect.Parameter.empty


# ===== 判定は LLM 任せ（数値ゲート撤廃・surprise は指標）=====
async def t_llm_authoritative() -> bool:
    sp = await _call(5, _fixed_speak)     # 低 surprise でも LLM が speak なら speak
    si = await _call(95, _fixed_silence)  # 高 surprise でも LLM が silence なら silence
    return sp.speak is True and si.speak is False


async def t_pending_hard_silence() -> bool:
    d = await _call(95, _fixed_speak, pending=True)
    return d.speak is False  # pending は唯一の hard ゲート（LLM speak でも沈黙）


async def t_speak_uses_llm_content() -> bool:
    d = await _call(40, _fixed_speak)
    return d.speak is True and d.content == "やあ"


async def t_empty_content_fallback() -> bool:
    async def speak_empty(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        return SpeechDecision(True, "話す", "")  # speak だが content 空

    d = await _call(40, speak_empty)
    return d.speak is True and d.content.strip() != ""  # 全 speak 経路で fallback 保証


async def t_last_feedback_passed_to_decider() -> bool:
    seen: dict = {}

    async def capture(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        seen["fb"] = last_feedback
        return SpeechDecision(False, "r", "")

    await should_speak(surprise=20, silence_seconds=5, recent_turns=[], topic_seeds=[],
                       decide_fn=capture, last_feedback="楽しい気分")
    return seen.get("fb") == "楽しい気分"


# ===== パーサ =====
def t_parse_yes() -> bool:
    d = parse_speech_decision("speak: yes\nreason: 話題がある\ncontent: 今日いい天気だね")
    return d.speak and d.content == "今日いい天気だね" and d.reason == "話題がある"


def t_parse_no() -> bool:
    d = parse_speech_decision("speak: no\nreason: 特にない")
    return (not d.speak) and d.reason == "特にない"


def t_parse_fullwidth() -> bool:
    d = parse_speech_decision("speak：yes\nreason：理由\ncontent：やあ")
    return d.speak and d.content == "やあ"


def t_parse_garbage_silence() -> bool:
    d = parse_speech_decision("ぐちゃぐちゃで何も無い文")
    return d.speak is False


def t_parse_ellipsis_silence() -> bool:
    return parse_speech_decision("…").speak is False


def t_parse_content_no_tag_silence() -> bool:
    # speak タグが無く content だけ → 保守的に silence（P6: content で speak 推定しない）。
    d = parse_speech_decision("content: うーん、やめておく")
    return d.speak is False


# ===== make_decide_fn（保守フォールバック）=====
async def t_decidefn_valid() -> bool:
    fn = make_decide_fn(_reg("speak: yes\nreason: r\ncontent: こんにちは"))
    d = await fn(surprise=40, silence_seconds=5, recent_turns=[], topic_seeds=[])
    return d.speak and d.content == "こんにちは"


async def t_decidefn_garbage_silence() -> bool:
    fn = make_decide_fn(_reg("???"))
    d = await fn(surprise=40, silence_seconds=5, recent_turns=[], topic_seeds=[])
    return d.speak is False


async def t_decidefn_apierror_silence() -> bool:
    async def boom(model, messages, **kw):
        raise RuntimeError("api down")

    fn = make_decide_fn(ModelRegistry(completion_fn=boom))
    d = await fn(surprise=40, silence_seconds=5, recent_turns=[], topic_seeds=[])
    return d.speak is False


# ===== C2 orchestrator 配線 =====
async def t_autonomous_injection() -> bool:
    captured: dict = {}

    async def stream_fn(messages):
        captured["m"] = messages
        yield "やあ、最近どう？"

    audio = AudioPlayQueue(play_fn=_noop_play)
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    cache.add_turn("eve", "やあ")
    rag = _store()
    await rag.add_chunk(text="夏祭りに行った思い出", summary="夏祭り", search_text="夏 祭り", importance=0.5)
    orch = ResponseOrchestrator(
        audio, stream_fn, _tts, ContextAssembler(system_prompt="S"),
        conversation_cache=cache, rag_store=rag,
    )
    worker = asyncio.create_task(audio.play_worker())
    users_before = sum(1 for t in cache.recent(99) if t.speaker == "user")
    await orch.handle(
        Stimulus(StimulusKind.AUTONOMOUS_SPEECH, AutonomousSpeech("天気の話を振ってみる", "間が空いたから")),
    )
    worker.cancel()
    msg = captured["m"][-1]["content"]
    users_after = sum(1 for t in cache.recent(99) if t.speaker == "user")
    eve_turns = [t.text for t in cache.recent(99) if t.speaker == "eve"]
    await cache.shutdown()
    return (
        "天気の話を振ってみる" in msg                 # content は注入される
        and "# 自分から話す" in msg                    # Fix3: イブ自身の発話として枠分け
        and "# ユーザ発話（今）" not in msg            # Fix3: ユーザ発話枠には入れない（話者取り違え防止）
        and "# 発話判定理由" in msg and "間が空いたから" in msg  # reason → speech_decision_reason
        and "話題の種" in msg                          # rag.random は as_topic_seed=True
        and users_after == users_before               # メモリ非汚染: user ターン増えない
        and any("やあ、最近どう" in t for t in eve_turns)  # eve 発話は記録される
    )


async def t_is_busy() -> bool:
    audio = AudioPlayQueue(play_fn=_noop_play)
    runner = PipelineRunner(StimulusQueue(), StubOrchestrator(audio), audio)
    idle = runner.is_busy() is False

    async def slow():
        await asyncio.sleep(0.15)

    runner._active = asyncio.create_task(slow())
    busy = runner.is_busy() is True
    await runner._active
    done = runner.is_busy() is False
    return idle and busy and done


# ===== C3 SpeechState / SilenceMonitor / SpeechDecider =====
class _FakeDecider:
    def __init__(self) -> None:
        self.triggers = 0
        self._idle = True

    def trigger(self) -> None:
        self.triggers += 1

    def is_idle(self) -> bool:
        return self._idle


def t_speechlog_rocket_pencil() -> bool:
    state = SpeechState(log_max=10)
    for i in range(12):
        state.record_decision(speak=(i % 2 == 0), reason=f"r{i}", content="")
    log = list(state.speech_log)
    speaks = [e["speak"] for e in log]
    return (
        len(log) == 10  # ロケット鉛筆（最古2件押し出る）
        and (True in speaks and False in speaks)  # True/False とも記録
        and log[0]["reason"] == "r2"
    )


def t_flat_cadence() -> bool:
    clk = [1000.0]
    state = SpeechState(now_fn=lambda: clk[0])
    clk[0] = 1006.0
    due1 = state.eval_due(5.0)  # baseline から 6s → 評価可
    state.record_decision(speak=False, reason="x", content="")  # last_eval=1006
    due2 = state.eval_due(5.0)  # 直後 → 不可（5秒連打しない）
    clk[0] = 1012.0
    due3 = state.eval_due(5.0)  # 6s 経過 → 再び可（フラット5秒）
    return due1 and (not due2) and due3


def t_monitor_guards() -> bool:
    clk = [1000.0]
    state = SpeechState(now_fn=lambda: clk[0])
    fd = _FakeDecider()
    busy = [False]
    mon = SilenceMonitor(state=state, decider=fd, is_busy_fn=lambda: busy[0], threshold_sec=5.0, tick_sec=0.7)
    clk[0] = 1003.0
    early = mon.tick()  # 3s < 5s → 発火しない
    clk[0] = 1006.0
    fired = mon.tick()  # 6s ≥ 5s → 発火
    n1 = fd.triggers
    clk[0] = 1020.0
    busy[0] = True
    busy_block = mon.tick()  # 応答中 → 発火しない
    busy[0] = False
    state.user_speaking = True
    user_block = mon.tick()  # ユーザ発話中 → 発火しない
    state.user_speaking = False
    fd._idle = False
    idle_block = mon.tick()  # decider 処理中 → 発火しない
    return (
        early is False and fired is True and n1 == 1
        and busy_block is False and user_block is False and idle_block is False
    )


async def t_decider_speak_injects() -> bool:
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    cache.add_turn("eve", "やあ")
    rag = _store()
    await rag.add_chunk(text="夏祭りの思い出", summary="夏", search_text="夏", importance=0.5)
    pred = PredictionState()  # surprise=20(NEUTRAL) → 中間帯 → LLM 判断

    async def fake_decide(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        return SpeechDecision(True, "話題がある", "天気の話を振る")

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred, queue=q, decide_fn=fake_decide)
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    log = list(state.speech_log)
    ok_queue = q.qsize() == 1
    s = await q.get() if ok_queue else None
    return (
        ok_queue and s.kind == StimulusKind.AUTONOMOUS_SPEECH
        and isinstance(s.payload, AutonomousSpeech) and s.payload.content == "天気の話を振る"
        and len(log) == 1 and log[0]["speak"] is True
    )


async def t_decider_silence_no_stimulus() -> bool:
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    rag = _store()
    pred = PredictionState()

    async def fake_silence(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        return SpeechDecision(False, "特に言うことがない", "")

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred, queue=q, decide_fn=fake_silence)
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    log = list(state.speech_log)
    # 黙る: 刺激は出ないが発話判定ログには記録（観測のため）
    return q.qsize() == 0 and len(log) == 1 and log[0]["speak"] is False


async def t_decider_single_flight() -> bool:
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    rag = _store()
    pred = PredictionState()
    gate = asyncio.Event()
    stats = {"concurrent": 0, "max": 0}

    async def gated(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        stats["concurrent"] += 1
        stats["max"] = max(stats["max"], stats["concurrent"])
        try:
            await gate.wait()
            return SpeechDecision(False, "r", "")
        finally:
            stats["concurrent"] -= 1

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred, queue=q, decide_fn=gated)
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.02)
    dec.trigger()  # 1件目処理中の2発目
    await asyncio.sleep(0.02)
    gate.set()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    return stats["max"] == 1  # 同時実行は常に1（single-flight）


def t_eve_activity_resets_silence() -> bool:
    """Eve が喋ったら沈黙時計がリセット（直後に自分へモノローグしない）+ user_speaking トグル。"""
    clk = [1000.0]
    state = SpeechState(now_fn=lambda: clk[0])
    clk[0] = 1006.0
    due_before = state.eval_due(5.0)  # baseline から 6s → 評価可
    state.mark_eve_activity()  # Eve 発話 → last_activity=1006
    due_after = state.eval_due(5.0)  # 直後 → 不可（モノローグ防止）
    state.mark_user_speech_start()
    speaking = state.user_speaking is True
    state.mark_user_utterance()
    not_speaking = state.user_speaking is False
    return due_before and (not due_after) and speaking and not_speaking


# ===== Fix1 barge-in が自発発話を中止・削除 =====
async def t_discard_kind() -> bool:
    q = StimulusQueue()
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "u"))
    await q.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, AutonomousSpeech("a", "r"), merge_key="autonomous"))
    n = q.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)
    kinds = [s.kind for s in q.snapshot()]
    return n == 1 and StimulusKind.AUTONOMOUS_SPEECH not in kinds and StimulusKind.USER_UTTERANCE in kinds


async def t_decider_user_preempts() -> bool:
    """判定中にユーザが話し始めたら自発発話を中止・破棄（put しない）。"""
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    rag = _store()
    pred = PredictionState()
    gate = asyncio.Event()

    async def gated(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        await gate.wait()
        return SpeechDecision(True, "話したい", "やあ")  # LLM は speak と判断

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred, queue=q, decide_fn=gated)
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.02)  # 判定中（gate 保持）
    state.mark_user_speech_start()  # ← ユーザが話し始めた（user_activity_seq 変化）
    gate.set()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    log = list(state.speech_log)
    # 刺激は出ず、ログは中止として記録される（speak=False・理由に「中止」）
    return q.qsize() == 0 and len(log) == 1 and log[0]["speak"] is False and "中止" in log[0]["reason"]


# ===== Fix3 自発発話の話者ロール枠分け（取り違え防止）=====
def t_autonomous_content_own_block() -> bool:
    r = ContextAssembler(system_prompt="S").assemble(autonomous_content="天気の話を振る").render()
    return "# 自分から話す" in r and "天気の話を振る" in r and "# ユーザ発話（今）" not in r


def t_user_text_still_user_block() -> bool:
    r = ContextAssembler(system_prompt="S").assemble(user_text="こんにちは").render()
    return "# ユーザ発話（今）" in r and "こんにちは" in r and "# 自分から話す" not in r


async def main() -> None:
    check("T2 surprise が判定に効く配線(反転)", await t_t2_surprise_wired())
    check("surprise は必須引数(既定なし)", t_surprise_is_required())
    check("判定は LLM 任せ(数値ゲート撤廃)", await t_llm_authoritative())
    check("pending は唯一の hard 沈黙", await t_pending_hard_silence())
    check("speak は LLM の content を使う", await t_speak_uses_llm_content())
    check("speak で content 空→fallback(全 speak 経路)", await t_empty_content_fallback())
    check("last_feedback(感情)を decider に渡す", await t_last_feedback_passed_to_decider())
    check("parse yes", t_parse_yes())
    check("parse no", t_parse_no())
    check("parse 全角コロン", t_parse_fullwidth())
    check("parse ゴミ→silence", t_parse_garbage_silence())
    check("parse …→silence", t_parse_ellipsis_silence())
    check("parse content のみ(タグ無)→silence", t_parse_content_no_tag_silence())
    check("decide_fn 正常", await t_decidefn_valid())
    check("decide_fn ゴミ→silence", await t_decidefn_garbage_silence())
    check("decide_fn API 例外→silence", await t_decidefn_apierror_silence())
    # C2
    check("C2 自発発話注入(content+理由+話題の種・非汚染)", await t_autonomous_injection())
    check("C2 runner.is_busy()", await t_is_busy())
    # C3
    check("C3 発話判定ログ ロケット鉛筆(T/F両方・maxlen10)", t_speechlog_rocket_pencil())
    check("C3 フラット5秒カデンス(連打しない)", t_flat_cadence())
    check("C3 SilenceMonitor ガード(busy/user/閾値/idle)", t_monitor_guards())
    check("C3 decider speak→AUTONOMOUS 刺激+ログ", await t_decider_speak_injects())
    check("C3 decider silence→刺激なし+ログ記録", await t_decider_silence_no_stimulus())
    check("C3 decider single-flight(同時=1)", await t_decider_single_flight())
    # C4
    check("C4 Eve 発話で沈黙時計リセット + user_speaking トグル", t_eve_activity_resets_silence())
    # Fix1
    check("Fix1 discard_kind が自発刺激を削除", await t_discard_kind())
    check("Fix1 判定中ユーザ発話で自発発話を中止・破棄", await t_decider_user_preempts())
    # Fix3
    check("Fix3 自発 content はイブ自身の発話枠(ユーザ枠でない)", t_autonomous_content_own_block())
    check("Fix3 USER は従来どおりユーザ発話枠", t_user_text_still_user_block())


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
