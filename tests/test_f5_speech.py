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
    build_decide_messages,
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


# ===== J-2 P2-3: 実行中タスクを判定材料に渡す（先回り回答の防止） =====
async def t_active_tasks_passed_to_decider() -> bool:
    seen: dict = {}

    async def capture(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None,
                      active_tasks=None):
        seen["tasks"] = active_tasks
        return SpeechDecision(False, "r", "")

    await should_speak(surprise=20, silence_seconds=5, recent_turns=[], topic_seeds=[],
                       decide_fn=capture, active_tasks=["・「モンハンの検索」（実行中）"])
    return seen.get("tasks") == ["・「モンハンの検索」（実行中）"]


async def t_active_tasks_not_forwarded_when_none() -> bool:
    # A6 と同じ規約: active_tasks=None(既定)の時は既存 decide_fn(active_tasks 未対応)を壊さない。
    async def legacy_decide_fn(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        return SpeechDecision(True, "r", "x")

    d = await should_speak(surprise=20, silence_seconds=5, recent_turns=[], topic_seeds=[],
                           decide_fn=legacy_decide_fn)
    return d.speak is True  # TypeError にならない


def t_build_decide_messages_active_tasks_block() -> bool:
    with_tasks = build_decide_messages(
        surprise=10, silence_seconds=5, recent_turns=[], topic_seeds=[],
        active_tasks=["・「富士山の標高」（実行中）"],
    )[1]["content"]
    zero_tasks = build_decide_messages(
        surprise=10, silence_seconds=5, recent_turns=[], topic_seeds=[], active_tasks=[],
    )[1]["content"]
    no_block = build_decide_messages(
        surprise=10, silence_seconds=5, recent_turns=[], topic_seeds=[],
    )[1]["content"]
    return (
        "実行中のタスク" in with_tasks and "富士山の標高" in with_tasks
        and "実行中のタスク" in zero_tasks and "実行中の予約タスクは無い" in zero_tasks
        and "実行中のタスク" not in no_block  # None=ブロック自体を出さない
    )


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
    last = captured["m"][-1]                                  # 自発指示は最終 user メッセージ
    joined = "\n".join(m["content"] for m in captured["m"])   # 文脈(理由/話題の種)は system に分散
    users_after = sum(1 for t in cache.recent(99) if t.speaker == "user")
    eve_turns = [t.text for t in cache.recent(99) if t.speaker == "eve"]
    await cache.shutdown()
    return (
        last["role"] == "user"
        and "天気の話を振ってみる" in last["content"]       # content は最終 user 指示に入る
        and "返事ではなく" in last["content"]               # Fix3/4: 自分から話す指示（返事でない）
        and "# 発話判定理由" in joined and "間が空いたから" in joined  # reason は system
        and "夏祭り" in joined                              # autonomous_memories の記憶材料が注入される
        and users_after == users_before                     # メモリ非汚染: user ターン増えない
        and any("やあ、最近どう" in t for t in eve_turns)   # eve 発話は記録される
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

    def trigger(self, source: str = "unknown") -> None:
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


# ===== J-2 ②-1: 同内容の自発発話の抑制（コードゲート） =====
# 実E2Eで観測された重複ペア（回帰の一次データそのもの）
_DUP_A1 = "さっきの3つ、比べるとスカイツリーよりエベレストはかなり高いし、琵琶湖は面積で見るとまた別の大きさ感があって面白いね。"
_DUP_A2 = "さっきの3つって、比べるとスカイツリーは高さの目安になって、エベレストは本当に別格だし、琵琶湖は面積で考えるとまた違うスケール感があって面白いね。"
_DUP_B1 = "へえ、よかったんだね。どんな味だったのか気になるよ。"
_DUP_B2 = "そのラーメン、どんな味だったのか気になるよ。こってり系だったのか、あっさり系だったのかも知りたいな。"
_DIFF_1 = "そういえば前にチーズケーキが好きって言ってたよね。最近食べた？"
_DIFF_2 = "まだ動いてるね、30秒の通知を待ってる間はこのまま見守るよ。"


def t_content_similarity_calibration() -> bool:
    from eve.speech.monitor import DUP_SIM_THRESHOLD, content_similarity
    th = DUP_SIM_THRESHOLD
    return (
        content_similarity(_DUP_A1, _DUP_A2) >= th  # S10 重複ペア（実測0.453）
        and content_similarity(_DUP_B1, _DUP_B2) >= th  # ラーメン2連発（実測0.286）
        and content_similarity(_DUP_A1, _DIFF_1) < th  # 別話題（実測≤0.074）
        and content_similarity(_DUP_A1, _DIFF_2) < th
        and content_similarity(_DIFF_1, _DIFF_2) < th
        and content_similarity("", _DUP_A1) == 0.0  # 空は常に0（ゼロ除算なし）
    )


async def t_decider_suppresses_duplicate_content() -> bool:
    # 1回目 speak→投入 / 2回目 ほぼ同内容→抑制（投入なし・理由と下書きを記録）/ 3回目 別内容→投入。
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    rag = _store()
    pred = PredictionState()
    contents = [_DUP_A1, _DUP_A2, _DIFF_1]
    calls = [0]

    async def fake_decide(*, surprise, silence_seconds, recent_turns, topic_seeds,
                          last_feedback=None, active_tasks=None):
        c = contents[min(calls[0], len(contents) - 1)]
        calls[0] += 1
        return SpeechDecision(True, "話す", c)

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred,
                        queue=q, decide_fn=fake_decide)
    dec.start()
    delivered = []  # 各回で drain（AUTONOMOUS は merge_key で畳まれるため qsize 累積では数えない）
    for _ in range(3):
        dec.trigger()
        await asyncio.sleep(0.05)
        while q.qsize() > 0:
            delivered.append((await q.get()).payload.content)
    await dec.stop()
    await cache.shutdown()
    log = list(state.speech_log)
    suppressed = [e for e in log if "抑制" in e["reason"]]
    return (
        delivered == [_DUP_A1, _DIFF_1]  # 1回目と3回目だけ投入（2回目は抑制）
        and len(suppressed) == 1
        and suppressed[0]["speak"] is False
        and suppressed[0]["content"] == _DUP_A2  # 何を言おうとしたかは記録に残す
    )


# ===== J-2 ②-2: 比較元を「自発発話専用の時間窓」に分離 + 意味の二段目 =====
# 実E2E(2026-07-23 e2e_autospeech)で観測された「語彙を変えただけの再提案」ペア。
# 文字bigram では 0.23 で閾値 0.25 をすり抜けた＝二段目（埋め込み）が要る理由の一次データ。
_PARA_1 = "買い物リスト、必要なら抜け漏れだけさらっと一緒に確認するよ。"
_PARA_2 = "じゃあ、次は買い物リストを軽く見て、抜けてるものだけさらっと確認しよっか。"


class _FakeEmbedder:
    """注入用 embedder（実モデルなし・決定論）。text→固定ベクトルの表引き。"""

    def __init__(self, table: dict):
        self.table = table
        self.calls = 0

    async def embed_query(self, text: str):
        self.calls += 1
        return self.table[text]


class _BoomEmbedder:
    async def embed_query(self, text: str):
        raise RuntimeError("埋め込み失敗（テスト）")


def t_cosine_basic() -> bool:
    from eve.speech.monitor import cosine
    return (
        abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
        and abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9
        and cosine([], [1.0]) == 0.0  # 長さ違い/空は0（例外にしない）
        and cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # ゼロ除算なし
    )


def t_paraphrase_slips_bigram() -> bool:
    # 二段目が必要な理由の回帰仕様: 同内容の言い換えは文字bigram では閾値未満（実測0.23）
    from eve.speech.monitor import DUP_SIM_THRESHOLD, content_similarity
    return content_similarity(_PARA_1, _PARA_2) < DUP_SIM_THRESHOLD


async def _run_decider(state, contents, *, embedder=None, hook=None, rag=None) -> tuple[list, list]:
    """contents を順に返す fake decide で decider を回し、(投入内容, 発話判定ログ) を返す。"""
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    rag = rag if rag is not None else _store()
    calls = [0]

    async def fake_decide(*, surprise, silence_seconds, recent_turns, topic_seeds,
                          last_feedback=None, active_tasks=None):
        d = contents[min(calls[0], len(contents) - 1)]
        calls[0] += 1
        return d

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=PredictionState(),
                        queue=q, decide_fn=fake_decide, embedder=embedder)
    dec.start()
    delivered = []
    for i in range(len(contents)):
        dec.trigger()
        await asyncio.sleep(0.05)
        while q.qsize() > 0:
            delivered.append((await q.get()).payload.content)
        if hook is not None:
            hook(i)
    await dec.stop()
    await cache.shutdown()
    return delivered, list(state.speech_log)


def t_turn_rendering_has_elapsed() -> bool:
    # 直近会話には「いつの発話か」を必ず添える。無いと起動直後に前セッションの続きを
    # 「今まさに返事待ち」と誤読して沈黙し続ける（実測 2026-07-26: 実起動状態で25判定中24が
    # 「直前に伝えたばかり」を理由に沈黙。実際の会話は9日前）。
    from eve.clock import Stamp
    from eve.context_assembler import Turn
    from eve.speech.decider import _render_turns
    now = "2026-07-26T12:00:00+00:00"
    turns = [Turn("user", "今タスクに入ってるんだよね", Stamp(mono=0.0, iso="2026-07-17T12:00:00+00:00")),
             Turn("eve", "うん、調べてるところだよ", Stamp(mono=0.0, iso="2026-07-26T11:58:00+00:00"))]
    s = _render_turns(turns, now)
    no_stamp = _render_turns([type("T", (), {"speaker": "user", "text": "素の発話"})()], now)
    return (
        "[ユーザ/9日前]" in s and "[イブ/2分前]" in s
        and "[user]" not in s
        and "[ユーザ] 素の発話" in no_stamp  # stamp を持たない相手でも壊れない
        and _render_turns([], now) == "（直近の会話なし）"
    )


def t_seed_rendering_summary_and_time() -> bool:
    # 話題の種は「要約1行 + いつの記憶か」で描く。text 全文（感情/次の予測/予測差/理由の
    # 内部ログ・平均162字）を渡すと話題として使える1行が埋もれ、時刻が無いと「そういえば前に」
    # が言えない（実測 2026-07-26: 現行形式では発話率7%・記憶接地も弱い）。
    from eve.context_assembler import RagChunk
    from eve.speech.decider import _render_seeds
    now = "2026-07-26T12:00:00+00:00"
    c1 = RagChunk(
        text="ユーザは空が青い理由を気にしていた\n感情: イブ=好奇心\n次の予測: 続けて質問しそう\n予測差: 92",
        iso="2026-07-23T12:00:00+00:00", as_topic_seed=True,
        summary="ユーザは空が青い理由を気にしていた")
    c2 = RagChunk(text="要約なし記録の1行目\n感情: イブ=平静", iso="2026-07-26T11:55:00+00:00")
    s = _render_seeds([c1, c2], now)
    return (
        "3日前" in s and "5分前" in s              # いつの記憶かを必ず添える
        and "次の予測" not in s and "予測差" not in s  # 内部ログは渡さない
        and "ユーザは空が青い理由を気にしていた" in s
        and "要約なし記録の1行目" in s              # summary 欠落の古い記録は1行目で代替
        and _render_seeds([], now) == "（なし）"
    )


async def t_seed_query_uses_last_user_turn() -> bool:
    # 話題の種のクエリは**直近のユーザ発話**で引く（イブ自身の自発発話で引くと自己強化ループ）
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "ラーメンの話をしてたね")
    cache.add_turn("eve", "買い物リストの抜け漏れを確認しようか")  # イブの自発発話が最後のターン
    captured = {}

    class _SpyRag:
        async def autonomous_memories(self, query, k, *, context_since_iso=None):
            captured["q"] = query
            captured["since"] = context_since_iso
            return []

    async def fake_decide(*, surprise, silence_seconds, recent_turns, topic_seeds,
                          last_feedback=None, active_tasks=None):
        return SpeechDecision(False, "黙る", "")

    dec = SpeechDecider(state=state, cache=cache, rag=_SpyRag(), prediction_state=PredictionState(),
                        queue=StimulusQueue(), decide_fn=fake_decide)
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    # ②-3: 今注入している会話区間（最古ターンの iso）を渡し、そこから生まれた記憶を除外させる
    return (
        "ラーメン" in captured.get("q", "") and "買い物" not in captured.get("q", "")
        and captured.get("since")  # 会話区間が渡っている（落とすと自己参照が復活する）
    )


async def t_dup_survives_speech_log_overflow() -> bool:
    # ⭐②-2 死活: 黙る判定で観測ログが溢れても同内容抑制は効き続ける。
    # 旧実装は比較元が speech_log（件数10・False 混在）だったため、放置中は 5-6秒毎の False で
    # 約50-60秒で前回発話が押し出され、同じ提案が通っていた（実測 8回中4回すり抜け）。
    state = SpeechState(log_max=10)
    silent = SpeechDecision(False, "黙る", "")
    contents = [SpeechDecision(True, "話す", _DUP_A1)] + [silent] * 12 + [SpeechDecision(True, "話す", _DUP_A2)]
    delivered, log = await _run_decider(state, contents)
    pushed_out = all(e["content"] != _DUP_A1 for e in log)  # 観測ログからは押し出されている
    return delivered == [_DUP_A1] and pushed_out and "抑制" in log[-1]["reason"]


async def t_dup_window_expires() -> bool:
    # 時間窓の外に出た自発発話とは比較しない（永久に同じ話題を封じない）
    clk = [1000.0]
    state = SpeechState(now_fn=lambda: clk[0], dup_window_sec=600.0)

    def advance(i):
        clk[0] += 601.0  # 1回目の直後に窓を跨ぐ

    contents = [SpeechDecision(True, "話す", _DUP_A1), SpeechDecision(True, "話す", _DUP_A2)]
    delivered, _ = await _run_decider(state, contents, hook=advance)
    return delivered == [_DUP_A1, _DUP_A2]


async def t_dup_embedding_second_stage() -> bool:
    # ⭐二段目: 文字bigram をすり抜けた言い換え(0.23)を埋め込み cos で抑制。別話題は通す。
    emb = _FakeEmbedder({
        _PARA_1: [1.0, 0.0],
        _PARA_2: [0.93, 0.3676],  # cos≈0.93 ≥ 0.87 → 抑制
        _DIFF_1: [0.8, 0.6],      # cos=0.80 < 0.87 → 通す
    })
    state = SpeechState()
    contents = [SpeechDecision(True, "話す", c) for c in (_PARA_1, _PARA_2, _DIFF_1)]
    delivered, log = await _run_decider(state, contents, embedder=emb)
    suppressed = [e for e in log if "抑制" in e["reason"]]
    return (
        delivered == [_PARA_1, _DIFF_1]
        and len(suppressed) == 1
        and "意味" in suppressed[0]["reason"]  # 二段目で捕まえたと分かる
        and emb.calls == 3  # 判定ごとに1回だけ（投入時の記録に再利用する）
    )


async def t_dup_embedding_failure_falls_back() -> bool:
    # 埋め込みが失敗しても落とさず、一段目(文字bigram)のみで継続する
    state = SpeechState()
    contents = [SpeechDecision(True, "話す", c) for c in (_PARA_1, _PARA_2, _DUP_A1, _DUP_A2)]
    delivered, log = await _run_decider(state, contents, embedder=_BoomEmbedder())
    return (
        delivered == [_PARA_1, _PARA_2, _DUP_A1]  # 言い換えは通る（二段目なし）が文字重複は止まる
        and any("抑制" in e["reason"] for e in log)
    )


# ===== J-2 ②-4: 根拠なき話題の丸投げ（空振り発話）の抑制 =====
# 実E2E(2026-07-26 実起動状態)で**実際に再生された**空振り発話＝回帰の一次データ
_WHIFF_1 = "予定ない時間って、逆に「今ならできること」も見つけやすいよね。今ふと、気になってることとかある？"
_WHIFF_2 = "じゃあ、今はのんびりタイムだね。もし気が向いたら、最近ちょっと気になってることをぽつっと聞かせて。"
# 同E2Eで正当と判断した発話（誤検出してはならない）
_CARE_1 = "休憩中なら、温かい飲み物でも用意してのんびりしよっか。"
_MEMO_1 = "休憩中って、ふと別のこと思い出したりするよね。前に電子レンジの仕組みに興味持ってたけど、ああいう「なんで？」って気になる話、私はけっこう好きだよ。"
_SEED_1 = "ユーザは電子レンジで食べ物が温まる仕組みに興味を持ち、マイクロ波と水分の関係を話題にしていた"
_ASK_GROUNDED = "さっきから同じような検索を何度も打ち直してるけど、何か困ってる？"
_VISION_1 = "検索ボックスに同じ語を打っては消す動作が繰り返されている"


def t_outsourcing_calibration() -> bool:
    from eve.speech.decider import is_topic_outsourcing as f
    return (
        f(_WHIFF_1, []) and f(_WHIFF_2, [])                    # 実測の空振り2件を捕まえる
        and not f(_MEMO_1, [_SEED_1])                          # 記憶起点は通す
        # ⭐死活A: 「丸投げ表現」アーム。これが死ぬと接地ゼロの気遣い発話まで潰れる
        # （接地アーム単独は実発話198件中104件=53%を潰す実測）
        and not f(_CARE_1, [])
        # ⭐死活B: 「材料接地」アーム。材料の有無で結果が反転する＝「見えている根拠がある時だけ
        # 困りごとを聞いてよい」がコードに配線されている保証
        and f(_ASK_GROUNDED, []) and not f(_ASK_GROUNDED, [_VISION_1])
        # 既存ゲートのテスト素材を巻き込まない（②-1/②-2 の意図が変わらない）
        and not f(_DUP_A1, []) and not f(_PARA_1, [])
    )


async def t_decider_suppresses_topic_outsourcing() -> bool:
    # worker 経路: 空振りは投入されず、記憶起点は投入される。何を言おうとしたかは記録に残す。
    state = SpeechState()
    contents = [SpeechDecision(True, "話す", _WHIFF_1), SpeechDecision(True, "話す", _MEMO_1)]

    class _SeedRag:
        async def autonomous_memories(self, query, k, *, context_since_iso=None):
            class _C:
                text = _SEED_1

                def seed_text(self):
                    return _SEED_1
            return [_C()]

    delivered, log = await _run_decider(state, contents, rag=_SeedRag())
    whiff = [e for e in log if "丸投げ" in e["reason"]]
    return (
        delivered == [_MEMO_1]
        and len(whiff) == 1 and whiff[0]["speak"] is False and whiff[0]["content"] == _WHIFF_1
    )


async def t_outsourcing_gate_precedes_dup_gate() -> bool:
    # 空振りゲートは同内容抑制より前（無駄な埋め込み呼び出しをしない）
    emb = _FakeEmbedder({})
    state = SpeechState()
    delivered, _ = await _run_decider(state, [SpeechDecision(True, "話す", _WHIFF_1)], embedder=emb)
    return delivered == [] and emb.calls == 0


# ===== J-2 ③-A: STT 待ち窓（発話終了〜テキスト投入）ガード =====
def t_stt_pending_flag_and_expiry() -> bool:
    clk = [1000.0]
    state = SpeechState(now_fn=lambda: clk[0], stt_pending_max_sec=10.0)
    none_before = not state.stt_pending
    state.mark_stt_pending()
    active = state.stt_pending
    clk[0] = 1009.9
    still = state.stt_pending
    clk[0] = 1010.0
    expired = not state.stt_pending  # 最大寿命で自動失効（STTハング/クリア漏れで固着しない）
    state.mark_stt_pending()
    state.clear_stt_pending()
    cleared = not state.stt_pending
    return none_before and active and still and expired and cleared


def t_monitor_blocks_during_stt_pending() -> bool:
    clk = [1000.0]
    state = SpeechState(now_fn=lambda: clk[0])
    fd = _FakeDecider()
    mon = SilenceMonitor(state=state, decider=fd, is_busy_fn=lambda: False, threshold_sec=5.0, tick_sec=0.7)
    clk[0] = 1006.0
    state.mark_stt_pending()
    blocked = mon.tick()  # 窓内 → 5秒沈黙でも発火しない
    state.clear_stt_pending()
    fired = mon.tick()  # 窓が閉じたら発火
    return blocked is False and fired is True and fd.triggers == 1


async def t_decider_discards_when_stt_pending_at_completion() -> bool:
    # 保留トリガで判定が窓内から開始した形の簡約: 判定中に窓が開く（seq は不変）→ 完了時に破棄。
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    rag = _store()
    pred = PredictionState()

    async def decide_then_window(*, surprise, silence_seconds, recent_turns, topic_seeds,
                                 last_feedback=None, active_tasks=None):
        state.mark_stt_pending()  # 判定 LLM 実行中に発話セグメント到着（STT 開始）を模す
        return SpeechDecision(True, "話す", "こんばんは")

    q = StimulusQueue()
    dec = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred,
                        queue=q, decide_fn=decide_then_window)
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    log = list(state.speech_log)
    return (q.qsize() == 0 and len(log) == 1
            and log[0]["speak"] is False and "STT処理中" in log[0]["reason"])


class _FakeAIOnce:
    """1回だけ pcm を返し、以後は永久待ち（_consume ループの単発駆動用）。"""

    def __init__(self) -> None:
        self._sent = False

    async def get_audio(self):
        if self._sent:
            await asyncio.Event().wait()
        self._sent = True
        return b"pcm"


async def t_input_source_stt_window_order() -> bool:
    # 窓は transcribe 前に開き、put 完了後に閉じる（put 前に閉じる微小窓を作らない）。
    from eve.response.input_source import MicSttInputSource

    events: list = []
    release = asyncio.Event()

    class FakeStt:
        async def transcribe(self, pcm):
            events.append("stt_begin")
            await release.wait()
            return "こんにちは"

    q = StimulusQueue()
    src = MicSttInputSource(
        q, FakeStt(),
        on_utterance=lambda: events.append("utterance"),
        on_stt_start=lambda: events.append("open"),
        on_stt_end=lambda: events.append("close"),
    )
    src._ai = _FakeAIOnce()
    task = asyncio.create_task(src._consume())
    await asyncio.sleep(0.05)
    during = list(events)  # STT 実行中: close は未発火のはず
    release.set()
    await asyncio.sleep(0.05)
    task.cancel()
    got = q.qsize() == 1
    s = await q.get() if got else None
    return (
        during == ["utterance", "open", "stt_begin"]
        and events == ["utterance", "open", "stt_begin", "close"]
        and got and s.payload == "こんにちは"
    )


async def t_input_source_stt_window_closes_on_failure() -> bool:
    # STT 失敗/空認識でも窓は必ず閉じる（固着防止）・投入なし・ループ継続（無クラッシュ）。
    from eve.response.input_source import MicSttInputSource

    events: list = []

    class FailStt:
        async def transcribe(self, pcm):
            raise RuntimeError("stt down")

    q = StimulusQueue()
    src = MicSttInputSource(
        q, FailStt(),
        on_stt_start=lambda: events.append("open"),
        on_stt_end=lambda: events.append("close"),
    )
    src._ai = _FakeAIOnce()
    task = asyncio.create_task(src._consume())
    await asyncio.sleep(0.05)
    task.cancel()
    return events == ["open", "close"] and q.qsize() == 0


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


async def t_decider_tasks_provider_wired() -> bool:
    # J-2 P2-3: tasks_provider の戻り値が decide_fn に active_tasks として届く。
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    rag = _store()
    pred = PredictionState()
    seen: dict = {}

    async def capture(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None,
                      active_tasks=None):
        seen["tasks"] = active_tasks
        return SpeechDecision(False, "r", "")

    q = StimulusQueue()
    dec = SpeechDecider(
        state=state, cache=cache, rag=rag, prediction_state=pred, queue=q, decide_fn=capture,
        tasks_provider=lambda: ["・「検索中のタスク」（実行中）"],
    )
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    return seen.get("tasks") == ["・「検索中のタスク」（実行中）"]


async def t_decider_tasks_provider_exception_safe() -> bool:
    # tasks_provider が例外を投げても発話判定自体はクラッシュせず継続する（注入なしで継続）。
    state = SpeechState()
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    cache.add_turn("user", "こんにちは")
    rag = _store()
    pred = PredictionState()
    seen: dict = {}

    async def capture(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None,
                      active_tasks=None):
        seen["tasks"] = active_tasks
        seen["called"] = True
        return SpeechDecision(False, "r", "")

    def boom():
        raise RuntimeError("store 未初期化")

    q = StimulusQueue()
    dec = SpeechDecider(
        state=state, cache=cache, rag=rag, prediction_state=pred, queue=q, decide_fn=capture,
        tasks_provider=boom,
    )
    dec.start()
    dec.trigger()
    await asyncio.sleep(0.05)
    await dec.stop()
    await cache.shutdown()
    return seen.get("called") is True and seen.get("tasks") is None


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
    msgs = ContextAssembler(system_prompt="S").assemble(autonomous_content="天気の話を振る")
    last = msgs[-1]
    # 自発: 最終 user 指示に下書きが入り「返事ではなく自分から」と明示。過去 user ターンには無い。
    return (
        last["role"] == "user"
        and "天気の話を振る" in last["content"]
        and "返事ではなく" in last["content"]
        and not any(m is not last and m["role"] == "user" and "天気の話を振る" in m["content"] for m in msgs)
    )


def t_user_text_still_user_block() -> bool:
    msgs = ContextAssembler(system_prompt="S").assemble(user_text="こんにちは")
    # USER 経路: 最終メッセージは user ロールでユーザの言葉そのまま（native ロール）。
    return msgs[-1]["role"] == "user" and msgs[-1]["content"] == "こんにちは"


async def main() -> None:
    check("T2 surprise が判定に効く配線(反転)", await t_t2_surprise_wired())
    check("surprise は必須引数(既定なし)", t_surprise_is_required())
    check("判定は LLM 任せ(数値ゲート撤廃)", await t_llm_authoritative())
    check("pending は唯一の hard 沈黙", await t_pending_hard_silence())
    check("speak は LLM の content を使う", await t_speak_uses_llm_content())
    check("speak で content 空→fallback(全 speak 経路)", await t_empty_content_fallback())
    check("last_feedback(感情)を decider に渡す", await t_last_feedback_passed_to_decider())
    check("J-2 P2-3: active_tasks を decide_fn に渡す", await t_active_tasks_passed_to_decider())
    check("J-2 P2-3: active_tasks=None は旧decide_fnを壊さない", await t_active_tasks_not_forwarded_when_none())
    check("J-2 P2-3: build_decide_messages の実行中タスクブロック", t_build_decide_messages_active_tasks_block())
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
    check("J-2 P2-3: tasks_provider の戻り値が届く", await t_decider_tasks_provider_wired())
    check("J-2 P2-3: tasks_provider 例外でも継続", await t_decider_tasks_provider_exception_safe())
    # J-2 ②-1: 同内容の自発発話の抑制
    check("J-2 ②-1: 類似度指標の実データ校正（重複≥閾値/別話題<閾値）", t_content_similarity_calibration())
    check("J-2 ②-1: 同内容の自発発話をコードゲートで抑制", await t_decider_suppresses_duplicate_content())
    check("直近会話に経過時間を添える(古い会話を返事待ちと誤読しない)", t_turn_rendering_has_elapsed())
    check("話題の種は要約1行+相対時刻で描く", t_seed_rendering_summary_and_time())
    check("話題の種は直近ユーザ発話で引く(自己強化しない)", await t_seed_query_uses_last_user_turn())
    check("⭐②-4 空振り判定の較正(丸投げ/接地の両アーム死活)", t_outsourcing_calibration())
    check("⭐②-4 空振りは投入せず記憶起点は通す", await t_decider_suppresses_topic_outsourcing())
    check("②-4 空振りゲートは同内容抑制より前(埋め込み節約)", await t_outsourcing_gate_precedes_dup_gate())
    check("cosine 基本(同一/直交/空/ゼロ)", t_cosine_basic())
    check("言い換えは文字bigramをすり抜ける(二段目の根拠)", t_paraphrase_slips_bigram())
    check("⭐②-2: 黙る判定でログが溢れても抑制が効く", await t_dup_survives_speech_log_overflow())
    check("②-2: 時間窓を出たら比較しない", await t_dup_window_expires())
    check("⭐②-2: 二段目(埋め込み)で言い換えを抑制・別話題は通す", await t_dup_embedding_second_stage())
    check("②-2: 埋め込み失敗は一段目で継続(落とさない)", await t_dup_embedding_failure_falls_back())
    # J-2 ③-A: STT 待ち窓ガード
    check("J-2 ③-A: stt_pending フラグ+最大寿命失効", t_stt_pending_flag_and_expiry())
    check("J-2 ③-A: 窓内は monitor が発火しない", t_monitor_blocks_during_stt_pending())
    check("J-2 ③-A: 判定完了時に窓内なら破棄", await t_decider_discards_when_stt_pending_at_completion())
    check("J-2 ③-A: 窓は transcribe 前に開き put 後に閉じる", await t_input_source_stt_window_order())
    check("J-2 ③-A: STT失敗でも窓は閉じる（固着防止）", await t_input_source_stt_window_closes_on_failure())
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
