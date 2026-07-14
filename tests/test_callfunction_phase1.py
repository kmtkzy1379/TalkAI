"""Call-Function（即時・read-only）の決定論テスト（API 不要・fake 注入）。

検証:
- Capability: read-only 実行・例外時「失敗」文字列・未対応「未対応」・tool_schemas が valid。
- merge_tool_call_deltas: streaming 断片を1件に結合。
- FunctionDispatcher: submit 非ブロッキング・背景 worker 逐次・dedup_key・未対応 ok=False・stop drain。
- ResponseOrchestrator: tool_calls 捕捉→応答完了後に submit / CALLFUNCTION_RESULT を「# 機能実行結果」へ
  （user 枠に repr を入れない）/ 1ホップ抑制（結果ターンに tools 無し・submit なし）/ barge-in で submit なし /
  無発話(tool のみ)でも無クラッシュ / dispatcher=None で従来挙動（旧 stream_fn シグネチャ不変）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_callfunction_phase1.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.config import Config  # noqa: E402
from eve.model_registry import merge_tool_call_deltas  # noqa: E402
from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402
from eve.pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher, parse_tool_call  # noqa: E402
from eve.response.orchestrator import ResponseOrchestrator  # noqa: E402

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


def _tc(call_id: str, name: str, args: str = "{}") -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": args}}


class RecordingRegistry:
    """実行順序を記録する fake registry。no_report に入れた名前は report_result=False を返す。"""

    def __init__(self, no_report: set | None = None, results: dict | None = None) -> None:
        self.calls: list[str] = []
        self._no_report = no_report or set()
        self._results = results or {}

    def has(self, name: str) -> bool:
        return name != "unknown_fn"

    def report_result(self, name: str) -> bool:
        return name not in self._no_report

    def execute(self, name: str, args: dict) -> str:
        self.calls.append(name)
        return self._results.get(name, f"result:{name}")

    def tool_schemas(self) -> list[dict]:
        return []


class FakeDispatcher:
    """orchestrator 用: submit を記録するだけ。"""

    def __init__(self, schemas: list[dict]) -> None:
        self._schemas = schemas
        self.submitted: list[list] = []

    def tool_schemas(self) -> list[dict]:
        return self._schemas

    def submit(self, tool_calls) -> None:
        self.submitted.append(list(tool_calls))


class FakeRag:
    """orchestrator 用: 検索クエリを記録する fake RagStore。"""

    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    async def search(self, q, k=None):
        self.queries.append(("search", q))
        return []

    async def autonomous_memories(self, q, k=3):
        self.queries.append(("auto", q))
        return []


# ========== Capability ==========
def t_capability_readonly() -> bool:
    reg = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 2)
    s = reg.execute("self_status", {})
    p = reg.execute("pc_status", {})
    u = reg.execute("nope", {})
    # 実行系(self_status/pc_status)は **応答LLM には非提示**・**TaskAgent 向け**に提示（責務分離）。
    resp_schemas = {x["function"]["name"] for x in reg.tool_schemas()}
    agent_schemas = {x["function"]["name"] for x in reg.agent_tool_schemas()}
    scope_ok = (
        bool([json.dumps(x) for x in reg.agent_tool_schemas()])
        and "self_status" in agent_schemas and "pc_status" in agent_schemas
        and "self_status" not in resp_schemas
    )
    return (
        "手が空いている" in s and "刺激は2件" in s
        and "現在時刻" in p
        and "未対応" in u
        and scope_ok
    )


def t_capability_exception() -> bool:
    reg = CapabilityRegistry()
    reg.register(Capability("boom", "x", {}, lambda a: 1 / 0))
    return "失敗" in reg.execute("boom", {})


def t_capability_failure_reason() -> bool:
    # 失敗時に「何で失敗したか」を結果文に含める（ユーザ要望: 失敗理由も話せるように）。
    reg = CapabilityRegistry()

    def boom(a):
        raise ValueError("disk full")

    reg.register(Capability("boom", "x", {}, boom))
    r = reg.execute("boom", {})
    return "失敗" in r and "disk full" in r


def t_parse_tool_call_robust() -> bool:
    name, args, cid = parse_tool_call(_tc("x", "self_status", '{"a": 1}'))
    bad_name, bad_args, _ = parse_tool_call(_tc("y", "f", "not-json"))  # 壊れた JSON → {}
    return name == "self_status" and args == {"a": 1} and cid == "x" and bad_args == {}


# ========== merge ==========
def t_merge_tool_call_deltas() -> bool:
    frags = [
        {"index": 0, "id": "c1", "function": {"name": "self_status", "arguments": ""}},
        {"index": 0, "function": {"name": "", "arguments": '{"a"'}},
        {"index": 0, "function": {"arguments": ":1}"}},
    ]
    out = merge_tool_call_deltas(frags)
    return (
        len(out) == 1 and out[0]["id"] == "c1"
        and out[0]["function"]["name"] == "self_status"
        and out[0]["function"]["arguments"] == '{"a":1}'
    )


def t_merge_multitool() -> bool:
    # マルチツール: index 0/1 の2件を取り違えず別々に結合（「時刻とシステム両方」のケース）。
    frags = [
        {"index": 0, "id": "a", "function": {"name": "pc_status", "arguments": "{}"}},
        {"index": 1, "id": "b", "function": {"name": "self_status", "arguments": ""}},
        {"index": 1, "function": {"arguments": "{}"}},
    ]
    out = merge_tool_call_deltas(frags)
    return len(out) == 2 and [o["function"]["name"] for o in out] == ["pc_status", "self_status"]


# ========== Dispatcher ==========
async def t_dispatcher_nonblocking_sequential() -> bool:
    reg = RecordingRegistry()
    q = StimulusQueue()
    d = FunctionDispatcher(registry=reg, queue=q)
    d.start()
    d.submit([_tc("a", "fa"), _tc("b", "fb")])
    immediate = q.qsize()  # submit は同期＝この時点では worker 未実行（output queue 空）
    await asyncio.sleep(0.05)
    await d.stop()
    results = [s for s in q.snapshot() if s.kind == StimulusKind.CALLFUNCTION_RESULT]
    return (
        immediate == 0
        and reg.calls == ["fa", "fb"]  # 逐次・順序保持
        and len(results) == 2
        and all((s.dedup_key or "").startswith("cf:") for s in results)
    )


async def t_dispatcher_dedup() -> bool:
    reg = RecordingRegistry()
    q = StimulusQueue()
    d = FunctionDispatcher(registry=reg, queue=q)
    d.start()
    d.submit([_tc("x", "fa")])
    await asyncio.sleep(0.03)
    d.submit([_tc("x", "fa")])  # 同 call_id → dedup_key cf:x で2件目は queue が捨てる
    await asyncio.sleep(0.03)
    await d.stop()
    results = [s for s in q.snapshot() if s.kind == StimulusKind.CALLFUNCTION_RESULT]
    return len(results) == 1


async def t_dispatcher_report_result_suppress() -> bool:
    # report_result=False の能力は**成功時は再投入しない**（create_task ack 重複防止）。失敗は報告。
    q = StimulusQueue()
    reg = RecordingRegistry(no_report={"create_task"},
                            results={"create_task": "予約したよ", "bad": "（予約できない）"})
    d = FunctionDispatcher(registry=reg, queue=q)
    d.start()
    d.submit([_tc("a", "create_task")])   # 成功 → 再投入されない
    await asyncio.sleep(0.03)
    after_success = q.qsize()
    d.submit([_tc("b", "bad")])           # 失敗(（…) → 再投入される
    await asyncio.sleep(0.03)
    await d.stop()
    results = [s for s in q.snapshot() if s.kind == StimulusKind.CALLFUNCTION_RESULT]
    return after_success == 0 and len(results) == 1 and results[0].payload.ok is False


async def t_dispatcher_unknown() -> bool:
    reg = CapabilityRegistry()
    q = StimulusQueue()
    d = FunctionDispatcher(registry=reg, queue=q)
    d.start()
    d.submit([_tc("u", "nope")])
    await asyncio.sleep(0.03)
    await d.stop()
    results = [s for s in q.snapshot() if s.kind == StimulusKind.CALLFUNCTION_RESULT]
    return len(results) == 1 and results[0].payload.ok is False and "未対応" in results[0].payload.content


# ========== Orchestrator ==========
async def _noop_play(audio) -> None:
    pass


async def _tts(s: str) -> bytes:
    return b"x"


async def t_orch_captures_and_submits() -> bool:
    Config.CALLFUNCTION_ENABLED = True
    try:
        audio = AudioPlayQueue(play_fn=_noop_play)
        disp = FakeDispatcher([{"type": "function", "function": {"name": "self_status"}}])
        seen: dict = {}

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            seen["tools"] = tools
            yield "ちょっと調べるね。"  # 前置きは喋る
            if tool_sink is not None:
                tool_sink.append(_tc("c1", "self_status"))  # call は無音（構造化）

        orch = ResponseOrchestrator(audio, stream_fn, _tts, dispatcher=disp)
        w = asyncio.create_task(audio.play_worker())
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "調子は？"))
        w.cancel()
        return (
            seen["tools"] is not None  # USER ターンに tools 渡る
            and orch.last_response == "ちょっと調べるね。"  # 前置きは発話
            and disp.submitted == [[_tc("c1", "self_status")]]  # 応答完了後に submit
        )
    finally:
        Config.CALLFUNCTION_ENABLED = False


async def t_orch_result_render_and_suppress() -> bool:
    Config.CALLFUNCTION_ENABLED = True
    try:
        audio = AudioPlayQueue(play_fn=_noop_play)
        disp = FakeDispatcher([{"type": "function", "function": {"name": "self_status"}}])
        seen: dict = {}

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            seen["tools"] = tools
            seen["sys"] = messages[0]["content"]
            seen["all"] = "\n".join(m.get("content", "") for m in messages)
            yield "今の状態を伝えるね。"

        orch = ResponseOrchestrator(audio, stream_fn, _tts, dispatcher=disp)
        w = asyncio.create_task(audio.play_worker())
        payload = CallFunctionResult("self_status", "今は手が空いている", True)
        await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, payload))
        w.cancel()
        return (
            seen["tools"] is None  # 1ホップ抑制（結果ターンに tools 無し）
            and "# 機能実行結果" in seen["sys"] and "今は手が空いている" in seen["sys"]
            and "CallFunctionResult(" not in seen["all"]  # repr が user 枠に流入していない
            and disp.submitted == []  # 結果ターンは何も submit しない
        )
    finally:
        Config.CALLFUNCTION_ENABLED = False


async def t_orch_bargein_no_submit() -> bool:
    Config.CALLFUNCTION_ENABLED = True
    try:
        audio = AudioPlayQueue(play_fn=_noop_play)
        disp = FakeDispatcher([{"type": "function", "function": {"name": "self_status"}}])
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_stream(messages, *, tools=None, tool_sink=None):
            started.set()
            await release.wait()
            if tool_sink is not None:
                tool_sink.append(_tc("c1", "self_status"))
            yield "x。"

        orch = ResponseOrchestrator(audio, slow_stream, _tts, dispatcher=disp)
        task = asyncio.create_task(orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "調子？")))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return disp.submitted == []  # barge-in → submit に到達しない
    finally:
        Config.CALLFUNCTION_ENABLED = False


async def t_orch_toolonly_no_content() -> bool:
    Config.CALLFUNCTION_ENABLED = True
    try:
        audio = AudioPlayQueue(play_fn=_noop_play)
        disp = FakeDispatcher([{"type": "function", "function": {"name": "self_status"}}])

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            if tool_sink is not None:
                tool_sink.append(_tc("c1", "self_status"))
            return
            yield  # async generator 化（到達しない）

        orch = ResponseOrchestrator(audio, stream_fn, _tts, dispatcher=disp)
        w = asyncio.create_task(audio.play_worker())
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "調子？"))
        w.cancel()
        return orch.last_response == "" and len(disp.submitted) == 1  # 無発話でも submit・無クラッシュ
    finally:
        Config.CALLFUNCTION_ENABLED = False


async def t_orch_result_rag_uses_content() -> bool:
    # CALLFUNCTION_RESULT は payload の repr でなく content で RAG 検索する（干渉/誤クエリ防止）。
    audio = AudioPlayQueue(play_fn=_noop_play)
    rag = FakeRag()

    async def stream_fn(messages):
        yield "報告するね。"

    orch = ResponseOrchestrator(audio, stream_fn, _tts, rag_store=rag)
    w = asyncio.create_task(audio.play_worker())
    payload = CallFunctionResult("self_status", "今は手が空いている", True)
    await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, payload))
    w.cancel()
    return rag.queries == [("search", "今は手が空いている")]


async def t_orch_no_dispatcher_unchanged() -> bool:
    # dispatcher 無し＝従来挙動。旧 stream_fn シグネチャ (messages) のまま動く。
    audio = AudioPlayQueue(play_fn=_noop_play)
    seen: dict = {}

    async def stream_fn(messages):  # 旧シグネチャ
        seen["called"] = True
        yield "はい。"

    orch = ResponseOrchestrator(audio, stream_fn, _tts)  # dispatcher=None
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    w.cancel()
    return seen.get("called") is True and orch.last_response == "はい。"


async def main() -> None:
    check("Capability: read-only 実行/未対応/schema", t_capability_readonly())
    check("Capability: 例外は「失敗」文字列", t_capability_exception())
    check("Capability: 失敗理由を結果に含める", t_capability_failure_reason())
    check("parse_tool_call: 頑健(壊れJSON→{})", t_parse_tool_call_robust())
    check("merge_tool_call_deltas: 断片を1件に結合", t_merge_tool_call_deltas())
    check("merge: マルチツール(index別)を別々に結合", t_merge_multitool())
    check("Dispatcher: submit 非ブロッキング・逐次・dedup_key", await t_dispatcher_nonblocking_sequential())
    check("Dispatcher: 同 call_id は dedup", await t_dispatcher_dedup())
    check("Dispatcher: report_result=False は成功時 再投入しない", await t_dispatcher_report_result_suppress())
    check("Dispatcher: 未対応は ok=False", await t_dispatcher_unknown())
    check("Orch: tool_calls 捕捉→完了後 submit・前置きは発話", await t_orch_captures_and_submits())
    check("Orch: 結果は「# 機能実行結果」へ・1ホップ抑制", await t_orch_result_render_and_suppress())
    check("Orch: barge-in では submit しない", await t_orch_bargein_no_submit())
    check("Orch: tool のみ(無発話)でも無クラッシュ・submit", await t_orch_toolonly_no_content())
    check("Orch: 結果は content で RAG 検索(repr 不使用)", await t_orch_result_rag_uses_content())
    check("Orch: dispatcher=None で従来挙動(旧シグネチャ)", await t_orch_no_dispatcher_unchanged())
    check("Fix#2: 予約タスク一覧を全刺激種別の system に注入", await t_orch_tasks_injected_all_kinds())
    check("Fix#2: ゼロ件明示 / provider 未配線はブロック無し", await t_orch_tasks_zero_and_default())
    check("Fix#2: provider 例外でも注入なしで応答継続", await t_orch_tasks_provider_error_resilient())
    check("再配達: cancel経路で attempts+1/dedup/内容/ok 伝搬・幻記憶なし", await t_redeliver_on_cancelled_report())
    check("再配達: stream前(RAG検索中)cancel も救済（事故窓の残り）", await t_redeliver_prestream_rag_cancel())
    check("再配達: gen-break 正常経路でも救済", await t_redeliver_gen_break())
    check("再配達: >5文字再生済みは再配達しない", await t_no_redeliver_after_spoken())
    check("再配達: ≤5文字の前置きのみ再生は再配達する（境界下側）", await t_redeliver_short_preamble())
    check("再配達: ≤5文字でも全文完走再生済みは再配達しない", await t_tiny_report_complete_no_redeliver())
    check("再配達: 途中停止文は聞かれた扱い→put直前 abort が反転（レース）", await t_redeliver_abort_after_midstop())
    check("再配達: USER kind は対象外", await t_no_redeliver_user_kind())
    check("再配達: 非 CallFunctionResult payload は無反応", await t_no_redeliver_bad_payload())
    check("再配達: cancel のみ(世代不変)は対象外", await t_no_redeliver_cancel_only())
    check("再配達: attempts 上限の両側(1→通る/2→断念)", await t_redeliver_attempts_boundary())
    check("再配達: 再報告プレフィクスは attempts>0 のみ", t_redeliver_prefix_in_messages())
    check("再配達: redeliver_fn 例外でも本流に漏れない", await t_redeliver_fn_exception_safe())
    check("再配達: queue で USER が再投入刺激より先に出る", await t_queue_user_priority_over_retry())


async def t_orch_tasks_injected_all_kinds() -> bool:
    # Fix#2: 予約タスク一覧は**全刺激種別**の system に入る（USER=再実行防止 /
    # CALLFUNCTION_RESULT=矛盾約束防止 / AUTONOMOUS=空約束防止）。
    from eve.speech.decider import AutonomousSpeech
    audio = AudioPlayQueue(play_fn=_noop_play)
    seen_sys: list = []

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        seen_sys.append(messages[0]["content"])
        yield "はい。"

    orch = ResponseOrchestrator(audio, stream_fn, _tts,
                                tasks_provider=lambda: ["・「50秒後に気持ち」（あと約30秒で実行）"])
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, CallFunctionResult("goal", "済んだよ", True)))
    await orch.handle(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, AutonomousSpeech(content="一言", reason="沈黙")))
    w.cancel()
    return (len(seen_sys) == 3
            and all("# 予約タスク" in s and "50秒後に気持ち" in s for s in seen_sys))


async def t_orch_tasks_zero_and_default() -> bool:
    # Fix#2: ゼロ件は「（予約タスクは無い）」を明示（矛盾発話はゼロ件を知らないのが原因）。
    # provider 未配線（None）ならブロック自体を出さない（既定挙動不変）。
    audio = AudioPlayQueue(play_fn=_noop_play)
    seen: dict = {}

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        seen["sys"] = messages[0]["content"]
        yield "はい。"

    w = asyncio.create_task(audio.play_worker())
    orch = ResponseOrchestrator(audio, stream_fn, _tts, tasks_provider=lambda: [])
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    zero_ok = "# 予約タスク" in seen["sys"] and "（予約タスクは無い）" in seen["sys"]
    orch2 = ResponseOrchestrator(audio, stream_fn, _tts)  # provider なし
    await orch2.handle(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    none_ok = "# 予約タスク" not in seen["sys"]
    w.cancel()
    return zero_ok and none_ok


async def t_orch_tasks_provider_error_resilient() -> bool:
    # Fix#2: provider 例外は注入なしで応答継続（A3 流儀・落とさない）。
    audio = AudioPlayQueue(play_fn=_noop_play)
    seen: dict = {}

    def boom() -> list:
        raise RuntimeError("store down")

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        seen["sys"] = messages[0]["content"]
        yield "はい。"

    orch = ResponseOrchestrator(audio, stream_fn, _tts, tasks_provider=boom)
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    w.cancel()
    return orch.last_response == "はい。" and "# 予約タスク" not in seen["sys"]


# ========== 再配達（barge-in で潰れた機能報告の救済・2026-07-13 21:20 実機事故対応） ==========

def _rd_payload(attempts=0, ok=True):
    return CallFunctionResult("goal", "10秒タスクの結果: 今は21時45分だよ", ok, attempts)


class FakeCache:
    """C5 検証用（実 ConversationCache と同じく空 text は無視）。"""

    def __init__(self):
        self.turns = []

    def recent_for_injection(self):
        return []

    def add_turn(self, role, text):
        if text:
            self.turns.append((role, text))


async def t_redeliver_on_cancelled_report() -> bool:
    # 事故の直接回帰: 発話前に cancel された報告 → attempts=1・同 dedup/内容/ok で再配達予約。
    audio = AudioPlayQueue(play_fn=_noop_play)
    captured = []

    def rd(s, a):
        captured.append((s, a))
    cache = FakeCache()
    started = asyncio.Event()

    async def stream_fn(messages):
        started.set()
        await asyncio.Event().wait()
        yield "x。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, conversation_cache=cache, redeliver_fn=rd)
    stim = Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(ok=False), dedup_key="task:t1")
    t = asyncio.create_task(orch.handle(stim))
    await started.wait()
    audio.interrupt(); t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    if len(captured) != 1:
        return False
    s, abort = captured[0]
    p = s.payload
    return (s.kind == StimulusKind.CALLFUNCTION_RESULT and s.dedup_key == "task:t1"
            and p.attempts == 1 and p.content == stim.payload.content
            and p.ok is False and p.function_name == "goal"
            and abort() is False  # 何も再生していない＝配達済みでない
            and not any(r == "eve" for r, _ in cache.turns))  # C5: 幻の記憶なし


async def t_redeliver_prestream_rag_cancel() -> bool:
    # 事故窓の残り（レッドチーム指摘#1）: stream 開始前（RAG 検索中）の cancel でも再配達される。
    class HangRag:
        async def search(self, q):
            await asyncio.Event().wait()
    audio = AudioPlayQueue(play_fn=_noop_play)
    captured = []

    def rd(s, a):
        captured.append((s, a))

    async def stream_fn(messages):
        yield "届かない。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, rag_store=HangRag(), redeliver_fn=rd)
    stim = Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1")
    t = asyncio.create_task(orch.handle(stim))
    await asyncio.sleep(0.05)  # RAG 検索で停止中
    audio.interrupt(); t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    return len(captured) == 1 and captured[0][0].payload.attempts == 1


async def t_redeliver_gen_break() -> bool:
    # cancel でなく世代変化（audio.interrupt のみ・stream break）の正常経路でも再配達される。
    audio = AudioPlayQueue(play_fn=_noop_play)
    captured = []

    def rd(s, a):
        captured.append((s, a))

    async def stream_fn(messages):
        audio.interrupt()
        yield "できたよ。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1"))
    return len(captured) == 1 and captured[0][0].payload.attempts == 1 and captured[0][1]() is False


async def t_no_redeliver_after_spoken() -> bool:
    # >5文字の文を再生済み（聞かれた扱い）→ 再配達しない（同じ報告を2回言わない）。
    played = asyncio.Event()

    async def play_fn(audio_bytes):
        played.set()  # on_played は play 直後に await なしで発火 → 再開時 spoken 充填済み
    audio = AudioPlayQueue(play_fn=play_fn)
    captured = []

    def rd(s, a):
        captured.append((s, a))

    async def stream_fn(messages):
        yield "タスク終わったよ、いまは二十一時。"
        await played.wait()
        audio.interrupt()
        yield "続きの文。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1"))
    w.cancel()
    return captured == []


async def t_redeliver_short_preamble() -> bool:
    # 境界の下側: 短い前置き（「了解。」=3文字 ≤5）しか再生できていない中断 → 残りは未配達＝再配達する
    # （stream 完走していない parts は「全文」でない＝完全配達と誤判定しない回帰）。
    played = asyncio.Event()

    async def play_fn(audio_bytes):
        played.set()
    audio = AudioPlayQueue(play_fn=play_fn)
    captured = []

    def rd(s, a):
        captured.append((s, a))

    async def stream_fn(messages):
        yield "了解。"
        await played.wait()
        audio.interrupt()
        yield "本題はこれからだった。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    w = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1"))
    w.cancel()
    return len(captured) == 1 and captured[0][1]() is False


async def t_tiny_report_complete_no_redeliver() -> bool:
    # ≤5文字の報告を**完走+全文再生済み**で世代変化 → 完全配達＝再配達しない（parts 一致ルール）。
    release = asyncio.Event()
    started = asyncio.Event()

    async def play_fn(audio_bytes):
        started.set()
        await release.wait()
    audio = AudioPlayQueue(play_fn=play_fn)
    captured = []

    def rd(s, a):
        captured.append((s, a))

    async def stream_fn(messages):
        yield "できたよ。"  # 4文字・これが全文（stream 完走）
    cache = FakeCache()
    orch = ResponseOrchestrator(audio, stream_fn, _tts, conversation_cache=cache, redeliver_fn=rd)
    w = asyncio.create_task(audio.play_worker())
    t = asyncio.create_task(orch.handle(
        Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1")))
    await started.wait()
    audio.interrupt()  # 再生中にユーザが被せた（世代変化）
    release.set()      # 再生完了 → on_played（聞かれた扱い）
    await t
    w.cancel()
    return captured == []


async def t_redeliver_abort_after_midstop() -> bool:
    # レッドチーム指摘#2の回帰: cancel 伝播は即時・on_played は遅れて発火。判定時は「未配達」でも
    # put 直前の should_abort が途中停止文（聞かれた扱い）を拾って True に反転＝二重発話を防ぐ。
    started = asyncio.Event()

    async def play_fn(audio_bytes, should_stop=None):
        started.set()
        for _ in range(100):
            if should_stop is not None and should_stop():
                return  # 途中停止（それでも on_played は呼ばれる＝C5）
            await asyncio.sleep(0.01)
    audio = AudioPlayQueue(play_fn=play_fn)
    captured = []

    def rd(s, a):
        captured.append((s, a))

    async def stream_fn(messages):
        yield "タスク終わったよ、いまは二十一時。"
        await asyncio.Event().wait()
        yield "x。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    w = asyncio.create_task(audio.play_worker())
    t = asyncio.create_task(orch.handle(
        Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1")))
    await started.wait()
    audio.interrupt(); t.cancel()  # 再生途中に barge-in
    try:
        await t
    except asyncio.CancelledError:
        pass
    pre = len(captured) == 1 and captured[0][1]() is False  # 判定時: spoken 未確定
    await asyncio.sleep(0.1)  # should_stop で play が戻り on_played 発火
    post = captured[0][1]() is True  # put 直前判定: 聞かれた扱い＝中止
    w.cancel()
    return pre and post


async def t_no_redeliver_user_kind() -> bool:
    # USER ターンの中断は再配達対象外（kind ガード）。
    audio = AudioPlayQueue(play_fn=_noop_play)
    captured = []

    def rd(s, a):
        captured.append((s, a))
    started = asyncio.Event()

    async def stream_fn(messages):
        started.set()
        await asyncio.Event().wait()
        yield "x。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    t = asyncio.create_task(orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "調子どう？")))
    await started.wait()
    audio.interrupt(); t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    return captured == []


async def t_no_redeliver_bad_payload() -> bool:
    # CALLFUNCTION_RESULT だが payload が CallFunctionResult でない → 無クラッシュ・再配達なし。
    audio = AudioPlayQueue(play_fn=_noop_play)
    captured = []

    def rd(s, a):
        captured.append((s, a))
    started = asyncio.Event()

    async def stream_fn(messages):
        started.set()
        await asyncio.Event().wait()
        yield "x。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    t = asyncio.create_task(orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, "生文字列")))
    await started.wait()
    audio.interrupt(); t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    return captured == []


async def t_no_redeliver_cancel_only() -> bool:
    # cancel のみ（世代不変＝シャットダウン相当）→ 再配達しない（stop() で余計な waiter を生まない）。
    audio = AudioPlayQueue(play_fn=_noop_play)
    captured = []

    def rd(s, a):
        captured.append((s, a))
    started = asyncio.Event()

    async def stream_fn(messages):
        started.set()
        await asyncio.Event().wait()
        yield "x。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    t = asyncio.create_task(orch.handle(
        Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1")))
    await started.wait()
    t.cancel()  # audio.interrupt() は呼ばない
    try:
        await t
    except asyncio.CancelledError:
        pass
    return captured == []


async def t_redeliver_attempts_boundary() -> bool:
    # 上限の両側: attempts=1 は通る（→2）・attempts=2 は断念。
    async def run_with(att):
        audio = AudioPlayQueue(play_fn=_noop_play)
        captured = []

        def rd(s, a):
            captured.append(s)
        started = asyncio.Event()

        async def stream_fn(messages):
            started.set()
            await asyncio.Event().wait()
            yield "x。"
        orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
        t = asyncio.create_task(orch.handle(
            Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(attempts=att), dedup_key="task:t")))
        await started.wait()
        audio.interrupt(); t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return captured
    c1 = await run_with(1)
    c2 = await run_with(2)
    return len(c1) == 1 and c1[0].payload.attempts == 2 and c2 == []


def t_redeliver_prefix_in_messages() -> bool:
    # 再配達（attempts>0）は「# 機能実行結果」に「遮られた再報告」前置が入る。初回は入らない。
    orch = ResponseOrchestrator(AudioPlayQueue(play_fn=_noop_play), None, _tts)
    m1 = orch._build_messages(Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(attempts=1)))
    m0 = orch._build_messages(Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(attempts=0)))
    s1, s0 = m1[0]["content"], m0[0]["content"]
    return ("# 機能実行結果" in s1 and "読み上げが遮られて" in s1
            and s1.index("# 機能実行結果") < s1.index("読み上げが遮られて")
            and "21時45分" in s1
            and "遮られて" not in s0 and "# 機能実行結果" in s0)


async def t_redeliver_fn_exception_safe() -> bool:
    # redeliver_fn が例外を投げても、cancel 経路は CancelledError を正しく raise・gen-break 経路は正常 return。
    audio = AudioPlayQueue(play_fn=_noop_play)

    def rd(s, a):
        raise RuntimeError("boom")
    started = asyncio.Event()

    async def stream_fn(messages):
        started.set()
        await asyncio.Event().wait()
        yield "x。"
    orch = ResponseOrchestrator(audio, stream_fn, _tts, redeliver_fn=rd)
    t = asyncio.create_task(orch.handle(
        Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t1")))
    await started.wait()
    audio.interrupt(); t.cancel()
    cancelled_ok = False
    try:
        await t
    except asyncio.CancelledError:
        cancelled_ok = True
    except Exception:
        return False

    async def stream2(messages):
        audio.interrupt()
        yield "y。"
    orch2 = ResponseOrchestrator(audio, stream2, _tts, redeliver_fn=rd)
    await orch2.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(), dedup_key="task:t2"))
    return cancelled_ok


async def t_queue_user_priority_over_retry() -> bool:
    # 「ユーザ応答が先」のキュー側保証: 再投入刺激と USER が同時在中なら USER が先に出る。
    q = StimulusQueue()
    retry = Stimulus(StimulusKind.CALLFUNCTION_RESULT, _rd_payload(attempts=1), dedup_key="task:t1")
    await q.put(retry)
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "OK"))
    first = await q.get()
    second = await q.get()
    return first.kind == StimulusKind.USER_UTTERANCE and second is retry


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
