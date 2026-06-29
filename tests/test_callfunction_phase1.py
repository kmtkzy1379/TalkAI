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


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
