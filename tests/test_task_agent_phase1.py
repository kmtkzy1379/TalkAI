"""TaskAgent inc2（境界つき native ツール呼出ループ）の決定論テスト（API 不要・fake model/registry）。

検証:
- ループ: tool_call を順に実行→結果を messages へ→tool_calls 無し(最終答)で終了。content/tool_calls は
  dict・オブジェクト両形で頑健に取り出す（_extract）。
- 境界（コードが所有）: max_steps 打ち切り / timeout(monotonic) 打ち切り / 毎step キャンセル検知→破棄(None)。
- executor 分岐: task.goal → agent.run（Done+再投入 / None=取消破棄 / agent 無し=Failed）。task.what → 既存パス。
- capabilities: delegate_task が goal 付き(what 空)タスクを追加・report_result=False。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_task_agent_phase1.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import types
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.pipeline.stimulus import StimulusKind  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.task import (  # noqa: E402
    CANCELLED, DONE, FAILED, PENDING, RUNNING,
    Task, TaskExecutor, TaskStore, register_task_capabilities,
)
from eve.task.agent import TaskAgent  # noqa: E402

_passed = 0
_failed = 0
_dir = tempfile.mkdtemp(prefix="eve_agent_")
_n = 0
BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


def _tmp():
    global _n
    _n += 1
    return os.path.join(_dir, f"t{_n}.jsonl")


# ---- fake resp ファクトリ（dict 形 / object 形の両方を試す）----
def _resp(content=None, tool_calls=None):
    return {"choices": [{"message": {"content": content, "tool_calls": tool_calls}}]}


def _tc(cid, name, args="{}"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


def _ores(content=None, tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _otc(cid, name, args="{}"):
    return types.SimpleNamespace(id=cid, function=types.SimpleNamespace(name=name, arguments=args))


class FakeModel:
    """scripted resp を順に返す。尽きたら default（既定=最終答で停止）。"""

    def __init__(self, scripted, default=None):
        self.scripted = list(scripted)
        self.default = default
        self.calls = 0
        self.seen_tools = None

    async def complete(self, role, messages, tools=None):
        self.calls += 1
        self.seen_tools = tools
        if self.scripted:
            return self.scripted.pop(0)
        return self.default if self.default is not None else _resp(content="（台本終わり）")


class FakeReg:
    def __init__(self, results, on_exec=None):
        self.results = results
        self.calls = []
        self._on_exec = on_exec

    def agent_tool_schemas(self):  # agent はこちらを使う（実行系スコープ）
        return [{"type": "function",
                 "function": {"name": k, "description": "", "parameters": {"type": "object", "properties": {}}}}
                for k in self.results]

    def execute(self, name, args):
        self.calls.append((name, args))
        if self._on_exec:
            self._on_exec(name, args)
        return self.results.get(name, f"（未対応{name}）")

    def has(self, name):
        return name in self.results


class CancelStore:
    """agent の cancel 検知用の最小 store。get(tid).status を返す。"""

    def __init__(self, status=PENDING):
        self._s = {"t": status}

    def get(self, tid):
        st = self._s.get(tid)
        return None if st is None else types.SimpleNamespace(status=st)

    def cancel(self, tid="t"):
        self._s[tid] = CANCELLED


# ========== TaskAgent ループ ==========
async def t_agent_loop_basic_object_path():
    # object 形 resp（litellm 実物相当）で: 1回 tool_call → 最終答。
    model = FakeModel([_ores(tool_calls=[_otc("c1", "self_status")]), _ores(content="両方ともOKだよ")])
    reg = FakeReg({"self_status": "手が空いている", "pc_status": "OS Windows"})
    agent = TaskAgent(registry=reg, model_registry=model, store=CancelStore())
    out = await agent.run("状態を調べて", "t")
    return out == "両方ともOKだよ" and reg.calls == [("self_status", {})] and model.calls == 2


async def t_agent_two_step_then_final():
    # dict 形 resp で 2手（pc→self）→ 最終要約。messages に結果が積まれ順に実行される。
    model = FakeModel([
        _resp(tool_calls=[_tc("c1", "pc_status")]),
        _resp(tool_calls=[_tc("c2", "self_status")]),
        _resp(content="PCもイブも問題ないよ"),
    ])
    reg = FakeReg({"self_status": "手が空いている", "pc_status": "OS Windows / CPU 8"})
    agent = TaskAgent(registry=reg, model_registry=model, store=CancelStore())
    out = await agent.run("PCとイブの状態を調べてまとめて", "t")
    return out == "PCもイブも問題ないよ" and [c[0] for c in reg.calls] == ["pc_status", "self_status"]


async def t_agent_args_parsed():
    # arguments(JSON 文字列) が dict に parse されて execute に渡る。
    seen = []
    model = FakeModel([_resp(tool_calls=[_tc("c1", "echo", '{"message":"やあ"}')]), _resp(content="言ったよ")])
    reg = FakeReg({"echo": "ok"}, on_exec=lambda n, a: seen.append(a))
    agent = TaskAgent(registry=reg, model_registry=model, store=CancelStore())
    out = await agent.run("echo して", "t")
    return out == "言ったよ" and seen == [{"message": "やあ"}]


async def t_agent_max_steps():
    # 常に tool_call を返す（最終答が来ない）→ max_steps で打ち切り・フォールバック文。
    model = FakeModel([], default=_resp(tool_calls=[_tc("c", "self_status")]))
    reg = FakeReg({"self_status": "手が空いている"})
    agent = TaskAgent(registry=reg, model_registry=model, store=CancelStore(), max_steps=3)
    out = await agent.run("終わらないやつ", "t")
    return out.startswith("（手順の上限") and model.calls == 3 and len(reg.calls) == 3


async def t_agent_timeout():
    # now_fn を deadline 超過に飛ばす → 途中で打ち切り。
    clock = iter([0.0, 0.0, 99.0, 99.0, 99.0])  # deadline計算, step0 check, step1 check...

    def now():
        try:
            return next(clock)
        except StopIteration:
            return 99.0

    model = FakeModel([], default=_resp(tool_calls=[_tc("c", "self_status")]))
    reg = FakeReg({"self_status": "ok"})
    agent = TaskAgent(registry=reg, model_registry=model, store=CancelStore(), timeout_sec=10.0, now_fn=now)
    out = await agent.run("遅いやつ", "t")
    return out.startswith("（時間切れ") and model.calls == 1  # step0 で1回実行、step1 で timeout


async def t_agent_cancel_midloop():
    # 実行の最中に cancel が入る（execute が store を Cancelled に）→ 次 step で検知→None（破棄）。
    cs = CancelStore()
    model = FakeModel([], default=_resp(tool_calls=[_tc("c", "self_status")]))
    reg = FakeReg({"self_status": "ok"}, on_exec=lambda n, a: cs.cancel())
    agent = TaskAgent(registry=reg, model_registry=model, store=cs, max_steps=5)
    out = await agent.run("途中で取消されるやつ", "t")
    return out is None and len(reg.calls) == 1  # step0 実行→cancel→step1 で None


async def t_agent_missing_task_aborts():
    # store に居ない（get None）→ 即 None（破棄）。
    class EmptyStore:
        def get(self, tid):
            return None
    model = FakeModel([_resp(content="x")])
    agent = TaskAgent(registry=FakeReg({}), model_registry=model, store=EmptyStore())
    out = await agent.run("g", "t")
    return out is None and model.calls == 0


# ========== Executor 分岐（実 store/executor + fake agent）==========
class FakeAgent:
    def __init__(self, result, side_effect=None):
        self.result = result
        self._side = side_effect
        self.ran = []

    async def run(self, goal, task_id):
        self.ran.append((goal, task_id))
        if self._side:
            self._side(task_id)
        return self.result


async def t_executor_goal_done_reinject():
    q = StimulusQueue()
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="g1", what="", goal="PCとイブを調べてまとめて", when=None))
    agent = FakeAgent("PCもイブも問題ないよ")
    ex = TaskExecutor(store=s, registry=FakeReg({}), queue=q, agent=agent)
    ex.start(); ex.trigger()
    await asyncio.sleep(0.05)
    await ex.stop()
    results = [st for st in q.snapshot() if st.kind == StimulusKind.CALLFUNCTION_RESULT]
    ok = (s.get("g1").status == DONE and agent.ran == [("PCとイブを調べてまとめて", "g1")]
          and len(results) == 1 and "PCもイブも問題ないよ" in results[0].payload.content
          and (results[0].dedup_key or "").startswith("task:"))
    await s.shutdown()
    return ok


async def t_executor_goal_cancel_discard():
    # agent.run が（実行中の取消で）None を返す → 再投入しない・status は Cancelled。
    q = StimulusQueue()
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="g2", what="", goal="途中で取消", when=None))
    agent = FakeAgent(None, side_effect=lambda tid: s.set_status(tid, CANCELLED))
    ex = TaskExecutor(store=s, registry=FakeReg({}), queue=q, agent=agent)
    ex.start(); ex.trigger()
    await asyncio.sleep(0.05)
    await ex.stop()
    discarded = q.qsize() == 0 and s.get("g2").status == CANCELLED
    await s.shutdown()
    return discarded


async def t_executor_goal_no_agent_fails():
    q = StimulusQueue()
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="g3", what="", goal="何か", when=None))
    ex = TaskExecutor(store=s, registry=FakeReg({}), queue=q, agent=None)  # agent 無し
    ex.start(); ex.trigger()
    await asyncio.sleep(0.05)
    await ex.stop()
    failed = s.get("g3").status == FAILED
    await s.shutdown()
    return failed


# ========== capabilities: delegate_task ==========
async def t_delegate_task_capability():
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    reg = CapabilityRegistry()
    register_task_capabilities(reg, s)
    r = reg.execute("delegate_task", {"goal": "PCとイブの状態を調べてまとめて", "when_seconds": 5})
    t = s.list_all()[-1]
    r_empty = reg.execute("delegate_task", {"goal": "  "})
    cap = reg._caps["delegate_task"]
    resp = {x["function"]["name"] for x in reg.tool_schemas()}        # 応答LLM 向け（委譲/管理）
    agent = {x["function"]["name"] for x in reg.agent_tool_schemas()}  # TaskAgent 向け（実行系）
    await s.shutdown()
    return (
        "引き受けた" in r and t.goal == "PCとイブの状態を調べてまとめて" and t.what == "" and t.when is not None
        and "ゴールが空" in r_empty
        and cap.mutates_state is True and cap.report_result is False
        # 責務分離: 応答LLM は委譲/管理だけ・実行系は見えない。agent は実行系だけ・委譲系は見えない。
        and resp == {"delegate_task", "create_task", "cancel_task", "list_tasks"}
        and "self_status" in agent and "pc_status" in agent and "delegate_task" not in agent
        # delegate_task は予約 action に出ない（_MGMT 除外）= create_task で予約できない
        and "予約できない" in reg.execute("create_task", {"action": "delegate_task"})
        # 実行系は create_task で予約できる（agent_tool）
        and "作成した" in reg.execute("create_task", {"action": "pc_status"})
    )


async def main():
    check("Agent: 基本ループ(object形)→最終答", await t_agent_loop_basic_object_path())
    check("Agent: 2手(pc→self)→要約", await t_agent_two_step_then_final())
    check("Agent: arguments(JSON)→dict で execute", await t_agent_args_parsed())
    check("Agent: max_steps 打ち切り", await t_agent_max_steps())
    check("Agent: timeout 打ち切り", await t_agent_timeout())
    check("Agent: 実行中キャンセル→None(破棄)", await t_agent_cancel_midloop())
    check("Agent: task 消失→None", await t_agent_missing_task_aborts())
    check("Executor: goal→agent→Done+再投入", await t_executor_goal_done_reinject())
    check("Executor: goal が None→取消破棄", await t_executor_goal_cancel_discard())
    check("Executor: agent 無し goal→Failed", await t_executor_goal_no_agent_fails())
    check("capabilities: delegate_task(goal/markers)", await t_delegate_task_capability())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
