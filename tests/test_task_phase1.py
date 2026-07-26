"""タスク管理 inc1（土台ハーネス）の決定論テスト（API 不要・fake registry/clock）。

検証:
- Store: add/JSONL 往復・**terminal-stays-terminal**・due_now(when≤now)・claim_due(atomic)・孤児 age-out。
- Executor: due を逐次実行→決定論 verdict→status 確定→CALLFUNCTION_RESULT(dedup_key=task:id)・single-flight。
- Scheduler: fake clock で when 到来→trigger・executor idle ガード。
- capabilities: create_task(非ブロッキング)・cancel(Pending のみ)・remind(message)・mutates_state/offered マーカ。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_task_phase1.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.clock import humanize_eta  # noqa: E402
from eve.pipeline.stimulus import StimulusKind  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.task import (  # noqa: E402
    CANCELLED, DONE, FAILED, PENDING, RUNNING,
    ReconcileTimer, Task, TaskExecutor, TaskStore,
    active_tasks_for_context, register_task_capabilities,
)

_passed = 0
_failed = 0
_dir = tempfile.mkdtemp(prefix="eve_task_")
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
    return os.path.join(_dir, f"tasks{_n}.jsonl")


def _iso(dt):
    return dt.isoformat()


class FakeRegistry:
    """name->結果文字列。失敗は「（…」で始める（registry 規約）。has は results のキー。"""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def has(self, name):
        return name in self.results

    def execute(self, name, args):
        self.calls.append((name, args))
        return self.results.get(name, f"（未対応「{name}」）")

    async def execute_async(self, name, args=None):
        # 実 CapabilityRegistry と同契約（同期能力は execute と同値）。
        return self.execute(name, args)


class FakeExec:
    def __init__(self):
        self.triggered = 0
        self._idle = True

    def is_idle(self):
        return self._idle

    def trigger(self):
        self.triggered += 1


# ========== Store ==========
async def t_store_roundtrip_terminal():
    f = _tmp()
    s = TaskStore(task_file=f, now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="t1", what="self_status"))
    a = s.set_status("t1", RUNNING)
    b = s.set_status("t1", DONE, result="ok")
    c = s.set_status("t1", PENDING)  # terminal-stays-terminal → False
    await s.shutdown()
    s2 = TaskStore(task_file=f, now_fn=lambda: BASE)
    await s2.initialize()
    t2 = s2.get("t1")
    await s2.shutdown()
    return a and b and (c is False) and t2 is not None and t2.status == DONE and t2.result == "ok"


async def t_store_due_now():
    clock = [BASE]
    s = TaskStore(task_file=_tmp(), now_fn=lambda: clock[0])
    await s.initialize()
    s.add(Task(task_id="now", what="x", when=None))
    s.add(Task(task_id="past", what="x", when=_iso(BASE - timedelta(seconds=10))))
    s.add(Task(task_id="future", what="x", when=_iso(BASE + timedelta(seconds=10))))
    due = {t.task_id for t in s.due_now()}
    await s.shutdown()
    return due == {"now", "past"}


async def t_store_claim_atomic():
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="a", what="x", when=None))
    c1 = s.claim_due()
    c2 = s.claim_due()  # 既に Running・他に Pending 無し → None
    await s.shutdown()
    return c1 is not None and c1.task_id == "a" and c1.status == RUNNING and c2 is None


async def t_store_orphan_ageout():
    f = _tmp()
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"task_id": "r1", "what": "x", "status": "Running",
                             "created_at": _iso(BASE), "updated_at": _iso(BASE)}) + "\n")
    s = TaskStore(task_file=f, now_fn=lambda: BASE + timedelta(seconds=10))
    await s.initialize()
    t = s.get("r1")
    await s.shutdown()
    return t is not None and t.status == FAILED and "孤児" in (t.failure_reason or "")


# ========== Executor ==========
async def t_executor_runs_verdict_reinject():
    q = StimulusQueue()
    reg = FakeRegistry({"self_status": "今は手が空いている", "boom": "（失敗しました）"})
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="ok1", what="self_status", when=None))
    s.add(Task(task_id="bad1", what="boom", when=None))
    ex = TaskExecutor(store=s, registry=reg, queue=q)
    ex.start(); ex.trigger()
    await asyncio.sleep(0.05)
    idle = ex.is_idle()
    await ex.stop()
    results = [st for st in q.snapshot() if st.kind == StimulusKind.CALLFUNCTION_RESULT]
    ok = s.get("ok1").status == DONE and s.get("bad1").status == FAILED
    dedup = len(results) == 2 and all((r.dedup_key or "").startswith("task:") for r in results)
    content_ok = any("今は手が空いている" in r.payload.content for r in results)
    await s.shutdown()
    return ok and dedup and content_ok and idle


async def t_executor_skips_future():
    q = StimulusQueue()
    reg = FakeRegistry({"self_status": "ok"})
    clock = [BASE]
    s = TaskStore(task_file=_tmp(), now_fn=lambda: clock[0])
    await s.initialize()
    s.add(Task(task_id="later", what="self_status", when=_iso(BASE + timedelta(seconds=30))))
    ex = TaskExecutor(store=s, registry=reg, queue=q)
    ex.start(); ex.trigger()
    await asyncio.sleep(0.04)
    await ex.stop()
    not_run = s.get("later").status == PENDING and q.qsize() == 0  # 未来は実行されない
    await s.shutdown()
    return not_run


async def t_executor_discard_on_cancel():
    # 実行中に取り消されたら executor は結果を再投入しない（破棄）・status は Cancelled のまま。
    q = StimulusQueue()
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="c1", what="self_status", when=None))

    class CancelDuringExec:
        def has(self, name):
            return True

        def execute(self, name, args):
            s.set_status("c1", CANCELLED)  # 実行の最中に取消が入ったと仮定
            return "状態は良好"

        async def execute_async(self, name, args=None):
            return self.execute(name, args)

    ex = TaskExecutor(store=s, registry=CancelDuringExec(), queue=q)
    ex.start(); ex.trigger()
    await asyncio.sleep(0.05)
    await ex.stop()
    discarded = q.qsize() == 0 and s.get("c1").status == CANCELLED
    await s.shutdown()
    return discarded


async def t_cancel_running_and_latest():
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    reg = CapabilityRegistry()
    register_task_capabilities(reg, s)
    s.add(Task(task_id="run1", what="self_status", when=None))
    s.claim_due()  # Running
    r1 = reg.execute("cancel_task", {})  # 直近未終了(run1・実行中)を取り消す（フォールバック経路）
    running_cancelled = s.get("run1").status == CANCELLED
    s.add(Task(task_id="p2", what="self_status", when=_iso(BASE + timedelta(seconds=100))))
    r2 = reg.execute("cancel_task", {})  # 直近の未終了(p2)
    await s.shutdown()
    return running_cancelled and s.get("p2").status == CANCELLED and "取り消した" in r1 and "取り消した" in r2


# ========== Scheduler ==========
async def t_scheduler_triggers_when_due():
    clock = [BASE]
    s = TaskStore(task_file=_tmp(), now_fn=lambda: clock[0])
    await s.initialize()
    fe = FakeExec()
    rt = ReconcileTimer(store=s, executor=fe, tick_sec=0.01)
    s.add(Task(task_id="f1", what="x", when=_iso(BASE + timedelta(seconds=5))))
    before = rt.tick()  # not due
    clock[0] = BASE + timedelta(seconds=6)
    after = rt.tick()  # due → trigger
    fe._idle = False
    busy = rt.tick()  # executor busy → no trigger
    await s.shutdown()
    return before is False and after is True and busy is False and fe.triggered == 1


# ========== capabilities ==========
async def t_capabilities():
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    reg = CapabilityRegistry()
    register_task_capabilities(reg, s)
    r_del = reg.execute("delegate_task", {"goal": "30秒後に時刻を教えて", "when_seconds": 30})
    r_dup = reg.execute("delegate_task", {"goal": "30秒後に時刻を教えて", "when_seconds": 30})  # 同一→dedup で弾く
    r_empty = reg.execute("delegate_task", {"goal": "  "})
    tid = s.list_all()[0].task_id
    t = s.get(tid)
    r_cancel = reg.execute("cancel_task", {"task_id": tid})
    cancelled = s.get(tid).status == CANCELLED
    offered = {x["function"]["name"] for x in reg.tool_schemas()}
    agent = {x["function"]["name"] for x in reg.agent_tool_schemas()}
    del_cap = reg._caps["delegate_task"]
    await s.shutdown()
    return (
        "引き受けた" in r_del and len(s) == 1  # dup は作られない
        and "もう予約してる" in r_dup and "ゴールが空" in r_empty
        and t.what == "" and t.goal == "30秒後に時刻を教えて"  # goal タスク
        and cancelled and "取り消した" in r_cancel and t.goal in r_cancel  # 空名でなく goal 名で報告(RC4)
        and del_cap.mutates_state is True and del_cap.report_result is False
        and offered == {"delegate_task", "cancel_task"}  # 応答LLM は委譲/取消だけ
        and "list_tasks" in agent and "self_status" in agent  # 実行系は agent 専用
        and "create_task" not in offered and "remind" not in offered and "list_tasks" not in offered
    )


async def t_duplicate_running_goal_rejected():
    """J-2 P1-2: 実行中(Running)の同一ゴール再委譲が dedup で弾かれる（重複検索/重複報告の防止）。

    実機事故(2026-07-18 E2E): 検索実行中に無関係な話題転換発話で応答LLMが同じゴールを
    再委譲し、新規タスクが作られて検索/報告が2重になった。旧実装は PENDING のみ照合の
    ため claim_due() で Running 化した瞬間から素通りしていた。
    """
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    reg = CapabilityRegistry()
    register_task_capabilities(reg, s)
    r1 = reg.execute("delegate_task", {"goal": "モンハンの最新アップデートを調べて"})
    running = s.claim_due()  # PENDING→Running(atomic)。when=None なので即 due。
    r2 = reg.execute("delegate_task", {"goal": "モンハンの最新アップデートを調べて"})  # 再委譲(事故の再現)
    await s.shutdown()
    return (
        "引き受けた" in r1 and len(s) == 1  # Running 中の再委譲でも新規タスクは作られない
        and running is not None and running.status == RUNNING
        and "今ちょうど調べてるところ" in r2  # Running 向けの自然な文言
    )


async def t_executor_drain_slow_async():
    # 遅い async 能力の実行中に stop(短drain) → 偽の報告刺激を残さず、store は Running のまま
    # （次回起動の孤児 age-out が回収＝既存設計と整合。レッドチーム §2-2）。
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    reg2 = CapabilityRegistry()
    started = asyncio.Event()

    async def hang(args):
        started.set()
        await asyncio.sleep(10)
        return "遅い"
    reg2.register(Capability(name="hang", description="", params_schema={}, async_handler=hang,
                             offered=False, agent_tool=True))
    q = StimulusQueue()
    ex = TaskExecutor(store=s, registry=reg2, queue=q)
    s.add(Task(task_id="h1", what="hang", when=None))
    ex.start(); ex.trigger()
    await asyncio.wait_for(started.wait(), timeout=2)
    await ex.stop(drain_timeout=0.1)
    ok = q.qsize() == 0 and s.get("h1").status == RUNNING
    await s.shutdown()
    return ok


async def t_active_tasks_context():
    # Fix#2: 応答LLM 注入用の一覧。Pending/Running のみ・丸め残り時間・task_id を出さない（leak ガード）。
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    s.add(Task(task_id="p1", what="", goal="50秒後に気持ちを教えて", when=_iso(BASE + timedelta(seconds=35))))
    s.add(Task(task_id="p0", what="", goal="すぐのやつ", when=None))
    s.add(Task(task_id="r1", what="", goal="PCの状態を調べて", when=None))
    s.set_status("r1", RUNNING)
    s.add(Task(task_id="d1", what="", goal="終わったやつ", when=None))
    s.set_status("d1", DONE, result="x")
    lines = active_tasks_for_context(s, now=BASE)
    joined = "\n".join(lines)
    ok = (len(lines) == 3
          and "「50秒後に気持ちを教えて」（あと約30秒で実行）" in joined
          and "「すぐのやつ」（まもなく実行）" in joined
          and "「PCの状態を調べて」（実行中）" in joined
          and "終わったやつ" not in joined  # terminal は出さない
          and "p1" not in joined and "r1" not in joined and "d1" not in joined)  # ID leak ガード
    await s.shutdown()
    return ok


async def t_active_tasks_limit_and_eta():
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    for i in range(7):
        s.add(Task(task_id=f"t{i}", what="", goal=f"ゴール{i}", when=_iso(BASE + timedelta(seconds=100 + i))))
    lines = active_tasks_for_context(s, now=BASE)
    ok_limit = len(lines) == 6 and lines[-1] == "・ほか2件"
    ok_eta = (humanize_eta(5) == "まもなく" and humanize_eta(35) == "あと約30秒"
              and humanize_eta(59) == "あと約50秒" and humanize_eta(60) == "あと約1分"
              and humanize_eta(3599) == "あと約59分" and humanize_eta(3600) == "あと約1時間")
    await s.shutdown()
    return ok_limit and ok_eta


async def main():
    check("Store: 往復+terminal-stays-terminal", await t_store_roundtrip_terminal())
    check("Store: due_now(when≤now のみ)", await t_store_due_now())
    check("Store: claim_due は atomic(Running 化)", await t_store_claim_atomic())
    check("Store: 起動時 Running 孤児→Failed", await t_store_orphan_ageout())
    check("Executor: 逐次実行+決定論verdict+再投入(task:)", await t_executor_runs_verdict_reinject())
    check("Executor: 未来タスクは実行しない", await t_executor_skips_future())
    check("Executor: 実行中キャンセルは結果破棄", await t_executor_discard_on_cancel())
    check("cancel: Running も / ID省略で直近", await t_cancel_running_and_latest())
    check("Scheduler: when 到来で trigger・busy ガード", await t_scheduler_triggers_when_due())
    check("capabilities: create/cancel/remind/markers", await t_capabilities())
    check("J-2 P1-2: Running中の同一ゴール再委譲は dedup で弾く", await t_duplicate_running_goal_rejected())
    check("Executor: 遅いasync能力中のstop→偽報告なし/Running温存", await t_executor_drain_slow_async())
    check("Fix#2: active_tasks_for_context(状態/ETA/ID leak なし)", await t_active_tasks_context())
    check("Fix#2: 一覧上限+humanize_eta 丸め境界", await t_active_tasks_limit_and_eta())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
