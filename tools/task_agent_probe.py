r"""TaskAgent を実 gpt-5.5 で直接駆動して**ループ自体**を検証（応答LLM の委譲判断を介さない）。

Tier-2 通しでは応答LLM が read-only 能力を直接呼んでしまい agent が走らなかったので、ここでは
agent.run(goal, task_id) を直接叩いて、実モデルが「tool を連鎖→最終要約1つ」「失敗を正直に」
「達成不能を境界で打ち切り」できるかを見る。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\task_agent_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.task import Task, TaskAgent, TaskStore, new_task_id  # noqa: E402


async def main():
    reg = ModelRegistry(overrides={"task": "openai/gpt-5.5"})
    caps = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)

    def flaky(a):
        raise ConnectionError("外部サービスに接続できませんでした")
    caps.register(Capability("external_service_status", "外部サービスの稼働状況を確認する。引数なし。", {}, flaky))

    steps = []
    _exec = caps.execute

    def wrap(name, args=None):
        out = _exec(name, args)
        steps.append(name)
        return out
    caps.execute = wrap  # type: ignore

    store = TaskStore(task_file=os.path.join(tempfile.mkdtemp(), "t.jsonl"))
    await store.initialize()
    agent = TaskAgent(registry=caps, model_registry=reg, store=store, max_steps=6, timeout_sec=30.0)

    async def run(goal):
        steps.clear()
        tid = new_task_id()
        store.add(Task(task_id=tid, what="", goal=goal))
        store.claim_due()  # Running（agent の cancel 検知が通る状態）
        out = await agent.run(goal, tid)
        print(f"\n■ goal: {goal}")
        print(f"  agent が呼んだ tool: {steps}")
        print(f"  最終結果: {out}")

    await run("PCの状態とイブ自身の状態を両方調べて、ひとことでまとめて。")
    await run("外部サービス(external_service_status)が今使えるか調べて、状況を教えて。")
    await run("3を5で割った正確な小数値を、提供された関数だけで厳密に求めて。")  # 達成不能寄り→境界
    await store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
