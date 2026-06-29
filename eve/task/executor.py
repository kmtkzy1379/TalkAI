"""TaskExecutor — due タスクをコード実行するサイドカー（`FeedbackWorker` 先例・single-flight）。

inc1 は **コード実行のみ**（TaskLLM は inc2）: `claim_due()` で 1件ずつ atomic に Running 化 →
Capability 層を実行 → **決定論 verdict**（失敗マーカ `（…）` で Failed・他 Done）→ `set_status` →
結果を `CALLFUNCTION_RESULT` として StimulusQueue へ再投入（Eve が「やっといたよ」と報告できる）。
実行を応答経路で await しない＝本体応答をブロックしない（§9.4）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
from .schema import DONE, FAILED

logger = logging.getLogger(__name__)


class TaskExecutor:
    def __init__(self, *, store, registry, queue, agent=None) -> None:
        self._store = store
        self._registry = registry
        self._queue = queue
        self._agent = agent  # inc2 TaskAgent（None=goal タスクは未対応構成）
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._idle = asyncio.Event()
        self._idle.set()

    # --- ライフサイクル ---------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self, drain_timeout: float = 2.0) -> None:
        self._stopping = True
        self._event.set()
        if not self._idle.is_set():
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=drain_timeout)
            except asyncio.TimeoutError:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def trigger(self) -> None:
        self._event.set()

    def is_idle(self) -> bool:
        return self._idle.is_set() and not self._event.is_set()

    async def _run_forever(self) -> None:
        while not self._stopping:
            await self._event.wait()
            self._event.clear()
            if self._stopping:
                break
            try:
                await self._process_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("TaskExecutor 処理で例外（継続）")

    async def _process_once(self) -> None:
        self._idle.clear()
        try:
            while not self._stopping:
                task = self._store.claim_due()  # atomic Pending→Running（await を挟まない）
                if task is None:
                    break
                await self._run_task(task)
        finally:
            self._idle.set()

    async def _run_task(self, task) -> None:
        # 分岐: goal あり＝TaskAgent ループ（inc2）/ なし＝単純な能力実行（inc1）。
        if task.goal:
            await self._run_goal(task)
        else:
            await self._run_capability(task)

    async def _run_capability(self, task) -> None:
        content = self._registry.execute(task.what, task.args or {})
        # 決定論 verdict: 失敗/未対応マーカは「（…」で始まる（registry 規約）。
        ok = bool(self._registry.has(task.what)) and not content.startswith("（")
        report = f"（予約していたタスク）{content}"
        await self._finish(task, ok, content, report, label=task.what)

    async def _run_goal(self, task) -> None:
        if self._agent is None:
            await self._finish(task, False, "（ゴール実行はこの構成では未対応）",
                               "（任せてくれたタスク）ごめん、今はそれを自分で進められないみたい。",
                               label="goal")
            return
        result = await self._agent.run(task.goal, task.task_id)
        if result is None:
            # 実行中に取り消された（TaskAgent が検知）→ 結果を破棄して報告しない。
            logger.info("🗒 タスク(goal) %s は実行中に取消されたため破棄", task.task_id)
            return
        ok = not result.startswith("（")  # ゆるい成否（正直な失敗文は「（…」で始める規約）
        await self._finish(task, ok, result, f"（任せてくれたタスク）{result}", label="goal")

    async def _finish(self, task, ok: bool, content: str, report: str, *, label: str) -> None:
        advanced = self._store.set_status(
            task.task_id, DONE if ok else FAILED,
            result=content, failure_reason=None if ok else content,
        )
        if not advanced:
            # 完了直前/実行中に取り消された（既に terminal=Cancelled）→ **結果を破棄して報告しない**
            # （ユーザ設計: 完了させて結果は応答LLM に届けない）。status は Cancelled のまま。
            logger.info("🗒 タスク %s は取消されたため結果を破棄", label)
            return
        logger.info("🗒 タスク %s [%s] %s", label, "Done" if ok else "Failed", content[:60])
        await self._queue.put(
            Stimulus(
                kind=StimulusKind.CALLFUNCTION_RESULT,
                payload=CallFunctionResult(function_name=label, content=report, ok=ok),
                dedup_key=f"task:{task.task_id}",
            )
        )
