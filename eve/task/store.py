"""TaskStore — タスクの永続/状態所有（`RagStore._write_worker` + `ConversationCache` 先例）。

**status を書くのは `set_status()` / `claim_due()` だけ**（＝executor 経由・単一書込）。terminal は
再遷移禁止をコードゲートで弾く（V1「完了所有権の二転三転」の直接修復）。LLM は create/cancel を
*要求*するのみで Done を主張できない。

永続は JSONL の追記イベントログ（add も status 更新も全 Task を1行追記）。読み込みは task_id 毎に
最新行を採用。起動時に **Running の孤児**（前回未完）を Failed へ落とす（孤児 age-out）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from ..clock import now_iso
from .schema import FAILED, PENDING, RUNNING, TERMINAL, Task

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class TaskStore:
    def __init__(
        self,
        task_file: Optional[str] = None,
        max_tasks: int = 500,
        orphan_timeout_sec: float = 3600.0,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        path = task_file or "tasks.jsonl"
        self.task_path = Path(path)
        if not self.task_path.is_absolute():
            self.task_path = _REPO_ROOT / self.task_path
        self.max_tasks = max_tasks
        self.orphan_timeout_sec = orphan_timeout_sec
        self._now = now_fn
        self._tasks: "dict[str, Task]" = {}  # task_id -> 現在状態（loop 所有・単一書込）
        self._write_queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
        self._write_task: Optional[asyncio.Task] = None

    # --- ライフサイクル ---------------------------------------------------

    async def initialize(self) -> None:
        await self._load()
        self._age_out_orphans()
        self._write_task = asyncio.create_task(self._write_worker())

    async def _load(self) -> None:
        if not self.task_path.exists():
            return

        def _read() -> list[dict]:
            out: list[dict] = []
            with open(self.task_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return out

        try:
            records = await asyncio.to_thread(_read)
        except Exception:
            logger.exception("タスクログ読み込み失敗（空で続行）: %s", self.task_path)
            return
        for r in records:  # task_id 毎に最新行が勝つ（イベントログ）
            tid = r.get("task_id")
            if tid:
                self._tasks[tid] = Task.from_dict(r)
        logger.info("タスクを復元: %d 件", len(self._tasks))

    def _age_out_orphans(self) -> None:
        """前回未完の Running を Failed へ（executor が保持していないので孤児）。"""
        now = self._now()
        for t in self._tasks.values():
            if t.status == RUNNING:
                age = (now - (_parse_iso(t.updated_at) or now)).total_seconds()
                if age >= 0:  # 起動時の Running は全て孤児（保持していた executor が居ない）
                    t.status = FAILED
                    t.failure_reason = "孤児: 前回セッションで未完のまま中断"
                    t.updated_at = now_iso()
                    self._enqueue_persist(t)

    async def shutdown(self) -> None:
        if self._write_task is not None:
            await self._write_queue.put(None)
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
            self._write_task = None

    # --- 書き込み（status はここ経由のみ）---------------------------------

    def add(self, task: Task) -> None:
        """新規タスクを登録（put_nowait＝非ブロッキング。create_task から呼ぶ）。"""
        self._tasks[task.task_id] = task
        self._enqueue_persist(task)
        self._enforce_max()

    def set_status(
        self, task_id: str, new_status: str, *, result: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> bool:
        """status を更新（**唯一の更新口**）。terminal は no-op で False（再遷移禁止）。"""
        t = self._tasks.get(task_id)
        if t is None:
            return False
        if t.status in TERMINAL:
            return False  # terminal-stays-terminal（V1 修復のコードゲート）
        t.status = new_status
        if result is not None:
            t.result = result
        if failure_reason is not None:
            t.failure_reason = failure_reason
        t.updated_at = now_iso()
        self._enqueue_persist(t)
        return True

    def claim_due(self) -> Optional[Task]:
        """due な Pending を1件 **atomic に Running へ**して返す（await を挟まない＝二重発火防止）。"""
        for t in self._tasks.values():
            if t.status == PENDING and self._is_due(t):
                t.status = RUNNING
                t.attempts += 1
                t.updated_at = now_iso()
                self._enqueue_persist(t)
                return t
        return None

    # --- 読み取り（同期）-------------------------------------------------

    def _is_due(self, t: Task) -> bool:
        if t.when is None:
            return True
        w = _parse_iso(t.when)
        return w is None or w <= self._now()

    def due_now(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == PENDING and self._is_due(t)]

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_all(self) -> list[Task]:
        return list(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    # --- 内部 -------------------------------------------------------------

    def _enforce_max(self) -> None:
        if len(self._tasks) <= self.max_tasks:
            return
        # 上限超過: 古い terminal タスクから捨てる（in-memory のみ・ログは残る）。
        terminals = [tid for tid, t in self._tasks.items() if t.status in TERMINAL]
        for tid in terminals[: len(self._tasks) - self.max_tasks]:
            self._tasks.pop(tid, None)

    def _enqueue_persist(self, t: Task) -> None:
        try:
            self._write_queue.put_nowait(t.to_dict())
        except Exception:
            logger.exception("タスク永続キュー投入失敗（メモリ層は更新済み）")

    async def _write_worker(self) -> None:
        while True:
            try:
                record = await self._write_queue.get()
                if record is None:
                    self._write_queue.task_done()
                    break
                try:
                    await asyncio.to_thread(self._append_line, record)
                finally:
                    self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("タスク追記失敗（継続）")

    def _append_line(self, record: dict) -> None:
        with open(self.task_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
