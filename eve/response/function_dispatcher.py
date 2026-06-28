"""FunctionDispatcher — Call-Function 実行のサイドカー（§9.4・queue 駆動 background worker）。

応答LLM が出した tool_calls を、応答完了後に orchestrator が `submit()`（O(1)・非ブロッキング）。
background worker が**逐次**（同時1・§J「並列にしない」）に CapabilityRegistry で実行し、結果を
`CALLFUNCTION_RESULT` 刺激として StimulusQueue へ再投入する。実行を応答経路で `await` しないので
**長時間タスクでも本体応答をブロックしない**（イブは実行中も喋れる）＝§J line 99。

precedent: `RagStore._write_worker` / `ConversationCache` の background write-worker（queue 駆動）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from ..pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind

logger = logging.getLogger(__name__)


def parse_tool_call(tc: Any) -> tuple[str, dict, str]:
    """litellm の tool_call オブジェクト or dict から (name, args, call_id) を頑健に取り出す。

    arguments は JSON 文字列で来る（壊れていれば {} に倒す＝raise しない）。
    """
    def g(o: Any, k: str) -> Any:
        return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

    call_id = g(tc, "id") or ""
    fn = g(tc, "function")
    name = (g(fn, "name") or "") if fn is not None else ""
    raw = (g(fn, "arguments") if fn is not None else None) or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args, call_id


class FunctionDispatcher:
    def __init__(self, *, registry, queue) -> None:
        self._registry = registry
        self._queue = queue
        self._inbox: "asyncio.Queue[Any]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def tool_schemas(self) -> list[dict]:
        return self._registry.tool_schemas()

    # --- ライフサイクル ---------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self, drain_timeout: float = 2.0) -> None:
        self._stopping = True
        try:
            await asyncio.wait_for(self._inbox.join(), timeout=drain_timeout)  # 進行中＋未処理を drain
        except asyncio.TimeoutError:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # --- 投入（応答完了後・O(1)・非ブロッキング）-------------------------

    def submit(self, tool_calls) -> None:
        """tool_calls を inbox に積むだけ（実行は背景 worker・本体応答をブロックしない）。"""
        for tc in tool_calls or []:
            self._inbox.put_nowait(tc)

    # --- 背景 worker（逐次・single-flight）--------------------------------

    async def _run(self) -> None:
        while not self._stopping:
            tc = await self._inbox.get()
            try:
                await self._handle_one(tc)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Call-Function 実行で例外（継続）")
            finally:
                self._inbox.task_done()

    async def _handle_one(self, tc: Any) -> None:
        name, args, call_id = parse_tool_call(tc)
        ok = self._registry.has(name)
        # read-only 能力は sub-ms。将来の blocking 能力は handler 側で executor へ逃がす。
        content = self._registry.execute(name, args)
        await self._queue.put(
            Stimulus(
                kind=StimulusKind.CALLFUNCTION_RESULT,
                payload=CallFunctionResult(function_name=name or "(unknown)", content=content, ok=ok),
                dedup_key=f"cf:{call_id}" if call_id else None,
            )
        )
