"""CancelResolver — タスク取消を**タスク側**で解決するサイドカー（dispatcher の近クローン・別コルーチン）。

応答LLM は `cancel_task(reference)` で取消意図（ユーザの言い回し）を渡すだけ。ここが件数で解決:
- **0件** → コード即時「今止めるタスクは無いよ」（LLM 不要）。
- **1件** → コードが即時キャンセル→「『{名前}』を止めたよ」と名前つきで報告（LLM 不要・停止は安全操作）。
- **2件以上** → タスクLLM(gpt-5.5) が store を見て reference をファジー照合（match/none/ambiguous）。

**executor(single-flight) 経由にしない**＝取消が実行中タスクの後ろで待つのを防ぐ（別コルーチンで並行）。
実行中 goal は agent の毎step CANCELLED チェックが中断・破棄する（agent/executor 側で既存）。

**TOCTOU**: LLM await の前後で状態が変わり得る→**書込直前に store を読み直し**、`set_status` の戻り値 bool で
分岐（False=既に終了→「もう終わってた」）。スナップショットから「取り消した」と誤報しない。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
from .schema import CANCELLED, PENDING, RUNNING

logger = logging.getLogger(__name__)


@dataclass
class CancelRequest:
    reference: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


CANCEL_SYSTEM = (
    "あなたはAI VTuber「イブ」のタスク取消担当です。ユーザが取り消したい対象(reference)と、現在の"
    "未終了タスク一覧が与えられます。どれを取り消すべきか判断し、次の JSON だけを返してください:\n"
    '{"decision":"match|none|ambiguous","task_id":"<一致したIDまたは空>","message":"<ユーザに話す一言>"}\n'
    "・match: reference が1つのタスクに対応。task_id を入れ、message は「『〜』を止めるね」等。\n"
    "・none: 対応するタスクが無い。message は「そんなタスクは入ってないよ。今あるのは〜」等。\n"
    "・ambiguous: 複数該当で決められない。message は候補を挙げ「〜と〜、どっち?」と聞く。\n"
    "message は簡潔に事実だけ。JSON 以外は書かない。"
)


def _g(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _content(resp) -> str:
    choices = _g(resp, "choices") or []
    if not choices:
        return ""
    msg = _g(choices[0], "message")
    return (_g(msg, "content") or "") if msg is not None else ""


def _display_name(t) -> str:
    return t.goal or t.what or "(無題)"


class CancelResolver:
    def __init__(self, *, store, model_registry, queue) -> None:
        self._store = store
        self._model = model_registry
        self._queue = queue
        self._inbox: "asyncio.Queue[CancelRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # --- ライフサイクル（dispatcher と同形）---
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self, drain_timeout: float = 3.0) -> None:
        self._stopping = True
        try:
            await asyncio.wait_for(self._inbox.join(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def submit(self, reference: str) -> None:
        """取消意図(ユーザの言い回し)を積むだけ（O(1)・非ブロッキング）。"""
        self._inbox.put_nowait(CancelRequest(reference=reference or ""))

    async def _run(self) -> None:
        while not self._stopping:
            req = await self._inbox.get()
            try:
                await self._handle_one(req)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("キャンセル解決で例外（フォールバック報告）")
                await self._report(req, "ごめん、取り消しの確認でエラーが起きたよ。", ok=False)
            finally:
                self._inbox.task_done()

    def _actives(self):
        return [t for t in self._store.list_all() if t.status in (PENDING, RUNNING)]

    async def _handle_one(self, req: CancelRequest) -> None:
        actives = self._actives()
        if not actives:
            ref = req.reference.strip()
            msg = f"「{ref}」ってタスクは入ってないよ。今は止められる予約も無いよ。" if ref else "今は止められるタスクは無いよ。"
            await self._report(req, msg, ok=False)
        elif len(actives) == 1:
            # 1件＝曖昧さ無し→コードで即時キャンセル（LLM 不要・停止は速く確実に）。
            await self._cancel_and_report(req, actives[0].task_id)
        else:
            # 2件以上→タスクLLM がファジー照合。
            await self._resolve_with_llm(req, actives)

    async def _resolve_with_llm(self, req: CancelRequest, actives) -> None:
        listing = "\n".join(f'- id={t.task_id} 内容="{_display_name(t)}" 状態={t.status}' for t in actives)
        messages = [
            {"role": "system", "content": CANCEL_SYSTEM},
            {"role": "user", "content": f"取り消したい対象: {req.reference or '(指定なし)'}\n現在の未終了タスク:\n{listing}"},
        ]
        try:
            resp = await self._model.complete("task", messages)
        except Exception:
            logger.exception("キャンセル解決 LLM 失敗")
            await self._report(req, "ごめん、どのタスクか確認できなかったよ。", ok=False)
            return
        decision, task_id, message = self._parse(resp)
        if decision == "match" and task_id:
            await self._cancel_and_report(req, task_id, llm_message=message)
        else:
            # none / ambiguous / 解析失敗 → status は変えず message を報告。
            await self._report(req, message or "どのタスクのことか分からなかったよ。", ok=False)

    async def _cancel_and_report(self, req: CancelRequest, task_id: str, llm_message: str = "") -> None:
        # TOCTOU: 書込直前に読み直し、set_status の bool で成否判定（スナップショットから誤報しない）。
        t = self._store.get(task_id)
        if t is None:
            await self._report(req, "そのタスクはもう無くなってたよ。", ok=False)
            return
        name = _display_name(t)
        advanced = self._store.set_status(task_id, CANCELLED)
        if not advanced:
            await self._report(req, f"「{name}」はもう終わってたよ。", ok=False)
            return
        await self._report(req, llm_message or f"「{name}」を止めたよ。", ok=True)

    async def _report(self, req: CancelRequest, message: str, *, ok: bool) -> None:
        await self._queue.put(Stimulus(
            kind=StimulusKind.CALLFUNCTION_RESULT,
            payload=CallFunctionResult(function_name="cancel_task", content=message, ok=ok),
            dedup_key=f"cancel:{req.id}",
        ))

    @staticmethod
    def _parse(resp) -> tuple[str, str, str]:
        raw = _content(resp).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)  # フェンスや前後テキストを許容
        if m:
            try:
                d = json.loads(m.group(0))
                if isinstance(d, dict):
                    return (str(d.get("decision", "")), str(d.get("task_id") or "").strip(), str(d.get("message", "")))
            except (ValueError, TypeError):
                pass
        return ("none", "", "")
