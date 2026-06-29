"""TaskAgent — ゴールを完遂する境界つき native ツール呼出ループ（inc2）。

応答LLM は `delegate_task(goal=自然文)` で**何をしたいか**だけ委譲する。TaskAgent（賢い `task`
ロールの LLM・非同期）がゴールを受け、次のループを回して**最終結果を1つ**返す:

    LLM(tools=能力) → tool_calls → コードが能力を実行 → 結果を messages へ戻す → LLM が次の手 or 完了

失敗した tool 結果を見て**その場で別アプローチ**を試す（暗黙の Reflexion）。達成したら関数を呼ばず
**短い最終結果（文章）**を返す＝それで完了。**コードが境界を所有**する（V1 修復の核）:
- `max_steps`（OpenAI Agents SDK 既定=5 相当）/ `timeout`（壁時計でなく monotonic）で打ち切り。
- 毎 step `store.get().status==Cancelled` を見て**中断**（結果は破棄＝None を返す）。
- status 確定・終端不変は executor/store 側。LLM は Done を主張できない（最終文を返すだけ）。

eve.task を eve.response から独立に保つため tool_call パースは本ファイルに内製（function_dispatcher
の parse_tool_call と同等）。non-streaming `complete()` は完全な tool_calls を返す（断片結合は不要）。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..clock import now_mono
from .schema import CANCELLED

logger = logging.getLogger(__name__)


TASK_AGENT_SYSTEM = (
    "あなたはAI VTuber「イブ」のタスク実行担当です。与えられたゴールを、提供された関数(tools)を"
    "使って達成してください。各関数の結果を見て次の手を決め、達成できたら**短い最終結果を一言**で"
    "返してください（その時は関数を呼ばず、文章だけを返す）。\n"
    "・失敗した結果が返ってきたら、別の手段・引数で**やり直してよい**。\n"
    "・どうしても達成できない/できない依頼なら、**正直に理由を述べて**終えてください（憶測で成功と言わない）。\n"
    "・最終結果はイブがユーザに報告する一言になります。簡潔に、事実だけを。"
)


def _g(o, k):
    """dict / オブジェクト両対応の属性取り出し（litellm resp は object・fake は dict）。"""
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _extract(message) -> tuple[str, list[tuple[str, str, str]]]:
    """message から (content, [(call_id, name, raw_args_json)]) を頑健に取り出す。raise しない。"""
    content = _g(message, "content") or ""
    calls: list[tuple[str, str, str]] = []
    for tc in _g(message, "tool_calls") or []:
        cid = _g(tc, "id") or ""
        fn = _g(tc, "function")
        name = (_g(fn, "name") or "") if fn is not None else ""
        raw = (_g(fn, "arguments") if fn is not None else None) or "{}"
        calls.append((cid, name, raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)))
    return content, calls


def _parse_args(raw: str) -> dict:
    try:
        a = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        a = {}
    return a if isinstance(a, dict) else {}


class TaskAgent:
    """ゴール → 境界つきツール呼出ループ → 最終結果1つ。コードが境界(max_steps/timeout/cancel)を所有。"""

    def __init__(
        self, *, registry, model_registry, store,
        max_steps: int = 6, timeout_sec: float = 30.0, now_fn=now_mono,
    ) -> None:
        self._registry = registry          # CapabilityRegistry（tool_schemas + execute）
        self._model = model_registry        # ModelRegistry（complete("task", ...)）
        self._store = store                  # TaskStore（cancel 検知に get()）
        self._max_steps = max(1, int(max_steps))
        self._timeout = float(timeout_sec)
        self._now = now_fn

    def _cancelled(self, task_id: str) -> bool:
        t = self._store.get(task_id)
        return t is None or t.status == CANCELLED

    async def run(self, goal: str, task_id: str) -> Optional[str]:
        """ゴールを完遂し最終結果文字列を返す。実行中に取り消されたら None（executor が破棄）。"""
        deadline = self._now() + self._timeout
        messages: list[dict] = [
            {"role": "system", "content": TASK_AGENT_SYSTEM},
            {"role": "user", "content": f"ゴール: {goal}"},
        ]
        last_content = ""
        for step in range(self._max_steps):
            if self._cancelled(task_id):
                logger.info("🗒 TaskAgent: 実行中に取消（破棄） task=%s", task_id)
                return None
            if self._now() > deadline:
                logger.info("🗒 TaskAgent: timeout task=%s step=%d", task_id, step)
                return last_content or "（時間切れで完了できなかった）"
            try:
                resp = await self._model.complete(
                    "task", messages, tools=self._registry.tool_schemas(),
                )
            except Exception:
                logger.exception("TaskAgent LLM 呼び出し失敗 task=%s", task_id)
                return last_content or "（タスク処理中にエラーが起きて完了できなかった）"

            choices = _g(resp, "choices") or []
            message = _g(choices[0], "message") if choices else None
            if message is None:
                return last_content or "（応答が空で完了できなかった）"
            content, calls = _extract(message)
            if content:
                last_content = content
            if not calls:
                # 関数を呼ばない＝完了（達成 or 正直な失敗報告）。
                return content or last_content or "（結果が空でした）"

            # tool_calls あり → assistant メッセージ + 各 tool 結果を積んで次の step へ。
            messages.append({
                "role": "assistant",
                "content": content or None,  # tool_calls 同伴時は content 省略可
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": raw}}
                    for (cid, name, raw) in calls
                ],
            })
            for (cid, name, raw) in calls:
                result = self._registry.execute(name, _parse_args(raw))
                logger.info("🗒 TaskAgent step=%d %s -> %s", step, name, result[:50])
                messages.append({"role": "tool", "tool_call_id": cid, "content": result})

        logger.info("🗒 TaskAgent: max_steps 到達 task=%s", task_id)
        return last_content or "（手順の上限まで試したけど完了できなかった）"
