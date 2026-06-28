"""タスク管理の能力を CapabilityRegistry へ登録（既存 tool-calling 経路に乗る）。

- `create_task(action, when_seconds?, message?)`: 単純な予約タスクを作る（write）。応答LLM が
  「5分後に〜」「これはタスクにすべき」と**自律判断**して呼ぶ。
- `remind(message)`: 予約された時に message をそのまま結果へ返す純動作（executor が実行・**tool 非提供**）。
- `list_tasks` / `cancel_task(task_id)`: 一覧 / 取消（Pending のみ・write）。
handler は同期・非ブロッキング（store へ即追加 / 状態は store がコード一本化で持つ）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..capability.registry import Capability, CapabilityRegistry
from .schema import CANCELLED, PENDING, Task, new_task_id

# create_task が予約できる action（read-only 能力 + remind）。inc2 で delegate_task(goal) が増える。
_ALLOWED_ACTIONS = ["self_status", "pc_status", "remind"]


def register_task_capabilities(registry: CapabilityRegistry, store) -> None:
    def _create_task(args: dict) -> str:
        action = (args.get("action") or "").strip()
        if action not in _ALLOWED_ACTIONS:
            return f"（予約できない動作「{action}」です。{ '/'.join(_ALLOWED_ACTIONS) } から選んでください）"
        when = None
        ws = args.get("when_seconds")
        if isinstance(ws, (int, float)) and ws > 0:
            when = (datetime.now(timezone.utc) + timedelta(seconds=float(ws))).isoformat()
        task_args = {"message": args["message"]} if action == "remind" and args.get("message") else {}
        task = Task(task_id=new_task_id(), what=action, args=task_args, when=when, goal=args.get("goal"))
        store.add(task)
        eta = f"{int(ws)}秒後" if when else "すぐ"
        return f"タスクを作成したよ（{action} / {eta} / ID:{task.task_id}）。"

    def _remind(args: dict) -> str:
        return args.get("message") or "（リマインド内容が空でした）"

    def _list_tasks(args: dict) -> str:
        tasks = store.list_all()
        if not tasks:
            return "今は登録されているタスクは無いよ。"
        lines = [f"・{t.what}（{t.status}{'・' + t.when if t.when else ''}）ID:{t.task_id}" for t in tasks[-10:]]
        return "登録中のタスク:\n" + "\n".join(lines)

    def _cancel_task(args: dict) -> str:
        tid = (args.get("task_id") or "").strip()
        t = store.get(tid)
        if t is None:
            return f"（ID「{tid}」のタスクは見つからなかったよ）"
        if t.status != PENDING:
            return f"（「{t.what}」は既に {t.status} なので取り消せないよ）"
        store.set_status(tid, CANCELLED)
        return f"タスク「{t.what}」(ID:{tid}) を取り消したよ。"

    registry.register(Capability(
        name="create_task",
        description="後で自動実行する予約タスクを作る。例『5分後に状態を教えて』→ action=self_status, when_seconds=300。",
        params_schema={
            "action": {"type": "string", "enum": _ALLOWED_ACTIONS, "description": "予約する動作"},
            "when_seconds": {"type": "integer", "description": "何秒後に実行するか（省略=すぐ）"},
            "message": {"type": "string", "description": "action が remind の時に伝える内容"},
        },
        handler=_create_task, mutates_state=True,
    ))
    registry.register(Capability(
        name="remind", description="（内部）予約された内容を伝える。", params_schema={},
        handler=_remind, mutates_state=False, offered=False,
    ))
    registry.register(Capability(
        name="list_tasks", description="登録中の予約タスク一覧を見る。", params_schema={},
        handler=_list_tasks,
    ))
    registry.register(Capability(
        name="cancel_task", description="予約タスクを取り消す（Pending のみ）。",
        params_schema={"task_id": {"type": "string", "description": "取り消すタスクID"}},
        handler=_cancel_task, mutates_state=True,
    ))
