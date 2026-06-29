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
from .schema import CANCELLED, PENDING, RUNNING, TERMINAL, Task, new_task_id

# 予約 action から除外するタスク管理能力（自己再帰/無意味を防ぐ）。
_MGMT = {"create_task", "list_tasks", "cancel_task"}


def register_task_capabilities(registry: CapabilityRegistry, store) -> None:
    def _allowed_actions() -> set:
        # 予約できる action = 提示中(offered)で状態を変えない能力 + remind（内部）。
        # 動的＝将来 search/screen-op を能力層に足すと自動で予約可能になる（テストの flaky も同様）。
        names = {
            c.name for c in registry._caps.values()
            if c.offered and not c.mutates_state and c.name not in _MGMT
        }
        names.add("remind")
        return names

    def _create_task(args: dict) -> str:
        action = (args.get("action") or "").strip()
        if action not in _allowed_actions():
            return f"（予約できない動作「{action}」です。今できるのは {'/'.join(sorted(_allowed_actions()))} だよ）"
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
        if not tid:
            # ID 省略 → 直近の未終了タスク（待機 or 実行中）を取り消す（「さっきの予約キャンセルして」）。
            actives = [t for t in store.list_all() if t.status in (PENDING, RUNNING)]
            if not actives:
                return "（今は取り消せる予約タスクは無いよ）"
            t = actives[-1]; tid = t.task_id
        else:
            t = store.get(tid)
            if t is None:
                return f"（ID「{tid}」のタスクは見つからなかったよ）"
        if t.status in TERMINAL:
            return f"（「{t.what}」は既に {t.status} なので取り消せないよ）"
        # Pending/Running とも Cancelled へ（実行中に取り消した場合は executor が結果を破棄する）。
        store.set_status(tid, CANCELLED)
        return f"予約していた「{t.what}」を取り消したよ。"

    registry.register(Capability(
        name="create_task",
        description="後で自動実行する予約タスクを作る。action は実行する能力名（例: self_status / pc_status / remind）。"
                    "例『5分後に状態を教えて』→ action=self_status, when_seconds=300。",
        params_schema={
            "action": {"type": "string", "description": "予約する能力名（提示中の能力か remind）"},
            "when_seconds": {"type": "integer", "description": "何秒後に実行するか（省略=すぐ）"},
            "message": {"type": "string", "description": "action が remind の時に伝える内容"},
        },
        handler=_create_task, mutates_state=True, report_result=False,  # 成功 ack は応答本文が担う（重複/先出し防止）
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
        name="cancel_task",
        description="予約・実行中のタスクを取り消す。**task_id は省略可**＝直近の予約を取り消す。"
                    "『キャンセルして』『やっぱりいい』等と言われたら、list_tasks で確認せず**直接これを1回呼ぶ**。",
        params_schema={"task_id": {"type": "string", "description": "取り消すタスクID（省略すると直近の予約）"}},
        handler=_cancel_task, mutates_state=True,
    ))
