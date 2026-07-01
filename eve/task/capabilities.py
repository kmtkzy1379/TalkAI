"""タスク管理の能力を CapabilityRegistry へ登録（**応答LLM は委譲/取消の意図を渡すだけ**）。

- `delegate_task(goal, when_seconds?)`: 自然文ゴールを TaskAgent に委譲。即時も後回しも状態確認も全部これ。
- `cancel_task(reference?)`: 取消の意図をそのまま渡す（どのタスクかはタスク側が解決。Part B で CancelResolver 化）。
- `list_tasks`: **agent 専用**(agent_tool)＝「予約教えて」も委譲で答える。
実行系能力(self_status/pc_status/将来の検索・画面操作)は agent_tool=True で TaskAgent 専用。
handler は同期・非ブロッキング（store へ即追加 / 状態は store がコード一本化で持つ）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from ..capability.registry import Capability, CapabilityRegistry
from .schema import CANCELLED, PENDING, RUNNING, TERMINAL, Task, new_task_id

_DEDUP_TOL_SEC = 20.0  # 同一 goal で when がこの秒数以内なら重複予約とみなす


def _display_name(t) -> str:
    """一覧/取消の表示名。goal タスクは what="" なので goal を優先（RC4 空名対策）。"""
    return t.goal or t.what or "(無題)"


def _iso_in(seconds) -> Optional[str]:
    if isinstance(seconds, (int, float)) and seconds > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=float(seconds))).isoformat()
    return None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _near(a: Optional[datetime], b: Optional[datetime], tol: float = _DEDUP_TOL_SEC) -> bool:
    if a is None and b is None:
        return True  # 両方 即時（when None）は同一窓
    if a is None or b is None:
        return False
    return abs((a - b).total_seconds()) <= tol


def register_task_capabilities(registry: CapabilityRegistry, store) -> None:
    def _find_duplicate(goal: str, when: Optional[str]):
        # 同じ goal かつ when が近接の PENDING があれば重複（混乱ターンの再作成を弾く＝RC2）。
        target = _parse_iso(when)
        for t in store.list_all():
            if t.status == PENDING and (t.goal or "").strip() == goal and _near(_parse_iso(t.when), target):
                return t
        return None

    def _delegate_task(args: dict) -> str:
        goal = (args.get("goal") or "").strip()
        if not goal:
            return "（ゴールが空でした）"
        when = _iso_in(args.get("when_seconds"))
        dup = _find_duplicate(goal, when)
        if dup is not None:
            return f"それはもう予約してるよ（ID:{dup.task_id}）。"
        task = Task(task_id=new_task_id(), what="", goal=goal, when=when)
        store.add(task)
        eta = f"{int(args['when_seconds'])}秒後" if when else "すぐ"
        return f"タスクを引き受けたよ（{goal[:24]} / {eta} / ID:{task.task_id}）。"

    def _list_tasks(args: dict) -> str:
        tasks = store.list_all()
        if not tasks:
            return "今は登録されているタスクは無いよ。"
        lines = [f"・{_display_name(t)}（{t.status}{'・' + t.when if t.when else ''}）ID:{t.task_id}" for t in tasks[-10:]]
        return "登録中のタスク:\n" + "\n".join(lines)

    def _cancel_task(args: dict) -> str:
        # ※Part A 暫定（Part B で CancelResolver によるファジー解決へ置換）。id 省略→直近未終了。
        tid = (args.get("task_id") or "").strip()
        if not tid:
            actives = [t for t in store.list_all() if t.status in (PENDING, RUNNING)]
            if not actives:
                return "（今は取り消せる予約タスクは無いよ）"
            t = actives[-1]; tid = t.task_id
        else:
            t = store.get(tid)
            if t is None:
                return f"（ID「{tid}」のタスクは見つからなかったよ）"
        if t.status in TERMINAL:
            return f"（「{_display_name(t)}」は既に {t.status} なので取り消せないよ）"
        store.set_status(tid, CANCELLED)
        return f"予約していた「{_display_name(t)}」を取り消したよ。"

    registry.register(Capability(
        name="delegate_task",
        description="やりたいこと・調べ物・状態確認・後回しの予約を、賢いタスク担当(TaskAgent)に任せる。"
                    "goal に自然文で『何を達成/報告したいか』を、**ユーザの言い回しをそのまま**書く"
                    "（例: 30秒後に今の時刻を教えて / PCとイブの状態を調べてまとめて）。どんな簡単な事も"
                    "自分でやらずここへ。when_seconds 省略=すぐ、指定=その秒後。呼んだら『やっとくね』とだけ言う。",
        params_schema={
            "goal": {"type": "string", "description": "達成/報告したいこと（自然文・ユーザの言い回しを保つ）"},
            "when_seconds": {"type": "integer", "description": "何秒後に始めるか（省略=すぐ）"},
        },
        handler=_delegate_task, mutates_state=True, report_result=False,  # ack は応答本文が担う
    ))
    registry.register(Capability(
        name="list_tasks", description="登録中の予約タスク一覧を見る。引数なし。", params_schema={},
        handler=_list_tasks, offered=False, agent_tool=True,  # agent 専用（「予約教えて」も委譲で答える）
    ))
    registry.register(Capability(
        name="cancel_task",
        description="タスクの取り消し。『キャンセルして』『やっぱりいい』等と言われたら、ユーザの言い回しを"
                    "reference にそのまま入れて1回呼ぶ（どのタスクかはタスク側が判断する）。",
        params_schema={"reference": {"type": "string", "description": "取り消したいタスクのユーザの言い回し（省略可）"}},
        handler=_cancel_task, mutates_state=True,
    ))
