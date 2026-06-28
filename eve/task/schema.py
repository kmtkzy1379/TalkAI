"""Task レコードの schema（inc1 範囲・inc2 用フィールドは予約のみ）。

status は `Pending → Running → (Done | Failed | Cancelled)`。terminal は再遷移禁止（コードゲートは store 側）。
"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..clock import now_iso

PENDING = "Pending"
RUNNING = "Running"
DONE = "Done"
FAILED = "Failed"
CANCELLED = "Cancelled"
TERMINAL = {DONE, FAILED, CANCELLED}


def new_task_id() -> str:
    return "t_" + uuid.uuid4().hex[:10]


@dataclass
class Task:
    task_id: str
    what: str  # inc1: 実行する能力名（self_status / pc_status / remind 等）
    args: dict = field(default_factory=dict)
    when: Optional[str] = None  # UTC iso（絶対予約時刻）。None=応答後すぐ
    status: str = PENDING
    result: Optional[str] = None
    failure_reason: Optional[str] = None
    attempts: int = 0
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    # --- inc2 (TaskAgent) 用に予約（inc1 では未使用）---
    goal: Optional[str] = None          # 自然文ゴール（delegate_task）
    reflection: Optional[str] = None    # 失敗分析の持ち越し
    parent_id: Optional[str] = None     # 再計画チェーン
    depth: int = 0                      # 再計画の深さ（暴走防止）
    success_criterion: Optional[str] = None
    verdict_kind: str = "deterministic"  # deterministic | judged(inc3)
    depends_on: list = field(default_factory=list)
    input_from: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        # 未知キーは無視して安全に復元（schema 進化への耐性）。
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
