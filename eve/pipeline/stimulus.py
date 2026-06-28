"""刺激(Stimulus) = 応答LLM を起動する入力の共通エンベロープ。

全入力源（ユーザ発話 / CallFunction 結果 / 自発発話 / 画面更新）はこの型で
StimulusQueue に入る。実プロデューサ（STT/VLM/FeedbackLLM/task executor）は
将来この Stimulus を作るだけでよい（将来互換の要）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from ..clock import Stamp


class StimulusKind(IntEnum):
    """値が小さいほど高優先（基本優先度）。"""

    USER_UTTERANCE = 0
    CALLFUNCTION_RESULT = 1
    AUTONOMOUS_SPEECH = 2
    VISION_UPDATE = 3


@dataclass
class Stimulus:
    kind: StimulusKind
    payload: Any = None
    stamp: Stamp = field(default_factory=Stamp.now)
    # 同じ merge_key を持つ待機刺激は最新で1件に畳む（vision/feedback の連発対策）
    merge_key: Optional[str] = None
    # 同じ dedup_key を持つ刺激は二重投入を捨てる（CallFunction 結果の重複・二次注入対策）
    dedup_key: Optional[str] = None

    @property
    def base_priority(self) -> int:
        return int(self.kind)


@dataclass(frozen=True)
class CallFunctionResult:
    """Call-Function 実行結果の刺激 payload（AutonomousSpeech と同形）。

    content = 応答LLM へ渡す**人間可読**の結果文（再投入時に「# 機能実行結果」へそのまま入る）。
    function_name = 実行した能力名（ログ/文脈用）。ok = 成否（失敗も正直に伝えるため保持）。
    """

    function_name: str
    content: str
    ok: bool = True
