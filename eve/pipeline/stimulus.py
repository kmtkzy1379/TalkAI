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
