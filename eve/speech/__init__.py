"""F5 発話判定 / 自発発話パッケージ。

5秒沈黙で発話判定LLM(should_speak)が走り、中核原理 surprise を必須ゲートに自発発話するか決める。
True なら AUTONOMOUS_SPEECH 刺激を既存パイプラインに流す。詳細は docs/COMPONENT_LOGIC.md §F。
"""
from __future__ import annotations

from .decider import (
    AutonomousSpeech,
    SpeechDecision,
    build_decide_messages,
    make_decide_fn,
    parse_speech_decision,
    should_speak,
)

__all__ = [
    "AutonomousSpeech",
    "SpeechDecision",
    "should_speak",
    "make_decide_fn",
    "parse_speech_decision",
    "build_decide_messages",
]
