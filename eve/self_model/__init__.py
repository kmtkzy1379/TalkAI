"""自己形成サイドカー（イブの好み・趣向を育てる）。

ストアは `eve/memory/self_model.py` に置く（RagStore が `memory/long_term.py` にあり、
それを書く FeedbackWorker が `eve/feedback/` にあるのと同じ分け方）。
"""
from .trigger import TRIGGER_MARKERS, is_trigger_text, should_examine
from .worker import SelfModelWorker, parse_ops, render_eve_turns

__all__ = [
    "TRIGGER_MARKERS", "is_trigger_text", "should_examine",
    "SelfModelWorker", "parse_ops", "render_eve_turns",
]
