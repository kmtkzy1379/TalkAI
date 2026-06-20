"""F4 FeedbackLLM パッケージ（内分泌系・遅延許容の非同期サイドカー）。

各応答後に {要約 / 感情 / ユーザ感情 / 次予測 / 予測差(0-100) / 理由 / タグ} を生成し、
(a) RAG へ add_chunk（圧縮埋め込み/展開注入）、(b) prediction-diff=surprise を PredictionState へ、
(c) 直近フィードバックを ContextAssembler へ供給する。詳細は docs/COMPONENT_LOGIC.md §G と
docs/PIPELINE_DESIGN.md §9.4（サイドカー契約）。
"""
from __future__ import annotations

from .feedback_llm import FeedbackLLM, render_chunk_text, render_last_feedback
from .parser import FeedbackResult, parse_feedback
from .prediction_state import NEUTRAL_SURPRISE, PredictionState, clamp_surprise

__all__ = [
    "PredictionState",
    "NEUTRAL_SURPRISE",
    "clamp_surprise",
    "FeedbackResult",
    "parse_feedback",
    "FeedbackLLM",
    "render_chunk_text",
    "render_last_feedback",
]
