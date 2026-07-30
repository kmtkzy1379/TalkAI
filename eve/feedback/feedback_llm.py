"""FeedbackLLM — 各応答後に内省を生成するサイドカー本体（run の中身）。

単一 LLM 呼び出し（ModelRegistry role="feedback"）→ タグ付き出力をパース → 4効果:
(a) RAG add_chunk（圧縮埋め込み=summary+tags / 展開注入=完全版 text）、
(b) last_prediction 更新（FEP 繰越）、(c) surprise 更新（メソッド API）、
(d) last_feedback 文字列更新（応答LLM への文脈注入用・最新1件）。

規律: 失敗（例外/空出力/RAG 未保存）は logger + **no-op**（state を変えない）で落とさない。
**RAG 保存に成功した時だけ state を更新し、run は結果を返す**（= watermark を進めてよい合図）。
保存できなければ run は None を返し、worker(B5) は watermark を進めず同スパンを次 run で再試行する
（ユーザ要件「フィードバックしてない会話＝記憶喪失を作らない」）。
worker / lifecycle は worker.py 側。run は「turns を渡されたら内省して効果適用」に専念。
"""
from __future__ import annotations

import logging
from typing import Optional

from ..context_assembler import Turn
from ..model_registry import ModelRegistry
from .parser import FeedbackResult, parse_feedback
from .prediction_state import NEUTRAL_SURPRISE, PredictionState
from .prompts import build_messages

logger = logging.getLogger(__name__)


def _content(resp: object) -> str:
    """litellm/openai 応答 or テスト fake から本文テキストを頑健に取り出す。"""
    try:
        c = resp.choices[0].message.content  # type: ignore[attr-defined]
        if isinstance(c, str):
            return c
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    if isinstance(resp, dict):
        try:
            c = resp["choices"][0]["message"]["content"]
            if isinstance(c, str):
                return c
        except (KeyError, IndexError, TypeError):
            pass
        c2 = resp.get("content")
        if isinstance(c2, str):
            return c2
    if isinstance(resp, str):
        return resp
    return ""


def render_last_feedback(r: FeedbackResult) -> str:
    """応答LLM へ注入する短い1ブロック（≤3s 予算のため最小限）。"""
    head = r.summary or r.reason or "（要約なし）"
    bits = []
    if r.prediction_diff is not None:
        bits.append(f"予測差{r.prediction_diff}")
    if r.emotions:
        bits.append(r.emotions)
    suffix = f"（{' / '.join(bits)}）" if bits else ""
    return f"{head}{suffix}"


def render_chunk_text(r: FeedbackResult) -> str:
    """RAG 注入時に解凍される完全版テキスト（展開注入）。

    先頭はエピソード記憶（出来事の事実）を**ラベルなし**で置く。注入側が
    `[過去の記憶/N分前] …` と既に銘打つため「要約:」等の前置きは重複＝省く。
    """
    lines: list[str] = []
    if r.summary:
        lines.append(r.summary)
    affect = []
    if r.emotions:
        affect.append(f"イブ={r.emotions}")
    if r.user_emotion:
        affect.append(f"ユーザ={r.user_emotion}")
    if affect:
        lines.append("感情: " + " / ".join(affect))
    if r.next_prediction:
        lines.append(f"次の予測: {r.next_prediction}")
    if r.prediction_diff is not None:
        lines.append(f"予測差: {r.prediction_diff}")
    if r.reason:
        lines.append(f"理由: {r.reason}")
    return "\n".join(lines)


class FeedbackLLM:
    def __init__(
        self,
        registry: ModelRegistry,
        rag_store=None,
        prediction_state: Optional[PredictionState] = None,
    ) -> None:
        self._registry = registry
        self._rag = rag_store
        self._state = prediction_state or PredictionState()

    @property
    def state(self) -> PredictionState:
        return self._state

    async def run(self, turns: list[Turn]) -> Optional[FeedbackResult]:
        """会話スパン turns を内省し効果を適用。

        成功（RAG 保存済 or RAG 無し）で FeedbackResult、失敗で None（watermark を進めない合図）。
        """
        if not turns:
            return None
        messages = build_messages(turns, self._state.last_prediction)
        try:
            resp = await self._registry.complete("feedback", messages)
        except Exception:
            logger.exception("FeedbackLLM 呼び出し失敗（no-op で継続）")
            return None
        result = parse_feedback(_content(resp))
        if result.is_empty():
            logger.warning("FeedbackLLM 出力から何も抽出できず（no-op で継続）")
            return None
        # D1: スパン末尾の iso を渡す（チャンクの時刻は書込時刻ではなく対象会話の時刻）。
        if not await self._apply(result, turns[-1].stamp.iso):
            return None  # RAG 未保存 → state 据え置き・同スパンを次 run で再試行
        return result

    async def _apply(self, result: FeedbackResult, span_end_iso: str) -> bool:
        """効果適用。RAG 保存に成功（or RAG 無し）なら True を返し state を更新。

        span_end_iso: 内省した会話スパンの末尾 iso（チャンクの timestamp に使う）。
        """
        cold = self._state.last_prediction is None
        # (a) RAG: 圧縮埋め込み（summary+tags）/ 展開注入（完全版 text）。保存成否を len 差で判定。
        if self._rag is not None:
            before = len(self._rag)
            try:
                await self._rag.add_chunk(
                    text=render_chunk_text(result),
                    summary=result.summary or None,
                    emotions=result.emotions or None,
                    next_prediction=result.next_prediction or None,
                    prediction_diff=result.prediction_diff,
                    reason=result.reason or None,
                    topic_tags=result.topic_tags or None,
                    # D1: チャンクの時刻は**内省した会話スパンの末尾**（書込時刻ではない）。
                    # 未指定だと long_term が now_iso() を入れるため、起動時 catch-up が
                    # 前セッションを要約したチャンクが `[過去の記憶/たった今]` と描画され、
                    # 5時間前の依頼が「たった今の記憶」として応答LLM に渡っていた
                    # （実データ: ts=17:21:05 / スパン末尾=12:26:35＝4時間55分のズレ）。
                    timestamp=span_end_iso,
                )
            except Exception:
                logger.exception("FeedbackLLM の RAG 書込で例外（state 据え置き）")
                return False
            if len(self._rag) <= before:
                logger.warning("FeedbackLLM の RAG 保存が成立せず（state 据え置き）")
                return False
        # ここから state 更新（RAG 保存成功後のみ）。
        # (b) FEP 繰越
        if result.next_prediction:
            self._state.last_prediction = result.next_prediction
        # (c) surprise: cold は中立 / diff 抽出失敗は carry-forward / それ以外は反映
        if cold:
            self._state.note_feedback_surprise(NEUTRAL_SURPRISE)
        elif result.prediction_diff is not None:
            self._state.note_feedback_surprise(result.prediction_diff)
        # (d) 文脈注入用の短い1ブロック（最新1件）
        self._state.last_feedback = render_last_feedback(result)
        return True
