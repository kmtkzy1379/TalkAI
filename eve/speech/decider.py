"""F5 発話判定（should_speak）— 企画書フロー + 中核原理 surprise の code ゲート。

企画書: ユーザ5秒無言で `…` を発話判定LLMに送る。LLM は「話す/黙る」を判断し、
True なら(理由+応答LLMへ渡す content)、False なら(理由)を返す（理由は両方で必須＝
「楽な False」への偏り防止）。本モジュールはこの LLM 判断を**毎回必ず**呼んだ上で、
中核原理に従い surprise を **code ゲート**で合成する（surprise を Optional にしない）:

  surprise >= HI → 強制 speak（想定外が起きた→反応する。content は LLM のものを使う）
  surprise <  LO → 強制 silence（予測が当たりきって新しさが無い）
  それ以外       → LLM の判断どおり（NEUTRAL=20 含む。長い沈黙は話題の種で能動的に振れる）

T2 death-detection: 固定投票 fake decide_fn でも surprise を HI↔<LO に振ると speak↔silence が
反転する（surprise が唯一の決定要因）。反転しなければビルド失敗＝surprise の装飾化を防ぐ。
解析失敗は silence へ保守的フォールバック。「何を言うか」は強制せず下流の応答LLMが自然に生成。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ..config import Config

logger = logging.getLogger(__name__)

# 強制発話で LLM が content を出さなかった時の最小ヒント（応答LLMが文脈から膨らませる）。
_FALLBACK_CONTENT = "（今の状況に自然に一言）"


@dataclass(frozen=True)
class AutonomousSpeech:
    """自発発話の刺激 payload（content=応答LLMへ渡す内容 / reason=なぜ話すか）。"""

    content: str
    reason: str


@dataclass(frozen=True)
class SpeechDecision:
    speak: bool
    reason: str
    content: str = ""


# decide_fn の型: 文脈を受けて LLM 判断（SpeechDecision）を返す。テストは fake を注入。
DecideFn = Callable[..., Awaitable[SpeechDecision]]


# ---- 出力パーサ（タグ付きテキスト・raise しない）-------------------------
def _extract(text: str, *aliases: str) -> str:
    for a in aliases:
        m = re.search(rf"^\s*{re.escape(a)}\s*[:：]\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return ""


_YES = re.compile(r"(yes|true|話す|はい|y\b|1)", re.IGNORECASE)


def parse_speech_decision(text: Optional[str]) -> SpeechDecision:
    """発話判定LLM の出力 → SpeechDecision。raise しない・不明瞭は保守的に silence。"""
    text = text or ""
    raw_speak = _extract(text, "speak", "判定", "話す")
    reason = _extract(text, "reason", "理由")
    content = _extract(text, "content", "内容", "発話")
    if raw_speak:
        speak = bool(_YES.search(raw_speak)) and "no" not in raw_speak.lower() and "黙" not in raw_speak
    else:
        # speak タグが無い: 本文が実質 "…" だけ等なら silence、content があれば speak とみなす
        speak = bool(content) and "…" not in text[:6]
    if not (raw_speak or reason or content):
        return SpeechDecision(False, "発話判定の出力を解析できず沈黙（保守）", "")
    return SpeechDecision(speak=speak, reason=reason or "(理由なし)", content=content)


# ---- 本体ゲート -----------------------------------------------------------
async def should_speak(
    *,
    surprise: int,  # 必須・非 Optional（中核原理: surprise を装飾化させない）
    silence_seconds: float,
    recent_turns,
    topic_seeds,
    decide_fn: DecideFn,
    pending_obligation: bool = False,
) -> SpeechDecision:
    if pending_obligation:
        # 将来の Call-Function/task: 締切が近い予約があるなら自発発話を抑制（今は常に False）。
        return SpeechDecision(False, "保留中の予約/義務があるため沈黙", "")
    # 企画書どおり毎回 LLM に判断させる（＝沈黙中も連続して思考する）。
    llm = await decide_fn(
        surprise=surprise,
        silence_seconds=silence_seconds,
        recent_turns=recent_turns,
        topic_seeds=topic_seeds,
    )
    hi, lo = Config.SURPRISE_SPEAK_FORCE, Config.SURPRISE_SILENCE_FLOOR
    if surprise >= hi:
        return SpeechDecision(True, f"[surprise={surprise}≥{hi} 強制発話] {llm.reason}", llm.content or _FALLBACK_CONTENT)
    if surprise < lo:
        return SpeechDecision(False, f"[surprise={surprise}<{lo} 強制沈黙] {llm.reason}", "")
    return llm


# ---- 本番 decide_fn（ModelRegistry role=speech_decide）---------------------
_SPEAKER_LABEL = {"user": "ユーザ", "eve": "イブ"}

SPEECH_DECIDE_SYSTEM = """\
あなたはAI VTuber「イブ」の発話判定モジュールです。今は誰も話していない沈黙状態。
直近の会話・話題の種・状況から、イブが今"自分から"話すべきか黙るべきかを判断します。
- 話すべき: 文脈的に話す内容がある／場をつなぐ／気になることがある等。**挨拶の繰り返しはしない**。
- 黙るべき: 特に言うことがない／ユーザの間を尊重すべき。
必ず次の形式で1行ずつ出力（**理由は話す/黙る どちらでも必須**）:
speak: yes または no
reason: なぜそう判断したか（1文）
content: 話すなら、イブが実際に話す内容のたたき台（1文）。黙るなら空でよい。"""


def _render_turns(turns) -> str:
    lines = []
    for t in turns or []:
        label = _SPEAKER_LABEL.get(getattr(t, "speaker", ""), getattr(t, "speaker", ""))
        lines.append(f"[{label}] {getattr(t, 'text', '')}")
    return "\n".join(lines) or "（直近の会話なし）"


def _render_seeds(seeds) -> str:
    lines = [f"・{getattr(c, 'text', '')}" for c in (seeds or [])]
    return "\n".join(lines) or "（なし）"


def build_decide_messages(*, surprise: int, silence_seconds: float, recent_turns, topic_seeds) -> list[dict]:
    user = (
        "…\n\n"
        f"# 直近の会話\n{_render_turns(recent_turns)}\n\n"
        f"# 話題の種（新しい話を振る材料・思い出話に縛られない）\n{_render_seeds(topic_seeds)}\n\n"
        f"# 状況\n沈黙{silence_seconds:.0f}秒 / 予測差(surprise)={surprise}"
    )
    return [
        {"role": "system", "content": SPEECH_DECIDE_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_decide_fn(registry) -> DecideFn:
    """ModelRegistry role=speech_decide を叩く本番 decide_fn を作る。"""

    async def decide_fn(*, surprise, silence_seconds, recent_turns, topic_seeds) -> SpeechDecision:
        messages = build_decide_messages(
            surprise=surprise, silence_seconds=silence_seconds,
            recent_turns=recent_turns, topic_seeds=topic_seeds,
        )
        try:
            resp = await registry.complete("speech_decide", messages)
        except Exception:
            logger.exception("発話判定LLM 呼び出し失敗（沈黙にフォールバック）")
            return SpeechDecision(False, "発話判定LLM 失敗のため沈黙（保守）", "")
        return parse_speech_decision(_content(resp))

    return decide_fn


def _content(resp: object) -> str:
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
    if isinstance(resp, str):
        return resp
    return ""
