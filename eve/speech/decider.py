"""F5 発話判定（should_speak）— 企画書フロー + 中核原理 surprise を「指標」として総合判断。

企画書: ユーザ5秒無言で `…` を発話判定LLMに送る。LLM は「話す/黙る」を判断し、
True なら(理由+応答LLMへ渡す content)、False なら(理由)を返す（理由は両方で必須＝
「楽な False」への偏り防止）。

**surprise は数値で判定を絶対決定しない（ユーザ裁定）**: 人間も予想が外れたから必ず話す訳でも、
当たったから必ず黙る訳でもない（感情/思考が高ぶる/安定するだけ）。よって surprise は **各感情
(直近フィードバック) と内容と合わせて発話判定LLMが総合判断する「指標(必須入力)」**として渡す。
唯一の hard ゲートは pending_obligation（予約締切等の事実・将来 Call-Function）。

T2 death-detection（surprise の非装飾性）: surprise は **必須引数**（Optional 化禁止）で decide_fn へ
配線され、判定を動かしうる。surprise を読む fake decide_fn で値を振ると判定が反転することを検証
（surprise が決定に効く配線である保証）。解析失敗は silence へ保守的フォールバック。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# 話す判断だが LLM が content を出さなかった時の最小ヒント（応答LLMが文脈から膨らませる）。
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
        # 保守的: speak タグが無ければ黙る（content の有無で speak と推定しない＝不明瞭は silence）。
        speak = False
    if not (raw_speak or reason or content):
        return SpeechDecision(False, "発話判定の出力を解析できず沈黙（保守）", "")
    return SpeechDecision(speak=speak, reason=reason or "(理由なし)", content=content)


# ---- 本体（pending だけ hard・surprise は指標として LLM が総合判断）----------
async def should_speak(
    *,
    surprise: int,  # 必須・非 Optional（中核原理: surprise を装飾化させない・指標として配線）
    silence_seconds: float,
    recent_turns,
    topic_seeds,
    decide_fn: DecideFn,
    last_feedback: Optional[str] = None,  # イブの今の感情/要約（直近フィードバック）
    vision: Optional[str] = None,  # F6 直近の画面ナレーション（あれば判定材料に・None なら無し）
    pending_obligation: bool = False,
) -> SpeechDecision:
    if pending_obligation:
        # 唯一の hard ゲート（事実: 予約締切等。感情でないのでここだけ確定で沈黙）。
        # 将来 Call-Function/task が締切近接を計算して渡す（今は常に False）。
        return SpeechDecision(False, "保留中の予約/義務があるため沈黙", "")
    # surprise + 感情(last_feedback) + 会話 + 話題の種 (+ 画面) を渡し、LLM が総合判断。
    # vision は **非 None の時だけ** 転送する（A6: vision を受けない既存 decide_fn を壊さない）。
    kwargs = dict(
        surprise=surprise,
        silence_seconds=silence_seconds,
        recent_turns=recent_turns,
        topic_seeds=topic_seeds,
        last_feedback=last_feedback,
    )
    if vision is not None:
        kwargs["vision"] = vision
    d = await decide_fn(**kwargs)
    if d.speak and not (d.content or "").strip():
        # 話す判断だが content が空 → 全 speak 経路で最小ヒントを保証（応答LLMが膨らませる）。
        return SpeechDecision(True, d.reason, _FALLBACK_CONTENT)
    return d


# ---- 本番 decide_fn（ModelRegistry role=speech_decide）---------------------
_SPEAKER_LABEL = {"user": "ユーザ", "eve": "イブ"}

SPEECH_DECIDE_SYSTEM = """\
あなたはAI VTuber「イブ」の発話判定モジュールです。今は誰も話していない（沈黙 or 相手が画面を操作中）。
直近の会話・話題の種(記憶)・イブの今の感情(直近フィードバック)・画面(今見えているもの)・予測差(surprise)を
**総合的に**見て、イブが今"自分から"一言を言うべきか黙るべきかを判断します。

【話す(yes) — 相手がうれしい/役立つ一言があるなら積極的に言ってよい】
- 画面の内容を**過去の記憶と結びつける**（例:「前にチーズケーキ好きって言ってたね、これ良さそう」）。
- 画面で相手が**探している/迷っているものに気づいて手伝う**（例:「スポッチャ、室内で雨でも遊べていいね」）。
- 直近の会話を一歩進める／関連する**新しい話題を記憶から**振る／相手が一息ついた間に声をかける。

【黙る(no)】
- **直前にイブが自分から話して、相手がまだ返事していない**（畳みかけない・間を空けて相手の番を待つ）。
- 本当に言うことがない／さっきと同じことの繰り返しになる。
- 相手が集中して作業に没頭していて、口を挟むと**邪魔になりそう**な時。
- 相手が**疲れていたり休みたそう**な時（「疲れた」等の直後）は、話題を増やさずそっとしておく。

【禁止】
- **毎回は話しかけない**。本当に良い一言・気の利いた一言がある時だけに絞る（質問の連投・実況の垂れ流しはしない）。
- 過去の記憶は**時々の隠し味**。毎回「そういえば」と持ち出すと逆にしつこい（数回に1回で十分）。
- **画面変化のいちいちを実況報告しない**（「○○が表示されました」の垂れ流しは黙る）。挨拶の繰り返しもしない。
- surprise(予測差)は強い指標だが絶対ではない（高くても黙ってよいし、低くても話してよい）。

必ず次の形式で1行ずつ出力（**理由は yes/no どちらでも必須**）:
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


def build_decide_messages(
    *, surprise: int, silence_seconds: float, recent_turns, topic_seeds,
    last_feedback: Optional[str] = None, vision: Optional[str] = None,
) -> list[dict]:
    fb = (last_feedback or "").strip() or "（なし）"
    screen = (vision or "").strip()
    screen_block = f"\n\n# 画面（今この瞬間）\n{screen}" if screen else ""
    user = (
        "…\n\n"
        f"# 直近の会話\n{_render_turns(recent_turns)}\n\n"
        f"# 過去の記憶・話題の種（今の流れに合えば「そういえば前〜って言ってたね」と自然に話を広げてよい）\n{_render_seeds(topic_seeds)}\n\n"
        f"# イブの今の状態（直近フィードバック: 感情/要約）\n{fb}"
        f"{screen_block}\n\n"
        f"# 状況\n沈黙{silence_seconds:.0f}秒 / 予測差(surprise)={surprise}"
        "（指標。高=思考/感情が高ぶる・低=安定。これだけで決めない）"
    )
    return [
        {"role": "system", "content": SPEECH_DECIDE_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_decide_fn(registry) -> DecideFn:
    """ModelRegistry role=speech_decide を叩く本番 decide_fn を作る。"""

    async def decide_fn(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None, vision=None) -> SpeechDecision:
        messages = build_decide_messages(
            surprise=surprise, silence_seconds=silence_seconds,
            recent_turns=recent_turns, topic_seeds=topic_seeds, last_feedback=last_feedback, vision=vision,
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
