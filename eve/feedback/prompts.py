"""FeedbackLLM のプロンプト（lean FEP）。

v1 `feedback_prompts.build_fep_user_text` の**形だけ**を激縮（VLM/SceneGraph/Goal-Slot/
precision/ペルソナ等の v1 地層は持ち込まない＝v2 方針「ペルソナは一旦外す・挙動を先に見る」）。
出力は parser.py が読むタグ付きテキスト。
"""
from __future__ import annotations

from typing import Optional

from ..context_assembler import OMITTED_SPEAKER, Turn

_SPEAKER_LABEL = {"user": "ユーザ", "eve": "イブ"}

FEEDBACK_SYSTEM = """\
あなたはAI VTuber「イブ」の内省（フィードバック）モジュールです。
直前の会話を振り返り、次の各項目を**タグ付きで1行ずつ**日本語で出力してください。
余計な前置き・解説・マークダウンは書かない。各タグの値は簡潔に。

summary: 後で思い出して話のネタにできる「出来事の記憶」を1文で。ユーザが何をしていたか・
  何を求めていたか・好み・予定などの“事実”を中心に、状況がわかるように書く。
  「ユーザが尋ねイブが答えた」式の手続き説明や「覚えていないと答えた」式の非・事実は書かない。
  （良い例: ユーザはチーズケーキが好きで、カフェを探していた ／ ユーザは雨の日に室内で遊べる場所を探していた）
  （悪い例: ユーザが質問し、イブが候補を答えた ／ イブは覚えていないと伝えた）
emotion: イブが今感じている感情（短く）
user_emotion: ユーザの感情の推定（短く）
next_prediction: 次に何が起きる/話されるかの予測（1文）
surprise: 前回の予測と今回の現実のズレ（0=完全一致 〜 100=完全に外れ。整数のみ）
reason: その予測差・感情の理由（1文）
tags: 話題タグをカンマ区切りで最大6個（具体的な固有名詞・物事を優先）"""

_COLD_SENTINEL = "（初回・前回の予測なし。surprise は中立でよい）"


def _render_turns(turns: list[Turn]) -> str:
    lines: list[str] = []
    for t in turns:
        if t.speaker == OMITTED_SPEAKER:
            continue  # 省略マーカは内省入力に含めない
        label = _SPEAKER_LABEL.get(t.speaker, t.speaker)
        lines.append(f"[{label}] {t.text}")
    return "\n".join(lines)


def build_user_text(turns: list[Turn], last_prediction: Optional[str]) -> str:
    """前回予測 + 振り返り対象の会話スパンを描画。"""
    pred = last_prediction.strip() if last_prediction else _COLD_SENTINEL
    convo = _render_turns(turns) or "（会話なし）"
    return f"# 前回の予測\n{pred}\n\n# 直近の会話（これを振り返る）\n{convo}"


def build_messages(turns: list[Turn], last_prediction: Optional[str]) -> list[dict]:
    return [
        {"role": "system", "content": FEEDBACK_SYSTEM},
        {"role": "user", "content": build_user_text(turns, last_prediction)},
    ]
