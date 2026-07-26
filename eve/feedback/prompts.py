"""FeedbackLLM のプロンプト（lean FEP）。

v1 `feedback_prompts.build_fep_user_text` の**形だけ**を激縮（VLM/SceneGraph/Goal-Slot/
precision/ペルソナ等の v1 地層は持ち込まない＝v2 方針「ペルソナは一旦外す・挙動を先に見る」）。
出力は parser.py が読むタグ付きテキスト。

J-2 P2-4（2026-07-18 実機事故）: summary が「イブ自身の未回答オファー」や「前回の予測
(next_prediction)」をユーザの確定意図として書いてしまい、それが RAG に永続化→数十秒後に
自律発話の「話題の種」として自分に戻ってくる自己参照ループが発生（FeedbackLLM 自体の
prompt 欠陥）。summary=事実専用・next_prediction/reason=推測専用と役割を明確化して解消
（実測: 7/10→0/10・gpt-4o-mini/gpt-5.5 の両方で確認。モデル増強単独は6→3/10止まりだった）。
"""
from __future__ import annotations

from typing import Optional

from ..context_assembler import OMITTED_SPEAKER, Turn

_SPEAKER_LABEL = {"user": "ユーザ", "eve": "イブ"}

FEEDBACK_SYSTEM = """\
あなたはAI VTuber「イブ」の内省（フィードバック）モジュールです。
直前の会話を振り返り、次の各項目を**タグ付きで1行ずつ**日本語で出力してください。
余計な前置き・解説・マークダウンは書かない。各タグの値は簡潔に。

**事実(summary)と推測(next_prediction/reason)を明確に分けること。**

summary: 後で思い出して話のネタにできる「出来事の記憶」を**事実のみ**で1文に。実際に何が
  起きたか・誰が何を言った/したかを、状況がわかるように書く。
  相手（ユーザ）の意図・要望・状態は、**ユーザ自身の発言が今回の会話に無い限り書かない**
  （イブ自身の提案・オファー・独り言だけを根拠に「ユーザは〜を求めていた/待っている」と
  書くのは推測であり事実ではない＝禁止）。ユーザの発言が無いターンなら、イブが何をしたかを
  淡々と書く。
  「ユーザが尋ねイブが答えた」式の手続き説明や「覚えていないと答えた」式の非・事実も書かない。
  （良い例: ユーザはチーズケーキが好きで、カフェを探していた ／ イブは要点を先にまとめようかと
   提案した。ユーザからの新しい発言は無い）
  （悪い例: ユーザが質問し、イブが候補を答えた ／ ユーザは説明の確認を待っている状況だった
   ← ユーザは何も言っていないのに意図を断定している＝事実でなく推測の混入）
emotion: イブが今感じている感情（短く）
user_emotion: ユーザの感情の**推定**（ユーザ自身の発言/反応から読み取れる根拠がある時だけ書く。
  根拠が無ければ「不明」と書く。当てずっぽうで書かない）
next_prediction: **ここは推測の欄**。次に何が起きる/ユーザが何を望みそうかの、あなたの推測を
  書いてよい（例:「ユーザは〜を望むかもしれない」）。ただし「前回の予測」をそのまま
  正しかったこととして繰り返さない（前回はあくまで前回時点の推測であり、今回のユーザ発言で
  裏付けられていなければ検証されていない）。
surprise: 前回の予測と今回の現実のズレ（0=完全一致 〜 100=完全に外れ。整数のみ）
reason: next_prediction/surprise の**根拠**を1文で。前回予測を検証もせず単に引き継いだだけ
  なら、正直にその旨を書く（例:「前回予測の継続で新たな根拠は無い」）。
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
