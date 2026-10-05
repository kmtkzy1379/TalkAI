"""FeedbackLLM のプロンプト（lean FEP）。

v1 `feedback_prompts.build_fep_user_text` の**形だけ**を激縮（VLM/SceneGraph/Goal-Slot/
precision/ペルソナ等の v1 地層は持ち込まない＝v2 方針「ペルソナは一旦外す・挙動を先に見る」）。
出力は parser.py が読むタグ付きテキスト。

J-2 P2-4（2026-07-18 実機事故）: summary が「イブ自身の未回答オファー」や「前回の予測
(next_prediction)」をユーザの確定意図として書いてしまい、それが RAG に永続化→数十秒後に
自律発話の「話題の種」として自分に戻ってくる自己参照ループが発生（FeedbackLLM 自体の
prompt 欠陥）。summary=事実専用・next_prediction/reason=推測専用と役割を明確化して解消
（実測: 7/10→0/10・gpt-4o-mini/gpt-5.5 の両方で確認。モデル増強単独は6→3/10止まりだった）。

2026-09-30 ユーザ提案「summary は1文にしなくていい。内容が濃かったり話題が次々変わると1文で要約できない。
重要な要素を簡潔に事実のみ」を反映（`tools/prompt_eval/fb.py`・Sonnet 5.5・8スパン×2回）: 旧「1文に」でも
Sonnet 5.5 は全話題を読点でつないだ長い1文に詰めており取りこぼしは無かった。新しい書き方は同じ情報量で
複数の文に分かれて読みやすくなる。長さはほぼ同じ（+10字前後）・推測の混入なし（イブだけが話したスパン 4/4）・
記憶の引き当て（本番の記憶 439件に混ぜて問いから上位3件）は 0.886 で変わらず。
2026-10-01 ユーザ決定で**良い例・悪い例を外した**（見本の文は丸写しされやすい＝ユーザの原則）。外した版も
取りこぼし・推測の混入は無く、引き当ても 0.886 で同じ。「ユーザが尋ね、イブが答えた」式の手続きの書き方が
少し戻る傾向はあった（禁止の文は残している）。
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

summary: 後で思い出して話のネタにできる「出来事の記憶」を、**重要な要素を簡潔に・事実のみ**で書く
  （1文にまとめなくてよい。話題が複数あれば全部入れる。改行はしない）。実際に何が
  起きたか・誰が何を言った/したかを、状況がわかるように書く。
  相手（ユーザ）の意図・要望・状態は、**ユーザ自身の発言が今回の会話に無い限り書かない**
  （イブ自身の提案・オファー・独り言だけを根拠に「ユーザは〜を求めていた/待っている」と
  書くのは推測であり事実ではない＝禁止）。ユーザの発言が無いターンなら、イブが何をしたかを
  淡々と書く。
  「ユーザが尋ねイブが答えた」式の手続き説明や「覚えていないと答えた」式の非・事実も書かない。
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

# S5（ユーザ裁定 2026-09-26「フィードバックでも感情分析してるからフィードバックがきたらそれに更新」
# 「現感情を…フィードバックに入れれば大きく不自然な感情切り替わりは少ない」）。感情の機能が ON の時だけ足す
# （OFF なら従来と同じ文面）。ラベルと説明は `eve/emotion.py` の LABELS と同じ。
EMOTION_LABEL_RULE = (
    "emotion_label: イブの今の感情を次のどれか1つだけで書く: "
    "喜（嬉しい・喜んでいる） / 怒（怒っている・苛立っている） / 哀（悲しい・つらい） / "
    "楽（穏やかで楽しい・リラックスしている） / 笑い（笑っている・可笑しがっている）"
)
_NO_EMOTION = "なし（初回）"


def _render_turns(turns: list[Turn]) -> str:
    lines: list[str] = []
    for t in turns:
        if t.speaker == OMITTED_SPEAKER:
            continue  # 省略マーカは内省入力に含めない
        label = _SPEAKER_LABEL.get(t.speaker, t.speaker)
        lines.append(f"[{label}] {t.text}")
    return "\n".join(lines)


def build_user_text(turns: list[Turn], last_prediction: Optional[str], *,
                    with_emotion: bool = False, current_emotion: Optional[str] = None) -> str:
    """前回予測 + 振り返り対象の会話スパンを描画（感情の機能が ON なら、イブの今の感情も添える）。"""
    pred = last_prediction.strip() if last_prediction else _COLD_SENTINEL
    convo = _render_turns(turns) or "（会話なし）"
    text = f"# 前回の予測\n{pred}\n\n# 直近の会話（これを振り返る）\n{convo}"
    if with_emotion:
        text += f"\n\n# イブの今の感情\n{current_emotion or _NO_EMOTION}"
    return text


PERSONA_HOW = ("感情や次の予測は、この性格のイブとして考える。ただし summary は事実だけ。"
               "イブの冗談や作り話は、冗談として書く")


def build_messages(turns: list[Turn], last_prediction: Optional[str], *,
                   with_emotion: bool = False, current_emotion: Optional[str] = None,
                   persona: str = "") -> list[dict]:
    """persona（`load_persona()` の文字列）があれば system の最後に性格を添える（応答と食い違わないため）。"""
    from ..persona import persona_for

    system = (FEEDBACK_SYSTEM + ("\n" + EMOTION_LABEL_RULE if with_emotion else "")
              + persona_for(persona, PERSONA_HOW))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_text(turns, last_prediction, with_emotion=with_emotion,
                                                    current_emotion=current_emotion)},
    ]
