"""聞き取れなかった入力への聞き返し（決定論のゲート + 定型文の選択）。

## なぜコード側で持つか
STT は発音が崩れると**別言語として認識する**（実測: 韓国語・ドイツ語）。日本語として
意味を成さないので、そのまま応答LLM に渡すと知ったかぶりの元になる。`response/style.py`
には既に「聞き取れない語を作文しない・一度だけ聞き返す」というプロンプト規則があるが、
プロンプトで守られない規律はコードで止めるのが本プロジェクトの方針（CLAUDE.md）。

## 何を捕まえるか
`text_match.has_foreign_script` が拾う**非ラテン文字の別言語**だけ。ドイツ語のような
ラテン文字の言語は検出できない（言語判定器が要り、誤判定で正常な日本語を捨てる危険の方が
大きいというユーザ裁定 2026-08-15）。そちらは従来どおりプロンプト側の聞き返しに任せる。

## 記録の方針（ユーザ裁定 2026-08-15）
- **会話履歴には残す**。残さないとイブの「聞き取れなかった」という発話だけが記録され、
  その原因が消える＝次ターンで自分の発話に自答する形になる（ROLE_ANCHOR が防ごうとして
  いる話者取り違えを自分で誘発する）。
- **生の誤認識テキストは保存しない**。それ自体が別言語混入の素材になるため、
  `UNINTELLIGIBLE_PLACEHOLDER` に置き換える。
- **RAG には入れない**。内容ゼロのチャンクは検索を汚し、無言時の「話題の種」として
  引かれると最悪（`RAG_RANDOM_K`）。除外は FeedbackWorker 側で行う。
"""
from __future__ import annotations

import random
from typing import Optional

from ..text_match import has_foreign_script

# 会話履歴に残すユーザ側の代替テキスト（生の誤認識は残さない）。
UNINTELLIGIBLE_PLACEHOLDER = "（聞き取れなかった発話）"

# 1回目の聞き返し。**一度使ったものは次に選ばない**（同じ文の連発は不自然・ユーザ指示）。
ASK_BACK_LINES: tuple[str, ...] = (
    "ごめん、今の聞き取れなかった。もう一回言ってくれる？",
    "ごめんね、うまく聞き取れなかったよ。もう一度お願い。",
    "あれ、今のうまく拾えなかった。もう一回いい？",
    "ごめん、聞き取れなかったかも。もう一度言ってみて。",
)

# 2回続けて聞き取れなかった時。**同じ聞き返しを繰り返さず手段を変える**（往復ループの遮断）。
# UI にキーボード入力があるので、これは実際に前へ進む提案になる。
ASK_BACK_FALLBACK = "ごめん、何度もうまく聞き取れないみたい。キーボードで打ってもらえる？"


def is_unintelligible(text: str) -> bool:
    """日本語として受け取れない入力か（現状は非ラテン文字の別言語だけ）。"""
    return has_foreign_script(text)


class AskBackPicker:
    """聞き返し文言を選ぶ（直前に使ったものは選ばない・2回連続で手段を変える）。

    状態は「直前に選んだ文」と「連続失敗回数」だけ。loop 単一所有なのでロックは要らない。
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()
        self._last: Optional[str] = None
        self._streak = 0

    def reset(self) -> None:
        """聞き取れた発話が来たら連続カウントを戻す（次はまた1回目の文言から）。"""
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def next_line(self) -> str:
        """次に言う聞き返しを1つ返す（呼ぶたびに連続失敗回数が1つ進む）。"""
        self._streak += 1
        if self._streak >= 2:
            return ASK_BACK_FALLBACK  # 2回目以降は手段を変える（同じ文を繰り返さない）
        pool = [s for s in ASK_BACK_LINES if s != self._last] or list(ASK_BACK_LINES)
        line = self._rng.choice(pool)
        self._last = line
        return line
