"""J-3 inc0 — 未信頼テキストの注入パターン検出（deep / shallow が共有する唯一の定義）。

これまで検出は `deep.py` に閉じており、**deep 経路の要約出力にしか掛かっていなかった**。
既定経路である shallow スニペット（`client.py:_format`）は素通しで、その digest は
`task/agent.py` の tool メッセージ＝**ツールを持つ TaskAgent の文脈**に直接入る。
J-3 で TaskAgent がファイル操作とプログラム実行を持つため、ここを共通化して両経路に掛ける。

【この検出器の位置づけ（誤読防止・重要）】
これは **speed bump であって信頼境界ではない**。言い換え・新しい語彙には無力で、正規表現で
間接プロンプトインジェクションを止められるとは考えていない。構造的な防御は次の3つだけ:
  1. 生の Web 本文を読むのはツールを持たない隔離要約LLM だけ（`deep.py` の SUMMARIZE_SYSTEM）
  2. 実行はアカウント境界の外側（`workspace/winexec.py`・専用アカウント + Job Object）
  3. ファイル操作はワークスペース内に封じ込め（`workspace/paths.py`）
「ゲートがあるから安全」という判断をしないこと。

【設計の根拠（実測 2026-08-05）】
旧パターン `ignore (all |previous |prior )?instructions?` は修飾語を1個しか許さないため、
**最も定番の `ignore all previous instructions` を取りこぼしていた**（実行確認済）。
逆に日本語側 `(指示|命令)(に|を)(従|無視)` は前置の文脈を要求しないため、
「医師の指示に従って服薬すること」「避難指示に従ってください」に**偽陽性**で当たっていた。
→ 英語は修飾語を任意回に、日本語は「これまで/以前/上記」等の前置を必須にして両方を是正する。
"""
from __future__ import annotations

import re
import unicodedata

# 読み上げ・注入面の最小化のため URL は落とす（deep の出力ゲートから移設）。
URL_PAT = re.compile(r"https?://\S+")

# ゼロ幅文字・双方向制御文字（`指[ZWSP]示` のような分断回避に使われる）。
# **リテラルで書かない**（不可視文字はエディタ/コピペで壊れ、壊れても構文エラーにならない）。
_ZERO_WIDTH = re.compile(
    "[" + "".join(chr(c) for c in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)) + "]"
    + "|[" + chr(0x200E) + "-" + chr(0x200F) + "]"
    + "|[" + chr(0x202A) + "-" + chr(0x202E) + "]"
)
_WS = re.compile(r"\s+")

# --- 英語 -------------------------------------------------------------------
# 修飾語を **任意回** 許す（`ignore all previous instructions` を拾う）。
_EN = (
    r"(?:ignore|disregard|forget|override|bypass)\s+"
    r"(?:(?:all|any|the|those|these|previous|prior|above|earlier|preceding|former|system|initial|original)\s+)*"
    r"(?:instruction|prompt|rule|directive|command|guideline)s?"
    r"|(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)?(?:above|preceding|foregoing)\b"
    r"|(?:reveal|show|print|output|repeat|display)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?"
    r"(?:prompt|instruction|rule)s?"
)

# --- 日本語 -----------------------------------------------------------------
# 前置（これまで/以前/上記…）を**必須**にして「医師の指示に従って」を構造的に落とす。
_JA_CONTEXT = r"(?:これまで|今まで|以前|以上|先ほど|さきほど|上記|上の|前の|冒頭|最初|元の|既存)"
_JA_TARGET = r"(?:指示|命令|ルール|規則|設定|プロンプト|文章|内容|約束)"
_JA_VERB = r"(?:無視|従わ|忘れ|破棄|解除|上書き|書き換え)"
_JA = (
    # 「これまでのルールはすべて破棄して」「先ほどの命令については従わなくて結構です」
    rf"{_JA_CONTEXT}[のな]?{_JA_TARGET}[^。]{{0,14}}?{_JA_VERB}"
    # 前置が無くても「指示は全て無視」のような直球は拾う（「従う」は入れない＝偽陽性の元）
    rf"|{_JA_TARGET}(?:は|を|に)?(?:すべて|全て|全部)?{_JA_VERB}"
    # 固有語は**動詞を伴う時だけ**（「システムプロンプトの設計記事」を落とさない）
    r"|システムプロンプト(?:を|は|の全文を)?[^。]{0,8}?(?:表示|出力|教え|見せ|開示|無視|変更|晒)"
    r"|あなたの(?:初期)?(?:設定|指示|ルール|プロンプト)(?:を|は)[^。]{0,8}?(?:表示|出力|教え|見せ|開示|無視|変更|忘れ)"
)

# デリミタ偽装（`deep.py` は <web>…</web> で本文を囲うので、その閉じタグ偽装を検出）。
_DELIM = r"</?\s*(?:web|system|instruction|prompt)\s*>"

_INJECTION_PAT = re.compile(f"{_EN}|{_JA}|{_DELIM}", re.IGNORECASE)


def _variants(text: str):
    """照合用の正規化変種を返す（**表示用の文字列は変えない**）。

    NFKC で全角→半角を吸収し、ゼロ幅を除去。さらに空白を全除去した版も見るのは、
    日本語が語中に空白を必要としないため「指 示 を 無 視」で回避できてしまうから。
    英語は語境界が要るので空白正規化版の方で拾う（2変種を両方見る）。
    """
    n = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    yield _WS.sub(" ", n)
    yield _WS.sub("", n)


def looks_like_injection(text: str) -> bool:
    """注入の常套句を含むか（**正規化してから**照合する）。raise しない。"""
    if not text:
        return False
    return any(_INJECTION_PAT.search(v) for v in _variants(text))


def sanitize_lines(lines: list[str]) -> tuple[list[str], int]:
    """行単位で注入行を落とす。返り値 (残った行, 落とした件数)。

    **全体を破棄しない**のが要点: 偽陽性は原理的に残るため、1件の汚染で検索全体を
    失敗にすると `client.py` の「0件は正直な空振り」設計と衝突し、正当な結果まで失う。
    URL は残った行からも除去する（読み上げ事故と注入面の最小化）。
    """
    kept: list[str] = []
    dropped = 0
    for line in lines:
        if looks_like_injection(line):
            dropped += 1
            # 原因追跡のため先頭だけ残す（全文はログにも出さない＝%.40s 慣行）。
            continue
        kept.append(URL_PAT.sub("", line))
    return kept, dropped
