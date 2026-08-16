"""日本語の**照合用**トークン抽出（形態素解析なし・純関数・raise しない）。

用途は2つで、どちらも「その文が“何について”言っているか」を粗く突き合わせるだけ:
- 配達確認（`response/delivery_checker.py`): 報告の要点が実発話に出たか
- 報告の実質チェック（`task/agent.py`): 最終結果に対象語が入っているか

`eve.task` は `eve.response` に依存しない方針（`task/agent.py` 冒頭）なので、共有する
この関数は**どちらにも属さないこの層**に置く。

形態素解析器を入れない理由: 依存を増やさずに済む粒度で足りるため。ここで必要なのは
「内容語が重なるか」だけで、品詞の正確さは要らない（実測: 漢字/カタカナ/英数の2文字以上
という粗い規則で、実機事故の3ケースが正しく分類できる＝`tests/test_delivery_checker.py`）。
"""
from __future__ import annotations

import re

# executor が報告文の頭に付ける定型（`task/executor.py`）。照合前に剥がす
# （剥がさないと「タスク」が内容語として毎回一致し、判定が全部“伝わった”に潰れる）。
REPORT_PREFIXES = ("（予約していたタスク）", "（任せてくれたタスク）")

# 内容語らしいトークン: 漢字/カタカナ/英数の2文字以上の連なり。
# ひらがなは機能語がほとんどなので採らない（「したよ」「だね」等を内容語にしない）。
_TOKEN = re.compile(r"[一-龯々〆ヶ]{2,}|[ァ-ヴー]{2,}|[A-Za-z0-9][A-Za-z0-9._-]+")

# どのタスクの報告にも出る汎用の動作語＝**照合の材料にならない**ので内容語から除く。
# ここが薄いと「表示したよ」対「表示したよ」が“内容語一致”に見え、実質チェックが死ぬ。
GENERIC_WORDS = frozenset({
    "表示", "実行", "完了", "開始", "終了", "作成", "保存", "送信", "検索", "確認",
    "用意", "準備", "対応", "処理", "報告", "結果", "取得", "設定", "変更", "追加",
    "削除", "更新", "実施", "成功", "失敗", "タスク", "予約", "画面", "内容",
})


def strip_report_prefix(text: str) -> str:
    """executor が付ける定型接頭辞を剥がす（無ければそのまま）。"""
    t = (text or "").strip()
    for p in REPORT_PREFIXES:
        if t.startswith(p):
            return t[len(p):].strip()
    return t


def content_tokens(text: str) -> set[str]:
    """照合に使える内容語の集合（汎用動作語を除く）。"""
    return {m.group(0) for m in _TOKEN.finditer(text or "")} - GENERIC_WORDS


def has_substance(text: str) -> bool:
    """「何をしたか」が分かる内容語を含むか（接頭辞は除いて判定）。"""
    return bool(content_tokens(strip_report_prefix(text)))


# --- 日本語・英語以外の文字体系の検出 -------------------------------------------
#
# 用途は2つ:
# - RAG 書き込み時の検査: FeedbackLLM が根拠なく別言語を混ぜたものを落とす。
#   実データ 2026-08-15: `rag_memory.jsonl` の3レコードにデーヴァナーガリーが入っていた
#   （`イブ=शांत` / `प्रतिक्रियाするかもしれない` / topic_tags の `短文 प्रतिक्रिया`）。
#   会話に一切文脈が無く、うち1件は実際に応答LLM の `# 参照` に注入されていた。
# - STT の聞き返し: 発音が崩れて韓国語等として認識された入力を拾う。
#
# **既知の限界（意図的）**: ドイツ語などラテン文字の言語は検出できない。判別には言語判定器が
# 要り、誤判定でまともな日本語を捨てる危険の方が大きい（ユーザ裁定 2026-08-15）。
# ギリシャ文字も**わざと外している**（「βテスト」「α版」など日本語の通常表記に出るため）。
_FOREIGN_SCRIPT = re.compile(
    "["
    "가-힯ᄀ-ᇿ㄰-㆏"      # ハングル
    "Ѐ-ԯ"                                  # キリル
    "֐-׿؀-ۿݐ-ݿ"      # ヘブライ / アラビア
    "ऀ-ॿঀ-৿஀-௿"      # デーヴァナーガリー / ベンガル / タミル
    "฀-๿"                                  # タイ
    "]+"
)


def foreign_script_runs(text: str) -> list[str]:
    """日本語・英語以外の文字体系の連なりを列挙する（無ければ空）。"""
    return [m.group(0) for m in _FOREIGN_SCRIPT.finditer(text or "")]


def has_foreign_script(text: str) -> bool:
    return bool(_FOREIGN_SCRIPT.search(text or ""))


def strip_ungrounded_foreign(text: str, source: str) -> tuple[str, list[str]]:
    """`source`（根拠となる会話）に出てこない別言語の断片を落とす。

    戻り値は (整形後, 落とした断片)。**根拠がある分は残す**のが要点で、
    実データには「『수도권』は韓国語で首都圏という意味だよ」のような正当な用例があり、
    無条件に落とすとこれを壊す（ユーザ・イブ双方に韓国語の実績あり: user 6件 / eve 5件）。
    """
    if not text:
        return text or "", []
    src = source or ""
    removed: list[str] = []

    def _sub(m):
        run = m.group(0)
        if run in src:
            return run  # 会話に出ている＝正当
        removed.append(run)
        return ""

    return _FOREIGN_SCRIPT.sub(_sub, text), removed
