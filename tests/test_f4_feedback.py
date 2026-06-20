"""F4 FeedbackLLM の決定論テスト（API 不要・fake LLM/embedder）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f4_feedback.py
ハーネスは他 Tier-1 と同形（runner 無し・PASS/FAIL カウンタ・合計表示・exit code）。
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)  # ログを黙らせて出力をクリーンに

from eve.feedback import NEUTRAL_SURPRISE, PredictionState, parse_feedback  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


# ========== B1 PredictionState ==========
def t_predstate_defaults() -> bool:
    s = PredictionState()
    return (
        s.last_prediction is None
        and s.last_feedback is None
        and s.watermark is None
        and s.surprise == NEUTRAL_SURPRISE
    )


def t_predstate_surprise_method() -> bool:
    s = PredictionState()
    s.note_feedback_surprise(55)
    a = s.surprise == 55
    s.note_feedback_surprise(150)  # クランプ上限
    b = s.surprise == 100
    s.note_feedback_surprise(-10)  # クランプ下限
    c = s.surprise == 0
    return a and b and c


def t_predstate_surprise_garbage() -> bool:
    s = PredictionState()
    s.note_feedback_surprise("xx")  # 壊れた値 → 中立に倒す（例外を出さない）
    return s.surprise == NEUTRAL_SURPRISE


# ========== B2 parser ==========
_FULL = """summary: ユーザは夏祭りの話で盛り上がっている
emotion: 楽しさ
user_emotion: わくわく
next_prediction: 次は花火の話に移る
surprise: 35
reason: 直近の話題が祭り中心だから
tags: 夏祭り, 花火, 思い出
"""


def t_parser_full() -> bool:
    r = parse_feedback(_FULL)
    return (
        r.summary == "ユーザは夏祭りの話で盛り上がっている"
        and r.emotions == "楽しさ"
        and r.user_emotion == "わくわく"
        and r.next_prediction == "次は花火の話に移る"
        and r.prediction_diff == 35
        and r.reason == "直近の話題が祭り中心だから"
        and r.topic_tags == ["夏祭り", "花火", "思い出"]
    )


def t_parser_partial() -> bool:
    # summary と next_prediction のみ。diff は None（carry-forward の合図）。
    r = parse_feedback("summary: 部分的な出力\nnext_prediction: 続きを聞く")
    return (
        r.summary == "部分的な出力"
        and r.next_prediction == "続きを聞く"
        and r.prediction_diff is None  # 0 ではない
        and r.emotions == ""
        and r.topic_tags == []
    )


def t_parser_fullwidth_colon() -> bool:
    r = parse_feedback("要約：全角コロンの行\n予測差：72")
    return r.summary == "全角コロンの行" and r.prediction_diff == 72


def t_parser_garbage() -> bool:
    r = parse_feedback("これはタグのない自由文。なにも抽出できない。")
    return r.is_empty() and r.prediction_diff is None and r.topic_tags == []


def t_parser_diff_clamp_and_robust() -> bool:
    # 値に語が混じっても整数を拾い、上限はパース後にクランプ前提（ここでは生値）。
    r = parse_feedback("surprise: 約 88 くらい")
    return r.prediction_diff == 88


async def main() -> None:
    # B1
    check("B1 PredictionState 既定値", t_predstate_defaults())
    check("B1 surprise はメソッド書込+クランプ", t_predstate_surprise_method())
    check("B1 surprise 壊れた値は中立", t_predstate_surprise_garbage())
    # B2
    check("B2 parser 完全タグ", t_parser_full())
    check("B2 parser 部分出力(diff=None)", t_parser_partial())
    check("B2 parser 全角コロン", t_parser_fullwidth_colon())
    check("B2 parser 非タグはゴミ(raise しない)", t_parser_garbage())
    check("B2 parser diff は語混じりでも整数抽出", t_parser_diff_clamp_and_robust())


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
