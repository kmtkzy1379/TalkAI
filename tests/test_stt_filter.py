"""STT 層の決定論テスト（API 不要）: 出力フィルタとバックエンド factory。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_stt_filter.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.stt import clean_transcript, make_stt  # noqa: E402
from eve.stt.groq_whisper import GroqWhisperStt  # noqa: E402
from eve.stt.openai_transcribe import OpenAITranscribeStt  # noqa: E402

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


# 出力フィルタ（音イベントタグ等の括弧だけ注記を破棄、それ以外は保持）
check("tag [drumming] 破棄", clean_transcript("[drumming]") == "")
check("(music) 破棄", clean_transcript("(music)") == "")
check("【拍手】破棄", clean_transcript("【拍手】") == "")
check("空白のみ破棄", clean_transcript("   ") == "")
check("通常文は保持", clean_transcript("こんにちは。") == "こんにちは。")
check("文中の括弧は保持", clean_transcript("これは[重要]です。") == "これは[重要]です。")
check("短い英語片は保持(源ゲートはVADが担当)", clean_transcript("Thank you.") == "Thank you.")

# backend factory（API は叩かない＝クライアント遅延生成）
check("factory openai", isinstance(make_stt("openai"), OpenAITranscribeStt))
check("factory groq", isinstance(make_stt("groq"), GroqWhisperStt))
try:
    make_stt("nope")
    check("未知backendは ValueError", False)
except ValueError:
    check("未知backendは ValueError", True)


# ---- 聞き返し（日本語として受け取れない入力・ユーザ裁定 2026-08-15）----
import random  # noqa: E402

from eve.stt.askback import (  # noqa: E402
    ASK_BACK_FALLBACK,
    ASK_BACK_LINES,
    AskBackPicker,
    is_unintelligible,
)

# 検出: 非ラテン文字の別言語だけを拾う（実データ: user 6件 / eve 5件の韓国語が実在）
check("韓国語は聞き返し対象", is_unintelligible("오케이."))
check("ヒンディー語も対象", is_unintelligible("प्रतिक्रिया"))
check("ベンガル語も対象", is_unintelligible("খাবি"))
check("普通の日本語は対象外", not is_unintelligible("好きな食べ物の画像出してくれない？"))
check("英語混じりも対象外", not is_unintelligible("VOICEVOXって何？"))
check("記号・数字も対象外", not is_unintelligible("12:00だよ。"))
# ギリシャ文字は日本語の通常表記に出る（βテスト等）ので**わざと**対象外にしている
check("βテストは対象外（ギリシャ文字は除外）", not is_unintelligible("βテストの話"))

# 文言選び: 一度使ったものは次に選ばない（同じ文の連発を避ける・ユーザ指示）
_p = AskBackPicker(rng=random.Random(0))
_first = _p.next_line()
_p.reset()
_second = _p.next_line()
check("1回目は聞き返し文言から出る", _first in ASK_BACK_LINES)
check("直前と同じ文言は選ばれない", _second in ASK_BACK_LINES and _second != _first)

# 2回連続で聞き取れなかったら手段を変える（往復ループの遮断）
_p2 = AskBackPicker(rng=random.Random(1))
_a = _p2.next_line()
_b = _p2.next_line()
check("2回目は打ってもらう案内に切り替わる", _a in ASK_BACK_LINES and _b == ASK_BACK_FALLBACK)
check("3回目以降も案内のまま", _p2.next_line() == ASK_BACK_FALLBACK)
_p2.reset()
check("聞き取れたらカウントが戻る", _p2.next_line() in ASK_BACK_LINES)

# 全文言を引き切っても落ちない（候補が1つに減っても回る）
_p3 = AskBackPicker(rng=random.Random(2))
_ok = True
for _ in range(20):
    _p3.reset()
    if _p3.next_line() not in ASK_BACK_LINES:
        _ok = False
check("繰り返し引いても常に候補から出る", _ok)

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
