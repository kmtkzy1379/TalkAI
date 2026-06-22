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

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
