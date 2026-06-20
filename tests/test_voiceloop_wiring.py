"""VoiceLoop の組み立て(配線)スモークテスト（API/モデル/音声デバイス不要）。

VoiceLoop() の __init__ は全 backend を遅延 import で構築する（torch/pyaudio/network なしで構築可）。
これは「ロジックテストが全緑でも VoiceLoop のコンストラクタ引数ドリフトで“実起動だけ”壊れる」
（監査 Gap1）を防ぐ最小ガード。run() は呼ばない＝構築と結線のみ検証する。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_voiceloop_wiring.py
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.voice_loop import VoiceLoop  # noqa: E402

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


try:
    vl = VoiceLoop()
    check("VoiceLoop 構築成功(遅延importのみ)", True)
    check("orchestrator に短期記憶を配線", vl.orchestrator._cache is vl.cache)
    check("orchestrator に長期記憶(RAG)を配線", vl.orchestrator._rag is vl.rag)
    # F4: 内分泌系（PredictionState / FeedbackLLM / FeedbackWorker）の配線
    check("orchestrator に PredictionState を配線", vl.orchestrator._state is vl.prediction)
    check("orchestrator 完了トリガ=feedback worker.trigger", vl.orchestrator._on_complete == vl.feedback_worker.trigger)
    check("FeedbackLLM に RAG/PredictionState を配線", vl.feedback._rag is vl.rag and vl.feedback._state is vl.prediction)
    check("FeedbackWorker に feedback/cache を配線", vl.feedback_worker._fb is vl.feedback and vl.feedback_worker._cache is vl.cache)
    check("runner に orchestrator を配線", vl.runner._orch is vl.orchestrator)
    check("runner と input が同じ queue を共有", vl.runner._queue is vl.queue and vl.input._queue is vl.queue)
    check("input に barge-in callback を結線", callable(vl.input._on_speech_start))
    # play_fn が should_stop を取れる＝文途中 barge-in(B3) の配線が生きている
    check("audio が mid-sentence stop 対応 play_fn を保持", vl.audio._play_takes_stop is True)
except Exception as e:  # 構築自体が落ちたら配線ドリフト＝即失敗
    check(f"VoiceLoop 構築で例外: {e!r}", False)

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
