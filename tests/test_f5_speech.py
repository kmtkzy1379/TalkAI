"""F5 発話判定 / 自発発話の決定論テスト（API 不要・fake LLM）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f5_speech.py
ハーネスは他 Tier-1 と同形。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.config import Config  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.speech import (  # noqa: E402
    SpeechDecision,
    make_decide_fn,
    parse_speech_decision,
    should_speak,
)

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


HI = Config.SURPRISE_SPEAK_FORCE  # 60
LO = Config.SURPRISE_SILENCE_FLOOR  # 20


async def _fixed_silence(*, surprise, silence_seconds, recent_turns, topic_seeds):
    return SpeechDecision(False, "fixed-silence", "")


async def _fixed_speak(*, surprise, silence_seconds, recent_turns, topic_seeds):
    return SpeechDecision(True, "fixed-speak", "やあ")


def _reg(text: str) -> ModelRegistry:
    async def fake(model, messages, **kw):
        return {"choices": [{"message": {"content": text}}]}

    return ModelRegistry(completion_fn=fake)


async def _call(surprise, decide_fn, pending=False) -> SpeechDecision:
    return await should_speak(
        surprise=surprise, silence_seconds=10.0, recent_turns=[], topic_seeds=[],
        decide_fn=decide_fn, pending_obligation=pending,
    )


# ===== T2 death-detection =====
async def t_t2_inversion_silence_vote() -> bool:
    """固定 silence 投票でも、surprise を高/低に振ると speak/silence が反転（surprise が唯一の要因）。"""
    hi = await _call(HI + 20, _fixed_silence)
    lo = await _call(LO - 15, _fixed_silence)
    return hi.speak is True and lo.speak is False


async def t_t2_inversion_speak_vote() -> bool:
    """固定 speak 投票でも、低 surprise は強制 silence に反転（対称）。"""
    hi = await _call(HI + 20, _fixed_speak)
    lo = await _call(LO - 15, _fixed_speak)
    return hi.speak is True and lo.speak is False


def t_surprise_is_required() -> bool:
    p = inspect.signature(should_speak).parameters["surprise"]
    return p.default is inspect.Parameter.empty


# ===== ゲート帯 =====
async def t_middle_defers_to_llm() -> bool:
    mid = (HI + LO) // 2  # 40: LO<=surprise<HI → LLM の判断どおり
    sp = await _call(mid, _fixed_speak)
    si = await _call(mid, _fixed_silence)
    return sp.speak is True and si.speak is False


async def t_pending_overrides_even_high() -> bool:
    d = await _call(HI + 30, _fixed_speak, pending=True)
    return d.speak is False  # 強制発話帯でも pending が優先


async def t_force_speak_uses_llm_content() -> bool:
    d = await _call(HI + 10, _fixed_speak)
    return d.speak is True and d.content == "やあ"  # content は LLM のものを使う


async def t_force_speak_fallback_content() -> bool:
    d = await _call(HI + 10, _fixed_silence)  # LLM が content 無し → fallback
    return d.speak is True and d.content != ""


# ===== パーサ =====
def t_parse_yes() -> bool:
    d = parse_speech_decision("speak: yes\nreason: 話題がある\ncontent: 今日いい天気だね")
    return d.speak and d.content == "今日いい天気だね" and d.reason == "話題がある"


def t_parse_no() -> bool:
    d = parse_speech_decision("speak: no\nreason: 特にない")
    return (not d.speak) and d.reason == "特にない"


def t_parse_fullwidth() -> bool:
    d = parse_speech_decision("speak：yes\nreason：理由\ncontent：やあ")
    return d.speak and d.content == "やあ"


def t_parse_garbage_silence() -> bool:
    d = parse_speech_decision("ぐちゃぐちゃで何も無い文")
    return d.speak is False


def t_parse_ellipsis_silence() -> bool:
    return parse_speech_decision("…").speak is False


# ===== make_decide_fn（保守フォールバック）=====
async def t_decidefn_valid() -> bool:
    fn = make_decide_fn(_reg("speak: yes\nreason: r\ncontent: こんにちは"))
    d = await fn(surprise=40, silence_seconds=5, recent_turns=[], topic_seeds=[])
    return d.speak and d.content == "こんにちは"


async def t_decidefn_garbage_silence() -> bool:
    fn = make_decide_fn(_reg("???"))
    d = await fn(surprise=40, silence_seconds=5, recent_turns=[], topic_seeds=[])
    return d.speak is False


async def t_decidefn_apierror_silence() -> bool:
    async def boom(model, messages, **kw):
        raise RuntimeError("api down")

    fn = make_decide_fn(ModelRegistry(completion_fn=boom))
    d = await fn(surprise=40, silence_seconds=5, recent_turns=[], topic_seeds=[])
    return d.speak is False


async def main() -> None:
    check("T2 反転(silence 投票・surprise が唯一の要因)", await t_t2_inversion_silence_vote())
    check("T2 反転(speak 投票・対称)", await t_t2_inversion_speak_vote())
    check("surprise は必須引数(既定なし)", t_surprise_is_required())
    check("中間帯は LLM 判断どおり", await t_middle_defers_to_llm())
    check("pending は強制発話帯より優先(沈黙)", await t_pending_overrides_even_high())
    check("強制発話は LLM の content を使う", await t_force_speak_uses_llm_content())
    check("強制発話で content 無し→fallback", await t_force_speak_fallback_content())
    check("parse yes", t_parse_yes())
    check("parse no", t_parse_no())
    check("parse 全角コロン", t_parse_fullwidth())
    check("parse ゴミ→silence", t_parse_garbage_silence())
    check("parse …→silence", t_parse_ellipsis_silence())
    check("decide_fn 正常", await t_decidefn_valid())
    check("decide_fn ゴミ→silence", await t_decidefn_garbage_silence())
    check("decide_fn API 例外→silence", await t_decidefn_apierror_silence())


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
