"""F0 基盤の決定論テスト（API 不要・純 stdlib）。

検証: 二重タイムスタンプ / 相対時刻 / ModelRegistry の role 解決と override /
completion_fn 注入で解決済みモデルへ配線 / ContextAssembler の時間接地と
「話題の種 vs 過去の記憶」ラベル分離（T6 過去参照防止の器）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f0_foundation.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.clock import Stamp, humanize  # noqa: E402
from eve.context_assembler import ContextAssembler, RagChunk, Turn  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402

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


# 1. clock — 二重時刻と相対表現
check("humanize たった今", humanize(2) == "たった今")
check("humanize 秒", humanize(42) == "42秒前")
check("humanize 分", humanize(125) == "2分前")
check("humanize 時間", humanize(7200) == "2時間前")
s = Stamp.now()
check("Stamp 二重保持", isinstance(s.iso, str) and isinstance(s.mono, float))

# 2. ModelRegistry — role 解決と override
reg = ModelRegistry()
check("resolve response 既定", reg.resolve("response") == "openai/gpt-4o")
check("resolve vlm_leaf 既定", reg.resolve("vlm_leaf").startswith("gemini/"))
reg.set_override("response", "gemini/gemini-2.5-pro")
check("resolve override 優先", reg.resolve("response") == "gemini/gemini-2.5-pro")
try:
    reg.resolve("nope")
    check("未知 role は KeyError", False)
except KeyError:
    check("未知 role は KeyError", True)

# 3. completion_fn 注入 → 解決済みモデルへ配線（litellm 不要）
_seen: dict[str, object] = {}


async def _fake_completion(model, messages, **kwargs):
    _seen["model"] = model
    _seen["messages"] = messages
    return {"ok": True}


reg2 = ModelRegistry(completion_fn=_fake_completion)
asyncio.run(reg2.complete("feedback", [{"role": "user", "content": "hi"}]))
check("complete は解決済みモデルへ", _seen.get("model") == reg2.resolve("feedback"))

# 4. ContextAssembler — 時間接地 + 話題の種/過去の記憶のラベル分離
now = Stamp.now()
three_min_ago = Stamp(iso=now.iso, mono=now.mono - 180)
ctx = ContextAssembler(system_prompt="SYS").assemble(
    user_text="今の話をしよう",
    recent_turns=[Turn("user", "昔これ言った", three_min_ago)],
    rag_chunks=[
        RagChunk("ラーメンが好き", now.iso, as_topic_seed=True),
        RagChunk("昨日の出来事", now.iso, as_topic_seed=False),
    ],
    now=now,
)
rendered = ctx.render()
check("ctx 相対時刻を注入", "3分前" in rendered)
check("ctx 話題の種ラベル", "話題の種" in rendered)
check("ctx 過去の記憶ラベル", "過去の記憶" in rendered)
check("ctx ユーザ発話=今", "ユーザ発話（今）" in rendered)
check("ctx system 分離", ctx.system == "SYS")

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
