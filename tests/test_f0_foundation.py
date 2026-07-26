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
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.clock import Stamp, elapsed_wall, humanize  # noqa: E402
from eve.context_assembler import ContextAssembler, RagChunk, Turn, messages_to_text  # noqa: E402
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
# elapsed_wall は tz-naive な timestamp でも例外を投げない（防御的・UTC扱い）
check("elapsed_wall naive 非クラッシュ", isinstance(elapsed_wall("2026-06-20T00:00:00"), float))
check("elapsed_wall 壊れ値は0", elapsed_wall("not-a-date") == 0.0)

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
# 直近ターンは壁時計(iso)で接地する（再起動後も正しい）。iso を 180 秒過去にする。
past_iso = (datetime.fromisoformat(now.iso) - timedelta(seconds=180)).isoformat()
three_min_ago = Stamp(iso=past_iso, mono=now.mono - 180)
ctx = ContextAssembler(system_prompt="SYS").assemble(
    user_text="今の話をしよう",
    recent_turns=[Turn("user", "昔これ言った", three_min_ago)],
    rag_chunks=[
        RagChunk("ラーメンが好き", now.iso, as_topic_seed=True),
        RagChunk("昨日の出来事", now.iso, as_topic_seed=False),
    ],
    now=now,
)
msgs = ctx  # assemble は native ロール messages を返す
joined = messages_to_text(msgs)
sysmsg = msgs[0]["content"]
check("ctx RAG に相対時刻接地(system)", "たった今" in sysmsg)
check("ctx 話題の種ラベル(system)", "話題の種" in sysmsg)
check("ctx 過去の記憶ラベル(system)", "過去の記憶" in sysmsg)
check("ctx ユーザ発話は user ロール末尾", msgs[-1]["role"] == "user" and msgs[-1]["content"] == "今の話をしよう")
check("ctx 過去発話は assistant/user ロール", any(m["role"] == "user" and "昔これ言った" in m["content"] for m in msgs))
check("ctx system に SYS とロールアンカー", "SYS" in sysmsg and "イブ" in sysmsg)

# J-2 ①: tools_active ブロックに「追加依頼は新しい部分だけを自己完結で委譲・実行中の内容を
# goal に再包含しない」指示が入る（superset goal 重複タスクの回帰ガード）。
tool_sys = ContextAssembler(system_prompt="SYS").assemble(
    user_text="それと沖縄の人口も", tools_active=True,
)[0]["content"]
check("J-2 ①: 委譲 goal は自己完結の指示", "それ単体で意味が通る一文" in tool_sys)
check("J-2 ①: 追加依頼は新しい部分だけ・実行中を再包含しない",
      "新しく頼む部分だけ" in tool_sys and "再依頼しない" in tool_sys)
# tools_active=False（自発発話/報告ターン等）ではブロックを出さない（1ホップ抑制と整合）。
notool_sys = ContextAssembler(system_prompt="SYS").assemble(user_text="x", tools_active=False)[0]["content"]
check("J-2 ①: tools 無効時は機能ブロックを出さない", "Call-Function" not in notool_sys)

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
