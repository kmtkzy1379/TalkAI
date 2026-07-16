r"""Call-Function 実機スモーク（実LLM・要 .env キー・コスト小）。

主要不確実点の検証: litellm に `tools=` を渡し、streaming で `tool_calls` を回収できるか
（`model_registry.stream_with_tools` + `merge_tool_call_deltas` が実モデルで通るか）。
gpt-4o と gemini-2.5-flash の両方で:
 1) 「調子/キューはどう？」→ self_status の tool_call が出るか・捕捉できるか。
 2) 出たら CapabilityRegistry で実行→結果を「# 機能実行結果」で再投入→Eve が報告する文を1回生成。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\callfunction_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.response.function_dispatcher import parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402

MODELS = ["openai/gpt-4o", "gemini/gemini-2.5-flash"]

SYSTEM = (
    SPEECH_STYLE
    + "\n\n# 機能（Call-Function）の使い方\n"
    "必要なら提供された関数を呼んでよい。呼ぶ前に「ちょっと確認するね」等の短い前置きを一言添えてよい。"
    "関数名・引数・JSON は読み上げない。"
)


async def run_model(model: str) -> None:
    print(f"\n========== {model} ==========")
    reg = ModelRegistry(overrides={"response": model})
    caps = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)
    schemas = caps.tool_schemas()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "ねえイブ、今の調子とか、未処理のキューの状況ってどう？"},
    ]
    sink: list = []
    content = ""
    try:
        async for piece in reg.stream_with_tools("response", messages, tools=schemas, tool_sink=sink):
            content += piece
    except Exception as e:
        print(f"  ✗ stream_with_tools 例外: {type(e).__name__}: {e}")
        return

    print(f"  発話content: {content!r}")
    print(f"  捕捉 tool_calls: {len(sink)} 件")
    if not sink:
        print("  △ tool_call が出なかった（モデルが関数を呼ばなかった）")
        return
    # 実行 → 結果を再投入して報告文を生成
    for tc in sink:
        name, args, cid = parse_tool_call(tc)
        result = await caps.execute_async(name, args)  # 統一経路（async 専用能力にも対応）
        print(f"   → {name}({args}) = {result}")
        followup = [
            {"role": "system", "content": SPEECH_STYLE + f"\n\n# 機能実行結果\n{result}"},
            {"role": "user", "content": "（結果が出ました。今の状態を一言で教えて）"},
        ]
        out = ""
        async for piece in reg.stream("response", followup):  # 結果ターンは tools 無し（1ホップ抑制）
            out += piece
        print(f"   🤖報告: {out!r}")


async def main() -> None:
    for m in MODELS:
        try:
            await run_model(m)
        except Exception as e:
            print(f"  ✗ {m} 失敗: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
