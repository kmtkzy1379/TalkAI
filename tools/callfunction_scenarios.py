r"""Call-Function シナリオ A/B（実LLM・要 .env・コスト中）。

モデル横断で挙動を確認:
 S1 単発  : 「調子は？」→ 前置きの自然さ + self_status の tool_call 捕捉 → 報告。
 S2 マルチ: 「時刻とシステムの調子を両方教えて」→ tool_call が2件捕捉できるか。
 S3 失敗  : わざと失敗する能力(flaky)を呼ばせ、失敗を**正直に・理由つきで**報告できるか
            （ハルシネで「成功した」と言わないか）。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\callfunction_scenarios.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.response.function_dispatcher import parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402

MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5.5", "gemini/gemini-2.5-flash"]

TOOL_HINT = (
    "\n\n# 機能（Call-Function）の使い方\n"
    "必要なら提供された関数を呼んでよい。呼ぶ前に「ちょっと確認するね」等の短い前置きを一言添えてよい。"
    "関数名・引数・JSON は読み上げない。"
)


def make_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)

    def flaky(args):
        raise ConnectionError("外部サービスに接続できませんでした")

    reg.register(Capability(
        name="external_service_status",
        description="外部サービス(天気/在庫等)の稼働状況を問い合わせる。引数なし。",
        params_schema={}, handler=flaky,
    ))
    return reg


async def _turn(reg_model, registry, system, user_text):
    """1ターン: tools 付きで stream → (content, tool_calls)。"""
    schemas = registry.tool_schemas()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    sink: list = []
    content = ""
    async for piece in reg_model.stream_with_tools("response", messages, tools=schemas, tool_sink=sink):
        content += piece
    return content, sink


async def _report(reg_model, result_text):
    """結果ターン(tools 無し=1ホップ抑制): 結果を報告させる。"""
    msgs = [
        {"role": "system", "content": SPEECH_STYLE + f"\n\n# 機能実行結果\n{result_text}"},
        {"role": "user", "content": "（結果が出ました。一言で正直に報告して）"},
    ]
    out = ""
    async for piece in reg_model.stream("response", msgs):
        out += piece
    return out


async def run_model(model: str) -> None:
    print(f"\n========== {model} ==========")
    reg = ModelRegistry(overrides={"response": model})
    registry = make_registry()
    system = SPEECH_STYLE + TOOL_HINT
    try:
        # S1 単発
        c, sink = await _turn(reg, registry, system, "ねえイブ、今の調子とキューの状況どう？")
        print(f"  S1 前置き={c!r}  tool_calls={[parse_tool_call(t)[0] for t in sink]}")
        if sink:
            res = await registry.execute_async(*parse_tool_call(sink[0])[:2])
            print(f"     報告: {await _report(reg, res)!r}")

        # S2 マルチツール
        c2, sink2 = await _turn(reg, registry, system, "時刻とシステムの調子を両方教えて。")
        print(f"  S2 前置き={c2!r}  tool_calls={[parse_tool_call(t)[0] for t in sink2]}（{len(sink2)}件）")

        # S3 わざと失敗
        c3, sink3 = await _turn(reg, registry, system, "外部サービスの稼働状況を確認して。")
        names3 = [parse_tool_call(t)[0] for t in sink3]
        print(f"  S3 前置き={c3!r}  tool_calls={names3}")
        if sink3:
            res3 = await registry.execute_async(*parse_tool_call(sink3[0])[:2])
            print(f"     実行結果(失敗): {res3}")
            print(f"     正直報告: {await _report(reg, res3)!r}")
    except Exception as e:
        print(f"  ✗ {model} 失敗: {type(e).__name__}: {e}")


async def main() -> None:
    for m in MODELS:
        await run_model(m)


if __name__ == "__main__":
    asyncio.run(main())
