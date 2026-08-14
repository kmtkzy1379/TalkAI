"""Tier-1: API 使用量メーター（API 不要）。

**なぜ厳しく縛るか**: 計測のために本体を落としたら本末転倒。ここは応答の主経路
（`ModelRegistry.complete/stream/stream_with_tools`）に差し込むので、
**記録が失敗しても会話が続く**ことと、**既定 off で挙動が完全に不変**なことを固める。

もう1つは正直さ。費用は「引けなかった時に 0 として混ぜる」と、使っているのに
0 円に見える。桁を誤らせない方が、精度そのものより大事。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve import usage as U  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def tmp_meter(**kw):
    d = Path(tempfile.mkdtemp(prefix="eve_usage_"))
    return U.UsageMeter(d / "usage.jsonl", **kw)


class _Resp:
    def __init__(self, p, c) -> None:
        self.usage = {"prompt_tokens": p, "completion_tokens": c}
        self.model = "openai/gpt-4o"


def t_record() -> None:
    print("\n[1] 記録と集計")
    m = tmp_meter()
    cost = m.record(role="response", model="openai/gpt-4o",
                    prompt_tokens=1000, completion_tokens=100)
    check("費用が計算される", cost is not None and cost > 0, f"{cost}")
    s = m.summary()
    check("役割別に積む", s["by_role"]["response"]["calls"] == 1)
    check("モデル別にも積む", s["by_model"]["openai/gpt-4o"]["prompt_tokens"] == 1000)
    check("合計に入る", s["total"]["completion_tokens"] == 100)

    rows = [json.loads(x) for x in m.path.read_text(encoding="utf-8").splitlines()]
    check("1呼び出し1行の JSONL", len(rows) == 1, f"{len(rows)}")
    for key in ("ts", "role", "model", "prompt_tokens", "completion_tokens",
                "cost_usd", "accuracy"):
        check(f"  {key} が残る", key in rows[0], f"{rows[0]}")
    check("**トークン数を必ず残す**（価格が変わっても後で再計算できる）",
          rows[0]["prompt_tokens"] == 1000)


def t_unpriced_not_zero() -> None:
    print("\n[2] 価格が引けない時に 0 円と嘘をつかない")
    m = tmp_meter()
    cost = m.record(role="response", model="存在しない/モデル",
                    prompt_tokens=1000, completion_tokens=100)
    check("費用は None（0 ではない）", cost is None, f"{cost}")
    s = m.summary()
    check("**未価格の件数を別に数える**", s["total"]["unpriced_calls"] == 1)
    check("USD には足さない", s["total"]["cost_usd"] == 0.0)
    check("トークンは数える（使ったことは事実）", s["total"]["prompt_tokens"] == 1000)
    text = m.format_summary()
    check("集計表に警告が出る", "価格を引けなかった" in text, text)


def t_accuracy_labeled() -> None:
    print("\n[3] exact と approx を区別する")
    m = tmp_meter()
    m.record_response(role="feedback", model="openai/gpt-4o", response=_Resp(50, 5))
    m.record_stream(role="response", model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "こんにちは" * 20}],
                    output="やあ" * 10)
    rows = [json.loads(x) for x in m.path.read_text(encoding="utf-8").splitlines()]
    check("provider の usage は exact", rows[0]["accuracy"] == "exact", f"{rows[0]}")
    check("ストリーミングは approx", rows[1]["accuracy"] == "approx", f"{rows[1]}")
    check("approx でも 0 にはしない",
          rows[1]["prompt_tokens"] > 0 and rows[1]["completion_tokens"] > 0, f"{rows[1]}")


def t_price_override() -> None:
    print("\n[4] 価格の手動指定")
    old = os.environ.get("USAGE_PRICE_OVERRIDE")
    os.environ["USAGE_PRICE_OVERRIDE"] = "myco/model=1e-05:2e-05"
    try:
        p = U.price_of("myco/model")
        check("上書きが効く", p == (1e-05, 2e-05), f"{p}")
        m = tmp_meter()
        cost = m.record(role="x", model="myco/model", prompt_tokens=100, completion_tokens=10)
        check("その単価で計算される", abs(cost - (100 * 1e-05 + 10 * 2e-05)) < 1e-12, f"{cost}")
        os.environ["USAGE_PRICE_OVERRIDE"] = "壊れた指定"
        check("壊れた指定は無視して落ちない", U.price_of("myco/model") is None)
    finally:
        if old is None:
            os.environ.pop("USAGE_PRICE_OVERRIDE", None)
        else:
            os.environ["USAGE_PRICE_OVERRIDE"] = old


def t_never_breaks_pipeline() -> None:
    print("\n[5] 計測のために本体を止めない")
    m = tmp_meter()
    m.path = Path("Z:/存在しない/場所/usage.jsonl")  # 書き込めない場所
    cost = m.record(role="response", model="openai/gpt-4o",
                    prompt_tokens=10, completion_tokens=1)
    check("書き込めなくても例外を出さない", cost is not None)
    check("メモリ上の集計は生きている", m.summary()["total"]["calls"] == 1)

    m2 = tmp_meter()
    check("壊れた response でも落ちない",
          m2.record_response(role="x", model="openai/gpt-4o", response=object()) is not None)
    check("その場合 0 トークンで記録される", m2.summary()["total"]["prompt_tokens"] == 0)


def t_registry_wiring() -> None:
    print("\n[6] ModelRegistry への差し込み")

    async def scenario() -> None:
        U.install(None)
        calls = []

        async def fake_complete(model, messages, **kw):
            calls.append(model)
            return _Resp(11, 3)

        reg = ModelRegistry(completion_fn=fake_complete)
        await reg.complete("response", [{"role": "user", "content": "やあ"}])
        check("**既定（メーター無し）では何も起きない**", U.meter() is None)

        m = tmp_meter()
        U.install(m)
        try:
            await reg.complete("response", [{"role": "user", "content": "やあ"}])
            check("complete が記録される", m.summary()["total"]["calls"] == 1,
                  f"{m.summary()['total']}")
            check("役割名で積まれる", "response" in m.summary()["by_role"])

            async def fake_stream(model, messages, **kw):
                for piece in ("こん", "にちは"):
                    yield piece

            reg2 = ModelRegistry(stream_fn=fake_stream)
            out = [p async for p in reg2.stream("response", [{"role": "user", "content": "ねえ"}])]
            check("stream の中身は変わらない", "".join(out) == "こんにちは", f"{out}")
            check("stream も記録される", m.summary()["total"]["calls"] == 2,
                  f"{m.summary()['total']}")

            # 途中で切られても（barge-in）そこまでを数える
            async def half_stream(model, messages, **kw):
                yield "とちゅ"
                raise asyncio.CancelledError

            reg3 = ModelRegistry(stream_fn=half_stream)
            try:
                async for _ in reg3.stream("response", [{"role": "user", "content": "ねえ"}]):
                    pass
            except asyncio.CancelledError:
                pass
            check("**barge-in で切れてもそこまでを数える**（費用は発生している）",
                  m.summary()["total"]["calls"] == 3, f"{m.summary()['total']}")

            # 記録側が壊れていても応答は流れる
            m.record = lambda **kw: (_ for _ in ()).throw(RuntimeError("メーター故障"))
            out2 = [p async for p in reg2.stream("response",
                                                 [{"role": "user", "content": "ねえ"}])]
            check("**メーターが例外を投げても応答は流れる**",
                  "".join(out2) == "こんにちは", f"{out2}")
        finally:
            U.install(None)

    asyncio.run(scenario())


def t_tools_schema_counted() -> None:
    print("\n[7] tools のスキーマも入力に数える（過小評価しない）")

    async def scenario() -> None:
        m = tmp_meter()
        U.install(m)
        try:
            async def fake(model, messages, tools=None, tool_sink=None, **kw):
                yield "はい"

            reg = ModelRegistry(stream_fn=fake)
            msgs = [{"role": "user", "content": "やって"}]
            sink: list = []
            async for _ in reg.stream_with_tools("response", msgs, tools=[], tool_sink=sink):
                pass
            base = m.summary()["total"]["prompt_tokens"]
            m2 = tmp_meter()
            U.install(m2)
            big = [{"type": "function", "function": {"name": "delegate_task",
                                                     "description": "x" * 400,
                                                     "parameters": {}}}]
            sink2: list = []
            async for _ in reg.stream_with_tools("response", msgs, tools=big, tool_sink=sink2):
                pass
            with_tools = m2.summary()["total"]["prompt_tokens"]
            check("tools を渡すと入力トークンが増える", with_tools > base,
                  f"{with_tools} vs {base}")
        finally:
            U.install(None)

    asyncio.run(scenario())


def t_format() -> None:
    print("\n[8] 集計表の体裁")
    m = tmp_meter()
    m.record(role="response", model="openai/gpt-4o", prompt_tokens=100, completion_tokens=20)
    m.record(role="vlm_leaf", model="gemini/gemini-2.5-flash",
             prompt_tokens=900, completion_tokens=30)
    text = m.format_summary(per_hour=10.0)
    check("役割が並ぶ", "response" in text and "vlm_leaf" in text)
    check("合計が出る", "合計" in text)
    check("1時間あたりが出る", "1時間あたり" in text, text)
    check("費用の高い順に並ぶ",
          text.index("response") < text.index("vlm_leaf"), text)


if __name__ == "__main__":
    t_record()
    t_unpriced_not_zero()
    t_accuracy_labeled()
    t_price_override()
    t_never_breaks_pipeline()
    t_registry_wiring()
    t_tools_schema_counted()
    t_format()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
