"""J-2 P1-1: DeliveryChecker の決定論テスト（API 不要・fake model registry）。

検証:
- parse_delivery_verdict: 3値抽出・解析失敗/不明瞭は 'delivered'（安全側・過剰な再配達をしない）。
- forgotten → redeliver_fn を attempts+1/dedup_key保持/content・function_name・ok を維持して呼ぶ。
- delivered/withheld → redeliver_fn を呼ばない（誤爆しない）。
- attempts が上限に達していれば forgotten でも再配達しない（barge-in 側の上限と共有）。
- LLM 例外は再配達しない（fail-safe: 判定不能を「過剰な再送」に倒さない）。
- CallFunctionResult でない payload は無反応（クラッシュしない）。
- start/stop のライフサイクル（未 start でも stop が安全・drain）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_delivery_checker.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind  # noqa: E402
from eve.response.delivery_checker import DeliveryChecker, parse_delivery_verdict  # noqa: E402

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


class FakeModel:
    def __init__(self, content=None, raise_exc=None):
        self.content = content
        self.calls = 0
        self._raise_exc = raise_exc
        self.seen_messages = None

    async def complete(self, role, messages, **kw):
        self.calls += 1
        self.seen_messages = messages
        if self._raise_exc is not None:
            raise self._raise_exc
        return {"choices": [{"message": {"content": self.content}}]}


class RecordingRedeliver:
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, stim, should_abort):
        self.calls.append((stim, should_abort))


async def _run_once(model, redeliver, stim, *, max_attempts=2):
    dc = DeliveryChecker(model_registry=model, redeliver_fn=redeliver, max_attempts=max_attempts)
    dc.start()
    dc.submit(stim, recent_text="[ユーザ] 30秒後に時刻を教えて", spoken_text="うん。")
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()


def _stim(content="30秒後の時刻は12:00だよ", attempts=0, ok=True, dedup="task:t1"):
    return Stimulus(
        kind=StimulusKind.CALLFUNCTION_RESULT,
        payload=CallFunctionResult("delegate_task", content, ok, attempts),
        dedup_key=dedup,
    )


# ---- parse_delivery_verdict ----

def t_parse_verdict():
    return (
        parse_delivery_verdict("forgotten") == "forgotten"
        and parse_delivery_verdict("Delivered") == "delivered"
        and parse_delivery_verdict("  withheld  ") == "withheld"
        and parse_delivery_verdict("うーん、たぶん forgotten かな") == "forgotten"
        and parse_delivery_verdict("") == "delivered"  # 空は安全側
        and parse_delivery_verdict(None) == "delivered"  # 解析不能は安全側
        and parse_delivery_verdict("よくわからない") == "delivered"  # 不明瞭は安全側
    )


# ---- forgotten → 再配達 ----

async def t_forgotten_triggers_redeliver():
    m = FakeModel("forgotten")
    rd = RecordingRedeliver()
    stim = _stim(content="30秒後の時刻は12:00だよ", attempts=0, dedup="task:t1")
    await _run_once(m, rd, stim)
    if len(rd.calls) != 1:
        return False
    retry, should_abort = rd.calls[0]
    return (
        m.calls == 1
        and retry.kind == StimulusKind.CALLFUNCTION_RESULT
        and retry.dedup_key == "task:t1"  # dedup_key 保持
        and retry.payload.attempts == 1  # +1
        and retry.payload.content == "30秒後の時刻は12:00だよ"  # 内容保持
        and retry.payload.function_name == "delegate_task"
        and retry.payload.ok is True
        and should_abort() is False  # ターン完了済み＝再考すべきレースは無い
    )


async def t_delivered_no_redeliver():
    m = FakeModel("delivered")
    rd = RecordingRedeliver()
    await _run_once(m, rd, _stim())
    return m.calls == 1 and rd.calls == []


async def t_withheld_no_redeliver():
    m = FakeModel("withheld")
    rd = RecordingRedeliver()
    await _run_once(m, rd, _stim())
    return m.calls == 1 and rd.calls == []


async def t_max_attempts_blocks_redeliver():
    # forgotten でも attempts が上限(既定2)に達していれば再配達しない（barge-in 側と共有の上限）。
    m = FakeModel("forgotten")
    rd = RecordingRedeliver()
    await _run_once(m, rd, _stim(attempts=2), max_attempts=2)
    return m.calls == 1 and rd.calls == []


async def t_llm_exception_no_redeliver():
    # 判定不能(例外) → 再配達しない（fail-safe: 分からない時は過剰再送に倒さない）。
    m = FakeModel(raise_exc=RuntimeError("api down"))
    rd = RecordingRedeliver()
    await _run_once(m, rd, _stim())
    return m.calls == 1 and rd.calls == []


async def t_non_callfunction_payload_ignored():
    # payload が CallFunctionResult でない（想定外の注入）→ 無反応・無クラッシュ。
    m = FakeModel("forgotten")
    rd = RecordingRedeliver()
    stim = Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT, payload="ただの文字列", dedup_key="x")
    await _run_once(m, rd, stim)
    return m.calls == 0 and rd.calls == []


async def t_redeliver_fn_none_safe():
    # redeliver_fn が None（未配線）でも forgotten 判定でクラッシュしない。
    m = FakeModel("forgotten")
    dc = DeliveryChecker(model_registry=m, redeliver_fn=None, max_attempts=2)
    dc.start()
    dc.submit(_stim(), recent_text="", spoken_text="")
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()
    return m.calls == 1


async def t_stop_without_start_is_safe():
    # start() を呼ばずに stop() してもハングしない（未使用時の安全性）。
    m = FakeModel("delivered")
    dc = DeliveryChecker(model_registry=m, redeliver_fn=None, max_attempts=2)
    await asyncio.wait_for(dc.stop(), timeout=2.0)
    return True


async def t_prompt_includes_report_and_spoken():
    # judge へ渡す messages に報告内容/実発話/直近会話が含まれる（判定材料の取りこぼし防止）。
    m = FakeModel("delivered")
    rd = RecordingRedeliver()
    dc = DeliveryChecker(model_registry=m, redeliver_fn=rd, max_attempts=2)
    dc.start()
    stim = _stim(content="VOICEVOXは無料の音声合成ソフトです")
    dc.submit(stim, recent_text="[ユーザ] ボイスボックスについて調べて", spoken_text="30秒後に教えるね。")
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()
    user_msg = m.seen_messages[1]["content"]
    return (
        "VOICEVOXは無料の音声合成ソフトです" in user_msg
        and "30秒後に教えるね。" in user_msg
        and "ボイスボックスについて調べて" in user_msg
    )


async def main():
    check("parse_delivery_verdict: 3値抽出+安全側デフォルト", t_parse_verdict())
    check("forgotten→再配達(attempts+1/dedup保持/内容保持)", await t_forgotten_triggers_redeliver())
    check("delivered→再配達しない", await t_delivered_no_redeliver())
    check("withheld→再配達しない", await t_withheld_no_redeliver())
    check("上限到達済みは forgotten でも再配達しない", await t_max_attempts_blocks_redeliver())
    check("LLM例外→再配達しない(fail-safe)", await t_llm_exception_no_redeliver())
    check("非CallFunctionResult payloadは無反応", await t_non_callfunction_payload_ignored())
    check("redeliver_fn未配線でも無クラッシュ", await t_redeliver_fn_none_safe())
    check("未startでのstopは安全", await t_stop_without_start_is_safe())
    check("judgeへ報告内容/実発話/直近会話を渡す", await t_prompt_includes_report_and_spoken())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
