"""配達確認の決定論テスト（API 不要・LLM を使わない実装の固定）。

検証:
- decide_delivery の5規則（無発話 / 丸ごと含む / 内容語ゼロ / 内容語一致 / 不一致）。
- **実機事故 2026-08-15 02:44 の回帰**: 報告「（任せてくれたタスク）表示したよ。」× 実発話
  「表示したよ。」が delivered になる（旧 LLM 判定は forgotten を返し二重回答を起こした）。
- **元事故 2026-07-18 の保護**: 報告に触れず別の話だけをしたターンは not_conveyed になる。
- 未配達 → redeliver_fn を attempts+1 / dedup_key 保持 / 内容保持 / reason=not_conveyed で呼ぶ。
- delivered → redeliver_fn を呼ばない（誤爆しない）。
- attempts 上限で再配達しない / 非 CallFunctionResult は無反応 / 未配線・未 start が安全。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_delivery_checker.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.pipeline.stimulus import (  # noqa: E402
    REDELIVER_BARGE_IN,
    REDELIVER_NOT_CONVEYED,
    CallFunctionResult,
    Stimulus,
    StimulusKind,
)
from eve.response.delivery_checker import (  # noqa: E402
    VERDICT_DELIVERED,
    VERDICT_UNSURE,
    DeliveryChecker,
    decide_delivery_fast,
    parse_delivery_verdict,
)
from eve.text_match import content_tokens  # noqa: E402

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


class RecordingRedeliver:
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, stim, should_abort):
        self.calls.append((stim, should_abort))


class FakeModel:
    """判定LLM の代役。**呼ばれた回数**も見る（決定論で済む分は呼ばれてはいけない）。"""

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


async def _run_once(redeliver, stim, *, spoken="うん。", max_attempts=2, model=None):
    model = model if model is not None else FakeModel("conveyed")
    dc = DeliveryChecker(model_registry=model, redeliver_fn=redeliver, max_attempts=max_attempts)
    dc.start()
    dc.submit(stim, recent_text="[ユーザ] 30秒後に時刻を教えて", spoken_text=spoken)
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()
    return model


def _stim(content="30秒後の時刻は12:00だよ", attempts=0, ok=True, dedup="task:t1"):
    return Stimulus(
        kind=StimulusKind.CALLFUNCTION_RESULT,
        payload=CallFunctionResult("delegate_task", content, ok, attempts),
        dedup_key=dedup,
    )


# ---- 純関数 ----

def t_content_tokens():
    # 汎用動作語は内容語から除く / ひらがなだけの語は拾わない。
    return (
        content_tokens("表示したよ。") == set()
        and "チーズケーキ" in content_tokens("チーズケーキの画像を表示したよ。")
        and "表示" not in content_tokens("チーズケーキの画像を表示したよ。")
        and "VOICEVOX" in content_tokens("VOICEVOXは無料の音声合成ソフトです")
    )


def t_rules():
    return (
        # 1) 無発話（決定論・確実）
        decide_delivery_fast("ラーメンの画像を表示したよ。", "") == REDELIVER_NOT_CONVEYED
        # 2) 丸ごと含む
        and decide_delivery_fast("12:00だよ", "はい、12:00だよ。") == VERDICT_DELIVERED
        # 4) 内容語が重なる（言い換え可）
        and decide_delivery_fast(
            "（任せてくれたタスク）ラーメンの画像を表示したよ。",
            "今度はラーメンの画像を表示したよ。") == VERDICT_DELIVERED
        # 5) 全く重ならない → **決めない**（LLM に回す）
        and decide_delivery_fast(
            "（任せてくれたタスク）VOICEVOXは無料の音声合成ソフトだよ。",
            "うん、今日はいい天気だね。") == VERDICT_UNSURE
    )


def t_regression_20260815_contentless_report():
    """実機事故の回帰: 内容語ゼロの報告 × ほぼ同一の発話 → delivered（再配達しない）。

    旧実装(gpt-4o-mini 判定)はここで forgotten を返し、再配達された応答LLM が
    直前のユーザ発話への謝罪ごと作り直して二重回答になった。
    **決定論の関門で確定する＝LLM の気分で揺れない**ことが要点。
    """
    return (
        decide_delivery_fast("（任せてくれたタスク）表示したよ。", "表示したよ。") == VERDICT_DELIVERED
        # 定型接頭辞を剥がさずに照合していると「タスク」が一致して常に delivered になる。
        # 内容語ゼロ側の規則で通ることを、接頭辞なしでも確認する。
        and decide_delivery_fast("表示したよ。", "表示したよ。") == VERDICT_DELIVERED
        # 報告が内容語ゼロなら、発話が全然違う話でも再配達しない（捏造を強制しないため）。
        and decide_delivery_fast("（任せてくれたタスク）やっておいたよ。",
                                 "今日はいい天気だね。") == VERDICT_DELIVERED
    )


def t_paraphrase_goes_to_llm():
    """ユーザ指摘 2026-08-15: 「チーズケーキの画像を表示したよ」×「やったよ」。

    語は1つも重ならないが、直前にユーザ自身が頼んでいるので文脈では通じている。
    一方「天気は雨だよ」×「うん、そうみたい」は中身が本当に届いていない。
    **同じ形なのに答えが違う**＝決定論では決められないので、両方 LLM に回すのが正しい。
    """
    return (
        decide_delivery_fast("（任せてくれたタスク）チーズケーキの画像を表示したよ。",
                             "やったよ。") == VERDICT_UNSURE
        and decide_delivery_fast("（任せてくれたタスク）チーズケーキの画像を表示したよ。",
                                 "出しといたよ、見てみて。") == VERDICT_UNSURE
        and decide_delivery_fast("（任せてくれたタスク）東京の明日の天気は雨だよ。",
                                 "うん、そうみたい。") == VERDICT_UNSURE
    )


def t_parse_verdict():
    # "not_conveyed" は "conveyed" を部分文字列に含む＝先に見ないと取り違える。
    return (
        parse_delivery_verdict("conveyed") == VERDICT_DELIVERED
        and parse_delivery_verdict("not_conveyed") == REDELIVER_NOT_CONVEYED
        and parse_delivery_verdict("  NOT CONVEYED  ") == REDELIVER_NOT_CONVEYED
        and parse_delivery_verdict("") == VERDICT_DELIVERED          # 空は安全側
        and parse_delivery_verdict(None) == VERDICT_DELIVERED        # 解析不能も安全側
        and parse_delivery_verdict("よくわからない") == VERDICT_DELIVERED  # 不明瞭も安全側
    )


# ---- サイドカー ----

async def t_not_conveyed_triggers_redeliver():
    rd = RecordingRedeliver()
    stim = _stim(content="東京の明日の天気は雨だよ", attempts=0, dedup="task:t1")
    # 発話は報告と内容語が1つも重ならない → LLM に回り、not_conveyed が返る想定。
    m = await _run_once(rd, stim, spoken="さっきの話だけど、それ面白いね。",
                        model=FakeModel("not_conveyed"))
    if len(rd.calls) != 1 or m.calls != 1:
        return False
    retry, should_abort = rd.calls[0]
    return (
        retry.kind == StimulusKind.CALLFUNCTION_RESULT
        and retry.dedup_key == "task:t1"            # dedup_key 保持
        and retry.payload.attempts == 1             # +1
        and retry.payload.content == "東京の明日の天気は雨だよ"  # 内容保持
        and retry.payload.function_name == "delegate_task"
        and retry.payload.ok is True
        and retry.payload.reason == REDELIVER_NOT_CONVEYED  # barge-in と取り違えない
        and should_abort() is False
    )


async def t_delivered_no_redeliver():
    # 決定論で delivered が確定する＝**LLM を呼ばない**（コストを増やさない）。
    rd = RecordingRedeliver()
    m = await _run_once(rd, _stim(content="ラーメンの画像を表示したよ"),
                        spoken="今度はラーメンの画像を表示したよ。")
    return rd.calls == [] and m.calls == 0


async def t_contentless_no_redeliver():
    """事故の再現条件そのもの: 内容語ゼロの報告では再配達が起きず、LLM も呼ばれない。"""
    rd = RecordingRedeliver()
    m = await _run_once(rd, _stim(content="（任せてくれたタスク）表示したよ。"), spoken="表示したよ。")
    return rd.calls == [] and m.calls == 0


async def t_llm_says_conveyed_no_redeliver():
    """ユーザ指摘の形（語が重ならないが文脈で通じている）→ LLM が conveyed → 再配達しない。"""
    rd = RecordingRedeliver()
    m = await _run_once(rd, _stim(content="チーズケーキの画像を表示したよ"), spoken="やったよ。",
                        model=FakeModel("conveyed"))
    return rd.calls == [] and m.calls == 1


async def t_llm_exception_no_redeliver():
    """判定LLM が落ちたら delivered 扱い（不明を過剰な再送に倒さない）。"""
    rd = RecordingRedeliver()
    m = await _run_once(rd, _stim(content="東京の天気は雨だよ"), spoken="関係ない話。",
                        model=FakeModel(raise_exc=RuntimeError("api down")))
    return rd.calls == [] and m.calls == 1


async def t_llm_prompt_includes_context():
    """判定LLM には直近会話・報告・実発話がすべて渡る（文脈で判断させるため）。"""
    rd = RecordingRedeliver()
    m = FakeModel("conveyed")
    dc = DeliveryChecker(model_registry=m, redeliver_fn=rd, max_attempts=2)
    dc.start()
    dc.submit(_stim(content="VOICEVOXは無料の音声合成ソフトです"),
              recent_text="[ユーザ] ボイスボックスについて調べて", spoken_text="やっといたよ。")
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()
    user_msg = m.seen_messages[1]["content"]
    return ("VOICEVOXは無料の音声合成ソフトです" in user_msg
            and "やっといたよ。" in user_msg
            and "ボイスボックスについて調べて" in user_msg)


async def t_max_attempts_blocks_redeliver():
    rd = RecordingRedeliver()
    await _run_once(rd, _stim(content="東京の天気は雨だよ", attempts=2),
                    spoken="関係ない話。", max_attempts=2, model=FakeModel("not_conveyed"))
    return rd.calls == []


async def t_non_callfunction_payload_ignored():
    rd = RecordingRedeliver()
    stim = Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT, payload="ただの文字列", dedup_key="x")
    await _run_once(rd, stim)
    return rd.calls == []


async def t_redeliver_fn_none_safe():
    dc = DeliveryChecker(model_registry=FakeModel("not_conveyed"), redeliver_fn=None, max_attempts=2)
    dc.start()
    dc.submit(_stim(content="東京の天気は雨だよ"), recent_text="", spoken_text="別の話。")
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()
    return True


async def t_model_none_safe():
    # 判定LLM 未配線でも、曖昧ケースは delivered 扱いで無クラッシュ（過剰再送に倒さない）。
    rd = RecordingRedeliver()
    dc = DeliveryChecker(model_registry=None, redeliver_fn=rd, max_attempts=2)
    dc.start()
    dc.submit(_stim(content="東京の天気は雨だよ"), recent_text="", spoken_text="別の話。")
    await asyncio.wait_for(dc._inbox.join(), timeout=2.0)
    await dc.stop()
    return rd.calls == []


async def t_stop_without_start_is_safe():
    dc = DeliveryChecker(model_registry=FakeModel("conveyed"), redeliver_fn=None, max_attempts=2)
    await asyncio.wait_for(dc.stop(), timeout=2.0)
    return True


def t_default_reason_is_barge_in():
    # 既定は barge_in（既存の再配達経路の文言を変えない）。
    return CallFunctionResult("f", "c").reason == REDELIVER_BARGE_IN


async def main():
    check("content_tokens: 汎用動作語を除く", t_content_tokens())
    check("決定論の関門: 5規則(5番目は unsure)", t_rules())
    check("回帰(2026-08-15): 内容語ゼロ報告×同一発話→delivered", t_regression_20260815_contentless_report())
    check("言い換え・指示語は LLM に回す(ユーザ指摘)", t_paraphrase_goes_to_llm())
    check("parse: not_conveyed を conveyed と取り違えない", t_parse_verdict())
    check("CallFunctionResult の既定 reason は barge_in", t_default_reason_is_barge_in())
    check("未配達→再配達(attempts+1/dedup保持/reason=not_conveyed)", await t_not_conveyed_triggers_redeliver())
    check("delivered確定→再配達もLLM呼び出しもしない", await t_delivered_no_redeliver())
    check("内容語ゼロ報告→再配達もLLM呼び出しもしない(二重回答の根絶)", await t_contentless_no_redeliver())
    check("LLMがconveyed→再配達しない", await t_llm_says_conveyed_no_redeliver())
    check("LLM例外→再配達しない(fail-safe)", await t_llm_exception_no_redeliver())
    check("LLMへ直近会話/報告/実発話を渡す", await t_llm_prompt_includes_context())
    check("上限到達済みは再配達しない", await t_max_attempts_blocks_redeliver())
    check("非CallFunctionResult payloadは無反応", await t_non_callfunction_payload_ignored())
    check("redeliver_fn未配線でも無クラッシュ", await t_redeliver_fn_none_safe())
    check("判定LLM未配線でも安全側(再配達しない)", await t_model_none_safe())
    check("未startでのstopは安全", await t_stop_without_start_is_safe())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
