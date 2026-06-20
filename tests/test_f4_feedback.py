"""F4 FeedbackLLM の決定論テスト（API 不要・fake LLM/embedder）。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f4_feedback.py
ハーネスは他 Tier-1 と同形（runner 無し・PASS/FAIL カウンタ・合計表示・exit code）。
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)  # ログを黙らせて出力をクリーンに

from eve.clock import Stamp  # noqa: E402
from eve.context_assembler import Turn  # noqa: E402
from eve.feedback import (  # noqa: E402
    NEUTRAL_SURPRISE,
    FeedbackLLM,
    FeedbackWorker,
    PredictionState,
    parse_feedback,
)
from eve.feedback.prompts import build_messages, build_user_text  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import Embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, Stimulus, StimulusKind  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402


async def _noop_play(audio) -> None:
    pass

_tmpdir = tempfile.mkdtemp(prefix="eve_f4_")
_counter = 0


def _tmp() -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"rag{_counter}.jsonl")


def _turn(speaker: str, text: str) -> Turn:
    return Turn(speaker, text, Stamp.now())


class FakeEmbedder(Embedder):
    """キーワード軸の決定論ベクトル（test_f3_5_rag と同方式）。"""

    AXES = ["夏", "祭り", "ラーメン", "仕事", "旅行", "音楽"]

    def __init__(self) -> None:
        self.dim = len(self.AXES)

    def _vec(self, text: str) -> list[float]:
        v = [1.0 if ax in text else 0.0 for ax in self.AXES]
        if not any(v):
            v = [0.001] * len(self.AXES)
        return v

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _store(**kw) -> RagStore:
    s = RagStore(FakeEmbedder(), rag_file=_tmp(), **kw)
    s.rel_baseline = 0.0
    return s


def _reg_const(text: str) -> ModelRegistry:
    """常に同じタグ付きテキストを返す fake registry。"""

    async def fake(model, messages, **kw):
        return {"choices": [{"message": {"content": text}}]}

    return ModelRegistry(completion_fn=fake)


class GatedFake:
    """並行度・呼び出し回数・入力スパンを観測でき、gate で run を保持できる fake。"""

    def __init__(self, gate: "asyncio.Event | None" = None) -> None:
        self.gate = gate
        self.concurrent = 0
        self.max_concurrent = 0
        self.calls = 0
        self.seen_spans: list[str] = []

    async def __call__(self, model, messages, **kw):
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.seen_spans.append(messages[1]["content"])
        n = self.calls
        try:
            if self.gate is not None:
                await self.gate.wait()
            return {
                "choices": [
                    {"message": {"content": f"summary: s{n}\nnext_prediction: p{n}\nsurprise: {n * 10}"}}
                ]
            }
        finally:
            self.concurrent -= 1


def _cache() -> ConversationCache:
    return ConversationCache(history_file=_tmp())

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


# ========== B1 PredictionState ==========
def t_predstate_defaults() -> bool:
    s = PredictionState()
    return (
        s.last_prediction is None
        and s.last_feedback is None
        and s.watermark is None
        and s.surprise == NEUTRAL_SURPRISE
    )


def t_predstate_surprise_method() -> bool:
    s = PredictionState()
    s.note_feedback_surprise(55)
    a = s.surprise == 55
    s.note_feedback_surprise(150)  # クランプ上限
    b = s.surprise == 100
    s.note_feedback_surprise(-10)  # クランプ下限
    c = s.surprise == 0
    return a and b and c


def t_predstate_surprise_garbage() -> bool:
    s = PredictionState()
    s.note_feedback_surprise("xx")  # 壊れた値 → 中立に倒す（例外を出さない）
    return s.surprise == NEUTRAL_SURPRISE


# ========== B2 parser ==========
_FULL = """summary: ユーザは夏祭りの話で盛り上がっている
emotion: 楽しさ
user_emotion: わくわく
next_prediction: 次は花火の話に移る
surprise: 35
reason: 直近の話題が祭り中心だから
tags: 夏祭り, 花火, 思い出
"""


def t_parser_full() -> bool:
    r = parse_feedback(_FULL)
    return (
        r.summary == "ユーザは夏祭りの話で盛り上がっている"
        and r.emotions == "楽しさ"
        and r.user_emotion == "わくわく"
        and r.next_prediction == "次は花火の話に移る"
        and r.prediction_diff == 35
        and r.reason == "直近の話題が祭り中心だから"
        and r.topic_tags == ["夏祭り", "花火", "思い出"]
    )


def t_parser_partial() -> bool:
    # summary と next_prediction のみ。diff は None（carry-forward の合図）。
    r = parse_feedback("summary: 部分的な出力\nnext_prediction: 続きを聞く")
    return (
        r.summary == "部分的な出力"
        and r.next_prediction == "続きを聞く"
        and r.prediction_diff is None  # 0 ではない
        and r.emotions == ""
        and r.topic_tags == []
    )


def t_parser_fullwidth_colon() -> bool:
    r = parse_feedback("要約：全角コロンの行\n予測差：72")
    return r.summary == "全角コロンの行" and r.prediction_diff == 72


def t_parser_garbage() -> bool:
    r = parse_feedback("これはタグのない自由文。なにも抽出できない。")
    return r.is_empty() and r.prediction_diff is None and r.topic_tags == []


def t_parser_diff_clamp_and_robust() -> bool:
    # 値に語が混じっても整数を拾い、上限はパース後にクランプ前提（ここでは生値）。
    r = parse_feedback("surprise: 約 88 くらい")
    return r.prediction_diff == 88


# ========== B3 prompt builder ==========
def t_prompt_includes_prediction_and_turns() -> bool:
    turns = [_turn("user", "こんにちは"), _turn("eve", "やあ！")]
    msgs = build_messages(turns, "次は天気の話")
    sys_ok = msgs[0]["role"] == "system" and "summary:" in msgs[0]["content"]
    u = msgs[1]["content"]
    return (
        sys_ok
        and "# 前回の予測\n次は天気の話" in u
        and "[ユーザ] こんにちは" in u
        and "[イブ] やあ！" in u
    )


def t_prompt_cold_sentinel() -> bool:
    u = build_user_text([_turn("eve", "ひとりごと")], None)
    return "初回・前回の予測なし" in u and "[イブ] ひとりごと" in u


# ========== B4 FeedbackLLM.run ==========
async def t_fb_cold_start() -> bool:
    fb = FeedbackLLM(_reg_const(_FULL))  # rag 無し・state 新規(last_prediction None)
    res = await fb.run([_turn("user", "暑いね"), _turn("eve", "夏だね")])
    s = fb.state
    return (
        res is not None
        and s.last_prediction == "次は花火の話に移る"  # next_prediction が繰越された
        and s.surprise == NEUTRAL_SURPRISE  # cold: 35 ではなく中立
    )


async def t_fb_fep_close() -> bool:
    fb = FeedbackLLM(_reg_const("summary: x\nnext_prediction: 新しい予測\nsurprise: 40"))
    fb.state.last_prediction = "古い予測"  # not cold
    await fb.run([_turn("user", "a"), _turn("eve", "b")])
    return fb.state.last_prediction == "新しい予測" and fb.state.surprise == 40


async def t_fb_surprise_reactivity() -> bool:
    """予測が現実に現れれば低 surprise、現れなければ高 surprise（非装飾性の F4 版証明）。"""

    async def fake_fep(model, messages, **kw):
        user = messages[1]["content"]
        m = re.search(r"# 前回の予測\n(.+)", user)
        pred = m.group(1).strip() if m else ""
        convo = user.split("# 直近の会話")[-1]
        diff = 5 if pred and pred in convo else 90
        return {"choices": [{"message": {"content": f"summary: s\nnext_prediction: n\nsurprise: {diff}"}}]}

    reg = ModelRegistry(completion_fn=fake_fep)
    fb_match = FeedbackLLM(reg)
    fb_match.state.last_prediction = "天気の話"
    await fb_match.run([_turn("user", "今日は天気の話をしよう"), _turn("eve", "いいね")])
    low = fb_match.state.surprise

    fb_miss = FeedbackLLM(reg)
    fb_miss.state.last_prediction = "天気の話"
    await fb_miss.run([_turn("user", "ラーメン食べたい"), _turn("eve", "へえ")])
    high = fb_miss.state.surprise
    return low == 5 and high == 90 and high > low


async def t_fb_rag_write() -> bool:
    store = _store()
    text = "summary: 夏祭りの思い出話\nemotion: 楽しさ\nsurprise: 35\ntags: 夏, 祭り"
    fb = FeedbackLLM(_reg_const(text), rag_store=store)
    fb.state.last_prediction = "前の予測"  # not cold → surprise=35
    await fb.run([_turn("user", "夏の話"), _turn("eve", "夏祭り行きたい")])
    if len(store) != 1:
        return False
    rec = store._chunks[0]
    diff_ok = rec["prediction_diff"] == 35
    text_ok = "要約" in rec["text"] and "夏祭り" in rec["text"]
    res = await store.search("夏")  # 埋め込みターゲット=summary+tags＝「夏」でヒット
    search_ok = len(res) >= 1 and "夏祭り" in res[0].text
    return diff_ok and text_ok and search_ok and fb.state.surprise == 35


# ========== B5 worker（single-flight + watermark/span）==========
async def t_worker_nonblocking_trigger() -> bool:
    """trigger() は O(1) で即返り、処理は背景で進む。"""
    cache = _cache()
    fake = GatedFake()
    worker = FeedbackWorker(FeedbackLLM(ModelRegistry(completion_fn=fake)), cache)
    worker.start()
    cache.add_turn("user", "a")
    cache.add_turn("eve", "b")
    import time

    t0 = time.monotonic()
    worker.trigger()
    dt = time.monotonic() - t0
    await asyncio.sleep(0.05)
    await worker.stop()
    return dt < 0.01 and fake.calls >= 1


async def t_worker_single_flight() -> bool:
    cache = _cache()
    gate = asyncio.Event()
    fake = GatedFake(gate=gate)
    worker = FeedbackWorker(FeedbackLLM(ModelRegistry(completion_fn=fake)), cache)
    worker.start()
    cache.add_turn("user", "a")
    cache.add_turn("eve", "b")
    worker.trigger()
    await asyncio.sleep(0.02)  # 1件目が run に入り gate で保持される
    cache.add_turn("user", "c")
    cache.add_turn("eve", "d")
    worker.trigger()  # 2件目のトリガ（1件目 gate 中）
    await asyncio.sleep(0.02)
    gate.set()  # 解放
    await asyncio.sleep(0.05)
    await worker.stop()
    return fake.max_concurrent == 1  # 同時実行は常に1（single-flight）


async def t_worker_no_skip() -> bool:
    cache = _cache()
    gate = asyncio.Event()
    fake = GatedFake(gate=gate)
    fb = FeedbackLLM(ModelRegistry(completion_fn=fake))
    worker = FeedbackWorker(fb, cache)
    worker.start()
    cache.add_turn("user", "u1")
    cache.add_turn("eve", "e1")
    worker.trigger()
    await asyncio.sleep(0.02)  # span1 を処理中（gate 保持）
    # 処理中に3往復ぶん到着
    cache.add_turn("user", "u2")
    cache.add_turn("eve", "e2")
    cache.add_turn("user", "u3")
    cache.add_turn("eve", "e3")
    worker.trigger()
    gate.set()
    await asyncio.sleep(0.06)
    await worker.stop()
    leftover = cache.turns_since(fb.state.watermark)  # 未処理が残っていないこと
    seen_u3 = any("u3" in s for s in fake.seen_spans)  # 後続ターンもカバーされた
    return leftover == [] and seen_u3 and fb.state.watermark is not None


async def t_worker_surprise_not_frozen() -> bool:
    cache = _cache()
    fake = GatedFake()  # gate なし＝即返り、surprise=calls*10
    fb = FeedbackLLM(ModelRegistry(completion_fn=fake))
    fb.state.last_prediction = "seed"  # cold を避け diff を反映させる
    worker = FeedbackWorker(fb, cache)
    worker.start()
    seen: list[int] = []
    for i in range(3):
        cache.add_turn("user", f"u{i}")
        cache.add_turn("eve", f"e{i}")
        worker.trigger()
        await asyncio.sleep(0.03)
        seen.append(fb.state.surprise)
    await worker.stop()
    return len(set(seen)) > 1 and fb.state.watermark is not None  # 凍結しない


# ========== B6 orchestrator 配線 ==========
async def _tts_fn(s: str) -> bytes:
    return b"x"


async def t_orch_injects_last_feedback() -> bool:
    captured: dict = {}

    async def stream_fn(messages):
        captured["m"] = messages
        yield "はい。"

    audio = AudioPlayQueue(play_fn=_noop_play)
    state = PredictionState()
    state.last_feedback = "前回は天気の話で盛り上がった（予測差35 / 楽しさ）"
    orch = ResponseOrchestrator(audio, stream_fn, _tts_fn, prediction_state=state)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    user_msg = captured["m"][-1]["content"]
    return "# 直近フィードバック" in user_msg and "前回は天気の話" in user_msg


async def t_orch_injection_timing() -> bool:
    """feedback が build 前に完了していれば今の応答に、なければ次の応答に載る。"""
    seen: list[str] = []

    async def stream_fn(messages):
        seen.append(messages[-1]["content"])
        yield "ok。"

    audio = AudioPlayQueue(play_fn=_noop_play)
    state = PredictionState()
    orch = ResponseOrchestrator(audio, stream_fn, _tts_fn, prediction_state=state)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "1"))  # feedback まだ無い
    state.last_feedback = "新フィードバック"  # ターン間で feedback が完了したと仮定
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "2"))
    return "# 直近フィードバック" not in seen[0] and "新フィードバック" in seen[1]


async def t_orch_trigger_only_on_normal_completion() -> bool:
    n = {"c": 0}

    def on_complete() -> None:
        n["c"] += 1

    audio = AudioPlayQueue(play_fn=_noop_play)

    async def quick_stream(messages):
        yield "はい。"

    orch = ResponseOrchestrator(audio, quick_stream, _tts_fn, on_response_complete=on_complete)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    normal_ok = n["c"] == 1

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_stream(messages):
        started.set()
        await release.wait()
        yield "x。"

    orch2 = ResponseOrchestrator(audio, slow_stream, _tts_fn, on_response_complete=on_complete)
    task = asyncio.create_task(orch2.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hey")))
    await started.wait()
    task.cancel()  # barge-in
    try:
        await task
    except asyncio.CancelledError:
        pass
    return normal_ok and n["c"] == 1  # barge-in（CancelledError）はトリガしない


# ========== B7 起動 catch-up / shutdown 回収 ==========
async def t_startup_catchup() -> bool:
    """前回 feedback 途中で落ちた tail を起動時に取り戻す（記憶喪失なし）。"""
    cache = _cache()
    store = _store()
    # 前回までの feedback チャンク（古い timestamp = watermark の元）
    await store.add_chunk(
        text="要約: 古い話", summary="古い", search_text="夏",
        timestamp="2020-01-01T00:00:00+00:00",
    )
    # 復元会話: watermark(2020) より新しい未 feedback の tail
    cache.add_turn("user", "未処理の話題")
    cache.add_turn("eve", "そうだね")
    fb = FeedbackLLM(_reg_const("summary: 取り戻し\ntags: 夏"), rag_store=store, prediction_state=PredictionState())
    worker = FeedbackWorker(fb, cache)
    # 起動 catch-up 相当: watermark を RAG の最新 timestamp から復元 → 未処理あれば trigger
    fb.state.watermark = store.latest_timestamp()
    worker.start()
    if cache.turns_since(fb.state.watermark):
        worker.trigger()
    await asyncio.sleep(0.05)
    await worker.stop()
    advanced = fb.state.watermark != "2020-01-01T00:00:00+00:00"
    return advanced and cache.turns_since(fb.state.watermark) == [] and len(store) == 2


async def t_shutdown_recovery() -> bool:
    """進行中 feedback を stop で中断 → watermark 未前進 → 次回 catch-up 対象（喪失しない）。"""
    cache = _cache()
    store = _store()
    gate = asyncio.Event()  # 解放しない → run は保持され stop で cancel される
    fb = FeedbackLLM(ModelRegistry(completion_fn=GatedFake(gate=gate)), rag_store=store, prediction_state=PredictionState())
    worker = FeedbackWorker(fb, cache)
    worker.start()
    cache.add_turn("user", "x")
    cache.add_turn("eve", "y")
    worker.trigger()
    await asyncio.sleep(0.02)  # run に入り gate で保持
    await worker.stop(drain_timeout=0.05)  # drain 期限切れ → cancel
    leftover = cache.turns_since(fb.state.watermark)
    # 未完なので watermark 未前進・チャンク未保存・未処理が残る＝次回 catch-up で回収可能
    return fb.state.watermark is None and len(leftover) == 2 and len(store) == 0


async def main() -> None:
    # B1
    check("B1 PredictionState 既定値", t_predstate_defaults())
    check("B1 surprise はメソッド書込+クランプ", t_predstate_surprise_method())
    check("B1 surprise 壊れた値は中立", t_predstate_surprise_garbage())
    # B2
    check("B2 parser 完全タグ", t_parser_full())
    check("B2 parser 部分出力(diff=None)", t_parser_partial())
    check("B2 parser 全角コロン", t_parser_fullwidth_colon())
    check("B2 parser 非タグはゴミ(raise しない)", t_parser_garbage())
    check("B2 parser diff は語混じりでも整数抽出", t_parser_diff_clamp_and_robust())
    # B3
    check("B3 prompt 前回予測+会話を含む", t_prompt_includes_prediction_and_turns())
    check("B3 prompt cold センチネル", t_prompt_cold_sentinel())
    # B4
    check("B4 cold start(surprise 中立・next 繰越)", await t_fb_cold_start())
    check("B4 FEP ループ閉(last_prediction 更新)", await t_fb_fep_close())
    check("B4 surprise 反応性(一致 低/不一致 高)", await t_fb_surprise_reactivity())
    check("B4 RAG 書込(圧縮埋め込み/展開注入)", await t_fb_rag_write())
    # B5
    check("B5 trigger は非ブロッキング", await t_worker_nonblocking_trigger())
    check("B5 single-flight(同時実行=1)", await t_worker_single_flight())
    check("B5 no-skip(span カバー・記憶喪失なし)", await t_worker_no_skip())
    check("B5 surprise 凍結しない", await t_worker_surprise_not_frozen())
    # B6
    check("B6 last_feedback を文脈注入", await t_orch_injects_last_feedback())
    check("B6 注入タイミング(build 時点を読む)", await t_orch_injection_timing())
    check("B6 トリガは正常完了のみ(barge-in しない)", await t_orch_trigger_only_on_normal_completion())
    # B7
    check("B7 起動 catch-up(未処理 tail を取り戻す)", await t_startup_catchup())
    check("B7 shutdown 中断は watermark 未前進で回収可能", await t_shutdown_recovery())


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
