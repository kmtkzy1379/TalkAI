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
    PredictionState,
    parse_feedback,
)
from eve.feedback.prompts import build_messages, build_user_text  # noqa: E402
from eve.memory import RagStore  # noqa: E402
from eve.memory.embed import Embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402

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


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
