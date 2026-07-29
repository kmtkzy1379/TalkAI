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

# ---------------------------------------------------------------------------
# 5. D1: セッション間ギャップ（前セッションの未応答依頼を勝手に実行しない）
# 実機 2026-07-29: 起動直後の挨拶だけで、5時間前の「PCの状態を教えてくれる?」を
# 「さっきの」と呼んで delegate_task した。
# ---------------------------------------------------------------------------
from eve.context_assembler import (  # noqa: E402
    OMITTED_SPEAKER, SESSION_GAP_SEC, session_gap_seconds,
)

D1_NOW = Stamp(iso="2026-07-28T17:21:11.146034+00:00", mono=0.0)


def _iso(seconds_ago: float) -> str:
    return (datetime.fromisoformat(D1_NOW.iso) - timedelta(seconds=seconds_ago)).isoformat()


def _turn(speaker: str, text: str, seconds_ago: float) -> Turn:
    return Turn(speaker, text, Stamp(iso=_iso(seconds_ago), mono=0.0))


_ca = ContextAssembler(system_prompt="SYS")
_GAP_HEAD = "# 会話の間隔"
_D1_FB = "ユーザは…その後PCの状態を教えてくれるかと発言した。（予測差45）"

# ⭐死活: 実機 D1 の入力（末尾=5時間前の未応答 user 依頼 / 現発話=挨拶）
d1 = _ca.assemble(
    user_text="おはよう。今日はちょっと寒いね。",
    recent_turns=[_turn("eve", "ごめん、ネット検索はできないんだ。", 17690),
                  _turn("user", "じゃあPCの状態を教えてくれる?", 17676)],
    last_feedback=_D1_FB, last_feedback_iso=_iso(17676), now=D1_NOW,
)
d1_sys = d1[0]["content"]
check("D1 ギャップブロックが system に出る", _GAP_HEAD in d1_sys)
check("D1 経過時間が載る（固定文でない）", "4時間" in d1_sys)
check("D1 依頼の失効を伝える", "まだ生きているとは限らない" in d1_sys)
check("D1 予約タスクに道を譲る", "「# 予約タスク」が正" in d1_sys)
check("D1 読み上げ抑止がある", "そのまま読み上げない" in d1_sys)
# ⭐死活: 逐語トークンを応答LLM に渡さない（J-2 ①-b の逐語見本禁止と同じ規律）
check("D1 語彙指定を渡さない(さっき/前に/今の)",
      not any(w in d1_sys.split(_GAP_HEAD)[1] for w in ("「さっき」", "「前に」", "「今の」")))
# ⭐死活: ロール汚染しない（注記が会話/ユーザ発話の本文に混ざらない）
check("D1 ユーザ発話は逐語のまま", d1[-1]["content"] == "おはよう。今日はちょっと寒いね。")
check("D1 会話ターンは無改変", d1[-2]["content"] == "じゃあPCの状態を教えてくれる?")
check("D1 注記は会話メッセージに入らない",
      all(_GAP_HEAD not in m["content"] and "間隔" not in m["content"] for m in d1[1:]))
# ⭐死活: # 直近フィードバック は「内省が対象にした会話スパン」で接地する
check("D1 古い内省に時刻ラベルが付く", "の会話についての内省" in d1_sys)
check("D1 古い内省は現在の依頼でないと明示", "今この場で出ている依頼ではない" in d1_sys)

# ⭐陽性対照: ギャップ無しでは何も出さない（「常に出す」実装を落とす）
fresh = _ca.assemble(
    user_text="おはよう。", recent_turns=[_turn("user", "ねえ", 30)],
    last_feedback=_D1_FB, last_feedback_iso=_iso(30), now=D1_NOW,
)
check("D1 陽性対照: ギャップ無しなら間隔ブロック無し", _GAP_HEAD not in fresh[0]["content"])
check("D1 陽性対照: 新しい内省にラベルを付けない", "内省" not in fresh[0]["content"])
check("D1 陽性対照: ユーザ発話は逐語", fresh[-1]["content"] == "おはよう。")

# 閾値の境界（`>` を pin）
check("D1 閾値ちょうどでは出さない",
      session_gap_seconds([_turn("user", "x", SESSION_GAP_SEC)], D1_NOW) is None)
check("D1 閾値超で出す",
      session_gap_seconds([_turn("user", "x", SESSION_GAP_SEC + 1)], D1_NOW) is not None)

# 測る基準は「最後の**ユーザ**発話」（直前ターンだとイブの自発発話で新しくなり取り逃す）
check("D1 直前がイブの自発発話でも取り逃さない",
      session_gap_seconds([_turn("user", "PCの状態教えて", 17676),
                           _turn("eve", "ひとりごと", 5)], D1_NOW) is not None)

# 省略マーカが挟まる窓は「空白」ではない（実ターンを省略しただけ）
check("D1 省略マーカ入りでは出さない",
      session_gap_seconds([_turn("user", "a", 17676),
                           Turn(OMITTED_SPEAKER, "3", Stamp.now()),
                           _turn("eve", "b", 5)], D1_NOW) is None)

# 履歴なし / 時刻破損は「無言で 0 扱い」にしない（ガードが自分で自分を切らない）
check("D1 履歴なしは出さない", session_gap_seconds([], D1_NOW) is None
      and session_gap_seconds(None, D1_NOW) is None)
check("D1 時刻破損では判定を降りる",
      session_gap_seconds([Turn("user", "x", Stamp(iso="not-a-date", mono=0.0))], D1_NOW) is None
      and session_gap_seconds([Turn("user", "x", Stamp(iso="", mono=0.0))], D1_NOW) is None)
check("D1 未来ターンでは出さない",
      session_gap_seconds([_turn("user", "x", -3600)], D1_NOW) is None)

# 自発発話経路には出さない（判定LLM は「古い話題を振ってよい」と逆を明示指示されている）
auto = _ca.assemble(
    autonomous_content="空が青い理由の話をする",
    recent_turns=[_turn("user", "じゃあPCの状態を教えてくれる?", 17676)], now=D1_NOW,
)
check("D1 自発経路には間隔ブロックを出さない", _GAP_HEAD not in auto[0]["content"])
check("D1 自発経路の指示文は壊れない", "下書き" in auto[-1]["content"])

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
