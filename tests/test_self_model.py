"""自己モデル（イブの好み・趣向）の決定論テスト（API 不要）。

実機事故 2026-08-15 の回帰:
- イブの「好きな食べ物」がチーズケーキ→ラーメン→りんご→おかか/ツナマヨ/生姜焼きと毎回
  変わった（嗜好を保持する層が無く、RAG top-3 の入れ替わりで即興していた）。
- ユーザの「俺はおかかが好き」がイブの自己申告に転写された（ロンダリング）。

ここで固定する不変条件（**すべてコード側**・プロンプトに委ねない）:
- 極性は閉じた2値。対象は自由。→「好きな食べ物」と「苦手な食べ物」が混ざらない
  （埋め込みでは分離不能と実測済み: 好き/苦手=0.931 > 好き/食材=0.928）
- 対象の総数に上限（LRU で追い出す）。→ **言い換えで突破されても注入ブロックは有界**
- 1項目最大2件。3件目で最も古い値が落ちる
- 注入ブロックは `# 参照`(RAG) とは別枠で、「今の会話に絡める」指示を持たない

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_self_model.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)  # fail-safe テストが意図的に出す例外ログを黙らせる

from eve.clock import Stamp  # noqa: E402
from eve.context_assembler import ContextAssembler, Turn  # noqa: E402
from eve.memory.self_model import (  # noqa: E402
    MAX_PER_SLOT,
    MAX_TOPICS,
    MAX_VALUE_CHARS,
    POLARITIES,
    SelfModelStore,
    normalize_topic,
)
from eve.self_model.trigger import is_trigger_text, should_examine  # noqa: E402
from eve.self_model.worker import SelfModelWorker, parse_ops, render_eve_turns  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def _tmp() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="eve_self_"), "self_model.jsonl")


def _turn(speaker: str, text: str, iso: str) -> Turn:
    return Turn(speaker, text, Stamp(iso=iso, mono=0.0))


def _async(coro):
    return asyncio.run(coro)


# ==== 極性は閉じ、対象は開く ====
s = SelfModelStore()
check("好き/苦手は受け付ける", s.set("好き", "食べ物", "おかかおにぎり") and s.set("苦手", "食べ物", "セロリ"))
check("**好きと苦手は別物として並ぶ**（埋め込みでは分離できない組）",
      s.values("好き", "食べ物") == ["おかかおにぎり"] and s.values("苦手", "食べ物") == ["セロリ"])
check("極性は閉じている（勝手な極性は拒否）", not s.set("大好き", "食べ物", "プリン"))
check("対象は自由に作れる（ポケモンでもゲームでも）",
      s.set("好き", "ポケモン", "フシデ") and s.set("好き", "ゲーム", "テトリス"))
check("空値は拒否", not s.set("好き", "食べ物", "   "))
check("空の対象は拒否", not s.set("好き", "  ", "プリン"))
check("長すぎる値は拒否", not s.set("好き", "食べ物", "あ" * (MAX_VALUE_CHARS + 1)))
check("長すぎる対象は拒否", not s.set("好き", "あ" * 20, "プリン"))

# 対象名の表記ゆれは潰す（言い換えではなく**同じ語の揺れ**だけ）
check("『好きな食べ物』と書かれても対象は『食べ物』", normalize_topic("好きな食べ物") == "食べ物")
check("全角・空白の揺れを潰す", normalize_topic("　食べ物 ") == "食べ物")
check("『苦手な』接頭辞も落ちる", normalize_topic("苦手な食べ物") == "食べ物")
s_n = SelfModelStore()
s_n.set("好き", "好きな食べ物", "プリン")
check("表記ゆれで別項目にならない", s_n.values("好き", "食べ物") == ["プリン"] and s_n.topics == 1)

# ==== 1項目2件 ====
s2 = SelfModelStore()
s2.set("好き", "食べ物", "梅おにぎり", iso="2026-06-22T00:00:00+00:00")
s2.set("好き", "食べ物", "鮭おにぎり", iso="2026-06-22T00:01:00+00:00")
check("2件までは並ぶ", s2.values("好き", "食べ物") == ["梅おにぎり", "鮭おにぎり"])
s2.set("好き", "食べ物", "おかかおにぎり", iso="2026-08-14T06:08:00+00:00")
check("3件目で最も古いものが落ちる",
      s2.values("好き", "食べ物") == ["鮭おにぎり", "おかかおにぎり"], f"{s2.values('好き', '食べ物')}")
check("上限は MAX_PER_SLOT と一致", len(s2.values("好き", "食べ物")) == MAX_PER_SLOT)
s2.set("好き", "食べ物", "おかかおにぎり", iso="2026-08-15T02:45:00+00:00")
check("同じ値の再表明では増えない", len(s2.values("好き", "食べ物")) == 2)

# ==== 対象の総数上限（言い換えで突破されても有界）====
s3 = SelfModelStore()
for i in range(MAX_TOPICS + 4):
    s3.set("好き", f"分野{i:02d}", f"値{i}", iso=f"2026-08-15T00:{i:02d}:00+00:00")
check("対象は上限を超えない", s3.topics == MAX_TOPICS, f"{s3.topics}個")
check("最も古い項目から落ちる（LRU）", "好きな分野00" not in s3.as_dict())
check("新しい項目は残る", "好きな分野15" in s3.as_dict(), f"{list(s3.as_dict())}")

# ユーザ懸念そのもの: 言い換えで枠を増やそうとしても、総量は増えない
s4 = SelfModelStore()
for i, alias in enumerate(["食べ物", "食材", "料理", "好物", "グルメ"]):
    s4.set("好き", alias, f"品目{i}", iso=f"2026-08-15T00:{i:02d}:00+00:00")
check("言い換えは別項目になってしまう（自動判別は不能・既知の限界）", s4.topics == 5)
check("それでも総数上限は破れない（被害は有界）", s4.topics <= MAX_TOPICS)

# ==== 削除 ====
s5 = SelfModelStore()
s5.set("苦手", "食べ物", "セロリ")
check("消せる", s5.remove("苦手", "食べ物", "セロリ"))
check("空になった項目ごと消える", s5.values("苦手", "食べ物") == [] and s5.topics == 0)
check("無い値の削除は False", not s5.remove("苦手", "食べ物", "セロリ"))

# ==== 注入ブロック ====
s6 = SelfModelStore()
check("空なら見出しごと出さない", s6.render() == "")
s6.set("好き", "食べ物", "おかかおにぎり")
s6.set("好き", "食べ物", "ツナマヨおにぎり")
s6.set("苦手", "食べ物", "セロリ")
s6.set("好き", "色", "藍色")
block = s6.render()
check("値が入る", "おかかおにぎり" in block and "ツナマヨおにぎり" in block and "藍色" in block)
check("好きと苦手が別行で出る", "好きな食べ物" in block and "苦手な食べ物" in block)
check("2件並びの意味を説明する", "どちらも好きで迷う" in block)
check("無い項目は決めつけさせない", "まだ決まっていない" in block)
check("そのまま読み上げさせない", "そのまま読み上げない" in block)
# **RAG と同じ引力を持たせない**のが要点（02:41 の格上げ事故の再発防止）
check("「今の会話に絡める」指示を持たない", "絡める" not in block)

# 最大まで埋めてもブロックが暴れない（注入量の上界を実測で固定）
s7 = SelfModelStore()
for i in range(MAX_TOPICS):
    s7.set("好き", f"分野{i:02d}", "あ" * MAX_VALUE_CHARS, iso=f"2026-08-15T00:{i:02d}:00+00:00")
    s7.set("好き", f"分野{i:02d}", "い" * MAX_VALUE_CHARS, iso=f"2026-08-15T01:{i:02d}:00+00:00")
check("満杯でもブロックは有界（1000字未満）", len(s7.render()) < 1000, f"{len(s7.render())}字")

# ==== ContextAssembler への注入 ====
ctx = ContextAssembler(system_prompt="スタイル")
sysmsg = ctx.assemble(user_text="好きな食べ物なに？", self_model=block)[0]["content"]
check("system に入る", "おかかおにぎり" in sysmsg)
check("参照ブロックとは別枠", "# イブ自身のこと" in sysmsg)
check("未指定なら何も足さない",
      "イブ自身のこと" not in ctx.assemble(user_text="やあ")[0]["content"])

# ==== 永続化（追記＋再生）====
p = _tmp()
a = SelfModelStore(p)
a.set("好き", "食べ物", "おかかおにぎり", iso="2026-08-14T06:08:00+00:00")
a.set("好き", "食べ物", "ツナマヨおにぎり", iso="2026-08-14T06:09:00+00:00")
a.set("好き", "色", "藍色", iso="2026-08-14T06:10:00+00:00")
a.remove("好き", "色", "藍色")
b = SelfModelStore(p)
check("再起動で復元される", b.values("好き", "食べ物") == ["おかかおにぎり", "ツナマヨおにぎり"],
      f"{b.values('好き', '食べ物')}")
check("削除も再生される", b.values("好き", "色") == [])

p2 = _tmp()
c = SelfModelStore(p2)
for i, v in enumerate(["梅", "鮭", "おかか", "ツナマヨ"]):
    c.set("好き", "食べ物", v, iso=f"2026-06-2{i}T00:00:00+00:00")
check("復元後も1項目2件", len(SelfModelStore(p2).values("好き", "食べ物")) == MAX_PER_SLOT)

p3 = _tmp()
c3 = SelfModelStore(p3)
for i in range(MAX_TOPICS + 4):
    c3.set("好き", f"分野{i:02d}", f"値{i}", iso=f"2026-08-15T00:{i:02d}:00+00:00")
check("復元後も対象の上限を守る", SelfModelStore(p3).topics == MAX_TOPICS)

with open(p2, "a", encoding="utf-8") as f:
    f.write("これはJSONではない\n")
check("壊れた行を飛ばして復元できる",
      len(SelfModelStore(p2).values("好き", "食べ物")) == MAX_PER_SLOT)

check("極性は2値だけ", len(POLARITIES) == 2)


# ==== トリガ（決定論のふるい）====
check("合図の語で発火", is_trigger_text("イブは梅と鮭が好きだよ"))
# 実データで取りこぼしていた2件（今回の調査の発端）。「迷」を足して拾えるようになった。
check("『迷う』も拾う(:515 の回帰)",
      is_trigger_text("コンビニおにぎりだとイブはおかかかツナマヨでちょっと迷っちゃうんだよね。"))
check("『迷う』も拾う(02:45 の回帰)",
      is_trigger_text("固定するなら、今のところは「おにぎりならおかかとツナマヨで迷う」って覚え方が一番近いかも。"))
check("無関係な発話では発火しない", not is_trigger_text("画面にログが流れてるね。"))

_user_only = [_turn("user", "俺はおかかが好き。", "2026-08-15T00:00:00+00:00")]
_eve_too = _user_only + [_turn("eve", "そうなんだ。イブは梅が好きかな。", "2026-08-15T00:00:01+00:00")]
check("ユーザ発話だけでは起動しない（転写の遮断）", not should_examine(_user_only))
check("イブ自身の発話があれば起動する", should_examine(_eve_too))
check("LLM にはイブの発話だけを渡す（ユーザ発話は渡さない）",
      "俺はおかかが好き" not in render_eve_turns(_eve_too)
      and "イブは梅が好きかな" in render_eve_turns(_eve_too))

# ==== 出力パーサ ====
_ops = parse_ops("set: 好き / 食べ物 = おかかおにぎり\nremove: 好き / 色 = 藍色\nよくわからない行\n")
check("set/remove を拾う", _ops == [("set", "好き", "食べ物", "おかかおにぎり"),
                                    ("remove", "好き", "色", "藍色")], f"{_ops}")
check("全角の区切りも許す",
      parse_ops("set：好き／色＝藍色") == [("set", "好き", "色", "藍色")])
check("変更なし（空出力）は空リスト", parse_ops("") == [] and parse_ops(None) == [])


# ==== worker（LLM は fake）====

class _FakeModel:
    def __init__(self, content="", raise_exc=None):
        self.content = content
        self.calls = 0
        self.seen = None
        self._raise = raise_exc

    async def complete(self, role, messages, **kw):
        self.calls += 1
        self.seen = messages
        if self._raise:
            raise self._raise
        return {"choices": [{"message": {"content": self.content}}]}


class _FakeCache:
    def __init__(self, turns):
        self._turns = turns

    def turns_since(self, watermark):
        return [t for t in self._turns if t.stamp.iso > (watermark or "")]


async def _run(model, turns, store):
    w = SelfModelWorker(model_registry=model, store=store, cache=_FakeCache(turns))
    w.start()
    w.trigger()
    await asyncio.sleep(0.05)
    await w.stop()


_m = _FakeModel("set: 好き / 食べ物 = ラーメン")
_st = SelfModelStore()
_async(_run(_m, [_turn("eve", "画面にログが流れてるね。", "2026-08-15T00:00:01+00:00")], _st))
check("合図が無ければ LLM を呼ばない", _m.calls == 0 and _st.as_dict() == {})

_m2 = _FakeModel("set: 好き / 食べ物 = おかかおにぎり")
_st2 = SelfModelStore()
_async(_run(_m2, [_turn("eve", "イブはおかかおにぎりが好きだよ。", "2026-08-15T00:00:01+00:00")], _st2))
check("合図があれば台帳に入る",
      _m2.calls == 1 and _st2.values("好き", "食べ物") == ["おかかおにぎり"])

_m3 = _FakeModel("set: 好き / 食べ物 = おかか")
_st3 = SelfModelStore()
_async(_run(_m3, [_turn("user", "俺はおかかが好き。", "2026-08-15T00:00:01+00:00")], _st3))
check("ユーザ発話だけでは LLM を呼ばない", _m3.calls == 0 and _st3.as_dict() == {})

# **LLM が規則を破っても store が止める**（不変条件はコード側）
_m4 = _FakeModel(
    "set: 大好き / 食べ物 = プリン\n"                      # 極性が閉じた2値の外
    "set: 好き / 食べ物 = A\nset: 好き / 食べ物 = B\nset: 好き / 食べ物 = C\n"  # 3件
)
_st4 = SelfModelStore()
_async(_run(_m4, [_turn("eve", "イブは好きなものが多いよ。", "2026-08-15T00:00:01+00:00")], _st4))
check("閉じていない極性は入らない（LLM が出しても）", "大好きな食べ物" not in _st4.as_dict())
check("1項目2件は LLM の出力に依らず守られる",
      _st4.values("好き", "食べ物") == ["B", "C"], f"{_st4.values('好き', '食べ物')}")

# LLM が言い換えで枠を量産しても、総数上限は破れない
_m5 = _FakeModel("\n".join(
    f"set: 好き / 分野{i:02d} = 値{i}" for i in range(MAX_TOPICS + 6)))
_st5 = SelfModelStore()
_async(_run(_m5, [_turn("eve", "イブは好きなものが多いよ。", "2026-08-15T00:00:01+00:00")], _st5))
check("LLM が項目を量産しても上限を超えない", _st5.topics == MAX_TOPICS, f"{_st5.topics}個")

_m6 = _FakeModel(raise_exc=RuntimeError("api down"))
_st6 = SelfModelStore()
_st6.set("好き", "色", "藍色")
_async(_run(_m6, [_turn("eve", "イブは藍色が好き。", "2026-08-15T00:00:01+00:00")], _st6))
check("LLM 例外でも台帳は無傷", _st6.values("好き", "色") == ["藍色"])

_m7 = _FakeModel("")
_st7 = SelfModelStore()
_st7.set("好き", "色", "藍色")
_async(_run(_m7, [_turn("eve", "イブは藍色が好き。", "2026-08-15T00:00:01+00:00")], _st7))
check("空出力（変更なし）でも台帳は無傷", _st7.values("好き", "色") == ["藍色"])

_m8 = _FakeModel("remove: 好き / 色 = 藍色")
_st8 = SelfModelStore()
_st8.set("好き", "色", "藍色")
_async(_run(_m8, [_turn("eve", "藍色はもう好きじゃないかも。", "2026-08-15T00:00:01+00:00")], _st8))
check("remove で消える", _st8.values("好き", "色") == [])

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
