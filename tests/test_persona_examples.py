"""話し方の実例（`eve/persona_examples.py`）の決定論テスト（API 不要・埋め込みは偽物）。

2026-10-01 ユーザ「N5＋実例で」「その場面になったら、ほぼその実際の会話データの丸写しにならないかよく確認」。
実測（`tools/prompt_eval/copy_test.py`・gpt-6-luna）: その実例を見せると、同じ話しかけで 33/100・言い換えで 16/100 が
丸写しになり、見せなければ 0/50。だから**今の話しかけと同じ場面の実例は見せない**ことを、ここで固定する。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_persona_examples.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.persona import load_persona  # noqa: E402
from eve.persona_examples import (  # noqa: E402
    HEADING,
    SIM_EXCLUDE,
    StyleExamples,
    load_examples,
    similar_line,
)

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


# テストのための作り文（実在の会話例やユーザの会話例をテストに写さない・2026-10-06 ユーザ「実例データは除外」）
EX = [
    {"id": "a", "user": "魚を釣って逃がす時の教訓ってある？", "eve": "逃がした魚のぶんだけ、次の釣りが楽しみになるんですよ。"},
    {"id": "b", "user": "夏と冬ならどっち？", "eve": "冬ですね。こたつから出なくていい理由が一年でいちばん多いので。"},
    {"id": "c", "user": "9たす10は？", "eve": "19です。たぶん。"},
    {"id": "d", "user": "イブ、秘密を一つ教えてくれない？", "eve": "教えたら秘密じゃなくなるので、また今度にします。"},
]


EXT = [{"id": e["id"], "turns": [("user", e["user"]), ("eve", e["eve"])]} for e in EX]


class FakeEmbedder:
    """話しかけ → ベクトル。同じ場面（言い換え）は同じ向き、別の場面は直交させる。"""

    TOPIC = {"釣": 0, "魚": 0, "夏": 1, "冬": 1, "たす": 2, "足す": 2, "秘密": 3}

    def __init__(self):
        self.calls = 0

    async def embed_query(self, text: str):
        self.calls += 1
        v = [0.0] * 5
        for w, i in self.TOPIC.items():
            if w in text:
                v[i] = 1.0
        if not any(v):
            v[4] = 1.0  # どれにも当たらない話しかけ
        return v


def run(coro):
    return asyncio.run(coro)


# ==== ファイルの読み込み ====
d = tempfile.mkdtemp(prefix="eve_examples_")
p = os.path.join(d, "ex.yaml")
with open(p, "w", encoding="utf-8") as f:
    f.write("examples:\n  - {id: x, user: こんにちは, eve: よっ}\n  - {user: '', eve: 空}\n  - 壊れた行\n")
got = load_examples(p)
check("YAML を読む（壊れた項目は飛ばす）", got == [{"id": "x", "turns": [("user", "こんにちは"), ("eve", "よっ")]}], f"{got}")
check("ファイルが無ければ空（実例なしで動く）", load_examples(os.path.join(d, "nothing.yaml")) == [])
with open(os.path.join(d, "bad.yaml"), "w", encoding="utf-8") as f:
    f.write("examples: [\n")
check("壊れたファイルでも落ちない", load_examples(os.path.join(d, "bad.yaml")) == [])
real = load_examples()
check("リポジトリの persona_examples.yaml が読める（人が選ぶまでは空でよい）", isinstance(real, list))

# ==== 選び方 ====
se = StyleExamples(EXT, k=2, rng=random.Random(1))
block, shown = run(se.pick("今日めっちゃ寒いね"))
check("k 組を選ぶ", len(shown) == 2, f"{shown}")
check("見出しが付く（測った置き方と同じ）", block.startswith(HEADING))
check("ユーザ／イブの形", block.count("ユーザ: ") == 2 and block.count("イブ: ") == 2, block)
seen = set()
for i in range(40):
    _, s = run(se.pick("今日めっちゃ寒いね"))
    seen.update(s)
check("毎回選び直す（同じ組を見せ続けない）", len(seen) == len(EX), f"{len(seen)}")
check("数が 0 なら使わない", run(StyleExamples(EXT, k=0).pick("x")) == ("", []))
check("実例が無ければ使わない", run(StyleExamples([], k=8).pick("x")) == ("", []))

# ==== 同じ場面の実例は見せない（丸写しの元）====
emb = FakeEmbedder()
se = StyleExamples(EXT, k=4, embedder=emb, rng=random.Random(2))
run(se.prepare())
check("起動時に実例の話しかけを埋め込む", se._vecs is not None and len(se._vecs) == len(EX))
for _ in range(20):
    _, s = run(se.pick("釣りで魚を逃がす時の学びって何？"))  # a の言い換え
    if EX[0]["eve"] in s:
        break
check("**言い換えでも、同じ場面の実例は見せない**（意味の近さ）", EX[0]["eve"] not in s and len(s) == 3, f"{s}")
_, s = run(se.pick("今日めっちゃ寒いね"))
check("関係ない話しかけでは全部から選ぶ", len(s) == 4, f"{s}")
check("しきい値は実測の値", SIM_EXCLUDE == 0.88)

# 埋め込みが無い／失敗した時は文字の似かたで代用（同じ話しかけは必ず外れる）
se = StyleExamples(EXT, k=4, rng=random.Random(3))
_, s = run(se.pick("9たす10は？"))
check("埋め込みが無くても、同じ話しかけの実例は見せない（文字の似かた）", EX[2]["eve"] not in s and len(s) == 3, f"{s}")
check("文字の似かた: 同じ話しかけは 1.0", similar_line("9たす10は？", "9たす10は？") == 1.0)
check("文字の似かた: 無関係は低い", similar_line("夏と冬ならどっち？", "イブ、秘密を一つ教えてくれない？") < 0.3)


class BrokenEmbedder(FakeEmbedder):
    async def embed_query(self, text):
        if text.startswith("壊"):
            raise RuntimeError("x")
        return await super().embed_query(text)


se = StyleExamples(EXT, k=4, embedder=BrokenEmbedder(), rng=random.Random(4))
run(se.prepare())
_, s = run(se.pick("壊れる前に、9たす10は？"))
check("話しかけの埋め込みに失敗しても落ちない（文字の似かたで続ける）", len(s) >= 3, f"{s}")

# ==== 複数ターンの実例（2026-10-04・ユーザ作の会話例は1話題に複数ターン）====
with open(os.path.join(d, "multi.yaml"), "w", encoding="utf-8") as f:
    f.write("examples:\n  - id: m\n    turns:\n      - user: 晩ごはん何がいい？\n      - eve: カレーです。\n"
            "      - user: 辛さは？\n      - eve: 中辛で。\n  - {id: s, user: 雨やんだね, eve: 傘いらないですね}\n")
multi = load_examples(os.path.join(d, "multi.yaml"))
check("複数ターンの実例を読む（1往復の形も混ぜてよい）",
      [len(e["turns"]) for e in multi] == [4, 2], f"{multi}")
se = StyleExamples(multi, k=8, rng=random.Random(6))
block, shown = run(se.pick("今日めっちゃ寒いね"))
check("複数ターンは1件として順番どおりに見せる",
      "ユーザ: 晩ごはん何がいい？\nイブ: カレーです。\nユーザ: 辛さは？\nイブ: 中辛で。" in block, block)
check("見せたイブの返事は全部（丸写しのログ用）", set(shown) == {"カレーです。", "中辛で。", "傘いらないですね"})
_, s = run(se.pick("辛さは？"))
check("途中のどの話しかけと同じでも、その会話ごと外す", "カレーです。" not in s and s == ["傘いらないですね"], f"{s}")
se = StyleExamples(multi, k=8, rng=random.Random(7), max_chars=40)
_, s = run(se.pick("今日めっちゃ寒いね"))
check("合計の文字数の上限を超えない（少なくとも1件は入る）", 1 <= len({x for x in s}) < 3, f"{s}")

# ==== 丸写しの疑い（ログ用）====
check("丸写しを見つける", StyleExamples.copy_suspect(
    EX[0]["eve"] + "だから気にしなくていいです。", [EX[0]["eve"]]) == EX[0]["eve"])
check("短い実例の半分以上そのままも見つける",
      StyleExamples.copy_suspect("19です。たぶん。……いや、20かも。", [EX[2]["eve"]]) == EX[2]["eve"])
check("話題が同じだけなら見つけない",
      StyleExamples.copy_suspect("冬派です。寒いのは苦手ですけど、鍋がおいしいので。", [EX[1]["eve"]]) is None)

# ==== system への入り方 ====
persona = load_persona()
ctx = ContextAssembler(system_prompt="STYLE\n\n" + persona)
sysmsg = ctx.assemble(user_text="x", style_examples=f"{HEADING}\nユーザ: a\nイブ: b")[0]["content"]
check("性格の直後・役割アンカーの前に入る",
      sysmsg.index(HEADING) > sysmsg.index("# あなた（イブ）の性格と口調")
      and sysmsg.index(HEADING) < sysmsg.index("assistant のメッセージ"))
check("渡さなければ入らない", HEADING not in ctx.assemble(user_text="x")[0]["content"])


# ==== 応答の組み立て（ユーザ発話への応答だけ・丸写しはログに残る）====
async def _orch_case():
    from eve.pipeline.audio_play_queue import AudioPlayQueue
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.orchestrator import ResponseOrchestrator
    from eve.speech.decider import AutonomousSpeech

    seen_sys: list[str] = []
    reply = {"text": EX[0]["eve"]}

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        seen_sys.append(messages[0]["content"])
        yield reply["text"]

    async def _play(audio):
        pass

    async def _tts(s):
        return b"x"

    audio = AudioPlayQueue(play_fn=_play)
    ex = StyleExamples(EXT, k=4, rng=random.Random(5))
    orch = ResponseOrchestrator(audio, stream_fn, _tts, context_assembler=ContextAssembler(system_prompt="S"),
                                style_examples=ex)
    w = asyncio.create_task(audio.play_worker())
    logging.disable(logging.NOTSET)
    records: list[str] = []

    class H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    lg = logging.getLogger("eve.response.orchestrator")
    h = H(level=logging.WARNING)
    lg.addHandler(h)
    try:
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "今日めっちゃ寒いね"))
        reply["text"] = "うん、そうだね。"
        await orch.handle(Stimulus(StimulusKind.AUTONOMOUS_SPEECH,
                                   AutonomousSpeech(content="そういえば昨日の話", reason="r")))
    finally:
        lg.removeHandler(h)
        logging.disable(logging.CRITICAL)
        w.cancel()
    return seen_sys, records


seen_sys, records = run(_orch_case())
check("ユーザ発話への応答には実例が入る", HEADING in seen_sys[0])
check("自発発話には入れない（測っていない経路）", HEADING not in seen_sys[1])
check("見せた実例をそのまま言ったらログに残る", any("そのまま言った疑い" in r for r in records), f"{records}")

# 性格の中身（手元の persona.local.md）はリポジトリのテストでは見ない（2026-10-06 ユーザ「詳細なペルソナは除外」）。
# 書式と読み込みは tests/test_persona.py が見る

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
