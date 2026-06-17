"""F2 応答背骨の決定論テスト（API 不要・純 stdlib）。

SentenceSplitter / ResponseOrchestrator 配線・パイプライン性・barge-in / TextInputSource。
実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f2_response.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.pipeline import AudioPlayQueue, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import JapaneseSentenceSplitter, ResponseOrchestrator, TextInputSource  # noqa: E402

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


# ---------- SentenceSplitter ----------
def t_splitter() -> None:
    sp = JapaneseSentenceSplitter()
    check("分割: 境界ごとに emit", sp.feed("あ。い！う？") == ["あ。", "い！", "う？"])
    check("分割: 境界まで buffer", sp.feed("まだ") == [])
    check("分割: flush 残り", sp.flush() == ["まだ"])

    sp2 = JapaneseSentenceSplitter()
    check("分割: 小数を割らない", sp2.feed("3.14は円。") == ["3.14は円。"])

    sp3 = JapaneseSentenceSplitter()
    check("分割: token 途中分割の結合", sp3.feed("こんにち") == [] and sp3.feed("は。") == ["こんにちは。"])

    sp4 = JapaneseSentenceSplitter()
    check("分割: 改行境界", sp4.feed("はい\n次") == ["はい"] and sp4.flush() == ["次"])


# ---------- ResponseOrchestrator ----------
async def t_orch_order() -> bool:
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())

    async def fake_stream(messages):
        for c in ["A。", "B。", "C。"]:
            yield c

    async def fake_tts(s):
        return f"[wav:{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    await audio.join()
    worker.cancel()
    return played == ["[wav:A。]", "[wav:B。]", "[wav:C。]"] and orch.last_response == "A。B。C。"


async def t_orch_reorder() -> bool:
    """TTS が順不同に完了しても再生は seq 昇順（AudioPlayQueue 再整列）。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())

    async def fake_stream(messages):
        for c in ["A。", "B。", "C。"]:
            yield c

    # 1文目を最も遅く完了させる（逆順完了）→ それでも再生は A,B,C
    delays = {"A。": 0.06, "B。": 0.03, "C。": 0.0}

    async def fake_tts(s):
        await asyncio.sleep(delays.get(s, 0.0))
        return f"[wav:{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    await audio.join()
    worker.cancel()
    return played == ["[wav:A。]", "[wav:B。]", "[wav:C。]"]


async def t_orch_pipeline() -> bool:
    """1文目の TTS は stream 完了前に始まる（バッチ待ちしない=即TTS）。"""
    events: list = []

    async def play_fn(a):
        events.append(("play", a))

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())

    async def fake_stream(messages):
        yield "A。"
        await asyncio.sleep(0.05)  # 後続 token はまだ来ていない
        yield "B。"
        await asyncio.sleep(0.05)
        yield "C。"
        events.append(("stream_end",))

    async def fake_tts(s):
        events.append(("tts", s))
        return f"[wav:{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    await audio.join()
    worker.cancel()
    return ("tts", "A。") in events and events.index(("tts", "A。")) < events.index(("stream_end",))


async def t_orch_bargein() -> bool:
    """stream 途中で世代が進むと、以降の文の音声は再生されない。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())

    async def fake_stream(messages):
        yield "A。"
        await asyncio.sleep(0.01)
        audio.bump_generation()  # 外部 barge-in を模す
        yield "B。"
        yield "C。"

    async def fake_tts(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    await audio.join()
    worker.cancel()
    return "[B。]" not in played and "[C。]" not in played


async def t_text_input() -> bool:
    q = StimulusQueue()
    src = TextInputSource(q)
    await src.submit("やあ")
    s = await q.get()
    return s.kind == StimulusKind.USER_UTTERANCE and s.payload == "やあ"


async def main() -> None:
    t_splitter()
    check("配線: stream→split→seq順再生", await t_orch_order())
    check("再整列: 逆順完了でも seq 昇順", await t_orch_reorder())
    check("パイプライン: 1文目TTSがstream完了前", await t_orch_pipeline())
    check("barge-in: 以降の文を再生しない", await t_orch_bargein())
    check("TextInputSource が USER_UTTERANCE を投入", await t_text_input())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
