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
from eve.response import (  # noqa: E402
    JapaneseSentenceSplitter,
    ResponseOrchestrator,
    TextInputSource,
    sanitize_for_speech,
)

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
    # 「！」「？」で終わった節は次の文字を見るまで出さない（続く記号をくっつけるため）
    check("分割: 境界ごとに emit（末尾の？は次の文字まで保留）", sp.feed("あ。い！う？") == ["あ。", "い！"])
    check("分割: 次の文字で保留を出す", sp.feed("まだ") == ["う？"])
    check("分割: flush 残り", sp.flush() == ["まだ"])

    sp2 = JapaneseSentenceSplitter()
    check("分割: 小数を割らない", sp2.feed("3.14は円。") == ["3.14は円。"])

    sp3 = JapaneseSentenceSplitter()
    check("分割: token 途中分割の結合", sp3.feed("こんにち") == [] and sp3.feed("は。") == ["こんにちは。"])

    sp4 = JapaneseSentenceSplitter()
    check("分割: 改行境界", sp4.feed("はい\n次") == ["はい"] and sp4.flush() == ["次"])

    # 「、」でも区切る（ユーザ裁定 2026-09-22「、。までは喋らせる」＝割り込まれたら節を言い切って譲る単位）
    sp5 = JapaneseSentenceSplitter()
    check("分割: 読点でも区切る", sp5.feed("うん、それでね、") == ["うん、", "それでね、"]
          and sp5.feed("行こう。") == ["行こう。"])
    sp6 = JapaneseSentenceSplitter()
    check("分割: 半角カンマは割らない（1,000）", sp6.feed("1,000円、安い。") == ["1,000円、", "安い。"])

    # 続く記号は前の節にくっつける（ユーザ 2026-09-29「！？単体だけ生成してもただの呼吸音…最後の文に記号をくっつけたい」）
    def split_all(*deltas: str) -> list[str]:
        s = JapaneseSentenceSplitter()
        out: list[str] = []
        for d in deltas:
            out += s.feed(d)
        return out + s.flush()

    check("記号: ！？ を前の節に", split_all("え、マジで！？それはすごいね。") == ["え、", "マジで！？", "それはすごいね。"])
    check("記号: ？！ を前の節に", split_all("本当？！うそでしょ。") == ["本当？！", "うそでしょ。"])
    check("記号: ！！ を前の節に", split_all("えっ！！すごい。") == ["えっ！！", "すごい。"])
    check("記号: 半角 !? も", split_all("まじ!?うそ。") == ["まじ!?", "うそ。"])
    check("記号: token をまたいでもくっつく",
          split_all("マジで！", "？", "それは") == ["マジで！？", "それは"])
    sp7 = JapaneseSentenceSplitter()
    check("記号: 次の文字が来るまで出さない", sp7.feed("マジで！") == [] and sp7.feed("？") == [])
    check("記号: 記号以外が来たら出す", sp7.feed("そ") == ["マジで！？"])
    check("記号: 最後の節は flush で出る", split_all("すごい！") == ["すごい！"])
    check("記号: 改行の後も前の節は1つ", split_all("すごい！\nでも") == ["すごい！", "でも"])
    check("記号: くっつける先の無い記号だけの節は出さない", split_all("そう。？ほんと。") == ["そう。", "ほんと。"])
    sp8 = JapaneseSentenceSplitter()
    check("記号: 「、」「。」はすぐ出す（割り込みの単位は今までどおり）",
          sp8.feed("うん、") == ["うん、"] and sp8.feed("行く。") == ["行く。"])


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


def t_sanitize() -> None:
    check("sanitize: 強調記号除去", sanitize_for_speech("**寿司**") == "寿司")
    check("sanitize: 番号マーカー除去", sanitize_for_speech("1. ピザ") == "ピザ")
    check("sanitize: 箇条書き- 除去", sanitize_for_speech("- りんご") == "りんご")
    check("sanitize: コード/見出し記号除去", sanitize_for_speech("`code`") == "code")
    check("sanitize: 通常文は不変", sanitize_for_speech("今日はいい天気。") == "今日はいい天気。")
    # 2026-09-30 実測（gpt-6-luna 1308応答）: 絵文字・特殊トークンがプロンプトの禁止をすり抜けた
    check("sanitize: 絵文字を落とす", sanitize_for_speech("どこの星から来たの？👾") == "どこの星から来たの？")
    check("sanitize: 異体字セレクタ付き絵文字も落とす", sanitize_for_speech("青く感じます。🌤️") == "青く感じます。")
    check("sanitize: 特殊トークンの漏れを落とす",
          sanitize_for_speech("確実だよ。<|END|>ごめんね") == "確実だよ。ごめんね"
          and sanitize_for_speech("/final_element空が") == "空が")
    check("sanitize: 英字や数字・ー・〜は残す", sanitize_for_speech("VTuberのRPGを3時〜5時にやろー")
          == "VTuberのRPGを3時〜5時にやろー")
    # 2026-10-01 実測（gpt-6-luna 2587文中 4文）: 日本語の声では読めない文字の混入。実際の出力をそのまま使う
    check("sanitize: キリル文字の混入を落とす",
          sanitize_for_speech("ефонがまだ動くなら、ひびに保護フィルムを貼ってね") == "がまだ動くなら、ひびに保護フィルムを貼ってね")
    check("sanitize: アルメニア文字の混入を落とす（それだけの文は空になる＝読み上げない）",
          sanitize_for_speech("元気いっぱいな感じがずるい。եցնում") == "元気いっぱいな感じがずるい。"
          and sanitize_for_speech("եցնում") == "")
    check("sanitize: グジャラート＋キリルの混入を落とす",
          sanitize_for_speech("でどう？રાજકોટмунасそのまま送るなら") == "でどう？そのまま送るなら")
    check("sanitize: ギリシャ文字・ハングル・アクセント付きラテン文字は残す",
          sanitize_for_speech("α波とβ版とπの話") == "α波とβ版とπの話"
          and sanitize_for_speech("수도권は韓国語だよ") == "수도권は韓国語だよ"
          and sanitize_for_speech("カフェはcaféって書くよ") == "カフェはcaféって書くよ")


async def t_orch_repeat_gate() -> bool:
    """応答の二重化の関門: 言い切った後に同じ答えをもう一度出し始めたら、以降は読み上げない（stream は最後まで読む）。

    実測 2026-09-30（gpt-6-luna・reasoning_effort=none・1308応答中 18件）の形そのまま:
    「…散らばるからです。これをレイリー散乱といいます。🌤️ /final_element空が青く見えるのは、…散らばるからです。」
    """
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())
    consumed = []

    async def fake_stream(messages):
        for c in ["えっと、ちゃんと説明するね。", "空が青く見えるのは、", "太陽の光が空気の分子にぶつかって散らばるからです。",
                  "これをレイリー散乱といいます。🌤️ /final_element", "空が青く見えるのは、",
                  "太陽の光が空気の分子にぶつかって散らばるからです。", "うん、", "うん、そうそう。"]:
            consumed.append(c)
            yield c

    async def fake_tts(s):
        return f"[wav:{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "なんで空は青いの？"))
    await audio.join()
    worker.cancel()
    return (orch.last_response == "えっと、ちゃんと説明するね。空が青く見えるのは、"
                                  "太陽の光が空気の分子にぶつかって散らばるからです。これをレイリー散乱といいます。"
            and len(consumed) == 8)  # stream は最後まで読む（tool_calls を取りこぼさない）


async def t_orch_short_repeats_kept() -> bool:
    """短い節（「うん」「そっか」等）の繰り返しや、別の内容の文は切らない（関門の誤爆防止）。"""
    audio = AudioPlayQueue(play_fn=None)

    async def fake_stream(messages):
        for c in ["うん、", "うん、", "そっか。", "明日は雨みたいだね。", "傘持っていってね。", "そっか。"]:
            yield c

    async def fake_tts(s):
        return f"[wav:{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "明日雨だって"))
    return orch.last_response == "うん、うん、そっか。明日は雨みたいだね。傘持っていってね。そっか。"


async def t_orch_markdown() -> bool:
    """LLM が出すマークダウン/箇条書きは読み上げ前に除去される。"""
    played: list = []

    async def play_fn(a):
        played.append(a)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())

    async def fake_stream(messages):
        for c in ["**A**。", "1. B。"]:
            yield c

    async def fake_tts(s):
        return f"[wav:{s}]"

    orch = ResponseOrchestrator(audio, fake_stream, fake_tts)
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    await audio.join()
    worker.cancel()
    return played == ["[wav:A。]", "[wav:B。]"] and orch.last_response == "A。B。"


def t_default_style() -> None:
    audio = AudioPlayQueue(play_fn=None)
    orch = ResponseOrchestrator(audio, None, None)
    msgs = orch._build_messages(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    sys_msg = next((m for m in msgs if m["role"] == "system"), None)
    check("既定で発話スタイル system が入る", sys_msg is not None and "マークダウン" in sys_msg["content"])


async def t_text_input() -> bool:
    q = StimulusQueue()
    src = TextInputSource(q)
    await src.submit("やあ")
    s = await q.get()
    return s.kind == StimulusKind.USER_UTTERANCE and s.payload == "やあ"


async def main() -> None:
    t_splitter()
    t_sanitize()
    check("配線: stream→split→seq順再生", await t_orch_order())
    check("再整列: 逆順完了でも seq 昇順", await t_orch_reorder())
    check("パイプライン: 1文目TTSがstream完了前", await t_orch_pipeline())
    check("barge-in: 以降の文を再生しない", await t_orch_bargein())
    check("マークダウン除去して読み上げ", await t_orch_markdown())
    check("⭐二重化の関門: 同じ答えの2回目は読み上げない・stream は最後まで読む", await t_orch_repeat_gate())
    check("二重化の関門: 短い節の繰り返し・別内容は切らない", await t_orch_short_repeats_kept())
    t_default_style()
    check("TextInputSource が USER_UTTERANCE を投入", await t_text_input())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
