"""S6 笑いの決定論テスト（API・音声デバイス不要）。

- yes の確率（logprobs）の取り出し
- 判定の閾値: 0.9 以上＝その場で笑う＋フィラーに笑い / 0.75 以上＝フィラーに笑い / それ未満＝何もしない。
  同じ文字は2回判定しない・判定の失敗で落ちない
- ストリーミング文字起こしの途中の文字: 「、」「。」で区切って区間番号つきで渡す・確定時にも1回
- フィラーの再生係: 話している途中に単独で笑う（ゲートを閉じて言い切り）/ 1発話1回 / チェーン中なら次の1つに挟む /
  笑いの声色 / 応答LLM に笑ったことを伝える / 応答の再生中は笑わない / 話し終わりと重なっても2回笑わない
- 配線: LAUGH_ENABLED・区間番号・途中の文字の口

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_laugh.py
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

from eve.filler import audio as fa  # noqa: E402
from eve.filler.bank import Clip, FillerBank, FillerTable  # noqa: E402
from eve.filler.laugh import LaughJudge, yes_probability  # noqa: E402
from eve.filler.player import FillerPlayer  # noqa: E402
from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402

_passed = 0
_failed = 0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 8000


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}  {detail}")


def tone(sec: float, f: float = 300.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# 確率の取り出し・判定の閾値
# ---------------------------------------------------------------------------

class _Tok:
    def __init__(self, token, logprob):
        self.token, self.logprob = token, logprob


class _Resp:
    def __init__(self, tops):
        c = type("C", (), {})()
        c.logprobs = type("L", (), {"content": [type("E", (), {"top_logprobs": tops})()]})()
        self.choices = [c]


def t_prob() -> None:
    r = _Resp([_Tok("yes", math.log(0.6)), _Tok("no", math.log(0.2)), _Tok("Yes", math.log(0.1))])
    check("確率: yes を大文字小文字まとめて足し、yes+no で割る", abs(yes_probability(r) - 0.7 / 0.9) < 1e-6)
    r2 = _Resp([{"token": "はい", "logprob": math.log(0.5)}, {"token": "いいえ", "logprob": math.log(0.5)}])
    check("確率: はい/いいえ・dict 形式も読む", abs(yes_probability(r2) - 0.5) < 1e-6)
    check("確率: 取れない応答は 0（笑わない）", yes_probability(object()) == 0.0)


async def t_judge() -> None:
    probs = {"とても面白い。": 0.95, "少し面白い、": 0.8, "普通だね。": 0.3, "壊れる。": None}
    calls: list = []
    imm: list = []
    fil: list = []

    async def judge(ctx, said):
        calls.append((tuple(ctx), said))
        p = probs[said]
        if p is None:
            raise RuntimeError("API 失敗")
        return p

    lj = LaughJudge(judge_fn=judge, context_fn=lambda: ["ユーザ: 前の話"],
                    on_immediate=lambda s, p: imm.append(s), on_filler=lambda s, p: fil.append(s),
                    immediate=0.9, filler=0.75)
    lj.on_phrase(1, "とても面白い。")
    lj.on_phrase(2, "少し面白い、")
    lj.on_phrase(3, "普通だね。")
    lj.on_phrase(4, "壊れる。")
    lj.on_phrase(1, "とても面白い。")  # 同じ区間の同じ文字
    await asyncio.sleep(0.05)
    check("閾値: 0.9 以上はその場で笑う", imm == [1], f"{imm}")
    check("閾値: 0.75 以上はフィラーに笑い（0.9 以上も含む）", sorted(fil) == [1, 2], f"{fil}")
    check("閾値: それ未満は何もしない", 3 not in imm and 3 not in fil)
    check("判定: 同じ区間の同じ文字は2回聞かない", sum(1 for c in calls if c[1] == "とても面白い。") == 1)
    check("判定: 直前の会話を渡す", calls and calls[0][0] == ("ユーザ: 前の話",))
    check("判定: 失敗しても落ちない（笑わないだけ）", 4 not in imm and 4 not in fil)
    await lj.stop()


# ---------------------------------------------------------------------------
# ストリーミング文字起こしの途中の文字
# ---------------------------------------------------------------------------

async def t_phrases() -> None:
    import base64

    from eve.stt.openai_live import OpenAILiveStt

    class Conn:
        def __init__(self):
            self.q: asyncio.Queue = asyncio.Queue()
            self.closed = False
            self.buf = 0

        async def send(self, obj):
            ty = obj.get("type")
            if ty == "session.update":
                self.q.put_nowait({"type": "session.updated", "session": {}})
            elif ty == "input_audio_buffer.append":
                self.buf += len(base64.b64decode(obj["audio"]))
            elif ty == "input_audio_buffer.commit":
                self.q.put_nowait({"type": "input_audio_buffer.committed", "item_id": "it1"})
                self.q.put_nowait({"type": "conversation.item.input_audio_transcription.completed",
                                   "item_id": "it1", "transcript": "えっとね、財布を探してた。見つかった！"})

        async def recv(self):
            return await self.q.get()

        async def close(self):
            self.closed = True
            self.q.put_nowait(None)

    conn = Conn()

    async def connect():
        return conn

    stt = OpenAILiveStt(api_key="x", connect_fn=connect, src_rate=16000)
    got: list = []
    stt.on_phrase = lambda tag, text: got.append((tag, text))
    pcm = (0.3 * 32767 * np.sin(2 * np.pi * 300 * np.arange(8000) / 16000)).astype("<i2").tobytes()
    stt.start(pcm, tag=7)
    await asyncio.sleep(0.05)
    for d in ["えっとね", "、", "財布を", "探してた", "。見つ"]:
        conn.q.put_nowait({"type": "conversation.item.input_audio_transcription.delta", "item_id": "it1",
                           "delta": d})
    await asyncio.sleep(0.05)
    check("途中の文字: 「、」「。」で区切って、その発話の頭からの文字を渡す",
          got == [(7, "えっとね、"), (7, "えっとね、財布を探してた。見つ")], f"{got}")
    stt.commit()
    text = await stt.transcribe(b"")
    await asyncio.sleep(0.02)
    check("途中の文字: 確定した時にも確定した文字で1回", got[-1] == (7, text) and text.startswith("えっとね"),
          f"{got[-1:]} / {text}")
    # 区間番号の無い呼び出し（ツールの一括経路）は途中の文字を出さない
    n = len(got)
    stt.start(pcm)  # tag=0
    conn.q.put_nowait({"type": "conversation.item.input_audio_transcription.delta", "item_id": "it2",
                       "delta": "はい。"})
    await asyncio.sleep(0.05)
    check("途中の文字: 区間番号が無ければ出さない", len(got) == n)
    stt.clear()
    await stt.close()


# ---------------------------------------------------------------------------
# フィラーの再生係の笑い
# ---------------------------------------------------------------------------

def mk_table(**kw) -> FillerTable:
    base = dict(weights={"a": 50.0, "aa": 30.0}, texts={"a": "あ", "aa": "あー", "w_fufu": "ふふ"},
                silence_weight=0.0, silence_sec=(0.02, 0.03), stretch_prob=0.0,
                stretchable={}, max_chain_sec=5.0, min_utterance_sec=0.6,
                join_gap_sec=0.01, laugh_ids=("w_fufu",))
    base.update(kw)
    return FillerTable(**base)


def mk_bank(t: FillerTable, *, laughs=True, tone_words=True) -> FillerBank:
    clips = {w: [Clip(w, t.texts[w], tone(0.06, 200 + 40 * k), SR) for k in range(3)] for w in t.weights}
    lw = {w: [Clip(w, t.texts[w], tone(0.09, 500), SR) for _ in range(2)] for w in t.weights} if tone_words else {}
    la = {"w_fufu": [Clip("w_fufu", "ふふ", tone(0.07, 700), SR)]} if laughs else {}
    return FillerBank(t, clips, {}, SR, laugh_words=lw, laughs=la)


class Rec:
    def __init__(self):
        self.events: list = []
        self.t0 = time.monotonic()

    async def play(self, wav, should_stop=None):
        x, sr = fa.read_wav(wav)
        self.events.append(("start", len(x) / sr, time.monotonic() - self.t0))
        await asyncio.sleep(len(x) / sr)
        self.events.append(("end", len(x) / sr, time.monotonic() - self.t0))


def mk_player(t=None, **kw):
    t = t or mk_table()
    rec = Rec()
    audio = AudioPlayQueue(play_fn=rec.play)
    texts: list = []
    audio.on_sentence_start = lambda text, meta: texts.append(text)
    pl = FillerPlayer(mk_bank(t, **kw), audio=audio, play_fn=rec.play, is_busy=lambda: False,
                      rng=random.Random(3))
    audio.on_head_waiting = pl.on_head_ready
    return pl, audio, rec, texts


async def t_lone_laugh() -> None:
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.request_laugh(5)  # 話している途中（区間5はまだ終わっていない）
    check("笑い: 話している途中でもその場で笑う（ゲートを閉じる）", audio.floor_held() and pl.active)
    await asyncio.sleep(0.12)
    check("笑い: 笑いの語を言い切って鳴らす", texts == ["ふふ"], f"{texts}")
    check("笑い: 鳴らし終えたらゲートを開ける", not audio.floor_held() and not pl.active)
    pl.request_laugh(5)
    await asyncio.sleep(0.1)
    check("笑い: 1発話1回まで", texts == ["ふふ"], f"{texts}")
    # 話し終わると、笑ったことも応答LLM に伝える（短い発話でフィラーが無くても）
    pl.on_speech_end(0.5, 5)
    spoken = pl.take_spoken()
    check("笑い: 話している途中に笑った語も応答LLM に伝える", spoken == "ふふ", f"{spoken}")
    await pl.stop()


async def t_laugh_in_chain() -> None:
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.on_speech_end(1.5, 8)
    await asyncio.sleep(0.03)  # 1語目の途中
    pl.request_laugh(8)
    await asyncio.sleep(0.2)
    check("笑い: その発話のフィラーが鳴っていれば次の1つとして挟む",
          "ふふ" in texts and texts.index("ふふ") >= 1, f"{texts}")
    pl.on_response_near()
    await asyncio.sleep(0.15)
    await pl.stop()
    # 話し終わりと「その場で笑う」がほぼ同時（単独で笑う前に話し終わった）→ チェーンの頭で1回だけ
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.request_laugh(9)
    pl.on_speech_end(1.5, 9)  # 常駐タスクが単独の笑いを鳴らす前
    await asyncio.sleep(0.15)
    pl.on_response_near()
    await asyncio.sleep(0.15)
    check("笑い: 話し終わりと重なっても2回笑わない（チェーンの頭で1回）",
          texts.count("ふふ") == 1 and texts[0] == "ふふ", f"{texts}")
    await pl.stop()


async def t_laugh_in_filler() -> None:
    """確率 ≥ filler: フィラーの中で普通に1回笑う（ユーザ 2026-09-26「ただ普通に笑うほうがよさそう」）。"""
    # 話している途中に判定が来た → 話し終わりのフィラーの頭で笑う
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.laugh_in_filler(4)
    await asyncio.sleep(0.05)
    check("フィラーに笑い: 話している途中には鳴らさない（その場で笑うのは 0.9 以上だけ）",
          not texts and not audio.floor_held(), f"{texts}")
    pl.on_speech_end(1.5, 4)
    await asyncio.sleep(0.2)
    check("フィラーに笑い: 話し終わりのフィラーの頭で笑う", texts[:1] == ["ふふ"], f"{texts}")
    check("フィラーに笑い: 笑いは1回", texts.count("ふふ") == 1, f"{texts}")
    durs = [round(e[1], 2) for e in rec.events if e[0] == "start"][1:]
    check("フィラーに笑い: 続くフィラーは普通の声（声色は使わない）", durs and all(d < 0.09 for d in durs), f"{durs}")
    spoken = pl.take_spoken() or ""
    check("フィラーに笑い: 笑ったことを応答LLM に伝える（語として）",
          spoken.startswith("ふふ") and "（笑いながら）" not in spoken, f"{spoken}")
    pl.on_response_near()
    await asyncio.sleep(0.15)
    await pl.stop()
    # フィラーが鳴っている途中に判定が来た → 次の1つとして挟む
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.on_speech_end(1.5, 5)
    await asyncio.sleep(0.03)  # 1語目の途中
    pl.laugh_in_filler(5)
    await asyncio.sleep(0.2)
    check("フィラーに笑い: 鳴っているフィラーの次の1つとして挟む",
          "ふふ" in texts and texts.index("ふふ") >= 1 and texts.count("ふふ") == 1, f"{texts}")
    pl.request_laugh(5)  # 後から 0.9 以上が来ても
    await asyncio.sleep(0.1)
    check("フィラーに笑い: その後に「その場で笑う」が来ても2回笑わない", texts.count("ふふ") == 1, f"{texts}")
    pl.on_response_near()
    await asyncio.sleep(0.15)
    await pl.stop()
    # フィラーの予約の後に 0.9 以上 → その場で笑う方に回す（1回だけ）
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.laugh_in_filler(6)
    pl.request_laugh(6)
    await asyncio.sleep(0.12)
    check("フィラーに笑い: 後から 0.9 以上が来たら、その場で笑う", texts == ["ふふ"], f"{texts}")
    pl.on_speech_end(1.5, 6)
    await asyncio.sleep(0.2)
    check("フィラーに笑い: そのあとのフィラーでは笑わない（1発話1回）", texts.count("ふふ") == 1, f"{texts}")
    pl.on_response_near()
    await asyncio.sleep(0.15)
    await pl.stop()
    # フィラーが鳴らない発話（短い）では笑わない
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.laugh_in_filler(7)
    pl.on_speech_end(0.4, 7)
    await asyncio.sleep(0.15)
    check("フィラーに笑い: フィラーが鳴らない発話では笑わない", not texts, f"{texts}")
    await pl.stop()


async def t_laughing_tone() -> None:
    """laugh.filler_style: tone（旧方式）の時は、フィラーの語を笑いの声色で鳴らす。"""
    pl, audio, rec, texts = mk_player(mk_table(laugh_filler_style="tone"))
    pl.start()
    pl.laugh_in_filler(4)
    pl.on_speech_end(1.5, 4)
    await asyncio.sleep(0.25)
    durs = [round(e[1], 2) for e in rec.events if e[0] == "start"]
    check("笑いの声色（tone）: フィラーの語を笑いの声色の方から鳴らす", durs and all(d >= 0.09 for d in durs), f"{durs}")
    check("笑いの声色（tone）: 笑いの語は挟まない", "ふふ" not in texts, f"{texts}")
    check("笑いの声色: 応答LLM に「笑いながら」と伝える", (pl.take_spoken() or "").endswith("（笑いながら）"))
    pl.on_response_near()
    await asyncio.sleep(0.15)
    await pl.stop()
    # 笑いの判定が無い発話は普通の声
    pl, audio, rec, texts = mk_player(mk_table(laugh_filler_style="tone"))
    pl.start()
    pl.on_speech_end(1.5, 6)
    await asyncio.sleep(0.2)
    durs = [round(e[1], 2) for e in rec.events if e[0] == "start"]
    check("笑いの声色: 判定が無い発話は普通の声", durs and all(d < 0.09 for d in durs), f"{durs}")
    check("笑いの声色: 普通の時は「笑いながら」を付けない", not (pl.take_spoken() or "").endswith("（笑いながら）"))
    pl.on_response_near()
    await asyncio.sleep(0.15)
    await pl.stop()


async def t_no_laugh_while_speaking() -> None:
    pl, audio, rec, texts = mk_player()
    pl.start()
    g = audio.current_generation()
    audio.enqueue(g, audio.reserve_seq(g), "応答の音声")  # 再生待ちの応答がある
    pl.request_laugh(3)
    await asyncio.sleep(0.05)
    check("笑い: 応答の再生中（待ち）は笑わない", not texts and not audio.floor_held())
    await pl.stop()
    pl, audio, rec, texts = mk_player(laughs=False)
    pl.start()
    pl.request_laugh(3)
    await asyncio.sleep(0.05)
    check("笑い: 笑いのバンクが無ければ何もしない", not texts and not audio.floor_held())
    await pl.stop()


def t_filler_style_yaml() -> None:
    import tempfile

    import yaml

    src = yaml.safe_load(open(os.path.join(REPO, "fillers.yaml"), encoding="utf-8"))
    check("表: 既定の笑い方は laugh（普通に笑う）", FillerTable.from_yaml(
        Path(os.path.join(REPO, "fillers.yaml"))).laugh_filler_style == "laugh")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "f.yaml"
        src["laugh"]["filler_style"] = "tone"
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        check("表: filler_style: tone を読む", FillerTable.from_yaml(f).laugh_filler_style == "tone")
        src["laugh"].pop("filler_style")
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        check("表: 書いていなければ laugh", FillerTable.from_yaml(f).laugh_filler_style == "laugh")
        src["laugh"]["filler_style"] = "koe"
        f.write_text(yaml.safe_dump(src, allow_unicode=True), encoding="utf-8")
        try:
            FillerTable.from_yaml(f)
            ok = False
        except ValueError:
            ok = True
        check("表: 知らない値は黙って読み替えずに止める（フィラーごと無効＋ログ）", ok)


# ---------------------------------------------------------------------------
# 配線
# ---------------------------------------------------------------------------

def t_wiring() -> None:
    from eve.config import Config
    from eve.voice_loop import VoiceLoop

    saved = (Config.LAUGH_ENABLED, Config.FILLER_ENABLED, Config.FILLER_VOICE, Config.EMOTION_ENABLED,
             Config.FILLER_MOOD_ENABLED)
    try:
        # 途中の文字は感情の判定（S5）・場面の判定（S8）にも配るので、ここではそれも切って「笑いだけ」の配線を見る
        Config.LAUGH_ENABLED, Config.EMOTION_ENABLED, Config.FILLER_MOOD_ENABLED = False, False, False
        vl = VoiceLoop()
        check("配線: LAUGH_ENABLED=0 なら笑いは無い", vl.laugh is None and getattr(vl.stt, "on_phrase", None) is None)
        have = os.path.isfile(os.path.join(REPO, "bank", "eve_calm_laugh", "manifest.json"))
        if not have:
            print("SKIP 本物の笑いのバンクが無い")
            return
        Config.LAUGH_ENABLED, Config.FILLER_ENABLED, Config.FILLER_VOICE = True, True, "eve_calm"
        vl = VoiceLoop()
        check("配線: LAUGH_ENABLED=1 で本物の笑いのバンクを読む", vl.laugh is not None
              and len(vl.filler._bank.laughs) == 7 and len(vl.filler._bank.laugh_words) == 10,
              f"{len(vl.filler._bank.laughs) if vl.filler else None}")
        got: list = []
        vl.laugh.on_phrase = lambda seg, text: got.append((seg, text))
        vl.stt.on_phrase(3, "えっとね、")
        check("配線: 途中の文字 → 笑い判定", vl.stt.on_phrase == vl._on_phrase and got == [(3, "えっとね、")],
              f"{got}")
        check("配線: 閾値は fillers.yaml（0.9 / 0.75）", vl.laugh.immediate == 0.9 and vl.laugh.filler == 0.75)
        check("配線: 0.75 以上の笑い方は fillers.yaml で「普通に笑う」（laugh）",
              vl.filler._t.laugh_filler_style == "laugh", vl.filler._t.laugh_filler_style)
        # 区間番号: 発話開始の確定ごとに +1、STT と発話終了の両方に同じ番号
        seen: list = []
        vl.input._on_speech_end = lambda sec, seg: seen.append(seg)
        starts: list = []
        vl.stt.start = lambda pcm, tag=0: starts.append(tag)
        vl.stt.commit = lambda: None
        vl.input._segment_start(b"\x00" * 3200)
        vl.input._segment_end(True, 1.0)
        vl.input._segment_start(b"\x00" * 3200)
        vl.input._segment_end(True, 1.0)
        check("配線: 区間番号が STT と発話終了の両方に同じ番号で渡る", starts == [1, 2] and seen == [1, 2],
              f"{starts} {seen}")
        Config.FILLER_ENABLED = False
        vl = VoiceLoop()
        check("配線: フィラーが無効なら笑いも無効（笑いはフィラーの仕組みで鳴らす）", vl.laugh is None)
    finally:
        (Config.LAUGH_ENABLED, Config.FILLER_ENABLED, Config.FILLER_VOICE, Config.EMOTION_ENABLED,
         Config.FILLER_MOOD_ENABLED) = saved


async def main() -> None:
    t_prob()
    # 本物の fillers.yaml は手元だけ（.gitignore・2026-10-06）。無い環境では本物の表を見るテストを飛ばす
    if os.path.isfile(os.path.join(REPO, "fillers.yaml")):
        t_filler_style_yaml()
    else:
        print("SKIP 本物の fillers.yaml が無い（手元だけの表）: 笑い方の表")
    for fn in (t_judge, t_phrases, t_lone_laugh, t_laugh_in_chain, t_laugh_in_filler, t_laughing_tone,
               t_no_laugh_while_speaking):
        try:
            await asyncio.wait_for(fn(), 15.0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} が例外なく終わる", False, f"{type(e).__name__}: {e}")
    t_wiring()


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
