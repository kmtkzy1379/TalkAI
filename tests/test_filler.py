"""S3 応答非依存フィラーの決定論テスト（API・音声デバイス不要）。

`docs/FILLER_IMPL_PLAN.md` §5 S3 のテスト項目:
- 発話権ゲート: 閉じている間は応答を鳴らさず、開けた後は seq 順 / 待っている間の割り込みで古い文を鳴らさない /
  例外でも・割り込みでも・キャンセルでも**必ずゲートが開く**（開かないと全ターン無音）/ 安全弁 / フィラー中は is_quiet=False
- 畳み方: 応答の音声ができていれば言い終えて止まる / 「応答が近い」で次の語を始めない / 来なければ上限まで /
  割り込みは言い切って止まる（次の発話のフィラーには持ち越さない）/ 空認識でつながない
- 抽選の分布（数万回）: 無言19%・1段目の比率・伸ばし棒1%・直前と別の語・無言の長さ 0.3〜0.5・使えない枠の分は別の語へ
- 特殊フィラー（語＋っと・っとね をまとめて合成した組）: 独立したフィラーとして合計5%（系は 2:2:1.5:0.5 で按分・系の中は
  等分・無言と語は同じ比率で縮める）/ お・あの・その の組は無い / バンクに無い組は外して鳴らす / 1つの音として鳴らす
- 鳴らさない条件: 0.6秒以内の発話 / 応答の生成中 / 再生中 / ミュート / FILLER_ENABLED=0 / つなぎの間が未決
- 応答LLM への伝え方: 前置き＋system の規則（USER 発話だけ）・payload と会話履歴は変えない・1回だけ渡す
- 配線: 自発発話・画面実況のガード・ゲートの合図・本物の fillers.yaml とバンク

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_filler.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

from eve.filler import audio as fa  # noqa: E402
from eve.filler import player as player_mod  # noqa: E402
from eve.filler.bank import Clip, FillerBank, FillerTable, Piece, draw  # noqa: E402
from eve.filler.player import FillerPlayer  # noqa: E402
from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402
from eve.pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind  # noqa: E402

_passed = 0
_failed = 0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_TABLE = os.path.join(REPO, "fillers.yaml")


def _real_table() -> bool:
    """本物の fillers.yaml（手元だけ・.gitignore・2026-10-06）が無い環境では、本物の表を見るテストを飛ばす。"""
    if os.path.isfile(REAL_TABLE):
        return True
    print("SKIP 本物の fillers.yaml が無い（手元だけの表）")
    return False


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}  {detail}")


SR = 8000


def tone(sec: float, f: float = 300.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def mk_table(**kw) -> FillerTable:
    base = dict(
        weights={"a": 50.0, "aa": 20.0, "ano": 10.0}, texts={"a": "あ", "aa": "あー", "ano": "あの"},
        silence_weight=20.0, silence_sec=(0.02, 0.03), stretch_prob=0.01,
        stretchable={"a": "a", "aa": "a", "ano": "o"},
        max_chain_sec=0.6, min_utterance_sec=0.6, join_gap_sec=0.01, stretch_adopted=None)
    base.update(kw)
    return FillerTable(**base)


def mk_bank(table: FillerTable, word_sec: float = 0.06) -> FillerBank:
    clips = {}
    for wid in table.weights:
        clips[wid] = [Clip(wid, table.texts[wid], tone(word_sec, 200 + 50 * k), SR) for k in range(3)]
    return FillerBank(table, clips, {}, SR)


class Rec:
    """再生の記録（フィラーの語と応答の文の順序・時刻）。"""

    def __init__(self) -> None:
        self.events: list = []  # (種類, 文字, 時刻)
        self.t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def filler_play(self, wav: bytes, should_stop=None) -> None:
        x, sr = fa.read_wav(wav)
        self.events.append(("f_start", None, self.now()))
        await asyncio.sleep(len(x) / sr)
        self.events.append(("f_end", None, self.now()))

    async def queue_play(self, audio, should_stop=None) -> None:
        self.events.append(("r_start", audio, self.now()))
        await asyncio.sleep(0.02)
        self.events.append(("r_end", audio, self.now()))


def mk_player(table=None, *, busy=False, muted=False, rec=None, word_sec=0.06, rng_seed=1):
    table = table or mk_table()
    rec = rec or Rec()
    audio = AudioPlayQueue(play_fn=rec.queue_play)
    texts: list = []
    audio.on_sentence_start = lambda text, meta: texts.append((text, rec.now()))
    pl = FillerPlayer(mk_bank(table, word_sec), audio=audio, play_fn=rec.filler_play,
                      is_busy=lambda: busy, muted=lambda: muted, rng=random.Random(rng_seed))
    audio.on_head_waiting = pl.on_head_ready
    return pl, audio, rec, texts


# ---------------------------------------------------------------------------
# 発話権ゲート
# ---------------------------------------------------------------------------

async def t_gate() -> None:
    rec = Rec()
    q = AudioPlayQueue(play_fn=rec.queue_play)
    w = asyncio.create_task(q.play_worker())
    q.hold_floor()
    check("ゲート: フィラー中は is_quiet=False", not q.is_quiet())
    g = q.current_generation()
    s0, s1 = q.reserve_seq(g), q.reserve_seq(g)
    q.enqueue(g, s1, "B")
    q.enqueue(g, s0, "A")
    await asyncio.sleep(0.1)
    check("ゲート: 閉じている間は応答を鳴らさない", not rec.events)
    check("ゲート: 先頭の音声ができたことは分かる（head_ready）", q.head_ready())
    q.release_floor()
    await asyncio.sleep(0.15)
    played = [e[1] for e in rec.events if e[0] == "r_start"]
    check("ゲート: 開けた後は seq 順に鳴らす", played == ["A", "B"], f"{played}")
    check("ゲート: 開いていれば is_quiet=True", q.is_quiet())

    # 待っている間に割り込み → 古い世代は鳴らさない（取り出す前に待つ）
    rec.events.clear()
    q.hold_floor()
    g = q.current_generation()
    q.enqueue(g, q.reserve_seq(g), "OLD")
    await asyncio.sleep(0.05)
    q.interrupt()
    q.release_floor()
    await asyncio.sleep(0.1)
    check("ゲート: 待っている間に割り込まれた古い文は鳴らさない", not rec.events, f"{rec.events}")

    # 安全弁: 開かないまま上限を過ぎたら開けて鳴らす（全ターン無音にしない）
    rec.events.clear()
    q.floor_timeout_sec = 0.2
    q.hold_floor()
    g = q.current_generation()
    q.enqueue(g, q.reserve_seq(g), "LATE")
    await asyncio.sleep(0.4)
    check("ゲート: 開かないまま上限を過ぎたら開けて鳴らす（安全弁）",
          [e[1] for e in rec.events if e[0] == "r_start"] == ["LATE"])
    check("ゲート: 既定は開いている（フィラー無しでは挙動が変わらない）", AudioPlayQueue(play_fn=None).is_quiet())
    w.cancel()


# ---------------------------------------------------------------------------
# 抽選の分布
# ---------------------------------------------------------------------------

def t_lottery() -> None:
    import yaml

    if not _real_table():
        return
    real = FillerTable.from_yaml(os.path.join(REPO, "fillers.yaml"))
    N = 200_000
    rng = random.Random(42)
    c = Counter(draw(real, rng, first=True, last_word=None, last_kind=None, stretch_ok=True).word_id or "無言"
                for _ in range(N))
    total_w = sum(real.weights.values())
    check("抽選(本物の表): 無言 + 語 + 特殊フィラーの重み = 100", abs(total_w + real.silence_weight - 100.0) < 1e-6,
          f"{total_w + real.silence_weight}")
    # 特殊フィラー 5% の分だけ無言 20% を縮めた（ユーザ 2026-09-29「２は按分するから変わらない」）
    check("抽選: 1段目の無言 ≈ 19%", abs(c["無言"] / N - 0.19) < 0.005, f"{c['無言'] / N:.4f}")
    worst = max(abs(c[w] / N - real.weights[w] / 100.0) for w in real.weights)
    check("抽選: 1段目の語・特殊フィラーの比率 ≈ 表の重み（あ 36.37% 等・誤差 0.5% 以内）", worst < 0.005,
          f"最大ずれ {worst:.4f}")
    sp = sum(c[w] for w in real.special) / N
    check("抽選: 1段目の特殊フィラー ≈ 5%（独立したフィラー）", abs(sp - 0.05) < 0.003, f"{sp:.4f}")
    check("抽選: 1段目に伸ばし棒は出ない",
          all(draw(real, rng, first=True, last_word="a", last_kind="word", stretch_ok=True).kind in ("word", "silence")
              for _ in range(20000)))

    kinds = Counter()
    words = Counter()
    same = 0
    for _ in range(N):
        p = draw(real, rng, first=False, last_word="a", last_kind="word", stretch_ok=True)
        kinds[p.kind] += 1
        if p.kind == "word":
            words[p.word_id] += 1
            same += p.word_id == "a"
    check("抽選(2段目): 無言 ≈ 19%", abs(kinds["silence"] / N - 0.19) < 0.005, f"{kinds['silence'] / N:.4f}")
    check("抽選(2段目): 伸ばし棒 ≈ 無言以外の1%（全体の0.81%）", abs(kinds["stretch"] / N - 0.0081) < 0.0015,
          f"{kinds['stretch'] / N:.4f}")
    # 語の後ろに っと・っとね を付ける仕組み（旧 terminal_prob 0.2%）は廃止＝独立した特殊フィラーの組（2026-09-29）
    check("抽選(2段目): 出るのは 無言・語（組を含む）・伸ばし棒 だけ", set(kinds) <= {"silence", "word", "stretch"},
          f"{dict(kinds)}")
    check("抽選(2段目): 直前と同じ語は出ない", same == 0, f"{same}")
    rest = {w: real.weights[w] for w in real.weights if w != "a"}
    tot = sum(rest.values())
    nw = sum(words.values())
    worst = max(abs(words[w] / nw - rest[w] / tot) for w in rest)
    check("抽選(2段目): 別の語の比率は weight のまま（直前の語を除いて割り直し）", worst < 0.006, f"{worst:.4f}")
    after = [draw(real, rng, first=False, last_word="e+tto", last_kind="word", stretch_ok=True)
             for _ in range(50_000)]
    check("抽選(2段目): 特殊フィラーの組の後は、同じ組が続かず・伸ばし棒も付かない（組は伸ばせない）",
          not any(p.word_id == "e+tto" or p.kind == "stretch" for p in after))

    sil = Counter(draw(real, rng, first=False, last_word="a", last_kind="silence", stretch_ok=True).kind
                  for _ in range(N))
    check("抽選: 無言の後に 伸ばし棒 は付かない", sil["stretch"] == 0)
    nost = Counter(draw(real, rng, first=False, last_word="a", last_kind="word", stretch_ok=False).kind
                   for _ in range(N))
    check("抽選: 伸ばし棒が使えない時はその1%を別の語へ回す",
          nost["stretch"] == 0 and abs(nost["word"] / N - 0.81) < 0.005, f"{nost['word'] / N:.4f}")
    # 同じ語の繰り返しを少し減らす補正（ユーザ 2026-09-26・repeat_factor=0.5）
    check("表: 同じ語の補正は 0.5（すこし補正）", real.repeat_factor == 0.5)
    base = Counter(draw(real, rng, first=False, last_word="aa", last_kind="word", stretch_ok=False).word_id
                   for _ in range(N))
    used = Counter(draw(real, rng, first=False, last_word="aa", last_kind="word", stretch_ok=False,
                        used={"a": 1}).word_id for _ in range(N))
    ratio = (used["a"] / used["e"]) / (base["a"] / base["e"])
    check("抽選: そのチェーンで使った語は重みが半分になる（あ を1回使うと え に対する比が 0.5倍）",
          abs(ratio - 0.5) < 0.05, f"{ratio:.3f}")
    check("抽選: 1段目は補正しない（使った語が無い）",
          abs(Counter(draw(real, rng, first=True, last_word=None, last_kind=None, stretch_ok=True).word_id
                      for _ in range(N))["a"] / N - real.weights["a"] / 100) < 0.005)
    lens = [draw(real, rng, first=True, last_word=None, last_kind=None, stretch_ok=True) for _ in range(20000)]
    ls = [p.sec for p in lens if p.kind == "silence"]
    check("抽選: 無言の長さは 0.3〜0.5秒", ls and min(ls) >= 0.3 and max(ls) <= 0.5, f"{min(ls):.3f}-{max(ls):.3f}")
    check("抽選: 無言の長さはばらつく（ランダム）", ls and np.std(ls) > 0.04)
    cfg = yaml.safe_load(open(os.path.join(REPO, "fillers.yaml"), encoding="utf-8"))
    check("表: 語そのものは変えていない（応答非依存10語＋っと/っとね）",
          [e["text"] for e in cfg["entries"] if "independent" in e["role"]]
          == ["んー", "あー", "えー", "え", "あ", "う", "お", "うー", "あの", "その"]
          and [e["text"] for e in cfg["entries"] if "terminal" in e["role"]] == ["っと", "っとね"])
    # 上限はユーザ 2026-09-29「4.5秒に少し引き上げる」（元 4秒・2026-09-25）
    check("表: 上限4.5秒・最短発話0.6秒・無言0.3〜0.5", real.max_chain_sec == 4.5 and real.min_utterance_sec == 0.6
          and real.silence_sec == (0.3, 0.5))
    # ユーザ 2026-09-28「間を1.2秒にしてそもそも2回目ほぼ出させない」（っと・っとね は語とまとめて合成＝間の設定は無い）
    check("表: 語と語の間 1.2秒", real.join_gap_sec == 1.2, f"{real.join_gap_sec}")


# ---------------------------------------------------------------------------
# フィラーの再生と畳み方
# ---------------------------------------------------------------------------

async def _response(audio: AudioPlayQueue, texts=("R1", "R2")) -> None:
    g = audio.current_generation()
    seqs = [audio.reserve_seq(g) for _ in texts]
    for s, t in zip(seqs, texts):
        audio.enqueue(g, s, t)


async def t_no_filler() -> None:
    pl, audio, rec, texts = mk_player()
    pl.start()
    pl.on_speech_end(0.6)
    await asyncio.sleep(0.05)
    check("鳴らさない: 0.6秒以内の発話（うん・へー）", not audio.floor_held() and not texts)
    await pl.stop()
    for kw, name in ((dict(busy=True), "応答の生成中"), (dict(muted=True), "ミュート中")):
        pl, audio, rec, texts = mk_player(**kw)
        pl.start()
        pl.on_speech_end(1.2)
        await asyncio.sleep(0.05)
        check(f"鳴らさない: {name}", not audio.floor_held() and not texts)
        await pl.stop()
    pl, audio, rec, texts = mk_player()
    pl.start()
    await _response(audio, ("R",))  # 再生待ちの応答がある
    pl.on_speech_end(1.2)
    check("鳴らさない: 応答の再生待ちがある", not audio.floor_held())
    await pl.stop()
    pl, audio, rec, texts = mk_player(mk_table(join_gap_sec=None))
    pl.start()
    pl.on_speech_end(1.2)
    check("鳴らさない: つなぎの間が未決（推測で決めない）", not audio.floor_held())
    await pl.stop()


async def t_fold_on_response() -> None:
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0)
    pl, audio, rec, texts = mk_player(t)
    w = asyncio.create_task(audio.play_worker())
    pl.start()
    pl.on_speech_end(1.5)
    check("再生: 発話終了の瞬間にゲートを閉じる（同期）", audio.floor_held())
    await asyncio.sleep(0.15)  # 数語ぶん鳴らす
    await _response(audio)
    await asyncio.sleep(0.4)
    f_ends = [e[2] for e in rec.events if e[0] == "f_end"]
    r_starts = [e[2] for e in rec.events if e[0] == "r_start"]
    check("畳む: 応答の音声ができたら言い終えて止まり、応答が鳴る", f_ends and r_starts, f"{rec.events[-4:]}")
    f_starts = [e[2] for e in rec.events if e[0] == "f_start"]
    spans = list(zip(f_starts, f_ends))
    check("畳む: 応答はフィラーの語が言い終わってから（どの語の途中にも重ならない）",
          r_starts and len(spans) == len(f_starts)
          and not any(a < r_starts[0] < b - 1e-3 for a, b in spans), f"{spans} r={r_starts[:1]}")
    check("口パク: フィラーの語も UI の再生開始通知を出す（音量から作った計画つき）",
          any(txt in ("あ", "あー", "あの") for txt, _ in texts))
    check("畳む: 応答が鳴り始めた後にフィラーの新しい語は始まらない", all(s < r_starts[0] for s in f_starts))
    check("畳む: 応答が鳴った後はゲートが開いている", not audio.floor_held() and not pl.active)
    played = [e[1] for e in rec.events if e[0] == "r_start"]
    check("畳む: 応答は seq 順", played == ["R1", "R2"], f"{played}")
    await pl.stop()
    w.cancel()


async def t_fold_on_near() -> None:
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0)
    pl, audio, rec, texts = mk_player(t, word_sec=0.08)
    w = asyncio.create_task(audio.play_worker())
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.12)  # 2語目の途中
    t_near = rec.now()
    pl.on_response_near()
    await asyncio.sleep(0.3)
    starts = [e[2] for e in rec.events if e[0] == "f_start"]
    ends = [e[2] for e in rec.events if e[0] == "f_end"]
    check("畳む: 「応答が近い」の後に新しい語を始めない", all(s <= t_near for s in starts), f"{starts} near={t_near:.3f}")
    check("畳む: 鳴っていた語は言い切る（途中で切らない）", len(ends) == len(starts))
    check("畳む: 畳んだらゲートを開ける", not audio.floor_held())
    await pl.stop()
    w.cancel()


async def t_fold_in_gap() -> None:
    """「、」の間の途中で応答が近い合図が来たら、次の語は言わない（ユーザ 2026-09-28「間を1.2秒にして
    そもそも2回目ほぼ出させない」。以前は間を次の語の音に含めていたので、間の途中の合図でも次の語まで言い切っていた）。"""
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0, join_gap_sec=0.3)
    pl, audio, rec, texts = mk_player(t, word_sec=0.05)
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.15)  # 1語目（0.05秒）を言い終えて、間（0.3秒）の途中
    t_near = rec.now()
    pl.on_response_near()
    await asyncio.sleep(0.5)
    starts = [e[2] for e in rec.events if e[0] == "f_start"]
    check("間: 間の途中で合図が来たら次の語を言わない", len(starts) == 1, f"{starts} near={t_near:.3f}")
    check("間: 合図の後すぐ畳む（間を待ち切らない）", not pl.active and not audio.floor_held())
    await pl.stop()


def t_silence_after_word() -> None:
    """語の直後に出た無言は、語と語の間（join_gap_sec）より短くしない（無言を挟んでも2つのフィラーの間は 1.2秒）。"""
    t = mk_table(join_gap_sec=1.2, silence_sec=(0.3, 0.5))
    pl, audio, rec, texts = mk_player(t)
    check("無言: 1段目の無言はそのまま（最初に黙る 0.3〜0.5秒）", pl._silence_len(0.4, True, None) == 0.4)
    check("無言: 語の直後の無言は語と語の間まで延ばす（0.4 → 1.2）", pl._silence_len(0.4, False, "word") == 1.2)
    check("無言: 伸ばし棒・笑いの直後も同じ", pl._silence_len(0.3, False, "stretch") == 1.2
          and pl._silence_len(0.3, False, "laugh") == 1.2)
    check("無言: 無言の直後の無言は延ばさない", pl._silence_len(0.35, False, "silence") == 0.35)


async def t_combo() -> None:
    """特殊フィラー（例 あっと）は語と っと を**まとめて合成した1本**を、1つのフィラーとして鳴らす（ユーザ 2026-09-29
    「普通に繋げてすべて作成」→「今回の方針で独立することが決まった」）。単独の っと は鳴らさない・他の語と同じく
    「、」の間を空ける。感情の声の組があればそれ（無ければ普通の声・他の語と同じ）。"""
    t = mk_table(weights={"a": 50.0, "a+tto": 50.0}, texts={"a": "あ", "a+tto": "あっと"}, special={"a+tto": "あ"},
                 silence_weight=0.0, stretch_prob=0.0, max_chain_sec=5.0, join_gap_sec=0.05)
    rec = Rec()
    audio = AudioPlayQueue(play_fn=rec.queue_play)
    said: list = []
    audio.on_sentence_start = lambda text, meta: said.append(text)
    pl = FillerPlayer(mk_bank(t, word_sec=0.05), audio=audio, play_fn=rec.filler_play, is_busy=lambda: False,
                      rng=random.Random(3))
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.4)
    pl.on_response_near()
    await asyncio.sleep(0.1)
    await pl.stop()
    check("組: まとめて合成した1本（あっと）を1つのフィラーとして鳴らす（単独の っと は鳴らさない）",
          "あっと" in said and set(said) <= {"あ", "あっと"}, f"{said}")
    check("組: 同じものは続けない（直前と同じは除く）", all(a != b for a, b in zip(said, said[1:])), f"{said}")
    starts = [e[2] for e in rec.events if e[0] == "f_start"]
    ends = [e[2] for e in rec.events if e[0] == "f_end"]
    gaps = [s - e for e, s in zip(ends, starts[1:])]
    check("組: 他の語と同じく「、」の間を空ける（0.05秒）", gaps and min(gaps) >= 0.04, f"{[round(g, 3) for g in gaps]}")
    emo_bank = mk_bank(t, word_sec=0.05)
    emo_clip = Clip("a+tto", "あっと", tone(0.05, 550), SR)
    emo_bank.by_emotion["喜"] = {"a+tto": [emo_clip]}
    check("組: その感情の声の組があればそれで鳴らす", emo_bank.pick("a+tto", random.Random(1), emotion="喜")[1] is emo_clip)
    check("組: その感情の声が無い語は普通の声", emo_bank.pick("a", random.Random(1), emotion="喜")[1] in emo_bank.clips["a"])


def t_special() -> None:
    """特殊フィラーの表（本物の fillers.yaml の special）。ユーザ 2026-09-29「確率0.2は1段目にくっつく確率であり、
    今回の方針で独立することが決まったため5％にする。ん系特殊2、え系特殊2、うん系特殊1.5、あ系特殊0.5」→
    「１は按分でいい。２は按分するから変わらない。…４は均等でいい。」"""
    import json
    import tempfile

    import yaml

    if not _real_table():
        return
    path = os.path.join(REPO, "fillers.yaml")
    real = FillerTable.from_yaml(path)
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    fam: dict = {}
    for cid, f in real.special.items():
        fam.setdefault(f, []).append(cid)
    want = {"ん": ["nn"], "え": ["e", "ee"], "うん": ["u", "uu"], "あ": ["a", "aa"]}
    check("特殊: 組は系の語 × っと・っとね（ん2・え4・うん4・あ4＝14組・お・あの・その の組は無い）",
          {f: sorted(v) for f, v in fam.items()}
          == {f: sorted(f"{w}+{x}" for w in ws for x in ("tto", "ttone")) for f, ws in want.items()}, f"{fam}")
    tot = {f: sum(real.weights[c] for c in v) for f, v in fam.items()}
    check("特殊: 合計 5%", abs(sum(tot.values()) - 5.0) < 1e-9, f"{tot}")
    check("特殊: 系は 2 : 2 : 1.5 : 0.5 で按分（ん 1.67・え 1.67・うん 1.25・あ 0.42）",
          all(abs(tot[f] - 5.0 * s / 6.0) < 1e-9 for f, s in (("ん", 2.0), ("え", 2.0), ("うん", 1.5), ("あ", 0.5))),
          f"{tot}")
    check("特殊: 系の中は等分", all(len({round(real.weights[c], 9) for c in v}) == 1 for v in fam.values()))
    yw = {e["id"]: float(e["weight"]) for e in cfg["entries"] if "independent" in e["role"]}
    check("特殊: 無言と語は同じ比率で縮める（×0.95＝無言 19・語の比は表のまま）",
          abs(real.silence_weight - 19.0) < 1e-9 and all(abs(real.weights[w] - yw[w] * 0.95) < 1e-9 for w in yw),
          f"無言 {real.silence_weight}")
    check("特殊: 組の文字（え → えっと・う → うんっと・うー → うーんっとね・んー → んーっとね）",
          real.texts["e+tto"] == "えっと" and real.texts["u+tto"] == "うんっと"
          and real.texts["uu+ttone"] == "うーんっとね" and real.texts["nn+ttone"] == "んーっとね",
          f"{ {c: real.texts[c] for c in real.special} }")
    t2 = real.without({"e+tto"})
    check("特殊: 無い組を外した表（重み・系・場面から消え、元の表は変えない）",
          "e+tto" not in t2.weights and "e+tto" not in t2.special
          and all("e+tto" not in o for o in t2.mood_first.values())
          and all("e+tto" not in d for d in t2.mood_special.values())
          and "e+tto" in real.weights and "e+tto" in real.mood_first["通常"])
    with tempfile.TemporaryDirectory() as d:
        bad = yaml.safe_load(open(path, encoding="utf-8"))
        bad["special"]["families"]["あ"]["words"] = ["あ", "あの"]
        p = os.path.join(d, "f.yaml")
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(bad, f, allow_unicode=True)
        try:
            FillerTable.from_yaml(p)
            ok = False
        except ValueError:
            ok = True
        check("特殊: っと・っとね を付けない語（あの・その・お）を special に書いたら止める", ok)
        # バンクに無い組は外して鳴らす（語だけでも鳴らせる）・ある組は語と同じく読む
        t = mk_table(weights={"a": 90.0, "a+tto": 5.0, "a+ttone": 5.0},
                     texts={"a": "あ", "a+tto": "あっと", "a+ttone": "あっとね"},
                     special={"a+tto": "あ", "a+ttone": "あ"}, mood_first={"通常": ("a", "a+tto", "a+ttone")})
        man = {"clips": []}
        for cid, role in (("a", ["independent"]), ("a+tto", ["combo_terminal"])):
            fn = f"{cid}.wav"
            with open(os.path.join(d, fn), "wb") as f:
                f.write(fa.to_wav(tone(0.1), SR))
            man["clips"].append({"file": fn, "id": cid, "text": t.texts[cid], "role": role})
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False)
        from pathlib import Path

        b = FillerBank.load(t, Path(d))
        check("特殊: バンクに無い組（あっとね）は抽選から外して鳴らす・ある組（あっと）は読む",
              "a+ttone" not in b.table.weights and "a+ttone" not in b.table.mood_first["通常"]
              and "a+tto" in b.clips and b.clips["a+tto"][0].text == "あっと" and "a+ttone" in t.weights,
              f"{list(b.table.weights)}")


async def t_limit() -> None:
    t = mk_table(max_chain_sec=0.4)
    pl, audio, rec, texts = mk_player(t, word_sec=0.07)
    pl.start()
    t0 = rec.now()
    pl.on_speech_end(2.0)
    await asyncio.sleep(0.8)
    ends = [e[2] for e in rec.events if e[0] == "f_end"]
    check("上限: 合図が来なければ上限まで鳴らして止める", ends and not audio.floor_held() and not pl.active)
    check("上限: 上限を超える語は始めない（全体が上限以内）", ends and max(ends) - t0 <= 0.4 + 0.05,
          f"{max(ends) - t0 if ends else None:.3f}")
    await pl.stop()


async def t_barge_in() -> None:
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0)
    pl, audio, rec, texts = mk_player(t, word_sec=0.08)
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.05)  # 1語目の途中
    t_b = rec.now()
    pl.on_barge_in()
    await asyncio.sleep(0.25)
    starts = [e[2] for e in rec.events if e[0] == "f_start"]
    ends = [e[2] for e in rec.events if e[0] == "f_end"]
    check("割り込み: 鳴っている語は言い切る", len(starts) == 1 and len(ends) == 1, f"{starts} {ends}")
    check("割り込み: 次の語は始めない", all(s <= t_b for s in starts))
    # 次の発話のフィラーには持ち越さない
    n = len(starts)
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.2)
    check("割り込み: 前の発話への合図は次の発話のフィラーを止めない",
          len([e for e in rec.events if e[0] == "f_start"]) > n)
    await pl.stop()


async def t_empty_stt() -> None:
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0)
    pl, audio, rec, texts = mk_player(t, word_sec=0.08)
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.05)
    pl.on_stt_result(False)
    t_e = rec.now()
    await asyncio.sleep(0.25)
    starts = [e[2] for e in rec.events if e[0] == "f_start"]
    check("空認識: 次の語につながない", all(s <= t_e for s in starts) and not audio.floor_held())
    check("空認識: つなぎ言葉を次の応答に持ち越さない", pl.take_spoken() is None)
    await pl.stop()


async def t_always_release() -> None:
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0)
    rec = Rec()
    audio = AudioPlayQueue(play_fn=rec.queue_play)

    async def boom(wav, should_stop=None):
        raise RuntimeError("再生デバイスが消えた")

    pl = FillerPlayer(mk_bank(t), audio=audio, play_fn=boom, is_busy=lambda: False, rng=random.Random(3))
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.05)
    check("必ず開く: 再生で例外が出てもゲートを開ける", not audio.floor_held() and not pl.active)
    pl._play = rec.filler_play
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.05)
    check("必ず開く: 例外の後も次の発話で鳴らせる（常駐タスクが生きている）",
          any(e[0] == "f_start" for e in rec.events))
    await pl.stop()
    check("必ず開く: 止めた（キャンセル）時もゲートを開ける", not audio.floor_held())
    # 発話の直後に止める（鳴らし始める前）
    pl2, audio2, _, _ = mk_player(t)
    pl2.start()
    pl2.on_speech_end(1.5)
    await pl2.stop()
    check("必ず開く: 鳴らし始める前に止めてもゲートを開ける", not audio2.floor_held())


async def t_spoken_and_gap() -> None:
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0, join_gap_sec=0.05)
    pl, audio, rec, texts = mk_player(t, word_sec=0.05)
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.17)  # 2語ぶん
    spoken = pl.take_spoken()
    check("伝える: 鳴らし始めた語を「、」でつないで渡す", spoken is not None and len(spoken.split("、")) >= 2
          and all(s in ("あ", "あー", "あの", "っと") for s in spoken.split("、")), f"{spoken}")
    check("伝える: 1回だけ（次の刺激に持ち越さない）", pl.take_spoken() is None)
    pl.on_response_near()
    await asyncio.sleep(0.15)
    starts = [e[2] for e in rec.events if e[0] == "f_start"]
    ends = [e[2] for e in rec.events if e[0] == "f_end"]
    durs = [b - a for a, b in zip(starts, ends)]
    check("つなぎ: 2語目の前に「、」の間（無音）を空け、語の音には間を含めない",
          len(starts) >= 2 and starts[1] - ends[0] >= 0.05 - 0.01 and all(d < 0.05 + 0.03 for d in durs[:2]),
          f"starts={starts} ends={ends}")
    # 鮮度
    saved = player_mod._SPOKEN_FRESH_SEC
    try:
        player_mod._SPOKEN_FRESH_SEC = 0.01
        pl.on_speech_end(1.5)
        await asyncio.sleep(0.08)
        pl.on_response_near()
        check("伝える: 古いつなぎ言葉（鮮度切れ）は渡さない", pl.take_spoken() is None)
    finally:
        player_mod._SPOKEN_FRESH_SEC = saved
    await pl.stop()
    # 同じ語の同じ1本は続けない
    bank = mk_bank(t)
    rng = random.Random(5)
    ks = [bank.pick("a", rng)[0] for _ in range(300)]
    check("バンク: 同じ語の同じ1本を続けて鳴らさない", all(a != b for a, b in zip(ks, ks[1:])))


def t_contour() -> None:
    """抑揚の物差し（ユーザ 2026-09-26「フィラーは…音程同じか、若干下がる」）。"""
    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    flat = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    f_rise = 300 * 2 ** (6 * t / 0.5 / 12)  # 0.5秒で 6半音上がる
    rise = (0.3 * np.sin(2 * np.pi * np.cumsum(f_rise) / sr)).astype(np.float32)
    f_fall = 300 * 2 ** (-1 * t / 0.5 / 12)  # 1半音下がる
    fall = (0.3 * np.sin(2 * np.pi * np.cumsum(f_fall) / sr)).astype(np.float32)
    cf, cr, cd = fa.contour(flat, sr, 300.0), fa.contour(rise, sr, 300.0), fa.contour(fall, sr, 300.0)
    check("抑揚: 平らな音は語尾 ≈ 0半音・高さ ≈ 0半音", cf and abs(cf[1]) < 0.3 and abs(cf[0]) < 0.3, f"{cf}")
    check("抑揚: 上がる音は語尾が＋（6半音上げ→約+4以上）", cr and cr[1] > 3.5, f"{cr}")
    check("抑揚: 若干下がる音は語尾が少し−", cd and -1.5 < cd[1] < -0.3, f"{cd}")
    check("抑揚: 落ち着いた点数は 平ら・若干下がる < 上がる",
          fa.calm_score(cd) <= fa.calm_score(cf) + 0.2 and fa.calm_score(cf) < fa.calm_score(cr))
    check("抑揚: 測れない音は選ばれにくい（大きい点数）", fa.calm_score(None) > fa.calm_score(cr))
    # 伸ばし棒: 継ぎ目で高さが跳ぶ組み合わせは伸ばさない
    word = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    high = (0.3 * np.sin(2 * np.pi * 600 * np.arange(int(1.5 * sr)) / sr)).astype(np.float32)
    same = (0.3 * np.sin(2 * np.pi * 300 * np.arange(int(1.5 * sr)) / sr)).astype(np.float32)
    check("伸ばし棒: 継ぎ目で1オクターブ跳ぶ素材では伸ばさない（None）", fa.stretch(word, [high], sr) is None)
    out = fa.stretch(word, [same], sr)
    check("伸ばし棒: 同じ高さの素材なら伸ばす（長くなる）", out is not None and len(out) > len(fa.trim(word, sr)))


async def t_stretch_path() -> None:
    """伸ばし棒（採用された場合）: 先読みで次が伸ばし棒なら、その語を「ー」を継いだ音で鳴らす（間を空けない）。"""
    t = mk_table(silence_weight=0.0, max_chain_sec=5.0, stretch_prob=1.0,
                 stretch_adopted=True, join_gap_sec=0.05)
    bank = mk_bank(t, word_sec=0.05)
    for wid in t.weights:
        for k in range(3):
            bank.stretched[(wid, k)] = Clip(wid, t.texts[wid] + "ー", tone(0.12), SR)
    rec = Rec()
    audio = AudioPlayQueue(play_fn=rec.queue_play)
    texts: list = []
    audio.on_sentence_start = lambda text, meta: texts.append(text)
    pl = FillerPlayer(bank, audio=audio, play_fn=rec.filler_play, is_busy=lambda: False, rng=random.Random(9))
    pl.start()
    pl.on_speech_end(1.5)
    await asyncio.sleep(0.2)  # 継いだ1語目（0.12秒）が言い終わるまで
    first_dur = next((b[2] - a[2] for a, b in zip(rec.events, rec.events[1:])
                      if a[0] == "f_start" and b[0] == "f_end"), 0.0)
    check("伸ばし棒: 継いだ音で鳴らす（あ→あー）", texts and texts[0].endswith("ー"), f"{texts}")
    check("伸ばし棒: 1語目は「、」の間を付けず継いだ長さで鳴る", 0.11 <= first_dur < 0.16, f"{first_dur:.3f}")
    await pl.stop()
    t2 = mk_table(stretch_adopted=None)
    check("伸ばし棒: 未決（null）なら使わない", not FillerBank(t2, mk_bank(t2).clips, {("a", 0): None}, SR).stretch_ok())


# ---------------------------------------------------------------------------
# 応答LLM への伝え方（会話履歴・RAG には入れない）
# ---------------------------------------------------------------------------

async def t_orchestrator() -> None:
    from eve.context_assembler import FILLER_PREFIX, FILLER_RULE
    from eve.memory import ConversationCache
    from eve.response.orchestrator import ResponseOrchestrator

    rec = Rec()
    audio = AudioPlayQueue(play_fn=rec.queue_play)
    w = asyncio.create_task(audio.play_worker())
    seen: list = []
    near: list = []

    async def stream(messages, **kw):
        seen.append(messages)
        for d in ["そうなんだ。", "いいね。"]:
            yield d

    async def tts(s):
        return f"wav:{s}"

    import tempfile
    cache = ConversationCache(history_file=os.path.join(tempfile.mkdtemp(), "h.jsonl"))
    await cache.initialize()
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return "あー、えー"

    orch = ResponseOrchestrator(audio, stream, tts, conversation_cache=cache,
                                on_first_sentence=lambda: near.append(1), filler_provider=provider)
    stim = Stimulus(StimulusKind.USER_UTTERANCE, "今日は雨だった")
    await orch.handle(stim)
    msgs = seen[-1]
    last_user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    system = [m for m in msgs if m["role"] == "system"][0]["content"]
    check("伝える: 最後の user メッセージの前置きにする",
          last_user == FILLER_PREFIX.format("あー、えー") + "\n今日は雨だった", repr(last_user))
    check("伝える: system に「同じつなぎ言葉で書き出さない」規則", FILLER_RULE in system)
    check("伝える: stimulus.payload は変えない", stim.payload == "今日は雨だった")
    turns = [(t.speaker, t.text) for t in cache.recent()]
    check("会話履歴: user はフィラー無しの原文・eve は応答の文だけ",
          ("user", "今日は雨だった") in turns and ("eve", "そうなんだ。いいね。") in turns
          and not any("あー" in txt for _, txt in turns), f"{turns}")
    check("応答が近い: そのターンで1回だけ（1文目の合成開始で）", near == [1], f"{near}")

    # 自発発話・機能報告ではつなぎ言葉を扱わない
    calls["n"] = 0
    await orch.handle(Stimulus(StimulusKind.CALLFUNCTION_RESULT, CallFunctionResult("x", "結果です")))
    msgs = seen[-1]
    check("伝える: 機能報告のターンでは読まない・前置きも規則も無い",
          calls["n"] == 0 and FILLER_RULE not in msgs[0]["content"]
          and not any(FILLER_PREFIX.split("{")[0] in (m.get("content") or "") for m in msgs[1:]))
    # フィラーが無いターンは前置きも規則も無い
    orch2 = ResponseOrchestrator(audio, stream, tts, filler_provider=lambda: None)
    await orch2.handle(Stimulus(StimulusKind.USER_UTTERANCE, "こんにちは"))
    msgs = seen[-1]
    check("伝える: 鳴らしていない時は前置きも規則も出さない（プロンプトを太らせない）",
          msgs[-1]["content"] == "こんにちは" and FILLER_RULE not in msgs[0]["content"])
    await cache.shutdown()
    w.cancel()


# ---------------------------------------------------------------------------
# S4 応答依存フィラー（表を応答LLM の system に入れる）
# ---------------------------------------------------------------------------

def t_reply_opener() -> None:
    import tempfile

    import yaml

    if not _real_table():
        return

    from eve.context_assembler import ContextAssembler
    from eve.filler.prompt import HEADING, reply_opener_block

    import re

    path = os.path.join(REPO, "fillers.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    block = reply_opener_block(path)
    # ユーザ 2026-09-28 のルナ検証で使い 2026-09-29 に採用した表（fillers.yaml の opener_prompt）:
    # 初期案（v1）＋後から足した語の意味（v2）＋S8 の場面。語は足さない・外さない
    listed: set = set()
    for ln in (block or "").splitlines():
        if ln.startswith("- "):
            listed.update(w.strip() for w in ln[2:].split(": ", 1)[0].split("・"))
        m = re.match(r"笑い（(.+?)）", ln)
        if m:
            listed.update(m.group(1).split("・"))
        if "っと・っとね" in ln:
            listed.update({"っと", "っとね"})
    texts = {e["text"] for e in cfg["entries"] if "{user_name}" not in e["text"]}
    check("S4: 表の語（応答非依存・特別・応答依存・組み合わせ・笑い）をすべて載せる（外さない）",
          block and not (texts - listed), f"{sorted(texts - listed)}")
    check("S4: 表に無い語を足していない（v1 の「そ」だけ・ユーザの表 v1 の語）",
          listed - texts == {"そ"}, f"{sorted(listed - texts)}")
    check("S4: 一例・特定の語に偏らない・置かない場面も大事（ユーザ 2026-09-28）",
          "一例" in block and "偏らない" in block and "置かない場面も大事" in block)
    check("S4: 応答非依存の語に S8 の場面（通常・悩む・驚く・悲しむ・不満）を添える",
          all(k in block for k in ("通常", "悩む", "驚く", "悲しむ", "不満")))
    check("S4: へー の会話例は載せない（特定の語に寄せない）", "水素の原子記号はHだよ" not in block)
    check("S4: 名前が未設定なら呼びかけの行ごと消す", "{user_name_line}" not in block and "呼びかけ\n" not in block)
    cfg2 = dict(cfg, user_name="タロウ")
    d = tempfile.mkdtemp(prefix="eve_s4_")
    p2 = os.path.join(d, "f.yaml")
    with open(p2, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, allow_unicode=True)
    check("S4: 名前が決まれば呼びかけも入る", "- タロウ: 呼びかけ" in (reply_opener_block(p2) or ""))

    ca = ContextAssembler(system_prompt="S", reply_opener=block)
    user = ca.assemble(user_text="こんにちは")
    auto = ca.assemble(autonomous_content="下書き")
    cf = ca.assemble(callfunction_result="結果")
    check("S4: ユーザ発話への応答の system にだけ入る", HEADING in user[0]["content"]
          and HEADING not in auto[0]["content"] and HEADING not in cf[0]["content"])
    check("S4: 表が無ければ何も足さない", HEADING not in ContextAssembler("S").assemble(user_text="x")[0]["content"])

    from eve.config import Config
    from eve.voice_loop import VoiceLoop

    saved = Config.FILLER_DEPENDENT_ENABLED
    try:
        Config.FILLER_DEPENDENT_ENABLED = False
        check("S4: FILLER_DEPENDENT_ENABLED=0 なら入れない", VoiceLoop().orchestrator._ctx.reply_opener == "")
        Config.FILLER_DEPENDENT_ENABLED = True
        check("S4: FILLER_DEPENDENT_ENABLED=1 なら本物の表を入れる",
              VoiceLoop().orchestrator._ctx.reply_opener == block)
    finally:
        Config.FILLER_DEPENDENT_ENABLED = saved


# ---------------------------------------------------------------------------
# 配線（VoiceLoop）と本物の表・バンク
# ---------------------------------------------------------------------------

def t_wiring() -> None:
    import tempfile

    import yaml

    from eve.config import Config
    from eve.voice_loop import VoiceLoop

    saved = (Config.FILLER_ENABLED, Config.FILLER_TABLE)
    try:
        Config.FILLER_ENABLED = False
        vl = VoiceLoop()
        check("配線: FILLER_ENABLED=0 なら何もしない（フックも無い）",
              vl.filler is None and vl.orchestrator._filler_provider is None and vl.audio.on_head_waiting is None)
        Config.FILLER_ENABLED = True
        if not _real_table():
            return
        # つなぎの間が未決（null）の表では作らない（推測で決めない）。本物の表は決まっている（1.2）ので一時ファイルで
        cfg0 = yaml.safe_load(open(os.path.join(REPO, "fillers.yaml"), encoding="utf-8"))
        cfg0["timing"]["join_gap_sec"] = None
        d0 = tempfile.mkdtemp(prefix="eve_filler_")
        p0 = os.path.join(d0, "fillers.yaml")
        with open(p0, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg0, f, allow_unicode=True)
        Config.FILLER_TABLE = p0
        vl = VoiceLoop()
        check("配線: つなぎの間が未決（null）なら鳴らさない＝作らない（推測で決めない）", vl.filler is None)
        Config.FILLER_TABLE = os.path.join(REPO, "fillers.yaml")
        vl = VoiceLoop()
        check("配線: 本物の表（つなぎの間 1.2・ユーザ決定 2026-09-28）で作る",
              vl.filler is not None and vl.filler._t.join_gap_sec == 1.2)
        bank_ok = os.path.isfile(os.path.join(REPO, "bank", Config.FILLER_VOICE, "manifest.json"))
        if not bank_ok:
            print("SKIP 本物のバンクが無い（tools/build_filler_bank.py で作る）")
            return
        cfg = yaml.safe_load(open(os.path.join(REPO, "fillers.yaml"), encoding="utf-8"))
        cfg["timing"]["join_gap_sec"] = 0.2
        d = tempfile.mkdtemp(prefix="eve_filler_")
        p = os.path.join(d, "fillers.yaml")
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        Config.FILLER_TABLE = p
        vl = VoiceLoop()
        check("配線: つなぎの間が決まっていれば本物のバンクで稼働", vl.filler is not None)
        if vl.filler is None:
            return
        check("配線: 応答LLM へのつなぎ言葉・応答が近い合図・畳む合図",
              vl.orchestrator._filler_provider == vl.filler.take_spoken
              and vl.orchestrator._on_first_sentence == vl._on_response_near
              and vl.audio.on_head_waiting == vl.filler.on_head_ready)
        check("配線: 発話終了・文字起こし結果のフック", vl.input._on_speech_end == vl._on_speech_end
              and vl.input._on_stt_result == vl._on_stt_result)
        vl.filler._running_rid = 1  # フィラー中
        check("配線: フィラー中は自発発話を発火しない", vl._busy_for_autospeech() is True)
        check("配線: フィラー中は画面の実況を差し込まない", vl._vision_can_speak() is False)
        vl.filler._running_rid = 0
        check("配線: フィラーが終われば自発発話のガードは元どおり", vl._busy_for_autospeech() is False)
        bank = vl.filler._bank
        import json

        man = json.loads(open(os.path.join(REPO, "bank", Config.FILLER_VOICE, "manifest.json"), encoding="utf-8").read())
        # 長さごとに3本（長さの数は目録から・長い版は 1.0 秒だけ＝ユーザ 2026-09-29）。
        # っと・っとね は単独では鳴らさない＝読まない（語とまとめて合成した組だけを使う・2026-09-29）
        tt = vl.filler._t
        n_len = {wid: len({c["requested_sec"] for c in man["clips"] if c["id"] == wid}) for wid in bank.clips}
        words = [w for w in bank.clips if w not in tt.special]
        check("本物のバンク: 応答非依存10語がそろい、長さごとに3本（っと・っとね は単独では読まない）",
              len(words) == 10 and all(len(bank.clips[k]) == 3 * n_len[k] for k in words),
              f"{ {k: len(bank.clips[k]) for k in words} } 長さの数 {n_len}")
        full = FillerTable.from_yaml(p)
        combos = [w for w in bank.clips if w in tt.special]
        check("本物のバンク: 特殊フィラーの組が special のぶん（7語×2＝14組）そろい各3本（抽選から外れた組は無い）",
              len(full.special) == 14 and set(combos) == set(full.special) == set(tt.special)
              and all(len(bank.clips[k]) == 3 for k in combos),
              f"{len(combos)}組 / 不足 {sorted(set(full.special) - set(combos))}")
        from pathlib import Path

        from eve.emotion import BANK_SUFFIX, load_voice_emoji
        from eve.response.tts_irodori import resolve_repo_path

        emoji = load_voice_emoji(Path(resolve_repo_path(Config.EMOTION_TABLE)))
        bd = os.path.join(REPO, "bank", Config.FILLER_VOICE)
        eb = FillerBank.load(full, Path(bd), emotion_dirs={k: Path(bd + BANK_SUFFIX[k]) for k in emoji},
                             emotion_emoji=emoji)
        check("本物のバンク: 感情の声（喜怒哀楽笑い）にも特殊フィラーの組がそろう",
              len(eb.by_emotion) == len(emoji)
              and all(set(full.special) <= set(pool) for pool in eb.by_emotion.values()),
              f"{ {k: len(set(full.special) & set(v)) for k, v in eb.by_emotion.items()} }")
        # ユーザ採用 2026-09-26「伸ばし棒も採用」（fillers.yaml の stretch.adopted: true）
        check("本物のバンク: 伸ばし棒を使う（採用・素材から継げる）",
              bank.stretch_ok() and len(bank.stretched) > 0, f"{len(bank.stretched)}")
        # 長い版でも短い版でも成り立つ形: 合成した長さより短い＝刈ってある
        req = max(c["requested_sec"] for c in man["clips"] if c["id"] == "a")
        check(f"本物のバンク: 前後の無音を刈ってある（あ は合成した長さ {req}秒より短い）",
              all(c.sec < req for c in bank.clips["a"]), f"{[round(c.sec, 2) for c in bank.clips['a']]}")
    finally:
        Config.FILLER_ENABLED, Config.FILLER_TABLE = saved


async def main() -> None:
    await t_gate()
    t_lottery()
    t_special()
    t_silence_after_word()
    for fn in (t_no_filler, t_fold_on_response, t_fold_on_near, t_fold_in_gap, t_combo, t_limit,
               t_barge_in, t_empty_stt, t_always_release, t_spoken_and_gap, t_stretch_path, t_orchestrator):
        try:
            await asyncio.wait_for(fn(), 15.0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} が例外なく終わる", False, f"{type(e).__name__}: {e}")
    t_contour()
    t_reply_opener()
    t_wiring()


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
