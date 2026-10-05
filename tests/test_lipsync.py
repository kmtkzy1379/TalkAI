"""Tier-1: リップシンク（口の開閉）— 純ロジック + 再生開始/終了フックの配線。

決定論のみ（API・VOICEVOX・GUI 不要）。狙いは「UI を付けたことで本体が変わっていない」を
構造的に固定すること。特に:
  - フックが **既定 None なら既存挙動が完全に不変**
  - フックが例外を投げても**再生と on_played が続く**
  - barge-in で**必ず終了通知が出る**（口が開きっぱなしにならない）
  - `enqueue` のシグネチャが変わっていない（E2E ハーネスが固定引数で wrap している）
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402
from eve.response.lipsync import (  # noqa: E402
    MouthSmoother, level_at, plan_duration, plan_from_query, plan_from_query_safe,
)

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def _mora(text, vowel, vl, consonant=None, cl=None):
    return {"text": text, "vowel": vowel, "vowel_length": vl,
            "consonant": consonant, "consonant_length": cl}


def _query(moras, speed=1.0, pre=0.1, post=0.1, pause=None):
    return {"speedScale": speed, "prePhonemeLength": pre, "postPhonemeLength": post,
            "accent_phrases": [{"moras": moras, "pause_mora": pause}]}


# --------------------------------------------------------------- 1. 計画の組み立て

def t_plan_basic() -> None:
    print("\n[1] audio_query → 口の開閉計画")
    q = _query([_mora("コ", "o", 0.15, "k", 0.10), _mora("ン", "N", 0.08)])
    plan = plan_from_query(q)
    check("子音と母音で区間が分かれる", len(plan) == 3, f"len={len(plan)}")
    check("prePhonemeLength ぶん開始が遅れる", abs(plan[0][0] - 0.1) < 1e-6, f"{plan[0][0]}")
    check("破裂音 k では口が閉じる", plan[0][2] == 0.0, f"{plan[0][2]}")
    check("母音 o の開きが VOWEL_OPEN と一致", abs(plan[1][2] - 0.62) < 1e-6, f"{plan[1][2]}")
    check("ん は小さく開く", abs(plan[2][2] - 0.18) < 1e-6, f"{plan[2][2]}")
    check("長い母音は減衰して終わる", plan[1][3] < plan[1][2], f"{plan[1]}")

    q2 = _query([_mora("サ", "a", 0.15, "s", 0.06)])
    p2 = plan_from_query(q2)
    check("破裂音以外の子音はわずかに開く", abs(p2[0][2] - 0.08) < 1e-6, f"{p2[0][2]}")


def t_plan_speed() -> None:
    print("\n[2] speedScale が長さに効く")
    moras = [_mora("ア", "a", 0.20), _mora("イ", "i", 0.20)]
    slow = plan_duration(plan_from_query(_query(moras, speed=1.0)))
    fast = plan_duration(plan_from_query(_query(moras, speed=2.0)))
    check("speed=2.0 で総尺がおよそ半分", abs(fast - slow / 2) < 1e-6, f"{slow} vs {fast}")
    check("speed=0 でも落ちない（1.0 扱い）",
          abs(plan_duration(plan_from_query(_query(moras, speed=0))) - slow) < 1e-6)


def t_plan_pause_and_devoiced() -> None:
    print("\n[3] 無音・無声化母音")
    q = _query([_mora("ス", "U", 0.08, "s", 0.05)], pause={"vowel_length": 0.3})
    plan = plan_from_query(q)
    check("無声化母音（大文字）は浅い", abs(plan[1][2] - 0.34 * 0.45) < 1e-6, f"{plan[1][2]}")
    check("pause_mora で完全に閉じる区間が入る", plan[-1][2] == 0.0 and plan[-1][3] == 0.0)
    check("pause の長さが反映される", abs((plan[-1][1] - plan[-1][0]) - 0.3) < 1e-6)


def t_level_at() -> None:
    print("\n[4] level_at のサンプリング")
    plan = plan_from_query(_query([_mora("ア", "a", 0.20)], pre=0.0, post=0.0))
    check("開始前は 0", level_at(plan, -0.01) == 0.0)
    check("区間の頭は満開", abs(level_at(plan, 0.0) - 1.0) < 1e-6, f"{level_at(plan, 0.0)}")
    check("区間内は単調に減衰", level_at(plan, 0.19) < level_at(plan, 0.01))
    check("終了後は 0（口が閉じる）", level_at(plan, 0.5) == 0.0)
    check("空の計画でも落ちない", level_at([], 0.1) == 0.0)


def t_plan_safe() -> None:
    print("\n[5] 壊れた query でも音声を止めない")
    check("None なら None", plan_from_query_safe(None) is None)
    check("空 dict なら None", plan_from_query_safe({}) is None)
    check("accent_phrases が壊れていても例外を出さない",
          plan_from_query_safe({"accent_phrases": None, "speedScale": "x"}) is None)
    check("型が変でも例外を出さない",
          plan_from_query_safe({"accent_phrases": [{"moras": [{"vowel": None,
                                                               "vowel_length": "abc"}]}]}) is None)


def t_smoother() -> None:
    print("\n[6] MouthSmoother（非対称スムージング）")
    s = MouthSmoother(attack_sec=0.02, release_sec=0.04)
    for _ in range(10):
        s.step(1.0, 1 / 60)
    opened = s.value
    check("attack で開く", opened > 0.9, f"{opened:.3f}")
    a = MouthSmoother(0.02, 0.04)
    b = MouthSmoother(0.02, 0.04)
    a.value = b.value = 1.0
    a.step(0.0, 1 / 60)
    for _ in range(2):
        b.step(0.0, 1 / 120)
    check("フレームレート非依存（60fps と 120fps×2 が一致）", abs(a.value - b.value) < 1e-3,
          f"{a.value:.4f} vs {b.value:.4f}")
    # 同じ位置(0.5)から同じ dt で、開く側の方が多く進む＝閉じる側が遅い
    up, down = MouthSmoother(0.02, 0.04), MouthSmoother(0.02, 0.04)
    up.value = down.value = 0.5
    d_up = up.step(1.0, 0.01) - 0.5
    d_down = 0.5 - down.step(0.0, 0.01)
    check("release は attack より遅い（閉じる方がゆっくり）", d_up > d_down,
          f"開く {d_up:.4f} / 閉じる {d_down:.4f}")
    s.reset()
    check("reset で 0 に戻る", s.value == 0.0)


# --------------------------------------------------------------- 7. フックの配線

def t_hooks() -> None:
    print("\n[7] 再生開始/終了フック")

    async def run() -> None:
        played: list = []

        async def play_fn(audio, should_stop=None):
            played.append(audio)

        # (a) 既定 None で既存挙動が不変
        q = AudioPlayQueue(play_fn=play_fn)
        check("既定でフックは None（既存挙動が不変）",
              q.on_sentence_start is None and q.on_sentence_end is None)
        worker = asyncio.create_task(q.play_worker())
        q.enqueue(0, q.reserve_seq(0), b"a", text="あ。")
        await q.join()
        check("フック未設定でも再生される", played == [b"a"])

        # (b) フックが呼ばれる / meta が届く
        starts: list = []
        ends: list = []
        q.on_sentence_start = lambda text, meta: starts.append((text, meta))
        q.on_sentence_end = ends.append
        seq = q.reserve_seq(0)
        q.note_meta(0, seq, [(0.0, 0.1, 1.0, 1.0)])
        q.enqueue(0, seq, b"b", text="いろは。")
        await q.join()
        check("再生開始フックが文と計画を受け取る",
              starts == [("いろは。", [(0.0, 0.1, 1.0, 1.0)])], f"{starts}")
        check("再生終了フックが呼ばれる", ends == ["いろは。"], f"{ends}")
        check("meta は pop される（溜め込まない）", q._meta == {}, f"{q._meta}")

        # (c) 番兵（audio=None）では通知しない＝on_played と同じ判定
        starts.clear()
        ends.clear()
        q.enqueue(0, q.reserve_seq(0), None, text="スキップ。")
        await q.join()
        check("番兵(audio=None)では開始/終了とも通知しない", starts == [] and ends == [])

        # (d) フックの例外が再生と on_played を止めない
        got: list = []
        q.on_sentence_start = lambda text, meta: (_ for _ in ()).throw(RuntimeError("boom"))
        q.on_sentence_end = lambda text: (_ for _ in ()).throw(RuntimeError("boom2"))
        q.enqueue(0, q.reserve_seq(0), b"c", text="に。", on_played=got.append)
        await q.join()
        check("開始/終了フックが例外でも再生は続く", played[-1] == b"c", f"{played}")
        check("開始/終了フックが例外でも on_played は呼ばれる", got == ["に。"], f"{got}")

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def t_barge_in_closes_mouth() -> None:
    print("\n[8] barge-in でも必ず口が閉じる（言い切った後に終了通知が出る）")

    async def run() -> None:
        starts: list = []
        ends: list = []
        stopped = asyncio.Event()

        async def play_fn(audio):
            # 節を鳴らす（割り込まれても最後まで鳴る＝言い切る）
            for _ in range(20):
                stopped.set()
                await asyncio.sleep(0.005)

        q = AudioPlayQueue(play_fn=play_fn)
        q.on_sentence_start = lambda text, meta: starts.append(text)
        q.on_sentence_end = ends.append
        worker = asyncio.create_task(q.play_worker())
        q.enqueue(0, q.reserve_seq(0), b"a", text="長い文。")
        await asyncio.wait_for(stopped.wait(), timeout=2.0)
        check("再生開始が通知されている", starts == ["長い文。"], f"{starts}")
        q.bump_generation()  # barge-in
        for _ in range(100):
            if ends:
                break
            await asyncio.sleep(0.01)
        check("割り込まれても言い切った後に終了通知が出る（口を閉じられる）", ends == ["長い文。"], f"{ends}")

        # barge-in で捨てられた文のメタが残らない
        q.note_meta(q.current_generation(), 0, [(0.0, 1.0, 1.0, 1.0)])
        q.bump_generation()
        check("世代が変わると古いメタは捨てられる", q._meta == {}, f"{q._meta}")

        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def t_signatures() -> None:
    print("\n[9] シグネチャの固定（E2E ハーネス・barge-in 配線を守る）")
    enq = list(inspect.signature(AudioPlayQueue.enqueue).parameters)
    check("enqueue の引数が変わっていない",
          enq == ["self", "generation", "seq", "audio", "text", "on_played"], f"{enq}")
    init = list(inspect.signature(AudioPlayQueue.__init__).parameters)
    check("__init__ の引数が変わっていない", init == ["self", "play_fn"], f"{init}")

    from eve.response.player import RealAudioPlayer
    play = list(inspect.signature(RealAudioPlayer().play_fn).parameters)
    # 割り込みでも鳴っている節は止めない（言い切る・2026-09-22）＝止める合図は受け取らない
    check("play_fn は音声だけを受ける（割り込みで途中停止しない）", play == ["audio"], f"{play}")

    from eve.response.tts import VoicevoxTTS
    check("generate は従来どおり（bytes を返す契約の呼び出し元を守る）",
          list(inspect.signature(VoicevoxTTS.generate).parameters) == ["self", "text"])
    check("generate_with_meta が存在する", hasattr(VoicevoxTTS, "generate_with_meta"))


def t_orchestrator_duck_typing() -> None:
    print("\n[10] orchestrator が (wav, meta) と単一値の両方を受ける")

    async def run() -> None:
        from eve.context_assembler import ContextAssembler
        from eve.pipeline.stimulus import Stimulus, StimulusKind
        from eve.response.orchestrator import ResponseOrchestrator

        for label, tts_fn, want_meta in (
            ("単一値（従来の tts_fn）", lambda s: _ret(b"x"), False),
            ("(wav, meta) タプル", lambda s: _ret((b"x", [(0.0, 0.1, 1.0, 1.0)])), True),
        ):
            played: list = []
            starts: list = []

            async def play_fn(audio, should_stop=None):
                played.append(audio)

            q = AudioPlayQueue(play_fn=play_fn)
            q.on_sentence_start = lambda text, meta: starts.append(meta)
            worker = asyncio.create_task(q.play_worker())

            async def stream_fn(messages, **kw):
                yield "こんにちは。"

            orch = ResponseOrchestrator(q, stream_fn, tts_fn, ContextAssembler())
            await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, payload="やあ"))
            await q.join()
            check(f"{label}: 音声が再生される", played == [b"x"], f"{played}")
            check(f"{label}: meta の有無が期待どおり",
                  (starts and starts[0] is not None) == want_meta, f"{starts}")
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    asyncio.run(run())


def t_note_meta_failure_is_survivable() -> None:
    print("\n[11] note_meta が壊れても seq に穴を空けない（ターン丸ごと無音の防止）")

    async def run() -> None:
        from eve.context_assembler import ContextAssembler
        from eve.pipeline.stimulus import Stimulus, StimulusKind
        from eve.response.orchestrator import ResponseOrchestrator

        played: list = []

        async def play_fn(audio, should_stop=None):
            played.append(audio)

        q = AudioPlayQueue(play_fn=play_fn)

        def boom(*a, **k):
            raise RuntimeError("note_meta broken")

        q.note_meta = boom  # type: ignore[method-assign]
        worker = asyncio.create_task(q.play_worker())

        async def stream_fn(messages, **kw):
            yield "いち。にい。さん。"

        orch = ResponseOrchestrator(q, stream_fn,
                                    lambda s: _ret((b"x", [(0.0, 0.1, 1.0, 1.0)])),
                                    ContextAssembler())
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, payload="やあ"))
        await q.join()
        check("note_meta が例外でも全文が再生される（seq に穴が空かない）",
              played == [b"x", b"x", b"x"], f"{played}")
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


async def _ret(v):
    return v


if __name__ == "__main__":
    t_plan_basic()
    t_plan_speed()
    t_plan_pause_and_devoiced()
    t_level_at()
    t_plan_safe()
    t_smoother()
    t_hooks()
    t_barge_in_closes_mouth()
    t_signatures()
    t_orchestrator_duck_typing()
    t_note_meta_failure_is_survivable()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
