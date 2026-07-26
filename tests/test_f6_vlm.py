"""F6 VLM 画面認識の Tier-1 決定論テスト（mss/GPU/API/実スレッド 不使用）。

注入: フレーム列の fake capture / fake narrate_fn(Event gate 可) / fake clock / stub bridge。
実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f6_vlm.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.feedback import NEUTRAL_SURPRISE, PredictionState  # noqa: E402
from eve.vlm import (  # noqa: E402
    BLANK_MARKER,
    ChangeDetector,
    Frame,
    VisionResult,
    VisionState,
    VlmWorker,
    parse_vision,
)
from eve.speech.decider import (  # noqa: E402
    SpeechDecision,
    build_decide_messages,
    should_speak,
)
from eve.vlm.change_detector import hamming  # noqa: E402


def _frame(fid: int, blank: bool = False) -> Frame:
    return Frame(frame_id=fid, mono_ts=float(fid), jpeg_b64=f"b64-{fid}", blank=blank)


async def _pump(n: int = 12) -> None:
    """イベントループを n 回回して worker を進める。"""
    for _ in range(n):
        await asyncio.sleep(0)


class FakeNarrator:
    """注入用 narrate_fn。並行度を計測し、gate で完了タイミングを制御できる。"""

    def __init__(self, gate: asyncio.Event | None = None, result_fn=None):
        self.calls: list[list[int]] = []  # 各呼び出しに渡ったフレームID列
        self.concurrent = 0
        self.max_concurrent = 0
        self._gate = gate
        self._result_fn = result_fn or (
            lambda frames: VisionResult(narration=f"画面{frames[-1].frame_id}番", notable=True, surprise_diff=50)
        )

    async def __call__(self, frames: list[Frame]) -> VisionResult:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.calls.append([f.frame_id for f in frames])
        try:
            if self._gate is not None:
                await self._gate.wait()
            return self._result_fn(frames)
        finally:
            self.concurrent -= 1


def _mkworker(vs, pred, narr, **kw):
    kw.setdefault("frames_per_call", 4)
    kw.setdefault("min_interval_sec", 0.0)
    kw.setdefault("dedup_ratio", 1.1)  # 既定は dedup 無効（dedup テストのみ有効化）
    return VlmWorker(vision_state=vs, prediction_state=pred, narrate_fn=narr, **kw)

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


# ===== パーサ =====
def t_parse_valid() -> bool:
    r = parse_vision("narration: メモ帳に文章を入力中\nnotable: yes\nsurprise: 45")
    return r.visible and "メモ帳" in r.narration and r.notable and r.surprise_diff == 45


def t_parse_garbage_safe() -> bool:
    # 完全なゴミでも raise せず安全既定（visible=True・空）
    r = parse_vision("@@@##$$ 壊れた出力 \n\n {")
    return r.visible and r.narration == "" and r.surprise_diff is None and not r.notable


def t_parse_fullwidth_colon() -> bool:
    r = parse_vision("ナレーション：ブラウザでニュースを閲覧\n驚き：30")
    return "ブラウザ" in r.narration and r.surprise_diff == 30


def t_parse_invisible_marker() -> bool:
    # A11: 本文に視認不可マーカ → 中身は採用せず surprise も上げない
    r = parse_vision("narration: 黒い画面で視認不可\nsurprise: 90")
    return (not r.visible) and r.narration == "" and r.surprise_diff is None and not r.notable


def t_parse_visible_false_tag() -> bool:
    r = parse_vision("visible: no\nnarration: 何か映ってるかも")
    return (not r.visible) and r.narration == ""


def t_parse_none() -> bool:
    r = parse_vision(None)
    return r.visible and r.is_empty()


# ===== 変化ゲート =====
def t_gate_first_always() -> bool:
    cd = ChangeDetector()
    return cd.evaluate(0xFFFF) is True  # 初回は必ず True


def t_gate_identical_none() -> bool:
    cd = ChangeDetector()
    cd.evaluate(0xABCD)  # 参照確立
    return cd.evaluate(0xABCD) is False  # 同一 → 変化なし


def t_gate_far_changed() -> bool:
    cd = ChangeDetector(phash_threshold=12)
    cd.evaluate(0x0000000000000000)
    # 多数ビットが立つ → hamming 大 → True
    return cd.evaluate(0xFFFFFFFFFFFFFFFF) is True


def t_gate_small_change_below_threshold() -> bool:
    cd = ChangeDetector(phash_threshold=12)
    cd.evaluate(0x0)
    # 3 ビットだけ変化（< 12）→ False
    return cd.evaluate(0b111) is False


def t_gate_periodic_forced() -> bool:
    cd = ChangeDetector(phash_threshold=12, periodic_frames=3)
    cd.evaluate(0x0)  # 参照
    r1 = cd.evaluate(0x0)  # idle 1 → False
    r2 = cd.evaluate(0x0)  # idle 2 → False
    r3 = cd.evaluate(0x0)  # idle 3 → periodic 強制 True
    return r1 is False and r2 is False and r3 is True


def t_hamming() -> bool:
    return hamming(0b1010, 0b0001) == 3


# ===== VisionState（ring / snapshot）=====
def t_ring_cap_drop_oldest() -> bool:
    vs = VisionState(ring_max=3)
    for i in range(5):
        vs.add_frame(_frame(i))
    ids = [f.frame_id for f in vs.ring]
    return len(vs.ring) == 3 and ids == [2, 3, 4]  # 最古(0,1)が drop


def t_snapshot_last_k() -> bool:
    vs = VisionState(ring_max=6)
    for i in range(5):
        vs.add_frame(_frame(i))
    snap = vs.snapshot(2)
    return [f.frame_id for f in snap] == [3, 4] and isinstance(snap, list)


def t_fresh_vision_ttl() -> bool:
    # 鮮度 TTL: 新しければ返す・古ければ None（明らか過去を参照させない）
    vs = VisionState()
    vs.set_latest("画面の内容", mono=100.0)
    fresh = vs.fresh_vision(ttl=7.0, now=105.0)   # 5s 前 → 返す
    stale = vs.fresh_vision(ttl=7.0, now=109.0)   # 9s 前 → None
    none0 = VisionState().fresh_vision(ttl=7.0, now=0.0)  # 未設定 → None
    return fresh == "画面の内容" and stale is None and none0 is None


def t_vision_for_user_static_hold() -> bool:
    # 層分離: 画面が変化していない間、ユーザ応答用は据え置き可（正直な注記つき）。
    # 一方で自発発話系(fresh_vision)には渡らない＝静止中の画面固執を構造で断つ。
    vs = VisionState()
    vs.add_frame(_frame(0), True, now=100.0)   # 変化フレーム
    vs.set_latest("メモ帳が開いている", mono=101.0)
    vs.add_frame(_frame(1), False, now=129.5)  # 静止のまま capture は生きている
    held = vs.vision_for_user(ttl=7.0, static_max=180.0, now=130.0)
    strict = vs.fresh_vision(ttl=7.0, now=130.0)
    return (
        held is not None
        and "メモ帳" in held
        and "変化なし" in held  # 「N秒前の画面」だと明示（捏造しない）
        and strict is None      # 自発発話・発話判定には据え置きを渡さない
    )


def t_vision_for_user_invalidated() -> bool:
    # 据え置きが無効になる3条件: 実況後に変化が来た / capture 停止 / 上限超過
    vs = VisionState()
    vs.add_frame(_frame(0), True, now=100.0)
    vs.set_latest("メモ帳が開いている", mono=101.0)
    vs.add_frame(_frame(1), True, now=140.0)  # 実況より後に変化（まだ実況が追いついていない）
    after_change = vs.vision_for_user(7.0, 180.0, now=141.0)

    vs2 = VisionState()
    vs2.add_frame(_frame(0), True, now=100.0)
    vs2.set_latest("メモ帳が開いている", mono=101.0)  # capture は以後止まっている
    capture_dead = vs2.vision_for_user(7.0, 180.0, now=130.0)

    vs3 = VisionState()
    vs3.add_frame(_frame(0), True, now=100.0)
    vs3.set_latest("メモ帳が開いている", mono=101.0)
    vs3.add_frame(_frame(1), False, now=299.5)
    too_old = vs3.vision_for_user(7.0, 180.0, now=300.0)  # age 199s > 上限 180s
    return after_change is None and capture_dead is None and too_old is None


def t_vision_for_user_marker_not_held() -> bool:
    # A11 の正直マーカ（視認不可）は据え置かない（「30秒前は取得できなかった」は無意味/紛らわしい）
    vs = VisionState()
    vs.add_frame(_frame(0), True, now=100.0)
    vs.set_latest(BLANK_MARKER, mono=101.0, narration=False)
    vs.add_frame(_frame(1), False, now=129.5)
    return vs.vision_for_user(7.0, 180.0, now=130.0) is None


# ===== surprise 合成（most-recent-wins・A4/Q2）=====
def t_surprise_cold_neutral() -> bool:
    return PredictionState().surprise == NEUTRAL_SURPRISE


def t_surprise_vlm_only() -> bool:
    s = PredictionState()
    s.note_vlm_surprise(80)
    return s.surprise == 80


def t_surprise_most_recent_wins() -> bool:
    s = PredictionState()
    s.note_vlm_surprise(80)
    a = s.surprise  # 80
    s.note_feedback_surprise(10)
    b = s.surprise  # feedback が最新 → 10（max なら 80 のはず）
    s.note_vlm_surprise(70)
    c = s.surprise  # vlm が最新 → 70
    return a == 80 and b == 10 and c == 70


def t_surprise_vlm_clamp() -> bool:
    s = PredictionState()
    s.note_vlm_surprise(150)
    return s.surprise == 100  # 上限クランプ


# ===== VlmWorker（single-flight / backpressure / A1 / A9 / A11 / dedup / guard）=====
async def t_backpressure_one_call_freshest() -> bool:
    vs, pred = VisionState(ring_max=6), PredictionState()
    gate = asyncio.Event()
    narr = FakeNarrator(gate=gate)
    w = _mkworker(vs, pred, narr)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()  # narrate([0]) が gate 待ちに入る
    for i in range(1, 6):  # 処理中に5枚 変化が到着
        w.on_frame(_frame(i), True)
    await _pump()
    gate.set()
    await _pump()
    await w.stop()
    # 呼び出しは 2回のみ（[0] と 最新ウィンドウ）・stale バックログを1枚ずつ処理しない・最新(5)を含む
    return (
        len(narr.calls) == 2
        and narr.calls[0] == [0]
        and 5 in narr.calls[1]
        and narr.max_concurrent == 1
    )


async def t_single_flight_max_one() -> bool:
    vs, pred = VisionState(), PredictionState()
    gate = asyncio.Event()
    narr = FakeNarrator(gate=gate)
    w = _mkworker(vs, pred, narr)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()
    for i in range(1, 4):  # 処理中に複数トリガ
        w.on_frame(_frame(i), True)
    await _pump()
    gate.set()
    await _pump()
    await w.stop()
    return narr.max_concurrent == 1


async def t_last_frame_never_stranded() -> bool:
    # A1: narrate 中に来た最後のフレームを取り残さない（自己再トリガ・外部トリガ無しで最新に追いつく）
    vs, pred = VisionState(), PredictionState()
    gate = asyncio.Event()
    narr = FakeNarrator(gate=gate)
    w = _mkworker(vs, pred, narr)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()  # narrate([0]) gate 待ち
    w.on_frame(_frame(1), True)  # 処理中に新フレーム B
    await _pump()
    gate.set()
    await _pump()
    await w.stop()
    # 2回目の呼び出しが自己再トリガで起き（窓は[0,1]＝変化前アンカー込み・A8）、latest が最新1を反映
    return (
        len(narr.calls) == 2
        and narr.calls[1][-1] == 1  # 最新ウィンドウの末尾＝今のフレーム
        and vs.latest_vision == "画面1番"
    )


async def t_static_no_self_retrigger() -> bool:
    # ⭐A1 死活: narrate 中に来たのが**変化なし**フレームだけなら自己再トリガしない。
    # （旧実装は ring 末尾の frame_id で判定したため、capture が静止中も 2fps で積む＝常に真＝
    #  一度起動すると永久ループ。実機実測 229回/1061s・15秒以上の空白 0 の再発防止テスト）
    vs, pred = VisionState(), PredictionState()
    gate = asyncio.Event()
    narr = FakeNarrator(gate=gate)
    w = _mkworker(vs, pred, narr)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()  # narrate([0]) gate 待ち
    for i in range(1, 4):  # 処理中に「変化なし」フレームだけが積まれる
        w.on_frame(_frame(i), False)
    await _pump()
    gate.set()
    await _pump()
    for i in range(4, 8):  # 完了後も静止が続く
        w.on_frame(_frame(i), False)
    await _pump()
    await w.stop()
    return len(narr.calls) == 1  # 呼び出しは最初の1回だけ（静止では回さない）


async def t_change_after_static_retriggers() -> bool:
    # A1 の逆側（過剰修正の検出）: 静止が続いた後の**変化**フレームでは必ず起きる。
    vs, pred = VisionState(), PredictionState()
    narr = FakeNarrator()
    w = _mkworker(vs, pred, narr)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()
    for i in range(1, 4):
        w.on_frame(_frame(i), False)  # 静止（起きない）
    await _pump()
    stayed = len(narr.calls) == 1
    w.on_frame(_frame(4), True)  # 変化（起きる）
    await _pump()
    await w.stop()
    return stayed and len(narr.calls) == 2 and narr.calls[1][-1] == 4


async def t_latest_vision_written() -> bool:
    vs, pred = VisionState(), PredictionState()
    narr = FakeNarrator(result_fn=lambda f: VisionResult(narration="ブラウザ閲覧中", surprise_diff=40))
    w = _mkworker(vs, pred, narr)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()
    await w.stop()
    return vs.latest_vision == "ブラウザ閲覧中" and pred.surprise == 40


async def t_min_interval_paces() -> bool:
    # A9: min_interval 内は ≤1 呼び出し（deferred で律速・fake clock で確定的に）
    clock = [0.0]
    async def fsleep(d):
        clock[0] += d
    vs, pred = VisionState(), PredictionState()
    narr = FakeNarrator()  # 即時完了
    w = _mkworker(vs, pred, narr, min_interval_sec=2.0, now_fn=lambda: clock[0], sleep_fn=fsleep)
    w.start()
    w.on_frame(_frame(0), True)  # t=0 で1回目
    await _pump()
    for i in range(1, 6):  # 連続変化（同一 fake 時刻で殺到）
        w.on_frame(_frame(i), True)
    await _pump(20)
    await w.stop()
    # 連続変化でも呼び出しは律速され少数（≤2）・2回目は clock>=2.0 に進む
    return len(narr.calls) <= 2 and clock[0] >= 2.0


async def t_dedup_no_retrigger() -> bool:
    # 同一ナレーションの連続 → latest は更新するが発話は再トリガしない
    spk = []
    vs, pred = VisionState(), PredictionState()
    narr = FakeNarrator(result_fn=lambda f: VisionResult(narration="同じ実況です", notable=True, surprise_diff=50))
    w = _mkworker(vs, pred, narr, dedup_ratio=0.85, speak_trigger=lambda: spk.append(1))
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()
    w.on_frame(_frame(1), True)  # 2回目（同一ナレーション）
    await _pump()
    await w.stop()
    return len(narr.calls) == 2 and len(spk) == 1  # 発話トリガは初回のみ


async def t_narrate_exception_safe() -> bool:
    async def boom(frames):
        raise RuntimeError("VLM down")
    vs, pred = VisionState(), PredictionState()
    w = _mkworker(vs, pred, boom)
    w.start()
    w.on_frame(_frame(0), True)
    await _pump()
    alive = w.is_idle()  # 例外後も worker は生存・idle に戻る
    # 後続も処理できる
    w._narrate = FakeNarrator(result_fn=lambda f: VisionResult(narration="復活", surprise_diff=30))
    w.on_frame(_frame(1), True)
    await _pump()
    await w.stop()
    return alive and vs.latest_vision == "復活"


async def t_blank_honest() -> bool:
    # A11: blank フレーム → VLM を呼ばず正直マーカ・surprise/発話に触れない
    spk = []
    vs, pred = VisionState(), PredictionState()
    narr = FakeNarrator()
    w = _mkworker(vs, pred, narr, speak_trigger=lambda: spk.append(1))
    w.start()
    w.on_frame(_frame(0, blank=True), True)
    await _pump()
    await w.stop()
    return (
        len(narr.calls) == 0
        and vs.latest_vision == BLANK_MARKER
        and pred.surprise == NEUTRAL_SURPRISE
        and len(spk) == 0
    )


# ===== should_speak への vision 配線（A6 非回帰 + T2 vlm）=====
async def t_vision_forwarded_when_set() -> bool:
    seen = {}
    async def fake(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None, vision=None):
        seen["vision"] = vision
        return SpeechDecision(True, "r", "c")
    d = await should_speak(surprise=50, silence_seconds=5, recent_turns=[], topic_seeds=[], decide_fn=fake, vision="画面X")
    return seen.get("vision") == "画面X" and d.speak


async def t_vision_none_legacy_fake_ok() -> bool:
    # A6: vision を受けない既存型 fake。vision=None なら転送されず壊れない（F5 非回帰の核）。
    async def legacy(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None):
        return SpeechDecision(False, "r")
    d = await should_speak(surprise=10, silence_seconds=5, recent_turns=[], topic_seeds=[], decide_fn=legacy, vision=None)
    return not d.speak


def t_should_speak_surprise_required() -> bool:
    import inspect
    sig = inspect.signature(should_speak)
    return (
        sig.parameters["surprise"].default is inspect.Parameter.empty  # surprise は必須のまま
        and sig.parameters["vision"].default is None  # vision は任意
    )


def t_build_decide_vision_block() -> bool:
    with_v = build_decide_messages(surprise=20, silence_seconds=5, recent_turns=[], topic_seeds=[], vision="ブラウザ閲覧中")
    without = build_decide_messages(surprise=20, silence_seconds=5, recent_turns=[], topic_seeds=[])
    uw, un = with_v[-1]["content"], without[-1]["content"]
    return "# 画面（今この瞬間）" in uw and "ブラウザ閲覧中" in uw and "# 画面" not in un


async def t_vision_injected_into_messages() -> bool:
    # ResponseOrchestrator が latest_vision を「# 画面（今この瞬間）」として注入する
    from eve.pipeline import AudioPlayQueue, Stimulus, StimulusKind
    from eve.response import ResponseOrchestrator
    vs = VisionState()
    vs.set_latest("メモ帳に文章を書いている")  # 時刻付き（鮮度 TTL 内）
    async def stream_fn(m):
        yield "ok"
    async def tts(s):
        return b"x"
    orch = ResponseOrchestrator(AudioPlayQueue(), stream_fn, tts, vision_state=vs)
    msgs = orch._build_messages(Stimulus(StimulusKind.USER_UTTERANCE, "やあ"))
    sysmsg = msgs[0]["content"]
    return "# 画面（今この瞬間）" in sysmsg and "メモ帳に文章を書いている" in sysmsg


async def t_t2_vlm_surprise_flips() -> bool:
    # T2 を vlm 生産者へ拡張: note_vlm_surprise→pred.surprise→should_speak が反転
    async def reader(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None, vision=None):
        return SpeechDecision(surprise >= 50, "by surprise", "c" if surprise >= 50 else "")
    pred = PredictionState()
    pred.note_vlm_surprise(80)
    hi = await should_speak(surprise=pred.surprise, silence_seconds=5, recent_turns=[], topic_seeds=[], decide_fn=reader)
    pred.note_vlm_surprise(10)
    lo = await should_speak(surprise=pred.surprise, silence_seconds=5, recent_turns=[], topic_seeds=[], decide_fn=reader)
    return hi.speak and not lo.speak


async def t_notable_guarded_trigger() -> bool:
    # A5/Q4: guard False なら notable でも発話を叩かない / True なら叩く
    res = []
    for guard in (False, True):
        spk = []
        vs, pred = VisionState(), PredictionState()
        narr = FakeNarrator(result_fn=lambda f: VisionResult(narration="変化あり", notable=True, surprise_diff=60))
        w = _mkworker(vs, pred, narr, speak_trigger=lambda: spk.append(1), speak_guard=(lambda g=guard: g))
        w.start()
        w.on_frame(_frame(0), True)
        await _pump()
        await w.stop()
        res.append(len(spk))
    return res == [0, 1]  # guard False→0回 / True→1回


# ===== capture / capture_thread（grab は注入・mss 不使用）=====
def t_capture_blank_and_valid() -> bool:
    import numpy as np

    from eve.vlm.capture import ScreenCapture
    black = np.zeros((40, 60, 3), dtype=np.uint8)
    noisy = np.random.RandomState(0).randint(0, 255, (40, 60, 3)).astype(np.uint8)
    fb, _ = ScreenCapture(grab_fn=lambda: black, blank_std_threshold=6.0).capture_one()
    fn, ph = ScreenCapture(grab_fn=lambda: noisy, blank_std_threshold=6.0).capture_one()
    return fb.blank and (not fn.blank) and len(fn.jpeg_b64) > 0 and isinstance(ph, int)


def t_capture_thread_step_delivers() -> bool:
    from eve.vlm.capture_thread import CaptureThread
    delivered: list = []
    seq = iter([(_frame(0), 0x0), (_frame(1), (1 << 64) - 1)])

    class FakeCap:
        def capture_one(self):
            return next(seq)

        def close(self):
            pass

    ct = CaptureThread(
        capture=FakeCap(), change_detector=ChangeDetector(phash_threshold=12),
        deliver=lambda f, c: delivered.append((f.frame_id, c)), loop=None,
        schedule=lambda fn, *a: fn(*a),  # 橋渡しを同期実行（call_soon_threadsafe 相当）
    )
    s1, s2 = ct._step(), ct._step()
    return s1 and s2 and delivered == [(0, True), (1, True)]


def t_capture_thread_a10_safe_stop() -> bool:
    from eve.vlm.capture_thread import CaptureThread

    class BoomCap:
        def capture_one(self):
            raise RuntimeError("mss 初期化失敗(headless)")

        def close(self):
            pass

    ct = CaptureThread(
        capture=BoomCap(), change_detector=ChangeDetector(),
        deliver=lambda f, c: None, loop=None, schedule=lambda fn, *a: fn(*a),
    )
    return ct._step() is False  # 例外を投げず安全停止シグナル（A10）


async def main() -> None:
    check("parse valid", t_parse_valid())
    check("parse garbage 安全(raise しない)", t_parse_garbage_safe())
    check("parse 全角コロン", t_parse_fullwidth_colon())
    check("A11 parse 視認不可マーカ→中身/驚き不採用", t_parse_invisible_marker())
    check("A11 parse visible:no→不可視", t_parse_visible_false_tag())
    check("parse None 安全", t_parse_none())
    check("gate 初回は必ず True", t_gate_first_always())
    check("gate 同一→False", t_gate_identical_none())
    check("gate 大変化→True", t_gate_far_changed())
    check("gate 微小変化(<閾値)→False", t_gate_small_change_below_threshold())
    check("gate periodic 強制", t_gate_periodic_forced())
    check("hamming 距離", t_hamming())
    check("ring 上限 drop-oldest", t_ring_cap_drop_oldest())
    check("snapshot 直近k枚 value-copy", t_snapshot_last_k())
    check("鮮度TTL: 新しい→返す/古い→None", t_fresh_vision_ttl())
    check("⭐層分離: 静止中はユーザ応答のみ据え置き(自発は渡さない)", t_vision_for_user_static_hold())
    check("据え置き無効: 変化後/capture停止/上限超", t_vision_for_user_invalidated())
    check("正直マーカは据え置かない", t_vision_for_user_marker_not_held())
    check("surprise cold→NEUTRAL", t_surprise_cold_neutral())
    check("surprise vlm 単独", t_surprise_vlm_only())
    check("A4 surprise most-recent-wins(maxでない)", t_surprise_most_recent_wins())
    check("surprise vlm クランプ", t_surprise_vlm_clamp())
    # VlmWorker（async）
    check("⭐backpressure: 最新ウィンドウ1回・累積なし", await t_backpressure_one_call_freshest())
    check("⭐single-flight ≤1", await t_single_flight_max_one())
    check("⭐A1 最後フレーム非取り残し(自己再トリガ)", await t_last_frame_never_stranded())
    check("⭐A1 静止中は自己再トリガしない(永久ループ死活)", await t_static_no_self_retrigger())
    check("A1 静止後の変化では必ず起きる", await t_change_after_static_retriggers())
    check("latest_vision 書込 + surprise 反映", await t_latest_vision_written())
    check("A9 min-interval で連続変化を律速", await t_min_interval_paces())
    check("dedup: 同一実況は発話再トリガしない", await t_dedup_no_retrigger())
    check("narrate 例外→no-op・worker 生存", await t_narrate_exception_safe())
    check("⭐A11 blank→VLM呼ばず正直マーカ・surprise/発話不変", await t_blank_honest())
    check("A5/Q4 notable は guard 付きで発話トリガ", await t_notable_guarded_trigger())
    # should_speak への vision 配線（A6 非回帰 + T2 vlm）
    check("vision は set 時のみ decide_fn へ転送", await t_vision_forwarded_when_set())
    check("A6 vision=None は legacy fake を壊さない", await t_vision_none_legacy_fake_ok())
    check("should_speak surprise 必須/vision 任意(signature)", t_should_speak_surprise_required())
    check("build_decide_messages の画面ブロック", t_build_decide_vision_block())
    check("T2 vlm: vlm surprise で should_speak 反転", await t_t2_vlm_surprise_flips())
    check("vision を応答文脈に注入(# 画面)", await t_vision_injected_into_messages())
    # capture / capture_thread
    check("capture: 黒→blank / 通常→有効 b64+phash", t_capture_blank_and_valid())
    check("capture_thread step: gate→deliver 橋渡し", t_capture_thread_step_delivers())
    check("A10 capture 失敗→安全停止(例外投げない)", t_capture_thread_a10_safe_stop())


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
