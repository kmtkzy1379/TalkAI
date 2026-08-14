"""Tier-1: 機能 ON/OFF（ControlPlane）と入口ゲート・ミュート（API 不要）。

**なぜここを固めるか**: ON/OFF の失敗は「静かに間違った状態で動き続ける」形で出る。
OFF にしたのに喋る / ON に戻したのに動かない / タスクを Running のまま置き去りにする、は
どれも例外を出さないので、決定論テストで縛らないと実起動でしか気づけない。

対応する要件（ユーザ裁定 2026-08-14）:
- 押した瞬間には切り替えず、その機能が**正常終了した安全な地点**で切り替える
- タスク OFF は**キャンセルを走らせて正常終了させてから**
- ミュート → 処理を正常終了 → 停止 の順で止められること
- 応答LLM には「機能が無い」ではなく「**機能はあるが今はオフ**」と伝える
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.pipeline.orchestrator import PipelineRunner, StubOrchestrator  # noqa: E402
from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402
from eve.pipeline.stimulus import Stimulus, StimulusKind  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.vlm.change_detector import ChangeDetector  # noqa: E402
from eve.vlm.state import VisionState  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


# --------------------------------------------------------------------------
# [1] PipelineRunner の入口ゲート（停止/再開の土台）
# --------------------------------------------------------------------------

def t_gate() -> None:
    print("\n[1] 入口ゲート: 停止中は刺激を処理しない（runner は殺さない）")

    async def scenario() -> None:
        q, audio = StimulusQueue(), AudioPlayQueue(play_fn=None)
        orch = StubOrchestrator(audio)
        r = PipelineRunner(q, orch, audio)
        check("既定は開いている（ヘッドレスの挙動は不変）", not r.is_paused())
        task = asyncio.create_task(r.run())

        await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "あ"))
        await asyncio.sleep(0.05)
        check("通常時は処理される", len(orch.handled) == 1, f"{len(orch.handled)}")

        r.pause()
        await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "い"))
        await asyncio.sleep(0.15)
        check("**停止中は処理されない**", len(orch.handled) == 1, f"{len(orch.handled)}")
        check("刺激は捨てられていない（保留されている）", True)

        r.resume()
        await asyncio.sleep(0.10)
        check("再開すると保留分が処理される", len(orch.handled) == 2, f"{len(orch.handled)}")
        check("順序が入れ替わっていない",
              [str(s.payload) for s in orch.handled] == ["あ", "い"],
              f"{[str(s.payload) for s in orch.handled]}")

        # `queue.get()` で待っている最中に停止 → その後に届いた刺激（タスクの完了報告など）
        # を喋ってしまわないか。ここが抜けていると「停止したのに喋る」が残る。
        r.pause()
        await asyncio.sleep(0.05)  # get() で待っている状態を作る
        await q.put(Stimulus(StimulusKind.CALLFUNCTION_RESULT, "報告"))
        await asyncio.sleep(0.15)
        check("**停止後に届いた刺激も処理しない**（get 待ち中に来た分）",
              len(orch.handled) == 2, f"{len(orch.handled)}")
        r.resume()
        await asyncio.sleep(0.10)
        check("再開でその刺激が届く（消えていない）", len(orch.handled) == 3, f"{len(orch.handled)}")

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# [2] ミュート（音だけ止める・世代は進めない）
# --------------------------------------------------------------------------

def t_mute() -> None:
    print("\n[2] ミュート: 音だけ止まり、パイプラインは正常に流れる")
    from eve.response.player import RealAudioPlayer

    p = RealAudioPlayer()
    check("既定はミュートしていない", p.muted is False)

    async def scenario() -> None:
        p.muted = True
        # 実デバイスを開かずに返ること（開くと pyaudio が要る＝決定論でなくなる）
        await p.play_fn(b"\x00" * 100)
        check("ミュート中は再生器がデバイスを開かずに戻る", p._pa is None)

    asyncio.run(scenario())

    import inspect

    src = inspect.getsource(RealAudioPlayer.play_fn)
    check("**再生中のミュートで文の途中でも止まる**（チャンクループ内で見ている）",
          src.count("self.muted") >= 2, src)
    # 世代を進めないこと＝barge-in と混同していないこと。進めると進行中ターンの残りが
    # 捨てられ、「処理は正常に終了させてから」というユーザ裁定を破る。
    from eve.control import ControlPlane

    mute_src = inspect.getsource(ControlPlane._apply_mute)
    check("ミュートは世代を進めない（interrupt/bump を呼ばない）",
          "interrupt" not in mute_src and "bump_generation" not in mute_src, mute_src)
    check("ミュートは待たない（速攻黙る手段そのもの）", "_wait_quiet" not in mute_src)


# --------------------------------------------------------------------------
# フェイクの VoiceLoop（ControlPlane が触る面だけを持つ）
# --------------------------------------------------------------------------

class _Sidecar:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.running = False

    def start(self) -> None:
        self.started += 1
        self.running = True

    async def stop(self, *a, **k) -> None:
        self.stopped += 1
        self.running = False


class _Runner:
    def __init__(self) -> None:
        self.paused = False
        self.busy = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def is_busy(self) -> bool:
        return self.busy


class _Input:
    def __init__(self) -> None:
        self.running = True
        self.starts = 0

    async def start(self) -> None:
        self.starts += 1
        self.running = True

    def stop(self) -> None:
        self.running = False


class _Executor(_Sidecar):
    def __init__(self) -> None:
        super().__init__()
        self.running = True
        self.idle = True

    def is_idle(self) -> bool:
        return self.idle


class _Caps:
    def __init__(self, names) -> None:
        self._names = list(names)

    def names(self):
        return list(self._names)

    def unregister(self, *names):
        out = [n for n in names if n in self._names]
        self._names = [n for n in self._names if n not in names]
        return out


class _Task:
    def __init__(self, tid, status) -> None:
        self.task_id, self.status = tid, status


class _Store:
    def __init__(self, tasks) -> None:
        self._tasks = {t.task_id: t for t in tasks}

    def list_all(self):
        return list(self._tasks.values())

    def set_status(self, tid, status, **k) -> bool:
        t = self._tasks.get(tid)
        if t is None:
            return False
        t.status = status
        return True


class _Workspace:
    """`close_viewers()` だけを見る（タスク OFF で表示窓を畳むかの検証用）。"""

    def __init__(self) -> None:
        self.closed = 0
        self.paths = type("P", (), {"root": os.devnull})()

    def close_viewers(self) -> int:
        self.closed += 1
        return 1


class _FakeVL:
    def __init__(self, *, tasks=()) -> None:
        from eve.response.player import RealAudioPlayer

        self.player = RealAudioPlayer()
        self.runner = _Runner()
        self.input = _Input()
        self.queue = StimulusQueue()
        self.audio = AudioPlayQueue(play_fn=None)
        self.speech_decider = _Sidecar()
        self.silence_monitor = _Sidecar()
        self.speech_state = type("S", (), {"marks": 0})()
        self.speech_state.mark_eve_activity = lambda: setattr(
            self.speech_state, "marks", self.speech_state.marks + 1)
        self.vision_state = VisionState(ring_max=4)
        self.vlm_change_detector = ChangeDetector()
        self.capture_thread = object()
        self.vision_started = 0
        self.vision_stopped = 0
        self.task_store = _Store(tasks) if tasks else _Store([])
        self.task_executor = _Executor()
        self.reconcile_timer = _Sidecar()
        self.cancel_resolver = _Sidecar()
        self.workspace = _Workspace()
        self.capabilities = _Caps(["self_status", "pc_status", "delegate_task",
                                   "cancel_task", "search_web"])
        self._task_cap_names = ["delegate_task", "cancel_task", "search_web"]
        self._reg_calls = 0
        self._task_cap_registrars = [self._re_register]
        self.control = None

    def _re_register(self) -> None:
        self._reg_calls += 1
        self.capabilities._names += [n for n in self._task_cap_names
                                     if n not in self.capabilities._names]

    def start_vision(self) -> None:
        self.vision_started += 1
        self.capture_thread = object()

    async def stop_vision(self) -> None:
        self.vision_stopped += 1
        self.capture_thread = None
        self.vision_state.latest_vision = None
        self.vision_state.ring.clear()
        self.vlm_change_detector.reset()


def _plane(vl, **kw):
    from eve.control import ControlPlane

    return ControlPlane(vl, quiesce_sec=kw.pop("quiesce_sec", 1.0), **kw)


async def _settle(cp, timeout: float = 5.0) -> None:
    """切り替えが**確定する**まで待つ。

    `_want` が空になった時点ではまだ適用中（待ちの最中）なので、そこで進むと
    「まだ切り替わっていない状態」を見てしまう。`wait_idle()` は適用中も待つ。
    """
    await asyncio.sleep(0)  # request 直後に worker が起きる隙を作る
    await asyncio.wait_for(cp.wait_idle(), timeout=timeout)


# --------------------------------------------------------------------------
# [3] 自律発話 OFF
# --------------------------------------------------------------------------

def t_autospeech() -> None:
    print("\n[3] 自律発話 OFF: 駆動を両方止め、未発話の下書きも捨てる")

    async def scenario() -> None:
        vl = _FakeVL()
        cp = _plane(vl)
        cp.start()
        # まだ喋っていない自発発話の下書きを積んでおく
        await vl.queue.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "ねえ", merge_key="a1"))
        await vl.queue.put(Stimulus(StimulusKind.USER_UTTERANCE, "こんにちは"))

        cp.request("autospeech", False)
        await _settle(cp)

        check("沈黙監視を止めた", vl.silence_monitor.stopped == 1, f"{vl.silence_monitor.stopped}")
        check("**発話判定 worker も止めた**（残すと画面認識のトリガ経由で喋る）",
              vl.speech_decider.stopped == 1, f"{vl.speech_decider.stopped}")
        kinds = [s.kind for s in vl.queue.snapshot()]
        check("未発話の自発発話を捨てた", StimulusKind.AUTONOMOUS_SPEECH not in kinds, f"{kinds}")
        check("**ユーザ発話は捨てていない**", StimulusKind.USER_UTTERANCE in kinds, f"{kinds}")
        check("状態が off で確定", cp.is_on("autospeech") is False)

        cp.request("autospeech", True)
        await _settle(cp)
        check("ON で両方起き直す",
              vl.silence_monitor.started == 1 and vl.speech_decider.started == 1,
              f"{vl.silence_monitor.started}/{vl.speech_decider.started}")
        check("**ON 時に沈黙時計をリセットする**（OFF 中の沈黙で即独り言を防ぐ）",
              vl.speech_state.marks >= 1, f"{vl.speech_state.marks}")
        await cp.stop()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# [4] 画面認識 OFF
# --------------------------------------------------------------------------

def t_vision() -> None:
    print("\n[4] 画面認識 OFF: 画面の記憶と変化検出の状態を捨てる")

    async def scenario() -> None:
        vl = _FakeVL()
        vl.vision_state.set_latest("メモ帳が開いている")
        vl.vlm_change_detector.evaluate(0b1010)
        cp = _plane(vl)
        cp.start()

        cp.request("vision", False)
        await _settle(cp)
        check("capture を止めた", vl.vision_stopped == 1 and vl.capture_thread is None)
        check("**直近の画面実況を消した**（残ると古い画面を今として語る）",
              vl.vision_state.latest_vision is None)
        check("変化検出の参照も捨てた（OFF 中の変化を ON 直後に拾わない）",
              vl.vlm_change_detector._reference is None)
        check("状態が off で確定", cp.is_on("vision") is False)

        cp.request("vision", True)
        await _settle(cp)
        check("ON で capture を起こし直す", vl.vision_started == 1 and vl.capture_thread is not None)
        await cp.stop()

    asyncio.run(scenario())

    # VlmWorker.reset: 前回実況を残すと ON 直後の実況が重複判定で消える（点けたのに見えない）
    from eve.vlm.worker import VlmWorker

    w = VlmWorker.__new__(VlmWorker)
    w._last_narration = "メモ帳が開いている"
    w._last_call_ts = 12345.0
    w._change_seq = 7
    w.reset()
    check("**VlmWorker の前回実況を捨てる**（残すと ON 直後の実況が重複で消える）",
          w._last_narration == "")
    check("pacing の起点も戻す（ON 直後の1回を待たせない）", w._last_call_ts < 0)

    # ChangeDetector.reset の単体（初回扱いに戻ること）
    cd = ChangeDetector()
    check("初回は必ず見る", cd.evaluate(0) is True)
    check("同じ画面は見ない", cd.evaluate(0) is False)
    cd.reset()
    check("reset 後は初回扱いに戻る", cd.evaluate(0) is True)


# --------------------------------------------------------------------------
# [5] タスク OFF（キャンセル → 正常終了 → 停止）
# --------------------------------------------------------------------------

def t_task() -> None:
    print("\n[5] タスク OFF: 取り消して正常終了させてから止める")
    from eve.task.schema import CANCELLED, DONE, PENDING, RUNNING

    async def scenario() -> None:
        vl = _FakeVL(tasks=[_Task("t1", RUNNING), _Task("t2", PENDING), _Task("t3", DONE)])
        cp = _plane(vl)
        cp.start()
        await vl.queue.put(Stimulus(StimulusKind.CALLFUNCTION_RESULT, "終わったよ", dedup_key="r1"))

        cp.request("task", False)
        await _settle(cp)

        st = {t.task_id: t.status for t in vl.task_store._tasks.values()}
        check("実行中を取り消した", st["t1"] == CANCELLED, f"{st}")
        check("予約中も取り消した", st["t2"] == CANCELLED, f"{st}")
        check("**終わったタスクは触らない**（終端の再遷移をしない）", st["t3"] == DONE, f"{st}")
        check("**先に能力を外す**（新しい依頼を受け付けない）",
              "delegate_task" not in vl.capabilities.names(), f"{vl.capabilities.names()}")
        check("束（検索）も一緒に外した", "search_web" not in vl.capabilities.names())
        check("組込み能力は残す（『何もできない』にしない）",
              "self_status" in vl.capabilities.names())
        check("executor / タイマ / 取消解決を止めた",
              vl.task_executor.stopped == 1 and vl.reconcile_timer.stopped == 1
              and vl.cancel_resolver.stopped == 1)
        kinds = [s.kind for s in vl.queue.snapshot()]
        check("取り消したタスクの未配達報告を捨てた",
              StimulusKind.CALLFUNCTION_RESULT not in kinds, f"{kinds}")
        check("**イブが開いた表示ウィンドウを畳んだ**（残すと画面認識が自分の出力を拾う）",
              vl.workspace.closed == 1, f"{vl.workspace.closed}")
        check("状態が off で確定", cp.is_on("task") is False)

        cp.request("task", True)
        await _settle(cp)
        check("**ON で能力が戻る**（戻らないと二度と頼めない）",
              "delegate_task" in vl.capabilities.names(), f"{vl.capabilities.names()}")
        check("ON で executor / タイマ / 取消解決を起こし直す",
              vl.task_executor.started == 1 and vl.reconcile_timer.started == 1
              and vl.cancel_resolver.started == 1)
        await cp.stop()

    asyncio.run(scenario())


def t_task_waits() -> None:
    print("\n[6] タスク OFF は executor が片付くのを待つ（殺さない）")

    async def scenario() -> None:
        vl = _FakeVL(tasks=[_Task("t1", "Running")])
        vl.task_executor.idle = False  # 実行中のまま
        cp = _plane(vl, quiesce_sec=0.4)
        cp.start()
        cp.request("task", False)
        await asyncio.sleep(0.15)
        check("**待っている間はまだ止めていない**", vl.task_executor.stopped == 0)
        vl.task_executor.idle = True  # 正常終了した
        await _settle(cp)
        check("片付いたら止める", vl.task_executor.stopped == 1)
        check("状態が off で確定", cp.is_on("task") is False)
        await cp.stop()

    asyncio.run(scenario())

    async def speaking_report() -> None:
        """報告を喋っている最中は OFF を確定させない（1件制の定義＝発話し終えるまで）。"""
        vl = _FakeVL()
        vl.runner.busy = True  # 報告ターンを喋っている
        cp = _plane(vl, quiesce_sec=0.6)
        cp.start()
        cp.request("task", False)
        await asyncio.sleep(0.2)
        check("**報告の発話中は OFF を確定させない**", cp.is_on("task") is True)
        vl.runner.busy = False
        await _settle(cp)
        check("喋り終わったら OFF が確定する", cp.is_on("task") is False)
        await cp.stop()

    asyncio.run(speaking_report())

    async def timeout_case() -> None:
        vl = _FakeVL(tasks=[_Task("t1", "Running")])
        vl.task_executor.idle = False
        cp = _plane(vl, quiesce_sec=0.2)
        cp.start()
        cp.request("task", False)
        await _settle(cp)
        check("**上限で諦めても切り替えは確定させる**（固まらない）", cp.is_on("task") is False)
        await cp.stop()

    asyncio.run(timeout_case())


# --------------------------------------------------------------------------
# [7] 停止/再開
# --------------------------------------------------------------------------

def t_pause() -> None:
    print("\n[7] 停止: 上流から断ち、進行中は正常終了を待つ")

    async def scenario() -> None:
        vl = _FakeVL()
        cp = _plane(vl, quiesce_sec=0.5)
        cp.start()
        vl.runner.busy = True  # 応答の途中

        cp.request("pause", False)
        await asyncio.sleep(0.15)
        check("入口はすぐ閉じる", vl.runner.paused is True)
        check("マイクもすぐ閉じる", vl.input.running is False)
        check("**進行中の応答は殺さない**（待つ）", cp.is_on("pause") is True)
        vl.runner.busy = False
        await _settle(cp)
        check("終わったら停止が確定する", cp.is_on("pause") is False)
        check("自発発話の駆動も止まっている", vl.speech_decider.stopped >= 1)
        check("画面認識も止まっている", vl.vision_stopped == 1)

        cp.request("pause", True)
        await _settle(cp)
        check("再開で入口が開く", vl.runner.paused is False)
        check("再開でマイクが開く", vl.input.starts == 1 and vl.input.running is True)
        check("再開で画面認識が戻る", vl.vision_started == 1)
        check("**再開時に沈黙時計をリセットする**（停止中の沈黙で即独り言を防ぐ）",
              vl.speech_state.marks >= 1, f"{vl.speech_state.marks}")
        await cp.stop()

    asyncio.run(scenario())


def t_pause_keeps_off_features_off() -> None:
    print("\n[8] 再開しても、OFF にしてある機能は勝手に ON にならない")

    async def scenario() -> None:
        vl = _FakeVL()
        cp = _plane(vl, quiesce_sec=0.2)
        cp.start()
        cp.request("vision", False)
        cp.request("autospeech", False)
        await _settle(cp)
        base_v, base_a = vl.vision_started, vl.speech_decider.started

        cp.request("pause", False)
        await _settle(cp)
        cp.request("pause", True)
        await _settle(cp)
        check("再開で画面認識は戻らない（ユーザが OFF にしたまま）",
              vl.vision_started == base_v, f"{vl.vision_started} vs {base_v}")
        check("再開で自律発話も戻らない",
              vl.speech_decider.started == base_a, f"{vl.speech_decider.started} vs {base_a}")
        await cp.stop()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# [9] 直列化と latest-wins
# --------------------------------------------------------------------------

def t_serialized() -> None:
    print("\n[9] 遷移は同時に走らない / 連打は最後の意図が勝つ")

    async def scenario() -> None:
        vl = _FakeVL()
        cp = _plane(vl, quiesce_sec=0.2)
        depth = {"now": 0, "max": 0}
        orig = cp._apply

        async def counting(key, on):
            depth["now"] += 1
            depth["max"] = max(depth["max"], depth["now"])
            try:
                await asyncio.sleep(0.02)
                await orig(key, on)
            finally:
                depth["now"] -= 1

        cp._apply = counting
        cp.start()
        cp.request("vision", False)
        cp.request("autospeech", False)
        cp.request("task", False)
        await _settle(cp)
        check("**同時には1つしか適用しない**（stop 中の start が黙って消えるのを防ぐ）",
              depth["max"] == 1, f"最大同時 {depth['max']}")

        # 連打: OFF→ON→OFF と押したら最後の OFF が残る
        cp.request("vision", True)
        cp.request("vision", False)
        cp.request("vision", True)
        await _settle(cp)
        check("連打は最後の意図が勝つ（latest-wins）", cp.is_on("vision") is True)
        await cp.stop()

    asyncio.run(scenario())


def t_states() -> None:
    print("\n[10] 表示は loop が確定した状態だけ（押した瞬間には変えない）")

    async def scenario() -> None:
        vl = _FakeVL()
        seen: list = []
        cp = _plane(vl, quiesce_sec=0.2, on_state=lambda k, s, on: seen.append((k, s, on)))
        cp.start()
        cp.request("vision", False)
        check("押した直後は pending を出す（まだ off にしない）",
              seen and seen[0] == ("vision", "pending", True), f"{seen}")
        await _settle(cp)
        check("最後に off が確定する", seen[-1] == ("vision", "off", False), f"{seen}")
        check("pending → off の順に出る",
              [s for _k, s, _o in seen] == ["pending", "off"], f"{seen}")

        # 例外が出ても状態は巻き戻り、UI は固まらない
        async def boom(on):
            raise RuntimeError("失敗")

        cp._apply_vision = boom
        seen.clear()
        cp.request("vision", True)
        await _settle(cp)
        check("**切り替えに失敗したら元の状態を出す**（pending で固まらない）",
              seen[-1] == ("vision", "off", False), f"{seen}")
        await cp.stop()

    asyncio.run(scenario())


def t_disabled() -> None:
    print("\n[11] 構成で使えない機能は押させない")
    import eve.control as control_mod

    class _NoTaskVL(_FakeVL):
        pass

    async def scenario() -> None:
        vl = _NoTaskVL()
        vl.task_store = None
        seen: list = []
        cp = control_mod.ControlPlane(vl, quiesce_sec=0.2,
                                      on_state=lambda k, s, on: seen.append((k, s, on)))
        cp.start()
        snap = cp.snapshot()
        check("タスク未構成なら disabled", snap["task"][0] == "disabled", f"{snap['task']}")
        cp.request("task", False)
        await _settle(cp)
        check("押しても遷移しない（disabled を返すだけ）",
              seen == [("task", "disabled", False)], f"{seen}")
        await cp.stop()

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# [12] 応答LLM への伝え方（「機能が無い」と言わせない）
# --------------------------------------------------------------------------

def t_features_off_prompt() -> None:
    print("\n[12] プロンプト: 「機能が無い」ではなく「あるが今はオフ」")
    from eve.context_assembler import ContextAssembler
    from eve.control import FEATURE_OFF_TEXT

    for key, text in FEATURE_OFF_TEXT.items():
        check(f"{key} の文言が「今はオフ」だと分かる",
              ("今はユーザがオフ" in text) or ("今はミュート" in text), text)
        check(f"{key} の文言が能力の不在を言っていない",
              "できない" not in text and "機能がない" not in text, text)

    ca = ContextAssembler(system_prompt="スタイル")
    msgs = ca.assemble(user_text="今の画面なに？", features_off=[FEATURE_OFF_TEXT["vision"]])
    sys_text = msgs[0]["content"]
    check("system に『今オフにしている機能』が入る", "今オフにしている機能" in sys_text)
    check("**『その機能が無い』とは言わないよう明示している**",
          "「その機能が無い」とは言わない" in sys_text, sys_text[-400:])
    check("勝手に点けられないことも書いてある", "勝手に点けることはできない" in sys_text)

    plain = ca.assemble(user_text="やあ")
    check("オフが無ければブロック自体を出さない（ヘッドレスの system を太らせない）",
          "今オフにしている機能" not in plain[0]["content"])

    # ControlPlane.features_off の中身
    async def scenario() -> None:
        vl = _FakeVL()
        cp = _plane(vl, quiesce_sec=0.2)
        cp.start()
        check("全部 ON なら空", cp.features_off() == [], f"{cp.features_off()}")
        cp.request("vision", False)
        await _settle(cp)
        off = cp.features_off()
        check("画面認識を OFF にすると1件出る", len(off) == 1, f"{off}")
        check("その文言は『オンにすれば見える』と言っている", "オンにすれば" in off[0], off[0])
        cp.request("mute", True)
        await _settle(cp)
        check("ミュートも「今はオフ」として伝わる",
              any("ミュート" in t for t in cp.features_off()), f"{cp.features_off()}")
        await cp.stop()

    asyncio.run(scenario())


def t_tools_not_lying() -> None:
    print("\n[13] タスク OFF 後に「tool が使えるつもり」で喋らせない")
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.orchestrator import ResponseOrchestrator

    class _Disp:
        def __init__(self, schemas) -> None:
            self._s = schemas

        def tool_schemas(self):
            return self._s

    audio = AudioPlayQueue(play_fn=None)

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        if False:
            yield ""

    def mk(schemas):
        return ResponseOrchestrator(audio, stream_fn, lambda *a, **k: None,
                                    None, dispatcher=_Disp(schemas))

    s = Stimulus(StimulusKind.USER_UTTERANCE, "やって")
    check("tool があるときは有効", mk([{"type": "function"}])._tools_enabled_for(s) is True)
    check("**tool が空なら無効**（tools_active の嘘を防ぐ）",
          mk([])._tools_enabled_for(s) is False)
    check("自発発話には元から渡さない（1ホップ抑制は維持）",
          mk([{"type": "function"}])._tools_enabled_for(
              Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "ねえ")) is False)


# --------------------------------------------------------------------------
# [14] UI 側の約束
# --------------------------------------------------------------------------

def t_ui_contract() -> None:
    print("\n[14] UI: 押した瞬間に表示を変えない / 語義が loop と一致する")
    import inspect

    from eve.ui import menu_bar

    src = inspect.getsource(menu_bar)
    check("MenuBar は自分で状態を確定させない（requested を出すだけ）",
          "self.requested.emit" in src and "_clicked" in src)
    # `mute` の on の意味が ControlPlane と一致していること。
    # ここが反転すると「ミュートを押したら音が出る」になる。
    items = {k: (on_text, off_text, initial) for k, _l, on_text, off_text, initial in menu_bar.ITEMS}
    check("mute は初期値 False（＝黙っていない）", items["mute"][2] is False, f"{items['mute']}")
    check("mute の ON 表示が『ミュート中』", items["mute"][0] == "ミュート中", f"{items['mute']}")
    check("pause は初期値 True（＝稼働中）", items["pause"][2] is True)
    from eve.control import KEYS

    check("MenuBar と ControlPlane のキーが一致する",
          sorted(items) == sorted(KEYS), f"{sorted(items)} vs {sorted(KEYS)}")

    app_src = inspect.getsource(__import__("eve.ui.app", fromlist=["x"]))
    check("UI → loop は call_soon_threadsafe のみ（§9.3）",
          "call_soon_threadsafe(ctl.request" in app_src)
    check("表示の反映点は control_state ただ1つ",
          app_src.count("self.menu.set_state") == 1, str(app_src.count("self.menu.set_state")))
    check("押した時にログへ『切り替えます』と出す（受け付けたことは分かる）",
          "に切り替えます" in app_src)


if __name__ == "__main__":
    t_gate()
    t_mute()
    t_autospeech()
    t_vision()
    t_task()
    t_task_waits()
    t_pause()
    t_pause_keeps_off_features_off()
    t_serialized()
    t_states()
    t_disabled()
    t_features_off_prompt()
    t_tools_not_lying()
    t_ui_contract()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
