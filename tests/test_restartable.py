"""Tier-1: サイドカーの再開可能性 と CapabilityRegistry.unregister（UI ON/OFF の前提工事）。

**なぜ要るか**: UI のメニュー（停止/再開・自律発話・画面認識・タスク）は、機能を止めて
また動かす。ところが全サイドカーの `stop()` は `_stopping = True` を立てたままだったため、
**一度止めると二度と回らなかった**（`start()` は新しいタスクを作るが、その run ループが
最初の `while not self._stopping` で即座に抜ける）。

**欠陥クラスを守る**（E13 の反省: 個別の実例だけを見るテストは、同じ欠陥が別の場所に
再発した時に何も言わない）。ここでは `_stopping` を持つクラスを**実際に列挙して**
全件に対して同じ不変条件を課す。新しいサイドカーが同じ形で追加されたら自動的に検査対象に入る。
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import pkgutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


# 「停止フラグを立てる書き方」→「再開時に戻す書き方」の対応。
# **形が違うだけの同じ欠陥**を1つの規則で捕まえる（`_stopping` のブール型と、
# CaptureThread の `threading.Event` は別物に見えるが不変条件は同じ）。
STOP_FLAG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("_stopping = True", "_stopping = False"),
    ("_stop.set()", "_stop.clear()"),
    # stop() が wait を解くために立てる起床イベント。idle な時は set から cancel まで
    # await を挟まないので**立ったまま残る**（実測 2026-08-14: 再開直後に無駄な1周が走り、
    # その間 is_idle() が False で沈黙監視が発火できず初回が11秒遅れた）。
    ("_event.set()", "_event.clear()"),
)


def _sidecar_classes() -> list[type]:
    """`eve/` 配下から「start/stop を持ち、stop で停止フラグを立てるクラス」を実際に集める。"""
    import eve

    found: list[type] = []
    for mod in pkgutil.walk_packages(eve.__path__, "eve."):
        if ".ui" in mod.name or "__pycache__" in mod.name:
            continue  # UI は Qt 依存（ヘッドレスで import できない）
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue  # 重い依存（torch/mss 等）で読めないモジュールは飛ばす
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ != mod.name or not _safe_source(obj):
                continue
            if not (hasattr(obj, "start") and hasattr(obj, "stop")):
                continue
            try:
                stop_src = inspect.getsource(obj.stop)
            except (OSError, TypeError):
                continue
            if any(set_form in stop_src for set_form, _ in STOP_FLAG_PATTERNS):
                found.append(obj)
    return found


def _safe_source(obj: type) -> bool:
    try:
        inspect.getsource(obj)
        return True
    except (OSError, TypeError):
        return False


def t_all_sidecars_reset_stopping() -> None:
    print("\n[1] stop で停止フラグを立てる全クラスが start() で再開可能")
    classes = _sidecar_classes()
    check(f"対象クラスを検出できた（{len(classes)} 個）", len(classes) >= 9,
          f"{[c.__name__ for c in classes]}")
    for cls in sorted(classes, key=lambda c: c.__name__):
        start_src = inspect.getsource(cls.start)
        stop_src = inspect.getsource(cls.stop)
        for set_form, reset_form in STOP_FLAG_PATTERNS:
            if set_form not in stop_src:
                continue
            check(f"{cls.__name__}.start() が `{reset_form}` を行う",
                  reset_form in start_src, "→ 一度 stop すると二度と回らない")
        # 実行体のハンドルも戻さないと start() のガード（`is None`）に弾かれる
        handle = "_task = None" if "_task" in stop_src else "_thread = None"
        check(f"{cls.__name__}.stop() が `{handle}` を行う", handle in stop_src,
              "→ start() のガードに引っかかり再開できない")
        # single-flight 用の idle フラグ。処理中に cancel されると clear のまま固着し、
        # `is_idle()` が永久に False ＝ 二度と起動しなくなる（沈黙監視の発火条件）。
        # 「self._idle = ...」で**自分が所有している**クラスだけが対象。
        # `self._decider.is_idle()` のように他人のものを参照しているだけのクラス
        # （SilenceMonitor / ReconcileTimer）を巻き込まない。
        if "self._idle = " in inspect.getsource(cls):
            check(f"{cls.__name__}.start() が `_idle.set()` で idle に戻す",
                  "_idle.set()" in start_src, "→ drain 期限切れの cancel で固着する")


def t_restart_actually_works() -> None:
    """実際に stop → start して**もう一度動く**ことを確かめる（形だけでなく挙動）。"""
    print("\n[2] 停止→再開で実際にまた動く")

    async def run() -> None:
        from eve.task.scheduler import ReconcileTimer

        ticks = []

        class _Store:
            def due_now(self, *a, **k):
                ticks.append(1)   # tick が回った証拠（due は無いことにする）
                return False

        class _Exec:
            def is_idle(self):
                return True

            def submit(self, *a, **k):
                pass

        timer = ReconcileTimer(store=_Store(), executor=_Exec(), tick_sec=0.02)
        timer.start()
        await asyncio.sleep(0.12)
        first = len(ticks)
        check("起動して回っている", first > 0, f"{first}")

        await timer.stop()
        stopped_at = len(ticks)
        await asyncio.sleep(0.12)
        check("停止したら止まっている", len(ticks) == stopped_at,
              f"{stopped_at} → {len(ticks)}")

        timer.start()  # ← 前提工事の本体: これが効くようになった
        await asyncio.sleep(0.12)
        check("再開したらまた回る", len(ticks) > stopped_at,
              f"{stopped_at} → {len(ticks)}")
        await timer.stop()

        # 3回繰り返しても壊れない（UI のトグルは何度も押される）
        for i in range(3):
            timer.start()
            await asyncio.sleep(0.05)
            await timer.stop()
        check("ON/OFF を3往復しても壊れない", timer._task is None and timer._stopping is True)

    asyncio.run(run())


def t_speech_decider_restart() -> None:
    """自律発話 ON/OFF の実体（`SilenceMonitor` + `SpeechDecider`）。"""
    print("\n[3] 自律発話サイドカーの停止→再開")

    async def run() -> None:
        from eve.speech import SilenceMonitor, SpeechState

        fired = []

        class _Decider:
            def trigger(self, source=None):
                fired.append(source)

            def is_idle(self):
                return True

        st = SpeechState()
        st.mark_eve_activity()
        mon = SilenceMonitor(state=st, decider=_Decider(), is_busy_fn=lambda: False,
                             tick_sec=0.02, threshold_sec=0.0)
        mon.start()
        await asyncio.sleep(0.1)
        check("沈黙監視が発火している", len(fired) > 0, f"{len(fired)}")
        await mon.stop()
        n = len(fired)
        await asyncio.sleep(0.08)
        check("停止で発火が止まる", len(fired) == n, f"{n} → {len(fired)}")
        mon.start()
        await asyncio.sleep(0.1)
        check("再開で再び発火する", len(fired) > n, f"{n} → {len(fired)}")
        await mon.stop()

    asyncio.run(run())


def t_idempotent_and_zombie() -> None:
    """レッドチーム指摘（2026-08-14）で見つかった前提工事の漏れ2件。

    どちらも「UI の ON/OFF を何度も押す」ようになって初めて踏む種類の欠陥。
    """
    print("\n[5] 二重 ON とゾンビスレッド")

    # (a) マイク入力は冪等でなければならない（二重 ON でストリームが2本になる）
    import inspect

    from eve.response.input_source import MicSttInputSource

    start_src = inspect.getsource(MicSttInputSource.start)
    stop_src = inspect.getsource(MicSttInputSource.stop)
    check("マイク start が冪等（既に動いていたら何もしない）",
          "is not None" in start_src and "return" in start_src,
          "→ 二重 ON で PyAudio ストリームが2本になり古い方が孤児化する")
    check("マイク stop が _task を None に戻す", "_task = None" in stop_src)
    check("マイク stop が _ai を None に戻す", "_ai = None" in stop_src)

    # (b) capture スレッドは join 失敗時にハンドルを捨ててはいけない
    from eve.vlm.capture_thread import CaptureThread

    # docstring を除いた**実行される行だけ**を見る（説明文の中の語を拾わない）
    body = "\n".join(ln for ln in inspect.getsource(CaptureThread.stop).splitlines()
                     if ln.strip() and not ln.strip().startswith(("#", '"""', "'''"))
                     and "。" not in ln)
    check("capture stop が join 失敗を検出する", "is_alive()" in body,
          "→ 生き残ったスレッドが start() の _stop.clear() で復活し二重起動する")
    check("capture stop は join 失敗時に手放さない（早期 return）",
          body.index("is_alive()") < body.index("_thread = None"),
          "→ ハンドルを捨てると次の start() が新スレッドを作ってしまう")

    class _Zombie:
        """join しても死なないスレッドの模型。"""

        def __init__(self) -> None:
            self.joined = 0

        def join(self, timeout=None):
            self.joined += 1

        def is_alive(self):
            return True

    ct = CaptureThread.__new__(CaptureThread)
    ct._stop = __import__("threading").Event()
    ct._thread = _Zombie()
    closed = []
    ct._capture = type("C", (), {"close": lambda self: closed.append(1)})()
    ct.stop(timeout=0.01)
    check("join 失敗時はハンドルを保持する", ct._thread is not None)
    check("join 失敗時は capture を閉じない（生存スレッドの下で mss を壊さない）",
          closed == [], f"{closed}")


def t_unregister() -> None:
    print("\n[4] CapabilityRegistry.unregister")
    from eve.capability import CapabilityRegistry
    from eve.capability.registry import Capability

    reg = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)
    before = set(reg.names())
    reg.register(Capability(name="dummy_tool", description="ダミー。テスト用。",
                            params_schema={}, handler=lambda a: "ok",
                            offered=True, agent_tool=True))
    bundle = set(reg.names()) - before
    check("登録前後の差分で束を作れる", bundle == {"dummy_tool"}, f"{bundle}")
    check("登録すると tool_schemas に出る",
          any(s["function"]["name"] == "dummy_tool" for s in reg.tool_schemas()))
    check("登録すると outward_actions に出る",
          any("ダミー" in a for a in reg.outward_actions()), f"{reg.outward_actions()}")

    removed = reg.unregister(*bundle)
    check("取り外した名前が返る", removed == ["dummy_tool"], f"{removed}")
    check("has が False になる", not reg.has("dummy_tool"))
    check("tool_schemas から消える",
          all(s["function"]["name"] != "dummy_tool" for s in reg.tool_schemas()))
    check("agent_tool_schemas から消える",
          all(s["function"]["name"] != "dummy_tool" for s in reg.agent_tool_schemas()))
    check("**outward_actions からも自動で消える（二重管理ゼロ）**",
          not any("ダミー" in a for a in reg.outward_actions()), f"{reg.outward_actions()}")
    check("execute は未対応マーカを返す（例外を投げない）",
          "未対応" in reg.execute("dummy_tool"), reg.execute("dummy_tool"))
    check("未登録名を外しても落ちない", reg.unregister("nonexistent") == [])
    check("残りの能力は無傷", set(reg.names()) == before, f"{set(reg.names()) ^ before}")

    # 束で外す（タスク+検索+作業フォルダをまとめて外す本番の使い方）
    for n in ("a_tool", "b_tool"):
        reg.register(Capability(name=n, description=f"{n} のテスト。", params_schema={},
                                handler=lambda a: "ok", agent_tool=True))
    check("複数まとめて外せる", sorted(reg.unregister("a_tool", "b_tool")) == ["a_tool", "b_tool"])


if __name__ == "__main__":
    t_all_sidecars_reset_stopping()
    t_restart_actually_works()
    t_speech_decider_restart()
    t_idempotent_and_zombie()
    t_unregister()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
