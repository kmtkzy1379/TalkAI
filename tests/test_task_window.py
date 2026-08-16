"""Tier-1: タスクウィンドウへの表示ルーティング（`show_on_screen` の送り先切替）。

GUI は開かない。**「集約 ON なら UI へ、OFF なら別プロセスへ落ちる」**という分岐と、
既存の台帳（`live_viewers` / `close_viewers` / 上限判定）が**無改造で動く**ことを固める。

守りたい不変条件:
  - `viewer_open_fn` 既定 None ＝ **ヘッドレスは従来どおり別プロセス**（挙動不変）
  - フックが None を返したら別プロセスへ落ちる（UI 側トグルを再登録なしで切替）
  - フックが例外を投げても表示が死なない（別プロセスへ落ちる）
  - ハンドルは `poll()`/`terminate()` だけで `store.py` の全機能が成立する
  - 上限（`WS_VIEWER_MAX_WINDOWS`）が集約モードでも効く
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.workspace import Workspace, register_viewer_capability  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


class _Handle:
    """UI 側ハンドルの最小模型（本物と同じく poll/terminate だけ）。"""

    def __init__(self) -> None:
        self.closed = False
        self.protected = False   # ユーザが操作中（UI 側の `_ItemHandle` と同じ形）

    def poll(self):
        return None if not self.closed else 0

    def terminate(self) -> None:
        self.closed = True


def _make(tmp: Path):
    ws = Workspace(tmp, quota_bytes=10 ** 9, max_write_chars=1000, max_read_chars=1000,
                   max_files=100, rescan_sec=600)
    reg = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)
    register_viewer_capability(reg, ws, max_windows=3)
    (tmp / "a.txt").write_text("あ\n" * 5, encoding="utf-8")
    (tmp / "b.txt").write_text("い\n" * 5, encoding="utf-8")
    return ws, reg


def t_default_is_unchanged() -> None:
    print("\n[1] 既定（フック未設定）＝ヘッドレスは従来どおり")
    with tempfile.TemporaryDirectory() as d:
        ws, reg = _make(Path(d))
        check("viewer_open_fn の既定は None", ws.viewer_open_fn is None)
        import inspect

        from eve.workspace import capabilities as capmod
        src = inspect.getsource(capmod.register_viewer_capability)
        check("フック未設定なら Popen 経路へ行く分岐がある",
              "viewer_open_fn is not None" in src and "subprocess.Popen" in src)


def t_routes_to_ui() -> None:
    print("\n[2] 集約 ON: UI へ渡り、台帳に載る")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ws, reg = _make(tmp)
        opened = []
        handles = []

        def open_fn(path, name):
            opened.append((Path(path).name, name))
            h = _Handle()
            handles.append(h)
            return h

        ws.viewer_open_fn = open_fn
        msg = reg.execute("show_on_screen", {"path": "a.txt"})
        check("別プロセスを起動していない（UI へ渡った）", len(opened) == 1, f"{opened}")
        check("返答がタスクウィンドウを指す", "タスクウィンドウ" in msg, msg)
        check("台帳に載る", [n for (_h, n) in ws.live_viewers()] == ["a.txt"],
              f"{ws.live_viewers()}")
        check("VLM 用の自己認識名にも出る", ws.open_window_names() == ["a.txt"])

        # ユーザが閉じた（ハンドルが閉状態になる）→ 台帳から自然に落ちる
        handles[0].closed = True
        check("ユーザが閉じたら台帳から消える", ws.live_viewers() == [], f"{ws.live_viewers()}")


def t_close_window() -> None:
    print("\n[3] close_window / 終了時のまとめ閉じが効く")
    with tempfile.TemporaryDirectory() as d:
        ws, reg = _make(Path(d))
        made = []

        def open_fn(path, name):
            h = _Handle()
            made.append(h)
            return h

        ws.viewer_open_fn = open_fn
        reg.execute("show_on_screen", {"path": "a.txt"})
        reg.execute("show_on_screen", {"path": "b.txt"})
        check("2件開いている", len(ws.live_viewers()) == 2)
        out = reg.execute("close_window", {"path": "a.txt"})
        check("指定した窓だけ閉じる", made[0].closed and not made[1].closed, out)
        check("残りは台帳に残る", [n for (_h, n) in ws.live_viewers()] == ["b.txt"])
        check("まとめて閉じられる（終了時の経路）", ws.close_viewers() == 1)
        check("全部閉じた", ws.live_viewers() == [])


def t_fallback() -> None:
    print("\n[4] 集約 OFF / 失敗時は別プロセスへ落ちる")
    with tempfile.TemporaryDirectory() as d:
        ws, reg = _make(Path(d))
        spawned = []
        import eve.workspace.capabilities as capmod

        class _FakeProc:
            def poll(self):
                return None

            def terminate(self):
                pass

        real_popen = capmod.subprocess.Popen if hasattr(capmod, "subprocess") else None

        import subprocess as _sp
        orig = _sp.Popen

        def fake_popen(*a, **k):
            spawned.append(a[0] if a else None)
            return _FakeProc()

        _sp.Popen = fake_popen  # type: ignore[assignment]
        try:
            ws.viewer_open_fn = lambda p, n: None            # 集約 OFF の合図
            reg.execute("show_on_screen", {"path": "a.txt"})
            check("None を返したら別プロセスへ落ちる", len(spawned) == 1, f"{spawned}")

            ws.close_viewers()
            spawned.clear()

            def boom(p, n):
                raise RuntimeError("UI が死んでいる")

            ws.viewer_open_fn = boom
            msg = reg.execute("show_on_screen", {"path": "b.txt"})
            check("フックが例外でも表示は死なず別プロセスへ落ちる", len(spawned) == 1, f"{msg}")
        finally:
            _sp.Popen = orig  # type: ignore[assignment]
        assert real_popen is None or True


def t_max_windows() -> None:
    print("\n[5] 上限は集約モードでも効く")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ws, reg = _make(tmp)
        for i in range(4):
            (tmp / f"f{i}.txt").write_text("x", encoding="utf-8")
        ws.viewer_open_fn = lambda p, n: _Handle()
        outs = [reg.execute("show_on_screen", {"path": f"f{i}.txt"}) for i in range(4)]
        check("3件までしか開かない（max_windows=3）", len(ws.live_viewers()) == 3,
              f"{len(ws.live_viewers())}")
        check("4件目は理由を返す", "これ以上は開けなかった" in outs[3], outs[3])
        check("close_window の案内が出る", "close_window" in outs[3])


def t_user_engaged() -> None:
    """ユーザが操作中（3クリック以上）の窓は、イブ側から閉じない（2026-08-15 指定）。

    HTML が動くようになった＝**操作の途中**が存在する。電卓を叩いている最中に
    `close_window` や機能 OFF で消えるのを構造的に止める。
    守るのはイブ側の閉鎖だけで、停止処理（`force=True`）とユーザ自身の×は通す。
    """
    print("\n[7] ユーザが操作中の窓は閉じない")
    with tempfile.TemporaryDirectory() as d:
        ws, reg = _make(Path(d))
        made = []

        def open_fn(path, name):
            h = _Handle()
            made.append(h)
            return h

        ws.viewer_open_fn = open_fn
        reg.execute("show_on_screen", {"path": "a.txt"})
        reg.execute("show_on_screen", {"path": "b.txt"})
        made[0].protected = True          # a.txt をユーザが触っている

        check("protected_names に出る", ws.protected_names() == ["a.txt"],
              f"{ws.protected_names()}")
        out = reg.execute("close_window", {"path": "a.txt"})
        check("**イブの close_window では閉じない**", not made[0].closed, out)
        check("閉じたと嘘をつかない（操作中だと伝える）", "操作中" in out, out)
        check("台帳に残る", "a.txt" in ws.open_window_names(), f"{ws.open_window_names()}")

        out = reg.execute("close_window", {})
        check("全部閉じる要求でも操作中は残る", not made[0].closed and made[1].closed, out)
        check("残した事実を報告に含める", "残して" in out or "操作中" in out, out)

        # 停止処理は force=True（プロセスごと終わるので残しても孤児になるだけ）
        check("force=True なら閉じる（停止時）", ws.close_viewers(force=True) == 1)
        check("全部閉じた", ws.live_viewers() == [])


def t_protection_is_opt_in_attr() -> None:
    """`protected` は**属性が無ければ False**（別プロセスの `Popen` を壊さない）。"""
    print("\n[8] 保護属性を持たないハンドルは従来どおり閉じる")
    with tempfile.TemporaryDirectory() as d:
        ws, reg = _make(Path(d))

        class _Bare:
            """`Popen` の最小模型（`protected` を持たない）。"""

            def __init__(self):
                self.closed = False

            def poll(self):
                return 0 if self.closed else None

            def terminate(self):
                self.closed = True

        bare = _Bare()
        ws.viewer_open_fn = lambda p, n: bare
        reg.execute("show_on_screen", {"path": "a.txt"})
        check("protected 属性が無くても例外にならない", ws.protected_names() == [])
        check("従来どおり閉じられる", ws.close_viewers() == 1 and bare.closed)


def t_shared_rules() -> None:
    """安全規律を再実装していないこと（片方だけ古くなる事故を防ぐ）。"""
    print("\n[6] 判定・上限は viewer.py と共有している")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "eve", "ui", "task_window.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    check("viewer.py から import している", "from ..workspace.viewer import" in src)
    for name in ("MAX_BYTES", "MAX_PIXELS", "TEXT_MAX_CHARS", "sniff"):
        check(f"{name} を共有している", name in src)
    check("拡張子ではなくマジックバイト判定を使う（sniff の結果で分岐）", "kind = sniff(" in src)
    check("デコード失敗を握って表示を止めない", "except Exception" in src)


if __name__ == "__main__":
    t_default_is_unchanged()
    t_routes_to_ui()
    t_close_window()
    t_fallback()
    t_max_windows()
    t_user_engaged()
    t_protection_is_opt_in_attr()
    t_shared_rules()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
