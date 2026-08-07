"""起動時／終了時の共通掃除の決定論テスト（API 不要）。

検証（ユーザ裁定 2026-08-08「どんな終わり方をしても次の起動はクリーン」）:
- 未終了タスクが terminal になる。Running→Failed（やりかけ）/ Pending→Cancelled（やっていない）
  と**意味を分ける**（後で履歴を読んだとき「落ちた」と「やらなかった」を混同しないため）。
- 既に terminal のタスクは触らない（terminal-stays-terminal を壊さない）。
- `.eve_run` の実行残骸だけを消し、**ユーザ/AI のファイルは消さない**。
- 掃除の失敗で起動/終了を止めない（例外を上げない）。
- 起動時のみビューア掃除を試み、shutdown では試みない。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_cleanup.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve import cleanup  # noqa: E402
from eve.task import CANCELLED, DONE, FAILED, PENDING, RUNNING, Task, TaskStore  # noqa: E402

_passed = 0
_failed = 0
_dir = tempfile.mkdtemp(prefix="eve_cleanup_")
_n = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


def _tmp():
    global _n
    _n += 1
    return os.path.join(_dir, f"t{_n}.jsonl")


class _FakeWs:
    """`workspace.paths.root` だけを持つ最小スタブ。"""

    def __init__(self, root):
        self.paths = type("P", (), {"root": root})()


async def t_tasks_swept():
    s = TaskStore(task_file=_tmp())
    await s.initialize()
    s.add(Task(task_id="run", what="", goal="実行中だった", when=None))
    s.add(Task(task_id="pend", what="", goal="未着手だった", when=None))
    s.add(Task(task_id="done", what="", goal="終わってた", when=None))
    s.set_status("run", RUNNING)
    s.set_status("done", DONE, result="x")
    n = cleanup.sweep_tasks(s, phase="startup")
    ok = (n == 2
          and s.get("run").status == FAILED        # やりかけ
          and s.get("pend").status == CANCELLED    # やっていない
          and s.get("done").status == DONE         # terminal は触らない
          and "起動時の掃除" in (s.get("run").failure_reason or ""))
    # 2回目は何も残っていない（冪等）。
    again = cleanup.sweep_tasks(s, phase="startup")
    await s.shutdown()
    return ok and again == 0


async def t_shutdown_reason_differs():
    s = TaskStore(task_file=_tmp())
    await s.initialize()
    s.add(Task(task_id="p", what="", goal="g", when=None))
    cleanup.sweep_tasks(s, phase="shutdown")
    ok = "終了時の掃除" in (s.get("p").failure_reason or "")
    await s.shutdown()
    return ok


def t_workspace_residue_only():
    root = Path(_dir) / "ws"
    (root / ".eve_run" / "abc123").mkdir(parents=True, exist_ok=True)
    (root / ".eve_run" / "abc123" / "job.json").write_text("{}", encoding="utf-8")
    (root / ".eve_run" / "loose.tmp").write_text("x", encoding="utf-8")
    (root / "大事なメモ.txt").write_text("消してはいけない", encoding="utf-8")
    n = cleanup.sweep_workspace(_FakeWs(str(root)), phase="startup")
    return (n == 2
            and not (root / ".eve_run" / "abc123").exists()
            and not (root / ".eve_run" / "loose.tmp").exists()
            and (root / ".eve_run").is_dir()          # 置き場自体は残す
            and (root / "大事なメモ.txt").exists())    # AI/ユーザのファイルは消さない


def t_missing_paths_safe():
    """作業フォルダが無い／store が None でも例外を出さない。"""
    a = cleanup.sweep_workspace(_FakeWs(os.path.join(_dir, "存在しない")), phase="startup")
    b = cleanup.sweep_tasks(None, phase="startup")
    c = cleanup.sweep(task_store=None, workspace=None, phase="startup")
    return a == 0 and b == 0 and c == {"tasks": 0, "workspace": 0, "viewers": 0}


def t_broken_store_does_not_raise():
    """掃除の失敗で起動/終了を止めない（例外を握って続行）。"""
    class _Boom:
        def list_all(self):
            raise RuntimeError("store が壊れている")

    out = cleanup.sweep(task_store=_Boom(), workspace=None, phase="shutdown")
    return out["tasks"] == 0  # 例外が外に出ない


def t_viewer_sweep_startup_only():
    """ビューア掃除は startup のみ（shutdown は自分の Popen を持つ close_viewers が担当）。"""
    calls = []
    orig = cleanup.sweep_orphan_viewers
    cleanup.sweep_orphan_viewers = lambda: (calls.append(1), 3)[1]
    try:
        up = cleanup.sweep(task_store=None, workspace=None, phase="startup")
        down = cleanup.sweep(task_store=None, workspace=None, phase="shutdown")
    finally:
        cleanup.sweep_orphan_viewers = orig
    return len(calls) == 1 and up["viewers"] == 3 and down["viewers"] == 0


async def t_wiring_startup_before_executor():
    """★配線ガード: 起動時の掃除は **executor 起動より前**に呼ばれること。

    後だと reconcile が前回の Pending を掴んで走り出し、1件制では起動直後から
    ユーザの最初の依頼が断られる（この順序が壊れても掃除自体は成功するので、
    順序をテストで固定しないと静かに退行する）。
    """
    import inspect

    from eve import voice_loop

    src = inspect.getsource(voice_loop.VoiceLoop.run)
    i_sweep = src.find('phase="startup"')
    i_exec = src.find("self.task_executor.start()")
    i_stop = src.find('phase="shutdown"')
    return (i_sweep != -1 and i_exec != -1 and i_sweep < i_exec
            and i_stop == -1)  # shutdown 側は stop() にあり run() には無い


async def t_wiring_shutdown_after_executor_stop():
    """★配線ガード: 終了時の掃除は executor を止めた後・store を閉じる前。"""
    import inspect

    from eve import voice_loop

    src = inspect.getsource(voice_loop.VoiceLoop.stop)
    i_exec = src.find("self.task_executor.stop()")
    i_sweep = src.find('phase="shutdown"')
    i_close = src.find("self.task_store.shutdown()")
    return -1 < i_exec < i_sweep < i_close


async def main():
    check("未終了タスクを terminal 化（Running=Failed/Pending=Cancelled・冪等）", await t_tasks_swept())
    check("理由に phase が入る（起動時/終了時を区別）", await t_shutdown_reason_differs())
    check(".eve_run の残骸だけ消す（AI/ユーザのファイルは消さない）", t_workspace_residue_only())
    check("パス不在・store なしでも安全", t_missing_paths_safe())
    check("掃除の失敗で例外を出さない（起動/終了を止めない）", t_broken_store_does_not_raise())
    check("ビューア掃除は startup のみ", t_viewer_sweep_startup_only())
    check("⭐配線: 起動時の掃除は executor 起動より前", await t_wiring_startup_before_executor())
    check("⭐配線: 終了時の掃除は executor 停止後・store 閉鎖前", await t_wiring_shutdown_after_executor_stop())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
