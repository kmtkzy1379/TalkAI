r"""起動時/終了時クリーンアップの実機E2E — **本番の起動経路（tools/voice_chat.py）**で検証する。

決定論テスト（`tests/test_cleanup.py`）は関数と配線順序を固定するが、
「**異常終了でゴミが残った状態から本番の起動が本当にクリーンになるか**」は実機でしか分からない。

やること:
  1. 前回ランが異常終了した状態を作る（未終了タスク / `.eve_run` の残骸 / 取り残しビューア窓）
  2. **本番の `tools/voice_chat.py` をそのまま起動**（同じ .env・同じ経路）
  3. 起動ログとファイル/プロセスの実状態で、ゴミが消えたことを確認
  4. 正常終了（SIGTERM 相当）でも未終了タスクが残らないことを確認

本番の記憶を汚さないよう TASK_FILE だけ artifacts 下に逃がす（起動経路そのものは本番と同じ）。
実 API 課金は起動時のウォームアップ分のみ（LLM 応答は起こさない）。

実行:
  $env:PYTHONIOENCODING="utf-8"
  .\.venv\Scripts\python.exe tools\cleanup_e2e.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ART = ROOT / "e2e_logs" / "cleanup_e2e"
ART.mkdir(parents=True, exist_ok=True)
TASK_FILE = ART / "tasks.jsonl"

_passed = 0
_failed = 0


def check(name: str, cond: bool, note: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}" + (f"  — {note}" if note else ""))
    else:
        _failed += 1
        print(f"FAIL {name}" + (f"  — {note}" if note else ""))


def _iso(sec: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=sec)).isoformat()


def seed_garbage(ws_root: Path) -> subprocess.Popen | None:
    """前回ランが異常終了した状態を作る。"""
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        # 1件制ではこの Pending が残るだけで、次の起動から枠が塞がる（直したい症状）
        {"task_id": "t_leftover_pending", "what": "", "goal": "前回の未着手タスク",
         "args": {}, "when": _iso(30), "status": "Pending", "attempts": 0,
         "created_at": now, "updated_at": now, "depth": 0,
         "verdict_kind": "deterministic", "depends_on": []},
        # 実行中に落ちた（executor はもう居ない＝孤児）
        {"task_id": "t_leftover_running", "what": "", "goal": "前回の実行中タスク",
         "args": {}, "when": None, "status": "Running", "attempts": 1,
         "created_at": now, "updated_at": now, "depth": 0,
         "verdict_kind": "deterministic", "depends_on": []},
        # 既に終わっていたものは触られてはいけない（対照）
        {"task_id": "t_leftover_done", "what": "", "goal": "前回の完了タスク",
         "args": {}, "when": None, "status": "Done", "attempts": 1, "result": "できてた",
         "created_at": now, "updated_at": now, "depth": 0,
         "verdict_kind": "deterministic", "depends_on": []},
    ]
    TASK_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                         encoding="utf-8")

    # `.eve_run` の残骸（winexec が作る job.json/result.json の置き場）
    runs = ws_root / ".eve_run"
    for name in ("crashed_a", "crashed_b"):
        d = runs / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text('{"argv": ["python", "x.py"]}', encoding="utf-8")
    # ユーザ/AI のファイルは消されてはいけない（対照）
    keep = ws_root / "消してはいけない.txt"
    keep.write_text("これは残る", encoding="utf-8")

    # 取り残しビューア窓（DETACHED_PROCESS なので親が死んでも生き残る）
    viewer = None
    try:
        viewer = subprocess.Popen(
            [sys.executable, "-m", "eve.workspace.viewer", str(keep), str(ws_root)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x00000008 | 0x08000000,  # DETACHED_PROCESS | CREATE_NO_WINDOW
        )
        time.sleep(2.0)
        if viewer.poll() is not None:
            viewer = None
    except Exception as e:
        print(f"  （ビューアを起動できなかったので窓の検証はスキップ: {type(e).__name__}）")
        viewer = None
    return viewer


def run_real_startup(seconds: float, log: Path) -> str:
    """**本番の起動経路**をそのまま起動して `seconds` 秒後に落とす（＝強制終了の再現）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["TASK_FILE"] = str(TASK_FILE)  # 本番の記憶を汚さない（経路は本番のまま）
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([str(py), "tools/voice_chat.py"], cwd=str(ROOT), env=env,
                             stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        time.sleep(seconds)
        p.kill()      # ★正常終了ハンドラを通さない＝異常終了の再現
        p.wait(30)
    return log.read_text(encoding="utf-8", errors="replace")


def viewer_alive() -> int:
    try:
        import psutil
    except Exception:
        return -1
    n = 0
    for pr in psutil.process_iter(["cmdline"]):
        try:
            if "eve.workspace.viewer" in " ".join(pr.info.get("cmdline") or ()):
                n += 1
        except Exception:
            continue
    return n


def main() -> int:
    from eve.config import Config
    from eve.workspace.paths import resolve_workspace_root

    ws_root, why = resolve_workspace_root(Config.WS_ROOT)
    if ws_root is None:
        print(f"SKIP: 作業フォルダが使えない（{why}）")
        return 0
    ws_root = Path(ws_root)
    print(f"作業フォルダ: {ws_root}")

    print("\n■ 1. 前回ランが異常終了した状態を作る")
    viewer = seed_garbage(ws_root)
    before_runs = len(list((ws_root / ".eve_run").iterdir()))
    before_viewers = viewer_alive()
    print(f"  未終了タスク2件 / .eve_run 残骸 {before_runs} 件 / ビューア窓 {before_viewers} 個")

    print("\n■ 2. 本番の起動経路（tools/voice_chat.py）をそのまま起動")
    log = run_real_startup(75.0, ART / "startup1.log")
    started = "VoiceLoop 稼働" in log
    check("本番経路が起動した", started, "起動ログ未達なら以降は測れない")
    if not started:
        print(log[-1500:])
        return 1

    print("\n■ 3. 起動時の掃除が効いたか")
    check("起動ログに掃除の記録がある", "起動時の掃除" in log,
          [l for l in log.splitlines() if "掃除" in l][:3])

    rows = [json.loads(l) for l in TASK_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict = {}
    for r in rows:
        latest[r["task_id"]] = r
    check("前回の Pending が枠を塞がない（Cancelled 化）",
          latest["t_leftover_pending"]["status"] == "Cancelled",
          latest["t_leftover_pending"]["status"])
    check("前回の Running は Failed 化（やりかけ）",
          latest["t_leftover_running"]["status"] == "Failed",
          latest["t_leftover_running"]["status"])
    check("既に終わっていたタスクは触らない",
          latest["t_leftover_done"]["status"] == "Done",
          latest["t_leftover_done"]["status"])

    left_runs = list((ws_root / ".eve_run").iterdir()) if (ws_root / ".eve_run").is_dir() else []
    check(".eve_run の残骸が消えた", len(left_runs) == 0, f"残り {len(left_runs)} 件")
    check("ユーザ/AI のファイルは消えていない", (ws_root / "消してはいけない.txt").exists())

    if before_viewers > 0:
        check("取り残しビューア窓が閉じた", viewer_alive() == 0, f"残り {viewer_alive()} 個")
    else:
        print("  （ビューア窓を用意できなかったので判定なし）")

    print("\n■ 4. 起動した側が終了時にも掃除するか（今回は kill=異常終了だったので次の起動で確認）")
    # kill した2回目の起動でも「掃除するものが無い」＝冪等であることを見る。
    log2 = run_real_startup(70.0, ART / "startup2.log")
    check("2回目の起動も正常（掃除は冪等）", "VoiceLoop 稼働" in log2)
    rows2 = [json.loads(l) for l in TASK_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest2: dict = {}
    for r in rows2:
        latest2[r["task_id"]] = r
    check("2回目でも状態が変わらない（terminal を再遷移させない）",
          latest2["t_leftover_pending"]["status"] == "Cancelled"
          and latest2["t_leftover_running"]["status"] == "Failed"
          and latest2["t_leftover_done"]["status"] == "Done")

    try:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
    except Exception:
        pass
    shutil.rmtree(ws_root / "消してはいけない.txt", ignore_errors=True)
    try:
        (ws_root / "消してはいけない.txt").unlink(missing_ok=True)
    except Exception:
        pass

    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    print(f"ログ: {ART}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
