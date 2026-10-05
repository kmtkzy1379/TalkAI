"""起動時／終了時の共通掃除 — 「どんな終わり方をしても、次の起動はクリーン」を保証する。

## なぜ両端でやるか（ユーザ裁定 2026-08-08）
終了時だけでは**異常終了で残る**（それが直したい症状そのもの）。起動時だけでは、生きている
間に溜まった物が次の起動まで居座る。よって**同じ処理を起動時と終了時の両方から呼ぶ**。
異常終了で取りこぼした分は次回の起動時掃除が回収する＝片方が落ちても最終的に収束する。

## 掃除するもの
1. **未終了タスク**（Pending/Running）。1件制では未終了が1件でも残ると**次の起動直後から
   枠が塞がり、最初の依頼が断られる**（`store.py:91-101` の孤児処理は Running しか見ておらず、
   Pending は再起動を越えて生き残っていた）。
   - Running → Failed（実行中に落ちた＝やりかけ）
   - Pending → Cancelled（一度も始まっていない＝やっていない）
   意味を分けるのは、後で履歴を読んだときに「落ちた」と「やらなかった」を混同しないため。
2. **作業フォルダの実行残骸** `.eve_run/<id>/`（`winexec.py:248` が作る job.json/result.json の
   置き場）。正常に終われば消えるはずだが、強制終了すると残る。
3. **取り残されたビューア窓**。`DETACHED_PROCESS` で起動する（`capabilities.py`）ため
   **親が死んでも生き残る**。正常終了は `close_viewers()` が閉じるが、異常終了は残る。
   起動時は自分の Popen ハンドルを持っていないので、コマンドラインで見つけて畳む。
4. **取り残された Irodori サーバ**（S1）。`sweep_orphan_tts_servers()`。`sweep()` とは別に
   `VoiceLoop.run()` が**ウォームアップの前に**呼ぶ（`sweep()` はタスク機能が有効な時しか呼ばれず、
   しかもウォームアップの後なので、残骸を「既にあるサーバ」として使ってしまう）。
   終了時は自分が起動したものを `IrodoriServer.stop()` が止める。

## 方針
**例外で起動を止めない**。掃除は「あると嬉しい」であって、失敗したら握って続行する
（掃除できないことよりも、掃除でシステムが起動しないことの方が害が大きい）。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

VIEWER_MARKER = "eve.workspace.viewer"  # ビューア子プロセスの見分け（`python -m eve.workspace.viewer ...`）


def sweep_tasks(store, *, phase: str) -> int:
    """未終了タスクを terminal にする。戻り値=片付けた件数。"""
    from .task.schema import CANCELLED, FAILED, PENDING, RUNNING

    if store is None:
        return 0
    n = 0
    why = "起動時の掃除（前回の残り）" if phase == "startup" else "終了時の掃除"
    for t in list(store.list_all()):
        if t.status == RUNNING:
            if store.set_status(t.task_id, FAILED, failure_reason=f"{why}: 実行中のまま中断"):
                n += 1
        elif t.status == PENDING:
            if store.set_status(t.task_id, CANCELLED, failure_reason=f"{why}: 未着手のまま破棄"):
                n += 1
    if n:
        logger.info("🧹 未終了タスクを %d 件片付けた（%s）", n, phase)
    return n


def sweep_workspace(workspace, *, phase: str) -> int:
    """`.eve_run` の実行残骸を消す。戻り値=消したディレクトリ数。"""
    if workspace is None:
        return 0
    try:
        root = Path(workspace.paths.root)
    except Exception:
        return 0
    runs = root / ".eve_run"
    if not runs.is_dir():
        return 0
    n = 0
    for child in runs.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            n += 1
        except Exception:
            logger.debug("実行残骸の削除に失敗（無視）: %s", child, exc_info=True)
    if n:
        logger.info("🧹 実行残骸を %d 件消した（%s）", n, phase)
    return n


def sweep_orphan_viewers() -> int:
    """**前回のランが残した**ビューア窓を畳む（起動時のみ）。戻り値=畳んだ数。

    `DETACHED_PROCESS` で起動しているので親が死んでも生き残る。起動時点では自分が開いた窓は
    無いはずなので、このマーカを持つプロセスは全て前回の残骸とみなしてよい。
    """
    try:
        import psutil
    except Exception:
        logger.debug("psutil が無いので取り残しビューアの掃除をスキップ")
        return 0
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or ())
            if VIEWER_MARKER in cmd:
                p.terminate()
                n += 1
        except Exception:
            continue  # 消えた/権限なし。掃除は best-effort
    if n:
        logger.info("🧹 前回のビューア窓を %d 個閉じた（startup）", n)
    return n


TTS_SERVER_MARKER = "--eve-irodori-server"  # `eve/response/tts_irodori.py` の SERVER_MARKER と同じ


def _owner_pid(cmdline: list) -> int:
    """`--owner-pid N` を取り出す（無ければ 0＝手で起動したサーバ）。"""
    for i, a in enumerate(cmdline):
        if a == "--owner-pid" and i + 1 < len(cmdline):
            try:
                return int(cmdline[i + 1])
            except ValueError:
                return 0
        if a.startswith("--owner-pid="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def sweep_orphan_tts_servers(*, process_iter=None, pid_exists=None) -> int:
    """**持ち主の EVE がもういない** Irodori サーバを止める（起動時のみ）。戻り値=止めた数。

    EVE が起動したサーバは Job Object で EVE と一緒に落ち、サーバ自身も持ち主を見張って
    終わるので、ここに来るのは両方が効かなかった時だけ（保険の三段目）。
    **手で起動したサーバ（`--owner-pid` が無い）と、生きている EVE のサーバは止めない**
    （前者は「EVE が既にあるサーバを使う」前提の運用・後者は別の EVE が使用中）。
    """
    try:
        import psutil
    except Exception:
        logger.debug("psutil が無いので取り残しサーバの掃除をスキップ")
        return 0
    it = process_iter or (lambda: psutil.process_iter(["pid", "cmdline"]))
    alive = pid_exists or psutil.pid_exists
    n = 0
    for p in it():
        try:
            cmd = list(p.info.get("cmdline") or ())
            if TTS_SERVER_MARKER not in cmd:
                continue
            owner = _owner_pid(cmd)
            if owner <= 0 or alive(owner):
                continue
            try:  # venv のランチャの子（本体の python）は親の側で止めるので数えない
                par = p.parent()
                if par is not None and TTS_SERVER_MARKER in (par.cmdline() or ()):
                    continue
            except Exception:
                pass
            for c in p.children(recursive=True):  # venv のランチャ → 本体の python
                try:
                    c.kill()
                except Exception:
                    pass
            p.kill()
            n += 1
        except Exception:
            continue  # 消えた/権限なし。掃除は best-effort
    if n:
        logger.info("🧹 前回の Irodori サーバを %d 個止めた（startup）", n)
    return n


def sweep(*, task_store=None, workspace=None, phase: str) -> dict:
    """起動時／終了時の共通掃除。**例外を上げない**（掃除の失敗で起動/終了を止めない）。

    `phase` は "startup" | "shutdown"。ビューアの掃除は startup のみ
    （shutdown 側は自分の Popen を持っている `Workspace.close_viewers()` が確実で速い）。
    """
    out = {"tasks": 0, "workspace": 0, "viewers": 0}
    for key, fn in (("tasks", lambda: sweep_tasks(task_store, phase=phase)),
                    ("workspace", lambda: sweep_workspace(workspace, phase=phase))):
        try:
            out[key] = fn()
        except Exception:
            logger.warning("%s の掃除に失敗（続行）", key, exc_info=True)
    if phase == "startup":
        try:
            out["viewers"] = sweep_orphan_viewers()
        except Exception:
            logger.warning("取り残しビューアの掃除に失敗（続行）", exc_info=True)
    return out
