"""Windows: **自分のウィンドウを画面キャプチャから隠す**（イブの自己観察ループを断つ）。

## なぜ要るか
UI が画面に出ている間、イブの画面認識(VLM)は**自分のログと自分の顔**を「ユーザの画面」
として撮ってしまう（`docs/HANDOFF.md` の D4 / E11 自己観察ループ。実測では自分が7秒前に
開いた窓を「デスクトップ上に…が新規に開かれています」と実況した）。

## なぜ黒塗りではなくこれか（実測 2026-08-13）
キャプチャ画像上で矩形を黒く塗る案もあったが、**跡地が真っ黒になる**ので
「画面が真っ黒」という新しい実況材料をこちらで作ってしまう（ユーザ指摘）。
`WDA_EXCLUDEFROMCAPTURE` は DWM が**背後の内容を合成する**ため、**窓が最初から無かった
のと同じ画面**になる。実測（シアンの窓の上にマゼンタの窓を重ねて計測）:

| 方式 | 跡地の中身 | 黒画素 |
|---|---|---|
| 除外なし | マゼンタ 92.2% | 0.0% |
| **DisplayAffinity 除外** | **シアン 98.6%**（背後が見える） | **0.8%** |
| 黒塗りマスク | 黒 100% | 100% |

## 制約（いずれも実測で確認）
- **自分のプロセスの窓にしか効かない**。他プロセスの窓へ適用すると
  `ERROR_ACCESS_DENIED(5)` で失敗する（2026-08-14 実測）。よって別プロセスで動く
  ビューア（`eve/workspace/viewer.py`）は**自分で**呼ぶ必要がある。
- 有効な間は **OBS 等の画面キャプチャからも消える**。配信時は OFF にする
  （裁定 2026-08-13: 配信中は画面認識を使わない想定なので実運用で衝突しない）。
- **実行中に ON/OFF を切り替えて元に戻せる**（実測: 除外 0.0% → 解除 92.1%）。
- Windows 10 2004 以降。それ以前や非 Windows では何もせず False を返す（落とさない）。

## 対象は2系統に分かれる（2026-08-15・ユーザ要望）
以前は `UI_HIDE_FROM_CAPTURE` 1つで**メインウィンドウも表示物も**まとめて隠していた。
そのため「イブに自分の UI は見せたくないが、**自分が出した表示物は見せたい**」
（例: 電卓を作らせて、それを一緒に見ながら話す）ができなかった。scope で分ける:

| scope | 対象 | 環境変数 |
|---|---|---|
| `UI` | メインウィンドウ（Live2D・ログ・設定） | `UI_HIDE_FROM_CAPTURE` |
| `TASK` | タスクウィンドウと別プロセスのビューア | `TASKWIN_HIDE_FROM_CAPTURE` |

どちらも既定 ON（従来の挙動と同じ）。`TASK` を OFF にすると、イブは**自分が画面に
出したもの**を画面認識で見られるようになる（自分の UI は隠れたまま）。
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

UI = "ui"        # メインウィンドウ
TASK = "task"    # タスクウィンドウ + 別プロセスのビューア

_ENV = "UI_HIDE_FROM_CAPTURE"        # scope=UI の環境変数（旧 API 互換のため名前を残す）
_ENV_BY_SCOPE = {UI: _ENV, TASK: "TASKWIN_HIDE_FROM_CAPTURE"}
_STATE_BY_SCOPE = {UI: "eve_capture_guard.state", TASK: "eve_capture_guard_task.state"}

_user32 = None
_EnumProc = None


def env_name(scope: str = UI) -> str:
    """scope に対応する環境変数名（知らない scope は UI 扱い＝落とさない）。"""
    return _ENV_BY_SCOPE.get(scope, _ENV)


def _api():
    """user32 を遅延ロードして argtypes まで設定する（非 Windows では None）。"""
    global _user32, _EnumProc
    if _user32 is not None:
        return _user32
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        import ctypes.wintypes as wt

        u = ctypes.WinDLL("user32", use_last_error=True)
        # **argtypes を必ず指定する**: 既定では引数が int 扱いになり、64bit の HWND が
        # 黙って切り捨てられて「窓が見つからない/適用できない」になる（実際に踏んだ）。
        _EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        u.EnumWindows.argtypes = [_EnumProc, wt.LPARAM]
        u.EnumWindows.restype = ctypes.c_bool
        u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        u.GetWindowThreadProcessId.restype = wt.DWORD
        u.IsWindowVisible.argtypes = [wt.HWND]
        u.IsWindowVisible.restype = ctypes.c_bool
        u.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
        u.SetWindowDisplayAffinity.restype = ctypes.c_bool
        _user32 = u
        return u
    except Exception:  # ctypes が使えない環境でも落とさない
        logger.debug("capture_guard: user32 を読み込めなかった", exc_info=True)
        return None


def available() -> bool:
    """この環境でキャプチャ除外を使えるか（Windows かつ user32 が読めるか）。"""
    return _api() is not None


def enabled_by_env(scope: str = UI, default: bool = True) -> bool:
    """環境変数の解釈。子プロセス（ビューア）は環境変数を継承して読む。

    既定は **ON**。この設定は機能フラグではなく**ガード**で、OFF は「イブが自分の画面を
    見てしまう既知の不具合状態」だから（`CLAUDE.md` の「機能フラグはコード既定 off」は
    能力の話で、ここは当てはまらない）。ただし scope=TASK の OFF は不具合ではなく
    「自分が出したものを自分でも見る」という意図的な使い方（電卓を一緒に見る等）。
    """
    raw = os.environ.get(env_name(scope))
    if raw is None:
        return default
    return raw.strip() not in ("0", "false", "False", "no", "")


def set_window(hwnd: int, hide: bool) -> bool:
    """1つの窓に適用。成功なら True。**自分のプロセスの窓のみ**（他プロセスは失敗する）。"""
    u = _api()
    if u is None or not hwnd:
        return False
    try:
        ok = bool(u.SetWindowDisplayAffinity(
            hwnd, WDA_EXCLUDEFROMCAPTURE if hide else WDA_NONE))
        if not ok:
            import ctypes

            logger.warning("キャプチャ除外の設定に失敗 hwnd=%s hide=%s err=%s",
                           hwnd, hide, ctypes.get_last_error())
        return ok
    except Exception:
        logger.exception("キャプチャ除外の設定で例外（無視して続行）")
        return False


def set_own_windows(hide: bool) -> int:
    """**このプロセスが持つ可視の最上位ウィンドウ全部**に適用し、成功した数を返す。

    Tk は `winfo_id()` がウィジェット側のハンドルを返すなど取り回しが面倒なので、
    プロセス単位で列挙する方が確実（ビューアはこちらを使う）。
    """
    u = _api()
    if u is None:
        return 0
    try:
        import ctypes
        import ctypes.wintypes as wt

        mine = os.getpid()
        hits: list[int] = []

        def _cb(hwnd, _lparam):
            pid = wt.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == mine and u.IsWindowVisible(hwnd):
                hits.append(hwnd)
            return True

        cb = _EnumProc(_cb)  # 参照を保持したまま呼ぶ（GC されるとコールバックが死ぬ）
        u.EnumWindows(cb, 0)
        return sum(1 for h in hits if set_window(h, hide))
    except Exception:
        logger.exception("自プロセスのウィンドウ列挙で例外（無視して続行）")
        return 0


def state_path(scope: str = UI):
    """設定の共有先。**別プロセスの窓には親から適用できない**ので、状態だけを共有して
    各プロセスが自分で適用する（環境変数は起動時に固定されるので実行中の切替に使えない）。

    一時ディレクトリに置く: 作業フォルダに置くと `list_files` に見えてイブの視界に入り、
    リポジトリに置くと gitignore の管理が増える。落ちても既定に倒れるだけなので永続性は不要。

    **scope ごとに別ファイル**にする（1ファイルに2値を詰めると、片方だけ書いた古い
    プロセスが他方を巻き添えで戻す）。ビューアが読むのは `TASK` だけ。
    """
    import tempfile
    from pathlib import Path

    return Path(tempfile.gettempdir()) / _STATE_BY_SCOPE.get(scope, _STATE_BY_SCOPE[UI])


def publish(hide: bool, scope: str = UI) -> None:
    """今の設定を子プロセス（ビューア）へ共有する。失敗しても無視（見た目だけの機能）。"""
    try:
        state_path(scope).write_text("1" if hide else "0", encoding="ascii")
    except OSError:
        logger.debug("capture_guard: 状態の共有に失敗", exc_info=True)


def desired(scope: str = UI, default: bool = True) -> bool:
    """今あるべき状態。共有ファイル → 環境変数 → 既定 の順に見る。"""
    try:
        raw = state_path(scope).read_text(encoding="ascii").strip()
        if raw in ("0", "1"):
            return raw == "1"
    except OSError:
        pass
    return enabled_by_env(scope, default)


def describe(hide: bool, scope: str = UI) -> str:
    """UI に出す短い説明（状態の取り違えを防ぐため、効果を明示する）。"""
    if not available():
        return "キャプチャ除外はこの環境では使えません"
    what = "このウィンドウ" if scope == UI else "タスクウィンドウと表示ビューア"
    return (f"{what}をイブの画面認識から隠しています（OBS からも映りません）" if hide
            else f"{what}はイブの画面認識に映ります（OBS にも映ります）")
