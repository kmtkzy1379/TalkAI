r"""タスクウィンドウの実起動確認（本番 UI + 本番 VoiceLoop を起動して通しで見る）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\task_window_e2e.py

2段階で確かめる:
  **A. 配線**（API コスト ≒ 0）— 作業フォルダにファイルを置き、`show_on_screen` を
     loop 上で直接実行して、タスクウィンドウに出るか / 台帳に載るか / `close_window` で
     消えるか / キャプチャ除外が掛かるかを見る。TaskAgent を回さないので安い。
  **B. 通し**（実 API・TaskAgent が動く）— キーボードで「作って見せて」と頼み、
     イブが自分の判断で `show_on_screen` に到達するかを見る。`--skip-b` で省略できる。

判定は発話ではなく **実際の窓の数と台帳** で行う。
"""
from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0
SKIP_B = "--skip-b" in sys.argv


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}  {extra}", flush=True)
    else:
        FAIL += 1
        print(f"  ❌ {name}  {extra}", flush=True)


def main() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    import live2d.v3 as live2d
    from eve import capture_guard
    from eve.logsetup import configure
    from eve.ui.app import MainWindow

    configure()
    live2d.init()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(960, 760)
    win.show()
    QTimer.singleShot(50, win.apply_capture_guard)
    win.start_loop()

    done = threading.Event()

    def on_loop(fn):
        """loop 上で fn(vl) を実行して結果を受け取る（検証スレッドから）。"""
        box: "queue.Queue" = queue.Queue(maxsize=1)
        th = win.thread_

        def run():
            try:
                box.put(fn(th._vl))
            except Exception as e:
                box.put(e)

        th._loop.call_soon_threadsafe(run)
        got = box.get(timeout=30)
        if isinstance(got, Exception):
            raise got
        return got

    def worker() -> None:
        global FAIL
        try:
            _run_checks(on_loop, win, capture_guard)
        except Exception:
            import traceback

            traceback.print_exc()
            FAIL += 1
        finally:
            print(f"\n合計: PASS {PASS} / FAIL {FAIL}", flush=True)
            done.set()

    def poll_done() -> None:
        if done.is_set():
            win.close()
            app.quit()
        else:
            QTimer.singleShot(300, poll_done)

    threading.Thread(target=worker, daemon=True).start()
    QTimer.singleShot(1000, poll_done)
    app.exec()
    live2d.dispose()
    return 1 if FAIL else 0


def _run_checks(on_loop, win, capture_guard) -> None:
    print("起動待ち（ウォームアップに数分かかる）…", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 400:
        try:
            if on_loop(lambda vl: vl is not None and vl.input._ai is not None):
                break
        except Exception:
            pass
        time.sleep(1.0)
    else:
        check("起動した", False, "タイムアウト")
        return
    print(f"起動完了（{time.monotonic() - t0:.0f}秒）", flush=True)
    # **マイクを止める**。イヤホン無しだと STT が周囲音とイブ自身の声を拾い続け、
    # 打ち込んだ依頼が会話に埋もれる（実測 2026-08-14: 依頼の 0.9秒後から
    # 「ここにはいる」「Sveiki」等が連続して入り、TaskAgent に到達しなかった）。
    # ここで確かめたいのは表示経路なので、入力はキーボードだけにする。
    on_loop(lambda vl: vl.input.stop())
    print("マイクを停止（キーボード入力だけで検証する）\n", flush=True)
    time.sleep(2.0)

    ws_ok = on_loop(lambda vl: vl.workspace is not None)
    if not ws_ok:
        check("作業フォルダが有効（WS_ENABLED=1）", False, ".env を確認")
        return

    print("[A] 配線（show_on_screen → タスクウィンドウ）", flush=True)
    check("表示フックが刺さっている",
          on_loop(lambda vl: vl.workspace.viewer_open_fn is not None))

    # 作業フォルダに実ファイルを置く（イブ自身の write_file 経路を使う）
    out = on_loop(lambda vl: vl.capabilities.execute(
        "write_file", {"path": "taskwin_check.txt",
                       "content": "タスクウィンドウの確認用テキスト\n" * 20}))
    check("作業フォルダにファイルを作れた", "taskwin_check.txt" in str(out), str(out)[:60])

    before = len(win.taskwin._subs)
    msg = on_loop(lambda vl: vl.capabilities.execute(
        "show_on_screen", {"path": "taskwin_check.txt"}))
    time.sleep(1.5)
    check("**タスクウィンドウに子ウィンドウが増えた**",
          len(win.taskwin._subs) == before + 1, f"{before} → {len(win.taskwin._subs)}")
    check("返答がタスクウィンドウを指す", "タスクウィンドウ" in str(msg), str(msg)[:60])
    check("台帳（live_viewers）に載る",
          on_loop(lambda vl: [n for (_h, n) in vl.workspace.live_viewers()]) ==
          ["taskwin_check.txt"],
          f"{on_loop(lambda vl: vl.workspace.open_window_names())}")
    check("タスクウィンドウが表示されている", win.taskwin.isVisible())

    # キャプチャ除外（イブ自身の窓なので隠れているべき）
    if capture_guard.available():
        import numpy as np
        from mss import mss

        r = win.taskwin.frameGeometry()
        with mss() as s:
            mon = s.monitors[1]
            img = np.array(s.grab(mon))[:, :, :3]
        x0, y0 = max(0, r.x()) + 20, max(0, r.y()) + 40
        x1 = min(img.shape[1], r.x() + r.width()) - 20
        y1 = min(img.shape[0], r.y() + r.height()) - 20
        sub = img[y0:y1, x0:x1]
        # MDI の背景は一様な灰色。写っていればその色が支配的になる
        grayish = float(((abs(sub[:, :, 0].astype(int) - 128) < 25)
                         & (abs(sub[:, :, 1].astype(int) - 128) < 25)
                         & (abs(sub[:, :, 2].astype(int) - 128) < 25)).mean() * 100)
        check("**キャプチャから隠れている**（MDI の灰色背景が写らない）", grayish < 20,
              f"灰色 {grayish:.1f}%")

    out = on_loop(lambda vl: vl.capabilities.execute("close_window", {}))
    time.sleep(1.2)
    check("close_window で子ウィンドウが消える", len(win.taskwin._subs) == before,
          f"{len(win.taskwin._subs)}")
    check("台帳も空になる", on_loop(lambda vl: vl.workspace.live_viewers()) == [], str(out)[:50])

    if SKIP_B:
        print("\n[B] 通し: --skip-b のため省略", flush=True)
        return

    print("\n[B] 通し（イブ自身に頼む・TaskAgent が動く／実 API）", flush=True)
    n0 = len(win.taskwin._subs)
    win.thread_.submit_text(
        "作業フォルダに好きな短いテキストファイルを1つ作って、それを画面に見せて")
    t1 = time.monotonic()
    while time.monotonic() - t1 < 300:
        if len(win.taskwin._subs) > n0:
            break
        time.sleep(1.0)
    got = len(win.taskwin._subs) - n0
    check("**イブが自分で show_on_screen に到達し、タスクウィンドウに出た**", got > 0,
          f"{time.monotonic() - t1:.0f}秒で {got} 件")
    if got:
        names = on_loop(lambda vl: vl.workspace.open_window_names())
        check("台帳に載っている", bool(names), f"{names}")
        on_loop(lambda vl: vl.capabilities.execute("close_window", {}))
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
