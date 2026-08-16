r"""HTML 表示を入れた後の**タスク精度**を通しで測る（実 API・課金あり）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\html_task_e2e.py            # 全6件
    & .\.venv\Scripts\python.exe tools\html_task_e2e.py --only 1,2 # 番号で絞る

`tools/task_window_e2e.py` の [B] を「複数の依頼を順番に流す」形にしたもの。狙いは
**HTML 経路を足したことで既存のタスク（検索・画像・PC状態）が劣化していないか**を
同じ物差しで見ること。判定は発話ではなく **窓の実体・中身のウィジェット型・タスクの
verdict** で行う（喋りは当てにならない）。

出す依頼（ユーザ指定 2026-08-15）:
  1. 四則演算の電卓（HTML・押せる）      4. PC の調子（PC状態の能力）
  2. クリックカウンタ（HTML・押せる）    5. ポケモン図鑑300番の説明（知識/検索）
  3. 好きなゲームの画像（取得＋画像表示） 6. ポケモン図鑑543番の高さ重さ（検索の精度）

**必ず時間の上限を決めて起動する**（CLAUDE.md 進め方7）。`--deadline`（既定 2400秒）で
全体を打ち切り、1件ごとにも `--per`（既定 420秒）で見切る。
"""
from __future__ import annotations

import argparse
import os
import queue as _queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUESTS = [
    ("電卓", "簡易的な四則演算の電卓を作って画面に出して", "html"),
    ("カウンタ", "クリックカウンター作ってみて", "html"),
    ("画像", "好きなゲームタイトルの画像を表示して", "image"),
    ("PC状態", "PCの調子悪いんだけどなんでかな？", "info"),
    ("図鑑300", "ポケモンの図鑑番号300番のポケモンの図鑑説明なんだっけ？", "info"),
    ("図鑑543", "ポケモンの図鑑番号543番のポケモンの高さと重さを調べてみて", "info"),
]

results: list[dict] = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="1始まりの番号をカンマ区切りで")
    ap.add_argument("--text", default="", help="一覧を使わず、この1文だけを投げる")
    ap.add_argument("--per", type=float, default=420.0, help="1件あたりの上限秒")
    ap.add_argument("--deadline", type=float, default=2400.0, help="全体の上限秒")
    args = ap.parse_args()

    picked = [int(x) for x in args.only.split(",") if x.strip()] if args.only else []
    plan = ([("即席", args.text, "?")] if args.text
            else [r for i, r in enumerate(REQUESTS, 1) if not picked or i in picked])

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    import live2d.v3 as live2d
    from eve.logsetup import configure
    from eve.ui import web_view

    configure()
    web_view.preinit()          # QApplication より前（`app.py` の main と同じ順序）
    live2d.init()
    app = QApplication(sys.argv)
    from eve.ui.app import MainWindow

    win = MainWindow()
    win.resize(1100, 780)
    win.show()
    QTimer.singleShot(50, win.apply_capture_guard)
    win.start_loop()

    done = threading.Event()

    def on_loop(fn, timeout: float = 30.0):
        box: "_queue.Queue" = _queue.Queue(maxsize=1)
        th = win.thread_

        def run():
            try:
                box.put(fn(th._vl))
            except Exception as e:
                box.put(e)

        th._loop.call_soon_threadsafe(run)
        got = box.get(timeout=timeout)
        if isinstance(got, Exception):
            raise got
        return got

    def worker() -> None:
        try:
            _run(on_loop, win, plan, args)
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            done.set()

    def poll_done() -> None:
        if done.is_set():
            win.close()
            app.quit()
        else:
            QTimer.singleShot(500, poll_done)

    threading.Thread(target=worker, daemon=True).start()
    QTimer.singleShot(1000, poll_done)
    # 保険（worker が固まっても必ず終わる）
    QTimer.singleShot(int((args.deadline + 600) * 1000), app.quit)
    app.exec()
    live2d.dispose()
    return 0


def _snapshot(on_loop, win) -> dict:
    """今の窓と中身（型）を見る。**発話ではなく実体で判定する**ための一次データ。"""
    kinds = {}
    for wid, sub in list(win.taskwin._subs.items()):
        try:
            kinds[sub.windowTitle()] = type(sub.widget()).__name__
        except Exception:
            kinds[wid] = "?"
    return {"n": len(win.taskwin._subs), "kinds": kinds,
            "names": on_loop(lambda vl: vl.workspace.open_window_names())
            if on_loop(lambda vl: vl.workspace is not None) else []}


def _idle(on_loop) -> bool:
    """アクティブタスクも未配達の報告も無い（1件制の枠が空いている）。"""
    def probe(vl):
        from eve.task.cancel_resolver import undelivered_reports
        from eve.task.schema import PENDING, RUNNING

        if vl.task_store is None:
            return True
        active = [t for t in vl.task_store.list_all() if t.status in (PENDING, RUNNING)]
        undel = undelivered_reports(vl.task_store, vl.queue)
        return not active and not undel

    return bool(on_loop(probe))


def _tasks(on_loop) -> list:
    def probe(vl):
        if vl.task_store is None:
            return []
        return [(t.task_id, t.status, (t.goal or t.what or "")[:40],
                 (t.result or "")[:220]) for t in vl.task_store.list_all()]

    return on_loop(probe)


def _run(on_loop, win, plan, args) -> None:
    print("起動待ち（ウォームアップに数分かかる）…", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 500:
        try:
            if on_loop(lambda vl: vl is not None and vl.input._ai is not None):
                break
        except Exception:
            pass
        time.sleep(1.0)
    else:
        print("❌ 起動しなかった", flush=True)
        return
    print(f"起動完了（{time.monotonic() - t0:.0f}秒）", flush=True)

    # **マイクを止める**（イヤホン無しだと STT が周囲音とイブの声を拾って依頼が埋もれる。
    # 2026-08-14 の task_window_e2e で実測済み）。ここで見たいのはタスクの中身。
    on_loop(lambda vl: vl.input.stop())
    print("マイクを停止（キーボード入力だけで検証）", flush=True)
    time.sleep(2.0)
    if not on_loop(lambda vl: vl.workspace is not None):
        print("❌ 作業フォルダが無効（WS_ENABLED を確認）", flush=True)
        return

    started = time.monotonic()
    for idx, (label, text, want) in enumerate(plan, 1):
        if time.monotonic() - started > args.deadline:
            print("\n⏱ 全体の上限に達したので残りは省略", flush=True)
            break
        print(f"\n{'=' * 62}\n[{idx}/{len(plan)}] {label}: {text}", flush=True)
        before = _snapshot(on_loop, win)
        ids_before = {t[0] for t in _tasks(on_loop)}
        t1 = time.monotonic()
        win.thread_.submit_text(text)

        # 1件制なので「タスクが生えて、終わって、報告が配達されるまで」を待つ
        seen_task = False
        while time.monotonic() - t1 < args.per:
            time.sleep(2.0)
            now = _tasks(on_loop)
            mine = [t for t in now if t[0] not in ids_before]
            if mine:
                seen_task = True
            if seen_task and _idle(on_loop):
                time.sleep(3.0)          # 報告の発話が終わるまで少し待つ
                break
        after = _snapshot(on_loop, win)
        mine = [t for t in _tasks(on_loop) if t[0] not in ids_before]
        new_kinds = {k: v for k, v in after["kinds"].items() if k not in before["kinds"]}
        rec = {
            "label": label, "want": want, "sec": round(time.monotonic() - t1, 1),
            "task": mine[-1] if mine else None,
            "new_windows": new_kinds,
            "files": on_loop(lambda vl: vl.workspace.list(None))[:400],
        }
        results.append(rec)
        print(f"  経過 {rec['sec']}秒 / タスク {mine[-1][1] if mine else '生えず'}", flush=True)
        print(f"  新しい窓: {new_kinds or 'なし'}", flush=True)
        if mine:
            print(f"  報告: {mine[-1][3]}", flush=True)

        # 次の依頼に備えて窓を片付ける（保護が掛かっていない前提。上限5に当てない）
        try:
            on_loop(lambda vl: vl.capabilities.execute("close_window", {}))
        except Exception:
            pass
        time.sleep(2.0)

    print(f"\n{'=' * 62}\n結果一覧", flush=True)
    for r in results:
        st = r["task"][1] if r["task"] else "タスク無し"
        win_kind = "・".join(f"{k}={v}" for k, v in r["new_windows"].items()) or "なし"
        print(f"  [{r['label']:6}] {st:9} {r['sec']:6.1f}s  窓: {win_kind}", flush=True)
        if r["task"]:
            print(f"           報告: {r['task'][3]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
