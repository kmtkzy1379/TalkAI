r"""設定タブで**保存 → 再起動 → 新しい値で動く**の通し（実起動）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\restart_settings_e2e.py

**実 API 課金あり（ウォームアップ2回ぶん・実測でおよそ 0.09 USD）。**
Phase 2/3 の全体がここに懸かっている: 画面で変えた値が `.env` に落ち、再起動した
プロセスがそれを読み、消費者まで届くこと。途中のどれが欠けても「設定したのに効かない」。

**本物の `.env` には触らない**（`EVE_ENV_FILE` で一時ファイルを見せる）。
子プロセスも同じ `EVE_ENV_FILE` を引き継ぐ（`.env` のキーではないので落とされない）。

流れ:
  1回目: 一時 `.env` で起動 → 設定タブで値を変えて保存 → ［⟳ 再起動］を押す
  2回目: 自動で立ち上がったプロセスが、新しい値を読めているかを書き出して終わる
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent

# 変える設定と、2回目のプロセスで読み出す式。**再起動でしか反映されない**ものを選ぶ。
CHANGES = {
    "VV_SPEED": ("1.37", "vl.tts.speed"),
    "VV_INTONATION": ("1.29", "vl.tts.intonation"),
    "VLM_PHASH_THRESHOLD": ("7", "vl.vlm_change_detector._threshold"),
    "RAG_TOP_K": ("5", "vl.rag.top_k"),
    "SILENCE_THRESHOLD_SEC": ("12.0", "vl.silence_monitor._threshold"),
}
RESULT = "restart_e2e_result.json"


def phase2() -> int:
    """2回目のプロセス: 新しい値が届いているかを書き出して即終了（UI も loop も起こさない）。"""
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    out = {}
    for key, (_val, expr) in CHANGES.items():
        try:
            out[key] = str(eval(expr))
        except Exception as e:
            out[key] = f"ERR {type(e).__name__}: {e}"
    Path(os.environ["EVE_RESTART_RESULT"]).write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("2回目のプロセス: 読み出し完了", out)
    return 0


def phase1() -> int:
    """1回目のプロセス: UI を出して保存 → 再起動ボタンを押す。"""
    os.environ.setdefault("UI_HIDE_FROM_CAPTURE", "0")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    import live2d.v3 as live2d

    from eve.logsetup import configure
    from eve.ui.app import MainWindow

    configure()
    live2d.init()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 700)
    win.show()
    app.processEvents()
    QTimer.singleShot(80, win.start_loop)

    state = {"saved": False, "t0": time.monotonic()}

    def step() -> None:
        el = time.monotonic() - state["t0"]
        if state["saved"]:
            return
        if int(el) % 10 < 1:  # どこで待っているかを出す（黙って時間切れにしない）
            print(f"1回目: {el:5.0f}秒 ready={getattr(win, '_ready', False)} "
                  f"進捗={win.status.text()[:70]}", flush=True)
        # **準備完了を待ち切らない**: 設定の保存も再起動も loop の完成を必要としない。
        # 待つのは「ウォームアップ中でも drain して再起動できる」を確かめたいからで、
        # 上限を置かないと環境要因（マイク占有など）でここに張り付く。
        if not getattr(win, "_ready", False) and el < 90:
            return
        state["saved"] = True
        page = win.settings_page
        win.pages.setCurrentIndex(1)
        page.reload()
        for key, (val, _expr) in CHANGES.items():
            w = page._widgets[key]
            if hasattr(w, "setValue"):
                w.setValue(float(val))
            else:
                w.setCurrentText(val)
        page.save()
        print(f"1回目: 保存しました（{el:.0f}秒）: {page.status.text()[:60]}")
        app.processEvents()
        # 保存が `.env` に落ちたことを、押す前に確かめる
        from eve import settings_store as store

        env = store.read_env()
        bad = [k for k, (v, _e) in CHANGES.items()
               if abs(float(env.get(k, "nan") or "nan") - float(v)) > 1e-9]
        print("1回目: .env への反映", "OK" if not bad else f"NG {bad}")
        timer.stop()
        # 再起動は `sys.argv` をそのまま使って起こす。**ここで印を付けておく**と、
        # 立ち上がった側が「2回目」だと分かる（環境変数だと `.env` 由来でないため
        # そのまま引き継がれ、1回目と区別が付かない）。
        sys.argv = [str(Path(__file__)), "phase2"]
        print("1回目: ［⟳ 再起動］を押します")
        QTimer.singleShot(300, win._restart)

    timer = QTimer()
    timer.timeout.connect(step)
    timer.start(500)
    try:
        return app.exec()
    finally:
        live2d.dispose()


def main() -> int:
    if "phase2" in sys.argv:      # 再起動で立ち上がった側（UI も loop も起こさない＝無課金）
        return phase2()
    if "phase1" in sys.argv:      # 親が起こした1回目（UI を出す）
        return phase1()

    work = Path(tempfile.mkdtemp(prefix="eve_restart_e2e_"))
    env_file = work / ".env"
    shutil.copy(REPO / ".env", env_file)
    result = work / RESULT

    from eve import settings_store as store
    from eve.ui.restart import child_env

    # 1回目の起動: 変える前の値を入れておく（差が出るように既定と違う値から始める）
    base = store.read_env(env_file)
    for key, (val, _expr) in CHANGES.items():
        base[key] = "1.0" if "." in val else "1"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in base.items()) + "\n",
                        encoding="utf-8")
    before = store.read_env(env_file)

    env = child_env(dict(os.environ), env_keys=list(store.read_env(REPO / ".env")))
    env["EVE_ENV_FILE"] = str(env_file)
    env["EVE_RESTART_RESULT"] = str(result)
    env["PYTHONIOENCODING"] = "utf-8"
    env["UI_HIDE_FROM_CAPTURE"] = "0"

    print("1回目のプロセスを起動します（実 API 課金は**ウォームアップ1回ぶんだけ**。")
    print(" 2回目は UI も loop も起こさず値を読むだけなので無課金）")
    print(f"  一時 .env: {env_file}")
    r = subprocess.run([sys.executable, str(Path(__file__)), "phase1"], env=env,
                       cwd=str(REPO), text=True, encoding="utf-8")
    print(f"1回目 終了（exit={r.returncode}）")

    for _ in range(120):
        if result.exists():
            break
        time.sleep(1)
    if not result.exists():
        print("❌ 2回目のプロセスが結果を残しませんでした（再起動していない可能性）")
        return 1

    got = json.loads(result.read_text(encoding="utf-8"))
    after = store.read_env(env_file)
    print("\n" + "=" * 62)
    ok = True
    for key, (val, _expr) in CHANGES.items():
        wrote = after.get(key, "<無>")
        live = got.get(key, "<無>")
        same_env = abs(float(wrote or "nan") - float(val)) < 1e-9
        try:
            same_live = abs(float(live) - float(val)) < 1e-9
        except (TypeError, ValueError):
            same_live = False
        ok &= same_env and same_live
        mark = "✅" if (same_env and same_live) else "❌"
        print(f"{mark} {key:<24} 起動前 {before.get(key, '?'):<7} → "
              f".env {wrote:<7} → 再起動後の実体 {live}")
    print("\n" + ("✅ 保存 → 再起動 → 反映 が通りました。"
                  if ok else "❌ どこかで落ちています（上を参照）。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
