r"""キャプチャ除外が**実際に効いているか**を画面を撮って確かめる実測ハーネス。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\capture_guard_check.py

**LLM も VOICEVOX も使わない**（API コストゼロ・約30秒）。確かめること:
  ① Qt の窓（本体 UI と同じ経路）が ON で消え、OFF で戻る
  ② 跡地が**黒くならない**（背後が合成される）＝イブが「画面が真っ黒」と言う材料を作らない
  ③ **別プロセスのビューア**が自分で適用して消える（親からは適用できないため）
  ④ ON/OFF を実行中に何度でも切り替えられる

判定は `mss`（本番の `eve/vlm/capture.py` と同じ経路）で撮った画素で行う。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from eve import capture_guard  # noqa: E402

PASS = FAIL = 0
MARK = (255, 0, 255)      # UI 窓の目印（マゼンタ）
BEHIND = (0, 255, 255)    # 背後の窓（シアン）


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}  {extra}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {extra}")


def grab():
    from mss import mss
    with mss() as sct:
        mon = sct.monitors[1]
        return np.array(sct.grab(mon))[:, :, :3].copy(), mon  # BGR


def ratios(rect, mon, inset=40):
    """矩形の内側の「目印色 / 背後色 / 黒」の割合（%）。"""
    img, _ = grab()
    l, t, r, b = rect
    x0, y0 = max(0, l - mon["left"]) + inset, max(0, t - mon["top"]) + inset
    x1 = min(img.shape[1], r - mon["left"]) - inset
    y1 = min(img.shape[0], b - mon["top"]) - inset
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0
    sub = img[y0:y1, x0:x1].reshape(-1, 3).astype(int)
    n = len(sub)
    mark = ((sub[:, 0] > 200) & (sub[:, 1] < 60) & (sub[:, 2] > 200)).sum() / n * 100
    behind = ((sub[:, 0] > 200) & (sub[:, 1] > 200) & (sub[:, 2] < 60)).sum() / n * 100
    black = (sub.max(axis=1) < 30).sum() / n * 100
    return mark, behind, black


def window_rect(hwnd: int):
    import ctypes
    import ctypes.wintypes as wt

    r = wt.RECT()
    dwm = ctypes.windll.dwmapi
    if dwm.DwmGetWindowAttribute(wt.HWND(hwnd), ctypes.c_uint(9),
                                 ctypes.byref(r), ctypes.sizeof(r)) != 0:
        ctypes.windll.user32.GetWindowRect(wt.HWND(hwnd), ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def _kill_viewers(proc) -> None:
    """ビューアを確実に落とす。**孫まで消す**（venv の python.exe が実体を子で起動する）。"""
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def main() -> int:
    if not capture_guard.available():
        print("この環境ではキャプチャ除外を使えない（Windows 以外）")
        return 2

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel

    app = QApplication.instance() or QApplication(sys.argv)

    behind = QLabel("ユーザのアプリ（背後）")
    behind.setStyleSheet("background:#00ffff; color:black; font-size:30px;")
    behind.setAlignment(Qt.AlignCenter)
    behind.resize(900, 620)
    behind.move(120, 120)
    behind.show()

    ui = QLabel("EVE UI")
    ui.setStyleSheet("background:#ff00ff; color:white; font-size:30px;")
    ui.setAlignment(Qt.AlignCenter)
    ui.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    ui.resize(520, 340)
    ui.move(300, 260)
    ui.show()
    for _ in range(40):
        app.processEvents()
        time.sleep(0.02)

    hwnd = int(ui.winId())
    rect = window_rect(hwnd)
    _, mon = grab()

    print("\n[1] Qt の窓（本体 UI と同じ経路）")
    capture_guard.set_window(hwnd, False)
    time.sleep(0.5)
    m0, b0, k0 = ratios(rect, mon)
    check("OFF では窓が写る", m0 > 80, f"目印 {m0:.1f}%")

    ok = capture_guard.set_window(hwnd, True)
    time.sleep(0.5)
    m1, b1, k1 = ratios(rect, mon)
    check("ON で窓が消える", ok and m1 < 1.0, f"目印 {m1:.1f}%")

    print("\n[2] 跡地に何が写るか（黒い穴を作っていないか）")
    check("**背後の内容が合成される**", b1 > 90, f"背後色 {b1:.1f}%")
    check("**黒くならない**（黒に反応する材料を作らない）", k1 < 5, f"黒 {k1:.1f}%")

    print("\n[3] 実行中の ON/OFF 切り替え")
    states = []
    for want in (False, True, False, True):
        capture_guard.set_window(hwnd, want)
        time.sleep(0.35)
        m, _, _ = ratios(rect, mon)
        states.append(m < 1.0)
    check("OFF/ON を4回切り替えて毎回追従する", states == [False, True, False, True],
          f"{['隠' if s else '見' for s in states]}")

    ui.hide()
    behind.hide()
    app.processEvents()

    print("\n[4] 別プロセスのビューア（自分で適用する経路）")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(root, "spike", "out", "_capture_guard_sample.txt")
    os.makedirs(os.path.dirname(sample), exist_ok=True)
    with open(sample, "w", encoding="utf-8") as f:
        f.write("キャプチャ除外の確認用\n" * 40)

    def run_viewer(hide: bool):
        env = dict(os.environ, UI_HIDE_FROM_CAPTURE=("1" if hide else "0"))
        p = subprocess.Popen([sys.executable, "-m", "eve.workspace.viewer", sample,
                              os.path.dirname(sample)], cwd=root, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        time.sleep(3.0)
        # **タイトルで探す**（pid では見つからない）: venv の python.exe が実体を
        # 別プロセスで起動し直すため、Tk の窓は Popen の pid ではなくその子に属する
        # （2026-08-14 実測・この挙動は capture_guard 導入前からある）。
        # ビューア自身は自分のプロセスで `set_own_windows()` を呼ぶので除外は正しく効く。
        import ctypes
        import ctypes.wintypes as wt
        u = ctypes.WinDLL("user32", use_last_error=True)
        proc_t = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        u.EnumWindows.argtypes = [proc_t, wt.LPARAM]
        u.EnumWindows.restype = ctypes.c_bool
        u.IsWindowVisible.argtypes = [wt.HWND]
        u.IsWindowVisible.restype = ctypes.c_bool
        u.GetWindowTextW.argtypes = [wt.HWND, ctypes.c_wchar_p, ctypes.c_int]
        u.GetWindowTextW.restype = ctypes.c_int
        hits = []

        def cb(h, _l):
            b = ctypes.create_unicode_buffer(256)
            u.GetWindowTextW(h, b, 256)
            if "イブの作業フォルダ" in b.value and u.IsWindowVisible(h):
                hits.append(h)
            return True

        c = proc_t(cb)
        u.EnumWindows(c, 0)
        return p, (hits[0] if hits else 0)

    p_off, h_off = run_viewer(False)
    if not h_off:
        check("ビューアの窓を検出できた（OFF）", False, "窓が見つからない")
    else:
        r = window_rect(h_off)
        img, mon2 = grab()
        sub = img[max(0, r[1] - mon2["top"]) + 40:r[3] - mon2["top"] - 40,
                  max(0, r[0] - mon2["left"]) + 40:r[2] - mon2["left"] - 40]
        var_off = float(sub.std()) if sub.size else 0.0
        check("OFF ではビューアが写る（中身に変化がある）", var_off > 3.0, f"分散 {var_off:.1f}")
    _kill_viewers(p_off)
    time.sleep(1.0)

    p_on, h_on = run_viewer(True)
    if not h_on:
        check("ビューアの窓を検出できた（ON）", False, "窓が見つからない")
    else:
        r = window_rect(h_on)
        img, mon2 = grab()
        sub = img[max(0, r[1] - mon2["top"]) + 40:r[3] - mon2["top"] - 40,
                  max(0, r[0] - mon2["left"]) + 40:r[2] - mon2["left"] - 40]
        # 窓があった場所に「ビューアの白い本文」が写っていないこと＝背後が見えている
        whiteish = float(((sub > 235).all(axis=2)).mean() * 100) if sub.size else 0.0
        check("**ON でビューアが自分で消える（別プロセスでも効く）**", whiteish < 40,
              f"白地 {whiteish:.1f}%")
    _kill_viewers(p_on)
    try:
        os.remove(sample)
    except OSError:
        pass

    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
