"""Tier-1: タスクウィンドウの画像表示（**拡大しない・窓に収める**）。API 不要・画面は開かない（offscreen）。

ユーザ 2026-09-30「画像出すとき拡大されて出されることが多い」。旧実装は `pix.scaled(1100, 800)` を固定で作り
`QScrollArea` に入れていたため、小さな画像ほど引き伸ばされ、しかも最大 560x420 の子ウィンドウでは一部しか
見えなかった。守りたい不変条件:
  - 窓より小さい画像は**元の大きさのまま**（拡大しない）
  - 窓より大きい画像は**縦横比を保って窓に収まる**（はみ出さない・スクロールしないと見えない、にならない）
  - 窓の大きさを変えたら描き直す
  - 子ウィンドウは画像の大きさに合わせて開く（小さい画像を大きな窓に置かない）
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 画面を開かない
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

APP = QApplication.instance() or QApplication([])

from eve.ui.task_window import FitImage, TaskWindow  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def _png(tmp: Path, name: str, w: int, h: int) -> Path:
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor(200, 120, 40))
    p = tmp / name
    img.save(str(p), "PNG")
    return p


def _shown(label: FitImage) -> tuple[int, int]:
    """いま描いている絵の論理サイズ。"""
    pm = label.pixmap()
    dpr = pm.devicePixelRatio() or 1.0
    return int(round(pm.width() / dpr)), int(round(pm.height() / dpr))


def t_fit() -> None:
    print("\n[1] 窓より小さい画像は拡大しない / 大きい画像は縦横比を保って収める")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        win = TaskWindow()
        small = win._build(_png(tmp, "small.png", 120, 80), "small.png")
        check("画像は FitImage で出る（スクロール付きの固定拡大ではない）", isinstance(small, FitImage),
              type(small).__name__)
        small.resize(500, 400)
        small.show()
        APP.processEvents()
        check("小さい画像（120x80）は 500x400 の窓でも元の大きさのまま", _shown(small) == (120, 80), str(_shown(small)))

        big = win._build(_png(tmp, "big.png", 1600, 900), "big.png")
        big.resize(400, 300)
        big.show()
        APP.processEvents()
        w, h = _shown(big)
        check("大きい画像（1600x900）は 400x300 に収まる", w <= 400 and h <= 300, f"{w}x{h}")
        check("縦横比を保つ（16:9）", abs(w / h - 16 / 9) < 0.03, f"{w}x{h}")
        check("収めた時は枠いっぱいまで使う（横が枠に当たる）", w >= 396, f"{w}x{h}")

        big.resize(800, 600)
        APP.processEvents()
        w2, h2 = _shown(big)
        check("窓を大きくしたら描き直す（大きく出る）", w2 > w and w2 <= 800 and h2 <= 600, f"{w2}x{h2}")
        big.resize(2000, 1500)
        APP.processEvents()
        check("窓が画像より大きくなっても元の大きさ以上には拡大しない", _shown(big) == (1600, 900), str(_shown(big)))
        win.close()


def t_subwindow_size() -> None:
    print("\n[2] 子ウィンドウは画像の大きさに合わせて開く")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        win = TaskWindow()
        win.resize(1200, 900)
        win.show()
        APP.processEvents()
        win.open_file(str(_png(tmp, "tiny.png", 200, 150)), "tiny.png", "w1")
        APP.processEvents()
        sub = win._subs["w1"]
        check("小さい画像の窓は小さく開く（旧: いつも 560x420）", sub.width() < 560 and sub.height() < 420,
              f"{sub.width()}x{sub.height()}")
        win.open_file(str(_png(tmp, "huge.png", 3000, 2000)), "huge.png", "w2")
        APP.processEvents()
        area = win.mdi.viewport().size()
        sub2 = win._subs["w2"]
        check("大きい画像の窓は枠の中に収まる", sub2.width() <= area.width() and sub2.height() <= area.height(),
              f"{sub2.width()}x{sub2.height()} / 枠 {area.width()}x{area.height()}")
        w, h = _shown(sub2.widget())
        check("大きい画像は窓の中で縮めて全体が見える", w <= sub2.width() and h <= sub2.height(), f"{w}x{h}")
        win._force_close = True
        win.close()


if __name__ == "__main__":
    t_fit()
    t_subwindow_size()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
