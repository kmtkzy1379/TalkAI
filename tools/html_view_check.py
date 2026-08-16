r"""HTML 表示の実測ハーネス（**API コストゼロ**・約15秒・Chromium は起こす）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\html_view_check.py

`tests/test_html_view.py` は純関数（方針）を固めるが、**その方針が Chromium に
実際に効いているか**は本物を起こさないと分からない。ここは実物で測る:

  1. HTML が HTML として開く（画像・テキストは今までどおり）
  2. **JavaScript が動く**（ボタンを押すと表示が変わる＝「触れるもの」が成立する）
  3. **ネットに出られない**（fetch が失敗する）
  4. **他のファイルを読めない**（同じフォルダの隣のファイルすら）
  5. **alert() でモーダルに固まらない**（LLM は素で alert を書く）
  6. `window.open` で窓が増えない
  7. 3クリックで「操作中」になり、イブ側からは閉じられない

`tools/capture_guard_check.py` と同じ位置づけ（実起動の前に無料で確かめる層）。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0

PROBE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>probe</title></head>
<body>
<div id="out">0</div>
<button id="b" onclick="document.getElementById('out').textContent='42'">押す</button>
<script>
window.r = {};
fetch('https://example.com/x')
  .then(function(){ window.r.net = 'REACHED'; })
  .catch(function(e){ window.r.net = 'BLOCKED'; });
fetch('secret.txt').then(function(x){ return x.text(); })
  .then(function(t){ window.r.file = 'READ:' + t; })
  .catch(function(e){ window.r.file = 'BLOCKED'; });
var w = window.open('https://example.com');
window.r.popup = (w === null || w === undefined) ? 'NULL' : 'GOT_WINDOW';
setTimeout(function(){
  window.r.alertPhase = 'before';
  alert('モーダルが出たらここで止まる');
  window.r.alertPhase = 'after';
}, 150);
</script></body></html>"""


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}  {extra}", flush=True)
    else:
        FAIL += 1
        print(f"  ❌ {name}  {extra}", flush=True)


def main() -> int:
    from eve.ui import web_view

    if not web_view.preinit():
        print("QtWebEngine が入っていません（`pip install PySide6-Addons`）。")
        print("この場合 HTML はソース表示に落ちます＝機能は死にませんが、ここは測れません。")
        return 2

    from PySide6.QtCore import QTimer
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    app = QApplication.instance() or QApplication([])
    from eve.ui.task_window import TaskWindow

    work = Path(tempfile.mkdtemp(prefix="eve_htmlcheck_"))
    (work / "secret.txt").write_text("よそのファイルの中身", encoding="utf-8")
    (work / "probe.html").write_text(PROBE, encoding="utf-8")
    (work / "memo.txt").write_text("ふつうのメモ", encoding="utf-8")

    win = TaskWindow()
    win.resize(900, 620)
    win.open_file(str(work / "probe.html"), "probe.html", "w1")
    win.open_file(str(work / "memo.txt"), "memo.txt", "w2")
    win.show()

    print("\n[1] 種類ごとの中身")
    inner = win._subs["w1"].widget()
    check("HTML は WebView で開く", isinstance(inner, QWebEngineView),
          type(inner).__name__)
    check("テキストは今までどおり", isinstance(win._subs["w2"].widget(), QPlainTextEdit),
          type(win._subs["w2"].widget()).__name__)
    n_top_before = len(app.topLevelWidgets())

    def phase_click(_ignored=None) -> None:
        print("\n[2] JavaScript が動く（押せる）")
        inner.page().runJavaScript(
            "document.getElementById('b').click();"
            "document.getElementById('out').textContent", phase_results)

    def phase_results(after: str) -> None:
        check("**ボタンを押すと表示が変わる**", after == "42", f"out={after!r}")
        inner.page().runJavaScript("JSON.stringify(window.r)", phase_verdict)

    def phase_verdict(raw: str) -> None:
        import json

        try:
            r = json.loads(raw or "{}")
        except ValueError:
            r = {}
        print("\n[3] 柵（外に出られない・固まらない）")
        check("**ネットに出られない**（fetch が失敗する）", r.get("net") == "BLOCKED",
              f"net={r.get('net')}")
        check("**隣のファイルも読めない**", r.get("file") == "BLOCKED",
              f"file={r.get('file')}")
        check("window.open が窓を返さない", r.get("popup") == "NULL",
              f"popup={r.get('popup')}")
        check("`window.open` で窓が増えていない",
              len(app.topLevelWidgets()) <= n_top_before,
              f"{n_top_before} → {len(app.topLevelWidgets())}")
        check("**alert() でモーダルに固まらない**", r.get("alertPhase") == "after",
              f"alertPhase={r.get('alertPhase')}")

        print("\n[4] ユーザが操作中なら閉じない")
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        from eve.config import Config

        # **本物のマウスイベントで数える**。`_count_click` を直接呼ぶと、肝心の
        # 「HTML ビューの中のクリックが親のフィルタまで届くか」を素通りしてしまう
        # （Chromium の内部ウィジェットが受け取るので、素朴な実装だと届かない）。
        target = inner.focusProxy() or inner
        for _ in range(Config.TASKWIN_PROTECT_CLICKS):
            QTest.mouseClick(target, Qt.LeftButton, Qt.NoModifier, QPoint(20, 20))
        check(f"**HTML ビューの中を {Config.TASKWIN_PROTECT_CLICKS} 回クリックで操作中**"
              f"（受け手: {type(target).__name__}）", win.is_protected("w1"),
              f"clicks={win._clicks.get('w1')}")
        check("題名に錠前が付く（見て分かる）",
              win._subs["w1"].windowTitle().startswith("🔒"),
              win._subs["w1"].windowTitle())
        check("2件目は巻き込まれない（窓ごとに数える）", not win.is_protected("w2"))
        check("**イブ側からは閉じられない**", win.close_file("w1") is False)
        check("窓は残っている", "w1" in win._subs, f"{sorted(win._subs)}")
        check("停止処理（force）では閉じる", win.close_file("w1", force=True) is True)
        win.force_close()
        app.quit()

    # 読み込み完了を待ってから触る（`loadFinished` は WebView が出す）
    inner.loadFinished.connect(lambda ok: QTimer.singleShot(1200, phase_click))
    QTimer.singleShot(20000, app.quit)   # 何かで固まっても必ず終わる
    app.exec()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
