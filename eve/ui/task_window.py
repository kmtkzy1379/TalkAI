"""タスクウィンドウ — **イブが画面に出すものを1枚の窓の中に集約する**。

## 何を解決するか
現状 `show_on_screen` は別プロセスの Tk 窓を**画面の好きな位置に**開く（最大5枚）。
これを1枚の「タスクウィンドウ」の中に閉じ込める。中身は `QMdiArea`＝**窓の中の小さな
デスクトップ**で、子ウィンドウは中で自由に動かせるが**枠の外には出ない**。
外側の窓はふつうのウィンドウなので、ユーザが自由に移動・拡大縮小できる。

## なぜこの形か
- 隠す対象が **UI 1枚 + タスクウィンドウ1枚**に固定される。別プロセスの窓それぞれに
  キャプチャ除外を掛け直す仕組み（共有ファイル）が要らなくなる。
- **E2（窓がランを跨いで居残る）が構造的に消える**。別プロセスでないので孤児が出ない。
- 上限（`WS_VIEWER_MAX_WINDOWS`）はそのまま使える。

## 何を失うか（正直に）
別プロセスだった理由の1つは**クラッシュ隔離**（`workspace/viewer.py` の設計メモ）。
壊れた画像のデコードで落ちても本体が生き残る、という性質はここでは失われる。
そのため (a) 判定と上限は `viewer.py` のものを**再実装せず import して共有**し、
(b) デコードは必ず try で囲み、(c) `external` モードに戻せる逃げ道を残す。

## ユーザが触っている窓は閉じない（2026-08-15 ユーザ指定）
HTML を動かせるようになった＝**ユーザが操作している最中の窓**が生まれる（電卓を叩いて
いる途中に消えると困る）。子ウィンドウを `TASKWIN_PROTECT_CLICKS` 回（既定3）クリック
したら「操作中」と見なし、イブの `close_window` や機能 OFF 時の一括クローズから守る。
**1回にしない**のは誤クリックで居座らせないため（ユーザ指定）。
守るのは*イブ側からの*クローズだけで、ユーザ自身の×と「全部閉じる」は常に通す。
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMdiArea, QMdiSubWindow, QPlainTextEdit,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..config import Config
# **安全規律は viewer.py と共有する**（拡張子を信じない種類判定・サイズ上限・画素数上限）。
# ここで書き直すと、片方だけ直して片方が古いままになる。
from ..workspace.viewer import MAX_BYTES, MAX_PIXELS, TEXT_MAX_CHARS, _hexdump, sniff

logger = logging.getLogger(__name__)


def _message(text: str) -> QWidget:
    """エラー等の説明文。`QLabel` だと横に伸びて枠内で切れるので折り返す入れ物を使う。"""
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setPlainText(text)
    return box


class FitImage(QLabel):
    """画像を**窓に収まる大きさで**出す（元の大きさより大きくはしない）。窓の大きさが変われば描き直す。

    旧実装は `pix.scaled(1100, 800)` を固定で作って `QScrollArea` に入れていた。`scaled` は縮めるだけでなく
    **拡大もする**ので、小さな画像ほど 1100x800 近くまで引き伸ばされ、しかも子ウィンドウは最大 560x420
    なので一部しか見えなかった（ユーザ 2026-09-30「画像出すとき拡大されて出されることが多い」・
    2026-09-28 のルナ検証でもコーヒーの写真が「等倍表示で一部しか見えない」）。
    """

    def __init__(self, pix: QPixmap) -> None:
        super().__init__()
        self._pix = pix
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        # 中身（画像）の大きさで窓を押し広げない＝窓の大きさに合わせて絵の方を縮める
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._refit()

    def natural_size(self) -> QSize:
        """画面上の元の大きさ（論理ピクセル）。子ウィンドウの初期サイズを決めるのに使う。"""
        dpr = self._pix.devicePixelRatio() or 1.0
        return QSize(int(self._pix.width() / dpr), int(self._pix.height() / dpr))

    def resizeEvent(self, ev) -> None:  # noqa: N802（Qt の名前）
        super().resizeEvent(ev)
        self._refit()

    def showEvent(self, ev) -> None:  # noqa: N802（表示前に決まった大きさで描き直す）
        super().showEvent(ev)
        self._refit()

    def _refit(self) -> None:
        nat = self.natural_size()
        w, h = max(1, self.width()), max(1, self.height())
        if nat.width() <= w and nat.height() <= h:
            self.setPixmap(self._pix)  # 収まるなら元の大きさのまま（**拡大しない**）
            return
        # 高 DPI の画面ではぼやけないよう実画素で縮め、同じ倍率を付けて論理サイズ w x h に収める
        dpr = self.devicePixelRatioF() or 1.0
        fit = self._pix.scaled(max(1, int(w * dpr)), max(1, int(h * dpr)),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        fit.setDevicePixelRatio(dpr)
        self.setPixmap(fit)


class _SubWindow(QMdiSubWindow):
    """閉じた**瞬間**に通知する子ウィンドウ。

    `destroyed` シグナルだと Qt が実際にオブジェクトを破棄するまで飛ばない（実測: ユーザが
    × を押しても通知が来ず、アプリ終了時にまとめて飛んだ）。台帳の更新が遅れると
    「5枚開いている扱いなのに画面には無い」状態になり、上限判定が嘘をつく。
    """

    def __init__(self, window_id: str, on_closed) -> None:
        super().__init__()
        self._window_id = window_id
        self._on_closed = on_closed

    def silent_close(self) -> None:
        """イブ側からの閉鎖。**「ユーザが閉じた」として通知しない**（台帳は呼び出し側が更新済み）。"""
        self._on_closed = None
        self.close()

    def closeEvent(self, ev) -> None:
        if self._on_closed is not None:
            try:
                self._on_closed(self._window_id)
            except Exception:
                logger.debug("子ウィンドウの閉鎖通知に失敗", exc_info=True)
        super().closeEvent(ev)


class TaskWindow(QWidget):
    """イブの表示物を集約する窓。中の子ウィンドウは枠外へ出られない。"""

    closed_by_user = Signal(str)  # 子ウィンドウがユーザに閉じられた（window_id）
    user_engaged = Signal(str)    # ユーザが操作中と判定した（window_id）→ loop 側の台帳へ

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("イブのタスクウィンドウ")
        self.mdi = QMdiArea()
        self.mdi.setViewMode(QMdiArea.SubWindowView)
        # **スクロールを付けない**＝子ウィンドウを枠の中に留める（要件「表示範囲をウィンドウ内に
        # 制限する」）。AsNeeded にすると枠外へドラッグでき、スクロールしないと見えなくなる。
        self.mdi.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mdi.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._subs: dict[str, QMdiSubWindow] = {}
        self._clicks: dict[str, int] = {}     # window_id -> クリック数
        self._protected: set[str] = set()     # 「ユーザが操作中」と確定した window_id
        # **アプリ全体のイベントを覗く**。子ウィンドウ自身に付けても届かない:
        # HTML ビュー（QWebEngineView）のマウス入力は Chromium が作る内部ウィジェットが
        # 受け取るので、親のフィルタには上がってこない。アプリ側で拾って、受け取った
        # ウィジェットから親を辿ってどの子ウィンドウかを特定する（**握り潰さない**）。
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.note = QLabel("イブがここに画像やテキストを出します")
        self.note.setStyleSheet("color:#666;")
        tile = QPushButton("並べる")
        tile.clicked.connect(self.mdi.tileSubWindows)
        cascade = QPushButton("重ねる")
        cascade.clicked.connect(self.mdi.cascadeSubWindows)
        close_all = QPushButton("全部閉じる")
        close_all.clicked.connect(self.mdi.closeAllSubWindows)

        bar = QHBoxLayout()
        bar.addWidget(self.note, 1)
        bar.addWidget(tile)
        bar.addWidget(cascade)
        bar.addWidget(close_all)

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(self.mdi, 1)
        self.resize(900, 620)
        self._force_close = False

    # ------------------------------------------------------------ 開く/閉じる
    def open_file(self, path: str, name: str, window_id: str) -> None:
        """UI スレッドで呼ぶ（loop からは queued signal 経由）。失敗しても例外を出さない。"""
        try:
            inner = self._build(Path(path), name)
        except Exception as e:  # 隔離が無いぶん、ここで必ず止める
            logger.exception("タスクウィンドウの中身を作れなかった")
            inner = _message(f"表示できませんでした: {type(e).__name__}: {e}")
        sub = _SubWindow(window_id, self._on_sub_closed)
        sub.setWidget(inner)
        sub.setAttribute(Qt.WA_DeleteOnClose)
        sub.setWindowTitle(name)
        self.mdi.addSubWindow(sub)
        self._place(sub)
        sub.show()
        self._subs[window_id] = sub
        self._refresh_note()
        if not self.isVisible():
            self.show()
        self.raise_()

    def _place(self, sub) -> None:
        """**枠に収まるように**サイズと位置を決める（はみ出したまま開かない）。

        既定の cascade だと、開くたびに右下へずれて枠外へ出てしまう（実測）。
        枠の大きさに合わせて縮め、右下に着いたら左上へ折り返す。
        """
        area = self.mdi.viewport().size()
        w = max(240, min(560, area.width() - 40))
        h = max(180, min(420, area.height() - 40))
        inner = sub.widget()
        if isinstance(inner, FitImage):
            # 画像は**元の大きさ＋枠**で開く（小さい画像を大きな窓に置かない）。枠に入り切らなければ
            # 枠いっぱいまでにして、絵の方を縮める（FitImage）。枠の分（タイトル・縁）はおおよその値。
            nat = inner.natural_size()
            w = max(240, min(nat.width() + 16, area.width() - 40))
            h = max(180, min(nat.height() + 38, area.height() - 40))
        sub.resize(w, h)
        step, n = 28, len(self._subs)
        max_x, max_y = max(0, area.width() - w), max(0, area.height() - h)
        x = (n * step) % (max_x + 1) if max_x else 0
        y = (n * step) % (max_y + 1) if max_y else 0
        sub.move(x, y)

    def close_file(self, window_id: str, force: bool = False) -> bool:
        """イブ側（`close_window` / 終了処理）からの閉じる要求。閉じたら True。

        **「ユーザが閉じた」通知は出さない**（台帳は既に呼び出し側が更新している。
        ここで通知すると閉鎖が二重に伝わる）。

        ユーザが操作中の窓（3クリック以上）は `force` でない限り閉じない。台帳側
        （`store.close_viewers`）でも同じ判定をしているので普通はここまで来ないが、
        「守りは両側」にしておく（判定の間に人が触った場合の取りこぼしを塞ぐ）。
        """
        if not force and window_id in self._protected:
            logger.info("🔒 ユーザが操作中の窓なので閉じなかった（%s）", window_id)
            return False
        sub = self._subs.pop(window_id, None)
        self._clicks.pop(window_id, None)
        self._protected.discard(window_id)
        if sub is not None:
            try:
                sub.silent_close()
            except Exception:
                logger.debug("子ウィンドウを閉じられなかった", exc_info=True)
        self._refresh_note()
        return sub is not None

    # ------------------------------------------------------------ 操作中の判定
    def eventFilter(self, obj, ev) -> bool:
        """マウスの押下だけを数える。**必ず False を返す**（イベントは素通しする）。"""
        try:
            if ev.type() == QEvent.MouseButtonPress:
                wid = self._owner_of(obj)
                if wid is not None:
                    self._count_click(wid)
        except Exception:  # 数え損ねても表示は止めない
            logger.debug("クリック計数で例外（無視）", exc_info=True)
        return False

    def _owner_of(self, obj) -> str | None:
        """イベントを受け取ったウィジェットから、どの子ウィンドウの中かを辿る。"""
        if not isinstance(obj, QWidget):
            return None
        subs = {id(s): wid for wid, s in self._subs.items()}
        w = obj
        for _ in range(24):   # 深さは有限（無限親子ループへの保険）
            if w is None:
                return None
            got = subs.get(id(w))
            if got is not None:
                return got
            w = w.parentWidget()
        return None

    def _count_click(self, window_id: str) -> None:
        need = int(Config.TASKWIN_PROTECT_CLICKS)
        if need <= 0 or window_id in self._protected:
            return
        n = self._clicks.get(window_id, 0) + 1
        self._clicks[window_id] = n
        if n < need:
            return
        self._protected.add(window_id)
        sub = self._subs.get(window_id)
        if sub is not None:
            try:
                sub.setWindowTitle(f"🔒 {sub.windowTitle()}")
            except Exception:
                logger.debug("題名に錠前を付けられなかった", exc_info=True)
        logger.info("🔒 ユーザが操作中と判定（%d クリック）→ イブからは閉じない", n)
        self._refresh_note()
        self.user_engaged.emit(window_id)

    def is_protected(self, window_id: str) -> bool:
        return window_id in self._protected

    def _on_sub_closed(self, window_id: str) -> None:
        self._subs.pop(window_id, None)
        self._clicks.pop(window_id, None)
        self._protected.discard(window_id)
        self._refresh_note()
        self.closed_by_user.emit(window_id)

    def _refresh_note(self) -> None:
        n = len(self._subs)
        locked = len(self._protected)
        try:
            if not n:
                self.note.setText("イブがここに画像やテキストを出します")
            elif locked:
                self.note.setText(f"表示中 {n} 件（うち {locked} 件は操作中🔒＝"
                                  "あなたが閉じるまでイブは消しません）")
            else:
                self.note.setText(f"表示中 {n} 件（枠の中で自由に動かせます）")
        except RuntimeError:
            # 終了時: 親が破棄された後に子の closeEvent が走ると、C++ 側の QLabel は
            # 既に消えている（実測: `Internal C++ object already deleted`）。
            # 表示の更新だけなので黙って諦めてよい。
            pass

    # ------------------------------------------------------------ 中身
    @staticmethod
    def _build(target: Path, name: str) -> QWidget:
        """`viewer.py` と同じ判定・同じ上限で中身を作る。"""
        if not target.is_file():  # 呼び出し側でも見ているが、ここでも正直に返す
            return _message(f"{name} は見つかりませんでした")
        size = target.stat().st_size
        if size > MAX_BYTES:
            return _message(f"{name} は大きすぎて表示できません（{size} バイト）")
        kind = sniff(target)
        if kind == "HTML":
            view = TaskWindow._build_html(target, name)
            if view is not None:
                return view
            # 落ちた時は下のテキスト表示へ流す（ソースを見せる＝嘘をつかない）
        if kind in ("PNG", "JPEG", "GIF", "BMP", "WEBP"):
            reader = QImageReader(str(target))
            dim = reader.size()
            if dim.isValid() and dim.width() * dim.height() > MAX_PIXELS:
                return _message(f"{name} は画素数が多すぎて表示できません "
                                f"（{dim.width()}x{dim.height()}）")
            pix = QPixmap(str(target))
            if pix.isNull():
                return _message(f"{name} は画像として読めませんでした")
            return FitImage(pix)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        if kind == "HTML":
            text.setPlainText("（HTML を動かせる表示機能が使えないので、ソースを出します。"
                              "`pip install PySide6-Addons` で押せる状態になります）\n\n"
                              + target.read_text(encoding="utf-8",
                                                 errors="replace")[:TEXT_MAX_CHARS])
        elif kind == "TEXT":
            text.setPlainText(target.read_text(encoding="utf-8", errors="replace")[:TEXT_MAX_CHARS])
        else:
            # 表示できない形式は**勝手に別の解釈をせず**正直にそう言う（viewer.py と同じ）
            text.setPlainText(f"（{name} はテキストとしても画像としても読めませんでした。"
                              f"先頭を16進で表示します）\n\n" + _hexdump(target.read_bytes()))
        return text

    @staticmethod
    def _build_html(target: Path, name: str):
        """HTML を**動かせる状態**で出す。使えなければ None（呼び出し側がソース表示へ）。"""
        if not Config.TASKWIN_HTML_ENABLED:
            logger.info("TASKWIN_HTML_ENABLED=0 なので %s はソース表示にする", name)
            return None
        try:
            from . import web_view

            if not web_view.available():
                return None
            return web_view.build(target)
        except Exception:
            # WebEngine の初期化は環境依存で落ちうる（GPU/権限）。表示機能ごと死なせない。
            logger.exception("HTML ビューを作れなかった（ソース表示に落とす）")
            return None

    def force_close(self) -> None:
        """アプリ終了時に本当に閉じる（×は隠すだけなので明示の口を用意する）。"""
        self._force_close = True
        self.close()

    def closeEvent(self, ev) -> None:
        """×は「隠す」だけにする。イブが開いた表示物の台帳を、ユーザ操作で壊さないため
        （次に出す時に上限判定と食い違うと『5枚開いてるのに何も見えない』になる）。"""
        if self._force_close:
            ev.accept()
            return
        self.hide()
        ev.ignore()
