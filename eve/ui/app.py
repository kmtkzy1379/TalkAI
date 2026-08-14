"""UI アプリ本体: Qt（メインスレッド）＋ VoiceLoop（ワーカースレッドの asyncio ループ）。

**スレッド契約**（`docs/PIPELINE_DESIGN.md §9.1/§9.3` の UI 版・`docs/UI_DESIGN.md §3.2`）:
1. UI スレッドは loop 所有 state（StimulusQueue / AudioPlayQueue / caches …）を**読まない**。
   受け取るのは loop が投げた**値のコピー**だけ（ここでは文字列と float タプルのリスト）。
2. loop → UI は Qt signal（queued connection）。`emit` はイベントを積むだけで**ブロックしない**。
3. UI → loop は `loop.call_soon_threadsafe` のみ（現状は終了要求だけ）。
4. **UI が落ちても loop は生きる**: フックは `AudioPlayQueue` 側で try/except に包まれている。
   **loop が落ちたら UI は状態を表示して閉じられる**（`loop_stopped` シグナル）。

asyncio をメインスレッド以外で回してよい根拠: 本リポジトリは `asyncio.create_subprocess_*` も
`signal.*` も使っていない（子プロセスは素の `subprocess.Popen`）＝メインスレッド限定の機能に
依存していない。

この段階のスコープは「**AI の応答で口が動く**」ことだけ。メニュー・設定・ログ表示・
画面表示ペイン・キャプチャ除外は `docs/UI_DESIGN.md` の P1 以降で足す。
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import capture_guard
from ..config import Config
from .live2d_view import Live2DView, assets_ready
from .log_bridge import LogBridge
from .log_panel import LogPanel
from .menu_bar import MenuBar
from .task_window import TaskWindow
from .theme import BG, LOG_FRAME, TEXT_FRAME, app_qss, frame_qss

logger = logging.getLogger(__name__)


class LoopBridge(QObject):
    """loop スレッド → UI スレッド の唯一の通り道（値のコピーだけを運ぶ）。"""

    sentence_started = Signal(str, object)  # (文, 口の開閉計画 or None)
    sentence_ended = Signal(str)
    mood = Signal(str, int)                 # (イブの感情の自由記述, surprise 0-100)
    status = Signal(dict)                   # 表示用スナップショット（値のコピーのみ）
    progress = Signal(str, str)             # 起動の進み具合 (key, 表示文)
    open_item = Signal(str, str, str)       # タスクウィンドウに開く (path, 表示名, window_id)
    close_item = Signal(str)                # タスクウィンドウから閉じる (window_id)
    loop_stopped = Signal(str)              # 異常終了時の理由（正常終了は空文字）


class _ItemHandle:
    """`Popen` の代わりに `Workspace.viewers` へ入れるハンドル。

    `store.py` は `poll()`（開いていれば None）と `terminate()` しか呼ばないので、
    この2つだけ実装すれば `live_viewers()` / `close_viewers()` / 上限判定は**無改造で動く**。
    フラグを書くのは **loop スレッドだけ**（ユーザが閉じた時は UI から
    `call_soon_threadsafe` で戻してくる）＝単一書込を保つ。
    """

    def __init__(self, window_id: str, request_close) -> None:
        self.window_id = window_id
        self._closed = False
        self._request_close = request_close

    def poll(self):
        return None if not self._closed else 0

    def terminate(self) -> None:
        self._closed = True
        self._request_close(self.window_id)

    def mark_closed(self) -> None:
        self._closed = True


class VoiceLoopThread(threading.Thread):
    """専用スレッドで asyncio ループを回し、本番の `VoiceLoop` をそのまま起動する。"""

    def __init__(self, bridge: LoopBridge) -> None:
        super().__init__(name="eve-loop", daemon=True)
        self._bridge = bridge
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._vl = None  # 実体は loop スレッドが持つ。UI からは submit_text 経由でしか触らない
        self._handles: dict = {}   # window_id -> _ItemHandle（loop 所有）
        self._aggregate = Config.WS_VIEWER_MODE != "external"
        self.started_ok = threading.Event()

    # --- loop スレッド上で実行される（UI の状態には触らない） ---
    def _on_start(self, text: str, meta) -> None:
        self._bridge.sentence_started.emit(text, meta)

    def _on_end(self, text: str) -> None:
        self._bridge.sentence_ended.emit(text)

    async def _mood_poller(self, vl) -> None:
        """状態と感情を **loop 上で読んで**値のコピーだけを UI へ流す（4Hz）。

        UI スレッドが loop 所有 state を直接読むと §9.3 の「loop 所有 state を他スレッドが
        触らない」に反する。読むのはここ（loop タスク＝分類(i)）で、渡すのは
        **bool / float / str だけ**。**読むだけ**で書き戻さないので、surprise を発話判定に
        使っていないという中核原理の不変条件も侵さない。
        """
        last_mood: tuple[str, int] = ("", -1)
        while True:
            try:
                mood = (vl.prediction.last_emotion or "", vl.prediction.surprise)
                if mood != last_mood:
                    last_mood = mood
                    self._bridge.mood.emit(mood[0], mood[1])
                self._bridge.status.emit(self._snapshot(vl))
            except Exception:
                logger.exception("状態ポーリングで例外（UI の見た目だけの機能なので継続）")
            await asyncio.sleep(0.25)

    @staticmethod
    def _snapshot(vl) -> dict:
        """loop 上で作る表示用スナップショット（値のコピーのみ・参照を渡さない）。"""
        ai = vl.input._ai
        task = ""
        if vl.task_store is not None:
            try:
                from ..task import active_tasks_for_context, undelivered_reports

                rows = active_tasks_for_context(
                    vl.task_store, undelivered=undelivered_reports(vl.task_store, vl.queue))
                task = rows[0] if rows else ""
            except Exception:
                task = ""
        return {
            "mic_on": bool(ai is not None and ai.running),
            "mic_level": float(getattr(ai, "speech_prob", 0.0) or 0.0) if ai else 0.0,
            "user_speaking": bool(vl.speech_state.user_speaking),
            "stt_pending": bool(vl.speech_state.stt_pending),
            "busy": bool(vl.runner.is_busy()),
            "queued": int(vl.queue.qsize()),
            "task": str(task)[:80],
        }

    # --- タスクウィンドウ（イブの表示物の集約）---
    def open_viewer(self, path, name: str):
        """**loop 上**で `show_on_screen` から呼ばれる。集約 OFF なら None（＝別プロセス）。"""
        if not self._aggregate:
            return None
        import uuid

        wid = uuid.uuid4().hex[:12]
        handle = _ItemHandle(wid, self._request_close_item)
        self._handles[wid] = handle
        self._bridge.open_item.emit(str(path), name, wid)
        return handle

    def _request_close_item(self, window_id: str) -> None:
        """loop 上（`terminate()` から）。UI へ「閉じて」を queued で伝える。"""
        self._handles.pop(window_id, None)
        self._bridge.close_item.emit(window_id)

    def notify_user_closed(self, window_id: str) -> None:
        """UI スレッドから: ユーザが子ウィンドウを閉じた → loop 側の台帳に反映する。"""
        loop = self._loop
        if loop is None:
            return

        def _apply() -> None:
            h = self._handles.pop(window_id, None)
            if h is not None:
                h.mark_closed()

        try:
            loop.call_soon_threadsafe(_apply)
        except RuntimeError:
            pass

    def set_aggregate(self, on: bool) -> None:
        """UI スレッドから集約の ON/OFF を切り替える（既に開いている窓はそのまま）。"""
        loop = self._loop
        if loop is None:
            self._aggregate = on
            return
        try:
            loop.call_soon_threadsafe(setattr, self, "_aggregate", bool(on))
        except RuntimeError:
            self._aggregate = on

    def submit_text(self, text: str) -> None:
        """UI スレッドから安全に投入する（§9.3 手段1: `call_soon_threadsafe` のみ）。"""
        loop, vl = self._loop, self._vl
        if loop is None or vl is None:
            return
        try:
            loop.call_soon_threadsafe(vl.submit_user_text, text)
        except RuntimeError:
            pass  # 既に閉じている

    async def _main(self) -> None:
        from ..voice_loop import VoiceLoop

        self._loop = asyncio.get_running_loop()
        vl = VoiceLoop()
        self._vl = vl
        # 起動の進み具合（案B: 2分間なにも分からない状態をやめる）。emit するだけで戻る。
        vl.on_progress = lambda key, text: self._bridge.progress.emit(key, text)
        # 再生開始/終了フック（既定 None のところへ差し込む＝本体の挙動は変えない）
        vl.audio.on_sentence_start = self._on_start
        vl.audio.on_sentence_end = self._on_end
        # イブの表示物をタスクウィンドウへ集約する（既定 None のところへ差し込む）。
        # 作業フォルダが無効な構成では workspace 自体が None なので何もしない。
        if vl.workspace is not None:
            vl.workspace.viewer_open_fn = self.open_viewer
        self._task = asyncio.create_task(vl.run())
        mood = asyncio.create_task(self._mood_poller(vl))
        self.started_ok.set()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            mood.cancel()  # 見た目用のポーラを先に止める（drain 中に UI を触らせない）
            try:
                await vl.stop()
            except Exception:
                logger.exception("VoiceLoop.stop() で例外")

    def run(self) -> None:
        reason = ""
        try:
            asyncio.run(self._main())
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            logger.exception("VoiceLoop スレッドが異常終了")
        finally:
            self.started_ok.set()
            self._bridge.loop_stopped.emit(reason)

    def request_stop(self) -> None:
        """UI スレッドから安全に停止要求（§9.3 手段1: call_soon_threadsafe のみ）。"""
        loop, task = self._loop, self._task
        if loop is None or task is None:
            return
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass  # 既に閉じている


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("イブ — Live2D（リップシンク接続）")
        self.view = Live2DView()
        # **小さく縮められるようにする**。最小を大きく取ると、画面の狭い環境で
        # ウィンドウが画面高にクランプされ、下の入力欄が画面外へ押し出される
        # （実起動で発生: 論理 1120x700 の環境で 1280x860 が入らなかった）。
        self.view.setMinimumSize(320, 214)
        self.status = QLabel("起動中…（VOICEVOX とマイクを確認しています）")
        self.status.setWordWrap(True)
        self.line = QLabel("")
        self.line.setWordWrap(True)
        self.line.setStyleSheet("font-size: 15px;")

        # 状態インジケータ: 音声だけだと「今こっちの声が届いているのか」が分からない。
        # 値はすべて loop 側スナップショット由来（UI は loop 所有 state を読まない）。
        self.pill = QLabel("起動中")
        self.pill.setAlignment(Qt.AlignCenter)
        self.pill.setMinimumWidth(150)
        self.mic = QProgressBar()
        self.mic.setRange(0, 100)
        self.mic.setTextVisible(False)
        self.mic.setFixedHeight(10)
        self.task_line = QLabel("")
        self.task_line.setStyleSheet("color:#666;")

        # キーボード入力: マイクを使わずに話しかける。実起動での確認が
        # 「喋って待つ」から「打って Enter」になるので検証がずっと軽くなる。
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("キーボードで話しかける（Enter で送信・マイクと同じ扱い）")
        self.entry.returnPressed.connect(self._send_text)
        send = QPushButton("送信")
        send.clicked.connect(self._send_text)

        self.bridge = LoopBridge()
        self.bridge.sentence_started.connect(self._on_sentence_started)
        self.bridge.sentence_ended.connect(self._on_sentence_ended)
        self.bridge.mood.connect(self._on_mood)
        self.bridge.status.connect(self._on_status)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.loop_stopped.connect(self._on_loop_stopped)
        self.view.ready.connect(lambda ok, msg: self.status.setText(("✅ " if ok else "❌ ") + msg))

        self.thread_ = VoiceLoopThread(self.bridge)

        row = QHBoxLayout()
        for label, lo, hi, init, fn in (
            ("口ズレ(ms)", -200, 400, int(self.view.lipsync_offset_sec * 1000),
             lambda v: setattr(self.view, "lipsync_offset_sec", v / 1000)),
            ("口の開き", 20, 250, 100, lambda v: setattr(self.view, "mouth_gain", v / 100)),
            ("横揺れ", 0, 250, int(self.view.sway_x * 100),
             lambda v: setattr(self.view, "sway_x", v / 100)),
            ("縦揺れ", 0, 250, int(self.view.sway_y * 100),
             lambda v: setattr(self.view, "sway_y", v / 100)),
        ):
            row.addWidget(QLabel(label))
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(init)
            s.valueChanged.connect(fn)
            row.addWidget(s)

        # キャプチャ除外の ON/OFF。**この設定だけ即時反映**（OS API を直接叩くだけで
        # `Config` を読み直さないため。配信の直前に切り替えられる方が実用的）。
        self.hide_cb = QCheckBox("イブの画面認識からこのウィンドウを隠す")
        self.hide_cb.setChecked(Config.UI_HIDE_FROM_CAPTURE)
        self.hide_cb.setEnabled(capture_guard.available())
        self.hide_cb.toggled.connect(self._toggle_hide)
        self.hide_state = QLabel("")

        # イブの表示物をタスクウィンドウ内に閉じ込めるか（OFF なら従来どおり画面の好きな位置）
        self.taskwin = TaskWindow()
        self.taskwin.closed_by_user.connect(self.thread_.notify_user_closed)
        self.bridge.open_item.connect(self._open_item)
        self.bridge.close_item.connect(self.taskwin.close_file)
        self.aggregate_cb = QCheckBox("イブの表示をタスクウィンドウ内に集約する")
        self.aggregate_cb.setChecked(Config.WS_VIEWER_MODE != "external")
        self.aggregate_cb.toggled.connect(self._toggle_aggregate)
        show_task = QPushButton("タスクウィンドウを表示")
        show_task.clicked.connect(self._show_task_window)
        self.aggregate_state = QLabel("")
        task_row = QHBoxLayout()
        task_row.addWidget(self.aggregate_cb, 1)
        task_row.addWidget(self.aggregate_state, 2)
        task_row.addWidget(show_task)

        top = QHBoxLayout()
        top.addWidget(self.pill)
        top.addWidget(self.mic, 1)
        top.addWidget(self.task_line, 2)

        entry_row = QHBoxLayout()
        entry_row.addWidget(self.entry, 1)
        entry_row.addWidget(send)

        # 本体のログをタブへ振り分ける（パイプラインは無改造・`log_bridge.py`）
        self.log_bridge = LogBridge()
        self.log_bridge.line.connect(self._on_log_line)
        self.log_bridge.install()

        # --- メニュータブ（1-A では表示のみ。実際に止める配線は 1-B）---
        self.menu = MenuBar()
        self.menu.requested.connect(self._menu_requested)
        self.menu.settings_clicked.connect(lambda: self.pages.setCurrentIndex(1))

        # --- メインタブ: 左に Live2D、右にログ、下に発話テキスト（参考画像の構成）---
        self.logs = LogPanel()
        log_box = QWidget()
        # 枠画像は角に装飾がある。**スライス幅を小さく**して装飾の比率を保ち、
        # 中身は装飾の内側に十分な余白を取って置く（大きすぎると文字が枠に重なる）。
        # 枠画像は横長素材なので、縦長の枠に引き伸ばすと装飾と内側の白地がずれる。
        # スライスを小さめにして「細い装飾枠」として使い、中身は素直に内側へ置く。
        log_box.setStyleSheet(frame_qss(LOG_FRAME, border=60))
        log_box.setMinimumWidth(240)
        lb = QVBoxLayout(log_box)
        lb.setContentsMargins(22, 26, 22, 26)
        lb.addWidget(self.logs)

        stage = QHBoxLayout()
        stage.setSpacing(8)
        stage.addWidget(self.view, 5)
        stage.addWidget(log_box, 3)

        self.line.setObjectName("speech")
        speech_box = QWidget()
        speech_box.setStyleSheet(frame_qss(TEXT_FRAME, border=60))
        sb = QVBoxLayout(speech_box)
        sb.setContentsMargins(34, 20, 34, 18)
        sb.addWidget(self.line)
        speech_box.setFixedHeight(84)   # 固定＝下の入力欄を押し出さない

        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(6, 6, 6, 6)
        ml.addLayout(stage, 1)
        ml.addWidget(speech_box)
        ml.addLayout(top)
        ml.addLayout(entry_row)

        # --- 設定タブ（1-A では枠だけ。中身と機能は Phase 2）---
        settings = QWidget()
        sl = QVBoxLayout(settings)
        # **戻るボタンは一番上**に置く。下に置くと、ウィンドウが小さい時に画面外へ押し出されて
        # メインに戻れなくなる（実起動で実際に起きた）。
        head_row = QHBoxLayout()
        back = QPushButton("← メインに戻る")
        back.clicked.connect(self._show_main)
        head = QLabel("設定")
        head.setStyleSheet("font-size:17px; font-weight:bold;")
        head_row.addWidget(back)
        head_row.addWidget(head, 1)
        note = QLabel("（この画面の項目は次の段階で実装します。"
                      "設定は原則、保存して再起動してから反映されます）")
        note.setObjectName("hint")
        note.setWordWrap(True)
        sl.addLayout(head_row)
        sl.addWidget(note)
        adj = QLabel("調整（開発用・のちに設定の正式項目へ移動）")
        adj.setObjectName("hint")
        sl.addWidget(adj)
        sl.addLayout(row)
        sl.addLayout(task_row)
        sl.addWidget(self.hide_cb)
        sl.addWidget(self.hide_state)
        sl.addStretch(1)
        settings_scroll = QScrollArea()      # 小さい窓でも全項目に手が届く
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)

        self.pages = QStackedWidget()
        self.pages.addWidget(main)
        self.pages.addWidget(settings_scroll)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 6, 6)
        root.addWidget(self.menu)
        root.addWidget(self.pages, 1)
        root.addWidget(self.status)
        self.setStyleSheet(app_qss())

        # --- 読み込み中の覆い（ウィンドウは即座に出す）---
        # Cubism の初期化だけで約5秒 UI スレッドが止まる（実起動で確認）。
        # 何も出ないと固まったように見えるので、**先に覆いを描いてから**重い初期化に入る。
        self.loading = QWidget(self)
        ll = QVBoxLayout(self.loading)
        ll.addStretch(1)
        self.loading_title = QLabel("イブを起動しています…")
        self.loading_title.setAlignment(Qt.AlignCenter)
        self.loading_title.setStyleSheet("font-size:20px; font-weight:bold;")
        self.loading_note = QLabel("Live2D と記憶を読み込んでいます")
        self.loading_note.setAlignment(Qt.AlignCenter)
        self.loading_note.setObjectName("hint")
        ll.addWidget(self.loading_title)
        ll.addWidget(self.loading_note)
        ll.addStretch(1)
        self.loading.setStyleSheet(f"background:{BG};")
        self.loading.setGeometry(self.rect())
        self.loading.raise_()

    def _show_main(self) -> None:
        self.pages.setCurrentIndex(0)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        # **isVisible() で判定しない**: 子は親が show() されるまで不可視なので、
        # show() 前の resize が素通りし、覆いが構築時の小さいサイズのまま残る
        # （実起動で発生: メニューボタンが覆いの下に隠れた）。
        if getattr(self, "loading", None) is not None:
            self.loading.setGeometry(self.rect())

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if getattr(self, "loading", None) is not None:
            self.loading.setGeometry(self.rect())
            self.loading.raise_()

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._refresh)
        self._poll.start(250)
        self._loop_alive = True
        self._spoken = 0
        self._closing = False
        self._close_deadline = 0.0
        self._emotion_text = ""
        self._speaking = False
        self._progress_rows: dict = {}
        self._ready = False

    def start_loop(self) -> None:
        self.thread_.start()

    # --- キャプチャ除外 ---
    def apply_capture_guard(self) -> None:
        """自分の窓に適用する。**窓が実際に出てから**でないと HWND が確定しない。"""
        self._toggle_hide(self.hide_cb.isChecked())

    def _open_item(self, path: str, name: str, wid: str) -> None:
        self.taskwin.open_file(path, name, wid)
        # 開いた直後の窓にもキャプチャ除外を掛ける（イブが自分の出力を実況しないように）
        self._apply_hide_to_task_window()

    def _show_task_window(self) -> None:
        self.taskwin.show()
        self.taskwin.raise_()
        self._apply_hide_to_task_window()

    def _toggle_aggregate(self, on: bool) -> None:
        self.thread_.set_aggregate(on)
        if not on:
            self.aggregate_state.setText("OFF: 画面の好きな位置に別ウィンドウで出します"
                                         "（既に出ている分はそのまま）")
        else:
            self.aggregate_state.setText("ON: タスクウィンドウの中だけに出します")

    def _apply_hide_to_task_window(self) -> None:
        """タスクウィンドウは**イブ自身の窓**なので、UI と同じ設定で隠す。"""
        if self.taskwin.isVisible():
            capture_guard.set_window(int(self.taskwin.winId()), self.hide_cb.isChecked())

    def _toggle_hide(self, hide: bool) -> None:
        self._apply_hide_to_task_window()
        ok = capture_guard.set_window(int(self.winId()), hide)
        # ビューア（別プロセス）へは**状態だけ**を渡し、向こうが自分で掛け直す。
        # 他プロセスの窓へ直接は適用できない（実測 ERROR_ACCESS_DENIED）ため。
        # 環境変数はこれから起動する子の初期値、共有ファイルは**既に開いている窓**用。
        os.environ["UI_HIDE_FROM_CAPTURE"] = "1" if hide else "0"
        capture_guard.publish(hide)
        mark = "✅" if ok else ("—" if not capture_guard.available() else "⚠")
        self.hide_state.setText(f"{mark} {capture_guard.describe(hide)}"
                                + ("" if ok or not capture_guard.available()
                                   else "（適用に失敗）"))

    # --- UI スレッド（queued connection で呼ばれる） ---
    def _on_log_line(self, tab: str, who: str, text: str) -> None:
        # 会話タブは**会話だけ**（ユーザ裁定）。イブの発話は `sentence_started` 側で
        # 実際に喋った瞬間に載せるので、ここでは重複させない。
        if tab == "chat" and who == "eve":
            return
        self.logs.add(tab, who, text)

    _MENU_JA = {"pause": "停止/再開", "mute": "ミュート", "autospeech": "自律発話",
                "vision": "画面認識", "task": "タスク"}

    def _menu_requested(self, key: str, want_on: bool) -> None:
        """押された → 切替中 → 切替完了、を**目に見える形で**進める。

        **1-A の時点では実体（パイプライン）はまだ切り替わらない**。表示だけが変わる。
        黙って見た目だけ変えると「OFF にしたのに動いている」と誤解させるので、
        ログに「表示のみ・実体は次の段階」と明記する（嘘の状態を作らない）。
        """
        name = self._MENU_JA.get(key, key)
        self.menu.set_state(key, "pending")
        self.logs.add("chat", "system", f"{name} を {'ON' if want_on else 'OFF'} に切替中…")

        def done() -> None:
            self.menu.set_state(key, "on" if want_on else "off", on=want_on)
            self.logs.add("chat", "system",
                          f"{name} を {'ON' if want_on else 'OFF'} にしました"
                          "（※表示のみ。実際に止める配線は次の段階で入ります）")

        QTimer.singleShot(700, done)

    def _on_sentence_started(self, text: str, meta) -> None:
        self.view.set_plan(meta)
        self.line.setText(text)
        self.logs.add("chat", "eve", text)
        self._spoken += 1
        self._speaking = True

    def _on_sentence_ended(self, text: str) -> None:
        self.view.clear_plan()  # barge-in で口が開きっぱなしにならない
        self._speaking = False

    # 状態表示。優先順位は「今いちばん知りたいこと」の順（喋っている > 聞いている > 考えている）。
    _PILLS = (
        ("speaking", "🗣 喋ってる", "#e8f2ff", "#1f6feb"),
        ("user_speaking", "👂 聞いてる", "#e9f9ec", "#1a7f37"),
        ("stt_pending", "✍ 聞き取り中", "#fff8e1", "#9a6700"),
        ("busy", "💭 考え中", "#f3eaff", "#6f42c1"),
        ("idle", "🟢 待機中", "#f0f0f0", "#555"),
    )

    # 起動中に「何が終わって何を待っているか」を出す（順序は起動の流れどおり）
    _PROGRESS_ORDER = ("memory", "stt", "rag", "llm", "tts", "warmup", "features", "mic", "ready")

    def _on_progress(self, key: str, text: str) -> None:
        self._progress_rows[key] = text
        self.loading_note.setText(text)
        if key == "ready":
            self._ready = True
            # 起動中は描画を落として CPU をモデルのロードに譲っていた（§起動時間）。戻す。
            self.view.set_full_speed()
            self.loading.hide()          # 覆いを外してメインへ
            self.logs.add("chat", "system", "準備完了。話しかけてください")
        rows = [self._progress_rows[k] for k in self._PROGRESS_ORDER
                if k in self._progress_rows]
        self.status.setText("　/　".join(rows[-4:]))  # 直近4件だけ出す（横に伸びすぎない）
        self.pill.setText("⏳ 起動中" if not self._ready else self.pill.text())

    def _on_status(self, s: dict) -> None:
        if not self._ready:
            return  # 起動中は進捗表示を上書きしない
        if self._speaking:
            key = "speaking"
        elif s.get("user_speaking"):
            key = "user_speaking"
        elif s.get("stt_pending"):
            key = "stt_pending"
        elif s.get("busy"):
            key = "busy"
        else:
            key = "idle"
        for k, text, bg, fg in self._PILLS:
            if k == key:
                self.pill.setText(text)
                self.pill.setStyleSheet(
                    f"background:{bg}; color:{fg}; border-radius:9px; padding:5px 10px;"
                    "font-weight:bold;")
                break
        if not s.get("mic_on"):
            self.mic.setValue(0)
            self.mic.setEnabled(False)
        else:
            self.mic.setEnabled(True)
            self.mic.setValue(int(max(0.0, min(1.0, s.get("mic_level", 0.0))) * 100))
        task = s.get("task") or ""
        q = s.get("queued", 0)
        self.task_line.setText((f"🔍 {task}" if task else "")
                               + (f"　待ち {q}" if q else ""))

    def _send_text(self) -> None:
        text = self.entry.text().strip()
        if not text:
            return
        self.entry.clear()
        self.logs.add("chat", "user", text)
        self.thread_.submit_text(text)

    def _on_mood(self, emotion: str, surprise: int) -> None:
        self.view.set_mood(emotion, surprise)
        self._emotion_text = emotion

    def _on_loop_stopped(self, reason: str) -> None:
        self._loop_alive = False
        self.view.clear_plan()
        self.status.setText("⏹ VoiceLoop 停止" + (f": {reason}" if reason else "（正常終了）"))

    def _refresh(self) -> None:
        if not self._loop_alive or not self._ready:
            return  # 起動中は `_on_progress` の表示を残す
        self.status.setText(
            f"{self.view.fps:.0f} fps ／ 口={self.view.mouth:.2f} ／ 発話 {self._spoken} 文"
            f" ／ 感情={self.view.emotion_label}"
            f"（{self._emotion_text or '—'}） ／ surprise={self.view.surprise}")

    def closeEvent(self, ev) -> None:
        """×を押した時: **UI を固めずに** drain を待ってから閉じる。

        `thread.join()` で待つと GUI スレッドが止まり、Windows が「応答なし」にして
        ユーザが強制終了しがち（＝`cleanup.sweep(phase="shutdown")` が走らず、次回起動まで
        タスクやビューア窓が残る）。停止要求だけ出して、あとは QTimer で完了を見張る。
        """
        if self._closing:
            ev.accept()
            return
        self._closing = True
        self._close_deadline = time.monotonic() + 30.0
        self.status.setText("終了中…（パイプラインを安全に停止しています。閉じるまでお待ちください）")
        self.thread_.request_stop()
        self._close_timer = QTimer(self)
        self._close_timer.timeout.connect(self._check_closed)
        self._close_timer.start(100)
        ev.ignore()  # まだ閉じない（描画は回り続けるので固まって見えない）

    def _check_closed(self) -> None:
        if self.thread_.is_alive() and time.monotonic() < self._close_deadline:
            left = self._close_deadline - time.monotonic()
            self.status.setText(f"終了中…（パイプラインを drain 中・あと最大 {left:.0f} 秒）")
            return
        self._close_timer.stop()
        if self.thread_.is_alive():
            logger.warning("VoiceLoop スレッドが 30 秒で止まらなかった（そのまま終了する）")
        self.log_bridge.uninstall()  # 破棄後の Qt オブジェクトへ emit しない
        self.taskwin.force_close()  # ×は隠すだけなので明示的に閉じる
        self.view.release()
        self.close()


def main() -> int:
    from ..logsetup import configure

    configure()
    ok, why = assets_ready()
    if not ok:
        print(why)
        print("eve_ui/ に Live2D モデルと背景画像を置いてください（gitignore 対象）。")
        return 2
    import live2d.v3 as live2d

    live2d.init()
    try:
        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        # **画面に収まる大きさで開く**。大きすぎると Windows が画面高にクランプし、
        # 下の入力欄が切れる（実起動で発生）。作業領域の 92% を上限にする。
        screen = app.primaryScreen().availableGeometry()
        win.resize(min(1280, int(screen.width() * 0.92)),
                   min(860, int(screen.height() * 0.92)))
        win.show()
        # **先に覆いを描いてから**重い初期化に入る（Cubism の init だけで約5秒 UI が止まる）。
        app.processEvents()
        # show() の直後は HWND が確定していないことがあるので1拍おいて適用する
        QTimer.singleShot(50, win.apply_capture_guard)
        QTimer.singleShot(80, win.start_loop)
        if os.environ.get("UI_AUTOCLOSE_SEC"):  # スモークテスト用
            QTimer.singleShot(int(float(os.environ["UI_AUTOCLOSE_SEC"]) * 1000), win.close)
        return app.exec()
    finally:
        live2d.dispose()
