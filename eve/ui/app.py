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

from PySide6.QtCore import QObject, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from .. import capture_guard
from ..config import Config
from .live2d_view import Live2DView, assets_ready
from .log_bridge import LogBridge
from .log_panel import LogPanel
from .menu_bar import MenuBar
from .settings_page import SettingsPage
from .task_window import TaskWindow
from .theme import BG, LINE, app_qss

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
    control_state = Signal(str, str, bool)  # 機能 ON/OFF の**確定**通知 (key, state, on)
    stream_event = Signal(str, dict)        # 配信からの知らせ（キャプチャ除外の外す/戻す）


def _window_class(hwnd: int) -> str:
    """Windows の窓のクラス名（OBS で窓を指定する時に使う）。取れなければ ""。"""
    try:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.windll.user32.GetClassNameW
        fn.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]  # 64bit の HWND を切り捨てない
        buf = ctypes.create_unicode_buffer(256)
        return buf.value if fn(hwnd, buf, 256) else ""
    except Exception:
        return ""


def _process_exe() -> str:
    """このプロセスの実行ファイル名（venv の python.exe は実体を子で起動するので、実体の名前を取る）。"""
    try:
        import psutil

        return psutil.Process().name()
    except Exception:
        import sys

        return os.path.basename(sys.executable)


class StreamDialog(QDialog):
    """［配信］を押した時の確認（誤って押して配信が始まらないように）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("YouTube で配信する")
        lay = QVBoxLayout(self)
        msg = QLabel(
            "OBS で YouTube への配信を始めます。\n"
            "配信中は、画面認識とマイクを切り、自律発話とタスクを入れて、視聴者のチャットに返事をします。\n"
            "PC の音（BGM・ゲームの音）は OBS の設定のまま配信に乗ります。マイクのボタンを入れると、"
            "イブがあなたの声を聞き、配信にも声が乗ります。\n"
            "（終わったら元に戻します。イブのウィンドウは最小化しないでください＝OBS に映らなくなります）")
        msg.setWordWrap(True)
        lay.addWidget(msg)
        self.url = QLineEdit()
        self.url.setPlaceholderText("配信の URL（限定公開の時だけ。空欄なら公開中の配信を自動で探します）")
        lay.addWidget(self.url)
        self.chat_only = QCheckBox("OBS は操作しない（チャットだけ読む・試す時用）")
        lay.addWidget(self.chat_only)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("配信を始める")
        buttons.button(QDialogButtonBox.Cancel).setText("やめる")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(560, 220)


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
        # ユーザが操作中（3クリック以上）。`store.close_viewers()` はこの属性を
        # `getattr(..., False)` で見るので、`Popen`（別プロセスのビューア）は自然に
        # 「保護なし」になる。書くのは loop スレッドだけ（UI からは call_soon_threadsafe）。
        self.protected = False

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
        self._control = None  # ControlPlane（loop 所有・UI からは request_control 経由のみ）
        self._handles: dict = {}   # window_id -> _ItemHandle（loop 所有）
        self._aggregate = Config.WS_VIEWER_MODE != "external"
        self.started_ok = threading.Event()

    # --- loop スレッド上で実行される（UI の状態には触らない） ---
    def _on_start(self, text: str, meta) -> None:
        self._bridge.sentence_started.emit(text, meta)

    def _stream_ui(self, event: str, data: dict) -> None:
        try:
            self._bridge.stream_event.emit(event, data)
        except RuntimeError:
            pass  # UI が先に閉じた（終了処理の途中で配信を片付けた時）

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
        stream = None
        if getattr(vl, "stream", None) is not None:
            try:
                stream = vl.stream.snapshot()
            except Exception:
                stream = None
        return {
            "mic_on": bool(ai is not None and ai.running),
            "mic_level": float(getattr(ai, "speech_prob", 0.0) or 0.0) if ai else 0.0,
            "user_speaking": bool(vl.speech_state.user_speaking),
            "stt_pending": bool(vl.speech_state.stt_pending),
            "busy": bool(vl.runner.is_busy()),
            "queued": int(vl.queue.qsize()),
            "task": str(task)[:80],
            "stream": stream,
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

    def notify_user_engaged(self, window_id: str) -> None:
        """UI スレッドから: ユーザが操作中と判定 → loop 側の台帳に印を付ける。

        印を **loop 側にも**置くのが要点。UI だけで守ると、イブの `close_window` は
        「閉じた」と言いながら窓が残る（台帳と画面がズレて嘘の報告になる）。
        """
        loop = self._loop
        if loop is None:
            return

        def _apply() -> None:
            h = self._handles.get(window_id)
            if h is not None:
                h.protected = True

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

    def configure_stream(self, **kw) -> None:
        """UI スレッドから: 配信の確認で選んだ値と、OBS に映す窓の情報を loop へ渡す（値のコピーだけ）。

        `request_control("stream", True)` より**先に**呼ぶ（call_soon_threadsafe は順番どおりに届く）。
        """
        loop, vl = self._loop, self._vl
        if loop is None or vl is None or getattr(vl, "stream", None) is None:
            return
        import functools

        try:
            loop.call_soon_threadsafe(functools.partial(vl.stream.configure, **kw))
        except RuntimeError:
            pass

    def push_stream_geometry(self, geom: dict) -> None:
        """UI スレッドから: イブのウィンドウと Live2D の位置（配信中に大きさが変わった時）。"""
        loop, vl = self._loop, self._vl
        if loop is None or vl is None or getattr(vl, "stream", None) is None:
            return
        try:
            loop.call_soon_threadsafe(vl.stream.set_geometry, dict(geom))
        except RuntimeError:
            pass

    def request_control(self, key: str, on: bool) -> None:
        """UI スレッドから機能 ON/OFF を要求（§9.3 手段1: call_soon_threadsafe のみ）。

        **切り替えるのは loop 側**。UI は要求を出すだけで、表示は `control_state` で
        戻ってきた確定状態しか反映しない（押した瞬間に表示を変えると嘘になる）。
        """
        loop, ctl = self._loop, self._control
        if loop is None or ctl is None:
            return
        try:
            loop.call_soon_threadsafe(ctl.request, key, bool(on))
        except RuntimeError:
            pass  # 既に閉じている

    async def _main(self) -> None:
        from ..control import ControlPlane
        from ..voice_loop import VoiceLoop

        self._loop = asyncio.get_running_loop()
        vl = VoiceLoop()
        self._vl = vl
        # 機能 ON/OFF の実体。`on_state` は loop 上から同期で呼ばれるので emit だけして戻る。
        self._control = ControlPlane(vl, on_state=lambda k, s, on: self._bridge.control_state.emit(k, s, on))
        # 経過（何を待っているか）は `logger` 経由で会話ログへ出る（LogBridge の eve.control 経路）
        # ＝ここで別の通り道を作らない。
        vl.control = self._control
        if getattr(vl, "stream", None) is not None:
            # 配信からの知らせ（キャプチャ除外の外す/戻す）。loop 上から呼ばれる＝emit だけして戻る
            vl.stream.ui = self._stream_ui
        self._control.start()
        # 構成で使えない機能（`.env` で無効なもの）を最初に灰色にする。
        for key, (state, on) in self._control.snapshot().items():
            self._bridge.control_state.emit(key, state, on)
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
            if self._control is not None:
                try:
                    await self._control.stop()  # 遷移中に落とさない（stop() と競合させない）
                except Exception:
                    logger.exception("ControlPlane.stop() で例外")
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
            try:
                self._bridge.loop_stopped.emit(reason)
            except RuntimeError:
                # `Signal source has been deleted`: UI が先に畳まれると bridge の C++ 実体が
                # 消えており、emit がここで例外になる。**スレッドの最後の1行**なので、
                # 握らないと「Exception in thread eve-loop」がコンソールに出るだけで、
                # 誰も受け取れない通知のために異常終了の見た目が残る（実起動で確認）。
                pass

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
        self.line = QLabel("")   # 見た目は theme.app_qss の QLabel#speech に一本化する

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
        # 配信中だけ出す（チャットの状態・コメント数・視聴者数）
        self.stream_line = QLabel("")
        self.stream_line.setStyleSheet("color:#c93a3f; font-weight:bold;")
        self._streaming = False
        self._capture_before: Optional[tuple[bool, bool]] = None  # 配信前のキャプチャ除外（戻す用）
        self._geom_timer = QTimer(self)
        self._geom_timer.setSingleShot(True)
        self._geom_timer.timeout.connect(self._push_geometry)

        # キーボード入力: マイクを使わずに話しかける。実起動での確認が
        # 「喋って待つ」から「打って Enter」になるので検証がずっと軽くなる。
        # **マイクの ON/OFF は入力欄の左**（打つ時に押す物なので、手がある場所に置く）。
        self.mic_btn = QPushButton()
        self.mic_btn.setFixedSize(34, 34)
        self.mic_btn.setCursor(Qt.PointingHandCursor)
        self.mic_btn.clicked.connect(
            lambda: self._menu_requested("mic", not self._mic_on))
        self._mic_on = True
        self._set_mic_button("on", True)
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
        self.bridge.control_state.connect(self._on_control_state)
        self.bridge.stream_event.connect(self._on_stream_event)
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
        # **2枚に分ける**（2026-08-15 ユーザ指定）: 以前は1つの設定でメインウィンドウも
        # 表示物もまとめて隠していたため、「自分の UI は見せずに、作らせた電卓だけ
        # イブにも見せる」ができなかった。
        self.hide_cb = QCheckBox("イブの画面認識からこのウィンドウ（メイン）を隠す")
        self.hide_cb.setChecked(Config.UI_HIDE_FROM_CAPTURE)
        self.hide_cb.setEnabled(capture_guard.available())
        self.hide_cb.toggled.connect(self._toggle_hide)
        self.hide_state = QLabel("")
        self.hide_task_cb = QCheckBox("イブの画面認識からタスクウィンドウ（表示物）を隠す")
        self.hide_task_cb.setChecked(Config.TASKWIN_HIDE_FROM_CAPTURE)
        self.hide_task_cb.setEnabled(capture_guard.available())
        self.hide_task_cb.toggled.connect(self._toggle_hide_task)
        self.hide_task_state = QLabel("")

        # イブの表示物をタスクウィンドウ内に閉じ込めるか（OFF なら従来どおり画面の好きな位置）
        self.taskwin = TaskWindow()
        self.taskwin.closed_by_user.connect(self.thread_.notify_user_closed)
        self.taskwin.user_engaged.connect(self.thread_.notify_user_engaged)
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
        top.addWidget(self.stream_line, 2)

        entry_row = QHBoxLayout()
        entry_row.addWidget(self.mic_btn)   # 左端（ユーザ指定 2026-08-15）
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
        # **枠素材は使わない**（ユーザ裁定 2026-08-14）。Live2D 以外は仮置きの素の配置。
        log_box = QWidget()
        log_box.setStyleSheet(f"background: {BG}; border-left: 1px solid {LINE};")
        log_box.setMinimumWidth(250)
        lb = QVBoxLayout(log_box)
        lb.setContentsMargins(12, 8, 10, 8)
        lb.addWidget(self.logs)

        stage = QHBoxLayout()
        stage.setSpacing(10)
        stage.addWidget(self.view, 5)
        stage.addWidget(log_box, 3)

        # 発話テキストも枠素材は使わない（仮置き）。読めることだけを優先する。
        self.line.setObjectName("speech")
        self.line.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.line.setWordWrap(True)
        speech_box = QWidget()
        speech_box.setStyleSheet(f"background: {BG}; border-top: 1px solid {LINE};")
        sb = QVBoxLayout(speech_box)
        sb.setContentsMargins(16, 10, 16, 10)
        sb.addWidget(self.line)

        # 入力欄は面を持たせない（発話テキストの直下にもう一枚置くと箱が増える）。
        entry_box = QWidget()
        eb = QVBoxLayout(entry_box)
        eb.setContentsMargins(10, 0, 10, 0)
        eb.setSpacing(6)
        eb.addLayout(top)
        eb.addLayout(entry_row)

        # **縦比 発話テキスト 2 : 入力欄 1**（ユーザ指定 2026-08-14）
        bottom_box = QWidget()
        bottom = QVBoxLayout(bottom_box)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(6)
        bottom.addWidget(speech_box, 2)
        bottom.addWidget(entry_box, 1)
        bottom_box.setMinimumHeight(170)
        bottom_box.setMaximumHeight(230)

        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(10, 8, 10, 8)
        ml.setSpacing(10)
        ml.addLayout(stage, 1)
        ml.addWidget(bottom_box)

        # --- 設定タブ（Phase 2）---
        # **戻るボタンは一番上**に置く（`SettingsPage._head`）。下に置くと、ウィンドウが
        # 小さい時に画面外へ押し出されてメインに戻れなくなる（実起動で実際に起きた）。
        self.settings_page = SettingsPage()
        self.settings_page.back_clicked.connect(self._show_main)
        self.settings_page.restart_requested.connect(self._restart)
        settings = QWidget()
        sl = QVBoxLayout(settings)
        sl.addWidget(self.settings_page)
        adj = QLabel("開発用の即時調整（設定ではありません。保存されず、この起動のあいだだけ効きます）")
        adj.setObjectName("hint")
        adj.setWordWrap(True)
        sl.addWidget(adj)
        sl.addLayout(row)
        sl.addLayout(task_row)
        sl.addWidget(self.hide_cb)
        sl.addWidget(self.hide_state)
        sl.addWidget(self.hide_task_cb)
        sl.addWidget(self.hide_task_state)
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
        if getattr(self, "_streaming", False):
            self._geom_timer.start(300)  # 配信中: OBS の Live2D の切り出しを合わせ直す（大きさが落ち着いてから）

    def changeEvent(self, ev) -> None:
        super().changeEvent(ev)
        from PySide6.QtCore import QEvent

        if (ev.type() == QEvent.Type.WindowStateChange and getattr(self, "_streaming", False)
                and self.isMinimized()):
            self.logs.add("chat", "system", "⚠ 配信中にイブのウィンドウを最小化すると OBS に映りません（元に戻してください）")

    # --- 配信（YouTube）---
    def _live2d_geometry(self) -> dict:
        """イブのウィンドウの中での Live2D の位置（論理座標）。OBS がウィンドウから Live2D を切り出すのに使う。"""
        p = self.view.mapTo(self, QPoint(0, 0))
        return {"win": (self.width(), self.height()),
                "view": (p.x(), p.y(), self.view.width(), self.view.height())}

    def _push_geometry(self) -> None:
        self.thread_.push_stream_geometry(self._live2d_geometry())

    def _stream_requested(self, want_on: bool) -> None:
        """［配信］: 確認してから loop に要求を出す（表示は loop の確定を待つ＝他のボタンと同じ）。"""
        if want_on:
            dlg = StreamDialog(self)
            if dlg.exec() != QDialog.Accepted:
                return
            from ..stream.scene import WindowInfo

            self._show_main()  # 配信中は Live2D の画面にしておく（設定ページを映さない）
            exe = _process_exe()
            main = WindowInfo(self.windowTitle(), _window_class(int(self.winId())), exe)
            task = WindowInfo(self.taskwin.windowTitle(), _window_class(int(self.taskwin.winId())), exe)
            self.thread_.configure_stream(target=dlg.url.text().strip(), use_obs=not dlg.chat_only.isChecked(),
                                          main_window=main, task_window=task, geometry=self._live2d_geometry())
            self.logs.add("chat", "system", "配信を ON に切り替えます…")
            self.thread_.request_control("stream", True)
            return
        ans = QMessageBox.question(self, "配信を終える", "配信を終了しますか？（OBS の配信も止めます）")
        if ans != QMessageBox.StandardButton.Yes:
            return
        self.logs.add("chat", "system", "配信を OFF に切り替えます…")
        self.thread_.request_control("stream", False)

    def _on_stream_state(self, state: str, on: bool) -> None:
        """配信の確定状態: 配信中（と切り替え中）は設定を触らせない（Live2D 以外が映らないように）。"""
        self._streaming = bool(on) and state == "on"
        busy = self._streaming or state == "pending"
        self.menu.settings.setEnabled(not busy)
        if busy:
            self._show_main()
        if not self._streaming:
            self.stream_line.setText("")

    def _on_stream_event(self, event: str, data: dict) -> None:
        if event != "capture":
            return
        if data.get("hide") is False:
            # OBS に映るよう、イブの窓のキャプチャ除外を外す（配信中は画面認識を切っているので自分を見ない）
            if self._capture_before is None:
                self._capture_before = (self.hide_cb.isChecked(), self.hide_task_cb.isChecked())
            self.hide_cb.setChecked(False)
            self.hide_task_cb.setChecked(False)
            self._apply_hide_to_task_window()
            self.logs.add("chat", "system", "OBS に映るよう、イブのウィンドウのキャプチャ除外を外しました（配信が終わったら戻します）")
        elif data.get("restore") and self._capture_before is not None:
            ui_hide, task_hide = self._capture_before
            self._capture_before = None
            self.hide_cb.setChecked(ui_hide)
            self.hide_task_cb.setChecked(task_hide)
            self._apply_hide_to_task_window()

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
        self._restart_after_close = False  # ［⟳ 再起動］が立てる（drain 完了後に新プロセス）
        if Config.UI_ALWAYS_ON_TOP:  # 設定タブの「常に最前面」（次の起動から効く＝ここで1回だけ）
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
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
        self._toggle_hide_task(self.hide_task_cb.isChecked())

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
        """タスクウィンドウは**メインとは別の設定**で隠す（2026-08-15）。

        「イブに自分の UI は見せないが、自分が出した表示物は見せる」を成立させるため。
        """
        if self.taskwin.isVisible():
            capture_guard.set_window(int(self.taskwin.winId()),
                                     self.hide_task_cb.isChecked())

    def _toggle_hide(self, hide: bool) -> None:
        ok = capture_guard.set_window(int(self.winId()), hide)
        os.environ[capture_guard.env_name(capture_guard.UI)] = "1" if hide else "0"
        capture_guard.publish(hide, capture_guard.UI)
        mark = "✅" if ok else ("—" if not capture_guard.available() else "⚠")
        self.hide_state.setText(
            f"{mark} {capture_guard.describe(hide, capture_guard.UI)}"
            + ("" if ok or not capture_guard.available() else "（適用に失敗）"))

    def _toggle_hide_task(self, hide: bool) -> None:
        """タスクウィンドウと**別プロセスのビューア**の設定。

        ビューア（別プロセス）へは**状態だけ**を渡し、向こうが自分で掛け直す。
        他プロセスの窓へ直接は適用できない（実測 ERROR_ACCESS_DENIED）ため。
        環境変数はこれから起動する子の初期値、共有ファイルは**既に開いている窓**用。
        """
        ok = True
        if self.taskwin.isVisible():
            ok = capture_guard.set_window(int(self.taskwin.winId()), hide)
        os.environ[capture_guard.env_name(capture_guard.TASK)] = "1" if hide else "0"
        capture_guard.publish(hide, capture_guard.TASK)
        mark = "✅" if ok else ("—" if not capture_guard.available() else "⚠")
        self.hide_task_state.setText(
            f"{mark} {capture_guard.describe(hide, capture_guard.TASK)}"
            + ("" if ok or not capture_guard.available() else "（適用に失敗）"))

    # --- UI スレッド（queued connection で呼ばれる） ---
    def _on_log_line(self, tab: str, who: str, text: str) -> None:
        # 会話タブは**会話だけ**（ユーザ裁定）。イブの発話は `sentence_started` 側で
        # 実際に喋った瞬間に載せるので、ここでは重複させない。
        if tab == "chat" and who == "eve":
            return
        self.logs.add(tab, who, text)

    _MENU_JA = {"pause": "停止/再開", "mute": "ミュート", "mic": "マイク",
                "autospeech": "自律発話", "vision": "画面認識", "task": "タスク"}

    def _menu_requested(self, key: str, want_on: bool) -> None:
        """押された → loop へ要求を出すだけ。**表示は変えない**。

        表示を変えるのは `_on_control_state`（loop が「実際に切り替わった」と言った時だけ）。
        押した瞬間に見た目を変えると、実体が切り替わる前に ON/OFF を表示することになり、
        「OFF にしたのに普通に稼働している」という嘘になる（1-A で実際に起きた）。
        """
        if key == "stream":
            self._stream_requested(want_on)  # 確認を挟む（誤って配信が始まる/終わるのを防ぐ）
            return
        name = self._MENU_JA.get(key, key)
        verb = "ON" if want_on else "OFF"
        if key == "mute":  # ミュートだけは ON=黙る なので語を反転させないと読み違える
            verb = "ミュート" if want_on else "ミュート解除"
        self.logs.add("chat", "system", f"{name} を {verb} に切り替えます…")
        self.thread_.request_control(key, want_on)

    def _on_control_state(self, key: str, state: str, on: bool) -> None:
        """loop が確定させた状態だけを表示に反映する（唯一の反映点）。"""
        if key == "mic":
            self._set_mic_button(state, on)
            return
        if key == "stream":
            self._on_stream_state(state, on)
        self.menu.set_state(key, state, on=on)

    def _set_mic_button(self, state: str, on: bool) -> None:
        """マイクボタンの見た目。**OFF は斜線入り**（ユーザ指定 2026-08-15）。

        `pending`（切替中）は色だけで示す。ここでもアイコンを変えてしまうと、
        まだ切り替わっていないのに切り替わったように見える。
        """
        from .icons import mic_icon

        self._mic_on = bool(on)
        self.mic_btn.setIcon(mic_icon(on))
        self.mic_btn.setIconSize(QSize(20, 20))
        self.mic_btn.setEnabled(state != "disabled")
        self.mic_btn.setToolTip(
            "切替中…" if state == "pending"
            else ("マイク ON（クリックで切る。キーボードはいつでも使えます）" if on
                  else "マイク OFF（あなたの声は届いていません。クリックで戻す）"))
        base = ("border-radius:9px; border:1px solid %s;" % LINE)
        if state == "pending":
            self.mic_btn.setStyleSheet(
                f"QPushButton{{{base} background:#fff6de; border-color:#e8d08a;}}")
        elif on:
            self.mic_btn.setStyleSheet(
                f"QPushButton{{{base} background:rgba(255,255,255,0.86);}}"
                "QPushButton:hover{background:#eaf3ff;}")
        else:
            self.mic_btn.setStyleSheet(
                f"QPushButton{{{base} background:rgba(179,38,30,0.10);"
                "border-color:rgba(179,38,30,0.35);}"
                "QPushButton:hover{background:rgba(179,38,30,0.18);}")

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
        st = s.get("stream") or {}
        if st.get("active"):
            from ..stream.manager import duration_text

            chat = {"waiting": "配信を探しています", "live": "チャット接続中", "ended": "チャット終了",
                    "error": "チャットを読めない", "quota": "API の上限"}.get(st.get("chat"), "")
            parts = [f"● 配信中 {duration_text(st.get('elapsed') or 0)}", chat,
                     f"コメント {st.get('comments', 0)}"]
            if st.get("viewers") is not None:
                parts.append(f"視聴 {st['viewers']}")
            if st.get("mic"):
                parts.append("マイクが配信に乗っている")  # 持ち主の声が視聴者に聞こえている（気づけるように）
            self.stream_line.setText("・".join(p for p in parts if p))
            self.stream_line.setToolTip(st.get("note") or st.get("title") or "")

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

    def _restart(self) -> None:
        """［⟳ 再起動］: **普通の終了と同じ手順で止めてから**、新しいプロセスを起こす。

        いきなり殺すと `cleanup.sweep(phase="shutdown")` が走らず、タスクとビューア窓が
        残ったまま次のプロセスが立ち上がる。終了処理は `closeEvent` が既に持っているので、
        フラグを立てて**そこに合流する**（新プロセスの起動は drain 完了後・`_check_closed`）。
        """
        if self._closing:
            return
        self._restart_after_close = True
        self.settings_page.status.setText("再起動します（パイプラインを安全に停止しています）…")
        self.close()

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
        if self._restart_after_close:
            # **drain が終わってから**起こす（マイクと VOICEVOX を離した後・タスクは掃除済み）。
            # 数秒だけ2プロセスが重なるが、新しい方はここからウォームアップに入る。
            from .restart import spawn

            spawn()
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

    # **`QApplication` より前**に QtWebEngine を読み込む（`AA_ShareOpenGLContexts` を
    # 立てる必要があり、後からでは Qt が警告を出して Live2D と GL 文脈を共有できない）。
    # 入っていない環境では False が返るだけで、HTML はソース表示に落ちる。
    if Config.TASKWIN_HTML_ENABLED:
        from . import web_view

        web_view.preinit()

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
