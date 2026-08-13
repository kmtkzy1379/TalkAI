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
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from .. import capture_guard
from ..config import Config
from .live2d_view import Live2DView, assets_ready

logger = logging.getLogger(__name__)


class LoopBridge(QObject):
    """loop スレッド → UI スレッド の唯一の通り道（値のコピーだけを運ぶ）。"""

    sentence_started = Signal(str, object)  # (文, 口の開閉計画 or None)
    sentence_ended = Signal(str)
    mood = Signal(str, int)                 # (イブの感情の自由記述, surprise 0-100)
    status = Signal(dict)                   # 表示用スナップショット（値のコピーのみ）
    loop_stopped = Signal(str)              # 異常終了時の理由（正常終了は空文字）


class VoiceLoopThread(threading.Thread):
    """専用スレッドで asyncio ループを回し、本番の `VoiceLoop` をそのまま起動する。"""

    def __init__(self, bridge: LoopBridge) -> None:
        super().__init__(name="eve-loop", daemon=True)
        self._bridge = bridge
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._vl = None  # 実体は loop スレッドが持つ。UI からは submit_text 経由でしか触らない
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
        # 再生開始/終了フック（既定 None のところへ差し込む＝本体の挙動は変えない）
        vl.audio.on_sentence_start = self._on_start
        vl.audio.on_sentence_end = self._on_end
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
        self.view.setMinimumSize(640, 427)
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

        top = QHBoxLayout()
        top.addWidget(self.pill)
        top.addWidget(self.mic, 1)
        top.addWidget(self.task_line, 2)

        entry_row = QHBoxLayout()
        entry_row.addWidget(self.entry, 1)
        entry_row.addWidget(send)

        root = QVBoxLayout(self)
        root.addWidget(self.view, 1)
        root.addLayout(top)
        root.addWidget(self.line)
        root.addLayout(entry_row)
        root.addLayout(row)
        root.addWidget(self.hide_cb)
        root.addWidget(self.hide_state)
        root.addWidget(self.status)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._refresh)
        self._poll.start(250)
        self._loop_alive = True
        self._spoken = 0
        self._closing = False
        self._close_deadline = 0.0
        self._emotion_text = ""
        self._speaking = False

    def start_loop(self) -> None:
        self.thread_.start()

    # --- キャプチャ除外 ---
    def apply_capture_guard(self) -> None:
        """自分の窓に適用する。**窓が実際に出てから**でないと HWND が確定しない。"""
        self._toggle_hide(self.hide_cb.isChecked())

    def _toggle_hide(self, hide: bool) -> None:
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
    def _on_sentence_started(self, text: str, meta) -> None:
        self.view.set_plan(meta)
        self.line.setText(f"🤖 {text}")
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

    def _on_status(self, s: dict) -> None:
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
        self.line.setText(f"🧑 {text}")
        self.thread_.submit_text(text)

    def _on_mood(self, emotion: str, surprise: int) -> None:
        self.view.set_mood(emotion, surprise)
        self._emotion_text = emotion

    def _on_loop_stopped(self, reason: str) -> None:
        self._loop_alive = False
        self.view.clear_plan()
        self.status.setText("⏹ VoiceLoop 停止" + (f": {reason}" if reason else "（正常終了）"))

    def _refresh(self) -> None:
        if not self._loop_alive:
            return
        base = self.status.text().split(" ／ ")[0]
        self.status.setText(
            f"{base} ／ {self.view.fps:.0f} fps ／ 口={self.view.mouth:.2f} ／ 発話 {self._spoken} 文"
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
        win.resize(960, 760)
        win.show()
        # show() の直後は HWND が確定していないことがあるので1拍おいて適用する
        QTimer.singleShot(50, win.apply_capture_guard)
        win.start_loop()
        if os.environ.get("UI_AUTOCLOSE_SEC"):  # スモークテスト用
            QTimer.singleShot(int(float(os.environ["UI_AUTOCLOSE_SEC"]) * 1000), win.close)
        return app.exec()
    finally:
        live2d.dispose()
