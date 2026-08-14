"""ログ表示。**会話ログと「隠しログ」を分ける**（ユーザ裁定 2026-08-14）。

> タスク進行や画面認識のログは隠しログとして使う。できるならフィードバックも隠しログ。
> 全部同じく乗せるとぐちゃぐちゃで見にくいため。

タブで切り替える:
  - **会話**: ユーザ発話とイブの発話だけ。自発発話は色で区別する（経路が別物なので）
  - タスク: 依頼 → 実行 → 報告
  - 画面: 画面認識の実況
  - 内省: FeedbackLLM の感情・予測差（surprise）

**行数は上限つき**（既定 500行/タブ）。常駐アプリなので無制限に伸ばすとメモリと描画が
効いてくる。古い行から捨てる（ロケット鉛筆・本体の会話履歴と同じ流儀）。
"""
from __future__ import annotations

import html
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget, QTextEdit, QVBoxLayout, QWidget

from .theme import SPEAKER_COLORS, SPEAKER_LABELS, SUB

MAX_LINES = 500


class _Log(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self._lines = 0

    def add(self, when: str, who: str, text: str, color: str) -> None:
        at_bottom = self.verticalScrollBar().value() >= self.verticalScrollBar().maximum() - 4
        self.append(
            f'<div style="margin:2px 0;">'
            f'<span style="color:{SUB};font-size:11px;">{when}</span> '
            f'<span style="color:{color};font-weight:bold;">{html.escape(who)}</span> '
            f'<span style="color:{color};">{html.escape(text)}</span></div>')
        self._lines += 1
        if self._lines > MAX_LINES:  # 古い行から捨てる（無制限に伸ばさない）
            cur = self.textCursor()
            cur.movePosition(cur.MoveOperation.Start)
            cur.select(cur.SelectionType.BlockUnderCursor)
            cur.removeSelectedText()
            cur.deleteChar()
            self._lines -= 1
        if at_bottom:  # 上を読んでいる時は勝手に飛ばさない
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class LogPanel(QWidget):
    """会話ログ＋隠しログ。`add(kind, who, text)` だけで振り分ける。"""

    KINDS = (("chat", "会話"), ("task", "タスク"), ("vision", "画面"), ("mind", "内省"))

    def __init__(self) -> None:
        super().__init__()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self._logs: dict[str, _Log] = {}
        for key, label in self.KINDS:
            log = _Log()
            self._logs[key] = log
            self.tabs.addTab(log, label)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        self._unseen: dict[str, int] = {k: 0 for k, _ in self.KINDS}
        self.tabs.currentChanged.connect(self._clear_badge)

    def add(self, kind: str, who: str, text: str) -> None:
        log = self._logs.get(kind)
        if log is None or not text:
            return
        log.add(datetime.now().strftime("%H:%M:%S"),
                SPEAKER_LABELS.get(who, who), text, SPEAKER_COLORS.get(who, "#2b2b33"))
        idx = [k for k, _ in self.KINDS].index(kind)
        if self.tabs.currentIndex() != idx:  # 見ていないタブには件数バッジを出す
            self._unseen[kind] += 1
            label = dict(self.KINDS)[kind]
            self.tabs.setTabText(idx, f"{label} ({self._unseen[kind]})")

    def _clear_badge(self, idx: int) -> None:
        if 0 <= idx < len(self.KINDS):
            key, label = self.KINDS[idx]
            self._unseen[key] = 0
            self.tabs.setTabText(idx, label)
