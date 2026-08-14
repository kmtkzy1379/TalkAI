"""折りたたみのグループ（高級設定用）。

Qt にアコーディオンは無いので、**見出しボタン + 中身の表示切替**で作る。
`QToolButton` の checked 状態と矢印だけで表せるので、アニメーションは入れない
（開閉のたびに再レイアウトが走るスクロール領域の中では、動かす方が見づらい）。

**既定は閉じる**。高級設定は「普段は見えなくてよいもの」の置き場なので、開いた状態で
並ぶと設定タブが縦に伸びて基本設定が埋もれる。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

from .theme import LINE, SUB


class Accordion(QWidget):
    def __init__(self, title: str, body: QWidget, *, opened: bool = False) -> None:
        super().__init__()
        self.head = QToolButton()
        self.head.setText(title)
        self.head.setCheckable(True)
        self.head.setChecked(opened)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
        self.head.setStyleSheet(
            "QToolButton{border:none; font-weight:bold; font-size:14px; padding:8px 4px;"
            f" color:#2b2b33; text-align:left;}} QToolButton:hover{{color:{SUB};}}")
        self.head.toggled.connect(self._toggle)
        self.body = body
        self.body.setVisible(opened)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.head)
        lay.addWidget(self.body)
        self.setStyleSheet(f"Accordion{{border-top:1px solid {LINE};}}")

    def _toggle(self, on: bool) -> None:
        self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)

    def set_opened(self, on: bool) -> None:
        self.head.setChecked(on)

    def is_opened(self) -> bool:
        return self.head.isChecked()
