"""メニュータブ（停止/再開・自律発話・画面認識・タスク・ミュート・設定）。

**1-A の時点では見た目と状態表示だけ**。実際にパイプラインを止める配線は 1-B。
ここで先に4状態（on / off / pending / disabled）を作っておくのは、
「押した瞬間には切り替わらない」という要件が**表示の作りに影響する**ため。
押す → `pending`（切替中）→ loop が安全地点に着いたら確定、という流れを前提に組む。

`requested` は「押された」だけを伝える。**UI は自分で状態を変えない**
（実体が切り替わったかは loop 側からの通知でしか分からない＝嘘を表示しない）。
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from .theme import menu_button_qss

# (キー, ラベル, ONラベル, OFFラベル)。停止/再開だけは意味が反転するので個別に文言を持つ。
ITEMS = (
    ("pause", "停止/再開", "稼働中", "停止中"),
    ("mute", "ミュート", "音あり", "ミュート中"),
    ("autospeech", "自律発話", "自律発話 ON", "自律発話 OFF"),
    ("vision", "画面認識", "画面認識 ON", "画面認識 OFF"),
    ("task", "タスク", "タスク ON", "タスク OFF"),
)


class MenuBar(QWidget):
    requested = Signal(str, bool)   # (キー, これから ON にしたいか)
    settings_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.buttons: dict[str, QPushButton] = {}
        self._state: dict[str, str] = {}
        self._on: dict[str, bool] = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 6, 6, 0)
        row.setSpacing(6)
        for key, _label, on_text, _off in ITEMS:
            b = QPushButton(on_text)
            b.clicked.connect(lambda _=False, k=key: self._clicked(k))
            self.buttons[key] = b
            self._on[key] = True
            self.set_state(key, "on")
            row.addWidget(b)
        row.addStretch(1)
        self.settings = QPushButton("設定")
        self.settings.clicked.connect(self.settings_clicked.emit)
        self.settings.setStyleSheet(menu_button_qss("on"))
        row.addWidget(self.settings)

    def _clicked(self, key: str) -> None:
        if self._state.get(key) in ("pending", "disabled"):
            return  # 切替中の連打は無視（意図は loop 側で latest-wins）
        self.requested.emit(key, not self._on.get(key, True))

    def set_state(self, key: str, state: str, *, on: bool | None = None) -> None:
        """loop から確定した状態を反映する。**UI から勝手に変えない**。"""
        b = self.buttons.get(key)
        if b is None:
            return
        if on is not None:
            self._on[key] = on
        self._state[key] = state
        b.setStyleSheet(menu_button_qss(state))
        label = dict((k, (o, f)) for k, _l, o, f in ITEMS)[key]
        if state == "pending":
            b.setText("切替中…")
        elif state == "disabled":
            b.setText(label[0] if self._on[key] else label[1])
        else:
            b.setText(label[0] if self._on[key] else label[1])
        b.setEnabled(state != "disabled")
