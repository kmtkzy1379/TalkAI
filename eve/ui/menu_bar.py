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

# (キー, ラベル, ONラベル, OFFラベル, 初期値)。
# **`on` の意味は `eve/control.py` の ControlPlane と必ず一致させる**（ここだけ反転すると
# 押した向きと実際の切り替えが逆になる）。`mute` は「ON＝黙っている」なので初期値は False。
ITEMS = (
    ("pause", "停止/再開", "稼働中", "停止中", True),
    ("mute", "ミュート", "ミュート中", "音あり", False),
    ("autospeech", "自律発話", "自律発話 ON", "自律発話 OFF", True),
    ("vision", "画面認識", "画面認識 ON", "画面認識 OFF", True),
    ("task", "タスク", "タスク ON", "タスク OFF", True),
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
        for key, _label, _on_text, _off_text, initial in ITEMS:
            b = QPushButton()
            b.clicked.connect(lambda _=False, k=key: self._clicked(k))
            self.buttons[key] = b
            self._on[key] = initial
            self.set_state(key, "on" if initial else "off")
            row.addWidget(b)
        row.addStretch(1)
        self.settings = QPushButton("設定")
        self.settings.clicked.connect(self.settings_clicked.emit)
        self.settings.setStyleSheet(menu_button_qss("on"))
        row.addWidget(self.settings)

    def is_on(self, key: str) -> bool:
        return bool(self._on.get(key, False))

    def _clicked(self, key: str) -> None:
        if self._state.get(key) == "disabled":
            return  # 構成で使えない機能は押させない
        if self._state.get(key) == "pending":
            return  # 切替中の連打は無視（意図は loop 側で latest-wins）
        self.requested.emit(key, not self._on.get(key, False))

    def set_state(self, key: str, state: str, *, on: bool | None = None) -> None:
        """loop から確定した状態を反映する。**UI から勝手に変えない**。"""
        b = self.buttons.get(key)
        if b is None:
            return
        if on is not None:
            self._on[key] = on
        self._state[key] = state
        b.setStyleSheet(menu_button_qss(state))
        on_text, off_text = dict((k, (o, f)) for k, _l, o, f, _i in ITEMS)[key]
        b.setText("切替中…" if state == "pending" else (on_text if self._on[key] else off_text))
        b.setEnabled(state != "disabled")
