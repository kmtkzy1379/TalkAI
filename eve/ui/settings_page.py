"""設定タブ。**何を出すかは `settings_spec.py`**、ここは並べ方と読み書きだけを持つ。

## 設定と高級設定の分け方（裁定 2026-08-15）
基準は「よく変えるか」ではなく **「間違いに気づけるか」**。

- **設定**（常時表示）= 好みや状況で変えるもので、間違えても**すぐ耳や目で分かる**
  （声・発話頻度・画面まわり・体感が変わるモデル4役・API キー）。
- **高級設定**（折りたたみ）= 間違えると**静かに劣化する**（実測で校正した値）・
  **課金が増える**（画面認識のコスト）・**起動を壊す**（接続先）もの。

出さない項目とその理由は `settings_spec.py` の docstring にある。

## API キーを画面に出さない理由
イブ自身の画面認識が**画面を Gemini に送る**（`VLM_ENABLED=1` のとき）。配信中は
キャプチャ除外を切る運用なので OBS にも乗る。読み返す用途はほぼ無く入れ替えるだけなので、
**保存済みの値は一度も表示しない**（状態と末尾4文字だけ）。「表示」ボタンは作らない。

## 反映のタイミング
**保存しても即時には反映しない**（ユーザ要件）。`.env` に書くだけで、次の起動から効く。
右上の［⟳ 再起動］でその場で入れ直せる。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .. import settings_store as store
from ..config import Config
from .accordion import Accordion
from .settings_spec import ADVANCED, BASIC, Group, Item, MODEL_CHOICES, check_consistency
from .theme import LINE, SUB

logger = logging.getLogger(__name__)

# 自律発話の頻度。**秒数は見せない**（ユーザ要件「抽象化」）。
# 「ふつう」= 現行既定 5.0 秒（実測で 1〜2分に1回＝ユーザが理想と言った頻度）。
FREQUENCY = (
    ("よく話す", "3.0", "沈黙が短くても話しかける"),
    ("ふつう", "5.0", "今の設定（だいたい1〜2分に1回）"),
    ("控えめ", "12.0", "こちらが黙っている時間が長い時だけ"),
    ("自分からは話さない", "", "聞かれたら答えるのは普通にできる"),
)

# 設定タブに出すモデル役（`settings_spec.BASIC` から導出＝二重管理しない）。
MODEL_ROLES = tuple(
    (item.key, item.label, item.note)
    for group in BASIC if group.title == "モデル" for item in group.items
)

API_KEYS = (
    ("OPENAI_API_KEY", "OpenAI", "応答・STT・埋め込み"),
    ("GEMINI_API_KEY", "Gemini", "画面認識"),
    ("GROQ_API_KEY", "Groq", "STT の代替"),
    ("ANTHROPIC_API_KEY", "Anthropic", "現在停止中（未使用）"),
)


def _hint(text: str, *, warn: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    if warn:
        lab.setStyleSheet("color:#b3261e;")
    return lab


class SettingsPage(QWidget):
    """`.env` を読み込んで表示し、保存して「再起動してください」と伝えるだけの画面。"""

    back_clicked = Signal()
    restart_requested = Signal()
    saved = Signal(list)          # 保存できた（変わったキー）
    save_failed = Signal(str)     # 保存できなかった（理由・**値は含めない**）

    def __init__(self) -> None:
        super().__init__()
        self._widgets: dict[str, QWidget] = {}   # env キー -> 入力ウィジェット
        self._items: dict[str, Item] = {}        # env キー -> 定義
        self._keys: dict[str, QLineEdit] = {}    # API キーの入力欄（書き込み専用）
        self._key_labels: dict[str, QLabel] = {}
        self.accordions: list[Accordion] = []
        self._env = store.read_env()

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.addLayout(self._head())
        self.notice = _hint("")
        root.addWidget(self.notice)

        # 設定（常時表示）。頻度は2キーに割れる特殊項目なので声の直後に手で挟む。
        for group in BASIC:
            root.addWidget(self._group_box(group))
            if group.title == "声":
                root.addWidget(self._autospeech())
        root.addWidget(self._api_keys())

        adv_head = QLabel("高級設定")
        adv_head.setStyleSheet("font-size:15px; font-weight:bold; margin-top:6px;")
        root.addWidget(adv_head)
        root.addWidget(_hint(
            "間違えても気づきにくいもの（静かに劣化する値・API の請求に効く値・"
            "起動を壊す値）をここにまとめています。**既定のままで問題ありません。**"))
        for group in ADVANCED:
            acc = Accordion(group.title, self._group_box(group, flat=True))
            self.accordions.append(acc)
            root.addWidget(acc)

        root.addStretch(1)
        root.addLayout(self._footer())
        self.reload()

    # --- 見出しと足回り ---------------------------------------------------

    def _head(self) -> QHBoxLayout:
        row = QHBoxLayout()
        back = QPushButton("← メインに戻る")
        back.clicked.connect(self.back_clicked.emit)
        title = QLabel("設定")
        title.setStyleSheet("font-size:17px; font-weight:bold;")
        # **再起動は右上**（ユーザ指定）。設定は再起動で反映されるので、保存の隣ではなくここ。
        self.restart_btn = QPushButton("⟳ 再起動")
        self.restart_btn.setToolTip("パイプラインを安全に止めてから、新しい設定で起動し直します")
        self.restart_btn.clicked.connect(self.restart_requested.emit)
        row.addWidget(back)
        row.addWidget(title, 1)
        row.addWidget(self.restart_btn)
        return row

    def _footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        revert = QPushButton("変更を捨てる")
        revert.clicked.connect(self.reload)
        save = QPushButton("保存")
        save.setDefault(True)
        save.clicked.connect(self.save)
        row.addWidget(self.status, 1)
        row.addWidget(revert)
        row.addWidget(save)
        return row

    # --- 定義表 → ウィジェット --------------------------------------------

    def _group_box(self, group: Group, *, flat: bool = False) -> QWidget:
        box = QGroupBox("" if flat else group.title)
        if flat:
            box.setStyleSheet("QGroupBox{border:none; margin:0; padding:0 0 8px 10px;}")
        lay = QVBoxLayout(box)
        if group.note:
            lay.addWidget(_hint(group.note))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        for item in group.items:
            self._add(form, item)
        lay.addLayout(form)
        return box

    def _add(self, form: QFormLayout, item: Item) -> None:
        w = self._make(item)
        self._widgets[item.key] = w
        self._items[item.key] = item
        w.setToolTip(item.note or item.label)
        if item.kind == "bool":
            form.addRow(w)          # チェックボックスは文言を自分で持つ
        else:
            form.addRow(item.label, w)
        # 説明は**2列にまたがる1行**として置く（`addRow("", ...)` にすると空のラベル欄が
        # 幅を取り、項目ごとに字下げがばらつく＝実起動のスクショで確認）。
        if item.note:
            form.addRow(_hint(item.note))
        if item.warn:
            form.addRow(_hint(f"⚠ {item.warn}", warn=True))

    @staticmethod
    def _make(item: Item) -> QWidget:
        if item.kind == "bool":
            w = QCheckBox(item.label)
            w.setProperty("on_value", item.on)
            w.setProperty("off_value", item.off)
            return w
        if item.kind == "int":
            w = QSpinBox()
            w.setRange(int(item.lo), int(item.hi))
            w.setSingleStep(max(1, int(item.step)))
            return w
        if item.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(item.lo, item.hi)
            w.setSingleStep(item.step)
            w.setDecimals(2)
            return w
        if item.kind == "choice":
            w = QComboBox()
            w.setEditable(item.editable)
            w.addItems(list(item.choices))
            return w
        w = QLineEdit()
        return w

    # --- 特殊項目 ---------------------------------------------------------

    def _autospeech(self) -> QWidget:
        box = QGroupBox("自分から話しかける頻度")
        lay = QVBoxLayout(box)
        lay.addWidget(_hint(
            "起動した時の設定です。動かしている間はメニューの［自律発話］で切り替えられます。"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self.freq = QComboBox()
        for label, _sec, note in FREQUENCY:
            self.freq.addItem(f"{label}　—　{note}")
        form.addRow("頻度", self.freq)
        lay.addLayout(form)
        return box

    def _api_keys(self) -> QWidget:
        box = QGroupBox("API キー")
        lay = QVBoxLayout(box)
        lay.addWidget(_hint(
            "**保存されている値は表示しません**（イブの画面認識が画面を外部に送るため）。"
            "入れ替えたい時だけ貼り付けて保存してください。空のまま保存すると、"
            "そのキーは今のままです。"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        for key, label, note in API_KEYS:
            state = QLabel("")
            state.setWordWrap(True)
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("新しいキーを貼り付け（変更しないなら空のまま）")
            self._keys[key] = edit
            self._key_labels[key] = state
            form.addRow(label, edit)
            form.addRow(state)
            form.addRow(_hint(note))
        lay.addLayout(form)
        return box

    # --- 読み込み ---------------------------------------------------------

    def reload(self) -> None:
        """`.env` と現在の Config から画面を作り直す（「変更を捨てる」でも使う）。"""
        self._env = store.read_env()
        for key, w in self._widgets.items():
            default = self._default_of(key)
            raw = self._env.get(key) or default
            try:
                self._apply(w, self._items[key], raw)
            except (TypeError, ValueError):
                # `.env` 側が壊れている（数値のはずが文字列など）。**既定に戻して表示する**。
                # 落とすと設定タブ全体が開かなくなり、直す手段を奪う。放置すると入力欄の
                # 最小値が入って見え、そのまま保存すると壊れた値が別の壊れ方に変わる。
                logger.warning("設定 %s の値を解釈できませんでした（既定で表示します）", key)
                try:
                    self._apply(w, self._items[key], default)
                except (TypeError, ValueError):
                    logger.warning("設定 %s は既定値も解釈できませんでした", key)

        # 頻度: AUTOSPEECH_ENABLED=0 なら「話さない」、それ以外は閾値がいちばん近いものを選ぶ。
        if (self._env.get("AUTOSPEECH_ENABLED", "1") == "0") or not Config.AUTOSPEECH_ENABLED:
            self.freq.setCurrentIndex(len(FREQUENCY) - 1)
        else:
            try:
                cur = float(self._env.get("SILENCE_THRESHOLD_SEC")
                            or Config.SILENCE_THRESHOLD_SEC)
            except ValueError:
                cur = float(Config.SILENCE_THRESHOLD_SEC)
            best = min(range(len(FREQUENCY) - 1),
                       key=lambda i: abs(float(FREQUENCY[i][1]) - cur))
            self.freq.setCurrentIndex(best)

        for key, lab in self._key_labels.items():
            self._keys[key].clear()
            state, fp = store.key_status(key)
            if state == "env":
                lab.setText(f"OS の環境変数で設定済み（{fp}）"
                            "　※こちらが優先されるので、この画面から変えても効きません")
                self._keys[key].setEnabled(False)
            elif state == "file":
                lab.setText(f"設定済み（{fp}）")
                self._keys[key].setEnabled(True)
            else:
                lab.setText("未設定")
                self._keys[key].setEnabled(True)
            lab.setStyleSheet("color:#b3261e;" if state == "none" else f"color:{SUB};")
        self.status.setText("")
        self._set_notice(False)

    @staticmethod
    def _apply(w: QWidget, item: Item, raw: str) -> None:
        """文字列 → ウィジェット。解釈できなければ例外を投げる（呼び手が既定へ戻す）。"""
        if item.kind == "bool":
            w.setChecked(raw == item.on)
        elif item.kind == "int":
            w.setValue(int(float(raw)))
        elif item.kind == "float":
            w.setValue(float(raw))
        elif item.kind == "choice":
            w.setCurrentText(raw)
        else:
            w.setText(raw)

    @staticmethod
    def _default_of(key: str) -> str:
        """既定値は `Config` と `ROLE_ENV` から**実行時に引く**（書き写して腐らせない）。"""
        from ..model_registry import ROLE_ENV

        for _role, (env_key, default) in ROLE_ENV.items():
            if env_key == key:
                return default
        if hasattr(Config, key):
            v = getattr(Config, key)
            if isinstance(v, bool):
                return "1" if v else "0"
            return str(v)
        return ""

    # --- 保存 -------------------------------------------------------------

    def collect(self) -> dict[str, str]:
        """画面 → `.env` に書く値。**API キーは空欄なら含めない**（消さない）。"""
        out: dict[str, str] = {}
        for key, w in self._widgets.items():
            item = self._items[key]
            if item.kind == "bool":
                out[key] = str(item.on if w.isChecked() else item.off)
            elif item.kind == "int":
                out[key] = self._numeric(key, w.value(), str(w.value()))
            elif item.kind == "float":
                out[key] = self._numeric(key, w.value(), f"{w.value():.2f}")
            elif item.kind == "choice":
                out[key] = w.currentText().strip()
            else:
                out[key] = w.text().strip()
        return self._with_extras(out)

    def _numeric(self, key: str, value: float, formatted: str) -> str:
        """数値が**動いていないなら `.env` の表記をそのまま返す**。

        触っていない `VV_SPEED=1.0` を `1.00` に書き換えると、値は同じなのに差分が出る。
        ユーザが手で書いたファイルなので、意味が変わらない書き換えはしない。
        """
        raw = self._env.get(key, "")
        try:
            if raw and float(raw) == float(value):
                return raw
        except ValueError:
            pass  # `.env` 側が数値として壊れている → 正しい表記で上書きしてよい
        return formatted

    def _with_extras(self, out: dict[str, str]) -> dict[str, str]:
        """頻度（2キーに割れる）と API キー（空欄なら書かない）を足す。"""
        idx = self.freq.currentIndex()
        if idx == len(FREQUENCY) - 1:
            out["AUTOSPEECH_ENABLED"] = "0"   # 閾値は触らない（ON に戻した時に元へ返す）
        else:
            out["AUTOSPEECH_ENABLED"] = "1"
            out["SILENCE_THRESHOLD_SEC"] = self._numeric(
                "SILENCE_THRESHOLD_SEC", float(FREQUENCY[idx][1]), FREQUENCY[idx][1])
        for key, edit in self._keys.items():
            if not edit.isEnabled():
                continue  # OS 環境変数が優先されている＝ここから書いても効かない
            v = edit.text().strip()
            if v:
                out[key] = v
        return out

    def _only_meaningful(self, updates: dict[str, str]) -> dict[str, str]:
        """**既定のままの項目は `.env` に書かない**（既定を固定してしまわないため）。

        高級設定は 45 項目あり、そのほとんどは `.env` に書かれていない＝コード側の既定で
        動いている。何気なく［保存］を押しただけで全部が `.env` に焼き付くと、
        **後でコード側の既定を改善しても古い値が残り続ける**（しかも本人は設定した覚えが
        無い）。実測: 素の `.env` で初回保存すると 32 件が新規に書かれた。

        規則は2つだけ:
        - 既に `.env` にあるキーは常に反映する（ユーザが明示的に持っている値なので消さない）
        - 無いキーは、**既定と違う時だけ**書く
        """
        secret = {k for k, _l, _n in API_KEYS}
        out: dict[str, str] = {}
        for key, value in updates.items():
            if key in secret or key in self._env:
                out[key] = value
                continue
            if not self._is_default(key, value):
                out[key] = value
        return out

    def _is_default(self, key: str, value: str) -> bool:
        """既定と同じ値か。**数値は数として比べる**（`2.0` と `2.00` は同じ）。

        文字列だけで比べると、画面が `.2f` で整える float 項目が全部「変更あり」になり、
        既定を固定しない意図が丸ごと効かなくなる（実測: 15 件が余計に書かれた）。
        """
        default = self._default_of(key)
        item = self._items.get(key)
        if item is not None and item.kind in ("int", "float"):
            try:
                return float(default) == float(value)
            except ValueError:
                return False
        return default == value

    def save(self) -> None:
        try:
            updates = self.collect()
        except Exception as e:
            logger.exception("設定の読み取りに失敗")
            self._fail(f"設定を読み取れませんでした（{type(e).__name__}）")
            return
        # 「ON にしたのに前提が OFF」を保存**前**に言う。起動時は fail-soft で無効化されるだけ
        # なので、ここで言わないと「設定したのに効かない」で詰まる。
        # 判定は**画面の全項目**で行う（実際に書くのは下で絞るが、整合は全体で見る）。
        warnings = check_consistency(updates)
        try:
            changed = store.write_env(self._only_meaningful(updates))
        except store.InvalidValue as e:
            self._fail(str(e))       # 例外側が値を載せない契約
            return
        except OSError as e:
            logger.exception("`.env` の書き込みに失敗")
            self._fail(f"`.env` に書き込めませんでした（{e.strerror or type(e).__name__}）")
            return
        # `.env.example` にも足りないキーを補う（CLAUDE.md: 全フラグを明記する）。
        # **秘密は渡さない**（API キーは除外して、値は空にする）。
        try:
            secret = {k for k, _l, _n in API_KEYS}
            store.sync_example({k: ("" if k in secret else v)
                                for k, v in updates.items() if k not in secret})
        except Exception:
            logger.exception("`.env.example` の追記に失敗（保存自体は成功）")
        for edit in self._keys.values():
            edit.clear()  # 入力欄に秘密を残さない
        self.reload()
        if changed:
            self._set_notice(True)
            self.status.setText(f"保存しました（{len(changed)} 件）。**再起動してください**")
            self.status.setStyleSheet("color:#9a6700; font-weight:bold;")
        else:
            self.status.setText("変更はありませんでした")
            self.status.setStyleSheet(f"color:{SUB};")
        if warnings:
            self.status.setText(self.status.text() + "　／　⚠ " + " ".join(warnings))
            self.status.setStyleSheet("color:#b3261e; font-weight:bold;")
            self._open_advanced()  # 直す場所を隠したままにしない
        self.saved.emit(changed)

    def _open_advanced(self) -> None:
        for acc in self.accordions:
            if acc.head.text() == "機能の起動時 ON/OFF":
                acc.set_opened(True)

    def _fail(self, reason: str) -> None:
        self.status.setText(f"保存できませんでした — {reason}")
        self.status.setStyleSheet("color:#b3261e; font-weight:bold;")
        self.save_failed.emit(reason)

    def _set_notice(self, pending: bool) -> None:
        if pending:
            self.notice.setText(
                "⚠ 保存した内容は**まだ動いていません**。右上の［⟳ 再起動］で反映されます。")
            self.notice.setStyleSheet(
                "color:#9a6700; border:1px solid #e8d08a; border-radius:8px; padding:8px;")
        else:
            self.notice.setText(
                "設定は保存しても即座には反映されません（次の起動から効きます）。"
                "右上の［⟳ 再起動］でその場で入れ直せます。")
            self.notice.setStyleSheet(f"color:{SUB}; border:1px solid {LINE};"
                                      "border-radius:8px; padding:8px;")
