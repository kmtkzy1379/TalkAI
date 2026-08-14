"""設定タブ（Phase 2・基本設定）。

## 設定と高級設定の分け方（裁定 2026-08-15）
基準は「よく変えるか」ではなく **「間違いに気づけるか」**。

- **設定** = 好みや状況で変えるもので、間違えても**すぐ耳や目で分かる**（声・発話頻度・
  画面まわり・体感が変わるモデル4つ・API キー）。
- **高級設定**（Phase 3）= 間違えると**静かに劣化する**（実測で校正した閾値）・**課金が増える**
  （画面認識のコスト）・**起動を壊す**（接続先）もの。

`.env` にはあるが**UI に出さないもの**もある: 実測校正済みの重み/閾値（動かす判断材料が
ユーザ側に無い）、消費者ゼロの設定（`VLM_MERGE_MODEL` / `YOUTUBE_*` / `VTS_PATH`＝出すと
「設定したのに効かない」嘘になる）、記憶ファイル名（変えると記憶が消えたように見える）。

## API キーを画面に出さない理由
イブ自身の画面認識が**画面を Gemini に送る**（`VLM_ENABLED=1` のとき）。配信中は
キャプチャ除外を切る運用なので OBS にも乗る。読み返す用途はほぼ無く入れ替えるだけなので、
**保存済みの値は一度も表示しない**（状態と末尾4文字だけ）。「表示」ボタンは作らない。

## 反映のタイミング
**保存しても即時には反映しない**（ユーザ要件）。`.env` に書くだけで、次の起動から効く。
右上の［再起動］でその場で入れ直せる。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..config import Config
from .. import settings_store as store
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

# モデルの候補。**自由入力もできる**（新しいモデルが出た時に UI 改修を待たない）。
MODEL_CHOICES = (
    "openai/gpt-5.5", "openai/gpt-5.4-mini", "openai/gpt-4o", "openai/gpt-4o-mini",
    "gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash",
    "groq/llama-3.3-70b-versatile",
)

# 設定タブに出すモデル役（体感が変わる4つ）。残りは高級設定へ。
# `vlm_merge` / `youtube` は**消費者ゼロ**なので出さない（出すと効かない設定になる）。
MODEL_ROLES = (
    ("RESPONSE_MODEL", "応答", "あなたと話す本体。ここが速さと賢さを決める"),
    ("DECIDE_MODEL", "発話判定", "自分から話しかけるかを決める。安いモデルでよい"),
    ("TASK_MODEL", "タスク", "調べ物や作業を進める。賢いモデルほど完遂しやすい"),
    ("VLM_LEAF_MODEL", "画面認識", "画面を読む。画像を扱えるモデルにすること"),
)

API_KEYS = (
    ("OPENAI_API_KEY", "OpenAI", "応答・STT・埋め込み"),
    ("GEMINI_API_KEY", "Gemini", "画面認識"),
    ("GROQ_API_KEY", "Groq", "STT の代替"),
    ("ANTHROPIC_API_KEY", "Anthropic", "現在停止中（未使用）"),
)


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


class SettingsPage(QWidget):
    """`.env` を読み込んで表示し、保存して「再起動してください」と伝えるだけの画面。"""

    back_clicked = Signal()
    restart_requested = Signal()
    saved = Signal(list)          # 保存できた（変わったキー）
    save_failed = Signal(str)     # 保存できなかった（理由・**値は含めない**）

    def __init__(self) -> None:
        super().__init__()
        self._widgets: dict[str, object] = {}   # env キー -> 入力ウィジェット
        self._keys: dict[str, QLineEdit] = {}   # API キーの入力欄（書き込み専用）
        self._key_labels: dict[str, QLabel] = {}
        self._env = store.read_env()

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.addLayout(self._head())
        self.notice = _hint("")
        self.notice.setObjectName("hint")
        root.addWidget(self.notice)
        root.addWidget(self._voice())
        root.addWidget(self._autospeech())
        root.addWidget(self._window())
        root.addWidget(self._models())
        root.addWidget(self._api_keys())
        root.addWidget(self._advanced_placeholder())
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

    @staticmethod
    def _box(title: str, note: str = "") -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        if note:
            lay.addWidget(_hint(note))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        lay.addLayout(form)
        return box, form

    # --- 各セクション -----------------------------------------------------

    def _voice(self) -> QWidget:
        box, form = self._box("声", "VOICEVOX の合成パラメータ。話者は VOICEVOX の話者IDです。")
        self._add_spin(form, "VV_SPEAKER", "話者ID", 0, 200, 1)
        self._add_dspin(form, "VV_SPEED", "話す速さ", 0.50, 2.00, 0.05,
                        "1.00 が標準。大きいほど速い")
        self._add_dspin(form, "VV_PITCH", "声の高さ", -0.15, 0.15, 0.01,
                        "0.00 が標準。VOICEVOX の上限が ±0.15")
        self._add_dspin(form, "VV_INTONATION", "抑揚", 0.00, 2.00, 0.05,
                        "1.00 が標準。0 に近いほど平坦、大きいほど起伏が強い")
        return box

    def _autospeech(self) -> QWidget:
        box, form = self._box(
            "自分から話しかける頻度",
            "起動した時の設定です。動かしている間はメニューの［自律発話］で切り替えられます。")
        self.freq = QComboBox()
        for label, _sec, note in FREQUENCY:
            self.freq.addItem(f"{label}　—　{note}")
        form.addRow("頻度", self.freq)
        return box

    def _window(self) -> QWidget:
        box, form = self._box("ウィンドウと画面表示")
        self._add_check(form, "WS_VIEWER_MODE", "イブの表示物をタスクウィンドウ内に集める",
                        on="window", off="external",
                        note="切ると、これまでどおり画面の好きな位置に別ウィンドウで開きます")
        self._add_check(form, "UI_ALWAYS_ON_TOP", "このウィンドウを常に最前面にする")
        self._add_check(
            form, "UI_HIDE_FROM_CAPTURE", "このウィンドウを画面キャプチャから隠す",
            note="既定は ON。切ると、イブの画面認識が**自分のログと自分の顔**を"
                 "「ユーザの画面」として実況し始めます。配信で映したい時だけ切ってください")
        return box

    def _models(self) -> QWidget:
        box, form = self._box(
            "モデル",
            "`provider/model` の形式で書きます。一覧に無いものは直接入力できます。")
        for key, label, note in MODEL_ROLES:
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(MODEL_CHOICES)
            combo.setToolTip(note)
            self._widgets[key] = combo
            form.addRow(label, combo)
            form.addRow("", _hint(note))
        return box

    def _api_keys(self) -> QWidget:
        box, form = self._box(
            "API キー",
            "**保存されている値は表示しません**（イブの画面認識が画面を外部に送るため）。"
            "入れ替えたい時だけ貼り付けて保存してください。空のまま保存すると、"
            "そのキーは今のままです。")
        for key, label, note in API_KEYS:
            state = QLabel("")
            state.setWordWrap(True)
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("新しいキーを貼り付け（変更しないなら空のまま）")
            self._keys[key] = edit
            self._key_labels[key] = state
            form.addRow(label, edit)
            form.addRow("", state)
            form.addRow("", _hint(note))
        return box

    def _advanced_placeholder(self) -> QWidget:
        box = QGroupBox("高級設定")
        lay = QVBoxLayout(box)
        lay.addWidget(_hint(
            "機能の起動時 ON/OFF・音声入力（VAD）・画面認識のコスト・記憶・タスクと検索・接続先。"
            "**次の段階で実装します。**"))
        return box

    # --- 入力ウィジェットの作成 -------------------------------------------

    def _add_spin(self, form, key, label, lo, hi, step) -> None:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        self._widgets[key] = w
        form.addRow(label, w)

    def _add_dspin(self, form, key, label, lo, hi, step, note="") -> None:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        self._widgets[key] = w
        form.addRow(label, w)
        if note:
            form.addRow("", _hint(note))

    def _add_check(self, form, key, label, *, on="1", off="0", note="") -> None:
        w = QCheckBox(label)
        w.setProperty("on_value", on)
        w.setProperty("off_value", off)
        self._widgets[key] = w
        form.addRow("", w)
        if note:
            form.addRow("", _hint(note))

    # --- 読み込み / 保存 ---------------------------------------------------

    def reload(self) -> None:
        """`.env` と現在の Config から画面を作り直す（「変更を捨てる」でも使う）。"""
        self._env = store.read_env()
        defaults = {
            "VV_SPEAKER": Config.VV_SPEAKER, "VV_SPEED": Config.VV_SPEED,
            "VV_PITCH": Config.VV_PITCH, "VV_INTONATION": Config.VV_INTONATION,
            "WS_VIEWER_MODE": Config.WS_VIEWER_MODE,
            "UI_ALWAYS_ON_TOP": "1" if Config.UI_ALWAYS_ON_TOP else "0",
            "UI_HIDE_FROM_CAPTURE": "1" if Config.UI_HIDE_FROM_CAPTURE else "0",
        }
        for key, w in self._widgets.items():
            raw = self._env.get(key, "")
            if isinstance(w, QCheckBox):
                val = raw or str(defaults.get(key, "0"))
                w.setChecked(val == w.property("on_value"))
            elif isinstance(w, QComboBox):
                w.setCurrentText(raw or self._model_default(key))
            elif isinstance(w, QSpinBox):
                w.setValue(int(float(raw or defaults.get(key, 0))))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(raw or defaults.get(key, 0.0)))

        # 頻度: AUTOSPEECH_ENABLED=0 なら「話さない」、それ以外は閾値がいちばん近いものを選ぶ。
        if (self._env.get("AUTOSPEECH_ENABLED", "1") == "0") or not Config.AUTOSPEECH_ENABLED:
            self.freq.setCurrentIndex(len(FREQUENCY) - 1)
        else:
            cur = float(self._env.get("SILENCE_THRESHOLD_SEC") or Config.SILENCE_THRESHOLD_SEC)
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
            lab.setStyleSheet(
                "color:#b3261e;" if state == "none" else f"color:{SUB};")
        self.status.setText("")
        self._set_notice(False)

    @staticmethod
    def _model_default(key: str) -> str:
        from ..model_registry import ROLE_ENV

        for _role, (env_key, default) in ROLE_ENV.items():
            if env_key == key:
                return default
        return ""

    def collect(self) -> dict[str, str]:
        """画面 → `.env` に書く値。**API キーは空欄なら含めない**（消さない）。"""
        out: dict[str, str] = {}
        for key, w in self._widgets.items():
            if isinstance(w, QCheckBox):
                out[key] = str(w.property("on_value") if w.isChecked()
                               else w.property("off_value"))
            elif isinstance(w, QComboBox):
                out[key] = w.currentText().strip()
            elif isinstance(w, QSpinBox):
                out[key] = self._numeric(key, w.value(), str(w.value()))
            elif isinstance(w, QDoubleSpinBox):
                out[key] = self._numeric(key, w.value(), f"{w.value():.2f}")
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

    def save(self) -> None:
        try:
            updates = self.collect()
        except Exception as e:
            logger.exception("設定の読み取りに失敗")
            self._fail(f"設定を読み取れませんでした（{type(e).__name__}）")
            return
        try:
            changed = store.write_env(updates)
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
        self.saved.emit(changed)

    def _fail(self, reason: str) -> None:
        self.status.setText(f"保存できませんでした — {reason}")
        self.status.setStyleSheet("color:#b3261e; font-weight:bold;")
        self.save_failed.emit(reason)

    def _set_notice(self, pending: bool) -> None:
        if pending:
            self.notice.setText(
                "⚠ 保存した内容は**まだ動いていません**。右上の［⟳ 再起動］で反映されます。")
            self.notice.setStyleSheet(
                f"color:#9a6700; border:1px solid #e8d08a; border-radius:8px; padding:8px;")
        else:
            self.notice.setText(
                "設定は保存しても即座には反映されません（次の起動から効きます）。"
                "右上の［⟳ 再起動］でその場で入れ直せます。")
            self.notice.setStyleSheet(f"color:{SUB}; border:1px solid {LINE};"
                                      "border-radius:8px; padding:8px;")
