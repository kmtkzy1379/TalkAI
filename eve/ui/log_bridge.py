"""本体のログを UI のログパネルへ流す（**パイプラインには一切触らない**）。

1-A の原則は「画面の骨格だけを作り、対話の配線は変えない」。会話・タスク・画面認識・内省の
内容は**すでに logger に出ている**（`🧑` `🤖` `🗒` `👁` など）ので、`logging.Handler` を1つ
足して振り分ければ本体を1行も変えずに済む。

**契約**:
- `emit()` は loop スレッドから呼ばれる。**Qt signal を emit するだけ**で戻る（描画は UI 側）。
- 例外を絶対に外へ出さない（`logging` は握るが、握られた例外は握り潰されて原因が消える）。
- 対象を絞る（全ログを流すとログパネルが本体のデバッグ出力で埋まる）。

**既知の限界（1-B で解消する）**: 応答と自発発話はどちらも `🤖` で出るため、ログだけでは
区別できない。色分け（ユーザ要件）は 1-B で loop 側から刺激の種別を渡して行う。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

# ロガー名 → ログパネルのタブ。ここに無いものは流さない（＝ノイズを持ち込まない）。
_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("eve.response.input_source", "chat", "user"),      # 🧑 STT で確定したユーザ発話
    ("eve.response.orchestrator", "chat", "eve"),       # 🤖 イブの発話（応答/自発の区別は 1-B）
    ("eve.task.agent", "task", "system"),               # 🗒 TaskAgent の各ステップ
    ("eve.task.executor", "task", "system"),            # 🗒 タスクの完了/失敗
    ("eve.task.cancel_resolver", "task", "system"),
    ("eve.response.function_dispatcher", "task", "system"),  # ⚙ 能力の実行
    ("eve.vlm.worker", "vision", "system"),             # 画面の実況
    ("eve.vlm.capture_thread", "vision", "system"),
    ("eve.feedback.feedback_llm", "mind", "system"),    # 内省（感情・予測差）
    ("eve.feedback.worker", "mind", "system"),
    ("eve.speech.monitor", "mind", "system"),           # 🔔 発話判定・🔇 抑制
    ("eve.control", "chat", "system"),                  # 🎛 機能 ON/OFF の確定と待ち（会話タブに出す）
    ("eve.stream.chat", "chat", "viewer"),              # 💬 配信のコメント（イブに渡した1件）
    ("eve.stream.manager", "chat", "system"),           # 🔴 配信の開始・終了・チャットの状態
    ("eve.stream.scene", "chat", "system"),             # 🎬 OBS のシーンの用意
)


class LogBridge(QObject, logging.Handler):
    """`logging` → Qt signal。UI スレッドへは queued connection で届く。"""

    line = Signal(str, str, str)  # (タブ, 話者, 本文)

    def __init__(self) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self, level=logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: A003（logging の API 名）
        try:
            for name, tab, who in _ROUTES:
                if record.name == name:
                    text = record.getMessage()
                    if text:
                        self.line.emit(tab, who, text[:400])
                    return
        except Exception:
            pass  # ログ表示のために本体を止めない（handleError も呼ばない＝静かに捨てる）

    def install(self) -> None:
        logging.getLogger("eve").addHandler(self)

    def uninstall(self) -> None:
        logging.getLogger("eve").removeHandler(self)
