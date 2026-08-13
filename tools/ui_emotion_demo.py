r"""感情ごとの動きを目で確認するデモ（**テスト専用・本番 UI には載せない**）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\ui_emotion_demo.py

- 全感情を **30秒ずつ**自動で巡回し、**今どの感情かを画面に大きく表示**する。
- ボタンでいつでも切り替えられる（30秒待たなくてよい）。
- surprise スライダで「驚きが大きいほど揺れも頻度も増す」を同時に確認できる。
- **LLM も VOICEVOX も使わない**（VoiceLoop を起動しない）＝API コストゼロ。
  口パクは疑似的なモーラ計画で動かすので、感情ごとの口の形も見える。

`--sec N` で1感情あたりの秒数を変えられる（既定 30）。
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QGridLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

import live2d.v3 as live2d  # noqa: E402
from eve.ui.emotion import PROFILES  # noqa: E402
from eve.ui.live2d_view import Live2DView, assets_ready  # noqa: E402

# 各ラベルの代表的な実出力（本番 rag_memory.jsonl に実在した表現。怒/哀は未出現なので想定語）
SAMPLES: dict[str, str] = {
    "calm": "落ち着き", "relieved": "ほっとした", "warm": "親しみ",
    "happy": "うれしい", "eager": "興味津々", "sorry": "申し訳なさ",
    "worried": "心配", "puzzled": "戸惑い", "watching": "様子見",
    "angry": "怒り", "sad": "悲しい",
}
LABEL_JA: dict[str, str] = {
    "calm": "平静", "relieved": "安心", "warm": "親しみ", "happy": "喜（うれしい）",
    "eager": "興味・期待", "sorry": "申し訳なさ", "worried": "心配",
    "puzzled": "戸惑い", "watching": "様子見", "angry": "怒（いかり）", "sad": "哀（かなしみ）",
}


def fake_plan(seconds: float = 2.2) -> list[tuple[float, float, float, float]]:
    """疑似的な口の開閉計画（VOICEVOX を呼ばずに口を動かすため）。"""
    plan, t = [], 0.0
    vowels = (1.0, 0.42, 0.62, 0.72, 0.34)
    while t < seconds:
        c = random.uniform(0.03, 0.09)
        plan.append((t, t + c, 0.08, 0.08))
        t += c
        v = random.uniform(0.07, 0.16)
        lv = random.choice(vowels)
        plan.append((t, t + v, lv, lv * 0.78))
        t += v
        if random.random() < 0.12:
            plan.append((t, t + 0.28, 0.0, 0.0))
            t += 0.28
    return plan


class Demo(QWidget):
    def __init__(self, sec: float) -> None:
        super().__init__()
        self.setWindowTitle("感情デモ（テスト専用）")
        self.sec = sec
        self.view = Live2DView()
        self.view.setMinimumSize(640, 427)
        self.labels = list(PROFILES)
        self.idx = 0
        self.until = time.monotonic() + sec

        self.title = QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 26px; font-weight: bold; padding: 6px;")
        self.detail = QLabel()
        self.detail.setAlignment(Qt.AlignCenter)
        self.detail.setStyleSheet("font-size: 13px; color: #666;")

        grid = QGridLayout()
        for i, lbl in enumerate(self.labels):
            b = QPushButton(f"{LABEL_JA.get(lbl, lbl)}\n{lbl}")
            b.clicked.connect(lambda _=False, k=lbl: self.jump(k))
            grid.addWidget(b, i // 6, i % 6)

        srow = QVBoxLayout()
        srow.addWidget(QLabel("surprise（0-100・大きいほど揺れも頻度も増す）"))
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setRange(0, 100)
        self.sl.setValue(20)
        self.sl.valueChanged.connect(lambda _: self.apply())
        srow.addWidget(self.sl)

        root = QVBoxLayout(self)
        root.addWidget(self.view, 1)
        root.addWidget(self.title)
        root.addWidget(self.detail)
        root.addLayout(grid)
        root.addLayout(srow)

        self.tick = QTimer(self)
        self.tick.timeout.connect(self._tick)
        self.tick.start(200)
        # 感情ごとの口の形も見たいので、たまに喋らせる（音は出さない）
        self.talk = QTimer(self)
        self.talk.timeout.connect(lambda: self.view.set_plan(fake_plan()))
        self.talk.start(4000)
        self.apply()

    def jump(self, label: str) -> None:
        self.idx = self.labels.index(label)
        self.until = time.monotonic() + self.sec
        self.apply()

    def apply(self) -> None:
        label = self.labels[self.idx]
        self.view.set_mood(SAMPLES[label], self.sl.value())
        m = self.view._motion
        self.title.setText(f"{LABEL_JA.get(label, label)}    ［{label}］    「{SAMPLES[label]}」")
        self.detail.setText(
            f"揺れ {m.sway:.2f}倍 ／ 速さ {m.speed:.2f}倍 ／ うつむき {m.pitch_bias:+.2f} ／ "
            f"首の傾き {m.tilt_bias:+.2f} ／ まばたき {m.blink:.2f}倍 ／ "
            f"よそ見の間隔 {m.glance:.2f}倍 ／ 眉 {m.brow:+.2f}／形 {m.brow_form:+.2f} ／ "
            f"目の笑み {m.eye_smile:.2f} ／ 口角 {m.mouth_form:+.2f}")

    def _tick(self) -> None:
        left = self.until - time.monotonic()
        if left <= 0:
            self.idx = (self.idx + 1) % len(self.labels)
            self.until = time.monotonic() + self.sec
            self.apply()
            left = self.sec
        t = self.title.text().split("  ｜  ")[0]
        self.title.setText(f"{t}  ｜  あと {left:.0f} 秒")

    def closeEvent(self, ev) -> None:
        self.view.release()
        super().closeEvent(ev)


def main() -> int:
    ok, why = assets_ready()
    if not ok:
        print(why)
        return 2
    sec = 30.0
    if "--sec" in sys.argv:
        sec = float(sys.argv[sys.argv.index("--sec") + 1])
    live2d.init()
    try:
        app = QApplication(sys.argv)
        w = Demo(sec)
        w.resize(980, 860)
        w.show()
        return app.exec()
    finally:
        live2d.dispose()


if __name__ == "__main__":
    sys.exit(main())
