r"""UI 付きの実行経路（Live2D + リップシンク）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\ui_chat.py

前提は `tools/voice_chat.py` と同じ（.env / VOICEVOX 起動 / マイク+イヤホン）。
加えて `eve_ui/` に Live2D モデルと背景画像、`pip install PySide6-Essentials live2d-py` が要る。

**ヘッドレス経路 `tools/voice_chat.py` は今後も一級の実行経路として残す**（ユーザ裁定
2026-08-13）。E2E ハーネスは VoiceLoop を直接叩くので UI を通さずに回り続ける。
終了はウィンドウを閉じる（パイプラインを drain してから閉じる）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.ui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
