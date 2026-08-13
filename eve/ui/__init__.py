"""UI 層（PySide6 + Live2D）。**本体パイプラインからは一方向にしか依存しない**。

`eve/ui/` を import しても `eve/` の他モジュールの挙動は変わらない（フックは既定 None）。
ヘッドレス経路 `tools/voice_chat.py` は今後も一級の実行経路として維持する。
"""
