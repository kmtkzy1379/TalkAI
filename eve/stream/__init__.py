"""YouTube 配信モード（OBS で配信し、視聴者のチャットを入力にして話す）。本体は `manager.StreamManager`。"""
from .manager import StreamError, StreamManager, StreamSettings, format_comment

__all__ = ["StreamError", "StreamManager", "StreamSettings", "format_comment"]
