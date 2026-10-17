"""X（旧 Twitter）で話題を仕入れる（ユーザ指定 2026-10-17・`X_ENABLED`）。

起動中は6時間ごとに、情報収集AI が選んだ3件を X で読み、イブが感想を持ってから「Xで見た」記憶として別ファイルに残す。
流れと守りは `collector.py`、選び方は `curate.py`、感想は `impression.py`。
"""
from .client import Post, Trend, XApiError, XClient, XRateLimited
from .collector import Materials, XCollector, XSettings, build_collector

__all__ = ["Materials", "Post", "Trend", "XApiError", "XClient", "XCollector", "XRateLimited", "XSettings",
           "build_collector"]
