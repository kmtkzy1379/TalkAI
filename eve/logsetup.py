"""ロギング設定。

各モジュールは `logger = logging.getLogger(__name__)` を使う。例外は print でなく
logger.warning / logger.exception（traceback 付き＝デバッグしやすい）で出す。
エントリ/ツールが `configure()` を1回呼ぶ。

方針（例外時の継続/停止）:
- 起こりやすい・軽微で完全に潰せる失敗（TTS一時失敗 / LLM一時エラー）→ ログして継続。
- 想定外で頻発しない失敗 → 無理に継続させない（連続失敗は循環ブレーカで停止）。
"""
from __future__ import annotations

import logging


def configure(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 外部ライブラリの INFO ノイズを抑制（会話ログ 🧑/🤖/⏸ を見やすくする）
    for noisy in ("httpx", "LiteLLM", "litellm", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
