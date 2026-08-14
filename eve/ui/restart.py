"""アプリを起動し直す（設定タブの［⟳ 再起動］）。

## ここで一番危ないこと: **環境変数を引き継ぐと `.env` が効かない**
`eve/config.py` の `_load_dotenv` は `os.environ.setdefault` を使う。つまり
**既に環境変数があれば `.env` の値は無視される**。今のプロセスは起動時に `.env` を
全部 `os.environ` へ入れてしまっているので、素直に `subprocess` を起こすと子は
**古い値を全部引き継ぎ、保存した設定が1つも効かない**（しかもエラーは出ない）。

→ 子に渡す環境から **`.env` に載っているキーを全部落とす**。落とせば子は `.env` を
読み直す。`.env` に無いキー（PATH など）はそのまま渡す。

## 起動し直す先
`sys.argv` をそのまま使う（`tools/ui_chat.py` でも `python -m ...` でも同じ形で戻る）。
`sys.executable` は venv の python なので、venv も維持される。
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .. import settings_store as store

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def child_env(base: dict | None = None, env_keys: list | None = None) -> dict:
    """子プロセスに渡す環境。**`.env` 由来のキーを取り除いたもの**。

    取り除く判断は「`.env` にそのキーが書いてあるか」だけで行う（値の一致は見ない）。
    値で判定すると、ユーザが OS 環境変数と `.env` に同じ値を書いている時に
    OS 側まで消してしまう……のではなく逆で、**違う値なら消せず古い方が残る**。
    キー名で消せば `.env` が常に勝つ＝設定タブで保存した内容が必ず効く。
    """
    env = dict(os.environ if base is None else base)
    keys = env_keys if env_keys is not None else list(store.read_env())
    for k in keys:
        env.pop(k, None)
    return env


def build_command() -> list:
    """起動し直すコマンド。今のプロセスと同じ形にする。"""
    return [sys.executable, *sys.argv]


def spawn() -> subprocess.Popen | None:
    """新しいプロセスを起こす。**失敗しても例外を投げない**（呼び出し側は終了処理の途中）。"""
    cmd = build_command()
    try:
        # 親が終わっても生き残らせる（親はこの直後に終了する）。
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
                subprocess, "DETACHED_PROCESS", 0)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=child_env(),
                                creationflags=flags, close_fds=True)
        logger.info("⟳ 新しいプロセスを起動しました（pid=%s）", proc.pid)
        return proc
    except Exception:
        logger.exception("再起動に失敗しました（このプロセスはそのまま終了します）")
        return None
