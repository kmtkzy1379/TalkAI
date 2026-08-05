"""J-3 inc1 — 作業フォルダ能力の登録（**TaskAgent 専用**・応答LLM には提示しない）。

`eve/search/capability.py` を前例に同型で書く（新しい作法を増やさない）。

【description の1文目が仕様になる】
`registry.py:146` の `outward_actions()` が **1文目を「。」で切り、括弧の補足を落として**
そのままユーザ向けの発話文脈（`# 実際に実行できる手段`）に載せる。したがって:
- 1文目は**ユーザに読ませてよい日本語**にする（実装語は括弧の中へ）
- 1文目に括弧を使わない（中身が落ちて文が壊れる）
- 必ず「。」で終える

【同期 handler にしている理由】
`async_handler` にすると `registry.is_blocking()` が True になり、`agent.py:150` の
「重い能力は1 step 1件まで」ゲートにファイル操作まで乗る。「読む→書く→動かす」で
`TASK_AGENT_MAX_STEPS` を浪費するため、ローカル FS は同期のままにして書込は文字数上限で
担保する（`store.py` の設計メモ参照）。
"""
from __future__ import annotations

import logging

from ..capability.registry import Capability, CapabilityRegistry
from .paths import WorkspacePathError
from .store import Workspace

logger = logging.getLogger(__name__)


def _as_bool(v) -> bool:
    """LLM 由来 bool の厳密パース（`bool("false") == True` の罠を塞ぐ・search と同型）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def _guard(fn):
    """封じ込め違反を**マーカ文字列に変換**する（例外のまま registry へ投げない）。

    実機検証(2026-08-05)で判明: `registry.execute()` は例外を `logger.exception` で拾うため、
    LLM が悪いパスを1回試すたびにスタックトレースが1本ログに出る。封じ込め違反は
    **想定内の拒否**であって異常ではないので、トレースは出さず1行の warning に留める。
    マーカ（先頭「（」）は維持＝`executor.py:110` の verdict 判定は従来どおり Failed 側。
    返す文言も registry の定型（「機能「x」の実行に失敗しました: WorkspacePathError: …」）より
    短く具体的になり、LLM が次の手を選びやすい。
    """
    def wrapped(args: dict) -> str:
        try:
            return fn(args or {})
        except WorkspacePathError as e:
            logger.warning("🗂 作業フォルダの封じ込めで拒否: %.80s", e)
            return f"（{e}）"
    return wrapped


def register_workspace_capabilities(registry: CapabilityRegistry, ws: Workspace) -> None:
    def _write(args: dict) -> str:
        return ws.write((args or {}).get("path", ""), (args or {}).get("content", ""),
                        _as_bool((args or {}).get("append")))

    def _read(args: dict) -> str:
        return ws.read((args or {}).get("path", ""), (args or {}).get("max_chars"))

    def _list(args: dict) -> str:
        return ws.list((args or {}).get("path") or None)

    def _delete(args: dict) -> str:
        return ws.delete((args or {}).get("path", ""), _as_bool((args or {}).get("recursive")))

    registry.register(Capability(
        name="write_file",
        description="自分の作業フォルダにファイルを作る・書き換える。"
                    "path は作業フォルダからの相対パス（例: memo.txt / notes/todo.md）。"
                    "append=true で末尾に追記、省略すると上書き。",
        params_schema={
            "path": {"type": "string", "description": "作業フォルダからの相対パス"},
            "content": {"type": "string", "description": "書き込む内容"},
            "append": {"type": "boolean", "description": "末尾に追記する（省略=上書き）"},
        },
        handler=_guard(_write), offered=False, agent_tool=True, mutates_state=True,
    ))
    registry.register(Capability(
        name="read_file",
        description="自分の作業フォルダのファイルを読む。"
                    "長いファイルは途中までしか返らない（max_chars で調整可）。",
        params_schema={
            "path": {"type": "string", "description": "作業フォルダからの相対パス"},
            "max_chars": {"type": "integer", "description": "最大何文字読むか（省略=既定値）"},
        },
        handler=_guard(_read), offered=False, agent_tool=True,
    ))
    registry.register(Capability(
        name="list_files",
        description="自分の作業フォルダの中身を見る。"
                    "path を省略すると一番上の階層。使用容量も一緒に返る。",
        params_schema={"path": {"type": "string", "description": "見たいフォルダの相対パス（省略可）"}},
        handler=_guard(_list), offered=False, agent_tool=True,
    ))
    registry.register(Capability(
        name="delete_file",
        description="自分の作業フォルダのファイルやフォルダを消す。"
                    "フォルダを中ごと消すときだけ recursive=true を付ける。",
        params_schema={
            "path": {"type": "string", "description": "作業フォルダからの相対パス"},
            "recursive": {"type": "boolean", "description": "フォルダを中身ごと消す（省略=false）"},
        },
        handler=_guard(_delete), offered=False, agent_tool=True, mutates_state=True,
    ))
