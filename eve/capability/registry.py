"""J-0 共有 Capability 層（read-only サブセット）。

応答LLM が tool_call で呼べる「能力」を**構造化（enum + 型付き引数）**で1箇所に列挙する。
本 increment は read-only のみ（`self_status` / `pc_status`）＝OS 変更なし・生コマンドなし。
将来 J-0 の状態変更系（download/launch_app/window前面化）や J-2 search はここに足す（3経路が共有）。

- `tool_schemas()`: litellm の `tools=`（OpenAI function-calling schema）形式を返す。
- `execute(name, args)`: 能力を実行し**人間可読の結果文字列**を返す。例外は握って「失敗」文字列
  （応答LLM が「○○に失敗した」と正直に言えるように・落とさない＝CLAUDE.md 規律）。
"""
from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    params_schema: dict  # JSON schema の properties（引数なしなら空 dict）
    handler: Callable[[dict], str]  # (args) -> 人間可読の結果
    mutates_state: bool = False  # 状態を変える書込能力（将来の UI 承認/監査用マーカ）
    offered: bool = True  # 応答LLM に tool として提示するか（False=executor 専用の内部能力）文字列


class CapabilityRegistry:
    """read-only 能力の登録簿。`self_status` は live state を**注入**で読む（loop 所有を直接 import しない）。"""

    def __init__(
        self,
        *,
        is_busy: Optional[Callable[[], bool]] = None,
        qsize: Optional[Callable[[], int]] = None,
        recent_errors: Optional[Callable[[], list[str]]] = None,
    ) -> None:
        self._is_busy = is_busy
        self._qsize = qsize
        self._recent_errors = recent_errors
        self._caps: dict[str, Capability] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(Capability(
            name="self_status",
            description="イブ自身の今の状態（応答中か / 未処理の刺激数 / 直近のエラー）を確認する。引数なし。",
            params_schema={},
            handler=self._self_status,
        ))
        self.register(Capability(
            name="pc_status",
            description="PC の現在状態（現在時刻 / OS / CPUコア数 / メモリ）を確認する。引数なし。",
            params_schema={},
            handler=self._pc_status,
        ))

    def register(self, cap: Capability) -> None:
        self._caps[cap.name] = cap

    def has(self, name: str) -> bool:
        return name in self._caps

    def names(self) -> list[str]:
        return list(self._caps)

    def tool_schemas(self) -> list[dict]:
        """litellm `tools=`（OpenAI function schema）形式。offered=False（内部能力）は除外。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": c.name,
                    "description": c.description,
                    "parameters": {"type": "object", "properties": c.params_schema, "required": []},
                },
            }
            for c in self._caps.values()
            if c.offered
        ]

    def execute(self, name: str, args: Optional[dict] = None) -> str:
        cap = self._caps.get(name)
        if cap is None:
            return f"（未対応の機能「{name}」のため実行しませんでした）"
        try:
            return cap.handler(args or {})
        except Exception as e:
            logger.exception("Capability 実行で例外: %s", name)
            reason = f"{type(e).__name__}: {e}".strip()
            # 失敗理由を短く同梱（応答LLM が「○○が理由で失敗した」と正直に言えるように）。
            return f"（機能「{name}」の実行に失敗しました: {reason[:120]}）"

    # --- read-only handlers -------------------------------------------------

    def _self_status(self, args: dict) -> str:
        busy = self._is_busy() if self._is_busy is not None else None
        qn = self._qsize() if self._qsize is not None else None
        errs = self._recent_errors() if self._recent_errors is not None else []
        parts: list[str] = ["今は応答中" if busy else "今は手が空いている"]
        if qn is not None:
            parts.append(f"未処理の刺激は{qn}件")
        parts.append(f"直近のエラー: {errs[-1]}" if errs else "直近のエラーは無し")
        return " / ".join(parts)

    def _pc_status(self, args: dict) -> str:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        osname = f"{platform.system()} {platform.release()}".strip()
        cores = os.cpu_count() or "不明"
        return f"現在時刻 {now} / OS {osname} / CPU {cores}コア / メモリ {self._mem_str()}"

    @staticmethod
    def _mem_str() -> str:
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            return f"使用{vm.percent:.0f}%（空き{vm.available // (1024 ** 2)}MB）"
        except Exception:
            return "（詳細は psutil 未導入のため省略）"
