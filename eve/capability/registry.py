"""J-0 共有 Capability 層（read-only サブセット）。

応答LLM が tool_call で呼べる「能力」を**構造化（enum + 型付き引数）**で1箇所に列挙する。
本 increment は read-only のみ（`self_status` / `pc_status`）＝OS 変更なし・生コマンドなし。
将来 J-0 の状態変更系（download/launch_app/window前面化）や J-2 search はここに足す（3経路が共有）。

- `tool_schemas()`: litellm の `tools=`（OpenAI function-calling schema）形式を返す。
- `execute(name, args)`: 同期能力を実行し**人間可読の結果文字列**を返す。例外は握って「失敗」文字列
  （応答LLM が「○○に失敗した」と正直に言えるように・落とさない＝CLAUDE.md 規律）。
- `execute_async(name, args)`: **本番3経路（TaskAgent/executor/dispatcher）の統一実行経路**（J-2〜）。
  同期能力は execute() へ委譲（同値・計測モンキーパッチも効く）、async 能力はここで await する。
"""
from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    params_schema: dict  # JSON schema の properties（引数なしなら空 dict）
    handler: Optional[Callable[[dict], str]] = None  # (args) -> 人間可読の結果（同期・sub-ms 前提）
    mutates_state: bool = False  # 状態を変える書込能力（将来の UI 承認/監査用マーカ）
    offered: bool = True  # **応答LLM** に tool 提示するか（委譲/管理系=True・実行系/内部=False）
    agent_tool: bool = False  # **TaskAgent** に tool 提示するか（実行系=True）。＝責務分離の軸:
    # 応答LLM は delegate_task/create_task/cancel_task/list_tasks だけ（offered）＝何をしたいか振るだけ。
    # 実行系(self_status/pc_status/将来の検索・画面操作)は agent_tool=True で **TaskAgent 専用**。
    report_result: bool = True  # 即時実行の結果を CALLFUNCTION_RESULT で報告するか。
    # False=create_task 等の「予約しただけ」系（応答ターンの本文が既に確認済＝ack 重複/状態先出しハルシネ防止）。
    # ※失敗(「（…」マーカ)時は report_result に関係なく報告される（dispatcher 側）。
    # J-2 前提工事: ブロッキング能力（検索・将来の画面操作）用の非同期 handler。
    # 同期 handler（sub-ms の read-only 系）とどちらか一方を指定する（__post_init__ で強制）。
    # 【async_handler の契約（レッドチーム裁定）】
    # - 内部で asyncio.to_thread 等へ逃がす（loop を数秒塞ぐと mic/barge-in/VLM が全停止する）。
    # - blocking I/O は**必ずハードタイムアウト**（socket connect/read）を持つ。cancel は thread を
    #   殺せないため、無タイムアウトの残スレッドはプロセス終了(shutdown_default_executor)をブロックする。
    # - to_thread に渡す関数は純粋関数にする（loop 所有の共有状態＝キャッシュ等は await の前後で
    #   loop 側からのみ読み書き。thread 内から触ると単一ループ保証が破れる）。
    # - ブロッキング能力は agent_tool 専用とし offered=True にしない（dispatcher は逐次実行のため
    #   後続の cancel_task 等が検索の後ろで数秒待たされる＝CancelResolver 分離と同じ理由）。
    async_handler: Optional[Callable[[dict], Awaitable[str]]] = None

    def __post_init__(self) -> None:
        # 契約の fail-fast: handler / async_handler はちょうど一方（両方/両方None は登録時プログラミングエラー。
        # 実行時マーカで沈黙させると経路により別コードが走る事故になる）。
        if (self.handler is None) == (self.async_handler is None):
            raise ValueError(f"Capability「{self.name}」は handler / async_handler のどちらか一方だけを指定する")


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
        # 実行系能力＝TaskAgent 専用（応答LLM には提示しない＝offered=False/agent_tool=True）。
        self.register(Capability(
            name="self_status",
            description="イブ自身の今の状態（応答中か / 未処理の刺激数 / 直近のエラー）を確認する。引数なし。",
            params_schema={},
            handler=self._self_status, offered=False, agent_tool=True,
        ))
        self.register(Capability(
            name="pc_status",
            description="PC の現在状態（現在時刻 / OS / CPUコア数 / メモリ）を確認する。引数なし。",
            params_schema={},
            handler=self._pc_status, offered=False, agent_tool=True,
        ))

    def register(self, cap: Capability) -> None:
        self._caps[cap.name] = cap

    def has(self, name: str) -> bool:
        return name in self._caps

    def report_result(self, name: str) -> bool:
        """この能力の即時実行結果を報告するか（未知は True＝報告）。"""
        cap = self._caps.get(name)
        return cap.report_result if cap is not None else True

    def is_blocking(self, name: str) -> bool:
        """時間のかかる能力か（async_handler 持ち＝検索/将来の fetch・画面操作）。

        TaskAgent が 1 step 内の重い呼び出しを律速するのに使う（未知/同期は False）。
        """
        cap = self._caps.get(name)
        return cap is not None and cap.async_handler is not None

    def names(self) -> list[str]:
        return list(self._caps)

    @staticmethod
    def _schema(c: "Capability") -> dict:
        return {
            "type": "function",
            "function": {
                "name": c.name,
                "description": c.description,
                "parameters": {"type": "object", "properties": c.params_schema, "required": []},
            },
        }

    def tool_schemas(self) -> list[dict]:
        """**応答LLM** 向け（委譲/管理系＝offered=True）。実行系/内部は除外。"""
        return [self._schema(c) for c in self._caps.values() if c.offered]

    def agent_tool_schemas(self) -> list[dict]:
        """**TaskAgent** 向け（実行系＝agent_tool=True）。委譲/管理系は含めない（自己再帰防止）。"""
        return [self._schema(c) for c in self._caps.values() if c.agent_tool]

    def execute(self, name: str, args: Optional[dict] = None) -> str:
        """同期実行（sub-ms の同期 handler のみ対象）。

        async 専用能力（handler=None）は「同期経路では実行不可」の失敗マーカを返す
        （本番3経路は execute_async へ移行済みなので到達しない防御線。__post_init__ により
        handler=None なら必ず async_handler が存在する）。
        """
        cap = self._caps.get(name)
        if cap is None:
            return f"（未対応の機能「{name}」のため実行しませんでした）"
        if cap.handler is None:
            return f"（機能「{name}」は非同期専用のため、この経路では実行できませんでした）"
        try:
            result = cap.handler(args or {})
            # handler の非 str 戻り値（プログラミングエラー）で下流（startswith/スライス）が
            # 静かに壊れないよう正規化（報告消失・タスク Running 放置の防止）。
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.exception("Capability 実行で例外: %s", name)
            reason = f"{type(e).__name__}: {e}".strip()
            # 失敗理由を短く同梱（応答LLM が「○○が理由で失敗した」と正直に言えるように）。
            return f"（機能「{name}」の実行に失敗しました: {reason[:120]}）"

    async def execute_async(self, name: str, args: Optional[dict] = None) -> str:
        """非同期実行（J-2 前提工事・本番3経路の統一経路）。

        - async_handler はここで await（ブロッキング I/O は handler 内で to_thread 等へ逃がす契約）。
        - それ以外（同期能力・未対応名）は **execute() へ委譲**＝挙動・失敗マーカが構築的に同値で、
          計測用の execute モンキーパッチ（tools/ の exec_wrap 等）も従来どおり効く。
        - CancelledError は握らない（BaseException＝shutdown/barge の伝播を妨げない）。
        """
        cap = self._caps.get(name)
        if cap is not None and cap.async_handler is not None:
            try:
                result = await cap.async_handler(args or {})
                return result if isinstance(result, str) else str(result)  # 非 str 正規化（execute と対称）
            except Exception as e:
                logger.exception("Capability 実行で例外: %s", name)
                reason = f"{type(e).__name__}: {e}".strip()
                return f"（機能「{name}」の実行に失敗しました: {reason[:120]}）"
        return self.execute(name, args)

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
