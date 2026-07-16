"""search_web 能力の登録（TaskAgent 専用・応答LLM には提示しない）。

- offered=False / agent_tool=True: ブロッキング能力は agent 専用（inc0 契約:
  dispatcher は逐次実行のため offered にすると cancel 等が検索の後ろで数秒待たされる）。
- 予約検索（「30秒後に調べて」）は delegate_task 経由で既存タスク機構がそのまま処理する。
"""
from __future__ import annotations

from ..capability.registry import Capability, CapabilityRegistry
from .client import SearchClient


def _as_bool(v) -> bool:
    """LLM 由来の bool 引数の厳密パース（JSON 型崩れ耐性: bool("false")==True の罠を塞ぐ）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def register_search_capability(registry: CapabilityRegistry, client: SearchClient) -> None:
    async def _search(args: dict) -> str:
        return await client.search(
            (args or {}).get("query", ""), _as_bool((args or {}).get("fresh")))

    registry.register(Capability(
        name="search_web",
        description="Webを検索して上位結果のタイトルと概要（スニペット）を得る。調べ物の第一手。"
                    "結果が不十分・空振りだったら**別のキーワード**で呼び直してよい"
                    "（同じキーワードの単純反復はしない）。ユーザが「もう一回/最新を調べて」と"
                    "言った時は fresh=true でキャッシュを使わず再検索する。",
        params_schema={
            "query": {"type": "string", "description": "検索キーワード（日本語可・短く要点だけ）"},
            "fresh": {"type": "boolean", "description": "キャッシュを使わず再検索する（省略=false）"},
        },
        async_handler=_search,
        offered=False, agent_tool=True,
    ))
