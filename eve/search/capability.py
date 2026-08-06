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
            (args or {}).get("query", ""), _as_bool((args or {}).get("fresh")),
            _as_bool((args or {}).get("deep")))

    registry.register(Capability(
        name="search_web",
        # E5(2026-08-06): 1文目が「タイトルと概要」だけだったため、応答LLM が
        # `outward_actions()` 経由でそれを読み「ページの中身まで読む手段は持ってない」と
        # ユーザに答えた（同じランで deep=true を7回実行しており事実と違う）。
        # deep を含む実態に合わせる。**1文目に括弧を使わない**（`_PAREN` で落ちて文が壊れる）。
        description="Webを検索して調べ、必要ならページの本文まで読んで要約する。"
                    "スニペットだけで足りる調べ物の第一手でもある。"
                    "結果が不十分・空振りだったら**別のキーワード**で呼び直してよい"
                    "（同じキーワードの単純反復はしない）。ユーザが「もう一回/最新を調べて」と"
                    "言った時は fresh=true でキャッシュを使わず再検索する。",
        params_schema={
            "query": {"type": "string", "description": "検索キーワード（日本語可・短く要点だけ）"},
            "fresh": {"type": "boolean", "description": "キャッシュを使わず再検索する（省略=false）"},
            "deep": {"type": "boolean", "description": "上位ページの本文まで読んで要約する（時間がかかる。"
                     "スニペットで足りない詳しい調査のときだけ true）"},
        },
        async_handler=_search,
        offered=False, agent_tool=True,
    ))
