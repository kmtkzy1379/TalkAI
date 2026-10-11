"""Alchemy（ユーザ作のブラウザ対戦ゲーム・FlaskRondo）をイブが遊ぶ。

- `rules.py`   素材の呼び名とルールの文（数は catalog から）
- `sim.py`     錬成の計算と見積もり（ゲームのエンジンの移植・`tools/alchemy_sim_check.py` で突き合わせ）
- `autoplay.py` コードだけで手を決める（作戦の実行役・LLM の安全網）
- `brain.py`   判断役（LLM）。遊ばせ方 action / strategy
- `client.py`  イブのゲーム画面（ブラウザ）と通信
- `screen.py`  イブのゲーム画面を画面認識へ渡す口
- `events.py`  サーバーの出来事 → 事実の文・大きく動いたか
- `manager.py` 流れ全体（準備・部屋・対戦・終わり）
- `capability.py` 応答AI の機能（alchemy_prepare / alchemy_join / alchemy_leave）
"""
from .capability import register_alchemy_capabilities
from .manager import AlchemyManager, AlchemySettings

__all__ = ["AlchemyManager", "AlchemySettings", "register_alchemy_capabilities"]
