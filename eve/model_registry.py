"""役割→モデルの間接層（provider 非依存）。

- Claude は現在停止中 → response/youtube 役は GPT/Gemini で代用。間接層なので後で
  Claude に戻すのは既定値の変更だけ（呼び出し側は不変）。
- 完了呼び出しは litellm を**遅延 import**（provider 差を litellm が吸収）。テストや
  オフラインでは `completion_fn` を注入して litellm 抜きで配線を検証できる。

既定モデルIDは暫定（実装時に有効性を検証する）。`.env` で上書き可。
"""
from __future__ import annotations

import os
from typing import Awaitable, Callable, Optional

# role -> (env キー, 既定モデルID)。既定は Claude 停止中の暫定代用値。
ROLE_ENV: dict[str, tuple[str, str]] = {
    "response": ("RESPONSE_MODEL", "openai/gpt-4o"),
    "speech_decide": ("DECIDE_MODEL", "openai/gpt-4o-mini"),
    "feedback": ("FEEDBACK_MODEL", "openai/gpt-4o-mini"),
    "vlm_leaf": ("VLM_LEAF_MODEL", "gemini/gemini-2.5-flash"),
    "vlm_merge": ("VLM_MERGE_MODEL", "gemini/gemini-2.5-pro"),
    "youtube": ("YOUTUBE_MODEL", "openai/gpt-4o"),
    "summarize": ("SUMMARIZE_MODEL", "openai/gpt-4o-mini"),
}

CompletionFn = Callable[..., Awaitable[object]]


class ModelRegistry:
    """役割名から具体モデルIDを解決し、統一インターフェースで完了を呼ぶ。"""

    def __init__(
        self,
        overrides: Optional[dict[str, str]] = None,
        completion_fn: Optional[CompletionFn] = None,
    ) -> None:
        self._overrides: dict[str, str] = dict(overrides or {})
        self._completion_fn = completion_fn  # None なら litellm.acompletion を遅延使用

    def resolve(self, role: str) -> str:
        """役割→モデルID。優先順: ランタイム override > .env > 既定。"""
        if role in self._overrides:
            return self._overrides[role]
        if role not in ROLE_ENV:
            raise KeyError(f"unknown role: {role}")
        env_key, default = ROLE_ENV[role]
        return os.getenv(env_key, default)

    def set_override(self, role: str, model_id: str) -> None:
        """UI からのモデル差し替え用（RUNNING 中はロックする想定）。"""
        if role not in ROLE_ENV:
            raise KeyError(f"unknown role: {role}")
        self._overrides[role] = model_id

    def roles(self) -> list[str]:
        return list(ROLE_ENV)

    async def complete(self, role: str, messages: list[dict], **kwargs):
        """解決済みモデルへ完了要求。provider 差は litellm が吸収。"""
        model = self.resolve(role)
        fn = self._completion_fn
        if fn is None:
            from litellm import acompletion  # 遅延 import（テストは不要）

            fn = acompletion
        return await fn(model=model, messages=messages, **kwargs)
