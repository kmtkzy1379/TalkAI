"""役割→モデルの間接層（provider 非依存）。

- Claude は現在停止中 → response/youtube 役は GPT/Gemini で代用。間接層なので後で
  Claude に戻すのは既定値の変更だけ（呼び出し側は不変）。
- 完了呼び出しは litellm を**遅延 import**（provider 差を litellm が吸収）。テストや
  オフラインでは `completion_fn` を注入して litellm 抜きで配線を検証できる。

既定モデルIDは暫定（実装時に有効性を検証する）。`.env` で上書き可。
"""
from __future__ import annotations

import os
from typing import AsyncIterator, Awaitable, Callable, Optional

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
StreamFn = Callable[..., AsyncIterator[str]]  # (model=, messages=, **kw) -> 文字列デルタの async iter


def _g(o: object, k: str):
    """dict / オブジェクト両対応の属性取り出し（litellm の chunk は object・fake は dict）。"""
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def merge_tool_call_deltas(fragments) -> list[dict]:
    """streaming の tool_calls 断片（index 毎に id/name/arguments が分割）を1件ずつに結合する。

    各 fragment は `.index / .id / .function.{name,arguments}` を持つ（object or dict）。
    arguments は連結（JSON 文字列の断片）。name の無い空 slot は捨てる。raise しない。
    """
    acc: dict[int, dict] = {}
    for f in fragments or []:
        idx = _g(f, "index") or 0
        slot = acc.setdefault(int(idx), {"id": "", "name": "", "arguments": ""})
        fid = _g(f, "id")
        if fid:
            slot["id"] = fid
        fn = _g(f, "function")
        if fn is not None:
            nm = _g(fn, "name")
            if nm:
                slot["name"] = nm
            ar = _g(fn, "arguments")
            if ar:
                slot["arguments"] += ar
    out: list[dict] = []
    for idx in sorted(acc):
        s = acc[idx]
        if s["name"]:
            out.append(
                {"id": s["id"] or f"call_{idx}",
                 "function": {"name": s["name"], "arguments": s["arguments"] or "{}"}}
            )
    return out


class ModelRegistry:
    """役割名から具体モデルIDを解決し、統一インターフェースで完了を呼ぶ。"""

    def __init__(
        self,
        overrides: Optional[dict[str, str]] = None,
        completion_fn: Optional[CompletionFn] = None,
        stream_fn: Optional[StreamFn] = None,
    ) -> None:
        self._overrides: dict[str, str] = dict(overrides or {})
        self._completion_fn = completion_fn  # None なら litellm.acompletion を遅延使用
        self._stream_fn = stream_fn  # None なら litellm.acompletion(stream=True) を遅延使用

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

    async def stream(self, role: str, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """解決済みモデルへストリーミング要求。文字列デルタを逐次 yield する。"""
        model = self.resolve(role)
        if self._stream_fn is not None:
            async for piece in self._stream_fn(model=model, messages=messages, **kwargs):
                yield piece
            return
        from litellm import acompletion  # 遅延 import（テストは不要）

        resp = await acompletion(model=model, messages=messages, stream=True, **kwargs)
        async for chunk in resp:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError, KeyError, TypeError):
                delta = None
            if delta:
                yield delta

    async def stream_with_tools(
        self, role: str, messages: list[dict], *, tools: list[dict], tool_sink: list, **kwargs
    ) -> AsyncIterator[str]:
        """content デルタを yield しつつ、streamed tool_calls を index 毎に結合して `tool_sink` に積む。

        litellm は tool_calls をチャンクで断片配信する（`delta.tool_calls[i]` の id/name/arguments が
        複数チャンクに分かれる）。終端で完成した tool_calls（dict）を `tool_sink` へ追記する。
        Call-Function 有効ターンでのみ呼ぶ（content streaming は従来どおり＝発話順/barge-in 不変）。
        """
        model = self.resolve(role)
        if self._stream_fn is not None:  # テスト注入（tool path は tools/tool_sink を受ける fn を渡す）
            async for piece in self._stream_fn(
                model=model, messages=messages, tools=tools, tool_sink=tool_sink, **kwargs
            ):
                yield piece
            return
        from litellm import acompletion  # 遅延 import

        resp = await acompletion(model=model, messages=messages, stream=True, tools=tools, **kwargs)
        frags: list = []
        async for chunk in resp:
            try:
                delta = chunk.choices[0].delta
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content
            tcs = getattr(delta, "tool_calls", None)
            if tcs:
                frags.extend(tcs)
        tool_sink.extend(merge_tool_call_deltas(frags))
