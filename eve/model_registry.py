"""役割→モデルの間接層（provider 非依存）。

- Claude は現在停止中 → response/youtube 役は GPT/Gemini で代用。間接層なので後で
  Claude に戻すのは既定値の変更だけ（呼び出し側は不変）。
- 完了呼び出しは litellm を**遅延 import**（provider 差を litellm が吸収）。テストや
  オフラインでは `completion_fn` を注入して litellm 抜きで配線を検証できる。

既定モデルIDは暫定（実装時に有効性を検証する）。`.env` で上書き可。
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# role -> (env キー, 既定モデルID)。既定は Claude 停止中の暫定代用値。
ROLE_ENV: dict[str, tuple[str, str]] = {
    "response": ("RESPONSE_MODEL", "openai/gpt-4o"),
    "speech_decide": ("DECIDE_MODEL", "openai/gpt-4o-mini"),
    "feedback": ("FEEDBACK_MODEL", "openai/gpt-4o-mini"),
    "task": ("TASK_MODEL", "openai/gpt-5.5"),  # TaskAgent（非同期・遅延許容＝賢いモデル）
    "vlm_leaf": ("VLM_LEAF_MODEL", "gemini/gemini-2.5-flash"),
    "vlm_merge": ("VLM_MERGE_MODEL", "gemini/gemini-2.5-pro"),
    "youtube": ("YOUTUBE_MODEL", "openai/gpt-4o"),
    "summarize": ("SUMMARIZE_MODEL", "openai/gpt-4o-mini"),
    # 自己形成（イブの好み・趣向の台帳を更新する）。**速度は要らないが賢さは要る**:
    # 起動は合図が出た時だけ（実測 イブ発話の 15.5%）＝呼び出しが稀なので、
    # 安いモデルを常時回すより賢いモデルを稀に呼ぶ方が安く正確（裁定 2026-08-15）。
    "self_model": ("SELF_MODEL_MODEL", "openai/gpt-5.4"),
    # J-2 inc2: Web本文の隔離要約（タスクLLM2）。**ツールを一切渡さない**専用ロール＝
    # 信頼できない Web 本文を読むのはこのロールだけ（injection がツール実行に到達しない構造）。
    "search_summarize": ("SEARCH_SUMMARIZE_MODEL", "openai/gpt-4o-mini"),
    # 報告が実発話で伝わったかの確認係。**決定論で決められない分岐でしか呼ばれない**
    # （`response/delivery_checker.py` の規則5）。既定を旧 gpt-4o-mini から上げたのは、
    # 02:44 の誤判定（ほぼ同一の報告と発話を forgotten と判定→二重回答）を出したのが
    # その世代で、しかも**パイプラインで最も弱いモデルが最も強い権限＝再依頼を持つ**
    # 構図になっていたため。呼び出し頻度が低いので安さより賢さを取る。
    "delivery_check": ("DELIVERY_CHECK_MODEL", "openai/gpt-5.4"),
    # 笑い判定（S6・ユーザ裁定 2026-09-26）。ユーザの話の「、」「。」ごとに「イブが思わず笑う場面か」を
    # yes/no で聞き、logprobs を確率にする。実測（名大会話コーパス・2026-09-25）で閾値が機能したのは
    # gpt-6-luna だけ（Jev は確率が最大 0.67 で 0.85 に届かない）。1件 約0.6〜0.8秒・約$0.00008。
    "laugh_judge": ("LAUGH_MODEL", "openai/gpt-6-luna"),
    # 感情（S5・ユーザ裁定 2026-09-26「感情判定はルナでよさそう」）。ユーザの話の「、」「。」ごとに
    # 「この発話を聞いたイブが次に返す声に乗る感情」（喜怒哀楽笑い）を JSON で聞く。実測（JVNV・2026-09-25）:
    # 話者の6感情 正解率 97%・1件 中央 1.28秒（8並列）・1件 約$0.00005。
    "emotion_judge": ("EMOTION_MODEL", "openai/gpt-6-luna"),
    # Alchemy（ユーザ作のゲーム）の手を決める判断役（`eve/alchemy/brain.py`）。1ラウンドに1〜3回・締切つき
    "game": ("GAME_MODEL", "openai/gpt-6-luna"),
}

CompletionFn = Callable[..., Awaitable[object]]
StreamFn = Callable[..., AsyncIterator[str]]  # (model=, messages=, **kw) -> 文字列デルタの async iter


def _g(o: object, k: str):
    """dict / オブジェクト両対応の属性取り出し（litellm の chunk は object・fake は dict）。"""
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _finish_reason(chunk) -> Optional[str]:
    try:
        r = chunk.choices[0].finish_reason
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return str(r) if r else None


def _warn_cut(role: str, model: str, finish: Optional[str], out: list) -> None:
    """モデルが返事を途中で止めた（安全のフィルター・長さの上限）ことをログに残す（黙って途切れると原因が追えない）。

    実測 2026-10-15 の配信リハーサル: 応答が文の途中（「〜みたいなこと言」）で終わり、ログに何も残らなかった。
    """
    if finish in ("content_filter", "length", "safety", "SAFETY", "recitation", "RECITATION"):
        logger.warning("✂ %s（%s）の返事が途中で止められた（%s・ここまで %d 字: %.40s）", role, model, finish,
                       len("".join(out)), "".join(out)[-40:])


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


def _meter_record(fn) -> None:
    """使用量メーターがあれば記録する。**無ければ何もしない＝挙動は完全に不変**。

    ここは応答の主経路なので、記録の失敗で会話を止めない（`usage.record` 側も握るが、
    メーター取得やラムダ自体が落ちる可能性まで含めてここで塞ぐ）。
    """
    try:
        from .usage import meter

        m = meter()
        if m is not None:
            fn(m)
    except Exception:  # noqa: BLE001（計測のために本体を落とさない）
        logger.debug("使用量の記録に失敗（無視）", exc_info=True)


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
        resp = await fn(model=model, messages=messages, **kwargs)
        _meter_record(lambda m: m.record_response(role=role, model=model, response=resp))
        return resp

    async def stream(self, role: str, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """解決済みモデルへストリーミング要求。文字列デルタを逐次 yield する。"""
        model = self.resolve(role)
        # 途中で barge-in に切られても、そこまでの分は数える（finally で記録）。
        out: list[str] = []
        try:
            if self._stream_fn is not None:
                async for piece in self._stream_fn(model=model, messages=messages, **kwargs):
                    out.append(piece)
                    yield piece
                return
            from litellm import acompletion  # 遅延 import（テストは不要）

            resp = await acompletion(model=model, messages=messages, stream=True, **kwargs)
            finish = None
            async for chunk in resp:
                finish = _finish_reason(chunk) or finish
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError, KeyError, TypeError):
                    delta = None
                if delta:
                    out.append(delta)
                    yield delta
            _warn_cut(role, model, finish, out)
        finally:
            _meter_record(lambda m: m.record_stream(
                role=role, model=model, messages=messages, output="".join(out)))

    async def stream_with_tools(
        self, role: str, messages: list[dict], *, tools: list[dict], tool_sink: list, **kwargs
    ) -> AsyncIterator[str]:
        """content デルタを yield しつつ、streamed tool_calls を index 毎に結合して `tool_sink` に積む。

        litellm は tool_calls をチャンクで断片配信する（`delta.tool_calls[i]` の id/name/arguments が
        複数チャンクに分かれる）。終端で完成した tool_calls（dict）を `tool_sink` へ追記する。
        Call-Function 有効ターンでのみ呼ぶ（content streaming は従来どおり＝発話順/barge-in 不変）。
        """
        model = self.resolve(role)
        out: list[str] = []
        try:
            if self._stream_fn is not None:  # テスト注入（tool path は tools/tool_sink を受ける fn を渡す）
                async for piece in self._stream_fn(
                    model=model, messages=messages, tools=tools, tool_sink=tool_sink, **kwargs
                ):
                    out.append(piece)
                    yield piece
                return
            from litellm import acompletion  # 遅延 import

            resp = await acompletion(model=model, messages=messages, stream=True, tools=tools, **kwargs)
            frags: list = []
            finish = None
            async for chunk in resp:
                finish = _finish_reason(chunk) or finish
                try:
                    delta = chunk.choices[0].delta
                except (AttributeError, IndexError, KeyError, TypeError):
                    continue
                content = getattr(delta, "content", None)
                if content:
                    out.append(content)
                    yield content
                tcs = getattr(delta, "tool_calls", None)
                if tcs:
                    frags.extend(tcs)
            tool_sink.extend(merge_tool_call_deltas(frags))
            _warn_cut(role, model, finish, out)
        finally:
            # tools のスキーマも入力トークンに乗るので一緒に数える（無視すると過小評価になる）。
            _meter_record(lambda m: m.record_stream(
                role=role, model=model, messages=messages, output="".join(out), tools=tools))
