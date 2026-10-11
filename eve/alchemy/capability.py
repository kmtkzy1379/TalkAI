"""Alchemy の機能を応答AI に渡す（`CapabilityRegistry` へ登録・応答AI だけが使う＝タスク担当には渡さない）。

- `alchemy_prepare`: ルールを会話に入れる。部屋には入らない。**結果は報告しない**（報告ターンを起こすと、結果の文を
  毎回同じ型で言い直していた＝実測 2026-10-11 で17回中14回「ルール完璧に思い出しました・部屋にはまだ入ってません」。
  ユーザ「この報告が単純に要らない」）。代わりに `reanswer`: 返事を言う前にルールを入れ、同じ発話への返事を作り直す
  （報告しないだけだと、ルールを聞かれて「ちょっと待ってね」と言ったきり答えなかった）。
- `alchemy_join`: 合言葉の部屋に入って準備OKにする（対戦の後に次の対戦の準備OKにするのも）。入れた時は報告しない
  （ユーザ 2026-10-11「入室退出の報告はなくしていい」・受付の一言と重なる）。入れなかった時だけ報告する。
- `alchemy_leave`: やめる（対戦中でも抜ける）。

**ゲームを始められるのは `alchemy_join` だけ**。説明には呼ぶ条件だけを書き、**ユーザの言い回しの例は書かない**
（ユーザ 2026-10-11「例を入れるとLLM特性上その例の言い回しフォーマットだらけになりそう。最初は例を無くすでもいい」）。
3つとも `reply`（今すぐ返す一言）を必須にする（`delegate_task` と同じ理由: 応答AI は本文を書かずに機能だけ
返すことが多い・`response/orchestrator.py` の `_reply_from_tools`）。
"""
from __future__ import annotations

from ..capability.registry import Capability, CapabilityRegistry

_REPLY = {"type": "string",
          "description": "ユーザに今すぐ返す短い一言（自然な話し言葉・毎回同じ言い方にしない）"}

NAMES = ("alchemy_prepare", "alchemy_join", "alchemy_leave")


def register_alchemy_capabilities(registry: CapabilityRegistry, manager) -> None:
    kw = manager.s.keyword
    registry.register(Capability(
        name="alchemy_prepare",
        description="ユーザが作ったブラウザ対戦ゲーム「Alchemy」の話になった時に呼ぶ。このゲームのルールを会話に入れる"
                    "（部屋には入らない）。今から遊ぶ時は alchemy_join だけを呼べばよい。",
        params_schema={"reply": _REPLY},
        handler=manager.tool_prepare, offered=True, agent_tool=False, report_result=False,
        reanswer=True,  # 返事の前にルールを入れて、同じ発話への返事を作り直す（ルールを聞かれてもそのまま答える）
        required=("reply",),
    ))
    registry.register(Capability(
        name="alchemy_join",
        description=f"ユーザが今から一緒に Alchemy を遊ぶと決めた時に呼ぶ。合言葉「{kw}」の部屋に入って準備OKにする"
                    "（対戦が終わった後に、次の対戦の準備OKにするのもこの機能）。ルールも会話に入る。"
                    "今すぐ遊ぶと決まっていない時は呼ばない。",
        params_schema={"reply": _REPLY},
        handler=manager.tool_join, mutates_state=True, offered=True, agent_tool=False, report_result=False,
        required=("reply",),
    ))
    registry.register(Capability(
        name="alchemy_leave",
        description="ユーザが Alchemy をやめると決めた時に呼ぶ。合言葉の部屋や対戦から抜ける（対戦の途中でも抜ける）。",
        params_schema={"reply": _REPLY},
        handler=manager.tool_leave, mutates_state=True, offered=True, agent_tool=False, report_result=False,
        required=("reply",),
    ))
