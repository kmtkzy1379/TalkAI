r"""フィードバック記憶のエピソード化 A/B（実 gpt-4o-mini・要 .env キー・コスト小）。

ユーザ実機の「記憶の引っ掛かりが弱い＝『話したこと』しか覚えてない」を、
旧プロンプト（手続き型メタ要約）と新プロンプト（出来事の記憶＝事実）で同じ会話に対して
summary/tags を生成して並べ、改善を可視化する。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\feedback_episodic_ab.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.clock import Stamp  # noqa: E402
from eve.context_assembler import Turn  # noqa: E402
from eve.feedback import prompts as fb_prompts  # noqa: E402
from eve.feedback.parser import parse_feedback  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402

# 旧 summary 指示（エピソード化前）。A/B 比較のため当時の文面を再現。
OLD_SYSTEM = """\
あなたはAI VTuber「イブ」の内省（フィードバック）モジュールです。
直前の会話を振り返り、次の各項目を**タグ付きで1行ずつ**日本語で出力してください。
余計な前置き・解説・マークダウンは書かない。各タグの値は簡潔に。

summary: 今回のやり取りの要約（1文）
emotion: イブが今感じている感情（短く）
user_emotion: ユーザの感情の推定（短く）
next_prediction: 次に何が起きる/話されるかの予測（1文）
surprise: 前回の予測と今回の現実のズレ（0=完全一致 〜 100=完全に外れ。整数のみ）
reason: その予測差・感情の理由（1文）
tags: 話題タグをカンマ区切りで最大6個"""

# 実機セッションを模した会話スパン（user/eve）。
SPANS = [
    ("チーズケーキ画像選び", [
        ("user", "この画像の中で一番美味しそうなのどれだと思う？"),
        ("eve", "チーズケーキがしっかりのってて、表面がつやっとしたやつが美味しそうだよ。"),
        ("user", "わかる、それカフェで探してたんだよね。"),
    ]),
    ("過去に調べた食べ物", [
        ("user", "前にどんな食べ物調べてたか覚えてる？"),
        ("eve", "ごめん、今の会話だけだと、前に何を調べてたかまでは覚えてないよ。"),
        ("user", "チーズケーキだよ。"),
    ]),
    ("雨の日の室内遊び", [
        ("user", "雨だと家で何しようかな。"),
        ("eve", "家でゲームしたり、ボウリングゲームで遊ぶのもいいかもね。"),
    ]),
    ("おやつ相談", [
        ("user", "今日もおやつ何食べようかな。"),
        ("eve", "軽めならクッキー、しっかりならチーズケーキもありだよ。"),
        ("user", "なんか一つに絞ってよ。"),
        ("eve", "じゃあ今日はクッキーがいいよ。"),
    ]),
]


def _turns(pairs):
    return [Turn(sp, tx, Stamp.now()) for sp, tx in pairs]


async def _gen(reg, system, turns):
    user_text = fb_prompts.build_user_text(turns, None)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user_text}]
    resp = await reg.complete("feedback", msgs)
    txt = resp["choices"][0]["message"]["content"] if isinstance(resp, dict) else resp.choices[0].message.content
    return parse_feedback(txt)


async def main():
    reg = ModelRegistry(overrides={"feedback": "openai/gpt-4o-mini"})
    print(f"feedback model -> {reg.resolve('feedback')}\n")
    for title, pairs in SPANS:
        turns = _turns(pairs)
        old = await _gen(reg, OLD_SYSTEM, turns)
        new = await _gen(reg, fb_prompts.FEEDBACK_SYSTEM, turns)
        print(f"=== {title} ===")
        for sp, tx in pairs:
            print(f"   {'🧑' if sp=='user' else '🤖'} {tx}")
        print(f"  旧 summary: {old.summary}")
        print(f"     tags   : {old.topic_tags}")
        print(f"  新 summary: {new.summary}")
        print(f"     tags   : {new.topic_tags}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
