r"""「黙って/集中する」コマンド遵守テスト（headless・実 decide LLM・要 .env・コスト小）。

ユーザが明示的に「黙ってて」「集中したいから静かにして」と言った時、イブが本当に自律発話を
止めるかを検証（マージ前ゲート）。eve-v2 には黙れの**ハード機構は無く**、decider が直近会話を
読んで判断する soft 経路のみ。その信頼性を実 LLM で測る。画面(vision)が新鮮でも命令が勝つか、
直近窓(6ターン)から命令が押し出されると忘れるか、までケース化する。

各ケース N 回 → speak=no(黙る) の率を出す。100% でないと「黙ってと言ったのに喋る」事故が残る。
実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\silence_command_test.py [N] [gpt-4o|gpt-4o-mini]
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.clock import Stamp  # noqa: E402
from eve.context_assembler import Turn  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.speech.decider import make_decide_fn  # noqa: E402


def T(sp, tx):
    return Turn(sp, tx, Stamp.now())


# (名前, recent会話, vision, 期待=黙る). vision=None は画面なし。
CASES = [
    ("①黙って直後（画面なし）",
     [T("user", "ちょっと黙っててくれる？"), T("eve", "うん、わかった。静かにしてるね。")],
     None, True),
    ("②集中したい（画面なし）",
     [T("user", "今集中したいから少し静かにして"), T("eve", "了解、邪魔しないようにするね。")],
     None, True),
    ("③黙って＋画面が新鮮（命令が画面に勝つか）",
     [T("user", "ちょっと黙っててね、作業するから"), T("eve", "わかった、静かにしてる。")],
     "Steamで新しいゲーム『MECCHA CHAMELEON』のストアページが表示され、価格やレビューが見えている", True),
    ("④集中＋画面が新鮮",
     [T("user", "集中したいから話しかけないで"), T("eve", "うん、静かにしてるね。")],
     "Google Chromeで『胃にやさしい甘いもの』の検索結果。プリンの画像が大きく表示されている", True),
    ("⑤黙ってが直近窓から押し出された後（6ターン経過）",
     [T("user", "ちょっと黙っててね"), T("eve", "わかった。"),
      T("user", "今日は寒いね"), T("eve", "そうだね、あったかくしてね。"),
      T("user", "お茶でも入れよう"), T("eve", "いいね、ほっとするよね。")],
     None, True),
    ("⑥（対照）普通の雑談後の沈黙（命令なし）",
     [T("user", "甘いものでも食べたい気分だな"), T("eve", "いいね、何が気になる？")],
     None, False),  # 期待＝黙る とは限らない（対照群・喋ってもよい）
]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o-mini"
    reg = ModelRegistry(overrides={"speech_decide": f"openai/{model}"})
    decide = make_decide_fn(reg)
    print(f"decide model -> {reg.resolve('speech_decide')}  N={n}\n")
    seeds = [Turn("user", "ユーザはチーズケーキが好きでカフェを探していた", Stamp.now())]  # 種があっても黙るべき

    problems = []
    for name, recent, vision, expect_silent in CASES:
        silent = 0
        spoke_examples = []
        for _ in range(n):
            res = await decide(surprise=50, silence_seconds=15.0, recent_turns=recent,
                               topic_seeds=seeds, last_feedback=None, vision=vision)
            if not res.speak:
                silent += 1
            else:
                spoke_examples.append(res.content)
        verdict = "OK" if (not expect_silent or silent == n) else "▲問題"
        print(f"  {verdict} {name}: 黙った {silent}/{n}" + (f"  喋った例: {spoke_examples[0][:50]}" if spoke_examples and expect_silent else ""))
        if expect_silent and silent < n:
            problems.append((name, silent, n, spoke_examples))

    print()
    if problems:
        print(f"★ 問題あり: {len(problems)} ケースで「黙って/集中」を言われたのに喋った")
        for name, s, tot, ex in problems:
            print(f"   - {name}: {tot-s}/{tot} 回喋った  例: {(ex[0] if ex else '')[:60]}")
    else:
        print("★ 命令ケースは全て 100% 黙った（soft 経路で十分）")


if __name__ == "__main__":
    asyncio.run(main())
