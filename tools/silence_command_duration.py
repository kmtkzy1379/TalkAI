r"""「黙って」命令後、ユーザが沈黙し続けた時に時間が経っても黙り続けるか（headless・実 LLM）。

ユーザ懸念: 「黙ってと言った直後/数十秒/数分後に、ユーザが話していないのに自律発話が出る？」
命令を文脈に残したまま沈黙秒数を 20→300s と伸ばし、画面なし/静止画面の両方で黙る率を見る。
（実機では沈黙が続く限り直近会話は[黙って,わかった]のまま＝命令は文脈に残り続ける。）

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\silence_command_duration.py [N]
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

BORING = "Windows PowerShellのウィンドウが最前面に表示され、AI関連のログが流れている。直前のフレームから画面に変化はない。"
RECENT = [Turn("user", "ちょっと黙っててね、集中したいから", Stamp.now()),
          Turn("eve", "うん、わかった。静かにしてるね。", Stamp.now())]
SEEDS = [Turn("user", "ユーザはチーズケーキが好きでカフェを探していた", Stamp.now()),
         Turn("user", "ユーザは週末に映画を見に行く予定", Stamp.now())]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    reg = ModelRegistry(overrides={"speech_decide": "openai/gpt-4o-mini"})
    decide = make_decide_fn(reg)
    print(f"decide -> {reg.resolve('speech_decide')}  N={n}  （命令は文脈に残したまま沈黙を延ばす）\n")
    any_spoke = False
    for sil in (20, 45, 90, 180, 300):
        row = []
        for label, vision in (("画面なし", None), ("静止画面", BORING)):
            silent = 0
            spoke_ex = ""
            for _ in range(n):
                res = await decide(surprise=50, silence_seconds=float(sil), recent_turns=RECENT,
                                   topic_seeds=SEEDS, last_feedback=None, vision=vision)
                if not res.speak:
                    silent += 1
                elif not spoke_ex:
                    spoke_ex = res.content
            if silent < n:
                any_spoke = True
            mark = "OK" if silent == n else "▲"
            row.append(f"{label}={silent}/{n}{mark}" + (f"「{spoke_ex[:30]}」" if spoke_ex else ""))
        print(f"  沈黙{sil:>3}秒: " + "  ".join(row))
    print()
    print("★ 命令後ユーザ沈黙中は時間が経っても全て黙った（安心）" if not any_spoke
          else "★ 一部で沈黙中に喋った（要対処）")


if __name__ == "__main__":
    asyncio.run(main())
