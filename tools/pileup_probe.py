r"""畳みかけ(pile-up)防止の実測（headless・実 LLM）。

実機ログで「同じ切り口を3連続」が出た。直前にイブが自分から話して**ユーザ未返信**のとき、
次の沈黙窓で黙るか(畳みかけ防止が効くか)を実 gpt-4o-mini で測る。静止画面でも検証。
※単発1回/窓・single-flight なので暴走(runaway)ではないが、質として連投を抑えたい。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\pileup_probe.py [N]
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

BORING = "Windows PowerShellのウィンドウが最前面に表示され、ログが流れている。直前のフレームから変化はない。"
SEEDS = [Turn("user", "ユーザはチーズケーキが好き", Stamp.now()),
         Turn("user", "ユーザは週末に映画の予定", Stamp.now())]

# A: 直前にイブが自分から2連続で話し、ユーザは未返信（＝畳みかけ防止が効くべき＝黙る期待）。
A = [Turn("user", "なんか軽く見られるもの探そうかな", Stamp.now()),
     Turn("eve", "じゃあジャンルからゆるく絞っていこっか。", Stamp.now()),
     Turn("eve", "ゲーム・動画・買い物の中だと、今いちばん見たいのはどれ？", Stamp.now())]
# B: 対照。ユーザが最後に返事している（＝話してよい状況）。
B = [Turn("eve", "ゲーム・動画・買い物どれがいい？", Stamp.now()),
     Turn("user", "うーん、どうしようかな。", Stamp.now())]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    reg = ModelRegistry(overrides={"speech_decide": "openai/gpt-4o-mini"})
    decide = make_decide_fn(reg)
    print(f"decide -> {reg.resolve('speech_decide')}  N={n}\n")
    for name, recent, vision, expect in (
        ("A 自分から2連続・ユーザ未返信（黙る期待）", A, BORING, "黙る"),
        ("A' 同上・画面なし", A, None, "黙る"),
        ("B 対照・ユーザ返事済み（喋ってよい）", B, BORING, "—"),
    ):
        silent = 0
        ex = ""
        for _ in range(n):
            r = await decide(surprise=40, silence_seconds=18.0, recent_turns=recent,
                             topic_seeds=SEEDS, last_feedback=None, vision=vision)
            if not r.speak:
                silent += 1
            elif not ex:
                ex = r.content
        print(f"  {name}: 黙った {silent}/{n}（期待={expect}）" + (f"  喋った例:「{ex[:34]}」" if ex else ""))


if __name__ == "__main__":
    asyncio.run(main())
