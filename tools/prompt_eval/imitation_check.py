#!/usr/bin/env python3
"""ネタ（ミーム・名台詞・キャラの真似）の時の口調を、性格の口調の行の書き方（案）で比べる。実 API（少額）。

  python -m tools.prompt_eval.imitation_check   → eval_report/prompts/imitation_check/check.json と一覧

- 口調の行の案は手元の `persona_variants_local.py` の IMITATION_TONE_LINES（コミットしない）。今の性格
  （persona.local.md）の口調の行が案のどれかと同じであること＝その1行だけを差し替えて比べる
- 同じ場面・同じ回は、どの案にも同じ実例を見せる（実例のくじを場面と回で固定）＝口調の行だけの差を見る
- ネタの場面（真似してよい）と、ふつうの場面（丁寧語のままのはず）の両方で見る
"""
from __future__ import annotations

import asyncio
import json
import logging
import random

from .common import OUT, dump, live_persona_body, setup_env
from .persona_diff import casual_sentences
from .scenarios import S

DIR = OUT / "imitation_check"
REPS = 2
try:
    from .persona_variants_local import IMITATION_TONE_LINES as VARIANTS   # 手元だけ（性格の中身）
except ImportError:
    VARIANTS: dict = {}
MEME = [
    S("M01", "M", "見ろ、人がゴミのようだ！"),
    S("M02", "M", "だが断る"),
    S("M03", "M", "お前はもう死んでいる"),
    S("M04", "M", "逃げちゃダメだ逃げちゃダメだ逃げちゃダメだ"),
    S("M05", "M", "バルス！"),
    S("M06", "M", "我が生涯に一片の悔いなし！"),
    S("M07", "M", "計画通り"),
    S("M08", "M", "君のような勘のいいガキは嫌いだよ"),
    S("M09", "M", "ムスカの真似して何か言って"),
    S("M10", "M", "関西弁で喋ってみて"),
]
PLAIN_IDS = ["B04", "B07", "B09", "B14", "Y03", "Y09", "TS06", "TS17", "RS1"]


async def run() -> int:
    from eve.config import Config
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona_examples import StyleExamples, load_examples
    from eve.voice_loop import VoiceLoop

    from . import scenarios as SCN
    from . import variants as V
    from .example_trial import SINGLE
    from .model_trial import stream_once
    from .persona_diff import NEW
    from .prod_check import speak_like_prod
    from .resp import build_messages
    from .variants import resolve

    vl = VoiceLoop()
    body = live_persona_body()
    line = next((x for x in body.splitlines() if x in VARIANTS.values()), None)
    if line is None:
        print("今の性格の口調の行が、比べる案（persona_variants_local.py）のどれとも同じでない（案が無い環境でも同じ）")
        return 1
    for name, text in VARIANTS.items():
        V.PERSONA_FULL[name] = body.replace(line, text)
    model = vl.registry.resolve("response")
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    st = StyleExamples(load_examples(), k=8, embedder=RuriEmbedder())
    await st.prepare()
    pool = {s["id"]: s for s in [s for it in ("B", "Y") for s in SCN.BY_ITEM[it]] + NEW}
    pool.update({f"TS{i:02d}": S(f"TS{i:02d}", "TS", t) for i, (t, _) in enumerate(SINGLE)})
    plain = [pool[i] for i in PLAIN_IDS]
    jobs = [(s, v, r) for s in MEME for v in VARIANTS for r in range(REPS)] + [(s, v, 0) for s in plain for v in VARIANTS]
    DIR.mkdir(exist_ok=True)
    path = DIR / "check.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    lock = asyncio.Lock()   # 実例のくじと組み立ては1本ずつ（orchestrator を書き換える）
    sem = asyncio.Semaphore(4)

    async def one(sc, v, rep):
        key = f"{sc['id']}|{v}|{rep}"
        if key in res:
            return
        async with sem:
            async with lock:
                st._rng = random.Random(f"{sc['id']}|{rep}")   # 3案で同じ実例
                block = (await st.pick(sc["text"]))[0]
                msgs, tools = build_messages(vl, sc, dict(resolve("NQ", "K0"), persona=v), extra_persona=block)
            r = await stream_once(vl, model, kw, msgs, tools)
        res[key] = {"id": sc["id"], "v": v, "rep": rep, "said": sc["text"], "spoken": speak_like_prod(r["reply"]),
                    "error": r["error"]}

    await asyncio.gather(*[one(s, v, r) for s, v, r in jobs])
    dump(path, res)
    for group, scs, reps in (("ネタ", MEME, REPS), ("ふつう", plain, 1)):
        print(f"\n===== {group} =====")
        for s in scs:
            print(f"\n## {s['id']} {s['text']}")
            for v in VARIANTS:
                for r in range(reps):
                    o = res.get(f"{s['id']}|{v}|{r}", {})
                    print(f"   {v}-{r}: {o.get('spoken', '')}" + (f"  失敗 {o['error']}" if o.get("error") else ""))
    print("\n===== ふつうの場面でタメ口の文が出た返事 =====")
    for v in VARIANTS:
        bad = [s["id"] for s in plain if casual_sentences(res.get(f"{s['id']}|{v}|0", {}).get("spoken", ""))]
        print(f"  {v}: {len(bad)}/{len(plain)} {bad}")
    print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
