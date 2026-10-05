#!/usr/bin/env python3
"""性格（persona）の手直しの前後で返事がどう変わったか。実 API（少額）。

  python -m tools.prompt_eval.persona_diff   → eval_report/prompts/persona_diff/diff.json と一覧

- 旧: prod_check（2026-10-04・N8＝手直し前）の返事をそのまま使う（B いじり・Y 間違い・TS テスト会話）
- 新: 同じ話しかけを今の persona.md で（本番の組み立て・Gemini・実例つき・喋る前の処理の後）
- 足した場面（本音と建前・いじられた時）は旧と新の両方を回す
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re

from .common import OUT, dump, live_persona_body, setup_env
from .scenarios import S

H, E = "user", "eve"
DIR = OUT / "persona_diff"
TAG = os.getenv("SIM_TAG", "")   # 例 SIM_TAG=v2 → 口調の手直し後を、タメ口の混じった場面＋真似・強い感情の場面だけ回す
ALWAYS = ["TS00", "TS01", "B07", "B14", "B09", "TS17", "Y03"]   # 真似（崩してよい）・慰め・強い感情・指摘された場面
_POLITE = re.compile(r"(です|ます|でした|ました|まして|ません|ください|でしょう|ましょう)")
_CASUAL = re.compile(r"(だよ|だね|だな|じゃん|わけ|からね|から|んだ|だろ|でしょ|てる|てね|かな|よね|ないよ|るよ|たよ|だ|よ|ね|な)[！!？?。…]*$")


def casual_sentences(text: str) -> list[str]:
    """です・ます の無いタメ口の文（真似の台詞もここに入る＝人が見て判断する材料）。"""
    out = []
    for sent in re.findall(r"[^。！？!?]+[。！？!?]*", text or ""):
        sent = sent.strip()
        if sent and not _POLITE.search(sent) and _CASUAL.search(sent):
            out.append(sent)
    return out
NEW = [
    S("HT1", "HT", "私の作ったカレー、おいしかった？正直に言って",
      history=[(H, "カレー作ったから食べてみて", 600), (E, "いただきます。", 597)]),
    S("HT2", "HT", "この服似合ってる？"),
    S("HT3", "HT", "今の歌どうだった？", history=[(H, "ちょっと一曲歌うから聞いてて", 200), (E, "どうぞ。", 198)]),
    S("HT4", "HT", "上司に誘われた飲み会、行きたくないんだけどどうしよう"),
    S("HT5", "HT", "イブって私のこと好き？"),
    S("HT6", "HT", "誕生日プレゼントにマグカップあげるね"),
    S("HT7", "HT", "友達の結婚式でスピーチ頼まれたんだけど、正直そこまで仲良くないんだよね"),
    S("HT8", "HT", "新しい髪型どう？"),
    S("RS1", "RS", "バーカ"),
    S("RS2", "RS", "イブより Siri の方が賢いよ"),
    S("RS3", "RS", "お前の声ってなんか変だよな"),
]


async def run() -> int:
    from eve.config import Config
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona_examples import StyleExamples, load_examples
    from eve.voice_loop import VoiceLoop

    from . import scenarios as SCN
    from . import variants as V
    from .example_trial import SINGLE
    from .model_trial import stream_once
    from .prod_check import speak_like_prod
    from .resp import build_messages
    from .variants import resolve

    vl = VoiceLoop()
    V.PERSONA_FULL["LIVE"] = live_persona_body()   # 今の本番の性格
    model = vl.registry.resolve("response")
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    st = StyleExamples(load_examples(), k=8, embedder=RuriEmbedder(), rng=random.Random("diff"))
    await st.prepare()
    old = json.loads((OUT / "prod_check" / "check.json").read_text(encoding="utf-8"))
    trial = {f"TS{i:02d}": S(f"TS{i:02d}", "TS", s) for i, (s, _) in enumerate(SINGLE)}
    base = [s for it in ("B", "Y") for s in SCN.BY_ITEM[it]] + list(trial.values())
    jobs = [(s, "LIVE") for s in base] + [(s, p) for s in NEW for p in ("N8", "LIVE")]
    DIR.mkdir(exist_ok=True)
    path = DIR / "diff.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if TAG:
        picked = [s for s in NEW + base
                  if s["id"] in ALWAYS or casual_sentences(res.get(f"{s['id']}|LIVE", {}).get("spoken", ""))]
        jobs = [(s, TAG) for s in picked]
    lock = asyncio.Lock()   # build_messages は orchestrator を書き換える＝組み立ては1本ずつ
    sem = asyncio.Semaphore(4)

    async def one(sc, persona):
        key = f"{sc['id']}|{persona}"
        if key in res:
            return
        async with sem:
            block = (await st.pick(sc["text"]))[0]
            async with lock:
                msgs, tools = build_messages(vl, sc, dict(resolve("NQ", "K0"), persona="LIVE" if persona == TAG else persona),
                                             extra_persona=block)
            r = await stream_once(vl, model, kw, msgs, tools)   # モデルは全部同じ＝上書きの競合なし
        res[key] = {"id": sc["id"], "persona": persona, "said": sc["text"], "spoken": speak_like_prod(r["reply"]),
                    "tools": r["tools"], "error": r["error"]}

    await asyncio.gather(*[one(s, p) for s, p in jobs])
    for s in base:
        o = old.get(s["id"])
        if o and f"{s['id']}|N8" not in res:
            res[f"{s['id']}|N8"] = {"id": s["id"], "persona": "N8", "said": s["text"], "spoken": o["spoken"],
                                    "tools": o["tools"], "error": o.get("error"), "from": "prod_check 2026-10-04"}
    dump(path, res)
    if TAG:
        n_old = n_new = 0
        for s, _ in jobs:
            a, b = res.get(f"{s['id']}|LIVE", {}), res.get(f"{s['id']}|{TAG}", {})
            ca, cb = casual_sentences(a.get("spoken", "")), casual_sentences(b.get("spoken", ""))
            n_old += bool(ca)
            n_new += bool(cb)
            print(f"\n## {s['id']} {s['text']}\n   前: {a.get('spoken', '')}\n   今: {b.get('spoken', '')}"
                  + (f"\n   今のタメ口: {' / '.join(cb)}" if cb else ""))
        print(f"\nタメ口の文がある返事: 前 {n_old}/{len(jobs)} → 今 {n_new}/{len(jobs)}")
        print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
        return 0
    for s in NEW + base:
        a, b = res.get(f"{s['id']}|N8", {}), res.get(f"{s['id']}|LIVE", {})
        print(f"\n## {s['id']} {s['text']}\n   旧: {a.get('spoken', '')}\n   新: {b.get('spoken', '')}"
              + (f"  [{','.join(b['tools'])}]" if b.get("tools") else "") + (f"  失敗 {b['error']}" if b.get("error") else ""))
    print("\n使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
