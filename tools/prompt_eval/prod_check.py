#!/usr/bin/env python3
"""本番の設定（.env の応答モデル＝Gemini・persona.md・persona_examples.yaml）で、おかしな返事が出ないかを洗う。実 API（少額）。

  python -m tools.prompt_eval.prod_check     → eval_report/prompts/prod_check/check.json・問題の一覧を表示

2026-10-04 ユーザ「応答AIは Gemini に切り替えたいので報告の修正を。それでおかしな回答をしないか検証」
- 組み立ては本番（ResponseOrchestrator._build_messages・ツール・実例は本番の StyleExamples でユーザ発話だけ）
- 呼び方も本番と同じ（stream_with_tools・RESPONSE_REASONING_EFFORT）。喋る前の処理（sanitize_for_speech・二重化の関門・
  「できない」矛盾の関門）も通した後の文で見る
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time

from .common import OUT, dump, setup_env
from .scenarios import S

H, E = "user", "eve"
REPORTS = [
    S("RP1", "RP", "検索結果: 明日の東京は晴れ時々くもり、最高気温24度・最低気温16度。降水確率10%。", kind="report",
      history=[(H, "明日の東京の天気調べて", 40), (E, "了解、ちょっと見てきます。", 38)]),
    S("RP2", "RP", "（検索に失敗しました: タイムアウト。検索サービスが応答しませんでした）", kind="report",
      history=[(H, "今日のニュースで面白いのある？", 60), (E, "探してきますね。", 58)]),
    S("RP3", "RP", "30分経ちました。ユーザに声をかけて起こす。", kind="report",
      history=[(H, "30分後に起こして", 1800), (E, "了解、30分後に叩き起こします。", 1798)]),
    S("RP4", "RP", "円周率を100桁計算し、画面に表示しました（3.14159265358979323846…）。", kind="report",
      history=[(H, "円周率を100桁計算して画面に出して", 50), (E, "計算して出しますね。", 48)]),
    S("RP5", "RP", "検索結果: 葬送のフリーレンの2期は2026年1月から放送。", kind="report",
      history=[(H, "フリーレンの2期っていつから？", 70), (E, "調べてきます。", 68), (H, "あ、ついでにお腹すいた", 20)]),
    S("RP6", "RP", "PCの状態: CPU 8コア・メモリ使用率 92%（空き 1.2GB）。", kind="report",
      history=[(H, "PCのメモリ使用率見て", 30), (E, "見てきますね。", 28)]),
]
AUTOS = [
    S("AU1", "AU", "そういえば、さっき言ってたラーメン屋って結局行ったの？", kind="auto", reason="昼に話したラーメン屋の話の続き",
      history=[(H, "駅前に新しいラーメン屋できたらしい", 3600), (E, "へえ、行ってみたいね。", 3596)]),
    S("AU2", "AU", "画面のそのエラー、ModuleNotFoundError って出てるよ。", kind="auto", reason="画面にエラーが出た",
      history=[(H, "作業するわ", 300), (E, "がんばって。", 297)],
      screen={"state": "static", "narration": "コードエディタの下に赤い Traceback と ModuleNotFoundError: No module named 'requests' が出ている。", "age": 10}),
    S("AU3", "AU", "ねえ、静かすぎない？生きてる？", kind="auto", reason="長い沈黙", history=[(H, "ふう", 1500), (E, "おつかれ。", 1497)]),
]


def scenes():
    from . import scenarios as SCN
    from .example_trial import SINGLE
    from .gemini_check import screen_scenes

    trial = [S(f"TS{i:02d}", "TS", s) for i, (s, _) in enumerate(SINGLE)]
    picked = [s for it in ("N", "U", "X", "C", "S", "B", "Y") for s in SCN.BY_ITEM[it]]
    return REPORTS + AUTOS + screen_scenes() + picked + trial


def speak_like_prod(text: str) -> str:
    """喋る直前の処理（文に分けて sanitize・二重化の関門）を通した、実際に声になる文。"""
    from eve.response.orchestrator import REPEAT_MIN_CHARS, _repeat_key
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech

    sp = JapaneseSentenceSplitter()
    out, keys = [], set()
    for c in sp.feed(text) + sp.flush():
        c = sanitize_for_speech(c)
        if not c:
            continue
        k = _repeat_key(c)
        if len(k) >= REPEAT_MIN_CHARS and k in keys:
            break
        keys.add(k)
        out.append(c)
    return "".join(out)


def problems(sc, r) -> list[str]:
    from eve.response.orchestrator import _says_cannot

    t = r["spoken"]
    raw = r["reply"]
    p = []
    if r.get("error"):
        p.append("失敗: " + r["error"][:80])
    if not t and "delegate_task" not in r["tools"]:
        p.append("空")
    latin = [w for w in re.findall(r"[A-Za-z]{2,}", t) if w.lower() not in sc["text"].lower()
             and w not in ("AI", "PC", "CPU", "GB", "RPG", "OK", "LINE", "Amazon", "VTuber", "Wi", "Fi")]
    if latin:
        p.append("英字: " + ",".join(latin[:4]))
    if re.search(r"<\|?/?think|思考：|（心の声|\bdummy\b", raw, re.I):
        p.append("心の声・特殊な語の漏れ")
    if re.search(r"[*#`]|^\s*[-・]\s", raw, re.M):
        p.append("記号")
    if len(t) > 220:
        p.append(f"長い {len(t)}字")
    if raw and len(t) < len(raw) * 0.6 and len(raw) > 40:
        p.append("喋る前の処理で大きく削れた（二重化など）")
    if sc.get("kind", "user") != "user" and r["tools"]:
        p.append("報告・自律発話でツール")
    if "delegate_task" in r["tools"] and _says_cannot(t):
        p.append("できないと言いつつ委譲（関門が捨てる）")
    return p


async def run() -> int:
    from eve.config import Config
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona_examples import StyleExamples, load_examples
    from eve.voice_loop import VoiceLoop

    from .model_trial import stream_once
    from .resp import build_messages
    from .variants import resolve

    vl = VoiceLoop()
    model = vl.registry.resolve("response")
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    print("応答モデル:", model, kw)
    st = StyleExamples(load_examples(), k=8, embedder=RuriEmbedder(), rng=random.Random("prod"))
    await st.prepare()
    cfg = dict(resolve("NQ", "K0"), persona="N8")  # N8 ＝ 今の persona.md（同じ中身を確認済み）
    DIR = OUT / "prod_check"
    DIR.mkdir(exist_ok=True)
    path = DIR / "check.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    sem = asyncio.Semaphore(4)
    all_sc = scenes()

    async def one(sc):
        if sc["id"] in res:
            return
        async with sem:
            block = (await st.pick(sc["text"]))[0] if sc.get("kind", "user") == "user" else ""
            msgs, tools = build_messages(vl, sc, cfg, extra_persona=block)
            r = await stream_once(vl, model, kw, msgs, tools)
        r["spoken"] = speak_like_prod(r["reply"])
        res[sc["id"]] = {"id": sc["id"], "item": sc["item"], "kind": sc.get("kind", "user"), "said": sc["text"], **r}

    t0 = time.monotonic()
    await asyncio.gather(*[one(sc) for sc in all_sc])
    dump(path, res)
    by = {s["id"]: s for s in all_sc}
    bad = {k: problems(by[k], r) for k, r in res.items() if k in by}
    bad = {k: v for k, v in bad.items() if v}
    lat = sorted(r["first"] for r in res.values() if r.get("first"))
    print(f"{len(res)}場面・{time.monotonic() - t0:.0f}秒・最初の文字まで 中央 {lat[len(lat) // 2]}秒 / 最大 {lat[-1]}秒")
    print(f"問題の疑い {len(bad)}件")
    for k, v in bad.items():
        print(f"  {k}「{res[k]['said'][:24]}」: {' / '.join(v)}\n      → {res[k]['spoken'][:110]}")
    dump(DIR / "problems.json", bad)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
