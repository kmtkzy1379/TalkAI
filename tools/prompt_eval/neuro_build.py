#!/usr/bin/env python3
"""Neuro-sama のやりとり（`neifuisan/Neuro-sama-QnA`・Apache-2.0・配信のやりとりを作者が書き起こした 520組）を、
イブとユーザの日本語の会話に移す。実 API（Sonnet 5.5・少額）。

  python -m tools.prompt_eval.neuro_build
→ eval_report/prompts/neuro/neuro_ja.json（split=eval: 評価の場面 ／ split=pool: 話し方の例の置き場。重ならない）

2026-10-01 ユーザ「neuro-sama の対話抽出できるならそれを指標にしてほしい。AI が評価すると無難でマイルドな結果こそ自然とする」
→ 同じ話しかけに Neuro-sama が実際に返した言葉を、ものさしにする。
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
import re

from .common import OUT, dump, gather_limited, load, setup_env

SRC = OUT / "neuro" / "Manually_dataset.csv"
DST = OUT / "neuro" / "neuro_ja.json"
N_SAMPLE = 110
N_EVAL = 40
# 配信の外に持ち出せない話（特定の人・配信の出来事・投げ銭）は最初から外す
LORE = re.compile(r"\b(vedal|evil|anny|filian|bits?|twitch|chat|swarm|bilibili|stream\w*|koko|camila|toma|ellie|cerber|"
                  r"mini|layna|subathon|donat\w*|turtle|abandoned archive|minecraft|osu|discord|youtube|vtuber|collab)\b",
                  re.I)

SYSTEM = """\
あなたは英日の翻訳者です。英語圏の AI VTuber「Neuro-sama」の配信でのやりとり（相手の発言 → Neuro-sama の返事）を、
日本語の AI VTuber「イブ」と、イブの友達で開発者でもあるユーザとの1対1の会話に移します。

各項目について:
- ja_q: ユーザがイブに話しかける、自然な日本語の話し言葉（タメ口）。"Neuro"・"Neuro-sama" は「イブ」にする。
- ja_a: イブの返事。**Neuro-sama の返事の態度をそのまま残す**（遠慮のなさ・毒・皮肉・突飛さ・根拠のない自慢・とぼけ・間違いを
  堂々と言う・短さ）。丁寧にしない。和らげない。説明・気遣い・質問・謝罪を足さない。長さは元の返事と同じくらい。
  イブは女の子の AI で、くだけた話し言葉で話す。英語のダジャレや言葉遊びは、日本語で同じ種類の笑いに置き換えてよい。
- skip: 特定の人物・配信の出来事・英語でしか成り立たない話で、1対1の日本語の会話に移せない時だけ true（その時 ja_q/ja_a は空でよい）。

出力は JSON だけ: {"items": [{"i": 番号, "skip": false, "ja_q": "…", "ja_a": "…"}, ...]}。渡された番号をすべて含める。"""


def _content(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return ""


def _parse(text: str) -> list:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return []
    return d.get("items") or [] if isinstance(d, dict) else []


async def run() -> int:
    from eve.voice_loop import VoiceLoop

    with open(SRC, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("output") or "").strip() and (r.get("instruction") or "").strip()]
    cand = [(i, r) for i, r in enumerate(rows) if not LORE.search(r["instruction"] + " " + r["output"])]
    rng = random.Random(20261001)
    picked = rng.sample(cand, min(N_SAMPLE, len(cand)))
    print(f"元 {len(rows)}組 → 配信の外に持ち出せるもの {len(cand)}組 → 抽出 {len(picked)}組")

    vl = VoiceLoop()
    role = "feedback"  # 役割を借りてモデルだけ差し替える（使用量は役割別に記録される）
    vl.registry.set_override(role, "anthropic/claude-sonnet-5-5")
    done = {d["i"]: d for d in (load(DST)["items"] if DST.exists() else [])}
    todo = [(i, r) for i, r in picked if i not in done]
    batches = [todo[k:k + 22] for k in range(0, len(todo), 22)]

    async def one(batch):
        user = json.dumps([{"i": i, "said": r["instruction"].strip(), "neuro": r["output"].strip()} for i, r in batch],
                          ensure_ascii=False, indent=1)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        for attempt in range(3):
            try:
                got = {d.get("i"): d for d in _parse(_content(await vl.registry.complete(role, msgs)))}
            except Exception as e:  # noqa: BLE001
                print(f"失敗: {type(e).__name__}: {str(e)[:120]}")
                await asyncio.sleep(3 * (attempt + 1))
                continue
            if all(i in got for i, _ in batch):
                for i, r in batch:
                    d = got[i]
                    done[i] = {"i": i, "en_q": r["instruction"].strip(), "en_a": r["output"].strip(),
                               "skip": bool(d.get("skip")), "ja_q": (d.get("ja_q") or "").strip(),
                               "ja_a": (d.get("ja_a") or "").strip()}
                print(f"翻訳 {len(batch)}件")
                return
        print("この束は訳せなかった")

    await gather_limited([one(b) for b in batches], limit=3)
    order = [i for i, _ in picked]
    usable = [done[i] for i in order if i in done and not done[i]["skip"] and done[i]["ja_q"] and done[i]["ja_a"]]
    for k, d in enumerate(usable):
        d["split"] = "eval" if k < N_EVAL else "pool"
        d["id"] = f"NQ{k:02d}"
    dump(DST, {"source": "huggingface.co/datasets/neifuisan/Neuro-sama-QnA (Apache-2.0)",
               "items": usable + [done[i] for i in order if i in done and done[i] not in usable]})
    print(f"使えるもの {len(usable)}組（評価 {min(N_EVAL, len(usable))}・例の置き場 {max(0, len(usable) - N_EVAL)}）→ {DST}")
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
