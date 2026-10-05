#!/usr/bin/env python3
"""匿名シート（blind/sheet.json）を API のモデルに採点させる。どの案の出力かは渡さない。実 API（少額）。

  python -m tools.prompt_eval.judge_api --model anthropic/claude-sonnet-5-5 --name sonnet
  python -m tools.prompt_eval.judge_api --model openai/gpt-6-luna --name luna
→ eval_report/prompts/blind/judge_<name>.json（judges.py が集計する）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time

from .common import OUT, dump, gather_limited, load, setup_env

SYSTEM = """\
あなたは日本語の音声会話AIの応答を採点する審査員です（独立採点）。
対象は AI VTuber「イブ」（ユーザの友達として音声で話す AI。応答はそのまま声で読み上げられる。1秒に約5文字）。
1つの場面について、context（直前の会話・ユーザの発話・イブの今の気持ち・直前に口にしたつなぎ言葉・期待 など）と、
いくつかの応答（rid と text）が渡される。どの応答がどの作り方で出たかは伏せてある。
text の末尾の「〔※この後、調べ物・作業のタスクを開始〕」は、その一言の後に裏でタスクを始めたことを表す（場面に合っているかも判断に含める）。

各応答について、score_items に書かれた項目を付ける:
- 全場面: natural（1〜5の整数）と flags（当てはまる語のリスト。無ければ []。語は score_items に列挙されたものだけ）
- item が P2: expect_met（0/1/2）も
- item が R: reaction（1〜5）も
- item が L2: length（"short" / "ok" / "long"）も
- item が F: continuation（1〜3）も（直前に口にしたつなぎ言葉＋応答を続けて声に出したと想像して判断）

- item が B3: score_items に並んだ項目（natural・humor・distance・ai・expect_met・flags）をすべて
心がけ: 人間の友達が声で言いそうかを最優先。点は各応答について絶対評価（相対順位ではない）。迷ったら中間に逃げず、はっきりした方を選ぶ。
出力は JSON だけ（説明文・コードブロック記号は書かない）: {"<rid>": {"natural": 4, "flags": [], ...}, ...} で、渡された rid をすべて含める。"""


def _content(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return ""


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


async def run(args) -> int:
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    sheet = load(OUT / args.dir / "sheet.json")["scenes"]
    out_path = OUT / args.dir / f"judge_{args.name}.json"
    res = load(out_path) if out_path.exists() else {}
    role = "feedback"  # 採点の呼び出しは役割を借りる（set_override でモデルだけ差し替え・使用量は役割別に記録される）
    vl.registry.set_override(role, args.model)
    kw = {}
    if "gpt-6" in args.model:
        kw = {"response_format": {"type": "json_object"}}

    async def one(scene):
        rids = [r["rid"] for r in scene["responses"]]
        if all(rid in res for rid in rids):
            return
        user = json.dumps({"scene": scene["scene"], "item": scene["item"], "context": scene["context"],
                           "score_items": scene["score_items"], "responses": scene["responses"]},
                          ensure_ascii=False, indent=1)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        got = {}
        for attempt in range(3):
            try:
                got = _parse(_content(await vl.registry.complete(role, msgs, **kw)))
            except Exception as e:  # noqa: BLE001
                print(f"{scene['scene']} 失敗: {type(e).__name__}: {str(e)[:120]}")
                await asyncio.sleep(3 * (attempt + 1))
                continue
            if all(rid in got for rid in rids):
                break
        for rid in rids:
            if rid in got:
                res[rid] = got[rid]
        print(f"{scene['scene']:8s} {scene['item']:3s} {sum(1 for r in rids if r in got)}/{len(rids)}")

    t0 = time.monotonic()
    await gather_limited([one(s) for s in sheet], limit=args.parallel)
    dump(out_path, res)
    print(f"採点 {len(res)} 件 → {out_path}（{time.monotonic() - t0:.0f}秒）")
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--dir", default="blind", help="シートのフォルダ（blind / blind_final）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
