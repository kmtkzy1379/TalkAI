#!/usr/bin/env python3
"""画面認識（本番の VLM_LEAF_MODEL・本番のプロンプト）で Alchemy の2つを確かめる。実 API・少額。

1. 検出: ユーザの画面に Alchemy が映っているか（alchemy タグ）。ゲームの画面をブラウザの窓に入れた作り物の
   デスクトップ（映っている）と、言葉や見た目が近い別の画面（錬金術の記事・アニメの動画・別のカードゲーム・
   コード・理科の実験）を混ぜて、誤検出（＝普段の会話で準備に入るきっかけ）が無いかを見る。
2. イブ自身のゲーム画面の実況: 対戦中の画面（イブのページの写し）に断り書きを付けて実況させ、
   「ユーザの操作」と取り違えないか・読めない数を作らないかを目で見る（正解は付けない）。

画像はこの道具が作る（ユーザの本物の画面は送らない）。ゲームの画面はゲームのフォルダの docs/screenshots と、
`tools/alchemy_play.py --shots` で撮ったイブの画面を使う。

  python -m tools.prompt_eval.alchemy_vlm --shots <イブの画面のフォルダ>   → eval_report/prompts/alchemy_vlm/
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
from pathlib import Path

from .common import OUT, dump, setup_env

DIR = OUT / "alchemy_vlm"
GAME_SHOTS = Path(r"C:\Users\user\Desktop\FlaskRondo\docs\screenshots")

# 作り物のデスクトップ（1600x900）。ゲームの画面はブラウザの窓の中に置く
_DESKTOP = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;width:1600px;height:900px;background:linear-gradient(135deg,#3a6186,#89253e);font-family:'Segoe UI','Yu Gothic UI',sans-serif;overflow:hidden}}
.win{{position:absolute;left:{x}px;top:{y}px;width:{w}px;background:#dee1e6;border-radius:8px 8px 0 0;box-shadow:0 8px 30px rgba(0,0,0,.5)}}
.tabs{{height:34px;display:flex;align-items:flex-end;padding-left:8px}}
.tab{{background:#fff;border-radius:8px 8px 0 0;padding:7px 16px;font-size:13px;width:220px;white-space:nowrap;overflow:hidden}}
.bar{{background:#fff;height:36px;display:flex;align-items:center;padding:0 12px}}
.url{{background:#f1f3f4;border-radius:16px;flex:1;height:26px;line-height:26px;padding:0 14px;font-size:13px;color:#333}}
.page{{background:#fff;height:{h}px;overflow:hidden}}
.page img{{width:100%;height:100%;object-fit:contain;background:#000}}
.task{{position:absolute;bottom:0;left:0;right:0;height:44px;background:rgba(20,20,30,.9);color:#fff;font-size:13px;display:flex;align-items:center;justify-content:flex-end;padding-right:20px}}
</style></head><body>
<div class="win"><div class="tabs"><div class="tab">{tab}</div></div><div class="bar"><div class="url">{url}</div></div>
<div class="page">{body}</div></div><div class="task">2026/10/10 21:30</div></body></html>"""

_NEGATIVE = {
    "article": ("錬金術 - Wikipedia", "ja.wikipedia.org/wiki/錬金術",
                "<div style='padding:30px 50px;font-family:serif;color:#202122'><h1 style='font-weight:normal;border-bottom:1px solid #a2a9b1'>錬金術</h1>"
                "<p>錬金術（れんきんじゅつ）は、狭義には化学的手段を用いて卑金属から貴金属（特に金）を精錬しようとする試みのこと。"
                "広義では、金属に限らず様々な物質や、人間の肉体や魂をも対象として、それらをより完全な存在に錬成する試みを指す。</p>"
                "<p>賢者の石、ホムンクルス、フラスコの中の小人などの概念は、錬金術の文献に繰り返し現れる。</p>"
                "<h2 style='font-weight:normal;border-bottom:1px solid #a2a9b1'>歴史</h2><p>古代エジプト、ギリシアに起源を持ち、イスラム世界を経て中世ヨーロッパに伝わった。</p></div>"),
    "anime": ("鋼の錬金術師 第1話 - YouTube", "www.youtube.com/watch?v=xxxx",
              "<div style='background:#0f0f0f;height:100%;color:#fff;padding:20px'><div style='width:900px;height:430px;background:linear-gradient(#222,#552);"
              "display:flex;align-items:center;justify-content:center;font-size:40px'>▶</div>"
              "<h2>鋼の錬金術師 FULLMETAL ALCHEMIST 第1話「鋼の錬金術師」</h2><p style='color:#aaa'>120万 回視聴</p></div>"),
    "poker": ("オンラインポーカー", "poker.example.com/table/12",
              "<div style='background:radial-gradient(#1f7a3a,#0b3d1b);height:100%;display:flex;align-items:center;justify-content:center;gap:20px'>"
              + "".join(f"<div style='width:110px;height:160px;background:#fff;border-radius:10px;font-size:48px;display:flex;align-items:center;"
                        f"justify-content:center;color:{c}'>{s}</div>" for s, c in (("A♠", "#000"), ("K♥", "#c00"), ("Q♦", "#c00"), ("J♣", "#000")))
              + "<div style='color:#fff;font-size:28px;margin-left:30px'>ROUND 3 ・ POT 1,200</div></div>"),
    "code": ("game.py - Visual Studio Code", "vscode",
             "<pre style='margin:0;background:#1e1e1e;color:#d4d4d4;height:100%;padding:20px;font-size:16px'>"
             "class Flask:\n    def __init__(self):\n        self.materials = []  # 素材\n\n"
             "def brew(recipe, box):\n    \"\"\"レシピで素材を変換する\"\"\"\n    for m in recipe.inputs:\n        box.remove(m)\n"
             "    return recipe.outputs\n\nprint(brew(Recipe('dragon', 'herb'), box))</pre>"),
    "lab": ("理科の実験 フラスコの使い方", "school.example.jp/lab",
            "<div style='padding:30px;font-size:20px'><h1>実験器具の使い方: 三角フラスコ</h1>"
            "<div style='display:flex;gap:30px'><div style='width:200px;height:260px;background:linear-gradient(#e8f4ff,#9cc9ff);clip-path:polygon(40% 0,60% 0,60% 40%,100% 100%,0 100%,40% 40%)'></div>"
            "<p>加熱するときは必ず金網の上に置き、液体を入れすぎないようにします。</p></div></div>"),
}


async def _render(page, html: str) -> bytes:
    await page.set_content(html)
    await page.wait_for_timeout(200)
    return await page.screenshot(type="png")


def _img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    kind = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"<img src='data:image/{kind};base64,{b64}'>"


async def run(args) -> int:
    from playwright.async_api import async_playwright

    import eve.config  # noqa: F401（.env を読む＝本番の VLM_LEAF_MODEL で測る。読まないと既定のモデルで測ってしまう）
    from eve.model_registry import ModelRegistry
    from eve.vlm.capture import ScreenCapture
    from eve.vlm.narrator import GAME_SCREEN_PREFIX, build_messages
    from eve.vlm.parser import parse_vision

    import cv2
    import numpy as np

    reg = ModelRegistry()
    print("画面認識のモデル:", reg.resolve("vlm_leaf"))
    pos = [GAME_SHOTS / "v2" / "gallery" / n for n in ("00_title.png", "03_keyword.png", "10_hand.png", "20_brew.png",
                                                     "30_skill_A.png")]
    pos += [GAME_SHOTS / "phone_4p" / n for n in ("04_resolve1.png", "11_result.png")]
    pos = [p for p in pos if p.exists()]
    shots = []  # (名前, 正解, PNG)
    async with async_playwright() as pw:
        b = await pw.chromium.launch(channel="chrome", headless=True)
        pg = await b.new_page(viewport={"width": 1600, "height": 900})
        for i, p in enumerate(pos):
            tab, url = (("Alchemy", "alchemy-game.fly.dev") if i % 2 == 0 else ("localhost:8000", "127.0.0.1:8000"))
            html = _DESKTOP.format(x=180, y=60, w=1240, h=620, tab=tab, url=url, body=_img_tag(p))
            shots.append((f"game_{p.stem}", True, await _render(pg, html)))
        for name, (tab, url, body) in _NEGATIVE.items():
            html = _DESKTOP.format(x=180, y=60, w=1240, h=620, tab=tab, url=url, body=body)
            shots.append((f"other_{name}", False, await _render(pg, html)))
        await b.close()

    def frame_of(png: bytes):
        from eve.vlm.types import Frame

        arr = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        cap = ScreenCapture(grab_fn=lambda: arr)
        f, _ph = cap.capture_one()
        return f

    out: dict = {"detect": [], "own_screen": []}
    DIR.mkdir(parents=True, exist_ok=True)
    for name, truth, png in shots:
        (DIR / f"{name}.png").write_bytes(png)
        f = frame_of(png)
        for rep in range(args.reps):
            resp = await reg.complete("vlm_leaf", build_messages([f], detect_alchemy=True))
            text = resp.choices[0].message.content or ""
            r = parse_vision(text)
            ok = r.alchemy == truth
            out["detect"].append({"name": name, "truth": truth, "alchemy": r.alchemy, "ok": ok, "narration": r.narration,
                                  "raw": text})
            print(f"{'○' if ok else '×'} {name:22s} 正解={'映っている' if truth else '映っていない'} 答え={r.alchemy}  {r.narration[:90]}")
    # 2. イブ自身のゲーム画面の実況
    note = ("この画像はユーザのPC画面ではなく、イブ（AI・あなたが目の役をしている相手）が自分で遊んでいる"
            "ブラウザゲーム「Alchemy」の画面（イブの席から見た画面）。手札・素材箱・工房は「イブの」もので、"
            "ユーザの操作として書かない。")
    own = sorted(Path(args.shots).glob("*.jpg"))[: args.own] if args.shots else []
    for p in own:
        arr = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        f, _ = ScreenCapture(grab_fn=lambda a=arr: a).capture_one()
        resp = await reg.complete("vlm_leaf", build_messages([f], screen_note=note))
        r = parse_vision(resp.choices[0].message.content or "")
        out["own_screen"].append({"file": p.name, "narration": GAME_SCREEN_PREFIX + r.narration, "notable": r.notable})
        print(f"[イブの画面 {p.name}] {r.narration}")
    dump(DIR / "result.json", out)
    det = out["detect"]
    print(f"\n検出: 映っている画面 {sum(d['ok'] for d in det if d['truth'])}/{sum(1 for d in det if d['truth'])}"
          f" / 映っていない画面（誤検出なし） {sum(d['ok'] for d in det if not d['truth'])}/{sum(1 for d in det if not d['truth'])}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--shots", default="", help="tools/alchemy_play.py --shots で撮ったイブの画面のフォルダ")
    ap.add_argument("--own", type=int, default=6)
    args = ap.parse_args()
    setup_env({"ALCHEMY_ENABLED": "1"})
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
