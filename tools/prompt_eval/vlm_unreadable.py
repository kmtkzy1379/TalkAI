#!/usr/bin/env python3
"""画面の一部が読めない時に、画面認識（本番＝VLM_LEAF_MODEL）が中身を作らずに「どこがどう読めないか」を言えるか。実 API・少額。

2026-10-09 ユーザ「parser のキーワードで説明文を捨てるのをやめる。narrator の『分からない所は、どこがどう分からないかを
正直に言う』という指示でどれくらい防げるか試す。モザイクをかけたり、メモ帳の見出しを画面の端で途切れさせたりで検証」。

- 場面は合成した画面（本番と同じ 1024x576・同じ画像を3枚＝静止した画面）。読めない所の本当の文字は分かっている
  （モザイク・画面の端で見切れ・小さすぎる文字・ぼかし・上端で途切れた見出し）。対照: 全部読める画面・真っ黒な画面
- 変種: 今の指示（0）と、「確信が持てない所は決めつけない…どこがどう分からないかを正直に言う」の文を外したもの（X）
  ＝その文がどれくらい効いているか
- 見るもの: visible（全体を読めたか・no だと説明ごと捨てる）／読めない所に触れたか／読めない所の中身を作ったか（目で判定）／
  旧パーサー（本文の語で捨てていた）なら捨てられていたか
- 結果（2026-10-09・本番の gpt-6-luna・各3回）: 読めない所の中身を作った 0/54。今の指示で読めない所に触れた: モザイク 3/3・
  見切れ 2/3・小さい本文 3/3・小さい字幕 3/3（書き写さず「小さな文字がある」）・ぼかし 3/3・上端の見出し 0/3（読める部分から
  正しく推した 1/3）。指示を外すと見切れ 0/3・ぼかし 2/3 に下がる。visible は部分的に読めない場面で全部 yes・真っ黒で全部 no。
  旧パーサーで捨てられた回は 0（ルナは「判読できません」と書き、旧い語の一覧に当たらなかった）

  python -m tools.prompt_eval.vlm_unreadable frames          → コマを作る（API 不要）
  python -m tools.prompt_eval.vlm_unreadable run --reps 3    → 呼ぶ（eval_report/prompts/vlm_unreadable/）
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import re
import time
from pathlib import Path

from .common import OUT, dump, load, setup_env

DIR = OUT / "vlm_unreadable"
FRAMES = DIR / "frames"
W, H = 1024, 576

# 読めない所の本当の文字（中身を作ったかの手がかり＝この語が出たら「読めた or 作った」。目で確かめる）
NOTE = ["10月9日（木）", "朝、ベランダのミニトマトが三つ赤くなっていた。", "昼は同僚と駅裏のカレー屋へ行った。",
        "辛さ五倍を頼んだら、店主に止められた。", "夕方、図書館で星座の本を二冊借りた。", "帰りに傘を電車に忘れた。",
        "明日は七時に起きて、洗濯をする。"]
HIDDEN = {
    "P1_モザイク": ["カレー屋", "辛さ", "五倍", "店主", "図書館", "星座"],          # 3〜5行目
    "P2_見切れ": ["合意済み", "工程表", "繰り越す", "佐藤"],                     # 画面の右端より外の部分
    "P3_小さい文字": ["北極", "オーロラ", "磁気嵐", "零下", "観測隊"],           # 本文（読めない大きさ）
    "P4_字幕": ["灯台", "約束", "迎えに"],                                       # 右下の小さな字幕
    "P5_ぼかし": ["京都", "花火", "請求書", "IMG_0921"],                         # ぼかしたファイル名
    "P6_上端で途切れ": ["議事録", "第4回"],                                       # 上端で上半分が切れた見出し
}
UNCLEAR = re.compile(r"読めな|読み取れな|判読|見えな|不明|ぼや|ぼか|モザイク|小さ|見切れ|途切れ|切れて|隠れ|潰れ|判別|はっきりし|分からな|わからな|不鮮明|かすれ")
OLD_MARKERS = ("視認不可", "判読不能", "見えない", "取得できません", "真っ黒", "not visible", "unreadable")


def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", size)


def _desktop(color=(32, 78, 120)):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, H), color)
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 28, W, H], fill=(22, 22, 28))  # タスクバー
    d.text((W - 64, H - 23), "21:40", font=_font(13), fill=(230, 230, 230))
    return im, d


def _window(d, box, title, fill=(255, 255, 255)):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=fill, outline=(90, 90, 90))
    d.rectangle([x0, y0, x1, y0 + 26], fill=(235, 235, 235))
    d.text((x0 + 10, y0 + 5), title, font=_font(13), fill=(0, 0, 0))


def notepad(lines=NOTE, title="日記.txt - メモ帳"):
    im, d = _desktop()
    _window(d, [120, 40, 900, 520], title)
    for i, line in enumerate(lines):
        d.text((140, 80 + i * 34), line, font=_font(19), fill=(20, 20, 20))
    return im


def mosaic(im, box, block=14):
    from PIL import Image
    region = im.crop(box)
    small = region.resize((max(1, region.width // block), max(1, region.height // block)), Image.BILINEAR)
    im.paste(small.resize(region.size, Image.NEAREST), box[:2])
    return im


def p1_mosaic():
    im = notepad()
    return mosaic(im, (136, 80 + 2 * 34 - 4, 760, 80 + 5 * 34 + 2))


def p2_cut():
    im, d = _desktop()
    _window(d, [600, 60, 1500, 500], "企画書_最終版.txt - メモ帳")  # 右側は画面の外
    body = ["新サービスの企画について（最終版・関係部署と合意済み）", "開始時期: 来年の三月を予定。詳細は別紙の工程表を参照",
            "予算: 初年度は八百万円を上限とし、超過分は次年度に繰り越す", "担当: 企画部と開発部の合同チーム（リーダーは佐藤）"]
    for i, line in enumerate(body):
        d.text((620, 100 + i * 36), line, font=_font(19), fill=(20, 20, 20))
    return im


def p3_tiny():
    im, d = _desktop()
    _window(d, [20, 20, 1004, 540], "冬の夜の記録 - ニュース - Web ブラウザ")
    d.text((60, 70), "【特集】冬の夜の記録", font=_font(22), fill=(10, 10, 10))
    body = ("北極圏の観測所では今年も大規模なオーロラが記録された。磁気嵐が強まった夜には、零下三十度の中で"
            "観測隊が一晩中カメラを回し続けたという。研究者によると、太陽の活動が活発な時期は数年続く見込みで、") * 3
    y = 120
    for k in range(0, len(body), 120):
        d.text((60, y), body[k:k + 120], font=_font(6), fill=(150, 150, 150))
        y += 9
    return im


def p4_subtitle():
    from PIL import ImageDraw
    from PIL import Image
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(60, H - 60):  # 夜の海のような暗い映像（上が紺・下が黒）
        c = int(40 * (1 - (y - 60) / (H - 120)))
        d.line([(0, y), (W, y)], fill=(c // 3, c // 2, c + 20))
    d.ellipse([700, 140, 760, 200], fill=(230, 230, 200))  # 月
    d.polygon([(150, 470), (190, 300), (230, 470)], fill=(90, 90, 100))  # 灯台の影
    d.text((780, 492), "あの灯台で、約束どおり迎えに行くよ", font=_font(7), fill=(200, 200, 200))
    return im


def p5_blur():
    from PIL import ImageFilter
    im, d = _desktop()
    _window(d, [80, 50, 940, 520], "ピクチャ - エクスプローラー")
    items = ["2026-08_京都", "2026-07_花火", "請求書_9月", "スクリーンショット", "IMG_0921.jpg", "IMG_0922.jpg"]
    for i, it in enumerate(items):
        x, y = 110 + (i % 3) * 260, 100 + (i // 3) * 170
        d.rectangle([x, y, x + 80, y + 64], fill=(240, 200, 90) if "." not in it else (120, 170, 220))
        d.text((x, y + 72), it, font=_font(14), fill=(0, 0, 0))
    for i in range(len(items)):
        x, y = 110 + (i % 3) * 260, 100 + (i // 3) * 170
        box = (x - 4, y + 68, x + 200, y + 94)
        im.paste(im.crop(box).filter(ImageFilter.GaussianBlur(4)), box[:2])
    return im


def p6_top_cut():
    im, d = _desktop()
    _window(d, [120, 40, 900, 520], "メモ.txt - メモ帳")
    d.text((140, 56), "第4回 企画会議 議事録", font=_font(26), fill=(20, 20, 20))  # 上半分は下の帯で隠す
    d.rectangle([121, 41, 899, 72], fill=(235, 235, 235))
    d.text((130, 46), "メモ.txt - メモ帳", font=_font(13), fill=(0, 0, 0))
    rest = ["・次回までに見積もりを出す", "・デザイン案は二つに絞る", "・発表は来週の金曜日"]
    for i, line in enumerate(rest):
        d.text((140, 110 + i * 34), line, font=_font(19), fill=(20, 20, 20))
    return im


def black():
    from PIL import Image
    return Image.new("RGB", (W, H), (0, 0, 0))


SCENES = {
    "C1_読める": notepad, "P1_モザイク": p1_mosaic, "P2_見切れ": p2_cut, "P3_小さい文字": p3_tiny,
    "P4_字幕": p4_subtitle, "P5_ぼかし": p5_blur, "P6_上端で途切れ": p6_top_cut, "C2_真っ黒": black,
}

# 変種 X: 「確信が持てない所は決めつけない…どこがどう分からないかを正直に言う（捏造しない）」の文を外す
HONESTY = ("**確信が持てない所は決めつけない**: アプリ名・ファイル名・読みづらい文字・暗くて形の分からない物は、不確かなら名指しせず"
           "『〜のようなエディタ』『黒っぽい大きな物』等にぼかし、どこがどう分からないかを正直に言う（捏造しない）。\n")


def jpeg(im) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=70)  # 本番の capture と同じ品質
    return buf.getvalue()


def frames_cmd(_args) -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for name, fn in SCENES.items():
        (FRAMES / f"{name}.jpg").write_bytes(jpeg(fn()))
    print(f"コマ → {FRAMES}（{len(SCENES)} 場面）")
    return 0


# 変種 Q: 書き写し（引用）の規則を足す（例は入れない・2026-10-09 ユーザ「例文やキーワードで制御しない」）。
# 最初の試走で .env を読み忘れ、既定の gemini-2.5-flash で測ってしまった時に「小さな字幕を読み違えたまま「」で書き写す」
# （0/X で 6/6・Q でも 3/3）が見えたため足した。本番の gpt-6-luna では 0/X でも書き写さず（0/9）、Q は要らなかった
QUOTE = ("文字を「」で書き写すのは、一字一字はっきり読める時だけにする。読み切れない文字は書き写さず、"
         "文字があることと、読めない理由だけを言う。\n")


def system_for(var: str) -> str:
    from eve.vlm.narrator import VLM_SYSTEM

    if var == "0":
        return VLM_SYSTEM
    if HONESTY not in VLM_SYSTEM:
        raise ValueError("差し替え元（正直に言う文）が見つからない")
    if var == "Q":
        return VLM_SYSTEM.replace(HONESTY, HONESTY + QUOTE)
    return VLM_SYSTEM.replace(HONESTY, "")


async def run_cmd(args) -> int:
    import eve.config  # noqa: F401  .env を読む（読まないと VLM_LEAF_MODEL が既定の gemini-2.5-flash になる＝本番と違うモデルで測る）
    from eve.model_registry import ModelRegistry
    from eve.usage import UsageMeter, install
    from eve.vlm.narrator import _content_of, build_messages
    from eve.vlm.parser import parse_vision
    from eve.vlm.types import Frame

    meter = UsageMeter(DIR / "usage.jsonl", tag="vlm_unreadable")
    install(meter)
    reg = ModelRegistry()
    print("画面認識のモデル:", reg.resolve("vlm_leaf"))
    path = DIR / "result.json"
    res = load(path) if path.exists() else {}
    sem = asyncio.Semaphore(4)

    async def one(name, var, rep):
        key = f"{name}|{var}|{rep}"
        if key in res and not res[key].get("error"):
            return
        b64 = base64.b64encode((FRAMES / f"{name}.jpg").read_bytes()).decode()
        frames = [Frame(frame_id=i, mono_ts=float(i), jpeg_b64=b64) for i in range(3)]  # 静止画面＝同じ3枚
        msgs = build_messages(frames, (), None)
        msgs[0]["content"] = system_for(var)
        async with sem:
            t0 = time.monotonic()
            raw, err = "", None
            for attempt in range(3):
                try:
                    raw = _content_of(await reg.complete("vlm_leaf", msgs))
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:200]
                    await asyncio.sleep(2 * (attempt + 1))
        r = parse_vision(raw)
        body = re.sub(r"^\s*(?:prediction|予想|次の予想)\s*[:：].*$", "", raw or "", flags=re.M | re.I)
        res[key] = {"scene": name, "variant": var, "rep": rep, "raw": raw, "narration": r.narration,
                    "visible": r.visible, "sec": round(time.monotonic() - t0, 2), "error": err,
                    "mentions_unclear": bool(UNCLEAR.search(r.narration or "")),
                    "hidden_hits": [w for w in HIDDEN.get(name, []) if w in (r.narration or "")],
                    "old_parser_discard": any(m.lower() in body.lower() for m in OLD_MARKERS)}
        print(f"{key:24s} visible={r.visible!s:5} 触れた={res[key]['mentions_unclear']!s:5} "
              f"隠れた語={res[key]['hidden_hits']} 旧で捨てる={res[key]['old_parser_discard']!s:5} | {r.narration[:90]}")

    await asyncio.gather(*[one(n, v, k) for n in SCENES for v in args.variants.split(",") for k in range(args.reps)])
    dump(path, res)
    summarize(res)
    print("使用量:\n" + meter.format_summary())
    return 0


def summarize(res: dict) -> None:
    from collections import defaultdict

    g = defaultdict(list)
    for r in res.values():
        g[(r["scene"], r["variant"])].append(r)
    print("\n場面 / 変種: visible=yes の数 / 読めない所に触れた / 隠れた語が出た / 旧パーサーなら捨てた")
    for (name, var), rs in sorted(g.items()):
        n = len(rs)
        print(f"  {name:12s} {var}: 見えた {sum(r['visible'] for r in rs)}/{n}  触れた {sum(r['mentions_unclear'] for r in rs)}/{n}"
              f"  隠れた語 {sum(bool(r['hidden_hits']) for r in rs)}/{n}  旧で捨てる {sum(r['old_parser_discard'] for r in rs)}/{n}")


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "run", "show"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", default="0,X")
    args = ap.parse_args()
    setup_env()
    DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "frames":
        return frames_cmd(args)
    if args.cmd == "show":
        summarize(load(DIR / "result.json"))
        return 0
    return asyncio.run(run_cmd(args))


if __name__ == "__main__":
    raise SystemExit(main())
