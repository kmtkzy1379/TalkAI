#!/usr/bin/env python3
"""画面認識（本番＝VLM_LEAF_MODEL）の変種比べ。実 API・少額。

  narr : 実況の長さ（今＝1〜2文 / U＝重要な要素を簡潔に・事実のみ・感情を入れない）を Sintel の10場面（前回の評価と同じコマ・
         正解の要点つき）で比べる
  seq  : surprise を「前回の予想との食い違い」にする案（U）を、予想どおりの変化／予想外の変化が混ざった連続の場面で比べる
         （今＝予想を持たずに付ける surprise）。場面は合成した作業画面＋Sintel のコマ

実行（EVE の venv・TalkAI-live のフォルダで）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.vlm frames   # 連続の場面のコマを作る（API 不要）
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.vlm narr --reps 2
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.vlm seq --reps 2
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import re
import subprocess
import time
from pathlib import Path

from .common import OUT, dump, gather_limited, load, setup_env, sub

PREV = Path(r"C:\Users\user\AppData\Local\Temp\claude\c--Users-user-Desktop-TalkAI-live"
            r"\e80d2df5-2049-4ffc-a1ea-a599f28765bc\scratchpad\vlm_eval")
VIDEO = PREV / "sintel_seg.mkv"
FFMPEG = (r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe")
SEQ_DIR = OUT / "vlm_seq_frames"

# ── プロンプトの差し替え（U＝ユーザ案） ─────────────────────────────────────────────
NARR_0 = "画面で何が起きているか・どこが変わった/動いたかを日本語で簡潔に1〜2文で述べてください。\n"
NARR_U = ("画面で何が起きているか・どこが変わった/動いたかを日本語で述べてください。"
          "**重要な要素を簡潔に、見えている事実だけ**を書く（文の数は決めない。感想・感情・評価は入れない）。\n")
TAGS_0_NARR = "narration: <実況を1〜2文>\n"
TAGS_U_NARR = "narration: <実況（1行・改行しない）>\n"
TAGS_0_SUR = "surprise: <0-100＝直前の予想からの外れ具合>\n"
TAGS_U_SUR = ("prediction: <この後の画面がどうなりそうかの予想を短く1つ（見えている事実を根拠に）>\n"
              "surprise: <0-100＝「前回の予想」と今の画面の食い違い（0=予想どおり 100=まったく予想外。前回の予想が無ければ 20）>\n")


def system_for(var: str) -> str:
    from eve.vlm.narrator import VLM_SYSTEM

    if var == "0":
        return VLM_SYSTEM
    s = sub(VLM_SYSTEM, NARR_0, NARR_U)
    s = sub(s, TAGS_0_NARR, TAGS_U_NARR)
    s = sub(s, TAGS_0_SUR, TAGS_U_SUR)
    return s


def user_text(n: int, var: str, prediction: str | None) -> str:
    head = f"連続スクリーンショット {n} 枚（古い順・最後が今）。今の画面を実況して。"
    if var == "U":
        head += f"\n前回のあなたの予想: {prediction or 'なし（初回）'}"
    return head


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def messages(var: str, frames_b64: list[str], prediction: str | None) -> list[dict]:
    content = [{"type": "text", "text": user_text(len(frames_b64), var, prediction)}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames_b64]
    return [{"role": "system", "content": system_for(var)}, {"role": "user", "content": content}]


PRED = re.compile(r"^\s*prediction\s*[:：]\s*(.+?)\s*$", re.I | re.M)


async def ask(vl, msgs) -> dict:
    from eve.vlm.narrator import _content_of
    from eve.vlm.parser import parse_vision

    t0 = time.monotonic()
    err, raw = None, ""
    for attempt in range(3):
        try:
            raw = _content_of(await vl.registry.complete("vlm_leaf", msgs))
            err = None
            break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
            await asyncio.sleep(3 * (attempt + 1))
    r = parse_vision(raw)
    m = PRED.search(raw or "")
    return {"raw": raw, "narration": r.narration, "notable": r.notable, "surprise": r.surprise_diff,
            "visible": r.visible, "prediction": m.group(1) if m else None,
            "sec": round(time.monotonic() - t0, 2), "error": err}


# ── narr: 10場面 ──────────────────────────────────────────────────────────────
async def narr(args) -> int:
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    gt = json.loads((PREV / "ground_truth.json").read_text(encoding="utf-8"))
    out_path = OUT / "vlm_narr.json"
    res = load(out_path) if (out_path.exists() and not args.fresh) else {}
    jobs = []
    for mid in sorted(k for k in gt if k.startswith("M")):
        frames = [b64(PREV / "moments" / mid / f"f{i}.jpg") for i in range(3)]
        for var in ("0", "U"):
            for rep in range(args.reps):
                key = f"{mid}|{var}|{rep}"
                if key not in res or res[key].get("error"):
                    jobs.append((key, mid, var, rep, frames))
    print(f"呼び出し {len(jobs)} 件（vlm_leaf = {vl.registry.resolve('vlm_leaf')}）")

    async def one(j):
        key, mid, var, rep, frames = j
        r = await ask(vl, messages(var, frames, None))
        res[key] = {"mid": mid, "variant": var, "rep": rep, **r}
        print(f"{key:10s} {r['sec']:4.1f}s ({len(r['narration'])}字) {r['narration'][:80]}")

    await gather_limited([one(j) for j in jobs], limit=4)
    dump(out_path, res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


# ── seq: 予想との食い違い ─────────────────────────────────────────────────────────
def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", size)


def _desktop():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1024, 576), (32, 78, 120))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 548, 1024, 576], fill=(22, 22, 28))  # タスクバー
    d.text((960, 553), "14:05", font=_font(13), fill=(230, 230, 230))
    return im, d


def _window(d, box, title, fill=(255, 255, 255)):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=fill, outline=(90, 90, 90))
    d.rectangle([x0, y0, x1, y0 + 26], fill=(235, 235, 235))
    d.text((x0 + 10, y0 + 5), title, font=_font(13), fill=(0, 0, 0))


DIARY = ["9月30日（火）", "朝は雨だったけど、昼から晴れた。", "駅前のパン屋で新作のメロンパンを買った。",
         "外はカリカリで中がふわふわ。", "午後は企画書の続き。", "図を3枚描いたところで集中が切れた。",
         "夜はカレーを作る予定。", "玉ねぎを飴色になるまで炒めたい。", "明日は8時に家を出る。"]


def notepad(n_lines: int, dialog: bool = False, cursor: bool = True):
    im, d = _desktop()
    _window(d, [120, 40, 900, 520], "日記.txt - メモ帳")
    for i, line in enumerate(DIARY[:n_lines]):
        d.text((140, 80 + i * 30), line, font=_font(18), fill=(20, 20, 20))
    if cursor:
        w = d.textlength(DIARY[max(0, n_lines - 1)], font=_font(18))
        y = 80 + (n_lines - 1) * 30
        d.line([140 + w + 2, y, 140 + w + 2, y + 22], fill=(0, 0, 0), width=2)
    if dialog:
        d.rectangle([300, 200, 740, 360], fill=(250, 250, 250), outline=(60, 60, 60), width=2)
        d.rectangle([300, 200, 740, 228], fill=(200, 40, 40))
        d.text((312, 205), "エラー", font=_font(14), fill=(255, 255, 255))
        d.text((320, 250), "ディスクの空き容量が不足しているため、", font=_font(16), fill=(0, 0, 0))
        d.text((320, 276), "ファイルを保存できませんでした。", font=_font(16), fill=(0, 0, 0))
        d.rectangle([620, 318, 720, 346], fill=(225, 225, 225), outline=(90, 90, 90))
        d.text((655, 322), "OK", font=_font(15), fill=(0, 0, 0))
    return im


ARTICLE = ["【特集】秋の行楽シーズン、各地の紅葉情報", "北海道の大雪山ではすでに見頃を迎え、", "山頂付近は赤や黄色に色づいている。",
           "東北地方では10月中旬から下旬が見頃の予想。", "関東の日光いろは坂は10月下旬から。", "京都の嵐山は11月中旬が見頃とみられる。",
           "今年は夏の猛暑の影響で、色づきが", "例年より1週間ほど遅れる地域もある。", "気象庁によると、朝晩の冷え込みが",
           "紅葉の進み具合を左右するという。", "お出かけの際は防寒具を忘れずに。", "（写真はイメージです）"]


def browser(scroll: int, other: bool = False):
    from PIL import ImageDraw
    im, d = _desktop()
    _window(d, [20, 20, 1004, 540], "紅葉情報 2026 - ニュース - Web ブラウザ" if not other else "ゲーム実況 まとめ - 動画サイト")
    d.rectangle([20, 46, 1004, 74], fill=(245, 245, 250))
    d.text((40, 51), "https://news.example.jp/autumn/leaves" if not other else "https://video.example.jp/watch?v=abc",
           font=_font(13), fill=(60, 60, 60))
    if other:
        d.rectangle([60, 90, 700, 450], fill=(10, 10, 10))
        d.polygon([(350, 230), (350, 310), (420, 270)], fill=(240, 240, 240))
        d.text((60, 460), "【初見】新作ホラーゲームを実況プレイ #1", font=_font(18), fill=(0, 0, 0))
        for i in range(4):
            d.rectangle([720, 90 + i * 90, 990, 170 + i * 90], fill=(200, 200, 210))
        return im
    for i, line in enumerate(ARTICLE[scroll:scroll + 12]):
        d.text((60, 95 + i * 36), line, font=_font(19 if (i == 0 and scroll == 0) else 17), fill=(20, 20, 20))
    d.rectangle([990, 80 + scroll * 25, 1000, 200 + scroll * 25], fill=(160, 160, 160))
    return im


def explorer(folder: str = "PC"):
    im, d = _desktop()
    _window(d, [80, 50, 940, 520], f"{folder} - エクスプローラー")
    items = (["ドキュメント", "ピクチャ", "ダウンロード", "ミュージック", "ビデオ", "ローカルディスク (C:)"] if folder == "PC"
             else ["2026-09-28_旅行", "2026-09-15_花火", "スクリーンショット", "壁紙", "IMG_0412.jpg", "IMG_0413.jpg"])
    for i, it in enumerate(items):
        x, y = 110 + (i % 3) * 260, 100 + (i // 3) * 150
        d.rectangle([x, y, x + 80, y + 64], fill=(240, 200, 90) if "." not in it else (120, 170, 220))
        d.text((x, y + 72), it, font=_font(14), fill=(0, 0, 0))
    return im


def jpeg(im) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=70)
    return buf.getvalue()


def sintel(t: float) -> bytes:
    """全画面で見ている想定（1024x576・16:9 に黒帯）。前回の評価と同じ作り方。"""
    from PIL import Image
    raw = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", str(VIDEO), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    src = Image.open(io.BytesIO(raw)).convert("RGB")
    canvas = Image.new("RGB", (1024, 576), (0, 0, 0))
    w = 1024
    h = int(src.height * w / src.width)
    canvas.paste(src.resize((w, h)), (0, (576 - h) // 2))
    return jpeg(canvas)


def _cuts() -> list[float]:
    return [float(x) for x in (PREV / "cuts.txt").read_text().split()]


def _shot_times(start: float, n_calls: int) -> list[list[float]]:
    """start から切り替わりを跨がない 3コマ×n_calls（0.5秒おき・呼び出しの間 1.5秒）。"""
    return [[start + c * 1.5 + i * 0.5 for i in range(3)] for c in range(n_calls)]


def build_sequences() -> dict:
    """連続の場面。各呼び出し＝3コマ。label は「前の呼び出しからの変化」（0=予想どおりの続き / 1=予想外 / None=採点しない）。"""
    cuts = _cuts()

    def inside(t0, t1):  # 区間内に切り替わりが無いか
        return not any(t0 <= c <= t1 for c in cuts)

    # 切り替わりを跨がない長めのショットを探す（連続3呼び出し＝約4.5秒）。映像は 188.8秒（cuts.txt の後半は範囲外）
    long_shots = [a for a, b in zip(cuts, cuts[1:]) if b - a > 6.0 and b < 185.0]
    s1, s2, s3 = long_shots[1], long_shots[2], long_shots[-1]  # 64.7 / 85.8 / 162.1（101〜156 は検出漏れの疑いで使わない）
    assert inside(s1 + 0.2, s1 + 4.8) and inside(s2 + 0.2, s2 + 4.8) and inside(s3 + 0.2, s3 + 3.3)
    seqs = {
        "A_作業→エラー": [("np", [1, 2, 2]), ("np", [3, 4, 4]), ("np", [5, 5, 6]), ("npdlg", [6, 6, 6]), ("np", [7, 7, 8])],
        "B_作業→動画": [("np", [2, 3, 3]), ("np", [4, 4, 5]), ("sintel", _shot_times(s1 + 0.3, 1)[0]),
                     ("sintel", _shot_times(s1 + 0.3, 2)[1]), ("sintel", _shot_times(s1 + 0.3, 3)[2])],
        "C_動画→エクスプローラー": [("sintel", _shot_times(s2 + 0.3, 1)[0]), ("sintel", _shot_times(s2 + 0.3, 2)[1]),
                            ("exp", ["PC"] * 3), ("exp", ["PC", "ピクチャ", "ピクチャ"])],
        "D_記事→動画サイト": [("br", [0, 0, 1]), ("br", [1, 2, 2]), ("br", [2, 3, 3]), ("brother", [0, 0, 0])],
        "E_エクスプローラー→動画": [("exp", ["PC", "PC", "ピクチャ"]), ("exp", ["ピクチャ"] * 3),
                           ("sintel", _shot_times(s3 + 0.3, 1)[0]), ("sintel", _shot_times(s3 + 0.3, 2)[1])],
    }
    labels = {
        "A_作業→エラー": [None, 0, 0, 1, None],
        "B_作業→動画": [None, 0, 1, 0, 0],
        "C_動画→エクスプローラー": [None, 0, 1, 0],
        "D_記事→動画サイト": [None, 0, 0, 1],
        "E_エクスプローラー→動画": [None, 0, 1, 0],
    }
    return {"seqs": seqs, "labels": labels}


def frames_cmd(args) -> int:
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    spec = build_sequences()
    index = {}
    for name, calls in spec["seqs"].items():
        index[name] = []
        for ci, (kind, params) in enumerate(calls):
            paths = []
            for fi, p in enumerate(params):
                if kind == "np":
                    data = jpeg(notepad(p))
                elif kind == "npdlg":
                    data = jpeg(notepad(p, dialog=True))
                elif kind == "br":
                    data = jpeg(browser(p))
                elif kind == "brother":
                    data = jpeg(browser(0, other=True))
                elif kind == "exp":
                    data = jpeg(explorer(p))
                else:
                    data = sintel(p)
                path = SEQ_DIR / f"{name}_{ci}_{fi}.jpg"
                path.write_bytes(data)
                paths.append(str(path))
            index[name].append({"kind": kind, "params": params, "frames": paths,
                                "label": spec["labels"][name][ci]})
    dump(OUT / "vlm_seq_index.json", index)
    print(f"コマ → {SEQ_DIR}（{sum(len(v) for v in index.values())} 呼び出し分）")
    return 0


async def seq(args) -> int:
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    index = load(OUT / "vlm_seq_index.json")
    out_path = OUT / "vlm_seq.json"
    res = load(out_path) if (out_path.exists() and not args.fresh) else {}

    async def run_seq(name, var, rep):
        key = f"{name}|{var}|{rep}"
        if key in res and not any(c.get("error") for c in res[key]):
            return
        pred = None
        calls = []
        for ci, call in enumerate(index[name]):
            frames = [b64(Path(p)) for p in call["frames"]]
            r = await ask(vl, messages(var, frames, pred))
            calls.append({"call": ci, "label": call["label"], "given_prediction": pred, **r})
            if var == "U":
                pred = r.get("prediction") or pred
            print(f"{key:28s} #{ci} label={call['label']} surprise={r['surprise']} notable={r['notable']} "
                  f"予想={str(r.get('prediction'))[:40]} ／ {r['narration'][:50]}")
        res[key] = calls

    jobs = [run_seq(name, var, rep) for name in index for var in ("0", "U") for rep in range(args.reps)]
    await gather_limited(jobs, limit=4)
    dump(out_path, res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "narr", "seq"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    if args.cmd == "frames":
        return frames_cmd(args)
    return asyncio.run(narr(args) if args.cmd == "narr" else seq(args))


if __name__ == "__main__":
    raise SystemExit(main())
