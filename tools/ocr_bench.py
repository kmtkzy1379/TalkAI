r"""高速文字認識ライブラリのベンチ（画面の**文字変化**を検出するトリガに使えるか判定する）。

## なぜ要るか
pHash（`eve/vlm/change_detector.py`）はウィンドウの移動・切替のような**見た目の変化**は拾うが、
同じ枠の中で文字だけが変わる変化には弱い。実測（2026-08-06）: 窓の中身を全文差し替えても
hamming 0＝閾値12 には遠く及ばず「変化なし」と判定された。かといって VLM を常時回すのは
コストと遅延で不可能（`vlm_leaf` は1回 数百ms〜秒・課金あり）。

そこで **高速OCR で文字を読み、前回との差分が N% を超えたら VLM を起こす** 案を測る。

## この案の生死を決めるのは「速度」ではなく「安定性」
同じ画面を2回読んで違うテキストが返るなら、その揺れがそのまま偽の「変化」になり、
VLM を無駄に叩き続ける（＝コストも遅延も悪化して逆効果）。よって測るのは:

1. **静止時ノイズ床**: 画面が変わっていない連続フレームで観測される差分率。
   これが**閾値の下限**を決める。ここが高いライブラリは**採用不可**（速くても無意味）。
2. **検出力**: 実際に文字が変わった時の差分率。ノイズ床と明確に離れていること。
3. 速度: 1フレームあたりの ms（2fps の capture を邪魔しないこと）。

## 絶対の制約
**OCR の読み取りテキストは LLM に一切渡さない**（誤読がそのままハルシネーションになる）。
用途は「VLM を起こすかどうか」の boolean だけ。この境界は実装でも守ること。

実行:
  $env:PYTHONIOENCODING="utf-8"
  python tools\ocr_bench.py --engines win,rapid --shots 12 --interval 1.0
"""
from __future__ import annotations

import argparse
import difflib
import re
import statistics as st
import time
import unicodedata

# --- 正規化（比較の土俵をそろえる）------------------------------------------
# OCR は空白・改行・記号の扱いが不安定なので、比較前に潰す。ここを甘くすると
# ノイズ床が上がり、閾値が使い物にならなくなる。
_WS = re.compile(r"\s+")
_NOISE = re.compile(r"[|｜/\\_\-—–~〜`'\"“”‘’·・:：;；,、.。!！?？()（）\[\]【】{}<>＜＞]+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _NOISE.sub("", s)
    return _WS.sub("", s)


def diff_ratio(a: str, b: str) -> float:
    """0.0=同一 / 1.0=完全に別物。difflib の類似度の裏返し（順序も見る）。"""
    a, b = norm(a), norm(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


# --- エンジン ---------------------------------------------------------------
class WinOcr:
    """Windows 組込み OCR（Windows.Media.Ocr）。モデル配布不要・OS 同梱。

    要 `pip install winsdk`。日本語は OCR 言語パックが要る（未導入なら英語のみ）。
    """

    name = "win"

    def __init__(self) -> None:
        import asyncio

        from winsdk.windows.globalization import Language
        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        self._asyncio = asyncio
        self._BitmapDecoder = BitmapDecoder
        self._DataWriter = DataWriter
        self._Stream = InMemoryRandomAccessStream
        eng = None
        for tag in ("ja", "ja-JP", "en-US"):
            try:
                eng = OcrEngine.try_create_from_language(Language(tag))
            except Exception:
                eng = None
            if eng is not None:
                self.lang = tag
                break
        if eng is None:
            eng = OcrEngine.try_create_from_user_profile_languages()
            self.lang = "user-profile"
        if eng is None:
            raise RuntimeError("OcrEngine を作れない（OCR 言語パック未導入の可能性）")
        self._eng = eng

    @staticmethod
    def _seq(v):
        """WinRT の IVectorView は Python の iter プロトコルに乗らないことがある
        （実測: TypeError: iter() returned non-iterator）。添字アクセスに倒す。"""
        try:
            return list(v)
        except TypeError:
            return [v.get_at(i) for i in range(v.size)]

    async def _read(self, png: bytes) -> str:
        stream = self._Stream()
        w = self._DataWriter(stream.get_output_stream_at(0))
        w.write_bytes(png)
        await w.store_async()
        await w.flush_async()
        w.detach_stream()
        stream.seek(0)
        dec = await self._BitmapDecoder.create_async(stream)
        bmp = await dec.get_software_bitmap_async()
        res = await self._eng.recognize_async(bmp)
        return "\n".join(l.text for l in self._seq(res.lines))

    def read(self, png: bytes) -> str:
        return self._asyncio.run(self._read(png))


class RapidOcr:
    """RapidOCR（onnxruntime 版 PaddleOCR）。`pip install rapidocr-onnxruntime`。

    既に onnxruntime が入っているので追加は軽い。日本語モデル同梱。
    """

    name = "rapid"

    def __init__(self) -> None:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        self._np = np
        self._ocr = RapidOCR()
        self.lang = "bundled"

    def read(self, png: bytes) -> str:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        res, _ = self._ocr(img)
        return "\n".join(r[1] for r in (res or []))


ENGINES = {"win": WinOcr, "rapid": RapidOcr}


# --- キャプチャ -------------------------------------------------------------
def grab_png(downscale: int = 1600):
    """全画面を (PNG bytes, phash) で。VLM 経路と同じ mss/imagehash を使う（別物を測らない）。"""
    import io

    import imagehash
    import mss
    from PIL import Image

    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[0])
    im = Image.frombytes("RGB", raw.size, raw.rgb)
    ph = int(str(imagehash.phash(im)), 16)  # `eve/vlm/capture.py:62` と同一の作り方
    if downscale and im.width > downscale:
        h = int(im.height * downscale / im.width)
        im = im.resize((downscale, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), ph


# --- 制御下の静止ウィンドウ（真のノイズ床と検出力を同時に測る）------------------
# デスクトップをそのまま撮ると、時計・ターミナル・カーソル点滅が常に動いていて
# 「静止」が作れない（実測 2026-08-07: 画素完全一致ペアが 0/15）。
# 自分で中身を完全に制御した窓を出せば、(a) 一切変えない=ノイズ床、
# (b) N行だけ変える=検出力、を同じ経路（mss→OCR）で測れる。CI にも載せられる。
_LINES = [
    "第1章 タスク管理の設計について",
    "予約タスクは同時に一件だけ受け付ける",
    "The quick brown fox jumps over the lazy dog",
    "画面認識は毎秒二回のキャプチャで動作する",
    "surprise = 0.42  threshold = 0.75  count = 128",
    "エラーが出たら自分で検索して解決策を探す",
    "C:\\eve_ws\\workspace\\primes.txt を表示中",
    "応答レイテンシの中央値は三点一四秒だった",
    "OCR text must never be passed to the LLM",
    "取り消しの言い回しはそのまま渡してよい",
    "合計 477 件のテストが二回連続で成功した",
    "ウィンドウを閉じる能力を追加で実装した",
]


class StaticWindow:
    """中身を完全に制御した Tk 窓。`set_lines` で任意行だけ差し替えられる。"""

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("ocr_bench")
        self.root.geometry("1280x820+40+40")
        self.root.configure(bg="white")
        self.text = tk.Text(self.root, font=("Meiryo", 16), bg="white", fg="black",
                            bd=0, highlightthickness=0, insertwidth=0, wrap="none")
        self.text.pack(fill="both", expand=True, padx=24, pady=24)
        self.root.update()

    def set_lines(self, lines) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n\n".join(lines))
        self.text.configure(state="disabled")  # カーソル点滅を消す＝静止を作る
        self.root.update()
        self.root.update_idletasks()

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


def run_controlled(engines, downscale: int, settle: float = 1.2) -> int:
    """静止窓で「真のノイズ床」と「検出力」を測り、閾値が引けるか判定する。"""
    win = StaticWindow()
    try:
        win.set_lines(_LINES)
        time.sleep(settle)
        base = [grab_png(downscale)[0] for _ in range(6)]
        ident = sum(1 for i in range(len(base) - 1) if base[i] == base[i + 1])
        print(f"■ 静止窓: 6枚中 隣接ペア {ident}/{len(base) - 1} が画素完全一致\n")

        variants = {}
        for label, n in (("1行変更", 1), ("3行変更", 3), ("6行変更", 6)):
            mod = list(_LINES)
            for i in range(n):
                mod[i] = f"変更後の行 {i} — modified line {i} でテキストが入れ替わった"
            win.set_lines(mod)
            time.sleep(settle)
            variants[label] = grab_png(downscale)[0]
        win.set_lines(_LINES)  # 元に戻す

        for key in engines:
            cls = ENGINES.get(key)
            if cls is None:
                continue
            print(f"── {key} ──")
            try:
                eng = cls()
                texts = [eng.read(p) for p in base]
                vtexts = {k: eng.read(v) for k, v in variants.items()}
            except Exception as e:
                print(f"  使えない: {type(e).__name__}: {e}\n")
                continue
            noise = [diff_ratio(texts[i], texts[i + 1]) for i in range(len(texts) - 1)]
            nmax = max(noise) if noise else 0.0
            print(f"  読取文字数: {len(norm(texts[0]))}")
            print(f"  真ノイズ床: 中央値 {st.median(noise) * 100:.2f}% / 最大 {nmax * 100:.2f}%")
            det = []
            for k, t in vtexts.items():
                d = diff_ratio(texts[-1], t)
                det.append(d)
                print(f"  検出力 {k}: {d * 100:.2f}%")
            dmin = min(det) if det else 0.0
            sep = dmin > nmax * 2 and nmax < 0.03
            print(f"  判定     : {'○ 閾値を引ける' if sep else '× ノイズと検出が近すぎる'}"
                  f"（ノイズ最大 {nmax * 100:.2f}% / 最小検出 {dmin * 100:.2f}%）")
            if sep:
                print(f"  推奨閾値  : {max(nmax * 2, 0.01) * 100:.1f}%"
                      "（最大ノイズの2倍・1行の変化は拾える）")
            print()
    finally:
        win.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="win,rapid")
    ap.add_argument("--shots", type=int, default=12, help="連続キャプチャ枚数")
    ap.add_argument("--interval", type=float, default=1.0, help="キャプチャ間隔(秒)")
    ap.add_argument("--downscale", type=int, default=1600)
    ap.add_argument("--controlled", action="store_true",
                    help="制御下の静止窓で 真ノイズ床/検出力 を測る（推奨・CI 向き）")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if args.controlled:
        return run_controlled(engines, args.downscale)

    print(f"■ 連続 {args.shots} 枚を {args.interval}s 間隔でキャプチャ（この間 画面は触らないこと）")
    shots, phashes = [], []
    for i in range(args.shots):
        png, ph = grab_png(args.downscale)
        shots.append(png)
        phashes.append(ph)
        print(f"  {i + 1}/{args.shots}", end="\r")
        if i < args.shots - 1:
            time.sleep(args.interval)
    # **pHash が同一の隣接ペアだけ**を「静止」とみなす。これをやらないと、裏で動いている
    # ターミナルや時計の更新が OCR のノイズとして混入し、ノイズ床を大きく見誤る
    # （実測 2026-08-07: 混入込みで最大 20.6% → 実際の OCR 揺れはこれよりずっと小さい）。
    static = [i for i in range(len(shots) - 1) if phashes[i] == phashes[i + 1]]
    moved = len(shots) - 1 - len(static)
    # **真に静止**＝PNG バイト列まで一致（可逆圧縮なので画素が1つも変わっていない）。
    # OCR の純粋なノイズ床はここでしか測れない。pHash 一致は「見た目が似ている」でしかなく、
    # 実測 2026-08-07: pHash 一致の13ペア全部で文字は変わっていた（＝ライブ更新中のタイマー）。
    identical = [i for i in range(len(shots) - 1) if shots[i] == shots[i + 1]]
    print(f"\n  取得完了（1枚 {len(shots[0]) / 1024:.0f}KB）")
    print(f"  pHash 一致 {len(static)} / pHash 変化 {moved} ペア")
    print(f"  **画素まで完全一致** {len(identical)} ペア（真の静止・ノイズ床はここで測る）")
    print(f"  → pHash 一致なのに画素が違う: {len(static) - len(identical)} ペア"
          f"（＝pHash が見逃した変化。この案が要る理由そのもの）\n")
    if not static:
        print("  ★ 静止ペアがゼロ。画面が動き続けているのでノイズ床を測れない。")
        print("    メモ帳など静止した画面を最前面にして再実行すること。\n")

    for key in engines:
        cls = ENGINES.get(key)
        if cls is None:
            print(f"× 未知のエンジン {key}")
            continue
        print(f"── {key} ──")
        try:
            eng = cls()
        except Exception as e:
            print(f"  使えない: {type(e).__name__}: {e}\n")
            continue
        texts, times = [], []
        try:
            for png in shots:
                t0 = time.perf_counter()
                texts.append(eng.read(png))
                times.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  読取で失敗: {type(e).__name__}: {e}\n")
            continue

        # (a) 決定性: 同一画像を2回読んで同じか（エンジン内部の非決定性を切り分ける）
        rerun = eng.read(shots[0])
        deterministic = norm(rerun) == norm(texts[0])

        # (b) 静止時ノイズ床: **pHash が一致した**隣接ペアだけで測る（実変化を混ぜない）
        noise = [diff_ratio(texts[i], texts[i + 1]) for i in static]
        changed = [diff_ratio(texts[i], texts[i + 1])
                   for i in range(len(texts) - 1) if i not in static]
        chars = [len(norm(t)) for t in texts]

        print(f"  言語     : {getattr(eng, 'lang', '?')}")
        print(f"  速度     : 中央値 {st.median(times):.0f}ms / 最大 {max(times):.0f}ms")
        print(f"  文字数   : 中央値 {st.median(chars):.0f} / 最小 {min(chars)} / 最大 {max(chars)}")
        print(f"  決定性   : {'同一画像→同一テキスト' if deterministic else '★同一画像でも揺れる（採用不可）'}")
        pure = [diff_ratio(texts[i], texts[i + 1]) for i in identical]
        if pure:
            print(f"  真ノイズ床: 中央値 {st.median(pure) * 100:.2f}% / 最大 {max(pure) * 100:.2f}%"
                  f"（画素完全一致 {len(pure)} ペア）")
        else:
            print("  真ノイズ床: 測定不能（画素完全一致のペアなし＝画面が動き続けている）")
        if noise:
            print(f"  pHash一致 : 中央値 {st.median(noise) * 100:.2f}% / 最大 {max(noise) * 100:.2f}%"
                  f"（{len(noise)} ペア・**pHash が見逃した文字変化を含む**）")
        else:
            print("  pHash一致 : 測定不能（静止ペアなし）")
        if changed:
            print(f"  変化時   : 中央値 {st.median(changed) * 100:.2f}% / 最小 {min(changed) * 100:.2f}%"
                  f"（pHash が動いた {len(changed)} ペア）")
        ok = deterministic and noise and max(noise) < 0.05
        print(f"  判定     : {'○ トリガに使える' if ok else '× 静止時に揺れすぎ＝偽トリガ源' if noise else '- 保留'}")
        if noise and changed:
            sep = min(changed) > max(noise)
            print(f"  分離     : {'○ 変化の最小 > ノイズの最大＝閾値を引ける' if sep else '× 重なる＝閾値を引けない'}"
                  f"（ノイズ最大 {max(noise) * 100:.2f}% vs 変化最小 {min(changed) * 100:.2f}%）")
        if noise and max(noise) >= 0.05:
            worst = static[noise.index(max(noise))]
            print(f"    最悪の静止ペア {worst}->{worst + 1}（pHash 同一なのに差が出た＝純粋な誤読）:")
            a, b = norm(texts[worst]), norm(texts[worst + 1])
            sm = difflib.SequenceMatcher(None, a, b)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"      {tag}: {a[i1:i2][:60]!r} -> {b[j1:j2][:60]!r}")
        print()

    print("※ 検出力（実際に文字が変わった時の差分率）は、途中で画面の文字を変えて再実行し")
    print("   ノイズ床と比較すること。両者が重なるなら、その engine では閾値を引けない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
