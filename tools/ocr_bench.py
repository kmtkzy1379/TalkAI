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

    async def _read(self, png: bytes) -> str:
        stream = self._Stream()
        w = self._DataWriter(stream.get_output_stream_at(0))
        w.write_bytes(png)
        await w.store_async()
        dec = await self._BitmapDecoder.create_async(stream)
        bmp = await dec.get_software_bitmap_async()
        res = await self._eng.recognize_async(bmp)
        return "\n".join(l.text for l in res.lines)

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
def grab_png(downscale: int = 1600) -> bytes:
    """全画面を PNG bytes で。VLM 経路と同じ mss を使う（別物を測らない）。"""
    import io

    import mss
    from PIL import Image

    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[0])
    im = Image.frombytes("RGB", raw.size, raw.rgb)
    if downscale and im.width > downscale:
        h = int(im.height * downscale / im.width)
        im = im.resize((downscale, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="win,rapid")
    ap.add_argument("--shots", type=int, default=12, help="連続キャプチャ枚数")
    ap.add_argument("--interval", type=float, default=1.0, help="キャプチャ間隔(秒)")
    ap.add_argument("--downscale", type=int, default=1600)
    args = ap.parse_args()

    print(f"■ 連続 {args.shots} 枚を {args.interval}s 間隔でキャプチャ（この間 画面は触らないこと）")
    shots = []
    for i in range(args.shots):
        shots.append(grab_png(args.downscale))
        print(f"  {i + 1}/{args.shots}", end="\r")
        if i < args.shots - 1:
            time.sleep(args.interval)
    print(f"\n  取得完了（1枚 {len(shots[0]) / 1024:.0f}KB）\n")

    for key in [e.strip() for e in args.engines.split(",") if e.strip()]:
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

        # (b) 静止時ノイズ床: 画面が変わっていない連続フレーム間の差分率
        noise = [diff_ratio(texts[i], texts[i + 1]) for i in range(len(texts) - 1)]
        chars = [len(norm(t)) for t in texts]

        print(f"  言語     : {getattr(eng, 'lang', '?')}")
        print(f"  速度     : 中央値 {st.median(times):.0f}ms / 最大 {max(times):.0f}ms")
        print(f"  文字数   : 中央値 {st.median(chars):.0f} / 最小 {min(chars)} / 最大 {max(chars)}")
        print(f"  決定性   : {'同一画像→同一テキスト' if deterministic else '★同一画像でも揺れる（採用不可）'}")
        print(f"  ノイズ床 : 中央値 {st.median(noise) * 100:.2f}% / 最大 {max(noise) * 100:.2f}%")
        ok = deterministic and max(noise) < 0.05
        print(f"  判定     : {'○ トリガに使える（閾値は最大ノイズの2倍以上に置く）' if ok else '× 静止時に揺れすぎ＝偽トリガ源'}")
        if noise and max(noise) >= 0.05:
            worst = noise.index(max(noise))
            print(f"    最悪の隣接ペア {worst}->{worst + 1}:")
            print(f"      前: {norm(texts[worst])[:110]}")
            print(f"      後: {norm(texts[worst + 1])[:110]}")
        print()

    print("※ 検出力（実際に文字が変わった時の差分率）は、途中で画面の文字を変えて再実行し")
    print("   ノイズ床と比較すること。両者が重なるなら、その engine では閾値を引けない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
