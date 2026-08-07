"""ScreenTextGate — 画面の**文字だけ**が変わった時に VLM を起こす二段目のゲート。

## なぜ要るか（実測 2026-08-07 `tools/ocr_bench.py`）
pHash は枠内の文字変化を拾えない。連続16枚で pHash 一致 15/15 ペア、しかし画素完全一致は
**0/15** で、OCR テキストは最大 24% 変化していた。メモ帳やブラウザの中身が入れ替わっても
`ChangeDetector` は「変化なし」を返し続け、VLM が起きない。

## なぜ OCR の読み取りを LLM に渡さないか（絶対の制約）
高速OCR は必ず誤読する（実測で `実行`→`夷行`、`5`の脱落 等）。それを LLM に渡すと誤読が
そのままハルシネーションの材料になる。**このモジュールが外に出すのは bool だけ**で、
読み取ったテキストはインスタンスの中に留まり、差分計算にしか使わない。

## なぜ鈍くするか（ユーザ裁定 2026-08-08）
文章作成中は何十分も文字が増減し続ける。1文字ごとに反応すると「いちいち声をかけてくる」
状態になり実用にならない。よって既定閾値は **10%**（実測: 12行中3行の入れ替えが約10.5%）。
pHash（大きな見た目の変化でしか動かない）と合わせて、両方とも鈍い側に倒す。

## pHash 優先（ユーザ裁定 2026-08-08）
文字がぎっしり詰まったタブを閉じると pHash も OCR も同時に反応する。二重に VLM を起こさない
よう、**pHash が変化と言ったらそこで確定**し、このゲートは見ない（`capture_thread.py`）。
その際 `reset()` で基準を取り直す＝直後に文字ゲートが追い撃ちしないようにする。

## スレッド安全性
`vlm-capture` スレッドからだけ呼ばれる（`PIPELINE_DESIGN.md §9.3`）。loop 所有状態には
一切触れず、戻り値は bool のみ。内部で asyncio ループを1本持つが、これはこのスレッド専用で
本体の loop とは無関係。
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
import unicodedata
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_DIFF_PCT = 10.0     # 何%変われば「文字が変わった」とみなすか
DEFAULT_INTERVAL_SEC = 3.0  # OCR を回す最短間隔（実測 302ms/回なので duty ≈10%）
MIN_CHARS = 40              # これ未満しか読めない画面（動画/ゲーム等）は判定しない

# 比較の土俵をそろえる。ここを甘くするとノイズ床が上がり閾値が使い物にならない。
_WS = re.compile(r"\s+")
_NOISE = re.compile(r"[|｜/\\_\-—–~〜`'\"“”‘’·・:：;；,、.。!！?？()（）\[\]【】{}<>＜＞]+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return _WS.sub("", _NOISE.sub("", s))


def diff_pct(a: str, b: str) -> float:
    """0.0=同一 / 100.0=完全に別物。`tools/ocr_bench.py` と同一の尺度。"""
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 100.0
    return (1.0 - difflib.SequenceMatcher(None, a, b).ratio()) * 100.0


class WindowsOcr:
    """Windows 組込み OCR（Windows.Media.Ocr）。OS 同梱でモデル配布が要らない。

    実測（`tools/ocr_bench.py --controlled`・2026-08-07）:
      速度 302ms / 真ノイズ床 最大0.23% / 1行変更で2.89% ＝ 分離比 12.6倍。
      RapidOCR は 1800ms・ノイズ床 2.37% で、速度も安定性も劣ったため不採用。

    生成に失敗したら `available=False` になり、呼び出し側は文字ゲートごと無効化する
    （画面認識自体は pHash だけで従来どおり動く＝fail-soft）。
    """

    def __init__(self) -> None:
        self.available = False
        self.lang = ""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            from winsdk.windows.globalization import Language
            from winsdk.windows.graphics.imaging import BitmapDecoder
            from winsdk.windows.media.ocr import OcrEngine
            from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
        except Exception as e:
            logger.info("文字ゲート無効: winsdk を import できない（%s）", type(e).__name__)
            return
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
            try:
                eng = OcrEngine.try_create_from_user_profile_languages()
                self.lang = "user-profile"
            except Exception:
                eng = None
        if eng is None:
            logger.info("文字ゲート無効: OcrEngine を作れない（OCR 言語パック未導入の可能性）")
            return
        self._eng = eng
        self._loop = asyncio.new_event_loop()  # このスレッド専用（本体 loop とは無関係）
        self.available = True

    @staticmethod
    def _seq(v):
        """WinRT の IVectorView は iter プロトコルに乗らないことがある（実測: TypeError）。"""
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
        """失敗は空文字（例外を上げない＝キャプチャスレッドを落とさない）。"""
        if not self.available or self._loop is None:
            return ""
        try:
            return self._loop.run_until_complete(self._read(png))
        except Exception:
            logger.debug("OCR 読取に失敗（この回は判定しない）", exc_info=True)
            return ""

    def close(self) -> None:
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None


class ScreenTextGate:
    """`maybe_changed()` が True なら「pHash は静かだが文字が大きく変わった」。"""

    def __init__(
        self,
        *,
        ocr,
        grab_png: Callable[[], bytes],
        diff_threshold_pct: float = DEFAULT_DIFF_PCT,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ocr = ocr
        self._grab = grab_png
        self._threshold = diff_threshold_pct
        self._interval = interval_sec
        self._now = now_fn
        self._last_text = ""      # **ここから外へ出さない**（LLM に渡さない）
        # None = まだ一度も測っていない（初回は間隔チェックを通す）。0.0 で初期化すると
        # 単調時計が 0 起点の環境／注入時計で**初回が必ず間隔に弾かれる**（テストが検出）。
        self._last_at: Optional[float] = None

    @property
    def available(self) -> bool:
        return bool(getattr(self._ocr, "available", False))

    def reset(self) -> None:
        """pHash が変化を検出した時に基準を取り直す（文字ゲートの追い撃ちを防ぐ）。"""
        self._last_text = ""
        self._last_at = self._now()

    def maybe_changed(self) -> bool:
        """間隔を守りつつ OCR し、閾値超えなら True。判定できない回は False。"""
        if not self.available:
            return False
        now = self._now()
        if self._last_at is not None and now - self._last_at < self._interval:
            return False
        self._last_at = now
        try:
            png = self._grab()
        except Exception:
            logger.debug("文字ゲート用のキャプチャに失敗（この回は判定しない）", exc_info=True)
            return False
        text = self._ocr.read(png)
        if len(normalize(text)) < MIN_CHARS:
            # 文字がほとんど無い画面（動画・ゲーム・真っ黒）。ここで基準を更新すると
            # 「文字なし→文字あり」で毎回誤爆するので、**基準を消して**次回に持ち越さない。
            self._last_text = ""
            return False
        if not self._last_text:
            self._last_text = text     # 初回は基準を作るだけ
            return False
        d = diff_pct(self._last_text, text)
        if d < self._threshold:
            return False
        logger.info("📝 画面の文字が変化（%.1f%% ≥ %.0f%%）→ 画面認識を起こす", d, self._threshold)
        self._last_text = text
        return True
