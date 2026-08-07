r"""文字ゲートの実機E2E — 本番の CaptureThread / ChangeDetector / ScreenTextGate を通す。

決定論テスト（`tests/test_f6_vlm.py`）は fake OCR で分岐を固定するが、**本物の画面を本物の
OCR で読んだとき pHash と文字ゲートがそれぞれ意図どおり動くか**は実機でしか分からない。
ここで検証するのは4点:

1. 文字だけを大きく変えた時 → **pHash は反応せず**、文字ゲートだけが起こす（この案の存在理由）
2. 文字を少しだけ変えた時 → どちらも起こさない（文章作成中に鳴り続けない＝鈍さの検証）
3. 窓を閉じるような大きな見た目の変化 → **pHash が起こし、文字ゲートは見られない**（pHash 優先）
4. OCR の読み取りが VLM へ渡る経路に**一切載っていない**こと

実 API は使わない（narrate_fn をスタブにして「起こされたか」だけ見る）＝課金なし。

実行:
  $env:PYTHONIOENCODING="utf-8"
  .\.venv\Scripts\python.exe tools\text_gate_e2e.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.vlm.capture import ScreenCapture  # noqa: E402
from eve.vlm.change_detector import ChangeDetector  # noqa: E402
from eve.vlm.text_gate import ScreenTextGate, WindowsOcr, normalize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_bench import _LINES, StaticWindow  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, note: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}" + (f"  — {note}" if note else ""))
    else:
        _failed += 1
        print(f"FAIL {name}" + (f"  — {note}" if note else ""))


class Rig:
    """本番と同じ部品を組む（CaptureThread の _step をこちらから1回ずつ回す）。"""

    def __init__(self, diff_pct: float = 10.0) -> None:
        self.cap = ScreenCapture(monitor=1, downscale_max=1024, jpeg_quality=70)
        self.det = ChangeDetector(phash_threshold=12, periodic_frames=0)
        self.ocr = WindowsOcr()
        self.gate = ScreenTextGate(
            ocr=self.ocr, grab_png=lambda: self.cap.grab_png(1600),
            diff_threshold_pct=diff_pct, interval_sec=0.0,  # E2E では待たない
        )
        self.gate_checks = 0
        self.gate_resets = 0
        _mc, _rs = self.gate.maybe_changed, self.gate.reset

        def _counted_mc():
            self.gate_checks += 1
            return _mc()

        def _counted_reset():
            self.gate_resets += 1
            return _rs()

        self.gate.maybe_changed = _counted_mc      # type: ignore[method-assign]
        self.gate.reset = _counted_reset           # type: ignore[method-assign]

    def step(self):
        """`CaptureThread._step` と同じ順序（pHash 優先）を1回。→ (phash判定, 最終判定)"""
        _frame, phash = self.cap.capture_one()
        phash_changed = self.det.evaluate(phash)
        changed = phash_changed
        if phash_changed:
            self.gate.reset()
        elif self.gate.maybe_changed():
            changed = True
        return phash_changed, changed

    def settle(self, n: int = 4):
        """画面が落ち着くまで空回し（参照 pHash と文字の基準を作る）。"""
        for _ in range(n):
            self.step()
            time.sleep(0.3)

    def close(self):
        self.cap.close()
        self.ocr.close()


def main() -> int:
    rig = Rig()
    if not rig.ocr.available:
        print("SKIP: OCR を初期化できない（pip install winsdk / OCR 言語パック ja を確認）")
        return 0
    print(f"OCR: lang={rig.ocr.lang}")
    win = StaticWindow()
    try:
        win.set_lines(_LINES)
        time.sleep(1.2)
        rig.settle()

        # --- 1. 文字だけ大きく変える → pHash は反応せず文字ゲートが起こす -------------
        big = [f"総入れ替えされた行 {i} — この画面の文字は全部別物になった" for i in range(len(_LINES))]
        win.set_lines(big)
        time.sleep(1.0)
        ph, ch = rig.step()
        check("① 文字の大変化で起こす", ch is True, f"pHash={ph} 最終={ch}")
        check("① その時 pHash は無反応（この案の存在理由）", ph is False,
              "pHash が拾えるなら文字ゲートは不要だった" if ph else "pHash 一致のまま文字だけ変わった")

        rig.settle(2)

        # --- 2. 文字を少しだけ変える → どちらも起こさない（鈍さ）---------------------
        small = list(big)
        small[0] = small[0].replace("総入れ替え", "総入替")
        win.set_lines(small)
        time.sleep(1.0)
        ph2, ch2 = rig.step()
        check("② 小さな文字変化では起こさない（文章作成中に鳴らない）", ch2 is False,
              f"pHash={ph2} 最終={ch2}")

        rig.settle(2)

        # --- 3. 見た目の大変化 → pHash 優先（文字ゲートは見られない）-----------------
        checks_before, resets_before = rig.gate_checks, rig.gate_resets
        win.root.geometry("640x400+500+300")   # 窓を大きく動かす＝pHash が動く変化
        win.root.update()
        time.sleep(1.2)
        ph3, ch3 = rig.step()
        used_gate = rig.gate_checks > checks_before
        did_reset = rig.gate_resets > resets_before
        check("③ 見た目の大変化は pHash が起こす", ph3 is True and ch3 is True, f"pHash={ph3}")
        check("③ ★pHash 優先: その回は文字ゲートを見ない", (not used_gate) if ph3 else True,
              f"文字ゲート参照={used_gate}")
        check("③ ★pHash が起こしたら文字ゲートは reset される（追い撃ち防止）",
              did_reset if ph3 else True, f"reset={did_reset}")

        # --- 4. OCR テキストが外に漏れていないこと ----------------------------------
        # ゲートの内部にしか無い（public な属性・戻り値からは取れない）ことを構造で確認。
        public = [a for a in dir(rig.gate) if not a.startswith("_")]
        leaks = [a for a in public if "text" in a.lower()]
        sample = normalize(rig.ocr.read(rig.cap.grab_png(1600)))
        check("④ OCR テキストは public に露出しない（LLM 経路に載らない）",
              not leaks, f"public={public}")
        check("④ 実際に読めている（判定が空振りしていない）", len(sample) > 100,
              f"{len(sample)}文字")
    finally:
        win.close()
        rig.close()

    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
