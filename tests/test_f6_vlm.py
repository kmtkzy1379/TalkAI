"""F6 VLM 画面認識の Tier-1 決定論テスト（mss/GPU/API/実スレッド 不使用）。

注入: フレーム列の fake capture / fake narrate_fn(Event gate 可) / fake clock / stub bridge。
実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f6_vlm.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.vlm import ChangeDetector, parse_vision  # noqa: E402
from eve.vlm.change_detector import hamming  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


# ===== パーサ =====
def t_parse_valid() -> bool:
    r = parse_vision("narration: メモ帳に文章を入力中\nnotable: yes\nsurprise: 45")
    return r.visible and "メモ帳" in r.narration and r.notable and r.surprise_diff == 45


def t_parse_garbage_safe() -> bool:
    # 完全なゴミでも raise せず安全既定（visible=True・空）
    r = parse_vision("@@@##$$ 壊れた出力 \n\n {")
    return r.visible and r.narration == "" and r.surprise_diff is None and not r.notable


def t_parse_fullwidth_colon() -> bool:
    r = parse_vision("ナレーション：ブラウザでニュースを閲覧\n驚き：30")
    return "ブラウザ" in r.narration and r.surprise_diff == 30


def t_parse_invisible_marker() -> bool:
    # A11: 本文に視認不可マーカ → 中身は採用せず surprise も上げない
    r = parse_vision("narration: 黒い画面で視認不可\nsurprise: 90")
    return (not r.visible) and r.narration == "" and r.surprise_diff is None and not r.notable


def t_parse_visible_false_tag() -> bool:
    r = parse_vision("visible: no\nnarration: 何か映ってるかも")
    return (not r.visible) and r.narration == ""


def t_parse_none() -> bool:
    r = parse_vision(None)
    return r.visible and r.is_empty()


# ===== 変化ゲート =====
def t_gate_first_always() -> bool:
    cd = ChangeDetector()
    return cd.evaluate(0xFFFF) is True  # 初回は必ず True


def t_gate_identical_none() -> bool:
    cd = ChangeDetector()
    cd.evaluate(0xABCD)  # 参照確立
    return cd.evaluate(0xABCD) is False  # 同一 → 変化なし


def t_gate_far_changed() -> bool:
    cd = ChangeDetector(phash_threshold=12)
    cd.evaluate(0x0000000000000000)
    # 多数ビットが立つ → hamming 大 → True
    return cd.evaluate(0xFFFFFFFFFFFFFFFF) is True


def t_gate_small_change_below_threshold() -> bool:
    cd = ChangeDetector(phash_threshold=12)
    cd.evaluate(0x0)
    # 3 ビットだけ変化（< 12）→ False
    return cd.evaluate(0b111) is False


def t_gate_periodic_forced() -> bool:
    cd = ChangeDetector(phash_threshold=12, periodic_frames=3)
    cd.evaluate(0x0)  # 参照
    r1 = cd.evaluate(0x0)  # idle 1 → False
    r2 = cd.evaluate(0x0)  # idle 2 → False
    r3 = cd.evaluate(0x0)  # idle 3 → periodic 強制 True
    return r1 is False and r2 is False and r3 is True


def t_hamming() -> bool:
    return hamming(0b1010, 0b0001) == 3


def main() -> None:
    check("parse valid", t_parse_valid())
    check("parse garbage 安全(raise しない)", t_parse_garbage_safe())
    check("parse 全角コロン", t_parse_fullwidth_colon())
    check("A11 parse 視認不可マーカ→中身/驚き不採用", t_parse_invisible_marker())
    check("A11 parse visible:no→不可視", t_parse_visible_false_tag())
    check("parse None 安全", t_parse_none())
    check("gate 初回は必ず True", t_gate_first_always())
    check("gate 同一→False", t_gate_identical_none())
    check("gate 大変化→True", t_gate_far_changed())
    check("gate 微小変化(<閾値)→False", t_gate_small_change_below_threshold())
    check("gate periodic 強制", t_gate_periodic_forced())
    check("hamming 距離", t_hamming())


if __name__ == "__main__":
    main()
    print(f"\n合計: PASS {_passed} / FAIL {_failed}")
    sys.exit(1 if _failed else 0)
