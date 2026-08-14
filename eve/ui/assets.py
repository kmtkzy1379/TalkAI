"""枠素材の下ごしらえ（**実際に絵がある範囲だけを切り出す**）。

## なぜ要るか（実測 2026-08-14）
`eve_ui/*.png` は 2800x2000 のキャンバスだが、**絵はその一部にしか入っていない**:

| 素材 | 絵の範囲 | 透明率 |
|---|---|---|
| `log_bvackground.png` | x 1446..2507 / y 104..1883（右寄り） | **68.8%** |
| `text_background.png` | x 200..2679 / y 266..1800 | 34.1% |

キャンバスごと引き伸ばすと、絵が枠の一部にしか来ず、**中の紙の縁が2本目の枠に見える**
（ユーザ指摘「背景が2重」）。切り出してから使えば絵と枠が一致する。

## 何を返すか
切り出した PNG をキャッシュに書き、そのパスと **9スライスの余白（絵の縁から内側の紙まで）**
を返す。余白が分かれば `border-image` で四隅を保ったまま辺だけ伸ばせる。
元の素材は**書き換えない**（ユーザの資産なので触らない）。
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE = Path(tempfile.gettempdir()) / "eve_ui_frames"


@dataclass(frozen=True)
class Frame:
    path: Path | None          # 切り出し済み PNG（失敗時 None）
    left: int
    top: int
    right: int
    bottom: int
    paper: str                 # 内側の紙の色（枠が使えない時のフォールバックに使う）

    @property
    def ok(self) -> bool:
        return self.path is not None and self.path.exists()


def prepare(src: Path) -> Frame:
    """素材を切り出してキャッシュし、9スライス余白と紙色を測って返す。失敗しても落ちない。"""
    fallback = Frame(None, 0, 0, 0, 0, "#ffffff")
    if not src.exists():
        return fallback
    try:
        import numpy as np
        from PIL import Image

        CACHE.mkdir(parents=True, exist_ok=True)
        out = CACHE / (src.stem + ".png")
        if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            meta = _measure(out)
            if meta is not None:
                return meta

        im = Image.open(src).convert("RGBA")
        a = np.array(im)
        solid = a[:, :, 3] > 40                      # 絵がある画素
        ys, xs = np.nonzero(solid)
        if not len(xs):
            return fallback
        im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)).save(out)
        meta = _measure(out)
        return meta if meta is not None else fallback
    except Exception:
        logger.debug("枠素材の下ごしらえに失敗（無地で描く）", exc_info=True)
        return fallback


def _measure(path: Path) -> Frame | None:
    """切り出し済み画像から、内側の紙までの余白と紙色を測る。"""
    try:
        import numpy as np
        from PIL import Image

        a = np.array(Image.open(path).convert("RGBA"))
        h, w = a.shape[:2]
        rgb = a[:, :, :3].astype(int)
        paper = (a[:, :, 3] > 200) & (rgb.min(axis=2) > 236) & \
                (rgb.max(axis=2) - rgb.min(axis=2) < 12)
        ys, xs = np.nonzero(paper)
        if not len(xs):
            return Frame(path, 12, 12, 12, 12, "#ffffff")
        cy, cx = int(ys.mean()), int(xs.mean())
        c = rgb[cy, cx]
        # 余白は「絵の縁 → 内側の紙」まで。小さすぎると装飾が潰れるので下限を置く。
        return Frame(
            path,
            max(8, int(xs.min())), max(8, int(ys.min())),
            max(8, w - 1 - int(xs.max())), max(8, h - 1 - int(ys.max())),
            "#%02x%02x%02x" % (c[0], c[1], c[2]),
        )
    except Exception:
        return None
