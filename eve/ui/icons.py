"""小さなアイコンをその場で描く（画像素材を増やさない）。

絵文字（🎤）は環境によって色も形も変わり、「斜線入り」に対応する字も無い。
フォント頼みにすると OFF が表現できないので、**QPainter で描く**。
描いてしまえば色も太さも状態に合わせて変えられ、拡大しても崩れない。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def mic_icon(on: bool, size: int = 20, *,
             color: str = "#2b2b33", off_color: str = "#b3261e") -> QIcon:
    """マイクのアイコン。`on=False` のときは**斜線**を重ねる。

    形は3つの部品だけ: 頭（角丸の縦長）／顎（下向きの弧）／柱と台座。
    小さく描くので線は 2px 前後に揃える（細いと薄れ、太いと潰れる）。
    """
    scale = 4  # いったん大きく描いて縮める（斜め線のギザギザを抑える）
    px = QPixmap(size * scale, size * scale)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, True)

    s = size * scale
    # OFF でもマイクの形が読める濃さを保つ。薄い灰色にすると、下の斜線の白縁と相まって
    # 20px では「斜線だけ」に見える（実際に描いて確認した）。
    ink = QColor(color if on else "#6b6b76")
    pen = QPen(ink, s * 0.085, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)

    # 頭（マイク本体）
    head = QRectF(s * 0.36, s * 0.14, s * 0.28, s * 0.42)
    p.setBrush(ink)
    p.drawRoundedRect(head, s * 0.14, s * 0.14)

    # 顎（下向きの弧）
    p.setBrush(Qt.NoBrush)
    cradle = QRectF(s * 0.24, s * 0.36, s * 0.52, s * 0.40)
    p.drawArc(cradle, 180 * 16, 180 * 16)

    # 柱と台座
    p.drawLine(QPointF(s * 0.5, s * 0.72), QPointF(s * 0.5, s * 0.86))
    p.drawLine(QPointF(s * 0.34, s * 0.86), QPointF(s * 0.66, s * 0.86))

    if not on:
        # 斜線。**下に白い隙間を敷いてから**赤い線を引く（線と本体が溶けると
        # 「斜線が入っている」と読めない）。ただし白は**細く**する: 太くすると
        # マイク本体を消してしまい、20px では斜線だけの絵になる（描いて確認した）。
        for c, w in ((QColor("#ffffff"), s * 0.155), (QColor(off_color), s * 0.10)):
            p.setPen(QPen(c, w, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.16, s * 0.84), QPointF(s * 0.84, s * 0.16))

    p.end()
    return QIcon(px.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
