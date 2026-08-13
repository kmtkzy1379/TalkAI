"""Live2D 表示ウィジェット（奥背景 → モデル → 手前背景 の3層合成）。

`spike/live2d_test.py`（P0 検証・2026-08-13）で実測した構成をそのまま持ってきたもの:
- `QPainter` + `beginNativePainting()` で Qt の描画と生 GL を挟む。**`live2d.clearBuffer()` は
  呼ばない**（呼ぶと下に描いた奥背景が消える）。
- パラメータは `model.Update()` の**後**・`Draw()` の**前**に書く（Update に上書きされないため）。
- このモデル（nun model）は `model3.json` の EyeBlink/LipSync グループが**空**なので SDK の
  自動まばたき・自動リップシンクは効かない（実機確認済）。まばたきも呼吸も自前で書く。
- `ParamA/I/U/E/O` が無いので**母音別の口の形は作れない**。開き量だけで表現する。
- `ParamEyeBallX` が無いので**眼球の左右は動かない**。首（ParamAngleX）で代替する。

口の開きは外から `set_plan()` で与えられた計画（`eve/response/lipsync.py`）を時間で引く。
UI スレッド以外からは触らないこと（Qt signal 経由で渡す）。
"""
from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..response.lipsync import MouthPlan, MouthSmoother, level_at, plan_duration
from .emotion import Motion, resolve

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
UI_DIR = ROOT / "eve_ui"
MODEL_JSON = UI_DIR / "nun model" / "nun model.model3.json"
BG_INSIDE = UI_DIR / "Live2D_background_inside.png"
BG_FOREGROUND = UI_DIR / "Live2D_background_foreground.png"

# 眉の形（`ParamBrowLForm`/`ParamBrowRForm`）の符号はモデルの作り方次第で反転する。
# 実物を見て逆だったらここを -1 にする（プロファイル側は触らない）。
BROW_FORM_SIGN = 1.0


def _soft(v: float, a: float = 1.3) -> float:
    """飽和カーブ（tanh）。倍率を 2〜3倍に上げても**可動域の端に張り付かない**。

    線形にクランプすると、揺れが大きい感情（喜び・怒り）で首が最大角のまま固まる時間が
    生まれて不自然になる。tanh なら大きい入力ほど緩やかに詰まるので、振れ幅は増えるのに
    動き続ける。小さい入力（既定の calm 付近）ではほぼ線形＝従来の見た目を保つ。
    """
    return a * math.tanh(v / a)


def _approach(cur: Motion, want: Motion, dt: float, tau: float) -> Motion:
    """Motion の全係数を時定数 `tau` 秒で目標へ近づける（感情の切替で動きが飛ばない）。"""
    k = 1.0 - math.exp(-dt / max(1e-3, tau))
    return Motion(**{f: getattr(cur, f) + (getattr(want, f) - getattr(cur, f)) * k
                     for f in Motion.__dataclass_fields__})


def assets_ready() -> tuple[bool, str]:
    missing = [p.name for p in (MODEL_JSON, BG_INSIDE, BG_FOREGROUND) if not p.exists()]
    return (not missing), ("素材が見つかりません: " + ", ".join(missing) if missing else "")


class Live2DView(QOpenGLWidget):
    """Live2D の描画とアイドル動作。口の開閉は `set_plan()` / `clear_plan()` で駆動する。"""

    ready = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.err = ""
        self._bg = QImage(str(BG_INSIDE)) if BG_INSIDE.exists() else QImage()
        self._fg = QImage(str(BG_FOREGROUND)) if BG_FOREGROUND.exists() else QImage()
        self._bg_scaled: Optional[QImage] = None
        self._fg_scaled: Optional[QImage] = None

        # 調整値（P0 の掃引実測で決めた既定・`spike/README.md`）
        self.mouth_gain = 1.0
        self.lipsync_offset_sec = 0.10  # 発話開始フック→実際に音が鳴るまでの遅れ補正
        self.sway_x = 1.0               # 横揺れ（本実装の既定は要調整・UI で可変）
        self.sway_y = 0.5               # 縦揺れは横より鈍く（ユーザ指摘 2026-08-13）

        # 口
        self._smoother = MouthSmoother()
        self._plan: Optional[MouthPlan] = None
        self._plan_t0 = 0.0
        self._plan_dur = 0.0
        # 頭
        self._last = time.monotonic()
        self._next_blink = self._last + random.uniform(2.0, 5.0)
        self._blink_start = -1.0
        self._look = [0.0, 0.0]
        self._body = [0.0, 0.0]
        self._mouse = [0.0, 0.0]
        self._mouse_inside = False
        self._follow = 0.0
        self._glance = [0.0, 0.0]
        self._glance_next = self._last + 2.0
        # 首の傾き: 揺れとは独立にランダムで入れる（ユーザ指定「首を傾ける」）
        self._tilt = 0.0
        self._tilt_target = 0.0
        self._tilt_next = self._last + 3.0
        # 感情 × surprise（表示専用）。滑らかに追従させる＝感情が切り替わっても急変しない
        self._motion: Motion = resolve("", 20)
        self._motion_now: Motion = self._motion
        self.emotion_label = "calm"
        self.surprise = 20
        self._phase = 0.0  # 揺れの位相（速さを可変にするため時間そのものは使わない）
        self._fps = 0.0
        self._frames = 0
        self._fps_t0 = self._last

        self.setMouseTracking(True)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        # **起動中は描画を落とす**（既定 10fps 相当）。60fps で回したまま起動すると、
        # 埋め込みモデル（Ruri）のロードと GIL / CPU を奪い合って**起動が 2倍以上遅くなる**。
        # 実測 2026-08-14: Ruri のロードは UI 無し 11〜15秒 / UI 60fps 併走 26〜44秒。
        # 準備完了で `set_full_speed()` により 60fps へ戻す。
        self._timer.start(100)

    # ------------------------------------------------------------ 口の駆動（UI スレッドから）
    def set_plan(self, plan: Optional[MouthPlan]) -> None:
        """「この文を今から喋る」。計画が無い（VOICEVOX 以外の音源等）なら口は動かさない。"""
        if not plan:
            self._plan = None
            return
        self._plan = plan
        self._plan_t0 = time.monotonic() + self.lipsync_offset_sec
        self._plan_dur = plan_duration(plan)

    def set_full_speed(self) -> None:
        """起動が終わったら 60fps に戻す（起動中は CPU をモデルのロードに譲っている）。"""
        self._timer.start(16)

    def clear_plan(self) -> None:
        """発話終了・barge-in。**口を閉じる**（開きっぱなしを構造的に防ぐ）。"""
        self._plan = None

    def set_mood(self, emotion_text: str, surprise: int) -> None:
        """感情（FeedbackLLM の自由記述）と surprise を反映する。**表示専用**。

        値は UI スレッドが loop から受け取ったコピー。ここから loop へは何も書かない。
        """
        from .emotion import classify

        self.emotion_label = classify(emotion_text)[0]
        self.surprise = surprise
        self._motion = resolve(emotion_text, surprise)

    @property
    def mouth(self) -> float:
        return self._smoother.value

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------ GL
    def initializeGL(self) -> None:
        try:
            import live2d.v3 as live2d

            live2d.glInit()
            self.model = live2d.LAppModel()
            self.model.LoadModelJson(str(MODEL_JSON))
            self.model.SetAutoBreathEnable(False)  # グループが空で効かないため自前で書く
            self.model.SetAutoBlinkEnable(False)
            self.ready.emit(True, "Live2D 読み込み OK")
        except Exception as e:  # 落とさず理由を画面に出す（音声対話は続けられる）
            self.model = None
            self.err = f"{type(e).__name__}: {e}"
            logger.exception("Live2D の初期化に失敗（音声対話は継続）")
            self.ready.emit(False, self.err)

    def resizeGL(self, w: int, h: int) -> None:
        if self.model is not None:
            self.model.Resize(w, h)
        self._bg_scaled = self._fg_scaled = None

    def paintGL(self) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        now = time.monotonic()
        dt = min(0.1, now - self._last)
        self._last = now
        self._tick(dt, now)

        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if not self._bg.isNull():
            if self._bg_scaled is None or self._bg_scaled.size() != self.size():
                self._bg_scaled = self._cover(self._bg, w, h)
            p.drawImage(self._centered(self._bg_scaled, w, h), self._bg_scaled)
        else:
            p.fillRect(0, 0, w, h, Qt.darkGray)

        if self.model is not None:
            p.beginNativePainting()
            try:
                self.model.Update()
                self._apply(now)
                self.model.Draw()
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"
            p.endNativePainting()

        if not self._fg.isNull():
            if self._fg_scaled is None or self._fg_scaled.size() != self.size():
                self._fg_scaled = self._cover(self._fg, w, h)
            p.drawImage(self._centered(self._fg_scaled, w, h), self._fg_scaled)
        if self.err:
            p.setPen(Qt.red)
            p.drawText(QRectF(8, 8, w - 16, 80), Qt.TextWordWrap, f"Live2D エラー: {self.err}")
        p.end()

        self._frames += 1
        if now - self._fps_t0 >= 1.0:
            self._fps = self._frames / (now - self._fps_t0)
            self._frames, self._fps_t0 = 0, now

    def _cover(self, img: QImage, w: int, h: int) -> QImage:
        return img.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

    @staticmethod
    def _centered(img: QImage, w: int, h: int) -> QRectF:
        return QRectF((w - img.width()) / 2.0, (h - img.height()) / 2.0, img.width(), img.height())

    # ------------------------------------------------------------ パラメータ
    def _tick(self, dt: float, now: float) -> None:
        # 口: 計画を時間で引く。計画が尽きたら level_at が 0 を返すので自然に閉じる
        target = 0.0
        if self._plan is not None:
            t = now - self._plan_t0
            if t > self._plan_dur + 0.25:
                self._plan = None       # 保険: 終了通知が来なくても必ず閉じる
            else:
                target = min(1.0, level_at(self._plan, t) * self.mouth_gain)
        self._smoother.step(target, dt)

        # 感情の切り替わりで動きが飛ばないよう、係数自体を時定数 1.5 秒で追従させる
        self._motion_now = _approach(self._motion_now, self._motion, dt, 1.5)

        if self._blink_start < 0 and now >= self._next_blink:
            self._blink_start = now
        if self._blink_start >= 0 and now - self._blink_start > 0.22:
            self._blink_start = -1.0
            # 感情/surprise でまばたきの頻度が変わる（blink が大きいほど間隔が短い）
            rate = max(0.3, self._motion_now.blink)
            self._next_blink = now + random.uniform(2.5, 6.0) / rate
        self._tick_head(dt, now)

    def _tick_head(self, dt: float, now: float) -> None:
        """アイドルの揺れ ＋ ランダムなよそ見 ＋ マウス追従 の合成（`spike/` で調整した式）。"""
        m = self._motion_now
        if now >= self._glance_next:
            self._glance = ([0.0, 0.0] if random.random() < 0.35 else
                            [random.uniform(-0.80, 0.80), random.uniform(-0.35, 0.35)])
            # glance が小さいほど間隔が短い（うれしい/驚いた時によそ見が増える）
            self._glance_next = now + random.uniform(1.8, 4.5) * max(0.25, m.glance)
        if now >= self._tilt_next:  # 首の傾きは揺れとは独立にランダムで入れる
            self._tilt_target = random.uniform(-0.5, 0.5) if random.random() < 0.6 else 0.0
            self._tilt_next = now + random.uniform(3.0, 8.0) * max(0.25, m.glance)
        self._tilt += (self._tilt_target + m.tilt_bias - self._tilt) * (1.0 - math.exp(-dt / 1.1))

        # 感情/surprise で「速さ」を変える＝位相を進める速度そのものを可変にする
        self._phase += dt * max(0.2, m.speed)
        ph = self._phase
        gx, gy = self.sway_x * m.sway, self.sway_y * m.sway
        wx = gx * (0.40 * math.sin(ph * 2 * math.pi / 6.1)
                   + 0.22 * math.sin(ph * 2 * math.pi / 9.7 + 1.3)
                   + 0.14 * math.sin(ph * 2 * math.pi / 3.3 + 0.4))
        wy = gy * (0.16 * math.sin(ph * 2 * math.pi / 7.9 + 0.7)
                   + 0.08 * math.sin(ph * 2 * math.pi / 13.1 + 2.1))
        idle_x = _soft(self._glance[0] * gx + wx)
        idle_y = _soft(self._glance[1] * gy + wy)
        tau_f = 0.25 if self._mouse_inside else 1.4  # 外れた時に正面へ飛ばない
        self._follow += ((1.0 if self._mouse_inside else 0.0) - self._follow) \
            * (1.0 - math.exp(-dt / tau_f))
        tx = idle_x + (self._mouse[0] - idle_x) * self._follow
        ty = idle_y + (self._mouse[1] - idle_y) * self._follow
        # 感情の上下バイアスは**追従の後に上乗せ**する（申し訳なさそうにしていても
        # カーソルは追うが、うつむき加減は残る）。追従中は半分に薄める。
        # ※ blend の前に足すと、マウスが窓の上にあるだけで丸ごと上書きされて消える
        #   （実測 2026-08-13: look_y が全感情で 0.0 になり、うつむきが一切出なかった）。
        ty += m.pitch_bias * (1.0 - 0.5 * self._follow)
        tau_h = 0.18 + 0.32 * (1.0 - self._follow)
        for i, tv in enumerate((tx, ty)):
            self._look[i] += (tv - self._look[i]) * (1.0 - math.exp(-dt / tau_h))
            self._body[i] += (self._look[i] - self._body[i]) * (1.0 - math.exp(-dt / 0.75))

    def _blink(self) -> float:
        if self._blink_start < 0:
            return 1.0
        e = time.monotonic() - self._blink_start
        return 1.0 - e / 0.08 if e < 0.08 else min(1.0, (e - 0.08) / 0.14)

    def _apply(self, now: float) -> None:
        from live2d.v3.params import StandardParams as P

        m = self.model

        def c(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, v))

        lx, ly = c(self._look[0]), c(self._look[1])
        bx, by = c(self._body[0]), c(self._body[1])
        em = self._motion_now
        ph = self._phase
        gx, gy = self.sway_x * em.sway, self.sway_y * em.sway
        mouth = self._smoother.value
        m.SetParameterValue(P.ParamMouthOpenY, mouth, 1.0)
        m.SetParameterValue(P.ParamMouthForm, c(em.mouth_form + 0.25 * mouth), 1.0)
        m.SetParameterValue(P.ParamAngleX, 28.0 * lx, 1.0)
        m.SetParameterValue(P.ParamAngleY, 24.0 * ly, 1.0)
        # 首の傾き = 振り向きに伴う分 + 独立したランダムな傾き（+感情バイアス）
        m.SetParameterValue(P.ParamAngleZ, c(-0.4 * lx + self._tilt) * 28.0, 1.0)
        m.SetParameterValue(P.ParamEyeBallY, c(ly * 1.15), 1.0)
        m.SetParameterValue(P.ParamBodyAngleX,
                            c(_soft(bx * 0.80 + 0.30 * gx * math.sin(ph * 2 * math.pi / 7.3))) * 10, 1.0)
        m.SetParameterValue(P.ParamBodyAngleY,
                            c(_soft(by * 0.35 + 0.10 * gy * math.sin(ph * 2 * math.pi / 5.1 + 0.9))) * 10, 1.0)
        m.SetParameterValue(P.ParamBodyAngleZ,
                            c(_soft(-bx * 0.55 + 0.22 * gx * math.sin(ph * 2 * math.pi / 11.0))) * 10, 1.0)
        m.SetParameterValue(P.ParamBreath,
                            0.5 + 0.5 * math.sin(ph * 2 * math.pi / 3.5), 1.0)
        b = self._blink()
        # 感情の表情（このモデルは眉・目の笑み・頬を持っている）。喋っている間は
        # 目の笑みを弱める（口が開いた笑顔で目まで細めると崩れやすいため）。
        smile = em.eye_smile * (1.0 - 0.5 * mouth)
        m.SetParameterValue(P.ParamEyeLOpen, b, 1.0)
        m.SetParameterValue(P.ParamEyeROpen, b, 1.0)
        m.SetParameterValue(P.ParamEyeLSmile, c(smile, 0.0, 1.0), 1.0)
        m.SetParameterValue(P.ParamEyeRSmile, c(smile, 0.0, 1.0), 1.0)
        m.SetParameterValue(P.ParamCheek, c(em.cheek, 0.0, 1.0), 1.0)
        for pid in (P.ParamBrowLY, P.ParamBrowRY):
            m.SetParameterValue(pid, c(em.brow), 1.0)
        # 眉の形（怒り眉 / 困り眉）。**符号はモデル依存**なので `BROW_FORM_SIGN` で反転できる。
        for pid in (P.ParamBrowLForm, P.ParamBrowRForm):
            m.SetParameterValue(pid, c(em.brow_form * BROW_FORM_SIGN), 1.0)
        # 眉の角度は左右で対称に倒す（内側が下がると怒り顔になる）
        m.SetParameterValue(P.ParamBrowLAngle, c(em.brow_form * BROW_FORM_SIGN), 1.0)
        m.SetParameterValue(P.ParamBrowRAngle, c(em.brow_form * BROW_FORM_SIGN), 1.0)

    # ------------------------------------------------------------ 入力
    def mouseMoveEvent(self, ev) -> None:
        w, h = max(1, self.width()), max(1, self.height())
        self._mouse = [ev.position().x() / w * 2 - 1, -(ev.position().y() / h * 2 - 1)]
        self._mouse_inside = True

    def enterEvent(self, ev) -> None:
        self._mouse_inside = True

    def leaveEvent(self, ev) -> None:
        self._mouse_inside = False

    def release(self) -> None:
        """GL コンテキストを持っている間に Live2D を解放する（ウィンドウを閉じる時に呼ぶ）。"""
        try:
            import live2d.v3 as live2d

            self.makeCurrent()
            live2d.glRelease()
        except Exception:
            pass
