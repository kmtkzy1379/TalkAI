r"""spike/live2d_test.py — P0 技術検証スパイク（**捨てるコード**・本番には使わない）。

目的は `docs/UI_DESIGN.md` §7 P0 の5項目のうち ①②③ を実測で潰すこと:
  ① PySide6 + QOpenGLWidget + live2d-py で `eve_ui/nun model` が描画できるか
  ② 奥背景 → Live2D → 手前背景 の3層合成が成立するか
  ③ 口の開閉（ParamMouthOpenY）を外部から毎フレーム駆動できるか＝リップシンク

使い方:
  # 対話モード（ウィンドウが出る。VOICEVOX 起動中なら実音声で口が動く）
  $env:PYTHONIOENCODING="utf-8"
  & .\.venv\Scripts\python.exe spike\live2d_test.py

  # 自己検証モード（ウィンドウを一瞬出して PNG を書き、判定して終了。VOICEVOX 不要）
  & .\.venv\Scripts\python.exe spike\live2d_test.py --selftest

【リップシンクの方式】
本スパイクは **再生中の WAV の音量包絡（RMS）→ ParamMouthOpenY** で口を開閉する。
モデルに `ParamA/I/U/E/O` が無い（実測）ため母音別の口形は原理的に作れず、開閉量が上限。
「いい感じ」に見せているのは以下の後処理:
  - dB 変換 + パーセンタイル正規化（小声/大声で口の開きが破綻しない）
  - 非対称スムージング（速い attack / 遅い release ＝ パクパクしない）
  - 無音デッドゾーン（文の切れ目でちゃんと閉じる）
  - 口角 ParamMouthForm を開き量に応じて微調整（真顔で開閉するのを避ける）
本番では VOICEVOX の `audio_query`（モーラ単位の母音長）を使う方が正確だが、
それは「実装の差し替え」であって描画側の作りは変わらない（本スパイクの検証範囲外）。
"""
from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "eve_ui"
MODEL_JSON = UI_DIR / "nun model" / "nun model.model3.json"
BG_INSIDE = UI_DIR / "Live2D_background_inside.png"          # 奥背景（部屋）
BG_FOREGROUND = UI_DIR / "Live2D_background_foreground.png"  # 手前背景（机・キーボード）
OUT_DIR = Path(os.environ.get("SPIKE_OUT", ROOT / "spike" / "out"))

VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021")
VV_SPEAKER = int(os.environ.get("VV_SPEAKER", "1"))

from PySide6.QtCore import QRectF, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

import live2d.v3 as live2d  # noqa: E402
from live2d.v3.params import StandardParams as P  # noqa: E402


# ---------------------------------------------------------------- リップシンク

class MoraEnvelope:
    """**VOICEVOX の `audio_query` のモーラ長**から口の開きを組み立てる（本命の方式）。

    RMS 方式（下の `Envelope`）を実測した結果、日本語 TTS では母音が連続的に有声のまま
    繋がるため**モーラごとの谷が振幅に現れない**（掃引しても開閉 2.4回/秒 が上限＝
    実際の 6〜7モーラ/秒 に届かず「口が開きっぱなし」に見える）。
    query には consonant_length / vowel_length が入っているので、そこから直接作る方が
    正確かつ軽い（音声解析が要らない）。

    母音別の**口の形**はこのモデルでは作れない（ParamA/I/U/E/O が無い）が、
    母音別の**開き量**は変えられるので、あ=大きく い/う=小さく とすると自然に見える。
    """

    # 開き量（形は作れないので量で表現する）
    VOWEL_OPEN = {"a": 1.00, "e": 0.72, "o": 0.62, "i": 0.42, "u": 0.34,
                  "N": 0.18, "cl": 0.0, "pau": 0.0}
    PLOSIVE = {"p", "b", "t", "d", "k", "g", "ch", "ts", "ky", "gy", "by", "py", "ty", "dy"}
    # 破裂音以外の子音での開き（0.0=完全に閉じる）。掃引実測 2026-08-13（4文・VOICEVOX speaker1）:
    # 0.15 → 開口率 76% / 閉じ 4.6回/秒、**0.08 → 72% / 5.1回/秒**、0.00 → 72% / 5.1回/秒。
    # 実モーラ速度 4.5個/秒 に対し 0.08 は ほぼ 1:1 で追随しつつ、s/n で完全に閉じ切らない分だけ自然。
    CONSONANT_OPEN = 0.08

    def __init__(self, query: dict) -> None:
        speed = float(query.get("speedScale") or 1.0)
        t = float(query.get("prePhonemeLength") or 0.0) / speed
        segs: list[tuple[float, float, float, float]] = []   # (t0, t1, v0, v1)

        def add(dur: float, v0: float, v1: float) -> None:
            nonlocal t
            if dur > 0:
                segs.append((t, t + dur, v0, v1))
                t += dur

        for phrase in query.get("accent_phrases", []):
            for mora in phrase.get("moras", []):
                vowel = mora.get("vowel") or "a"
                # 大文字＝無声化母音（「です」の す 等）。口は開くが浅く
                devoiced = vowel.isupper() and vowel != "N"
                lv = self.VOWEL_OPEN.get(vowel.lower() if devoiced else vowel, 0.5)
                if devoiced:
                    lv *= 0.45
                cl = mora.get("consonant_length")
                if cl:
                    c = mora.get("consonant") or ""
                    lvc = 0.0 if c in self.PLOSIVE else self.CONSONANT_OPEN
                    add(cl / speed, lvc, lvc)
                vl = float(mora.get("vowel_length") or 0.0) / speed
                # 長い母音は少しずつ閉じていく（固まって見えないように）
                add(vl, lv, lv * (0.78 if vl > 0.12 else 1.0))
            pm = phrase.get("pause_mora")
            if pm:
                add(float(pm.get("vowel_length") or 0.0) / speed, 0.0, 0.0)
        t += float(query.get("postPhonemeLength") or 0.0) / speed

        self._t0 = np.array([s[0] for s in segs], dtype=np.float32)
        self._t1 = np.array([s[1] for s in segs], dtype=np.float32)
        self._v0 = np.array([s[2] for s in segs], dtype=np.float32)
        self._v1 = np.array([s[3] for s in segs], dtype=np.float32)
        self.duration = t

    def level(self, t: float) -> float:
        if t < 0 or len(self._t0) == 0:
            return 0.0
        i = int(np.searchsorted(self._t0, t, side="right")) - 1
        if i < 0 or t >= self._t1[i]:
            return 0.0  # 区間の隙間＝無音
        f = (t - self._t0[i]) / max(1e-6, self._t1[i] - self._t0[i])
        return float(self._v0[i] + (self._v1[i] - self._v0[i]) * f)


class Envelope:
    """WAV の音量包絡（フォールバック）。`level(t)` で経過秒に対する 0..1 を返す。

    VOICEVOX 以外の音（将来の別 TTS・録音）でも口を動かせるように残してある。
    モーラ情報が取れるなら `MoraEnvelope` の方が良い（上記コメント参照）。
    """

    HOP_SEC = 0.010   # 10ms 刻み
    WIN_SEC = 0.025   # 25ms 窓

    def __init__(self, samples: np.ndarray, rate: int) -> None:
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        hop = max(1, int(rate * self.HOP_SEC))
        win = max(hop, int(rate * self.WIN_SEC))
        n = max(1, (len(samples) - win) // hop + 1)
        # RMS（フレームごと）
        rms = np.empty(n, dtype=np.float32)
        for i in range(n):
            seg = samples[i * hop: i * hop + win]
            rms[i] = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2))) if len(seg) else 0.0
        db = 20.0 * np.log10(np.maximum(rms, 1e-6))
        # パーセンタイル正規化: 上位5%を「口全開」、そこから 32dB 下を「閉じ」とする。
        # 絶対値で正規化すると小声の発話で口が開かない/大声で開きっぱなしになる。
        ceil_db = float(np.percentile(db, 95))
        floor_db = ceil_db - 32.0
        lv = (db - floor_db) / max(1e-6, ceil_db - floor_db)
        lv = np.clip(lv, 0.0, 1.0) ** 0.85  # 軽いガンマ（小さい音を少し持ち上げる）
        lv[lv < 0.06] = 0.0                 # 無音デッドゾーン（文の切れ目で閉じる）
        self._lv = lv.astype(np.float32)
        self._hop = self.HOP_SEC
        self.duration = len(samples) / float(rate)

    def level(self, t: float) -> float:
        if t < 0.0:
            return 0.0
        i = int(t / self._hop)
        return float(self._lv[i]) if 0 <= i < len(self._lv) else 0.0

    @classmethod
    def from_wav_bytes(cls, wav: bytes) -> "Envelope":
        import io
        import soundfile as sf
        data, rate = sf.read(io.BytesIO(wav), dtype="float32")
        return cls(data, rate)

    @classmethod
    def synthetic(cls, seconds: float = 3.0, rate: int = 24000) -> "Envelope":
        """VOICEVOX が無い時用の疑似包絡（モーラ風の断続 + 文の切れ目）。"""
        t = np.arange(int(seconds * rate)) / rate
        mora = (np.sin(2 * np.pi * 6.5 * t) * 0.5 + 0.5) ** 1.5       # 6.5モーラ/秒くらい
        phrase = (np.sin(2 * np.pi * 0.45 * t) * 0.5 + 0.5) > 0.25    # 息継ぎ
        noise = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 1.7 * t + 1.0))
        return cls((mora * phrase * noise * 0.5).astype(np.float32), rate)


class SpeechPlayer:
    """VOICEVOX で合成して再生し、`level()` で「今の口の開き」を返す（スパイク用の簡易実装）。

    本番の `eve/response/{tts,player}.py` は asyncio だが、ここは描画検証が目的なので
    素のスレッド + 壁時計で十分（再生位置は「最初の書き込み時刻からの経過」で近似する）。
    """

    def __init__(self) -> None:
        self._env = None         # MoraEnvelope | Envelope | None
        self._t0 = 0.0
        self._gen = 0            # 世代（連打時に古い再生を捨てる＝barge-in の縮小版）
        self._pending = False    # 合成中（まだ包絡が無い）
        self._lock = threading.Lock()
        self.offset_ms = 40.0    # 再生バッファ分の遅れ補正（対話モードのスライダで調整）
        self.status = "待機中"

    # --- 呼び出し側 API ---
    def busy(self) -> bool:
        with self._lock:
            return self._pending or self._env is not None

    def speak_text(self, text: str) -> None:
        with self._lock:
            self._gen += 1
            self._pending = True
            gen = self._gen
        threading.Thread(target=self._worker, args=(text, gen), daemon=True).start()

    def speak_synthetic(self) -> None:
        with self._lock:
            self._gen += 1
            self._env = Envelope.synthetic()
            self._t0 = time.monotonic()
            self.status = "疑似包絡で口だけ動かしています（音は出ません）"

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            self._env = None
            self.status = "停止"

    def level(self) -> float:
        with self._lock:
            if self._env is None:
                return 0.0
            t = time.monotonic() - self._t0 - self.offset_ms / 1000.0
            if t > self._env.duration + 0.3:
                self._env = None
                return 0.0
            return self._env.level(t)

    # --- ワーカー ---
    def _worker(self, text: str, gen: int) -> None:
        try:
            wav, query = self._synthesize(text)
        except Exception as e:
            with self._lock:
                self._pending = False
                self.status = f"VOICEVOX に繋がりません（{type(e).__name__}）→ [疑似発話] で口だけ確認できます"
            return
        try:
            env = MoraEnvelope(query)          # 本命: モーラ長から作る
            if env.duration <= 0:
                raise ValueError("モーラが空")
            how = "モーラ"
        except Exception:
            env, how = Envelope.from_wav_bytes(wav), "RMS"   # フォールバック
        with self._lock:
            self._pending = False
            if gen != self._gen:
                return
            self._env, self._t0 = env, time.monotonic()
            self.status = f"再生中（{env.duration:.1f}秒・{how}方式）"
        self._play(wav, gen)
        with self._lock:
            if gen == self._gen:
                self.status = "待機中"

    def _synthesize(self, text: str) -> tuple[bytes, dict]:
        """WAV と **audio_query の両方**を返す（query にモーラ長が入っている）。

        本番の `eve/response/tts.py:37-58` は WAV しか返さず query を捨てているため、
        リップシンクを入れるならここを `(wav, query)` にする改修が要る（数行）。
        """
        import requests
        q = requests.post(f"{VOICEVOX_URL}/audio_query",
                          params={"text": text, "speaker": VV_SPEAKER}, timeout=10)
        q.raise_for_status()
        query = q.json()
        r = requests.post(f"{VOICEVOX_URL}/synthesis", params={"speaker": VV_SPEAKER},
                          json=query, timeout=30)
        r.raise_for_status()
        return r.content, query

    def _play(self, wav: bytes, gen: int) -> None:
        import io
        import pyaudio
        import soundfile as sf
        data, rate = sf.read(io.BytesIO(wav), dtype="float32")
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)
        try:
            # 実際に音が出始めた時刻を再取得（合成〜デバイス確保の分だけ後ろにずれるため）
            with self._lock:
                if gen != self._gen:
                    return
                self._t0 = time.monotonic()
            for i in range(0, len(data), 1024):
                with self._lock:
                    if gen != self._gen:      # 連打・停止で即座に譲る
                        break
                stream.write(data[i:i + 1024].tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


# ---------------------------------------------------------------- 描画

class Live2DView(QOpenGLWidget):
    """奥背景 → Live2D → 手前背景 を1つの GL ウィジェットで合成する。

    背景は QPainter、モデルは live2d-py（生 GL）。`beginNativePainting()` で挟むのが Qt の作法。
    **`live2d.clearBuffer()` は呼ばない**（呼ぶと下に描いた奥背景が消える）。
    """

    ready = Signal(bool, str)

    def __init__(self, player: SpeechPlayer) -> None:
        super().__init__()
        self.player = player
        self.model: Optional[live2d.LAppModel] = None
        self.err = ""
        self._bg = QImage(str(BG_INSIDE)) if BG_INSIDE.exists() else QImage()
        self._fg = QImage(str(BG_FOREGROUND)) if BG_FOREGROUND.exists() else QImage()
        self._bg_scaled: Optional[QImage] = None
        self._fg_scaled: Optional[QImage] = None

        # 調整パラメータ（対話モードのスライダで動かして、良い値を本番に焼き込む）
        self.mouth_gain = 1.0
        # 掃引実測 2026-08-13: attack は 15-22ms で差が出ない / release は 40ms が最良
        # （30ms=開口率67% と閉じすぎ、55ms=74% でやや開きっぱなし）。
        self.attack_sec = 0.020
        self.release_sec = 0.040
        self.model_scale = 1.0
        self.model_dx = 0.0
        self.model_dy = 0.0
        self.auto_blink = False       # モデルの EyeBlink グループが空なので既定は手動
        self.manual_idle = True

        # 揺れの強さ。**横と縦は別物**（実測フィードバック 2026-08-13: 縦は鈍い方が自然・
        # 横はもっと欲しい）。将来 surprise でこの倍率を動かす＝KIKAKUSHO §6 の実装点。
        # スパイクでは可動域を見るために既定を最大（2.5）で立ち上げる。本実装の既定は別途決める。
        self.sway_x = 2.5
        self.sway_y = 2.5

        # 内部状態
        self._mouth = 0.0
        self._last = time.monotonic()
        self._next_blink = time.monotonic() + random.uniform(2.0, 5.0)
        self._blink_start = -1.0
        # 頭の向き: アイドルの揺れ / ランダムなよそ見 / マウス追従 を1つにブレンドする
        self._look = [0.0, 0.0]       # 実際に適用中の向き（-1..1）
        self._body = [0.0, 0.0]       # 体は首に遅れて付いていく（追従遅れ＝生っぽさ）
        self._mouse = [0.0, 0.0]
        self._mouse_inside = False
        self._follow = 0.0            # 0=アイドル 1=マウス追従（ここを緩やかに動かす）
        self._glance = [0.0, 0.0]
        self._glance_next = time.monotonic() + 2.0
        self._fps = 0.0
        self._frames = 0
        self._fps_t0 = time.monotonic()

        self.setMouseTracking(True)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)  # 約60fps

    # --- GL ライフサイクル ---
    def initializeGL(self) -> None:
        try:
            live2d.glInit()
            self.model = live2d.LAppModel()
            self.model.LoadModelJson(str(MODEL_JSON))
            self.model.SetAutoBreathEnable(False)  # 呼吸も自前（比較できるように）
            self.model.SetAutoBlinkEnable(False)
            self.ready.emit(True, "モデル読み込み OK")
        except Exception as e:  # 失敗しても落とさず理由を画面に出す（P0 の判定材料にする）
            self.model = None
            self.err = f"{type(e).__name__}: {e}"
            self.ready.emit(False, self.err)

    def resizeGL(self, w: int, h: int) -> None:
        if self.model is not None:
            self.model.Resize(w, h)
        self._bg_scaled = self._fg_scaled = None

    def _cover(self, img: QImage, w: int, h: int) -> QImage:
        """アスペクトを保ったまま領域を埋める（余白を作らない＝cover）。"""
        return img.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

    def paintGL(self) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        dt = min(0.1, time.monotonic() - self._last)
        self._last = time.monotonic()
        self._tick_params(dt)

        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # ① 奥背景
        if not self._bg.isNull():
            if self._bg_scaled is None or self._bg_scaled.size() != self.size():
                self._bg_scaled = self._cover(self._bg, w, h)
            p.drawImage(self._centered(self._bg_scaled, w, h), self._bg_scaled)
        else:
            p.fillRect(0, 0, w, h, Qt.darkGray)

        # ② Live2D（生 GL）
        if self.model is not None:
            p.beginNativePainting()
            try:
                self.model.Update()
                self._apply_params()   # Update の後に入れる（Update 内で上書きされないように）
                self.model.Draw()
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"
            p.endNativePainting()

        # ③ 手前背景（RGBA・上部が透過）
        if not self._fg.isNull():
            if self._fg_scaled is None or self._fg_scaled.size() != self.size():
                self._fg_scaled = self._cover(self._fg, w, h)
            p.drawImage(self._centered(self._fg_scaled, w, h), self._fg_scaled)

        if self.err:
            p.setPen(Qt.red)
            p.drawText(QRectF(8, 8, w - 16, 80), Qt.TextWordWrap, f"Live2D エラー: {self.err}")
        p.end()

        self._frames += 1
        if time.monotonic() - self._fps_t0 >= 1.0:
            self._fps = self._frames / (time.monotonic() - self._fps_t0)
            self._frames, self._fps_t0 = 0, time.monotonic()

    @staticmethod
    def _centered(img: QImage, w: int, h: int) -> QRectF:
        return QRectF((w - img.width()) / 2.0, (h - img.height()) / 2.0, img.width(), img.height())

    # --- パラメータ駆動 ---
    def _tick_params(self, dt: float) -> None:
        """口の開き（非対称スムージング）とまばたきの時計を進める。"""
        target = min(1.0, self.player.level() * self.mouth_gain)
        tau = self.attack_sec if target > self._mouth else self.release_sec
        # フレームレート非依存の指数追従（dt が揺れても同じ聞こえ方になる）
        self._mouth += (target - self._mouth) * (1.0 - math.exp(-dt / max(1e-3, tau)))

        now = time.monotonic()
        if self._blink_start < 0 and now >= self._next_blink:
            self._blink_start = now
        if self._blink_start >= 0 and now - self._blink_start > 0.22:
            self._blink_start = -1.0
            self._next_blink = now + random.uniform(2.5, 6.0)
        self._tick_head(dt, now)

    def _tick_head(self, dt: float, now: float) -> None:
        """アイドルの揺れ ＋ ランダムなよそ見 ＋ マウス追従 を1つの向きに合成する。

        - **マウスが外れた瞬間に正面へ飛ばない**: 追従の重み `_follow` を時定数で落とす
          （入る時 0.25秒 / 抜ける時 1.4秒＝抜ける側をわざと遅くする）。
        - **追従していない間も止まらない**: 周期が互いに素な正弦を重ねた緩いうねりに、
          数秒おきの「よそ見」を足す（同じ動きの繰り返しに見せない）。
        - 体は首に遅れて付いていく（`_body` が `_look` を長い時定数で追う）。この遅れが
          髪の physics を動かすので、揺れが実際に見えるようになる。
        """
        if now >= self._glance_next:      # ① 数秒おきによそ見（35%で正面に戻る）
            self._glance = ([0.0, 0.0] if random.random() < 0.35 else
                            [random.uniform(-0.80, 0.80), random.uniform(-0.35, 0.35)])
            self._glance_next = now + random.uniform(1.8, 4.5)

        # ② 緩いうねり。周期が揃わないので繰り返して見えない。
        #    横だけ速い成分(3.3秒)を混ぜる＝**角速度が出て髪の physics が揺れる**
        #    （遅い正弦だけだと位置は変わっても速度が出ず、髪が付いてこない）。
        gx, gy = self.sway_x, self.sway_y
        wx = gx * (0.40 * math.sin(now * 2 * math.pi / 6.1)
                   + 0.22 * math.sin(now * 2 * math.pi / 9.7 + 1.3)
                   + 0.14 * math.sin(now * 2 * math.pi / 3.3 + 0.4))
        wy = gy * (0.16 * math.sin(now * 2 * math.pi / 7.9 + 0.7)
                   + 0.08 * math.sin(now * 2 * math.pi / 13.1 + 2.1))
        idle_x = self._glance[0] * gx + wx
        idle_y = self._glance[1] * gy + wy

        tau_f = 0.25 if self._mouse_inside else 1.4   # ③ 追従の重み（抜ける側を遅く）
        self._follow += ((1.0 if self._mouse_inside else 0.0) - self._follow) \
            * (1.0 - math.exp(-dt / tau_f))

        tx = idle_x + (self._mouse[0] - idle_x) * self._follow
        ty = idle_y + (self._mouse[1] - idle_y) * self._follow

        tau_h = 0.18 + 0.32 * (1.0 - self._follow)    # ④ 追従中は機敏・アイドルはゆったり
        for i, tv in enumerate((tx, ty)):
            self._look[i] += (tv - self._look[i]) * (1.0 - math.exp(-dt / tau_h))
            self._body[i] += (self._look[i] - self._body[i]) * (1.0 - math.exp(-dt / 0.75))

    def _blink_value(self) -> float:
        """0=閉じ 1=開き。閉じ 80ms → 開き 140ms の非対称カーブ。"""
        if self._blink_start < 0:
            return 1.0
        e = time.monotonic() - self._blink_start
        if e < 0.08:
            return 1.0 - e / 0.08
        return min(1.0, (e - 0.08) / 0.14)

    def _apply_params(self) -> None:
        m = self.model
        assert m is not None
        # 口: 開閉量 + 口角を少しだけ連動（真顔でパクパクさせない）
        m.SetParameterValue(P.ParamMouthOpenY, self._mouth, 1.0)
        m.SetParameterValue(P.ParamMouthForm, 0.35 + 0.25 * self._mouth, 1.0)
        if not self.manual_idle:
            return
        t = time.monotonic()
        gx, gy = self.sway_x, self.sway_y

        def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, v))

        lx, ly = clamp(self._look[0]), clamp(self._look[1])
        bx, by = clamp(self._body[0]), clamp(self._body[1])
        # 首（±30）。振り向くと少し首が傾く（ParamAngleZ）＝棒立ちに見せない
        m.SetParameterValue(P.ParamAngleX, 28.0 * lx, 1.0)
        m.SetParameterValue(P.ParamAngleY, 24.0 * ly, 1.0)
        m.SetParameterValue(P.ParamAngleZ, -11.0 * lx, 1.0)
        # 眼球は縦のみ（ParamEyeBallX がモデルに無い）。首より少し先に動くと生きて見える
        m.SetParameterValue(P.ParamEyeBallY, clamp(ly * 1.15), 1.0)
        # 体（±10）: 首に遅れて付いていく分 + 独立した重心移動。この「遅れ」が髪を揺らす。
        # 横（X=向き / Z=傾き）は大きく、縦（Y）は控えめ＝ユーザ指摘の配分。
        m.SetParameterValue(P.ParamBodyAngleX,
                            clamp(bx * 0.80 + 0.30 * gx * math.sin(t * 2 * math.pi / 7.3)) * 10, 1.0)
        m.SetParameterValue(P.ParamBodyAngleY,
                            clamp(by * 0.35 + 0.10 * gy * math.sin(t * 2 * math.pi / 5.1 + 0.9)) * 10, 1.0)
        m.SetParameterValue(P.ParamBodyAngleZ,
                            clamp(-bx * 0.55 + 0.22 * gx * math.sin(t * 2 * math.pi / 11.0)) * 10, 1.0)
        m.SetParameterValue(P.ParamBreath, 0.5 + 0.5 * math.sin(t * 2 * math.pi / 3.5), 1.0)
        if not self.auto_blink:
            b = self._blink_value()
            m.SetParameterValue(P.ParamEyeLOpen, b, 1.0)
            m.SetParameterValue(P.ParamEyeROpen, b, 1.0)

    # --- 入力 ---
    def mouseMoveEvent(self, ev) -> None:
        w, h = max(1, self.width()), max(1, self.height())
        self._mouse = [ev.position().x() / w * 2 - 1, -(ev.position().y() / h * 2 - 1)]
        self._mouse_inside = True

    def enterEvent(self, ev) -> None:
        self._mouse_inside = True

    def leaveEvent(self, ev) -> None:
        # 目標を 0 に飛ばさない。追従の重みだけ下ろして、アイドルの揺れへ**ゆっくり戻す**
        self._mouse_inside = False

    def apply_transform(self) -> None:
        if self.model is not None:
            self.model.SetScale(self.model_scale)
            self.model.SetOffset(self.model_dx, self.model_dy)


# ---------------------------------------------------------------- 対話ウィンドウ

DEMO_LINES = [
    # 口の動きの見どころを変えた5文（順に: 普通 / 母音単体 / 長め / 間と小さい口 / 早口）
    "こんにちは。イブだよ。口の動き、どうかな？",
    "あー。いー。うー。えー。おー。",
    "今日はいい天気だね。散歩でも行こうか？",
    "えっと、ちょっと待ってね。……うん、わかった！",
    "早口で言うよ。すもももももももものうち。",
]


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live2D スパイク — PySide6 + live2d-py（P0 検証・捨てるコード）")
        self.player = SpeechPlayer()
        self.view = Live2DView(self.player)
        self.view.setMinimumSize(600, 400)

        self.status = QLabel("起動中…")
        self.status.setWordWrap(True)
        self.view.ready.connect(lambda ok, msg: self.status.setText(("✅ " if ok else "❌ ") + msg))

        self.edit = QLineEdit("こんにちは。今日はいい天気だね。ちょっと画面を見せてくれる？")
        speak = QPushButton("VOICEVOX で喋らせる")
        speak.clicked.connect(lambda: self.player.speak_text(self.edit.text()))
        fake = QPushButton("疑似発話（音なし）")
        fake.clicked.connect(self.player.speak_synthetic)
        stop = QPushButton("停止")
        stop.clicked.connect(self.player.stop)

        row1 = QHBoxLayout()
        row1.addWidget(self.edit, 1)
        row1.addWidget(speak)
        row1.addWidget(fake)
        row1.addWidget(stop)

        rows = QVBoxLayout()
        rows.addLayout(row1)
        rows.addLayout(self._sliders())

        self.blink_cb = QCheckBox("SDK の自動まばたきを使う（このモデルでは効かないはず＝検証用）")
        self.blink_cb.toggled.connect(self._toggle_auto_blink)
        rows.addWidget(self.blink_cb)
        rows.addWidget(self.status)

        root = QVBoxLayout(self)
        root.addWidget(self.view, 1)
        root.addLayout(rows)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._refresh)
        self._poll.start(200)

        # --- デモ（--demo）: 触らずに眺めて口の動きを確かめるための自動再生 ---
        self._demo: list[str] = []
        self._demo_wait_until = 0.0
        self._demo_timer = QTimer(self)
        self._demo_timer.timeout.connect(self._demo_tick)

    def start_demo(self, loop: bool = False) -> None:
        self._demo = list(DEMO_LINES)
        self._demo_loop = loop
        self._demo_wait_until = time.monotonic() + 2.0  # 起動直後の warmup を避ける
        self._demo_timer.start(250)

    def _demo_tick(self) -> None:
        if time.monotonic() < self._demo_wait_until or self.player.busy():
            return
        if not self._demo:
            if getattr(self, "_demo_loop", False):
                self._demo = list(DEMO_LINES)
            else:
                self._demo_timer.stop()
                return
        line = self._demo.pop(0)
        self.edit.setText(line)
        self.player.speak_text(line)
        self._demo_wait_until = time.monotonic() + 1.2  # 文の間（合成待ちは busy() が見る）

    def _sliders(self) -> QHBoxLayout:
        box = QHBoxLayout()

        def add(label: str, lo: int, hi: int, init: int, fn) -> None:
            box.addWidget(QLabel(label))
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(init)
            s.valueChanged.connect(fn)
            box.addWidget(s)

        add("口の開き", 20, 300, 100, lambda v: setattr(self.view, "mouth_gain", v / 100))
        add("attack(ms)", 5, 150, 20, lambda v: setattr(self.view, "attack_sec", v / 1000))
        add("release(ms)", 20, 400, 40, lambda v: setattr(self.view, "release_sec", v / 1000))
        add("音ズレ(ms)", -200, 300, 40, lambda v: setattr(self.player, "offset_ms", float(v)))
        add("横揺れ", 0, 250, 250, lambda v: setattr(self.view, "sway_x", v / 100))
        add("縦揺れ", 0, 250, 250, lambda v: setattr(self.view, "sway_y", v / 100))
        add("拡大", 50, 250, 100, self._set_scale)
        add("上下", -100, 100, 0, self._set_dy)
        return box

    def _set_scale(self, v: int) -> None:
        self.view.model_scale = v / 100
        self.view.apply_transform()

    def _set_dy(self, v: int) -> None:
        self.view.model_dy = v / 100
        self.view.apply_transform()

    def _toggle_auto_blink(self, on: bool) -> None:
        self.view.auto_blink = on
        if self.view.model is not None:
            self.view.model.SetAutoBlinkEnable(on)

    def _refresh(self) -> None:
        base = self.status.text().split(" ／ ")[0]
        self.status.setText(
            f"{base} ／ {self.player.status} ／ {self.view._fps:.0f} fps ／ "
            f"口={self.view._mouth:.2f} 首=({self.view._look[0]:+.2f},{self.view._look[1]:+.2f}) "
            f"追従={self.view._follow:.2f} "
            f"（横揺れ {self.view.sway_x:.2f} / 縦揺れ {self.view.sway_y:.2f} / "
            f"拡大 {self.view.model_scale:.2f} / 上下 {self.view.model_dy:+.2f}）")

    def closeEvent(self, ev) -> None:
        self.player.stop()
        self.view.makeCurrent()
        try:
            live2d.glRelease()
        except Exception:
            pass
        super().closeEvent(ev)


# ---------------------------------------------------------------- 自己検証

def selftest() -> int:
    """ウィンドウを一瞬だけ出して PNG を書き、①②③ を機械判定する（VOICEVOX 不要）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    player = SpeechPlayer()
    view = Live2DView(player)
    view.resize(900, 600)
    view.show()
    for _ in range(30):          # コンテキスト生成 + 数フレーム回す
        app.processEvents()
        time.sleep(0.02)

    results: list[tuple[str, bool, str]] = []
    ok_model = view.model is not None and not view.err
    results.append(("① モデル読み込み・描画", ok_model, view.err or "例外なし"))
    if not ok_model:
        _report(results)
        return 1

    m = view.model
    assert m is not None

    # パラメータの実測値（本番の実装判断に使う一次データ）
    ids = m.GetParamIds()
    ranges = {}
    for i in range(m.GetParameterCount()):
        prm = m.GetParameter(i)
        ranges[prm.id] = (prm.min, prm.max, prm.default)
    print("\n--- モデルのパラメータ（実測） ---")
    for key in ("ParamMouthOpenY", "ParamMouthForm", "ParamEyeLOpen", "ParamAngleX",
                "ParamEyeBallY", "ParamBreath"):
        print(f"  {key:16} {ranges.get(key, '(無し)')}")
    print(f"  ParamA/I/U/E/O の有無: {[p for p in ('ParamA','ParamI','ParamU') if p in ids] or '無し'}")
    print(f"  ParamEyeBallX の有無 : {'あり' if 'ParamEyeBallX' in ids else '無し'}")

    def grab(mouth: float) -> QImage:
        player.stop()
        view._mouth = mouth
        view.manual_idle = False           # 揺れを止めて口の差分だけを見る
        view.makeCurrent()
        img = view.grabFramebuffer()
        return img

    closed = grab(0.0)
    opened = grab(1.0)
    closed.save(str(OUT_DIR / "mouth_closed.png"))
    opened.save(str(OUT_DIR / "mouth_open.png"))

    # ② 3層合成: 手前背景（机）が最後に描かれているか＝下端の行が背景画像と一致するか
    #    （モデルは机の後ろにいるので、下端は必ず手前背景の色になる）
    a = _to_array(closed)
    fg = QImage(str(BG_FOREGROUND))
    ok_layers = (not fg.isNull()) and a.shape[0] > 0 and float(a[-6:, :, :3].std()) > 1.0
    results.append(("② 3層合成（奥→モデル→手前）が描かれている", ok_layers,
                    f"下端6行の分散={float(a[-6:, :, :3].std()):.1f}（真っ黒/単色なら失敗）"))

    # ③ 口の開閉が実際に絵に出るか（同じ姿勢で ParamMouthOpenY だけを変えて比較）
    #    変化**量**ではなく「変化が口の位置に局在しているか」で判定する。900x600 では顔が
    #    150px 程度しかなく、口は 20x12px＝数百ピクセルにしかならない（当初 200 の閾値は
    #    絵が出ているのに FAIL を出した＝判定側の誤り）。
    b = _to_array(opened)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2)
    mask = diff > 12
    changed = int(mask.sum())
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    if changed:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        cx, cy = (bbox[0] + bbox[2]) / 2 / W, (bbox[1] + bbox[3]) / 2 / H
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / float(W * H)
        localized = area < 0.02 and 0.25 < cx < 0.75 and 0.10 < cy < 0.60  # 顔のあたりに小さく
        note = (f"変化{changed}px / 重心=({cx:.2f},{cy:.2f}) / 面積比={area:.4f} "
                f"→ {'口の位置に局在' if localized else '広範囲＝口以外も動いた'}")
    else:
        localized, note = False, "変化ピクセルが 0（パラメータが描画に届いていない）"
    ok_mouth = changed >= 40 and localized
    results.append(("③ ParamMouthOpenY の外部駆動が描画に反映される", ok_mouth, note))

    # ④ 描画性能（P8 の受け入れ条件の基準値を取っておく）
    view.manual_idle = True
    t0 = time.monotonic()
    for _ in range(120):
        view.update()
        app.processEvents()
    fps = 120.0 / max(1e-6, time.monotonic() - t0)
    results.append(("④ 描画レート（参考値・60fps 上限のタイマ駆動）", fps > 30.0,
                    f"{fps:.0f} fps / 解像度 {closed.width()}x{closed.height()}px "
                    f"（論理 {view.width()}x{view.height()} ＝ devicePixelRatio "
                    f"{closed.width() / max(1, view.width()):.2f}）"))

    # 参考: SDK の自動まばたきがこのモデルで効くか（model3.json の EyeBlink グループが空）
    m.SetAutoBlinkEnable(True)
    eye_idx = ids.index("ParamEyeLOpen") if "ParamEyeLOpen" in ids else -1
    seen = set()
    t_end = time.monotonic() + 6.0
    while time.monotonic() < t_end and eye_idx >= 0:
        m.Update()
        seen.add(round(m.GetParameterValue(eye_idx), 2))
        time.sleep(0.02)
    m.SetAutoBlinkEnable(False)
    auto_blink_works = len(seen) > 1
    results.append((f"（参考）SDK 自動まばたき: {'効く' if auto_blink_works else '効かない'}",
                    True, f"6秒間の ParamEyeLOpen の値={sorted(seen)}"))

    # 見て確認できるように 2枚を横に並べた PNG も出す
    _side_by_side(closed, opened, OUT_DIR / "mouth_compare.png")

    view.makeCurrent()
    try:
        live2d.glRelease()
    except Exception:
        pass
    view.close()
    return _report(results)


def lipsync_check(text: str) -> int:
    """VOICEVOX で実際に合成し、口の開き方を**音を出さずに数値と図で**確認する。

    「いい感じか」を耳ではなく形で見るための道具:
      - 開口率 = 口が開いている時間の割合（日本語の発話なら 60〜85% あたりが自然）
      - 開閉回数/秒 ≈ モーラ速度（日本語は 5〜8 モーラ/秒）。1〜2 なら鈍すぎ、15 ならパクパク
    """
    player = SpeechPlayer()
    print(f"合成中: {text}")
    wav, query = player._synthesize(text)

    def run(env, attack: float, release: float) -> np.ndarray:
        """60fps の描画ループを再現（`Live2DView._tick_params` と同じ式）。"""
        dt, cur, series, t = 1 / 60, 0.0, [], 0.0
        while t < env.duration:
            target = min(1.0, env.level(t))
            tau = attack if target > cur else release
            cur += (target - cur) * (1.0 - math.exp(-dt / tau))
            series.append(cur)
            t += dt
        return np.array(series)

    def stats(s: np.ndarray, dur: float) -> tuple[float, float]:
        """開口率と「閉じの回数/秒」。閉じ = 直前のピークの 55% 未満まで落ちた回数。

        固定閾値の横断回数で数えると、小さい母音（う=0.34）が閾値を跨がず**数え落とす**。
        モーラ速度と直接比べたいので相対的な谷で数える。
        """
        n, peak, armed = 0, 0.0, False
        for v in s:
            peak = max(peak, float(v))
            if armed and v < peak * 0.55:
                n += 1
                peak, armed = float(v), False
            if v > 0.25:
                armed = True
        return float((s > 0.10).mean()), n / max(1e-6, dur)

    mora = MoraEnvelope(query)
    rms = Envelope.from_wav_bytes(wav)
    s_mora = run(mora, 0.020, 0.040)
    s_rms = run(rms, 0.030, 0.090)
    r_m, c_m = stats(s_mora, mora.duration)
    r_r, c_r = stats(s_rms, rms.duration)

    n_mora = sum(len(p.get("moras", [])) for p in query.get("accent_phrases", []))
    rate = n_mora / mora.duration
    print(f"\n長さ {mora.duration:.2f}秒 / モーラ {n_mora}個 = {rate:.1f}モーラ/秒")
    print(f"  {'方式':10} {'開口率':>7} {'閉じ/秒':>8}  （目標: 閉じ ≈ モーラ速度）")
    print(f"  {'モーラ':10} {r_m * 100:6.0f}% {c_m:8.1f}   ← 採用")
    print(f"  {'RMS':10} {r_r * 100:6.0f}% {c_r:8.1f}   ← 参考（振幅に谷が出ないので追随できない）")

    for name, s in (("モーラ方式", s_mora), ("RMS 方式", s_rms)):
        rows, cols = 10, min(150, len(s))
        step = max(1, len(s) // cols)
        grid = [[" "] * cols for _ in range(rows)]
        for c in range(cols):
            v = float(s[c * step: (c + 1) * step].max())
            for r in range(int(v * rows)):
                grid[rows - 1 - r][c] = "|"
        print(f"\n{name}（縦=口の開き 0→1・横=時間）")
        for row in grid:
            print("  " + "".join(row))

    # モーラ速度に追随できているか（8割以上拾えていれば「モーラごとに口が動いている」）
    ok = 0.45 < r_m < 0.92 and c_m > 0.8 * rate
    print(f"\n判定: {'PASS' if ok else 'FAIL'}"
          f"（開口率 45-92% かつ 閉じがモーラ速度の8割以上 = {0.8 * rate:.1f} 回/秒）")
    return 0 if ok else 1


def _to_array(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGBA8888)
    ptr = img.constBits()
    return np.frombuffer(ptr, np.uint8).reshape(img.height(), img.width(), 4).copy()


def _side_by_side(a: QImage, b: QImage, path: Path) -> None:
    out = QImage(a.width() + b.width(), max(a.height(), b.height()), QImage.Format_RGBA8888)
    out.fill(Qt.black)
    p = QPainter(out)
    p.drawImage(0, 0, a)
    p.drawImage(a.width(), 0, b)
    p.setPen(Qt.white)
    p.drawText(12, 24, "ParamMouthOpenY = 0.0")
    p.drawText(a.width() + 12, 24, "ParamMouthOpenY = 1.0")
    p.end()
    out.save(str(path))


def _report(results: list[tuple[str, bool, str]]) -> int:
    print("\n=== P0 スパイク 自己検証 ===")
    fail = 0
    for name, ok, note in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}\n        {note}")
        fail += 0 if ok else 1
    print(f"\n  出力: {OUT_DIR}")
    print(f"  合計 PASS {len(results) - fail} / FAIL {fail}")
    return 1 if fail else 0


# ---------------------------------------------------------------- main

def main() -> int:
    for p in (MODEL_JSON, BG_INSIDE, BG_FOREGROUND):
        if not p.exists():
            print(f"素材が見つかりません: {p}", file=sys.stderr)
            return 2
    if "--lipsync-check" in sys.argv:  # GL も Live2D も要らない（音声解析だけ）
        i = sys.argv.index("--lipsync-check")
        text = sys.argv[i + 1] if len(sys.argv) > i + 1 else "こんにちは。今日はいい天気だね。"
        return lipsync_check(text)
    live2d.init()
    try:
        if "--selftest" in sys.argv:
            return selftest()
        app = QApplication(sys.argv)
        win = MainWindow()
        win.resize(960, 780)
        win.show()
        if "--demo" in sys.argv:      # 自動で5文しゃべる（--demo-loop で繰り返す）
            win.start_demo(loop="--demo-loop" in sys.argv)
        elif "--demo-loop" in sys.argv:
            win.start_demo(loop=True)
        # 自動終了（スモーク用）: SPIKE_AUTOCLOSE_SEC=3 で3秒後に閉じる
        if os.environ.get("SPIKE_AUTOCLOSE_SEC"):
            QTimer.singleShot(int(float(os.environ["SPIKE_AUTOCLOSE_SEC"]) * 1000), win.close)
        return app.exec()
    finally:
        live2d.dispose()


if __name__ == "__main__":
    sys.exit(main())
