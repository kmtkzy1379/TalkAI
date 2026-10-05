"""int16 モノラル PCM のサンプルレート変換（チャンクをまたいで状態を持つ・numpy のみ）。

なぜ要るか: マイクは 16kHz（`Config.SAMPLE_RATE`）だが、gpt-live-transcribe は
`rate` に 24000 未満を受け付けない（実測 2026-09-25: rate=16000 を送ると
"Expected a value >= 24000, but got 16000 instead." で session.update ごと拒否される）。

方式: 有理数比 L/M のポリフェーズ（L 倍にゼロ詰め → 低域通過 FIR → M 分の1 に間引き）。
チャンクごとに独立に変換すると継ぎ目で波形が切れるので、FIR の履歴と間引きの位相を持ち越す。
発話ごとに `reset()` する（発話と発話の間は時間が飛んでいるので履歴を繋がない）。
"""
from __future__ import annotations

from math import gcd

import numpy as np


class StreamResampler:
    def __init__(self, src_rate: int, dst_rate: int, taps_per_phase: int = 16) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"サンプルレートが不正: {src_rate} → {dst_rate}")
        g = gcd(int(src_rate), int(dst_rate))
        self.up = int(dst_rate) // g    # L
        self.down = int(src_rate) // g  # M
        self.passthrough = self.up == 1 and self.down == 1
        if self.passthrough:
            self._h = None
            self._n = 0
        else:
            m = max(self.up, self.down)
            n = 2 * taps_per_phase * m + 1  # 奇数長（遅れが整数サンプル）
            # 遮断は低い側のナイキストの 9 割（ゼロ詰めの折り返し像を落としつつ声の帯域は残す）
            fc = 0.9 * 0.5 / m  # 中間レート（src*L）での周波数（cycles/sample）
            k = np.arange(n) - (n - 1) / 2.0
            h = 2.0 * fc * np.sinc(2.0 * fc * k) * np.kaiser(n, 8.0)
            h *= self.up / h.sum()  # 直流利得を L に（ゼロ詰めで L 分の1 に目減りした分を戻す）
            self._h = h
            self._n = n
        self.reset()

    def reset(self) -> None:
        """発話の頭で呼ぶ（前の発話の履歴を持ち込まない）。"""
        self._hist = np.zeros(max(0, self._n - 1), dtype=np.float64)
        self._phase = 0

    def process(self, pcm: bytes) -> bytes:
        if self.passthrough or not pcm:
            return pcm
        x = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.float64)
        if x.size == 0:
            return b""
        up = np.zeros(x.size * self.up, dtype=np.float64)
        up[:: self.up] = x
        buf = np.concatenate([self._hist, up])
        y = np.convolve(buf, self._h, mode="valid")  # 長さ = len(up)
        self._hist = buf[len(buf) - (self._n - 1):]
        out = y[self._phase:: self.down]
        # 次のチャンクで最初に取る位置（このチャンクで取った最後の次の位置を持ち越す）
        self._phase = (self._phase - up.size) % self.down
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()
