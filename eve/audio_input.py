"""マイク入力 + Silero VAD（v1 modules/audio_input.py を PORT・整理）。

挙動はユーザ満足の v1 を踏襲: チャンクごとに Silero VAD で発話確率→閾値判定、発話中バッファ、
無音 SILENCE_LIMIT 秒で終端、最小0.3秒ゲートを通った発話だけ speech_queue に流す。
＝ endpointing ＋「非音声を STT に渡さない源ゲート」。声の判別はしない（イヤホン前提）。

torch / pyaudio は遅延 import（マイクを使う時だけ要る）。発話開始で callback_on_speech_start
を呼ぶ（barge-in の発火点。Eve が譲るトリガに使う）。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable, Optional

from .config import Config

logger = logging.getLogger(__name__)


def _load_silero_vad():
    """Silero VAD JIT を torch.hub キャッシュから直接ロード（torchaudio DLL 問題回避）。"""
    import torch

    hub_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub",
        "snakers4_silero-vad_master", "src", "silero_vad", "data",
    )
    jit_path = os.path.join(hub_cache, "silero_vad.jit")
    if not os.path.isfile(jit_path):
        logger.info("Silero VAD キャッシュ無し。torch.hub で取得します...")
        torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True, onnx=False)
        if not os.path.isfile(jit_path):
            raise FileNotFoundError(f"Silero VAD JIT not found: {jit_path}")
    torch.set_num_threads(1)
    model = torch.jit.load(jit_path)
    model.eval()
    return model


class AudioInput:
    def __init__(self, callback_on_speech_start: Optional[Callable] = None) -> None:
        import pyaudio

        self.chunk = 512
        self.format = pyaudio.paInt16
        self.channels = Config.CHANNELS
        self.rate = Config.SAMPLE_RATE
        self._pa = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        # 直近チャンクの VAD 発話確率（0-1）。**UI のマイクレベル表示専用**で、
        # 判定には一切使わない（書くのは `_process_loop` だけ＝loop 上の単一書込）。
        self.speech_prob = 0.0
        self.callback_on_speech_start = callback_on_speech_start
        self.speech_queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        logger.info("Silero VAD ロード中...")
        self.model = _load_silero_vad()
        logger.info("Silero VAD ロード完了")

    def start(self) -> None:
        import pyaudio  # noqa: F401  (format 定数は __init__ で取得済)

        last_err = None
        for attempt in range(3):  # デバイスビジーの一時失敗をリトライ
            try:
                self.stream = self._pa.open(
                    format=self.format, channels=self.channels, rate=self.rate,
                    input=True, frames_per_buffer=self.chunk,
                )
                break
            except OSError as e:
                last_err = e
                logger.warning("PyAudio open 失敗 %d回目: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(0.5)
        else:
            raise OSError(f"音声ストリームを3回開けず: {last_err}") from last_err
        self.running = True
        asyncio.create_task(self._process_loop())

    def stop(self) -> None:
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self._pa.terminate()

    async def get_audio(self) -> bytes:
        return await self.speech_queue.get()

    async def _process_loop(self) -> None:
        from collections import deque

        import numpy as np
        import torch

        logger.info("マイク監視を開始（バックグラウンド）")
        audio_buffer: list[bytes] = []
        is_speaking = False
        silence_start = None
        # pre-roll: 発話検知の直前チャンクを溜め、開始時に先頭へ継ぎ足す（頭欠け防止）
        preroll_chunks = max(1, int(Config.PREROLL_SEC * self.rate / self.chunk))
        preroll: "deque[bytes]" = deque(maxlen=preroll_chunks)
        # onset 確認: これだけ連続して発話が続いてから「発話開始」とする（単発スパイク無視）
        onset_chunks = max(1, int(Config.VAD_ONSET_SEC * self.rate / self.chunk))
        speech_run = 0
        # 音量ゲート（`Config.MIC_MIN_PEAK`）。**まとまりの最大振幅**で見る:
        # run_peak = onset 窓の最大（割り込み発火の判定）/ utt_peak = 発話全体の最大（STT 送りの判定）。
        # チャンクごとに切ると語尾や語間で落ちて実発話まで捨てるので、必ず max を使う。
        min_peak = float(Config.MIC_MIN_PEAK)
        run_peak = 0.0
        utt_peak = 0.0
        loop = asyncio.get_running_loop()

        while self.running:
            try:
                data = await loop.run_in_executor(
                    None, lambda: self.stream.read(self.chunk, exception_on_overflow=False)
                )
            except Exception as e:  # マイクの一時エラーは継続（起こりうる）
                logger.warning("Mic read エラー: %s", e)
                await asyncio.sleep(0.1)
                continue

            audio_f32 = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
            speech_prob = self.model(torch.from_numpy(audio_f32), self.rate).item()
            self.speech_prob = speech_prob  # UI 表示専用（判定はこの下の既存ロジックのまま）

            peak = float(np.abs(audio_f32).max())

            if speech_prob > Config.VAD_THRESHOLD:
                speech_run += 1
                run_peak = max(run_peak, peak)
                if is_speaking:
                    utt_peak = max(utt_peak, peak)
                    silence_start = None
                    audio_buffer.append(data)
                else:
                    preroll.append(data)  # 確定前は pre-roll に溜める
                    # 単発スパイク(クリック/打鍵)を無視: 連続して発話が続いた時だけ開始。
                    # 加えて音量ゲート: 漏れ音のような極小音では割り込まない。ここで見送っても
                    # speech_run は伸び続けるので、声が大きくなった次のチャンクで発火する
                    # （＝小声の出だしでも数十ms遅れるだけで取りこぼさない）。
                    if speech_run >= onset_chunks and run_peak >= min_peak:
                        is_speaking = True
                        utt_peak = run_peak
                        logger.debug("発話開始を確定")
                        audio_buffer = list(preroll)  # 確認チャンク+直前を先頭に（頭欠け防止）
                        preroll.clear()
                        silence_start = None
                        cb = self.callback_on_speech_start
                        if cb:
                            if asyncio.iscoroutinefunction(cb):
                                asyncio.create_task(cb())
                            else:
                                cb()
            else:
                speech_run = 0
                run_peak = 0.0
                if is_speaking:
                    utt_peak = max(utt_peak, peak)
                    audio_buffer.append(data)
                    if silence_start is None:
                        silence_start = time.time()
                    if time.time() - silence_start > Config.SILENCE_LIMIT:
                        is_speaking = False
                        full_audio = b"".join(audio_buffer)
                        audio_buffer = []
                        reached = utt_peak  # ログ用に退避してからリセットする
                        utt_peak = 0.0
                        if len(full_audio) > self.rate * 2 * 0.3:  # 最小0.3秒ゲート
                            if reached >= min_peak:
                                self.speech_queue.put_nowait(full_audio)
                            else:
                                # **黙って捨てない**: 「喋ったのに無視された」を後から追えるようにする。
                                # ここが続くなら MIC_MIN_PEAK が高すぎる。
                                logger.info(
                                    "音量ゲートで破棄（最大振幅 %.4f < %.4f）。"
                                    "自分の声が拾われない場合は MIC_MIN_PEAK を下げてください",
                                    reached, min_peak)
                else:
                    preroll.append(data)  # 非発話中は直近を pre-roll に溜める

            await asyncio.sleep(0)
