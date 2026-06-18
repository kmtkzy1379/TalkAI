"""実音声再生（PORT: v1 modules/player.py の _play_audio）。

AudioPlayQueue に注入する async な `play_fn` を提供する。pyaudio/soundfile/numpy は
遅延 import（Tier-1 決定論テストでは fake play_fn を使うため読み込まれない）。

注: 文の途中で止めるチャンク単位の割り込みは F2.5 以降。F2 の barge-in は
AudioPlayQueue の世代比較（項目単位）で「次の文を再生しない」までを担う。
"""
from __future__ import annotations

import asyncio
import io
from typing import Optional


class RealAudioPlayer:
    def __init__(self, channels: int = 1) -> None:
        self._pa = None
        self._channels = channels

    def _pyaudio(self):
        if self._pa is None:
            import pyaudio

            self._pa = pyaudio.PyAudio()
        return self._pa

    async def play_fn(self, audio: Optional[bytes], should_stop=None) -> None:
        if not audio:
            return
        import pyaudio
        import soundfile as sf

        with io.BytesIO(audio) as bio:
            data, samplerate = sf.read(bio, dtype="float32")
        pa = self._pyaudio()
        stream = pa.open(
            format=pyaudio.paFloat32, channels=self._channels, rate=samplerate, output=True
        )
        loop = asyncio.get_running_loop()
        chunk = 1024
        try:
            for i in range(0, len(data), chunk):
                if should_stop is not None and should_stop():
                    break  # B3: barge-in → 文の途中でも即停止（チャンク粒度で数十ms以内）
                buf = data[i : i + chunk].tobytes()
                await loop.run_in_executor(None, stream.write, buf)
        finally:
            stream.stop_stream()
            stream.close()

    def close(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
