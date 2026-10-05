"""実音声再生（PORT: v1 modules/player.py の _play_audio）。

AudioPlayQueue に注入する async な `play_fn` を提供する。pyaudio/soundfile/numpy は
遅延 import（Tier-1 決定論テストでは fake play_fn を使うため読み込まれない）。

割り込み（barge-in）では鳴っている項目を止めない＝言い切る（ユーザ裁定 2026-09-22「、。までは喋らせる」）。
次の項目を鳴らさないのは AudioPlayQueue の世代比較の担当。ここで途中で止めるのはミュートだけ。

**出力は開きっぱなし**（ユーザ 2026-09-26「開きっぱなしがいい」）。以前は項目ごとに開いて閉じており、
開くのに約0.2秒・出力の遅れ約0.09秒かかっていた（実測 2026-09-26・Windows の既定の出力）。
その分だけ毎回1回目の音が遅れ、区切りの間も空いていた。開き直すのはサンプルレートが変わった時と、
書き込みに失敗した時（出力機器が外れた等）だけ。区切りの間は AudioPlayQueue の `item_gap_sec` が作る。
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

from ..clock import now_mono

logger = logging.getLogger(__name__)


class RealAudioPlayer:
    """実再生。**ミュートはここだけで完結させる**（UI「ミュート」用）。

    なぜ再生器で止めるか: 上位（AudioPlayQueue の世代/seq）で止めると barge-in と同じ
    経路になり、進行中ターンの残りの文が捨てられる。ユーザ裁定は「速攻黙って、**処理は
    正常に終了させて**から Pause」なので、止めるのは音だけでよい。seq/generation・
    再生開始/終了フック・会話記憶はすべて通常どおり進む（ログには発話が残る＝読める）。
    """

    def __init__(self, channels: int = 1) -> None:
        self._pa = None
        self._channels = channels
        self.muted = False  # 単一書込（loop 上の ControlPlane のみ）・読みは再生中の同期参照
        self._stream = None
        self._rate: Optional[int] = None
        # 1本の出力を応答とフィラーで共有するので、同時に2つは書かない（設計上は重ならない＝保険）
        self._lock: Optional[asyncio.Lock] = None
        # 最後に音を書き終えた時刻（monotonic・0=まだ鳴らしていない）。区切りの間を測る基準
        self.last_end = 0.0

    def _pyaudio(self):
        if self._pa is None:
            import pyaudio

            self._pa = pyaudio.PyAudio()
        return self._pa

    def _open(self, rate: int):
        """出力を用意する（同じレートで開いていればそのまま使う）。ブロッキング＝executor で呼ぶ。"""
        if self._stream is not None and self._rate == rate:
            return self._stream
        import pyaudio

        self._close_stream()
        self._stream = self._pyaudio().open(
            format=pyaudio.paFloat32, channels=self._channels, rate=rate, output=True
        )
        self._rate = rate
        return self._stream

    def _close_stream(self) -> None:
        st, self._stream, self._rate = self._stream, None, None
        if st is None:
            return
        try:
            st.stop_stream()
            st.close()
        except Exception:
            logger.debug("出力を閉じる時に例外（無視）", exc_info=True)

    def prepare(self, rate: int) -> None:
        """起動時に出力を開いておく（1回目の応答で開く約0.2秒を払わない）。ブロッキング。"""
        self._open(int(rate))

    async def play_fn(self, audio: Optional[bytes]) -> None:
        if not audio:
            return
        if self.muted:
            return  # 音は出さない（デバイスも開かない）。呼び出し側から見れば「再生し終えた」
        import soundfile as sf

        with io.BytesIO(audio) as bio:
            data, samplerate = sf.read(bio, dtype="float32")
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        chunk = 1024
        async with self._lock:
            wrote = False
            try:
                stream = await loop.run_in_executor(None, self._open, int(samplerate))
                for i in range(0, len(data), chunk):
                    if self.muted:
                        break  # 再生中にミュートされた → 文の途中でも即黙る（チャンク粒度で数十ms以内）
                    buf = data[i : i + chunk].tobytes()
                    await loop.run_in_executor(None, stream.write, buf)
                    wrote = True
            except asyncio.CancelledError:
                raise
            except Exception:
                # 書けなくなった（出力機器が外れた等）: 閉じて次の項目で開き直す
                self._close_stream()
                raise
            finally:
                if wrote:
                    self.last_end = now_mono()

    def close(self) -> None:
        self._close_stream()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
