"""VOICEVOX TTS クライアント（PORT: v1 modules/tts.py）。

audio_query → synthesis の2段。ClientSession を使い回す。aiohttp は遅延 import
（Tier-1 決定論テストは tts_fn を注入するため本モジュールを読み込まない）。
"""
from __future__ import annotations

import logging
from typing import Optional

from ..config import Config

logger = logging.getLogger(__name__)


class VoicevoxTTS:
    def __init__(
        self,
        base_url: Optional[str] = None,
        speaker: Optional[int] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> None:
        self.base_url = base_url or Config.VOICEVOX_URL
        self.speaker = Config.VV_SPEAKER if speaker is None else speaker
        self.speed = Config.VV_SPEED if speed is None else speed
        self.pitch = Config.VV_PITCH if pitch is None else pitch
        self._session = None

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(self, text: str) -> Optional[bytes]:
        """文 → WAV バイト。失敗時は None（落とさない）。

        **公開契約は変えない**（bytes を返す）。`tools/` の多数のスクリプトがこの形に依存している。
        リップシンク付きが要る呼び出し側は `generate_with_meta()` を使う。
        """
        wav, _ = await self._generate(text)
        return wav

    async def generate_with_meta(self, text: str) -> tuple[Optional[bytes], Optional[list]]:
        """文 → `(WAV バイト, 口の開閉計画)`。UI のリップシンク用（本番 VoiceLoop はこちらを使う）。

        `audio_query` は元々この関数の中で取得して**捨てていた**ので、**HTTP 往復は増えない**。
        返す計画は float タプルのリスト（`eve/response/lipsync.py`）＝スレッドを跨いで渡しても
        loop 所有オブジェクトへの参照が漏れない。音声が出せない時は `(None, None)`。
        """
        wav, query = await self._generate(text)
        if wav is None:
            return None, None
        from .lipsync import plan_from_query_safe

        return wav, plan_from_query_safe(query)

    async def _generate(self, text: str) -> tuple[Optional[bytes], Optional[dict]]:
        """audio_query → synthesis の本体。`(WAV, 合成に使った query)` を返す。"""
        if not text or not text.strip():
            return None, None
        try:
            session = await self._get_session()
            params = {"text": text, "speaker": self.speaker}
            async with session.post(f"{self.base_url}/audio_query", params=params) as resp:
                if resp.status != 200:
                    return None, None
                query = await resp.json()
            query["speedScale"] = self.speed
            query["pitchScale"] = self.pitch
            async with session.post(
                f"{self.base_url}/synthesis", json=query, params={"speaker": self.speaker}
            ) as resp:
                if resp.status != 200:
                    return None, None
                # query は speedScale 適用後のものを返す（口の開閉計画は速度込みで作る）
                return await resp.read(), query
        except Exception as e:  # 起こりやすい一時失敗 → ログして None（呼び出し側が文をスキップ）
            logger.warning("TTS 失敗（None を返す）: %s", e)
            return None, None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
