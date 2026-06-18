"""ResponseOrchestrator — 応答の背骨（pipeline.Orchestrator Protocol を実装）。

刺激 → ContextAssembler で messages → stream_fn で token stream → SentenceSplitter で文 →
文ごとに seq 予約 + TTS(並行,上限) → AudioPlayQueue へ enqueue（seq 再整列）。
PipelineRunner には StubOrchestrator の代わりにこれを注入する。

barge-in（応答側）: stream 中に世代が変わったら生成停止。TTS 完了時にも世代を確認して
古い世代の音声は enqueue しない（AudioPlayQueue 側でも世代で破棄される＝二重の安全）。

依存性注入（stream_fn / tts_fn）で実 API とテスト用フェイクを切替（Tier-1 は実依存ゼロ）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Awaitable, Callable, Optional

from ..context_assembler import ContextAssembler
from ..pipeline.audio_play_queue import AudioPlayQueue
from ..pipeline.stimulus import Stimulus
from .splitter import JapaneseSentenceSplitter
from .style import SPEECH_STYLE, sanitize_for_speech

logger = logging.getLogger(__name__)

StreamFn = Callable[[list[dict]], AsyncIterator[str]]  # messages -> 文字列デルタの async iter
TtsFn = Callable[[str], Awaitable[Optional[bytes]]]  # 文 -> 音声バイト(or None)


class ResponseOrchestrator:
    def __init__(
        self,
        audio: AudioPlayQueue,
        stream_fn: StreamFn,
        tts_fn: TtsFn,
        context_assembler: Optional[ContextAssembler] = None,
        tts_concurrency: int = 3,
    ) -> None:
        self._audio = audio
        self._stream_fn = stream_fn
        self._tts_fn = tts_fn
        self._ctx = context_assembler or ContextAssembler(system_prompt=SPEECH_STYLE)
        self._tts_concurrency = tts_concurrency
        self.last_response = ""  # 自然さの目視・テスト用

    def _build_messages(self, stimulus: Stimulus) -> list[dict]:
        ctx = self._ctx.assemble(user_text=str(stimulus.payload))
        messages: list[dict] = []
        if ctx.system:
            messages.append({"role": "system", "content": ctx.system})
        messages.append({"role": "user", "content": ctx.render()})
        return messages

    async def handle(self, stimulus: Stimulus) -> None:
        gen = self._audio.current_generation()
        messages = self._build_messages(stimulus)
        splitter = JapaneseSentenceSplitter()
        sem = asyncio.Semaphore(self._tts_concurrency)
        tasks: list[asyncio.Task] = []
        parts: list[str] = []

        async def _tts_and_enqueue(seq: int, sentence: str) -> None:
            # A1: 同一世代である限り、TTS が None/例外でも **必ず seq を埋める**。
            # 埋めないと AudioPlayQueue._drain_buffer が連続 seq で永久停止
            # （= head-of-line デッドロック / 1文目失敗でターン丸ごと無音）。
            wav = None
            try:
                async with sem:
                    if self._audio.current_generation() != gen:
                        return  # barge-in 済み: 起こさない（古い世代は破棄される）
                    wav = await self._tts_fn(sentence)
            except Exception:
                logger.exception("TTS 失敗（文をスキップして継続）: %.20s", sentence)
                wav = None
            if wav is None:
                logger.warning("TTS が音声を返さず（文をスキップ）: %.20s", sentence)
            if self._audio.current_generation() == gen:
                self._audio.enqueue(gen, seq, wav)  # wav=None は番兵（再生時スキップ）

        def _emit(sentence: str) -> None:
            spoken = sanitize_for_speech(sentence)  # 残留マークダウン除去（コードゲート）
            if not spoken:
                return  # 記号だけの行は読み上げない
            parts.append(spoken)
            seq = self._audio.reserve_seq(gen)  # stream 順に予約（再生は seq 昇順）
            tasks.append(asyncio.create_task(_tts_and_enqueue(seq, spoken)))

        try:
            async for delta in self._stream_fn(messages):
                if self._audio.current_generation() != gen:
                    break  # barge-in: 生成停止（残りの文は出さない）
                for sentence in splitter.feed(delta):
                    _emit(sentence)
            else:
                # 正常終了時のみ末尾を flush（barge-in/エラー中断時は出さない）
                for sentence in splitter.flush():
                    _emit(sentence)
        except Exception:
            # A3: LLM/stream の一時エラーは起こりやすい → ログして途中までで打ち切り、継続。
            logger.exception("応答生成中にエラー（途中までで打ち切り）")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.last_response = "".join(parts)
