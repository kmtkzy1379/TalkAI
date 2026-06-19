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
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Optional

from ..context_assembler import ContextAssembler, RagChunk, Turn
from ..pipeline.audio_play_queue import AudioPlayQueue
from ..pipeline.stimulus import Stimulus, StimulusKind
from .splitter import JapaneseSentenceSplitter
from .style import SPEECH_STYLE, sanitize_for_speech

if TYPE_CHECKING:  # 実行時の循環 import を避ける（型注釈のみ）
    from ..memory.conversation_cache import ConversationCache
    from ..memory.long_term import RagStore

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
        conversation_cache: Optional["ConversationCache"] = None,
        rag_store: Optional["RagStore"] = None,
    ) -> None:
        self._audio = audio
        self._stream_fn = stream_fn
        self._tts_fn = tts_fn
        self._ctx = context_assembler or ContextAssembler(system_prompt=SPEECH_STYLE)
        self._tts_concurrency = tts_concurrency
        # 任意注入: None なら短期記憶を使わない（F2 系テストは従来どおりの経路）。
        self._cache = conversation_cache
        # 任意注入: 長期記憶(連想RAG)。None なら使わない。
        self._rag = rag_store
        self.last_response = ""  # 生成済み全文（自然さの目視・テスト用。記憶には使わない＝C5）

    def _build_messages(
        self,
        stimulus: Stimulus,
        recent_turns: Optional[list[Turn]] = None,
        rag_chunks: Optional[list[RagChunk]] = None,
    ) -> list[dict]:
        ctx = self._ctx.assemble(
            user_text=str(stimulus.payload),
            recent_turns=recent_turns,
            rag_chunks=rag_chunks,
        )
        messages: list[dict] = []
        if ctx.system:
            messages.append({"role": "system", "content": ctx.system})
        messages.append({"role": "user", "content": ctx.render()})
        return messages

    async def handle(self, stimulus: Stimulus) -> None:
        gen = self._audio.current_generation()
        # 記憶: 現ターンを記録する**前**に直近会話をスナップショット（現発話が二重表示されない）。
        recent = self._cache.recent_for_injection() if self._cache is not None else None
        # 長期記憶: 今の発話に連想する過去記憶を取得（クエリ埋め込みはここで・≤3s 内）。
        rag_chunks = None
        if self._rag is not None:
            try:
                rag_chunks = await self._rag.search(str(stimulus.payload))
            except Exception:
                logger.exception("RAG 検索に失敗（記憶なしで継続）")
        messages = self._build_messages(stimulus, recent, rag_chunks)
        # ユーザ発話なら user ターンを記録（自律/vision/callfunction には user ターンは無い）。
        if self._cache is not None and stimulus.kind == StimulusKind.USER_UTTERANCE:
            self._cache.add_turn("user", str(stimulus.payload))
        splitter = JapaneseSentenceSplitter()
        sem = asyncio.Semaphore(self._tts_concurrency)
        tasks: list[asyncio.Task] = []
        parts: list[str] = []
        spoken: list[str] = []  # C5: 実際に再生し終えた文だけが入る（生成≠発話）
        eve_recorded = False

        def _record_eve() -> None:
            nonlocal eve_recorded
            if eve_recorded or self._cache is None:
                return
            eve_recorded = True
            self._cache.add_turn("eve", "".join(spoken))  # 空なら add_turn 側で無視

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
                # on_played は「実際に再生し終えた時」だけ呼ばれる → spoken に積む（C5）。
                self._audio.enqueue(gen, seq, wav, text=sentence, on_played=spoken.append)

        def _emit(sentence: str) -> None:
            clean = sanitize_for_speech(sentence)  # 残留マークダウン除去（コードゲート）
            if not clean:
                return  # 記号だけの行は読み上げない
            logger.info("🤖 %s", clean)  # 文ごとに表示（ストリーミング表示）
            parts.append(clean)
            seq = self._audio.reserve_seq(gen)  # stream 順に予約（再生は seq 昇順）
            tasks.append(asyncio.create_task(_tts_and_enqueue(seq, clean)))

        try:
            try:
                async for delta in self._stream_fn(messages):
                    if self._audio.current_generation() != gen:
                        break  # barge-in(世代変化): 生成停止（残りの文は出さない）
                    for sentence in splitter.feed(delta):
                        _emit(sentence)
                else:
                    # 正常終了時のみ末尾を flush（barge-in/エラー中断時は出さない）
                    for sentence in splitter.flush():
                        _emit(sentence)
            except Exception:
                # A3: LLM/stream の一時エラーは起こりうる → ログして途中までで打ち切り、継続。
                logger.exception("応答生成中にエラー（途中までで打ち切り）")
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.last_response = "".join(parts)
            if self._cache is not None:
                # 現ターンの音声が再生し切るまで待ってから「実発話」を記録（C5）。
                await self._audio.join()
                _record_eve()
        except asyncio.CancelledError:
            # barge-in: ここまでに**実際に喋った分だけ**を記憶に記録（生成途中の文は残さない）。
            _record_eve()
            for t in tasks:  # 進行中の TTS タスクも片付ける（孤児化防止）
                if not t.done():
                    t.cancel()
            raise
