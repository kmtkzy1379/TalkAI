r"""自律発話(沈黙nudge)が inc2 後も生きているか単体確認（他機能 非回帰チェック）。

RAG に話題の種を入れ、1回だけ普通の雑談ターン→以後 沈黙。decider(沈黙監視)が自律的に一言
出すかを見る。inc2 は応答/decider 経路を触っていないので「壊れていない」ことの実証用。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\autonomous_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.feedback import PredictionState  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.speech import SilenceMonitor, SpeechDecider, SpeechState, make_decide_fn  # noqa: E402

T0 = time.monotonic()


def stamp():
    return f"T+{time.monotonic() - T0:5.1f}s"


class CapturePlayer:
    async def play_fn(self, audio, should_stop=None):
        if audio:
            await asyncio.sleep(0.05)


async def _tts(s):
    return b"x" * 4000


async def main():
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5"})
    art = tempfile.mkdtemp(prefix="auto_")
    cache = ConversationCache(history_file=os.path.join(art, "h.jsonl")); await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(art, "r.jsonl")); await rag.warmup()
    await rag.add_chunk(text="ユーザはチーズケーキが大好き", search_text="チーズケーキ 甘いもの", prediction_diff=60)
    pred = PredictionState(); state = SpeechState(); queue = StimulusQueue()
    audio = AudioPlayQueue(play_fn=CapturePlayer().play_fn)

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        async for x in reg.stream("response", messages, max_tokens=120):
            yield x

    orch = ResponseOrchestrator(audio, stream_fn, _tts, ContextAssembler(system_prompt=SPEECH_STYLE),
                                conversation_cache=cache, rag_store=rag, prediction_state=pred,
                                on_response_complete=lambda: state.mark_eve_activity())
    _handle = orch.handle

    async def handle_wrap(stim):
        await _handle(stim)
        kind = "自発" if stim.kind == StimulusKind.AUTONOMOUS_SPEECH else "応答"
        print(f"  {stamp()} 🤖[{kind}] {orch.last_response}")
    orch.handle = handle_wrap  # type: ignore

    runner = PipelineRunner(queue, orch, audio)
    _decide = make_decide_fn(reg)

    async def decide_logged(**kw):
        d = await _decide(**kw)
        print(f"  {stamp()} …decider: should_speak={getattr(d, 'should_speak', '?')} "
              f"silence={kw.get('silence_seconds'):.0f}s reason={getattr(d, 'reason', '')[:50]}")
        return d
    decider = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred,
                            queue=queue, decide_fn=decide_logged)
    silence = SilenceMonitor(state=state, decider=decider, is_busy_fn=runner.is_busy,
                             tick_sec=1.0, threshold_sec=5.0)
    pt = asyncio.create_task(audio.play_worker()); rt = asyncio.create_task(runner.run())
    decider.start()
    from eve.stt import make_stt
    stt = make_stt(); await stt.warmup()

    # 1回だけ雑談ターン
    state.mark_user_speech_start(); state.mark_user_utterance()
    print(f"  {stamp()} 🧑 最近ちょっと疲れてて、甘いものでも食べたい気分なんだよね")
    await queue.put(Stimulus(StimulusKind.USER_UTTERANCE, "最近ちょっと疲れてて、甘いものでも食べたい気分なんだよね"))
    t = time.monotonic()
    while runner.is_busy() or queue.qsize() > 0:
        await asyncio.sleep(0.1)
        if time.monotonic() - t > 20:
            break

    # 以後 沈黙。自律発話が出るか 25秒観測。
    print(f"  {stamp()} … 沈黙開始（自律発話を待つ・threshold 5s）")
    silence.start()
    await asyncio.sleep(25.0)

    await silence.stop(); await decider.stop()
    rt.cancel(); pt.cancel(); await asyncio.sleep(0.2)
    await rag.shutdown(); await cache.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
