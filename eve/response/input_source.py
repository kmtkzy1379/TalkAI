"""入力源の抽象（STT の差込口）。

- TextInputSource: テキスト発話を USER_UTTERANCE として投入（テスト/タイプ入力）。
- MicSttInputSource: mic → VAD → STT → USER_UTTERANCE。発話開始で barge-in（Eveが譲る）。

いずれも asyncio ループ内で動く（mic のブロッキング read のみ executor）。よって
StimulusQueue.put を直接 await でき、別スレッド用の thread-safe put は不要。
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from ..pipeline.stimulus import Stimulus, StimulusKind
from ..pipeline.stimulus_queue import StimulusQueue
from ..stt import Stt
from ..stt.askback import UNINTELLIGIBLE_PLACEHOLDER, is_unintelligible

logger = logging.getLogger(__name__)


class InputSource(ABC):
    def __init__(self, queue: StimulusQueue) -> None:
        self._queue = queue

    @abstractmethod
    async def start(self) -> None:
        """取り込みを開始する（mic 版はタスクを起動して即 return）。"""
        ...

    def user_in_speech(self) -> bool:
        """ユーザが今まさに話しているか（発話区間の中か）。マイクの無い入力源は常に False。"""
        return False


class TextInputSource(InputSource):
    async def submit(self, text: str) -> None:
        await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))

    async def start(self) -> None:
        return  # テキスト版は外部から submit() を呼ぶだけ


class MicSttInputSource(InputSource):
    """mic → VAD → STT → 刺激。発話開始で on_speech_start（barge-in＝Eveが即譲る）を呼ぶ。

    on_speech_start には「音声停止＋進行中応答キャンセル」をまとめた barge-in を渡す
    （VoiceLoop が audio.interrupt + runner.interrupt を束ねて渡す）。
    """

    def __init__(self, queue: StimulusQueue, stt: Stt, on_speech_start=None, on_utterance=None,
                 on_stt_start=None, on_stt_end=None, on_speech_end=None, on_stt_result=None,
                 on_phrase=None) -> None:
        super().__init__(queue)
        self._stt = stt
        # S0: 話している間に音声を送る方式か（True なら区間フックで start/feed/commit/clear を呼ぶ）
        self._streaming = bool(getattr(stt, "streaming", False))
        self._on_speech_start = on_speech_start
        # S3: VAD が発話の終わりを確定した瞬間（採用した区間のみ）。`fn(speech_sec, seg)`。
        # seg = 区間の番号（発話開始の確定ごとに +1・S6 笑い判定の結果と対応づける）。
        # フィラーの発火点。`on_utterance` は STT を待つ消費側で呼ぶので、前の発話の STT を
        # 待っている間は遅れる＝「発話終了の瞬間」にはこちらを使う。
        self._on_speech_end = on_speech_end
        # S3: STT の結果を刺激にしたか（`fn(put: bool)`）。False＝空認識・失敗
        # ＝応答が来ないのでフィラーを止める合図。`on_stt_end` の直前に呼ぶ。
        self._on_stt_result = on_stt_result
        # S6 笑い: 途中の文字（「、」「。」ごと・確定時）を `fn(seg, text)` で受ける。ストリーミングの時だけ。
        self._seg_no = 0
        if on_phrase is not None and self._streaming:
            stt.on_phrase = on_phrase
        # 発話セグメント到着（＝ユーザが話し終えた）時に呼ぶ。F5 の沈黙時計リセット用。
        self._on_utterance = on_utterance
        # J-2 ③-A: STT 待ち窓（発話終了〜テキスト投入）の可視化フック。この窓は user_speaking=False
        # かつ沈黙時計リセット済みで「静か」に見え、自発発話が差し込まれてユーザ応答を遅らせた
        # （E2E S8 実測）。start=transcribe 直前 / end=投入完了（失敗・空認識含む）後に呼ぶ。
        self._on_stt_start = on_stt_start
        self._on_stt_end = on_stt_end
        self._ai = None
        self._task = None

    async def start(self) -> None:
        # **冪等にする**: 二重 ON で PyAudio ストリームが2本になり、古い `_consume` が
        # 参照を失って孤児化する（旧 `speech_queue` を永久に待つ）。UI の停止/再開は
        # 何度も押されるので、守らないと確率的に必ず踏む。
        # 他のサイドカーは `if self._task is None` ガードを持っているが、ここだけ無かった
        # （`_stopping` を持たないため前提工事の構造テストの網からも漏れていた）。
        if self._ai is not None or self._task is not None:
            return
        from ..audio_input import AudioInput  # torch/pyaudio を要する→遅延 import

        if self._streaming:
            self._stt.reset()  # 前回の停止で取り残した送りかけ・待ち札を持ち越さない
        # 発話開始の瞬間に barge-in（STT 完了を待たず即停止＝自然な割り込み）
        self._ai = AudioInput(
            callback_on_speech_start=self._on_speech_start,
            on_segment_start=self._segment_start,
            on_segment_audio=self._segment_audio,
            on_segment_end=self._segment_end,
        )
        self._ai.start()
        self._task = asyncio.create_task(self._consume())

    def user_in_speech(self) -> bool:
        """VAD の発話区間の中か（発話開始の確定〜`SILENCE_LIMIT` の無音まで）。止まっていれば False。"""
        return self._ai is not None and self._ai.in_speech()

    def _hook(self, fn, name: str, *args) -> None:
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            logger.exception("%s フックで例外（無視して継続）", name)

    # --- マイクの区間フック（loop 上で同期に呼ばれる・例外は AudioInput 側でも握られる）---

    def _segment_start(self, pcm: bytes) -> None:
        self._seg_no += 1
        if self._streaming:
            self._stt.start(pcm, tag=self._seg_no)

    def _segment_audio(self, pcm: bytes) -> None:
        if self._streaming:
            self._stt.feed(pcm)

    def _segment_end(self, accepted: bool, speech_sec: float) -> None:
        if self._streaming:
            # commit は speech_queue に積まれる前に済ませる（待ち札と区間の順序を揃える）
            self._hook(self._stt.commit if accepted else self._stt.clear, "stt commit/clear")
        if accepted:
            self._hook(self._on_speech_end, "on_speech_end", speech_sec, self._seg_no)

    async def _consume(self) -> None:
        assert self._ai is not None
        while True:
            pcm = await self._ai.get_audio()
            # 発話セグメントが届いた＝ユーザは話し終えた（F5: user_speaking 解除 + 沈黙時計リセット）。
            self._hook(self._on_utterance, "on_utterance")
            # J-2 ③-A: STT 待ち窓を開く（この間は自発発話を差し込まない）。
            self._hook(self._on_stt_start, "on_stt_start")
            text = None
            put = False
            try:
                text = await self._stt.transcribe(pcm)
            except Exception:  # STT 失敗はこの発話を捨てて継続（起こりうる）
                logger.exception("STT 失敗（この発話を捨てて継続）")
            if text and is_unintelligible(text):
                # 日本語として受け取れない（発音が崩れて別言語として認識された等）。生の文字は渡さず
                # （知ったかぶりの元）、「聞き取れなかった」という事実だけを応答LLM に渡す。何と言うかは
                # イブが決める（決まり文句は 2026-10-09 に撤去・`stt/askback.py`）。会話履歴にも同じ代替が残る。
                logger.info("🧑? 聞き取れず（%d文字・別言語として認識）→ %s", len(text), UNINTELLIGIBLE_PLACEHOLDER)
                await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=UNINTELLIGIBLE_PLACEHOLDER))
                put = True
            elif text:
                logger.info("🧑 %s", text)
                await self._queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))
                put = True
            self._hook(self._on_stt_result, "on_stt_result", put)
            # J-2 ③-A: 窓を閉じるのは**投入完了後**（put 前に閉じると put の await 中に
            # 自発発話が滑り込む微小窓が残る）。失敗/空認識でも必ず閉じる（固着防止）。
            self._hook(self._on_stt_end, "on_stt_end")

    def stop(self) -> None:
        """止める。**ハンドルを None に戻す**（戻さないと二重起動の検出ができない）。"""
        if self._task:
            self._task.cancel()
            self._task = None
        if self._ai:
            self._ai.stop()
            self._ai = None
        if self._streaming:
            # 送りかけの発話と、受け取り手（`_consume`）を失った待ち札を捨てる。
            # 残すと再開後の最初の発話が、止める前の発話の文字を受け取ってしまう。
            try:
                self._stt.reset()
            except Exception:
                logger.exception("STT の後片付けで例外（無視して継続）")
