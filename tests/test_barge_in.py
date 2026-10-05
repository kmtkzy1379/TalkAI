"""割り込み（barge-in）: 「、」「。」まで話して譲る（API・音声デバイス不要の決定論テスト）。

ユーザ裁定:
- 2026-09-22「今まではユーザーしゃべったら速攻黙ったけど不自然。、。までは喋らせる」
- 2026-09-26「話の途中で応答が始まってもいい。0.5秒も黙れば人間でも話し終わったと思って話し始めて
  しまう。相手がまだ話したら、。まで話して再度割り込ませればいい」

確かめること:
  [1] 鳴っている途中にユーザが話し出す → その節は言い切り、次の節は鳴らない・記憶は鳴った節まで
  [2] ユーザが話している途中にイブの音が鳴り始める → 最初の節を言い切って譲る（割り込みのかけ直し）
  [3] ユーザが話していなければ最後まで話す
  [4] 聞き返しの決まり文句も節ごとに鳴らす（割り込まれたら、その節で譲る）
  [5] 「今まさに話しているか」は VAD の発話区間（発話開始の確定〜SILENCE_LIMIT の無音まで）

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_barge_in.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.pipeline.audio_play_queue import AudioPlayQueue  # noqa: E402
from eve.pipeline.stimulus import Stimulus, StimulusKind  # noqa: E402
from eve.response.orchestrator import ResponseOrchestrator  # noqa: E402
from eve.voice_loop import VoiceLoop  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}" + (f"  ({detail})" if detail else ""))


class FakeCache:
    def __init__(self) -> None:
        self.turns: list = []

    def recent_for_injection(self):
        return []

    def add_turn(self, role, text):
        if text:
            self.turns.append((role, text))

    def eve(self) -> list:
        return [t for r, t in self.turns if r == "eve"]


CHUNKS = 10  # 1節 = 10チャンク（1チャンク 5ms）


class Rig:
    """VoiceLoop の割り込みまわり（`_barge_in` / `_yield_if_user_speaking` / `_speak_canned`）を
    本物のメソッドのまま、周り（マイク・応答の実行役・TTS）だけ偽物にして組む。"""

    def __init__(self) -> None:
        self.speaking = False          # ユーザが今まさに話しているか（VAD の区間の中）
        self.played: list = []         # 鳴らし始めた節
        self.chunks: dict = {}         # 節 → 鳴ったチャンク数
        self.on_chunk = None           # 鳴っている途中に呼ぶ（ユーザが話し出す、を起こす）
        self.task = None               # 応答の実行（runner.interrupt で cancel する）
        self.whys: list = []
        self.audio = AudioPlayQueue(play_fn=self._play)
        self.filler = None
        self.runner = SimpleNamespace(interrupt=self._cancel)
        self.speech_state = SimpleNamespace(mark_user_speech_start=lambda: None)
        self.queue = SimpleNamespace(discard_kind=lambda kind: None)
        self.input = SimpleNamespace(user_in_speech=lambda: self.speaking)
        self.tts = SimpleNamespace(generate_with_meta=self._tts_meta)
        self.audio.add_start_listener(self._yield_if_user_speaking)

    # --- VoiceLoop の本物のメソッド ---
    def _barge_in(self, why: str = "発話検知（割り込み）") -> None:
        self.whys.append(why)
        VoiceLoop._barge_in(self, why)

    def _yield_if_user_speaking(self) -> None:
        VoiceLoop._yield_if_user_speaking(self)

    async def speak_canned(self, text: str) -> None:
        await VoiceLoop._speak_canned(self, text)

    # --- 偽物 ---
    def _cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def _play(self, audio) -> None:
        text = audio.decode() if isinstance(audio, bytes) else str(audio)
        self.played.append(text)
        self.chunks[text] = 0
        for _ in range(CHUNKS):
            self.chunks[text] += 1
            if self.on_chunk is not None:
                self.on_chunk(text)
            await asyncio.sleep(0.005)

    async def _tts_meta(self, text: str):
        return text.encode(), None

    async def settle(self) -> None:
        for _ in range(200):
            if not self.audio.is_playing() and not self.audio.pending_audio():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.03)

    def full(self, text: str) -> bool:
        return self.chunks.get(text) == CHUNKS


REPLY = "うん、それでね、明日行こう。"


async def _respond(rig: Rig, cache: FakeCache) -> None:
    async def tts(s: str) -> bytes:
        return s.encode()

    async def stream_fn(messages):
        yield REPLY

    orch = ResponseOrchestrator(rig.audio, stream_fn, tts, conversation_cache=cache)
    worker = asyncio.create_task(rig.audio.play_worker())
    rig.task = asyncio.create_task(orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "ねえ")))
    try:
        await rig.task
    except asyncio.CancelledError:
        pass
    await rig.settle()
    worker.cancel()


def t_onset_during_playback() -> None:
    print("\n[1] 鳴っている途中にユーザが話し出す → その節は言い切り、次は鳴らない")

    async def run() -> None:
        rig, cache = Rig(), FakeCache()

        def onset(text):
            if text == "それでね、" and not rig.whys:
                rig.speaking = True
                rig._barge_in()  # マイクの発話開始の確定（callback_on_speech_start）

        rig.on_chunk = onset
        await _respond(rig, cache)
        check("鳴った節: うん、→ それでね、（明日行こう。は鳴らない）",
              rig.played == ["うん、", "それでね、"], f"{rig.played}")
        check("割り込まれた節も最後まで鳴った（言い切り）", rig.full("それでね、"), f"{rig.chunks}")
        check("記憶は鳴った節まで", cache.eve() == ["うん、それでね、"], f"{cache.turns}")
        check("割り込みは1回", rig.whys == ["発話検知（割り込み）"], f"{rig.whys}")

    asyncio.run(run())


def t_start_while_user_speaking() -> None:
    print("\n[2] ユーザが話している途中にイブの音が鳴り始める → 最初の節で譲る")

    async def run() -> None:
        rig, cache = Rig(), FakeCache()
        rig.speaking = True  # 前半への応答が、後半を話している途中で鳴り始める
        await _respond(rig, cache)
        check("最初の節だけ鳴った", rig.played == ["うん、"], f"{rig.played}")
        check("その節は最後まで鳴った", rig.full("うん、"), f"{rig.chunks}")
        check("記憶はその節だけ（割り込みより前に確定＝消えない）", cache.eve() == ["うん、"], f"{cache.turns}")
        check("割り込みのかけ直しとしてログに理由が残る",
              len(rig.whys) == 1 and "まだ話している途中" in rig.whys[0], f"{rig.whys}")
        check("応答の実行は止まった（cancel）", rig.task.cancelled())

    asyncio.run(run())


def t_not_speaking() -> None:
    print("\n[3] ユーザが話していなければ最後まで話す")

    async def run() -> None:
        rig, cache = Rig(), FakeCache()
        await _respond(rig, cache)
        check("3節とも鳴った", rig.played == ["うん、", "それでね、", "明日行こう。"], f"{rig.played}")
        check("記憶は全文", cache.eve() == [REPLY], f"{cache.turns}")
        check("割り込みは起きない", rig.whys == [], f"{rig.whys}")

    asyncio.run(run())


def t_yield_guard() -> None:
    print("\n[3b] 話しているかの確認が失敗しても譲らずに続ける（落ちない）")
    rig = Rig()

    def boom():
        raise RuntimeError("mic")

    rig.input = SimpleNamespace(user_in_speech=boom)
    ok = True
    try:
        rig._yield_if_user_speaking()
    except Exception:
        ok = False
    check("例外を外に出さない・割り込まない", ok and rig.whys == [], f"{rig.whys}")


def t_canned() -> None:
    print("\n[4] 聞き返しの決まり文句も節ごとに鳴らす")
    line = "ごめん、今の聞き取れなかった。もう一回言ってくれる？"

    async def run() -> None:
        rig = Rig()
        worker = asyncio.create_task(rig.audio.play_worker())
        await rig.speak_canned(line)
        await rig.settle()
        check("節ごとに順番どおり鳴る",
              rig.played == ["ごめん、", "今の聞き取れなかった。", "もう一回言ってくれる？"], f"{rig.played}")

        rig2 = Rig()

        def onset(text):
            if text == "ごめん、" and not rig2.whys:
                rig2._barge_in()

        rig2.on_chunk = onset
        worker2 = asyncio.create_task(rig2.audio.play_worker())
        await rig2.speak_canned(line)
        await rig2.settle()
        check("割り込まれたら、その節で譲る（言い直しに2文とも被せない）",
              rig2.played == ["ごめん、"] and rig2.full("ごめん、"), f"{rig2.played} {rig2.chunks}")
        worker.cancel()
        worker2.cancel()

    asyncio.run(run())


def t_in_speech() -> None:
    print("\n[5] 「今まさに話しているか」＝ VAD の発話区間の中")
    from eve.audio_input import AudioInput, _SegmentTracker
    from eve.config import Config
    from eve.pipeline.stimulus_queue import StimulusQueue
    from eve.response.input_source import MicSttInputSource, TextInputSource

    src = MicSttInputSource(StimulusQueue(), SimpleNamespace(streaming=False))
    check("マイクが止まっていれば False", src.user_in_speech() is False)
    src._ai = SimpleNamespace(in_speech=lambda: True)
    check("マイクの区間の中なら True", src.user_in_speech() is True)
    check("マイクの無い入力源は常に False", TextInputSource(StimulusQueue()).user_in_speech() is False)

    saved = (Config.VAD_THRESHOLD, Config.VAD_ONSET_SEC, Config.SILENCE_LIMIT, Config.MIC_MIN_PEAK)
    Config.VAD_THRESHOLD, Config.VAD_ONSET_SEC, Config.SILENCE_LIMIT, Config.MIC_MIN_PEAK = 0.5, 0.12, 0.5, 0.045
    try:
        ai = AudioInput.__new__(AudioInput)  # マイク・VAD を開かない
        ai._tracker = None
        check("監視を始める前は False", ai.in_speech() is False)
        ai._tracker = _SegmentTracker(rate=16000, chunk=512, put=lambda b: None)
        ch = b"\x01\x00" * 512
        t = 0.0

        def feed(prob: float, peak: float, n: int) -> None:
            nonlocal t
            for _ in range(n):
                ai._tracker.step(ch, prob, peak, t)
                t += 0.032

        feed(0.1, 0.01, 5)
        check("無音: False", ai.in_speech() is False)
        feed(0.9, 0.2, 6)
        check("発話開始の確定後: True", ai.in_speech() is True)
        feed(0.1, 0.01, 9)   # 約0.29秒の間（SILENCE_LIMIT 0.5 未満）
        check("話の途中の短い間はまだ区間の中: True", ai.in_speech() is True)
        feed(0.1, 0.01, 10)  # 合わせて 0.5秒を超える
        check("SILENCE_LIMIT を超える無音で区間が終わる: False", ai.in_speech() is False)
    finally:
        Config.VAD_THRESHOLD, Config.VAD_ONSET_SEC, Config.SILENCE_LIMIT, Config.MIC_MIN_PEAK = saved


t_onset_during_playback()
t_start_while_user_speaking()
t_not_speaking()
t_yield_guard()
t_canned()
t_in_speech()
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
