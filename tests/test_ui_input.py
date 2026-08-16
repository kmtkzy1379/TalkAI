"""Tier-1: キーボード入力の投入経路（`VoiceLoop.submit_user_text`）。

UI を作らずに、**マイク発話と同じ state 遷移になっているか**を決定論で固める。
ここを間違えると症状が遠くに出る:
  - `mark_user_utterance()` を忘れる → `user_speaking` が True のまま張り付き、
    沈黙監視の `tick()` が永久に発火しない＝**自律発話が二度と起きない**
  - `_barge_in()` を忘れる → 打ち込んでもイブが喋り続け、割り込めない
  - 自発発話の刺激を捨てない → 打った直後に無関係な独り言が先に出る
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.pipeline.stimulus import Stimulus, StimulusKind  # noqa: E402
from eve.stt.askback import AskBackPicker  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


class _FakeLoop:
    """`VoiceLoop.submit_user_text` の依存だけを持つ最小の器（本物は API 鍵が要る）。"""

    def __init__(self) -> None:
        from eve.pipeline.stimulus_queue import StimulusQueue
        from eve.speech import SpeechState

        self.queue = StimulusQueue()
        self.speech_state = SpeechState()
        self._ui_tasks: set = set()
        self.barge_ins = 0
        # 打てている＝意思疎通できているので、連続聞き返しのカウントを戻す（本体と同じ持ち物）。
        self._askback = AskBackPicker()

    def _barge_in(self) -> None:
        self.barge_ins += 1
        self.speech_state.mark_user_speech_start()
        self.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)

    # 本体の実装をそのまま借りる（コピーしない＝実装が変わればテストも追随する）
    from eve.voice_loop import VoiceLoop
    submit_user_text = VoiceLoop.submit_user_text


def t_submit() -> None:
    print("\n[1] 打った文がマイク発話と同じ経路で入る")

    async def run() -> None:
        vl = _FakeLoop()
        vl.speech_state.mark_eve_activity()
        await asyncio.sleep(0.01)

        vl.submit_user_text("  こんにちは  ")
        await asyncio.sleep(0.05)
        stim = await asyncio.wait_for(vl.queue.get(), timeout=1.0)
        check("USER_UTTERANCE として投入される", stim.kind is StimulusKind.USER_UTTERANCE)
        check("前後の空白が落ちている", stim.payload == "こんにちは", f"{stim.payload!r}")
        check("barge-in している（喋っていたら譲る）", vl.barge_ins == 1)
        check("**user_speaking が戻っている**（沈黙監視が止まらない）",
              vl.speech_state.user_speaking is False)
        check("STT を通らないので stt_pending は開かない",
              vl.speech_state.stt_pending is False)

    asyncio.run(run())


def t_empty() -> None:
    print("\n[2] 空入力は何も起こさない")

    async def run() -> None:
        vl = _FakeLoop()
        for bad in ("", "   ", "\n\t", None):
            vl.submit_user_text(bad)  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        check("刺激が1件も入らない", vl.queue.qsize() == 0, f"{vl.queue.qsize()}")
        check("barge-in もしない（喋っている最中に誤爆しない）", vl.barge_ins == 0)

    asyncio.run(run())


def t_discards_autonomous() -> None:
    print("\n[3] 打ち込んだら、待機中の自発発話は捨てられる")

    async def run() -> None:
        vl = _FakeLoop()
        await vl.queue.put(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, payload="ひとりごと"))
        check("先に自発発話が並んでいる", vl.queue.qsize() == 1)
        vl.submit_user_text("ねえ")
        await asyncio.sleep(0.05)
        kinds = []
        while vl.queue.qsize():
            kinds.append((await vl.queue.get()).kind)
        check("自発発話が消え、ユーザ発話だけが残る",
              kinds == [StimulusKind.USER_UTTERANCE], f"{kinds}")

    asyncio.run(run())


def t_silence_clock() -> None:
    """打った直後は「今しゃべった」扱い＝直後に自律発話が割り込まない。"""
    print("\n[4] 沈黙時計がリセットされる")

    async def run() -> None:
        vl = _FakeLoop()
        vl.speech_state.last_activity_mono -= 60.0  # 1分沈黙していたことにする
        before = vl.speech_state.last_activity_mono
        vl.submit_user_text("やあ")
        await asyncio.sleep(0.05)
        check("last_activity が今に更新される",
              vl.speech_state.last_activity_mono > before,
              f"{before} → {vl.speech_state.last_activity_mono}")

    asyncio.run(run())


def t_wiring() -> None:
    print("\n[5] 本体側の配線（UI が読む観測点）")
    import inspect

    from eve.voice_loop import VoiceLoop

    src = inspect.getsource(VoiceLoop.submit_user_text)
    check("barge-in を通している", "_barge_in()" in src)
    check("mark_user_utterance を呼んでいる", "mark_user_utterance()" in src)
    check("create_task の参照を保持している（GC で消えない）", "_ui_tasks.add" in src)

    from eve.audio_input import AudioInput
    check("AudioInput が UI 表示用の speech_prob を持つ",
          "speech_prob" in inspect.getsource(AudioInput.__init__))
    loop_src = inspect.getsource(AudioInput._process_loop)
    check("speech_prob を毎チャンク更新している", "self.speech_prob = speech_prob" in loop_src)
    # 表示専用であること: 判定は元の speech_prob（ローカル変数）で行われ続けている
    check("判定は self.speech_prob ではなくローカル変数のまま（表示専用）",
          "if speech_prob > Config.VAD_THRESHOLD" in loop_src)

    from eve.ui import app as ui_app
    snap = inspect.getsource(ui_app.VoiceLoopThread._snapshot)
    for key in ("mic_on", "mic_level", "user_speaking", "stt_pending", "busy", "task"):
        check(f"スナップショットに {key} がある", f'"{key}"' in snap)
    check("スナップショットは値のコピーだけ（bool/float/str/int に変換している）",
          all(t in snap for t in ("bool(", "float(", "str(", "int(")))
    check("UI→loop は call_soon_threadsafe のみ",
          "call_soon_threadsafe" in inspect.getsource(ui_app.VoiceLoopThread.submit_text))


if __name__ == "__main__":
    t_submit()
    t_empty()
    t_discards_autonomous()
    t_silence_clock()
    t_wiring()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
