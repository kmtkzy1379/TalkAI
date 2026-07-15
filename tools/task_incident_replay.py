r"""実機事故 2026-07-13（予約タスク誤取消）の音声E2E再現テスト。

事故: 「30秒後に気持ち教えて」→「やっぱり50秒後に変えて」→「お願い。」で、応答LLMが
完了済み変更を再実行し、CancelResolver の1件無照合 fast path が残すべき50秒タスクを削除。
タスクは発火せずユーザは結果を受け取れなかった（tasks.jsonl t_ce8cb4e62e / t_46482b46e1）。

Phase A（確率的・N回）: VOICEVOX 合成ユーザ音声→実STT→実パイプライン（実LLM gpt-5.5・
  CancelResolver 配線済み・実 VOICEVOX TTS・再生時間忠実プレイヤー）で事故台本をリプレイし、
  50秒タスクが「一度も Cancelled にならず Done + 報告発話」に到達するかを検証。
  ※本台本にはタスク取消を頼む発話が一切無い → 50秒タスクへの Cancelled イベント=即バグ。
Phase B（決定論）: CancelResolver へ reference を直接注入するマトリクス（実 task LLM）。
  「30秒のやつ」×アクティブ1件(50秒)=温存 / 「やっぱりいいや」=即取消 等。
  同一ターン delegate+cancel ペア（65μ秒自己取消の再現形）も検証。

既存 tools/task_full_test.py との差分（事故がテストをすり抜けた構造的理由の修正）:
  - CancelResolver を本番同等に配線（既存ツールは resolver 未配線のフォールバック経路だった）
  - 実 VoicevoxTTS + DurationSimPlayer（orchestrator は audio.join 後に tool submit するため、
    再生時間がレース窓を支配する。フェイク20msでは事故タイミングを再現できない）

実行: $env:PYTHONIOENCODING="utf-8"; ..\portfolio8-VLM-AI\venv\Scripts\python.exe tools\task_incident_replay.py
変数: N_RUNS(既定3) / REAL_AUDIO=1(実スピーカー再生) / SKIP_A=1 / SKIP_B=1
前提: VOICEVOX 起動（127.0.0.1:50021）+ .env の OPENAI_API_KEY。Phase B のみなら VOICEVOX 不要。
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402

Config.CALLFUNCTION_ENABLED = True
Config.TASK_ENABLED = True

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.memory import ConversationCache  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import AudioPlayQueue, PipelineRunner, Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.response import ResponseOrchestrator  # noqa: E402
from eve.response.function_dispatcher import FunctionDispatcher, parse_tool_call  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.response.tts import VoicevoxTTS  # noqa: E402
from eve.speech import SpeechState  # noqa: E402
from eve.stt import make_stt  # noqa: E402
from eve.task import (  # noqa: E402
    CANCELLED, DONE, FAILED, PENDING, RUNNING,
    CancelResolver, ReconcileTimer, TaskAgent, TaskExecutor, TaskStore,
    active_tasks_for_context, register_task_capabilities,
)

T0 = time.monotonic()
CUR = ["init"]


def ev(tag, d):
    print(f"  {time.monotonic() - T0:6.1f} [{CUR[0]:12}] {tag} {d}", flush=True)


def synth(text, speaker=8, rate=16000):
    """ユーザ音声の合成（VOICEVOX）。task_full_test.py と同一。"""
    import requests
    q = requests.post(f"{Config.VOICEVOX_URL}/audio_query", params={"text": text, "speaker": speaker}, timeout=10).json()
    q["outputSamplingRate"] = rate; q["outputStereoToMono"] = True
    return requests.post(f"{Config.VOICEVOX_URL}/synthesis", json=q, params={"speaker": speaker}, timeout=20).content


def wav_pcm(b):
    with wave.open(io.BytesIO(b), "rb") as wf:
        return wf.readframes(wf.getnframes())


class DurationSimPlayer:
    """実再生時間を忠実に再現する無音プレイヤー（WAV 実長ぶん sleep・barge-in 対応）。

    orchestrator は「再生完了(audio.join)→tool submit」の順のため、再生時間が
    取消レースの窓を決める。実 TTS の WAV を実長で消費し、音だけ出さない。
    """

    async def play_fn(self, audio, should_stop=None):
        if not audio:
            return
        try:
            with wave.open(io.BytesIO(audio), "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate() or 24000)
        except Exception:
            dur = max(0.5, len(audio) / 48000.0)
        t = 0.0
        while t < dur:
            if should_stop is not None and should_stop():
                return
            await asyncio.sleep(0.02)
            t += 0.02


def _make_player():
    if os.getenv("REAL_AUDIO") == "1":
        from eve.response.player import RealAudioPlayer
        return RealAudioPlayer()
    return DurationSimPlayer()


class World:
    """1 run ぶんの本番同等パイプライン（毎 run 作り直し＝会話履歴/dedup/queue 残留の汚染防止）。"""

    def __init__(self, reg, stt, art_dir, run_id):
        self.reg = reg
        self.stt = stt
        self.art = os.path.join(art_dir, f"run{run_id}")
        os.makedirs(self.art, exist_ok=True)
        self.task_file = os.path.join(self.art, "tasks.jsonl")
        self.handles = []  # 完了ターン: (kind, dedup_key, attempts, 生成テキスト)
        self.handle_starts = []  # 開始ターン: (mono, kind, dedup_key, attempts)
        self.cancelled = []  # 中断されたターン: (kind, dedup_key)
        self.report_started = {}  # dedup_key -> asyncio.Event（barge タイミング制御用）
        self.turn_played_chars = 0  # 現報告ターンで実再生完了した文字数（barge 有効性ゲート）
        self.speech_state = SpeechState()  # 本番同等の user_speaking（再配達 WHEN 制御）
        self._redeliver_tasks = []
        # 異常検出: ターン cancel 後も stream が回り続けたら記録（barge が効いていない疑いの実証用）
        self._turn_id = 0
        self._active_turn = 0
        self.turn_cancelled_at = {}  # turn_id -> mono
        self.anomalies = []  # (turn_id, 遅れ秒, 直近delta断片)

    async def start(self):
        self.cache = ConversationCache(history_file=os.path.join(self.art, "h.jsonl"))
        await self.cache.initialize()
        self.queue = StimulusQueue()
        self.store = TaskStore(task_file=self.task_file)
        self.caps = CapabilityRegistry(is_busy=lambda: self.runner.is_busy(), qsize=lambda: self.queue.qsize())
        # 本番同等の要: CancelResolver を配線（voice_loop.py と同じ）。既存ツールはここが欠けていた。
        self.resolver = CancelResolver(store=self.store, model_registry=self.reg, queue=self.queue)
        register_task_capabilities(self.caps, self.store, cancel_resolver=self.resolver)
        self.dispatcher = FunctionDispatcher(registry=self.caps, queue=self.queue)
        _submit = self.dispatcher.submit

        def submit_wrap(tcs):
            ev("⚙submit", [f"{parse_tool_call(t)[0]}{parse_tool_call(t)[1]}" for t in tcs])
            _submit(tcs)
        self.dispatcher.submit = submit_wrap  # type: ignore

        agent = TaskAgent(registry=self.caps, model_registry=self.reg, store=self.store,
                          max_steps=Config.TASK_AGENT_MAX_STEPS, timeout_sec=Config.TASK_AGENT_TIMEOUT_SEC)
        self.executor = TaskExecutor(store=self.store, registry=self.caps, queue=self.queue, agent=agent)
        self.timer = ReconcileTimer(store=self.store, executor=self.executor, tick_sec=1.0)
        self.player = _make_player()
        self.audio = AudioPlayQueue(play_fn=self.player.play_fn)
        self.tts = VoicevoxTTS()

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            # 本番 voice_loop.py と同一（max_tokens は付けない: gpt-5.x 系は reasoning が
            # 予算を食い尽くし content が空になる＝発話ゼロで再現にならない）。
            # 計装: この turn が cancel された後に delta が流れ続けたら異常記録（barge 有効性の実証）。
            tid = world._active_turn

            def _check(x):
                cat = world.turn_cancelled_at.get(tid)
                if cat is not None:
                    late = time.monotonic() - cat
                    if late > 0.3:
                        world.anomalies.append((tid, round(late, 2), str(x)[:30]))
                return x
            if tools:
                async for x in self.reg.stream_with_tools("response", messages, tools=tools,
                                                          tool_sink=tool_sink):
                    yield _check(x)
            else:
                async for x in self.reg.stream("response", messages):
                    yield _check(x)

        # 実再生完了（on_played=聞かれた扱い）の文字数を観測（barge 有効性ゲート用）。
        _enq = self.audio.enqueue
        world = self

        def enqueue_wrap(gen, seq, wav, text="", on_played=None):
            def played(t):
                world.turn_played_chars += len(t)
                if on_played is not None:
                    on_played(t)
            _enq(gen, seq, wav, text=text, on_played=played)
        self.audio.enqueue = enqueue_wrap  # type: ignore

        orch = ResponseOrchestrator(self.audio, stream_fn, self.tts.generate,
                                    ContextAssembler(system_prompt=SPEECH_STYLE),
                                    conversation_cache=self.cache, dispatcher=self.dispatcher,
                                    tasks_provider=lambda: active_tasks_for_context(self.store),
                                    redeliver_fn=self._redeliver_stimulus)
        _handle = orch.handle

        async def handle_wrap(stim):
            att = getattr(stim.payload, "attempts", None)
            dk = getattr(stim, "dedup_key", None)
            world._turn_id += 1
            world._active_turn = world._turn_id
            tid = world._turn_id
            world.handle_starts.append((time.monotonic(), stim.kind, dk, att))
            if stim.kind == StimulusKind.CALLFUNCTION_RESULT:
                world.turn_played_chars = 0
                world.report_started.setdefault(dk, asyncio.Event()).set()
                ev("▶報告開始", f"dedup={dk} attempts={att}")
            try:
                await _handle(stim)
            except asyncio.CancelledError:
                world.turn_cancelled_at[tid] = time.monotonic()
                world.cancelled.append((stim.kind, dk))
                ev("✂中断", f"{stim.kind.name} dedup={dk}")
                raise
            world.handles.append((stim.kind, dk, att, orch.last_response or ""))
            tag = {StimulusKind.USER_UTTERANCE: "🤖", StimulusKind.CALLFUNCTION_RESULT: "🛠報告"}.get(stim.kind, "🤖?")
            ev(tag, (orch.last_response or "（空）")[:90])
        orch.handle = handle_wrap  # type: ignore
        self.orch = orch
        self.runner = PipelineRunner(self.queue, orch, self.audio)
        self._bg = [asyncio.create_task(self.audio.play_worker()), asyncio.create_task(self.runner.run())]
        await self.store.initialize()
        self.dispatcher.start(); self.executor.start(); self.timer.start(); self.resolver.start()

    async def stop(self):
        for t in self._redeliver_tasks:
            if not t.done():
                t.cancel()
        await self.timer.stop(); await self.resolver.stop(); await self.executor.stop()
        await self.dispatcher.stop()
        for t in self._bg:
            t.cancel()
        await asyncio.sleep(0.2)
        await self.cache.shutdown(); await self.store.shutdown(); await self.tts.close()
        if hasattr(self.player, "close"):
            self.player.close()

    # --- 駆動ヘルパ ---------------------------------------------------------

    def _redeliver_stimulus(self, stim, should_abort):
        """本番 voice_loop._redeliver_stimulus と同一ロジック（WHEN 制御込みの複製）。"""
        async def _wait_and_put():
            t0 = time.monotonic()
            while self.speech_state.user_speaking and time.monotonic() - t0 < 60.0:
                await asyncio.sleep(0.1)
            await asyncio.sleep(2.0)  # STT 完了猶予
            if should_abort():
                ev("🔁中止", "再生済み確定（二重発話防止）")
                return
            ev("🔁再投入", f"dedup={stim.dedup_key} attempts={stim.payload.attempts}")
            await self.queue.put(stim)
        self._redeliver_tasks.append(asyncio.create_task(_wait_and_put()))

    def barge(self):
        self.audio.interrupt(); self.runner.interrupt()
        self.speech_state.mark_user_speech_start()  # 本番 _barge_in と同等

    def _sidecars_idle(self):
        return (self.dispatcher._inbox.empty() and self.dispatcher._inbox._unfinished_tasks == 0
                and self.resolver._inbox.empty() and self.resolver._inbox._unfinished_tasks == 0
                and all(t.done() for t in self._redeliver_tasks))  # 再配達待機中は「静止」でない

    async def wait_idle(self, timeout=60.0):
        """応答・tool 実行・取消解決すべての静止を待つ（resolver の LLM 照合中に早抜けしない）。"""
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if (not self.runner.is_busy() and self.queue.qsize() == 0
                    and self.executor.is_idle() and self._sidecars_idle()):
                await asyncio.sleep(0.3)
                if (not self.runner.is_busy() and self.queue.qsize() == 0
                        and self.executor.is_idle() and self._sidecars_idle()):
                    return True
            await asyncio.sleep(0.1)
        return False

    async def hear(self, line, must=()):
        """合成音声→実STT。台本キーワード(must)が欠けたら1回リトライ、それでもダメなら台本を使う
        （誤聴で台本自体が変わると再現テストにならない。音声経路は通した上で文言だけ保証する）。"""
        for attempt in range(2):
            wav = await asyncio.to_thread(synth, line)
            try:
                text = await self.stt.transcribe(wav_pcm(wav))
            except Exception as e:
                ev("STT失敗", e); text = None
            if text and all(m in text for m in must):
                return text
            ev("👂誤聴", f"{attempt + 1}回目: {text!r}（必須語 {must}）")
        return line

    async def say(self, line, *, barge=False, must=()):
        heard = await self.hear(line, must=must)
        if barge:
            self.barge()
        self.speech_state.mark_user_utterance()  # 話し終わり（本番: STT 前に解除される）
        ev("🧑", heard)
        await self.queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))
        await self.wait_idle()

    def goal_tasks(self):
        return [t for t in self.store.list_all() if t.goal]

    def cancelled_ids_from_file(self):
        """永続イベントログから Cancelled になった task_id 集合（一度でも取消イベントがあれば検出）。"""
        out = set()
        with open(self.task_file, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == CANCELLED:
                    out.add(r.get("task_id"))
        return out


# ============================ Phase A: 音声リプレイ ============================

U1 = "30秒後に今の気持ち教えてくれる？"
# 実機事故でユーザが言い直した表現（06:08:45）に合わせ cancel+delegate の両方を明示的に誘発
# （「50秒後に変えて」だけだと gpt-5.5 の解釈が delegate のみ/cancel のみにばらける）。
U2 = "あ、やっぱりさっきの30秒のは止めて、50秒後に変えてくれる？"
U3 = "お願い。"


async def phase_a_run(reg, stt, art_dir, run_id):
    """1 run。返り値: ('PASS'|'FAIL'|'INCONCLUSIVE', 理由)。"""
    CUR[0] = f"A-run{run_id}"
    w = World(reg, stt, art_dir, run_id)
    await w.start()
    try:
        await w.say(U1, must=("30", "秒"))
        pend1 = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend1) != 1:
            return ("INCONCLUSIVE", f"U1後の PENDING が {len(pend1)} 件（期待1）")
        t30 = pend1[0]
        ev("📋", f"30秒タスク: {t30.task_id} 「{t30.goal[:20]}」")

        await w.say(U2, must=("30", "50"))
        await asyncio.sleep(2.0)  # resolver の LLM 照合の余韻
        await w.wait_idle()
        pend2 = [t for t in w.goal_tasks() if t.status in (PENDING, RUNNING)]
        if not (t30.task_id in w.cancelled_ids_from_file() and len(pend2) == 1):
            return ("INCONCLUSIVE",
                    f"U2後: 30秒版取消={t30.task_id in w.cancelled_ids_from_file()} アクティブ={len(pend2)}件（期待: 取消済+1件）")
        t50 = pend2[0]
        ev("📋", f"50秒タスク: {t50.task_id} 「{t50.goal[:20]}」 when={t50.when}")

        # 核心ターン: 「お願い。」（実機同様 barge-in してから）
        await w.say(U3, barge=True)
        st = w.store.get(t50.task_id).status
        ev("📋", f"U3後の50秒タスク: {st}")
        if st == CANCELLED:
            return ("FAIL", "事故再現: 『お願い。』ターンで50秒タスクが誤取消された")

        # 発火待ち（登録+50秒 は U2〜U3 で一部消化済み → 残り + 報告余裕で最大80秒 poll）
        deadline = time.monotonic() + 80
        while time.monotonic() < deadline:
            st = w.store.get(t50.task_id).status
            if st in (DONE, FAILED, CANCELLED):
                break
            await asyncio.sleep(0.5)
        await w.wait_idle(30)
        st = w.store.get(t50.task_id).status
        cancelled = w.cancelled_ids_from_file()
        report_spoken = any(k == StimulusKind.CALLFUNCTION_RESULT and (d or "") == f"task:{t50.task_id}" and sp
                            for k, d, _att, sp in w.handles)
        ev("📋", f"最終: 50秒タスク={st} 取消集合={sorted(cancelled)} 報告発話={report_spoken}")
        if t50.task_id in cancelled:
            return ("FAIL", "50秒タスクに Cancelled イベント（台本に取消依頼は無い＝バグ）")
        if st != DONE:
            return ("FAIL", f"50秒タスクが Done に到達せず（{st}）")
        if not report_spoken:
            return ("FAIL", "タスク結果の報告が発話されていない")
        extra = [t for t in w.goal_tasks() if t.task_id not in (t30.task_id, t50.task_id)]
        if extra:
            ev("⚠", f"想定外のタスク再作成 {len(extra)} 件（Fix#2 検証観点）: {[t.task_id for t in extra]}")
        return ("PASS", "50秒タスク温存→発火→報告 まで完走")
    finally:
        await w.stop()


# ============================ Phase C: 報告 barge-in 再配達（2026-07-13 21:20 事故） ============================

async def _mimic_user_speech(w, heard, hold_sec=0.8, stt_sec=0.5):
    """barge 済みの状態から「話し終わり→STT→刺激投入」の本番イベント順序を模倣。"""
    await asyncio.sleep(hold_sec)          # 発話時間
    w.speech_state.mark_user_utterance()   # 本番: STT 前に user_speaking 解除
    await asyncio.sleep(stt_sec)           # 擬似 STT 遅延（< 再配達猶予 2s）
    ev("🧑", heard)
    await w.queue.put(Stimulus(StimulusKind.USER_UTTERANCE, heard))


def _report_stats(w, dedup):
    starts = [(m, a) for m, k, d, a in w.handle_starts
              if k == StimulusKind.CALLFUNCTION_RESULT and d == dedup]
    dones = [(a, sp) for k, d, a, sp in w.handles
             if k == StimulusKind.CALLFUNCTION_RESULT and d == dedup and sp]
    cut = [(k, d) for k, d in w.cancelled if d == dedup]
    return starts, dones, cut


async def phase_c1(reg, stt, art, run_id):
    """事故再現→救済: 報告ターン開始直後（発話前）に barge → OK 応答が先 → 再配達が一度だけ。"""
    CUR[0] = f"C1-run{run_id}"
    w = World(reg, stt, art, f"c1_{run_id}")
    await w.start()
    try:
        heard_ok = await w.hear("OK、ありがとう")  # 音声→実STT を事前に済ませタイミングを分離
        await w.say("30秒後に今の時刻を教えてくれる？", must=("30", "秒"))
        pend = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend) != 1:
            return ("INCONCLUSIVE", f"予約が {len(pend)} 件（期待1）")
        dedup = f"task:{pend[0].task_id}"
        evt = w.report_started.setdefault(dedup, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=90)
        except asyncio.TimeoutError:
            return ("INCONCLUSIVE", "報告ターンが開始しなかった")
        await asyncio.sleep(0.4)
        if w.turn_played_chars > 5:
            return ("INCONCLUSIVE", "barge 前に本文が再生されていた（barge が遅すぎ）")
        w.barge()  # 発話前の報告を潰す（事故の再現）
        await _mimic_user_speech(w, heard_ok)
        # 再配達報告（attempts=1）の完了を待つ
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            if any(a == 1 for a, _ in _report_stats(w, dedup)[1]):
                break
            await asyncio.sleep(0.5)
        await w.wait_idle(30)
        starts, dones, cut = _report_stats(w, dedup)
        st = w.store.get(pend[0].task_id).status
        ev("📋", f"starts={[(f'{a}',) for _, a in starts]} dones={[(a, s[:30]) for a, s in dones]} "
                 f"中断={len(cut)} store={st}")
        if [a for _, a in starts] != [0, 1]:
            return ("FAIL", f"報告ターン開始列が {[a for _, a in starts]}（期待 [0,1]=潰れ+再配達1回）")
        if len(cut) != 1:
            return ("FAIL", f"初回報告の中断記録が {len(cut)} 件（期待1）")
        if len(dones) != 1 or dones[0][0] != 1:
            return ("FAIL", f"発話された報告が {len(dones)} 件（期待: attempts=1 の1件のみ＝2回言わない）")
        # 順序: 初回報告開始より後に来た USER（=OK）の応答が、再配達報告より先に始まっていること
        first_report_start = starts[0][0]
        redeliver_start = [m for m, a in starts if a == 1][0]
        ok_user_starts = [m for m, k, d, a in w.handle_starts
                          if k == StimulusKind.USER_UTTERANCE and m > first_report_start]
        if not ok_user_starts or min(ok_user_starts) > redeliver_start:
            return ("FAIL", "OK への応答より先に再配達報告が始まった（ユーザ優先の破れ）")
        if st != DONE or pend[0].task_id in w.cancelled_ids_from_file():
            return ("FAIL", f"store 状態異常（{st}）")
        return ("PASS", f"潰れた報告を再配達で救済（報告発話1回・OK応答が先・内容: {dones[0][1][:40]}）")
    finally:
        await w.stop()


async def phase_c2(reg, stt, art, run_id):
    """二重発話防止: 報告を >5文字 再生済みの途中で barge → 再配達されない（聞かれた扱い）。"""
    CUR[0] = f"C2-run{run_id}"
    w = World(reg, stt, art, f"c2_{run_id}")
    await w.start()
    try:
        heard = await w.hear("ありがとう")
        await w.say("30秒後にあなたの好きな色を教えてくれる？", must=("30", "秒"))
        pend = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend) != 1:
            return ("INCONCLUSIVE", f"予約が {len(pend)} 件（期待1）")
        dedup = f"task:{pend[0].task_id}"
        evt = w.report_started.setdefault(dedup, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=90)
        except asyncio.TimeoutError:
            return ("INCONCLUSIVE", "報告ターンが開始しなかった")
        # 本文が >5文字 再生されるまで待ってから barge（実長再生なので現実的な窓がある）
        t0 = time.monotonic()
        while w.turn_played_chars <= 5 and time.monotonic() - t0 < 60:
            if _report_stats(w, dedup)[1]:
                return ("INCONCLUSIVE", "barge 前に報告が完了した")
            await asyncio.sleep(0.05)
        if w.turn_played_chars <= 5:
            return ("INCONCLUSIVE", "本文再生を検知できなかった")
        played = w.turn_played_chars
        w.barge()
        await _mimic_user_speech(w, heard)
        await asyncio.sleep(8.0)  # 再配達猶予(2s)を大きく超えて観測
        await w.wait_idle(30)
        starts, dones, cut = _report_stats(w, dedup)
        retry_starts = [a for _, a in starts if a and a >= 1]
        ev("📋", f"再生済み{played}文字で barge → starts={[a for _, a in starts]} 再配達開始={retry_starts}")
        if retry_starts:
            return ("FAIL", f"再生済み({played}文字)なのに再配達が実行された＝同じ報告を2回言う")
        return ("PASS", f"{played}文字再生済み（聞かれた扱い）→ 再配達なし")
    finally:
        await w.stop()


async def phase_c3(reg, stt, art, run_id):
    """複数タスク兼ね合い: 2予約中に先行報告を barge → 両報告が各1回ずつ・消失ゼロ・二重ゼロ。"""
    CUR[0] = f"C3-run{run_id}"
    w = World(reg, stt, art, f"c3_{run_id}")
    await w.start()
    try:
        heard = await w.hear("うん、お願い")
        await w.say("30秒後に今の時刻を教えてくれる？", must=("30", "秒"))
        pend1 = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend1) != 1:
            return ("INCONCLUSIVE", f"予約1が {len(pend1)} 件")
        t_time = pend1[0]
        await w.say("それと50秒後にあなたの好きな色も教えて。", must=("50",))
        pend2 = [t for t in w.goal_tasks() if t.status == PENDING and t.task_id != t_time.task_id]
        if len(pend2) != 1:
            return ("INCONCLUSIVE", f"予約2が {len(pend2)} 件")
        t_color = pend2[0]
        d_time, d_color = f"task:{t_time.task_id}", f"task:{t_color.task_id}"
        evt = w.report_started.setdefault(d_time, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=120)
        except asyncio.TimeoutError:
            return ("INCONCLUSIVE", "時刻報告が開始しなかった")
        await asyncio.sleep(0.4)
        if w.turn_played_chars > 5:
            return ("INCONCLUSIVE", "barge 前に本文再生済み")
        w.barge()
        await _mimic_user_speech(w, heard)
        # 両報告の完了を待つ
        t0 = time.monotonic()
        while time.monotonic() - t0 < 180:
            if _report_stats(w, d_time)[1] and _report_stats(w, d_color)[1]:
                break
            await asyncio.sleep(0.5)
        await w.wait_idle(30)
        s_t, d_t, _ = _report_stats(w, d_time)
        s_c, d_c, _ = _report_stats(w, d_color)
        ev("📋", f"時刻: starts={[a for _, a in s_t]} dones={len(d_t)} / 色: starts={[a for _, a in s_c]} dones={len(d_c)}")
        if len(d_t) != 1 or d_t[0][0] != 1:
            return ("FAIL", f"時刻報告の発話が {len(d_t)} 件（期待: 再配達1件のみ）")
        if len(d_c) != 1 or d_c[0][0] != 0:
            return ("FAIL", f"色報告の発話が {len(d_c)} 件/attempts={d_c[0][0] if d_c else '-'}（期待: 通常配達1件）")
        if (w.store.get(t_time.task_id).status != DONE or w.store.get(t_color.task_id).status != DONE):
            return ("FAIL", "store 状態異常")
        return ("PASS", "潰した時刻報告=再配達1回・色報告=通常1回・消失/二重ゼロ")
    finally:
        await w.stop()


# ============================ Phase D: タスク待機中の混線ストレス ============================

def _full_delivery_count(w, marker):
    """完了ターンのうち marker（結果の中核語）を含む発話の数（完全配達の重複検出）。"""
    return sum(1 for _k, _d, _a, sp in w.handles if marker in sp)


async def phase_d1(reg, stt, art, run_id):
    """雑談混線+発話前barge: タスク待機中に雑談2連発→報告を潰す→追い質問→全部整合するか。"""
    CUR[0] = f"D1-run{run_id}"
    w = World(reg, stt, art, f"d1_{run_id}")
    await w.start()
    try:
        heard_q3 = await w.hear("ごめん、続けて。")
        await w.say("30秒後に今の時刻を教えてくれる？", must=("30", "秒"))
        pend = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend) != 1:
            return ("INCONCLUSIVE", f"予約が {len(pend)} 件")
        dedup = f"task:{pend[0].task_id}"
        # 待機中の雑談（タスクと無関係な質問を挟む）
        await w.say("ところで富士山ってどれくらいの高さだっけ？")
        await w.say("へえ。じゃあ海で一番深いところは何メートル？")
        evt = w.report_started.setdefault(dedup, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=90)
        except asyncio.TimeoutError:
            return ("INCONCLUSIVE", "報告ターンが開始しなかった")
        await asyncio.sleep(0.4)
        if w.turn_played_chars > 5:
            return ("INCONCLUSIVE", "barge 前に本文再生済み")
        w.barge()
        await _mimic_user_speech(w, heard_q3)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            if any(a == 1 for a, _ in _report_stats(w, dedup)[1]):
                break
            await asyncio.sleep(0.5)
        await w.wait_idle(30)
        starts, dones, cut = _report_stats(w, dedup)
        result_core = (w.store.get(pend[0].task_id).result or "")[:10]
        n_user_done = sum(1 for k, _d, _a, sp in w.handles if k == StimulusKind.USER_UTTERANCE and sp)
        ev("📋", f"starts={[a for _, a in starts]} dones={len(dones)} user応答={n_user_done} "
                 f"anomalies={w.anomalies}")
        if w.anomalies:
            return ("FAIL", f"cancel 後も stream 継続の異常: {w.anomalies}")
        if len(dones) != 1 or dones[0][0] != 1:
            return ("FAIL", f"報告発話が {len(dones)} 件（期待: 再配達の1件）")
        if n_user_done < 2:  # 雑談2件ぶん（予約 ack と「続けて」は tool のみ無発話になり得る既知挙動）
            return ("FAIL", f"ユーザ発話への応答が {n_user_done} 件（雑談が無視された）")
        if w.store.get(pend[0].task_id).status != DONE:
            return ("FAIL", "store 状態異常")
        return ("PASS", f"雑談2件応答+報告再配達1回（結果: {result_core}…）・stream異常ゼロ")
    finally:
        await w.stop()


async def phase_d2(reg, stt, art, run_id):
    """本文再生中barge+追い質問: >5文字再生済みで潰す→即質問→重複配達なし・stream異常なし。"""
    CUR[0] = f"D2-run{run_id}"
    w = World(reg, stt, art, f"d2_{run_id}")
    await w.start()
    try:
        heard_q = await w.hear("ありがとう。ところで水って何度で凍るんだっけ？")
        await w.say("30秒後にあなたの好きな食べ物を教えてくれる？", must=("30", "秒"))
        pend = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend) != 1:
            return ("INCONCLUSIVE", f"予約が {len(pend)} 件")
        dedup = f"task:{pend[0].task_id}"
        evt = w.report_started.setdefault(dedup, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=90)
        except asyncio.TimeoutError:
            return ("INCONCLUSIVE", "報告ターンが開始しなかった")
        t0 = time.monotonic()
        while w.turn_played_chars <= 5 and time.monotonic() - t0 < 60:
            if _report_stats(w, dedup)[1]:
                return ("INCONCLUSIVE", "barge 前に報告が完了した")
            await asyncio.sleep(0.05)
        if w.turn_played_chars <= 5:
            return ("INCONCLUSIVE", "本文再生を検知できず")
        t_barge = time.monotonic()
        w.barge()
        await _mimic_user_speech(w, heard_q)
        await asyncio.sleep(10.0)
        await w.wait_idle(30)
        starts, dones, _ = _report_stats(w, dedup)
        retry_starts = [a for _, a in starts if a and a >= 1]
        result_core = (w.store.get(pend[0].task_id).result or "")[:8]
        dup = _full_delivery_count(w, result_core) if result_core else 0
        # 追い質問の処理判定: barge 後に USER ターンが開始し、かつ非空のユーザ応答が存在すること
        # （予約 ack が tool のみ無発話になる既知挙動があるため総数閾値では判定しない）。
        followup_started = any(m > t_barge and k == StimulusKind.USER_UTTERANCE
                               for m, k, _d, _a in w.handle_starts)
        answered = any(k == StimulusKind.USER_UTTERANCE and sp for k, _d, _a, sp in w.handles)
        ev("📋", f"再配達開始={retry_starts} 完全配達数={dup} 追い質問開始={followup_started} "
                 f"応答有={answered} anomalies={w.anomalies}")
        if w.anomalies:
            return ("FAIL", f"cancel 後も stream 継続の異常: {w.anomalies}")
        if retry_starts:
            return ("FAIL", "再生済みなのに再配達された（二重発話）")
        if dup > 1:
            return ("FAIL", f"結果の完全配達が {dup} 回（重複発話）")
        if not (followup_started and answered):
            return ("FAIL", "追い質問が処理されなかった")
        return ("PASS", "再生中barge→再配達なし・重複なし・追い質問に応答・stream異常ゼロ")
    finally:
        await w.stop()


async def phase_d3(reg, stt, art, run_id):
    """再配達の連続potsし: 初回と再配達1回目を両方潰す→2回目の再配達で届く（上限内の粘り）。"""
    CUR[0] = f"D3-run{run_id}"
    w = World(reg, stt, art, f"d3_{run_id}")
    await w.start()
    try:
        heard1 = await w.hear("ちょっと待って。")
        heard2 = await w.hear("ごめん、もう大丈夫。")
        await w.say("30秒後に今の時刻を教えてくれる？", must=("30", "秒"))
        pend = [t for t in w.goal_tasks() if t.status == PENDING]
        if len(pend) != 1:
            return ("INCONCLUSIVE", f"予約が {len(pend)} 件")
        dedup = f"task:{pend[0].task_id}"
        evt = w.report_started.setdefault(dedup, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=90)
        except asyncio.TimeoutError:
            return ("INCONCLUSIVE", "報告ターン未開始")
        await asyncio.sleep(0.4)
        if w.turn_played_chars > 5:
            return ("INCONCLUSIVE", "barge1 前に本文再生済み")
        w.barge()
        await _mimic_user_speech(w, heard1)
        # 再配達1回目（attempts=1）の開始を待って、それも発話前に潰す
        t0 = time.monotonic()
        retry1_started = False
        while time.monotonic() - t0 < 90:
            if any(a == 1 for _m, a in _report_stats(w, dedup)[0]):
                retry1_started = True
                break
            await asyncio.sleep(0.1)
        if not retry1_started:
            return ("INCONCLUSIVE", "再配達1回目が開始しなかった")
        await asyncio.sleep(0.4)
        if w.turn_played_chars > 5:
            return ("INCONCLUSIVE", "barge2 前に本文再生済み")
        w.barge()
        await _mimic_user_speech(w, heard2)
        # 再配達2回目（attempts=2・上限）で届くはず
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120:
            if any(a == 2 for a, _ in _report_stats(w, dedup)[1]):
                break
            await asyncio.sleep(0.5)
        await w.wait_idle(30)
        starts, dones, cut = _report_stats(w, dedup)
        ev("📋", f"starts={[a for _, a in starts]} dones={[(a, s[:25]) for a, s in dones]} "
                 f"中断={len(cut)} anomalies={w.anomalies}")
        if w.anomalies:
            return ("FAIL", f"cancel 後も stream 継続の異常: {w.anomalies}")
        if [a for _, a in starts] != [0, 1, 2]:
            return ("FAIL", f"開始列 {[a for _, a in starts]}（期待 [0,1,2]）")
        if len(dones) != 1 or dones[0][0] != 2:
            return ("FAIL", f"発話された報告 {len(dones)} 件（期待: attempts=2 の1件）")
        return ("PASS", "2連続で潰しても3回目（上限）で届いた・発話1回のみ")
    finally:
        await w.stop()


# ============================ Phase B: 決定論注入 ============================

# (reference, 温存すべきか) — アクティブは常に「50秒後に今の気持ち教えて」1件。
B_MATRIX = [
    ("30秒のやつ", True),            # 事故の直接回帰（不一致 reference は温存すべき）
    ("さっきの30秒の予約", True),     # 言い換え耐性
    ("やっぱりいいや", False),        # 基準ケース: 曖昧取消は即キャンセルを維持
    ("", False),                      # reference 省略 = 今のをやめて
    ("50秒のやつ", False),            # 正引き一致は取消
]


async def phase_b_probe(reg, art_dir, n, reference, expect_keep):
    q = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art_dir, f"b{n}.jsonl"))
    await store.initialize()
    caps = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)
    resolver = CancelResolver(store=store, model_registry=reg, queue=q)
    register_task_capabilities(caps, store, cancel_resolver=resolver)
    resolver.start()
    try:
        caps.execute("delegate_task", {"goal": "50秒後に今の気持ち教えて", "when_seconds": 50})
        t0 = time.monotonic()
        resolver.submit(reference)
        while resolver._inbox._unfinished_tasks > 0 and time.monotonic() - t0 < 30:
            await asyncio.sleep(0.05)
        dt = time.monotonic() - t0
        actives = [t for t in store.list_all() if t.status in (PENDING, RUNNING)]
        kept = len(actives) == 1
        rep = [s.payload.content for s in q.snapshot() if s.kind == StimulusKind.CALLFUNCTION_RESULT]
        ok = kept == expect_keep
        ev("PASS" if ok else "FAIL",
           f'ref={reference!r} → {"温存" if kept else "取消"}（期待:{"温存" if expect_keep else "取消"}） '
           f'{dt:.2f}s 報告={rep[-1][:40] if rep else "（無）"}')
        return ok
    finally:
        await resolver.stop()
        await store.shutdown()


async def phase_b_same_turn(reg, art_dir):
    """同一ターン delegate+cancel ペア（65μ秒自己取消の再現形）: 生まれたてタスクが温存されるべき。"""
    q = StimulusQueue()
    store = TaskStore(task_file=os.path.join(art_dir, "b_pair.jsonl"))
    await store.initialize()
    caps = CapabilityRegistry(is_busy=lambda: False, qsize=lambda: 0)
    resolver = CancelResolver(store=store, model_registry=reg, queue=q)
    register_task_capabilities(caps, store, cancel_resolver=resolver)
    dispatcher = FunctionDispatcher(registry=caps, queue=q)
    dispatcher.start(); resolver.start()
    try:
        def tc(name, **args):
            return {"id": f"call_{name}", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        dispatcher.submit([
            tc("delegate_task", goal="50秒後に今の気持ちを伝えて", when_seconds=50),
            tc("cancel_task", reference="30秒のやつ"),
        ])
        t0 = time.monotonic()
        while time.monotonic() - t0 < 30:
            if (dispatcher._inbox.empty() and dispatcher._inbox._unfinished_tasks == 0
                    and resolver._inbox.empty() and resolver._inbox._unfinished_tasks == 0):
                break
            await asyncio.sleep(0.05)
        actives = [t for t in store.list_all() if t.status in (PENDING, RUNNING)]
        ok = len(actives) == 1
        ev("PASS" if ok else "FAIL",
           f'同一ターン delegate+cancel(ref=30秒のやつ) → 新規タスク{"温存" if ok else "即死（65μ秒事故の再現）"}')
        return ok
    finally:
        await resolver.stop(); await dispatcher.stop()
        await store.shutdown()


# ============================ main ============================

async def main():
    art = tempfile.mkdtemp(prefix="incident_replay_")
    reg = ModelRegistry(overrides={"response": "openai/gpt-5.5", "task": "openai/gpt-5.5"})
    results_a, results_b = [], []

    if os.getenv("SKIP_B") != "1":
        CUR[0] = "B"
        print("\n═══ Phase B: 決定論 reference マトリクス（実 task LLM）═══")
        for i, (ref, keep) in enumerate(B_MATRIX):
            for j in range(3):
                results_b.append(await phase_b_probe(reg, art, f"{i}_{j}", ref, keep))
        results_b.append(await phase_b_same_turn(reg, art))

    stt = None
    if os.getenv("SKIP_A") != "1":
        print("\n═══ Phase A: 音声リプレイ（30秒→50秒に変更→お願い→発火待ち）═══")
        stt = make_stt()
        await stt.warmup()
        n_runs = int(os.getenv("N_RUNS", "3"))
        for r in range(1, n_runs + 1):
            verdict, why = await phase_a_run(reg, stt, art, r)
            results_a.append((verdict, why))
            print(f"\n  ── run{r}: {verdict} — {why}\n")

    results_c = []
    if os.getenv("SKIP_C") != "1":
        print("\n═══ Phase C: 報告 barge-in → 再配達（2026-07-13 21:20 事故）═══")
        if stt is None:
            stt = make_stt()
            await stt.warmup()
        n_c = int(os.getenv("N_RUNS_C", "1"))
        for r in range(1, n_c + 1):
            for name, fn in (("C1事故再現→救済", phase_c1), ("C2二重発話防止", phase_c2), ("C3複数タスク", phase_c3)):
                verdict, why = await fn(reg, stt, art, r)
                results_c.append((name, verdict, why))
                print(f"\n  ── {name} run{r}: {verdict} — {why}\n")

    results_d = []
    if os.getenv("SKIP_D") != "1":
        print("\n═══ Phase D: タスク待機中の混線ストレス ═══")
        if stt is None:
            stt = make_stt()
            await stt.warmup()
        n_d = int(os.getenv("N_RUNS_D", "1"))
        for r in range(1, n_d + 1):
            for name, fn in (("D1雑談混線+発話前barge", phase_d1),
                             ("D2再生中barge+追い質問", phase_d2),
                             ("D3再配達2連続潰し", phase_d3)):
                verdict, why = await fn(reg, stt, art, r)
                results_d.append((name, verdict, why))
                print(f"\n  ── {name} run{r}: {verdict} — {why}\n")

    print("\n════════════ 判定サマリ ════════════")
    if results_b:
        print(f"Phase B: {sum(results_b)}/{len(results_b)} PASS")
    for i, (v, why) in enumerate(results_a, 1):
        print(f"Phase A run{i}: {v} — {why}")
    for name, v, why in results_c:
        print(f"Phase {name}: {v} — {why}")
    for name, v, why in results_d:
        print(f"Phase {name}: {v} — {why}")
    print(f"artifacts: {art}")
    fail = ((results_b and not all(results_b)) or any(v == "FAIL" for v, _ in results_a)
            or any(v == "FAIL" for _, v, _ in results_c)
            or any(v == "FAIL" for _, v, _ in results_d))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
