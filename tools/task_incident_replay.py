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
from eve.stt import make_stt  # noqa: E402
from eve.task import (  # noqa: E402
    CANCELLED, DONE, FAILED, PENDING, RUNNING,
    CancelResolver, ReconcileTimer, TaskAgent, TaskExecutor, TaskStore, register_task_capabilities,
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
        self.handles = []  # (kind, dedup_key, 生成テキスト) の記録

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
            if tools:
                async for x in self.reg.stream_with_tools("response", messages, tools=tools,
                                                          tool_sink=tool_sink):
                    yield x
            else:
                async for x in self.reg.stream("response", messages):
                    yield x

        orch = ResponseOrchestrator(self.audio, stream_fn, self.tts.generate,
                                    ContextAssembler(system_prompt=SPEECH_STYLE),
                                    conversation_cache=self.cache, dispatcher=self.dispatcher)
        _handle = orch.handle
        world = self

        async def handle_wrap(stim):
            await _handle(stim)
            world.handles.append((stim.kind, getattr(stim, "dedup_key", None), orch.last_response or ""))
            tag = {StimulusKind.USER_UTTERANCE: "🤖", StimulusKind.CALLFUNCTION_RESULT: "🛠報告"}.get(stim.kind, "🤖?")
            ev(tag, (orch.last_response or "（空）")[:90])
        orch.handle = handle_wrap  # type: ignore
        self.orch = orch
        self.runner = PipelineRunner(self.queue, orch, self.audio)
        self._bg = [asyncio.create_task(self.audio.play_worker()), asyncio.create_task(self.runner.run())]
        await self.store.initialize()
        self.dispatcher.start(); self.executor.start(); self.timer.start(); self.resolver.start()

    async def stop(self):
        await self.timer.stop(); await self.resolver.stop(); await self.executor.stop()
        await self.dispatcher.stop()
        for t in self._bg:
            t.cancel()
        await asyncio.sleep(0.2)
        await self.cache.shutdown(); await self.store.shutdown(); await self.tts.close()
        if hasattr(self.player, "close"):
            self.player.close()

    # --- 駆動ヘルパ ---------------------------------------------------------

    def barge(self):
        self.audio.interrupt(); self.runner.interrupt()

    def _sidecars_idle(self):
        return (self.dispatcher._inbox.empty() and self.dispatcher._inbox._unfinished_tasks == 0
                and self.resolver._inbox.empty() and self.resolver._inbox._unfinished_tasks == 0)

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
                            for k, d, sp in w.handles)
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

    if os.getenv("SKIP_A") != "1":
        print("\n═══ Phase A: 音声リプレイ（30秒→50秒に変更→お願い→発火待ち）═══")
        stt = make_stt()
        await stt.warmup()
        n_runs = int(os.getenv("N_RUNS", "3"))
        for r in range(1, n_runs + 1):
            verdict, why = await phase_a_run(reg, stt, art, r)
            results_a.append((verdict, why))
            print(f"\n  ── run{r}: {verdict} — {why}\n")

    print("\n════════════ 判定サマリ ════════════")
    if results_b:
        print(f"Phase B: {sum(results_b)}/{len(results_b)} PASS")
    for i, (v, why) in enumerate(results_a, 1):
        print(f"Phase A run{i}: {v} — {why}")
    print(f"artifacts: {art}")
    fail = (results_b and not all(results_b)) or any(v == "FAIL" for v, _ in results_a)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
