r"""J-2 inc2 検索の音声E2E通しテスト（VOICEVOX ユーザ音声 × フル VoiceLoop）。

本番 VoiceLoop（RAG/feedback/VLM/発話判定/タスク/検索すべて実配線）をそのまま起動し、
マイク入力だけスタブ化して「VOICEVOX 合成ユーザ音声 → 実STT → USER_UTTERANCE」で
実機と同じイベント順（発話開始=barge-in → 発話終了 → STT → 刺激投入）を再現する。

シナリオ（J-2 inc2 検証観点）:
  S1 通常検索 + 検索中に別の話題（混線・処理順）
  S2 検索中に別タスク（30秒後の予約と検索報告の共存）
  S3 検索中に別検索（executor single-flight の直列性と両報告）
  S4 検索キャンセル（実行中 goal の取消 → 結果破棄・報告なし）
  S5 検索中に画面移動（メモ帳を開閉・VLM と検索の相互干渉）
  S6 深掘り検索（deep=true の実選択と隔離要約ダイジェスト）
  S7 RAG 想起（過去の検索内容を後から聞く）

実行: $env:PYTHONIOENCODING="utf-8"; ..\portfolio8-VLM-AI\venv\Scripts\python.exe tools\search_e2e_test.py
変数: E2E_ART=artifacts出力先 / REAL_AUDIO=1(実スピーカー再生) / SKIP="S1,S4"(スキップ)
      REAL_STATE=1(**実起動と同じ状態**: .env のフラグ/モデルのまま + 本番の記憶をコピーして使う)
      IDLE_NEGLECT_SEC=180(SIDLE の1ラウンド放置秒) / AUTO_ROUNDS / AUTO_NEGLECT_SEC(SAUTO)
前提: VOICEVOX 起動 + .env のキー。データは artifacts 下に隔離（本物の記憶を汚さない）。
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 応答モデルの上書き（.env の RESPONSE_MODEL より優先・eve.config インポート前に必須）。
# 既定 gpt-5.5: a(tool_calls JSON漏れ)が gpt-5.4-mini→gpt-5.5 で解消するか徹底検証するため。
# REAL_STATE=1 の時は上書きしない（実起動と同じ .env の値で測る）。
if os.getenv("REAL_STATE") != "1":
    os.environ["RESPONSE_MODEL"] = os.environ.get("E2E_RESPONSE_MODEL", "openai/gpt-5.5")

from eve.config import Config  # noqa: E402

ART = os.environ.get("E2E_ART") or os.path.join(
    tempfile.gettempdir(), f"search_e2e_{time.strftime('%H%M%S')}")
os.makedirs(ART, exist_ok=True)

# --- フラグ/データは VoiceLoop 構築前に（Config は構築時に読まれる） ---
# REAL_STATE=1: **実起動(tools/voice_chat.py の VoiceLoop())と同じ状態**で測る。
#   - 機能フラグ/モデルは .env のまま（強制 ON も RESPONSE_MODEL 上書きもしない）
#   - 記憶は本番ファイルを artifacts に**コピーして**使う（起動時の状態は同一・書込は本番を汚さない）
#   自律発話の頻度や RAG 起点の想起は、空の記憶では原理的に測れないため（実測 2026-07-26:
#   隔離モードのRAGは同一セッション5件のみ＝過去の記憶が存在しない）。
REAL_STATE = os.getenv("REAL_STATE") == "1"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seed_from_real(name: str, dest: str) -> str:
    """本番ファイルを artifacts にコピー（無ければ空で開始）。返り値はコピー先パス。"""
    src = os.path.join(_ROOT, name)
    dst = os.path.join(ART, dest)
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst)
    return dst


if REAL_STATE:
    # E2E_FEATURES=1: 記憶は実状態のまま、タスク/検索/Call-Function を明示的に有効化する。
    # .env にこれらのフラグが無いため実起動では無効だが、機能自体のテストには必要
    # （フラグの既定をどうするかは別途ユーザ判断）。
    if os.getenv("E2E_FEATURES") == "1":
        Config.CALLFUNCTION_ENABLED = True
        Config.TASK_ENABLED = True
        Config.SEARCH_ENABLED = True
    Config.HISTORY_FILE = _seed_from_real(Config.HISTORY_FILE, "history.jsonl")
    Config.RAG_FILE = _seed_from_real(Config.RAG_FILE, "rag_memory.jsonl")
    Config.TASK_FILE = _seed_from_real(Config.TASK_FILE, "tasks.jsonl")
else:
    Config.CALLFUNCTION_ENABLED = True
    Config.TASK_ENABLED = True
    Config.SEARCH_ENABLED = True
    Config.HISTORY_FILE = os.path.join(ART, "history.jsonl")
    Config.RAG_FILE = os.path.join(ART, "rag_memory.jsonl")
    Config.TASK_FILE = os.path.join(ART, "tasks.jsonl")

from eve.logsetup import configure  # noqa: E402
from eve.pipeline.stimulus import Stimulus, StimulusKind  # noqa: E402
from eve.task.schema import CANCELLED, DONE, FAILED  # noqa: E402
from eve.voice_loop import VoiceLoop  # noqa: E402

logger = logging.getLogger("e2e")

T0 = time.monotonic()
_tl = open(os.path.join(ART, "timeline.jsonl"), "a", encoding="utf-8")


def ev(tag: str, msg: str, quiet: bool = False, **kw) -> None:
    rec = {"t": round(time.monotonic() - T0, 2), "tag": tag, "msg": msg, **kw}
    if not quiet:
        print(f"{rec['t']:7.1f} [{tag}] {msg}", flush=True)
    _tl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _tl.flush()


def synth(text: str, speaker: int = 8, rate: int = 16000) -> bytes:
    """ユーザ音声の合成（VOICEVOX）。task_incident_replay.py と同一。"""
    import requests
    q = requests.post(f"{Config.VOICEVOX_URL}/audio_query",
                      params={"text": text, "speaker": speaker}, timeout=10).json()
    q["outputSamplingRate"] = rate
    q["outputStereoToMono"] = True
    return requests.post(f"{Config.VOICEVOX_URL}/synthesis", json=q,
                         params={"speaker": speaker}, timeout=30).content


def wav_pcm(b: bytes) -> bytes:
    with wave.open(io.BytesIO(b), "rb") as wf:
        return wf.readframes(wf.getnframes())


def wav_dur(b: bytes) -> float:
    with wave.open(io.BytesIO(b), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 16000)


class DurationSimPlayer:
    """実再生時間を忠実に再現する無音プレイヤー（barge-in 対応・replay と同一）。"""

    async def play_fn(self, audio, should_stop=None):
        if not audio:
            return
        try:
            dur = wav_dur(audio)
        except Exception:
            dur = max(0.5, len(audio) / 48000.0)
        t = 0.0
        while t < dur:
            if should_stop is not None and should_stop():
                return
            await asyncio.sleep(0.02)
            t += 0.02


def screen_activity_loop(duration_sec: float) -> None:
    """放置計測用: duration_sec の間、メモ帳を開いて動かし続ける（VLM notable トリガの誘発）。"""
    import ctypes
    end = time.time() + duration_sec
    try:
        subprocess.Popen(["notepad.exe"])
        time.sleep(3.0)
        u = ctypes.windll.user32
        hwnd = u.FindWindowW("Notepad", None)
        positions = [(100, 100), (700, 350), (250, 500), (850, 120), (400, 200), (600, 450)]
        i = 0
        while time.time() < end:
            try:
                if hwnd:
                    x, y = positions[i % len(positions)]
                    u.MoveWindow(hwnd, x, y, 900, 600, True)
                    i += 1
            except Exception:
                pass
            time.sleep(4.0)  # VLM が変化を拾える程度の間隔で動かす
    except Exception:
        time.sleep(max(0.0, end - time.time()))
    finally:
        subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"], capture_output=True)


def screen_move_sync() -> None:
    """S5: メモ帳を開いて動かして閉じる（実画面変化の発生・純同期=to_thread 実行）。"""
    try:
        subprocess.Popen(["notepad.exe"])
        time.sleep(3.0)
        try:
            import ctypes
            u = ctypes.windll.user32
            hwnd = u.FindWindowW("Notepad", None)
            if hwnd:
                for x, y in ((100, 100), (700, 350), (250, 500), (850, 120)):
                    u.MoveWindow(hwnd, x, y, 900, 600, True)
                    time.sleep(1.5)
        except Exception:
            time.sleep(5.0)
        time.sleep(2.0)
    finally:
        subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"],
                       capture_output=True)


class Harness:
    def __init__(self) -> None:
        self.vl = VoiceLoop()
        if os.getenv("REAL_AUDIO") != "1":
            # 署名が RealAudioPlayer.play_fn と同形（should_stop対応判定は有効なまま）
            self.vl.audio._play_fn = DurationSimPlayer().play_fn
        self.started = asyncio.Event()
        outer = self

        class StubInput:
            async def start(self):
                outer.started.set()

            def stop(self):
                pass

        self.vl.input = StubInput()
        self.handles: list[tuple[float, str, str, str, bool]] = []  # (mono, kind, dedup, text, cancelled)
        self.search_calls: list[dict] = []
        self.cap_calls: list[dict] = []
        self.played: list[tuple[float, str]] = []
        self.latencies: list[dict] = []
        self.search_started: asyncio.Event = asyncio.Event()
        self.cur_turn: dict = {}
        # 自律発話計測: speech_log(deque maxlen=10)は長時間で押し出されるため定期スナップで全捕捉。
        self.decisions: list[dict] = []          # 全発話判定（speak True/False・理由・content）
        self._seen_decision_ts: set = set()      # ts（マイクロ秒精度）で重複排除
        self.suppressions: list[dict] = []        # 🔇 同内容抑制（terminal.log からは取れないので judge 記録）
        self.seed_calls: list[dict] = []          # 発話判定に渡った話題の種（RAG起点かの事後判定用）
        self._instrument()
        self.run_task = None
        self.sampler = None

    # --- 計装（インスタンス属性の wrap のみ・本体コード不変） -----------------
    def _instrument(self) -> None:
        orch = self.vl.orchestrator
        _handle = orch.handle

        # 話題の種の記録: 自律発話が「記憶起点か」を事後に判定するため、判定へ渡った種を残す。
        _autonomous_memories = self.vl.rag.autonomous_memories

        async def seeds_wrap(query, k=3, *, context_since_iso=None):
            seeds = await _autonomous_memories(query, k, context_since_iso=context_since_iso)
            self.seed_calls.append({
                "t": round(time.monotonic() - T0, 2),
                "query": (query or "")[:80],
                "since": context_since_iso,  # ②-3: 自己参照除外の起点（効いているかの一次データ）
                "seeds": [(s.seed_text() if hasattr(s, "seed_text") else s.text)[:120] for s in seeds],
            })
            return seeds

        self.vl.rag.autonomous_memories = seeds_wrap

        async def handle_wrap(stim):
            t0 = time.monotonic()
            dk = getattr(stim, "dedup_key", None) or ""
            payload = getattr(stim, "payload", None)
            ptxt = getattr(payload, "content", None) or (payload if isinstance(payload, str) else "")
            self.cur_turn = {"t0": t0, "kind": stim.kind.name, "first_audio": None}
            ev("▶turn", f"{stim.kind.name} dedup={dk} {str(ptxt)[:70]}")
            try:
                await _handle(stim)
            except asyncio.CancelledError:
                self.handles.append((t0, stim.kind.name, dk, orch.last_response or "", True))
                ev("✂中断", f"{stim.kind.name} dedup={dk}")
                raise
            resp = orch.last_response or ""
            self.handles.append((t0, stim.kind.name, dk, resp, False))
            fa = self.cur_turn.get("first_audio")
            lat = round(fa - t0, 2) if fa else None
            self.latencies.append({"kind": stim.kind.name, "first_audio_sec": lat, "text": resp[:60]})
            ev("🤖", f"[{stim.kind.name}] {resp[:110]}", first_audio_sec=lat)

        orch.handle = handle_wrap  # type: ignore

        _enq = self.vl.audio.enqueue

        def enqueue_wrap(gen, seq, wav, text="", on_played=None):
            if self.cur_turn and self.cur_turn.get("first_audio") is None:
                self.cur_turn["first_audio"] = time.monotonic()

            def played(t):
                self.played.append((time.monotonic(), t))
                if on_played is not None:
                    on_played(t)

            _enq(gen, seq, wav, text=text, on_played=played)

        self.vl.audio.enqueue = enqueue_wrap  # type: ignore

        caps = self.vl.capabilities
        _ex = caps.execute_async

        async def ex_wrap(name, args=None):
            t0 = time.monotonic()
            try:
                args_s = json.dumps(args or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                args_s = str(args)
            ev("⚙call", f"{name}({args_s[:110]})")
            try:
                r = await _ex(name, args)
            except asyncio.CancelledError:
                ev("⚙cancel", f"{name} {time.monotonic() - t0:.2f}s")
                raise
            self.cap_calls.append({"t": round(t0 - T0, 2), "name": name, "args": args_s[:150],
                                   "sec": round(time.monotonic() - t0, 2), "head": r[:100]})
            ev("⚙done", f"{name} {time.monotonic() - t0:.2f}s -> {r[:90]}")
            return r

        caps.execute_async = ex_wrap  # type: ignore

        sc = self.vl.search_client
        if sc is None:
            # REAL_STATE では .env に SEARCH_ENABLED が無ければ検索は配線されない（実起動と同じ）。
            assert REAL_STATE, "SEARCH_ENABLED 配線に失敗（ddgs 未導入?）"
            return
        _search = sc.search

        async def search_wrap(query, fresh=False, deep=False):
            t0 = time.monotonic()
            ev("🔍start", f"q={query!r} fresh={fresh} deep={deep}")
            if not self.search_started.is_set():
                self.search_started.set()
            try:
                out = await _search(query, fresh, deep)
            except asyncio.CancelledError:
                ev("🔍cancel", f"q={query!r} {time.monotonic() - t0:.2f}s")
                raise
            self.search_calls.append({"t": round(t0 - T0, 2), "q": query, "fresh": bool(fresh),
                                      "deep": bool(deep), "sec": round(time.monotonic() - t0, 2),
                                      "head": out[:100]})
            ev("🔍done", f"{time.monotonic() - t0:.2f}s deep={deep} -> {out[:90]}")
            return out

        sc.search = search_wrap  # type: ignore

    # --- ライフサイクル -------------------------------------------------------
    async def start(self) -> None:
        self.run_task = asyncio.create_task(self.vl.run())
        await asyncio.wait_for(self.started.wait(), timeout=600)  # 初回は Ruri ロード込み
        ev("🚀", f"VoiceLoop 稼働（artifacts: {ART} / response={self.vl.registry.resolve('response')}）")
        self.sampler = asyncio.create_task(self._sample())

    async def _sample(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            # 発話判定ログを押し出される前に捕捉（判定は約5〜7秒毎・2秒サンプルで取りこぼさない）。
            for e in list(self.vl.speech_state.speech_log):
                ts = e.get("ts")
                if ts and ts not in self._seen_decision_ts:
                    self._seen_decision_ts.add(ts)
                    rec = {"t": round(time.monotonic() - T0, 2), **e}
                    self.decisions.append(rec)
                    if "抑制" in (e.get("reason") or ""):
                        self.suppressions.append(rec)
            ev("📈", f"surprise={self.vl.prediction.surprise}", quiet=True,
               busy=self.vl.runner.is_busy(), q=self.vl.queue.qsize())

    async def stop(self) -> None:
        if self.sampler is not None:
            self.sampler.cancel()
        try:
            await self.vl.stop()
        except Exception:
            logger.exception("stop で例外")
        if self.run_task is not None:
            self.run_task.cancel()
            try:
                await self.run_task
            except (asyncio.CancelledError, Exception):
                pass

    # --- 駆動（実機のイベント順: barge→発話→utterance→STT→投入） -------------
    async def prep(self, line: str):
        wav = await asyncio.to_thread(synth, line)
        return (line, wav, wav_pcm(wav), wav_dur(wav))

    async def say_prepared(self, prepared, must=()) -> str:
        line, wav, pcm, dur = prepared
        self.vl._barge_in()  # 発話開始（実機は毎発話 VAD onset で発火）
        await asyncio.sleep(min(dur, 6.0))
        self.vl.speech_state.mark_user_utterance()  # 発話終了（STT 前）
        self.vl.speech_state.mark_stt_pending()  # ③-A: 実機 MicSttInputSource と同じ STT 待ち窓
        t_end = time.monotonic()
        text = None
        try:
            text = await self.vl.stt.transcribe(pcm)
        except Exception as e:
            ev("STT失敗", str(e))
        if not text or (must and not all(m in text for m in must)):
            ev("👂誤聴", f"{text!r} → 台本使用（必須語 {list(must)}）")
            text = line
        ev("🧑", text, stt_sec=round(time.monotonic() - t_end, 2))
        await self.vl.queue.put(Stimulus(StimulusKind.USER_UTTERANCE, payload=text))
        self.vl.speech_state.clear_stt_pending()  # ③-A: 投入完了で窓を閉じる（実機と同順）
        return text

    async def say(self, line: str, must=()) -> str:
        return await self.say_prepared(await self.prep(line), must=must)

    # --- 待機/検査ヘルパ ------------------------------------------------------
    def arm_search(self) -> None:
        self.search_started = asyncio.Event()

    async def wait_search_start(self, timeout=90.0) -> bool:
        try:
            await asyncio.wait_for(self.search_started.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            ev("⚠", f"検索開始を {timeout}s 待っても観測できず")
            return False

    def goal_tasks(self):
        return [t for t in self.vl.task_store.list_all() if t.goal]

    async def wait_new_task(self, known_ids, timeout=45.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            for tk in self.goal_tasks():
                if tk.task_id not in known_ids:
                    ev("📋", f"新タスク {tk.task_id} 「{tk.goal[:40]}」 status={tk.status}")
                    return tk
            await asyncio.sleep(0.2)
        ev("⚠", "新タスクが現れなかった")
        return None

    async def wait_report(self, dedup=None, dedup_prefix=None, contains=None,
                          timeout=180.0, after=0.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            for (m, kind, dk, text, cancelled) in self.handles:
                if kind != "CALLFUNCTION_RESULT" or cancelled or m < after:
                    continue
                if dedup is not None and dk != dedup:
                    continue
                if dedup_prefix is not None and not (dk or "").startswith(dedup_prefix):
                    continue
                if contains is not None and contains not in text:
                    continue
                return (m, dk, text)
            await asyncio.sleep(0.3)
        return None

    def _sidecars_idle(self) -> bool:
        # REAL_STATE では .env にフラグが無ければ task/callfunction は配線されない（実起動と同じ）。
        d = self.vl.dispatcher
        r = self.vl.cancel_resolver
        ex = self.vl.task_executor
        return ((d is None or (d._inbox.empty() and d._inbox._unfinished_tasks == 0))
                and (r is None or (r._inbox.empty() and r._inbox._unfinished_tasks == 0))
                and (ex is None or ex.is_idle()))

    async def wait_idle(self, timeout=120.0) -> bool:
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if (not self.vl.runner.is_busy() and self.vl.queue.qsize() == 0
                    and self._sidecars_idle()):
                await asyncio.sleep(0.4)
                if (not self.vl.runner.is_busy() and self.vl.queue.qsize() == 0
                        and self._sidecars_idle()):
                    return True
            await asyncio.sleep(0.15)
        ev("⚠", f"wait_idle timeout {timeout}s（自発発話/長考中の可能性・続行）")
        return False


# ============================ シナリオ ============================

async def s1_search_plus_topic(h: Harness):
    """通常検索 + 検索中に別の話題（混線・処理順・報告の正当性）。"""
    prep_topic = await h.prep("あ、そういえば昨日ね、駅前の新しいラーメン屋さんに行ってきたんだ。すごく美味しかったよ。")
    known = {t.task_id for t in h.goal_tasks()}
    h.arm_search()
    await h.say("ねえイブ、モンハンワイルズの最新アップデートの内容を検索して教えてくれる？", must=("検索",))
    tk = await h.wait_new_task(known)
    await h.wait_search_start()
    await h.say_prepared(prep_topic)  # 検索実行中に別の話題
    if tk is not None:
        rep = await h.wait_report(dedup=f"task:{tk.task_id}", timeout=240)
        ev("✅" if rep else "❌", f"S1 検索報告: {rep[2][:120] if rep else '届かず'}")
    await h.wait_idle()


async def s2_search_plus_task(h: Harness):
    """検索中に別タスク（30秒予約）: 両方が消えず正しい順で届くか。"""
    prep_task = await h.prep("あとね、30秒後に今の時刻を教えて。")
    known = {t.task_id for t in h.goal_tasks()}
    h.arm_search()
    await h.say("VOICEVOXっていうソフトについて検索して教えて。", must=("検索",))
    tk1 = await h.wait_new_task(known)
    await h.wait_search_start()
    await h.say_prepared(prep_task, must=("30",))
    known2 = known | ({tk1.task_id} if tk1 else set())
    tk2 = await h.wait_new_task(known2)
    if tk1 is not None:
        rep1 = await h.wait_report(dedup=f"task:{tk1.task_id}", timeout=240)
        ev("✅" if rep1 else "❌", f"S2 検索報告: {rep1[2][:120] if rep1 else '届かず'}")
    if tk2 is not None:
        rep2 = await h.wait_report(dedup=f"task:{tk2.task_id}", timeout=180)
        ev("✅" if rep2 else "❌", f"S2 30秒予約報告: {rep2[2][:120] if rep2 else '届かず'}")
    await h.wait_idle()


async def s3_search_plus_search(h: Harness):
    """検索中に別検索: single-flight の直列性・両報告・取り違えなし。"""
    prep2 = await h.prep("それとね、世界で一番深い海の名前と深さも調べてほしいな。")
    known = {t.task_id for t in h.goal_tasks()}
    h.arm_search()
    await h.say("日本で一番高い山の標高を検索して教えて。", must=("山",))
    tk1 = await h.wait_new_task(known)
    await h.wait_search_start()
    await h.say_prepared(prep2, must=("海",))
    known2 = known | ({tk1.task_id} if tk1 else set())
    tk2 = await h.wait_new_task(known2)
    if tk1 is not None:
        rep1 = await h.wait_report(dedup=f"task:{tk1.task_id}", timeout=240)
        ev("✅" if rep1 else "❌", f"S3 検索1報告: {rep1[2][:120] if rep1 else '届かず'}")
    if tk2 is not None:
        rep2 = await h.wait_report(dedup=f"task:{tk2.task_id}", timeout=240)
        ev("✅" if rep2 else "❌", f"S3 検索2報告: {rep2[2][:120] if rep2 else '届かず'}")
    await h.wait_idle()


async def s4_cancel_mid_search(h: Harness):
    """検索キャンセル: 実行中 goal の取消 → 結果破棄（報告なし）・取消の一言はある。"""
    prep_cancel = await h.prep("あ、ごめん。やっぱりさっきの検索はやめて。")
    known = {t.task_id for t in h.goal_tasks()}
    h.arm_search()
    await h.say("富士山の初冠雪はいつ頃か検索して教えて。", must=("富士山",))
    tk = await h.wait_new_task(known)
    await h.wait_search_start()
    t_cancel = time.monotonic()
    await h.say_prepared(prep_cancel, must=("やめ",))
    rep_c = await h.wait_report(dedup_prefix="cancel:", timeout=90, after=t_cancel)
    ev("✅" if rep_c else "❌", f"S4 取消の返答: {rep_c[2][:100] if rep_c else '届かず'}")
    if tk is not None:
        # 取消後にタスク報告が来ない（結果破棄）ことを 45s 観測
        ghost = await h.wait_report(dedup=f"task:{tk.task_id}", timeout=45, after=t_cancel)
        st = h.vl.task_store.get(tk.task_id)
        ev("✅" if (ghost is None and st is not None and st.status == CANCELLED) else "❌",
           f"S4 取消後: 報告={'無し(正)' if ghost is None else '来た(誤)'} status={st.status if st else '?'}")
    await h.wait_idle()


async def s5_screen_move_mid_search(h: Harness):
    """検索中に画面移動: VLM の実況と検索が互いを壊さないか。"""
    known = {t.task_id for t in h.goal_tasks()}
    h.arm_search()
    await h.say("ポケモンの最新作がどうなってるか検索して教えて。", must=("ポケモン",))
    tk = await h.wait_new_task(known)
    await h.wait_search_start()
    ev("🖥", "メモ帳を開いて動かす（画面変化の発生）")
    mover = asyncio.create_task(asyncio.to_thread(screen_move_sync))
    if tk is not None:
        rep = await h.wait_report(dedup=f"task:{tk.task_id}", timeout=240)
        ev("✅" if rep else "❌", f"S5 検索報告: {rep[2][:120] if rep else '届かず'}")
    await mover
    await h.wait_idle()


async def s6_deep_search(h: Harness):
    """深掘り検索: deep=true が選ばれ隔離要約ダイジェストで報告されるか。"""
    known = {t.task_id for t in h.goal_tasks()}
    calls_before = len(h.search_calls)  # J-2バグ修正: セッション累積でなく S6 中の呼び出しだけ見る
    h.arm_search()
    await h.say("超かぐや姫っていう映画について、ページの中身まで読んで詳しく調べてまとめてほしいな。",
                must=("かぐや",))
    tk = await h.wait_new_task(known)
    await h.wait_search_start()
    if tk is not None:
        rep = await h.wait_report(dedup=f"task:{tk.task_id}", timeout=300)
        ev("✅" if rep else "❌", f"S6 深掘り報告: {rep[2][:150] if rep else '届かず'}")
    new_calls = h.search_calls[calls_before:]
    deep_used = any(c["deep"] for c in new_calls)
    ev("✅" if deep_used else "⚠", f"S6 deep=true の実選択: {deep_used}（S6中の呼び出し{len(new_calls)}件）")
    await h.wait_idle()


async def s7_rag_recall(h: Harness):
    """RAG 想起: 最初の検索内容（直近6ターン外）を後から聞く。"""
    await h.say("ねえ、最初の方に調べてくれたモンハンのアップデートって、どんな内容だったっけ？",
                must=("モンハン",))
    await h.wait_idle(150)


async def s8_barge_in_mid_speech(h: Harness):
    """発話途中の明示的な割り込み: 話している最中に別発話で遮り、自然に切り替わるか。"""
    prep_new = await h.prep("ごめん、ちょっと聞きたいことがあるんだけど、VOICEVOXの読み方のコツってある？")
    await h.say("最近見た中で面白かった話ある？なんでもいいから話してみて。")
    t0 = time.monotonic()
    while h.cur_turn.get("first_audio") is None and time.monotonic() - t0 < 20:
        await asyncio.sleep(0.05)
    if h.cur_turn.get("first_audio") is None:
        ev("⚠", "S8: 発話開始を検知できなかった（割り込みタイミングは目視で確認）")
    else:
        await asyncio.sleep(1.2)  # 発話が乗っている最中に割り込む
    t_barge = time.monotonic()
    await h.say_prepared(prep_new)
    await h.wait_idle(60)
    # 判定は「barge 直前30秒以内に開始した USER ターンが中断された」（handles の m はターン開始時刻。
    # 旧判定 m >= t_barge-3 は長めの前置き応答だと開始が3秒より前になり False になる計測癖があった）
    interrupted = any(
        t_barge - 30 <= m <= t_barge and k == "USER_UTTERANCE" and cancelled
        for (m, k, d, tx, cancelled) in h.handles)
    ev("✅" if interrupted else "⚠", f"S8 割り込みで前ターンの中断記録: {interrupted}（詳細はtimeline参照）")


async def s9_multi_task_burst(h: Harness):
    """複数タスクをほぼ同時に連投: 3件がすべて欠落/混線なく届くか。"""
    known = {t.task_id for t in h.goal_tasks()}
    await h.say("東京スカイツリーの高さを調べて教えて。", must=("スカイツリー",))
    await h.say("それとエベレストの標高も調べてほしいな。", must=("エベレスト",))
    await h.say("あと琵琶湖の面積も教えて。", must=("琵琶湖",))
    new_tasks: list = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and len(new_tasks) < 3:
        new_tasks = [t for t in h.goal_tasks() if t.task_id not in known]
        await asyncio.sleep(0.3)
    ev("📋", f"S9 burst: 新規タスク{len(new_tasks)}件（期待3件）: {[t.goal[:15] for t in new_tasks]}")
    for tk in new_tasks:
        rep = await h.wait_report(dedup=f"task:{tk.task_id}", timeout=240)
        ev("✅" if rep else "❌", f"S9 報告[{tk.goal[:20]}]: {rep[2][:100] if rep else '届かず'}")
    await h.wait_idle()


async def s10_extended_neglect(h: Harness):
    """長時間放置: 自律発話の頻度・内容の多様性・単調な繰り返しの有無を観察する。"""
    t_start = time.monotonic()
    neglect_sec = float(os.getenv("S10_NEGLECT_SEC", "200"))
    ev("💤", f"{neglect_sec:.0f}秒放置開始（自律発話を観察・画面はそのまま）")
    await asyncio.sleep(neglect_sec)
    autos = [(m, tx) for (m, k, d, tx, c) in h.handles if k == "AUTONOMOUS_SPEECH" and m >= t_start and not c]
    ev("📊", f"S10 放置中の自律発話 {len(autos)}件")
    for m, tx in autos:
        ev("🗣", f"  +{m - t_start:.0f}s: {tx[:90]}")
    await h.say("ただいま、待たせてごめんね。")
    await h.wait_idle(60)


async def s11_repeated_overlap_stress(h: Harness):
    """反復ストレス: 「事実A→直後に事実B」ペアを毎回別の題材で繰り返し、
    a(JSON漏れ)/b(数値歪曲)/c(先回り)型の不具合が gpt-5.5 でも再発しないか確認する。"""
    pairs = [
        ("北海道の面積を検索して教えて。", "それと沖縄の人口も調べてほしいな。", ("北海道",), ("沖縄",)),
        ("琵琶湖の水深を検索して教えて。", "それと東京タワーの高さも教えて。", ("琵琶湖",), ("東京タワー",)),
        ("大阪城の築城年を検索して教えて。", "それと日本一長い川の名前も調べてほしいな。", ("大阪城",), ("川",)),
    ]
    for i, (q1, q2, must1, must2) in enumerate(pairs):
        known = {t.task_id for t in h.goal_tasks()}
        await h.say(q1, must=must1)
        tk1 = await h.wait_new_task(known, timeout=20)
        await asyncio.sleep(1.0)
        await h.say(q2, must=must2)
        known2 = known | ({tk1.task_id} if tk1 else set())
        tk2 = await h.wait_new_task(known2, timeout=20)
        if tk1 is not None:
            rep1 = await h.wait_report(dedup=f"task:{tk1.task_id}", timeout=200)
            ev("✅" if rep1 else "❌", f"S11-{i} 報告1: {rep1[2][:100] if rep1 else '届かず'}")
        if tk2 is not None:
            rep2 = await h.wait_report(dedup=f"task:{tk2.task_id}", timeout=200)
            ev("✅" if rep2 else "❌", f"S11-{i} 報告2: {rep2[2][:100] if rep2 else '届かず'}")
        await h.wait_idle(60)


async def s_auto_speech(h: Harness):
    """自律発話の頻度・間隔・同話題連続性の計測（複数文脈の放置ラウンド）。

    各ラウンド: 短い区切り発話 → wait_idle → 放置。放置中の自律発話を全記録。
    文脈を変えるのは decide_fn の判断（返事待ち→黙る / 手が空いた合図→話す）を多面的に見るため。
    画面変化ラウンドは VLM notable トリガ経路も込みで観測する。
    """
    rounds = int(os.getenv("AUTO_ROUNDS", "4"))
    neglect = float(os.getenv("AUTO_NEGLECT_SEC", "240"))
    # (区切り発話, 画面変化させるか) — 返事待ちになりにくい「離脱/独白」系で自律発話を誘発。
    starters = [
        ("じゃあちょっと自分の作業に集中するね。何かあったら声かけて。", False),
        ("今日はのんびりしようかな。", False),
        ("ちょっとこのまま画面で作業するね、見てて。", True),
        ("うーん、次何しようかな。", False),
    ]
    for i in range(rounds):
        line, screen_move = starters[i % len(starters)]
        ev("═══", f"AUTO-round{i}（画面変化={screen_move}・放置{neglect:.0f}s）")
        await h.say(line)
        await h.wait_idle(30)
        t_round = time.monotonic()
        base_auto = sum(1 for (m, k, d, tx, c) in h.handles if k == "AUTONOMOUS_SPEECH")
        mover = None
        if screen_move:
            mover = asyncio.create_task(asyncio.to_thread(screen_activity_loop, neglect))
        await asyncio.sleep(neglect)
        if mover is not None:
            await mover
        autos = [(round(m - t_round, 1), tx, c)
                 for (m, k, d, tx, c) in h.handles if k == "AUTONOMOUS_SPEECH" and m >= t_round]
        ev("📊", f"AUTO-round{i} 自律発話 {len(autos)}件（放置開始からの秒・中断含む）")
        for sec, tx, cancelled in autos:
            ev("🗣", f"  +{sec}s {'[中断]' if cancelled else ''} {tx[:80]}")
        await h.wait_idle(20)


async def s_idle3(h: Harness):
    """3分放置を複数ラウンド: 実運用で「どれくらいの頻度で話しかけてくるか」を測る。

    画面は触らない（VLM 起因を混ぜない）。文脈は自然な区切り発話のみ、または起動直後の純沈黙。
    「作業に集中する/見てて」系は decider が正しく黙るため、頻度計測には使わない。
    """
    neglect = float(os.getenv("IDLE_NEGLECT_SEC", "180"))
    starters = [
        None,  # 起動直後の純沈黙（会話文脈なし）
        "ふー、ちょっと一息つこうかな。",
        "特に予定はないんだよね。",
    ]
    for i, line in enumerate(starters):
        ev("═══", f"IDLE-round{i}（{'区切り発話なし' if line is None else line}・放置{neglect:.0f}s）")
        if line is not None:
            await h.say(line)
            await h.wait_idle(30)
        t_round = time.monotonic()
        await asyncio.sleep(neglect)
        autos = [(round(m - t_round, 1), tx, c)
                 for (m, k, d, tx, c) in h.handles if k == "AUTONOMOUS_SPEECH" and m >= t_round]
        rate = len(autos) / (neglect / 60.0)
        ev("📊", f"IDLE-round{i} 自律発話 {len(autos)}件 = {rate:.1f}件/分（放置{neglect:.0f}s）")
        for sec, tx, cancelled in autos:
            ev("🗣", f"  +{sec}s {'[中断]' if cancelled else ''} {tx[:90]}")
        await h.wait_idle(20)


async def s_vlm_states(h: Harness):
    """画面認識まわりの全場面（J-2 ③ 沈黙バイアス撤廃の回帰）。

    1. 静止画面で「今何が見える?」→ 変化していない間は据え置きで答えられる（層分離）
    2. 画面変化中 → 画面に触れた応答/自発発話ができる
    3. 静止で放置 → 自発発話が画面を**捏造しない**
    4. イブの発話中にユーザが割り込む → 中断され、その後の応答が壊れない
    判定は自動化せず一次データ（timeline/terminal.log）に残す。捏造だけ自動チェックする。
    """
    ev("═══", "VLM-1 静止画面で『今何が見えてる?』（据え置きが働くか・捏造しないか）")
    await asyncio.sleep(25)  # 画面を触らない＝VLM が止まり latest_vision が TTL 超過する
    await h.say("ねえ、今わたしの画面って何が見えてる？")
    await h.wait_idle(40)

    ev("═══", "VLM-2 画面変化中（メモ帳を開いて動かす）")
    mover = asyncio.create_task(asyncio.to_thread(screen_move_sync))
    await asyncio.sleep(8)  # 変化フレームが VLM に届くのを待つ
    await h.say("今、画面どうなってる？")
    await h.wait_idle(40)
    await mover

    ev("═══", "VLM-3 静止に戻して放置150s（自発発話が画面を捏造しないか）")
    t0 = time.monotonic()
    await asyncio.sleep(150)
    autos = [(round(m - t0, 1), tx) for (m, k, d, tx, c) in h.handles
             if k == "AUTONOMOUS_SPEECH" and m >= t0]
    bad = [(s, tx) for s, tx in autos
           if re.search(r"画面|表示|見え(て|る)|映っ|ウィンドウ|タブ", tx)
           and not re.search(r"画面[^。]{0,8}(見えて(い)?な|分からな|無い|ない)", tx)]
    ev("📊", f"VLM-3 自発発話 {len(autos)}件 / 画面に言及した疑い {len(bad)}件")
    for s, tx in autos:
        ev("🗣", f"  +{s}s {tx[:90]}")
    for s, tx in bad:
        ev("⚠", f"  画面捏造の疑い: {tx[:90]}")

    ev("═══", "VLM-4 発話中の割り込みとその後")
    prep = await h.prep("ごめん、今の話じゃなくて、ちょっと別のこと聞いていい？")
    await h.say("なんでもいいから、最近の話をひとつ聞かせて。")
    t1 = time.monotonic()
    while h.cur_turn.get("first_audio") is None and time.monotonic() - t1 < 20:
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.2)  # 発話が乗っている最中に割り込む
    t_barge = time.monotonic()
    await h.say_prepared(prep)
    await h.wait_idle(60)
    interrupted = any(t_barge - 30 <= m <= t_barge and k == "USER_UTTERANCE" and c
                      for (m, k, d, tx, c) in h.handles)
    ev("✅" if interrupted else "⚠", f"VLM-4 割り込みで前ターン中断: {interrupted}")
    await asyncio.sleep(45)  # 割り込み直後の自発発話の様子も残す
    await h.wait_idle(30)


async def s_speech_sources(h: Harness):
    """自発発話の**由来**を切り分ける（画面 / 直近会話 / 記憶）。

    ユーザ要望「RAGだけでなく画面認識や直近会話からも自発発話するか」の確認。
    各フェーズの自発発話を記録し、事後に語彙・画面ブロック有無で帰属を判定する。
    """
    dur = float(os.getenv("SRC_PHASE_SEC", "120"))

    ev("═══", f"SRC-1 画面が動いている最中に放置{dur:.0f}s（画面起点が出るか）")
    t0 = time.monotonic()
    mover = asyncio.create_task(asyncio.to_thread(screen_activity_loop, dur))
    await asyncio.sleep(dur)
    await mover
    for (m, k, d, tx, c) in h.handles:
        if k == "AUTONOMOUS_SPEECH" and m >= t0:
            ev("🗣", f"  SRC-1 +{m - t0:.0f}s {tx[:95]}")

    ev("═══", f"SRC-2 具体的な話題を振ってから放置{dur:.0f}s（会話起点が出るか）")
    await h.say("最近ぜんぜんゲームやってなくてさ、積んでるソフトばっかり増えてるんだよね。")
    await h.wait_idle(40)
    t1 = time.monotonic()
    await asyncio.sleep(dur)
    for (m, k, d, tx, c) in h.handles:
        if k == "AUTONOMOUS_SPEECH" and m >= t1:
            ev("🗣", f"  SRC-2 +{m - t1:.0f}s {tx[:95]}")

    ev("═══", f"SRC-3 中立な区切りから放置{dur:.0f}s（記憶起点が出るか）")
    await h.say("うん、まあそんな感じかな。")
    await h.wait_idle(40)
    t2 = time.monotonic()
    await asyncio.sleep(dur)
    for (m, k, d, tx, c) in h.handles:
        if k == "AUTONOMOUS_SPEECH" and m >= t2:
            ev("🗣", f"  SRC-3 +{m - t2:.0f}s {tx[:95]}")
    await h.wait_idle(30)


SCENARIOS = [
    ("SSRC_自発発話の由来", s_speech_sources),
    ("SVLM_画面認識の全場面", s_vlm_states),
    ("SIDLE_3分放置頻度", s_idle3),
    ("SAUTO_自律発話計測", s_auto_speech),
    ("S1_通常検索+別話題", s1_search_plus_topic),
    ("S2_検索中に別タスク", s2_search_plus_task),
    ("S3_検索中に別検索", s3_search_plus_search),
    ("S4_検索キャンセル", s4_cancel_mid_search),
    ("S5_検索中に画面移動", s5_screen_move_mid_search),
    ("S6_深掘り検索", s6_deep_search),
    ("S7_RAG想起", s7_rag_recall),
    ("S8_発話中割り込み", s8_barge_in_mid_speech),
    ("S9_複数タスク連投", s9_multi_task_burst),
    ("S10_長時間放置", s10_extended_neglect),
    ("S11_反復ストレスabc再現", s11_repeated_overlap_stress),
]


async def main() -> None:
    configure()
    root = logging.getLogger()
    fh = logging.FileHandler(os.path.join(ART, "terminal.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    logging.getLogger("primp").setLevel(logging.WARNING)  # ddgs auto の HTTP ノイズ

    skip = set((os.getenv("SKIP") or "").replace(" ", "").split(","))
    h = Harness()
    await h.start()
    try:
        for name, fn in SCENARIOS:
            if name.split("_")[0] in skip:
                ev("⏭", f"skip {name}")
                continue
            ev("═══", name)
            try:
                await fn(h)
            except Exception:
                logger.exception("シナリオ %s で例外（続行）", name)
                ev("💥", f"{name} で例外（terminal.log 参照）")
            await asyncio.sleep(2.0)
    finally:
        # 総括 artifacts
        summary = {
            "handles": [{"t": round(m - T0, 2), "kind": k, "dedup": d,
                         "text": tx[:200], "cancelled": c}
                        for (m, k, d, tx, c) in h.handles],
            "search_calls": h.search_calls,
            "cap_calls": h.cap_calls,
            "latencies": h.latencies,
            "played_chars": sum(len(t) for _, t in h.played),
            "speech_log": [str(x) for x in list(h.vl.speech_state.speech_log)],
            "decisions": h.decisions,       # 自律発話計測: 全判定（押し出し前に捕捉した全件）
            "suppressions": h.suppressions,  # 同内容抑制（②-1 の発火）
            "seed_calls": h.seed_calls,      # 判定へ渡った話題の種（RAG起点かの事後判定用）
            "tasks_final": [{"id": t.task_id, "goal": t.goal, "status": t.status,
                             "result": (t.result or "")[:120]}
                            for t in (h.vl.task_store.list_all() if h.vl.task_store else [])],
        }
        with open(os.path.join(ART, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        ev("🏁", f"シナリオ完了 → 停止処理へ（summary.json 書出済）")
        await h.stop()
        ev("🏁", "停止完了")


if __name__ == "__main__":
    asyncio.run(main())
