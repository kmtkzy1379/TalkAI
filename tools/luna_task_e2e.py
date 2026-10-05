#!/usr/bin/env python3
r"""ルナ（応答）× フィラー初期案（書き出し）× タスク4種 の通し確認（本番 UI＋本番 VoiceLoop・実 API・時間上限つき）。

ユーザ 2026-09-28:
  「フィラー初期案…それをシステムプロンプトに入れてルナで応答検証。検証内容は
   ポケモン図鑑番号９８７は？（これで検索をするか、応答AIのみで答えられるか、ハルシネーションしなければ合格）
   PCの調子悪いかも。（意図をくみ取り自律的にタスクを実行できれば合格）
   電卓作って（バグなく電卓作って画面に出せたら合格）
   おすすめのコーヒー画像見せて（ネットから画像拾ってこれれば合格）
   その他応答の自然さや速度は私が確認するため全会話報告」

構成:
  - 本番の UI（`tools/ui_chat.py` と同じ MainWindow）を起動する。HTML が**押せる状態で**動くのは
    UI のタスクウィンドウだけ（ヘッドレスの別プロセス表示はソースを出す＝`eve/workspace/viewer.py`）
  - 応答モデル・reasoning_effort・書き出しの表は**本番の設定のまま**（2026-09-29 採用: .env の RESPONSE_MODEL=gpt-6-luna・
    RESPONSE_REASONING_EFFORT=none／fillers.yaml の opener_prompt）。最初の検証（2026-09-28）はこのツールの中で差し込んでいた
  - マイクだけ差し替え: ユーザの発話を Irodori で作り、本番と同じ VAD → ストリーミング文字起こしへ実時間で流す
    （フィラー・笑い・感情・場面は本番どおり動く）。スピーカーからは本番どおり音が出る
  - 画面認識と自発発話は切る（画面を送らない・検証と関係ない発話を混ぜない）。記憶は出力フォルダで空から
  - タスク担当（TaskAgent）・検索・作業フォルダは .env のまま
  - 電卓は、タスクウィンドウに出た HTML の中で JS からボタンを押して表示を読む（実際に表示されたものを試す）
出力（`e2e_logs/luna_<時刻>/`）: report.md（全会話・時刻）/ terminal.log / events.json /
  conversation.wav・turnN.wav（ユーザの声＋フィラー＋応答の録音）/ taskwin_*.png / 作られたファイルの写し

実行（TalkAI-live のフォルダで・EVE の venv）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\luna_task_e2e.py
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent

# (名前, 発話, 合格条件, 必須語)。必須語が文字起こしに無ければ台本の文を使う（合成音声の聞き違いで
# 別の依頼にすり替わるのを防ぐ・旧 `search_e2e_test.py` の「誤聴→台本使用」と同じ。報告に両方残す）。
# 実測 2026-09-28 1回目: 「おすすめのコーヒー画像見せて」→「おすすめのコーヒーどうぞ、見せて。」で「画像」が落ち、
# タスクが画像ではなく HTML のおすすめカードになった。
TESTS = [
    ("ポケモン", "ポケモン図鑑番号987は？",
     "検索をするか、応答AIのみで答えられるか。ハルシネーションしなければ合格（987 は ハバタクカミ）", ("ポケモン",)),
    ("PC", "PCの調子悪いかも。", "意図をくみ取り自律的にタスクを実行できれば合格", ("PC",)),
    ("電卓", "電卓作って", "バグなく電卓作って画面に出せたら合格", ("電卓",)),
    ("コーヒー", "おすすめのコーヒー画像見せて", "ネットから画像拾ってこれれば合格", ("コーヒー", "画像")),
]
# 通常会話（ユーザ 2026-09-28「今のやり取りだと｢やっとくね｣しかなく通常会話がないため通常会話シナリオを適当に作り検証して」）。
# イブの返事に依らず流れがつながる並び（疲れ → 不満 → 驚き・かわいい → 困った → イブへの質問 → 予定 → 締め）。
CHAT = [
    ("ただいま", "ただいま。今日めっちゃ疲れたー。", "通常会話", ("疲れ",)),
    ("会議", "会議が三時間もあってさ、結局なにも決まらなかったんだよね。", "通常会話", ("会議",)),
    ("猫", "そういえば帰り道に、猫がずっと後ろをついてきたんだよ。", "通常会話", ("猫",)),
    ("困った", "家の前まで来ちゃったから、ちょっと困った。", "通常会話", ("困",)),
    ("質問", "イブは猫と犬だったら、どっちが好き？", "通常会話", ("犬",)),
    ("予定", "明日は休みだから、昼まで寝ようかな。", "通常会話", ("休み",)),
    ("締め", "じゃあ、そろそろご飯作ってくるね。", "通常会話", ("ご飯",)),
]
SCENARIOS = {"tasks": TESTS, "chat": CHAT}
# 合成した声に VAD が反応しなかった時は、語は変えず句読点だけ変えて合成し直す（`Harness.speak`）。
# 実測 2026-09-28 1回目: 「電卓作って」の合成音声は VAD が 3枠しか反応せず（最大 0.65）発話にならなかった
# （Irodori の seed は固定なので同じ文は同じ音になる）。
MIN_SPEECH_FRAMES = 10  # 32ms × 10 = 0.32秒

T0 = time.monotonic()
EVENTS: list = []   # (t_mono, 種類, 文)
LOGS: list = []     # (t_mono, logger 名, 文)
log = logging.getLogger("luna_e2e")


def ev(kind: str, text: str) -> None:
    EVENTS.append((time.monotonic(), kind, text))
    log.info("%s %s", kind, text)


def setup_env(art: Path, filler_voice: str = "") -> None:
    """eve を import する前に環境変数で上書きする（.env は既存の値を上書きしない）。"""
    if filler_voice:
        os.environ["FILLER_VOICE"] = filler_voice  # 例 eve_calm（短い版と聴き比べる時）。無ければ .env のまま
    os.environ.update({
        "VLM_ENABLED": "0", "AUTOSPEECH_ENABLED": "0",
        "HISTORY_FILE": str(art / "history.jsonl"), "RAG_FILE": str(art / "rag.jsonl"),
        "TASK_FILE": str(art / "tasks.jsonl"), "SELF_MODEL_FILE": str(art / "self_model.jsonl"),
        "USAGE_LOG_ENABLED": "1", "USAGE_LOG_FILE": str(art / "usage.jsonl"),
    })


class Capture(logging.Handler):
    """eve.* の INFO 以上を時刻つきで貯める（報告の時系列に使う）。"""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.INFO:
            return
        if not (record.name.startswith("eve") or record.name == "luna_e2e"):
            return
        try:
            LOGS.append((time.monotonic(), record.name, record.getMessage()))
        except Exception:
            pass


class Recorder:
    """鳴らした音（ユーザの声・フィラー・応答）を鳴らし始めた時刻で並べて、あとで1本に混ぜる。"""

    def __init__(self) -> None:
        self.items: list = []  # (t_mono, sr, float32 mono, 種類)

    def add_wav(self, wav, kind: str) -> None:
        import numpy as np

        from eve.filler import audio as fa

        if not wav:
            return
        try:
            x, sr = fa.read_wav(wav)  # float32 モノラル（PCM16 でも float でも）
            self.items.append((time.monotonic(), sr, np.asarray(x, np.float32).reshape(-1), kind))
        except Exception:
            log.exception("録音に足せなかった（続行）")

    def add_array(self, x, sr: int, kind: str, t: float) -> None:
        self.items.append((t, sr, x, kind))

    def render(self, t_from: float, t_to: float, sr_out: int = 24000):
        import numpy as np

        n = max(1, int((t_to - t_from) * sr_out))
        out = np.zeros(n, np.float32)
        for (t, sr, x, _k) in self.items:
            if t > t_to or t + len(x) / sr < t_from:
                continue
            y = x if sr == sr_out else np.interp(
                np.arange(int(len(x) * sr_out / sr)) * (sr / sr_out), np.arange(len(x)), x).astype(np.float32)
            i0 = int(round((t - t_from) * sr_out))
            a, b = max(0, i0), min(n, i0 + len(y))
            if b > a:
                out[a:b] += y[a - i0:b - i0]
        return np.clip(out, -1.0, 1.0)


class FakeMic:
    """マイクの代わり（`MicSttInputSource._ai` に入る）。UI の状態表示が読む属性も持つ。"""

    running = True
    speech_prob = 0.0

    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()
        self.tracker = None

    async def get_audio(self) -> bytes:
        return await self.q.get()

    def in_speech(self) -> bool:
        return self.tracker is not None and self.tracker.is_speaking

    def stop(self) -> None:
        pass


# 電卓の確認（タスクウィンドウの HTML の中で実行する・同期で終わる）
CALC_JS = r"""
(() => {
  const out = {buttons: [], display: null, tests: [], errors: [], note: ''};
  window.__eveErrs = window.__eveErrs || [];
  window.addEventListener('error', e => window.__eveErrs.push(String(e.message)));
  const norm = s => String(s == null ? '' : s).replace(/\s+/g, '');
  const SEL = 'button, input[type=button], input[type=submit], [role=button], [onclick], [data-value], [data-key], [data-num], [data-op], [data-action], .btn, .button, .key';
  const btns = Array.from(document.querySelectorAll(SEL));
  const label = e => norm(e.tagName === 'INPUT' ? e.value : e.textContent);
  out.buttons = btns.map(label).slice(0, 60);
  const ALIAS = {'*': ['×', '*', 'x', 'X', '✕', '✖', '＊'], '/': ['÷', '/', '／'], '-': ['−', '-', '－', '–', 'ー'],
                 '+': ['+', '＋'], '=': ['=', '＝'], 'C': ['AC', 'C', 'CE', 'クリア', 'Clear', 'CLEAR', 'CLR', 'ＡＣ', 'Ｃ'],
                 '.': ['.', '．']};
  const find = k => {
    for (const n of (ALIAS[k] || [k])) {
      const e = btns.find(b => label(b) === n || b.getAttribute('data-value') === n || b.getAttribute('data-key') === n);
      if (e) return e;
    }
    return null;
  };
  const press = seq => { for (const k of seq) { const e = find(k); if (!e) return 'ボタンが無い: ' + k; e.click(); } return null; };
  const cands = Array.from(document.querySelectorAll('input, output, textarea, div, span, p, h1, h2, h3, pre, td'))
    .filter(e => !btns.includes(e) && !btns.some(b => b.contains(e)));
  const val = e => (e.tagName === 'INPUT' || e.tagName === 'TEXTAREA') ? e.value : e.textContent;
  press(['C']);
  const before = cands.map(val);
  const miss = press(['7']);
  if (miss) { out.note = miss; out.errors = window.__eveErrs.slice(0, 10); return JSON.stringify(out); }
  const changed = cands.filter((e, i) => val(e) !== before[i] && norm(val(e)).includes('7'));
  changed.sort((a, b) => norm(val(a)).length - norm(val(b)).length);
  const disp = changed[0];
  if (!disp) { out.note = '表示欄が見つからない'; out.errors = window.__eveErrs.slice(0, 10); return JSON.stringify(out); }
  out.display = disp.tagName + (disp.id ? '#' + disp.id : '') + (disp.className ? '.' + String(disp.className).trim().split(/\s+/).join('.') : '');
  const TESTS = [
    [['1', '+', '2', '='], '3'],
    [['7', '*', '8', '='], '56'],
    [['9', '/', '3', '='], '3'],
    [['5', '-', '8', '='], '-3'],
    [['1', '2', '.', '5', '+', '0', '.', '5', '='], '13'],
    [['0', '.', '1', '+', '0', '.', '2', '='], '0.3'],
    [['2', '+', '3', '*', '4', '='], '14 か 20（計算の順の作りによる）'],
    [['1', '/', '0', '='], 'エラー等の表示（固まらない）'],
    [['1', '+', '2', '=', '5'], '5（結果の後に打つと新しい数）'],
    [['8', 'C'], '0 か 空'],
  ];
  for (const [seq, expect] of TESTS) {
    press(['C']);
    const m = press(seq);
    out.tests.push({seq: seq.join(' '), got: m || norm(val(disp)), expect});
  }
  out.errors = window.__eveErrs.slice(0, 10);
  return JSON.stringify(out);
})()
"""


class Harness:
    """VoiceLoop のループ上で動く（UI スレッドの物は `ui_*` 経由でだけ触る）。"""

    def __init__(self, win, invoker, art: Path, args) -> None:
        self.win, self.inv, self.art, self.args = win, invoker, art, args
        self.vl = None
        self.rec = Recorder()
        self.handles: list = []   # (t0, kind, dedup, text, cancelled)
        self.cur = {"kind": None}
        self.turns: list = []
        self.calc: dict = {}
        self.files: list = []

    # ---------------------------------------------------------------- UI スレッドへの橋
    def ui(self, fn) -> "concurrent.futures.Future":
        """fn(fut) を UI スレッドで実行する（fn が結果を入れる。入れずに返せば戻り値が結果）。"""
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def run() -> None:
            try:
                r = fn(fut)
                if not fut.done() and r is not None:
                    fut.set_result(r)
            except Exception as e:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(e)

        self.inv.call.emit(run)
        return fut

    async def ui_await(self, fn, timeout: float = 20.0):
        return await asyncio.wait_for(asyncio.wrap_future(self.ui(fn)), timeout)

    def _grab(self, name: str):
        def fn(_fut):
            path = self.art / name
            self.win.taskwin.grab().save(str(path))
            return str(path)
        return fn

    def _sub_titles(self, _fut):
        return [s.windowTitle() for s in self.win.taskwin._subs.values()] or ["（なし）"]

    def _run_calc_js(self, fut):
        from PySide6.QtWebEngineWidgets import QWebEngineView

        views = []
        for sub in list(self.win.taskwin._subs.values()):
            w = sub.widget()
            cand = ([w] if isinstance(w, QWebEngineView) else []) + (
                w.findChildren(QWebEngineView) if w is not None else [])
            views += [(sub.windowTitle(), v) for v in cand]
        html = [(t, v) for (t, v) in views if t.lower().endswith((".html", ".htm"))] or views
        if not html:
            return {"window": None, "note": "動く HTML の窓が無い（ソース表示か、表示していない）"}
        title, view = html[-1]

        def cb(res) -> None:
            if not fut.done():
                fut.set_result({"window": title, "raw": res})
        page = view.page()
        try:
            page.runJavaScript(CALC_JS, 0, cb)
        except TypeError:
            page.runJavaScript(CALC_JS, cb)
        return None

    # ---------------------------------------------------------------- 準備（ループ上）
    def prepare(self, vl) -> None:
        self.vl = vl
        orch = vl.orchestrator
        reg = vl.registry
        # 応答モデル・reasoning_effort・書き出しの表は**本番の設定のまま**（2026-09-29 に採用＝.env と fillers.yaml）。
        # ここでは時間を測るために包むだけ
        _stream = orch._stream_fn

        async def stream_fn(messages, *, tools=None, tool_sink=None):
            t0, first = time.monotonic(), True
            gen = _stream(messages, tools=tools, tool_sink=tool_sink) if tools is not None else _stream(messages)
            async for d in gen:
                if first:
                    first = False
                    ev("🧠", f"応答LLM 最初の文字 +{time.monotonic() - t0:.2f}秒")
                yield d
            calls = [f"{c['function']['name']}({c['function']['arguments'][:100]})" for c in (tool_sink or [])]
            ev("🧠", f"応答LLM 終わり +{time.monotonic() - t0:.2f}秒" + (f" ツール {calls}" if calls else ""))

        orch._stream_fn = stream_fn

        _handle = orch.handle

        async def handle(stim):
            t0 = time.monotonic()
            dk = getattr(stim, "dedup_key", None) or ""
            self.cur = {"kind": stim.kind.name, "t0": t0}
            p = getattr(stim, "payload", None)
            ev("▶", f"{stim.kind.name} {dk} {str(getattr(p, 'content', None) or p)[:80]}")
            try:
                await _handle(stim)
            except asyncio.CancelledError:
                self.handles.append((t0, stim.kind.name, dk, orch.last_response or "", True))
                ev("✂", f"{stim.kind.name} 中断 {dk}")
                raise
            self.handles.append((t0, stim.kind.name, dk, orch.last_response or "", False))
            ev("✔", f"{stim.kind.name} {dk} 応答={orch.last_response!r}")

        orch.handle = handle

        caps = vl.capabilities
        _ex = caps.execute_async

        async def ex(name, args=None):
            t0 = time.monotonic()
            a = json.dumps(args or {}, ensure_ascii=False)
            ev("⚙", f"{name}({a[:160]})")
            r = await _ex(name, args)
            ev("⚙✔", f"{name} {time.monotonic() - t0:.2f}秒 → {str(r)[:160]}")
            if name in ("write_file", "show_on_screen"):
                self.files.append((name, args or {}, str(r)))
            return r

        caps.execute_async = ex

        prev = vl.audio.on_sentence_start

        def on_start(text, meta):
            ev("🔊", f"[{self.cur.get('kind')}] {text}")
            if prev is not None:
                prev(text, meta)

        vl.audio.on_sentence_start = on_start
        _play = vl.audio._play_fn

        async def play(audio):
            self.rec.add_wav(audio, "eve")
            ev("🔈", f"応答の音 鳴らし始め [{self.cur.get('kind')}]")
            return await _play(audio)

        vl.audio._play_fn = play
        # 合成音声の聞き違いで依頼がすり替わらないように（必須語が無ければ台本の文）
        self.cur_script, self.cur_must = "", ()
        _tr = vl.input._stt.transcribe

        async def transcribe(pcm):
            text = await _tr(pcm)
            if text and self.cur_must and not all(m in text for m in self.cur_must):
                ev("👂", f"聞き違い「{text}」→ 台本「{self.cur_script}」を使う（必須語 {list(self.cur_must)}）")
                return self.cur_script
            return text

        vl.input._stt.transcribe = transcribe
        if vl.filler is not None:
            _fplay = vl.filler._play

            async def fplay(audio):
                self.rec.add_wav(audio, "filler")
                ev("🫧", "フィラーの音 鳴らし始め")
                return await _fplay(audio)

            vl.filler._play = fplay
        # マイクを止めて差し替える（本物のマイクは周りの音とスピーカーの声を拾う）
        vl.input.stop()
        self.mic = FakeMic()
        vl.input._ai = self.mic
        vl.input._task = asyncio.create_task(vl.input._consume())
        from eve.config import Config

        ev("🛠", f"準備: 応答={reg.resolve('response')}・reasoning_effort={Config.RESPONSE_REASONING_EFFORT or '（既定）'}"
                 f"・書き出し=fillers.yaml の opener_prompt・マイク→合成音声")

    # ---------------------------------------------------------------- 1発話
    async def speak(self, text: str, vad) -> float:
        import numpy as np
        import torch

        from eve.audio_input import _SegmentTracker
        from eve.filler import audio as fa
        from eve.stt.resample import StreamResampler

        from eve.config import Config

        vl = self.vl
        rate, chunk = 16000, 512
        base = text.rstrip("。？！?!")
        for said in dict.fromkeys([text, base + "。", base + "！"]):  # 語は変えず句読点だけ変える
            wav = await vl.tts.generate(said)
            x, sr0 = fa.read_wav(wav)
            pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
            sp = np.frombuffer(StreamResampler(sr0, rate).process(pcm), dtype="<i2").copy()
            # 流す前に VAD に掛けて、発話として拾われる音か確かめる（拾われなければ言い方を変えて作り直す）
            if hasattr(vad, "reset_states"):
                vad.reset_states()
            frames = sum(1 for i in range(0, len(sp) - chunk + 1, chunk)
                         if vad(torch.from_numpy(sp[i:i + chunk].astype(np.float32) / 32768.0), rate).item()
                         > Config.VAD_THRESHOLD)
            if frames >= MIN_SPEECH_FRAMES:
                break
            ev("🔁", f"合成した「{said}」は VAD が {frames}枠しか反応しない → 言い方を変えて作り直す")
        if said != text:
            ev("🔁", f"合成は「{said}」で流す（VAD {frames}枠）")
        d = np.concatenate([np.zeros(rate // 2, np.int16), sp, np.zeros(rate, np.int16)])

        def on_end(accepted: bool, speech_sec: float) -> None:
            ev("〈", f"話し終わり（VAD 確定）・声 {speech_sec:.2f}秒{'' if accepted else '・不採用'}")
            vl.input._segment_end(accepted, speech_sec)

        tr = _SegmentTracker(rate=rate, chunk=chunk, put=self.mic.q.put_nowait, on_speech_start=vl._barge_in,
                             on_segment_start=vl.input._segment_start, on_segment_audio=vl.input._segment_audio,
                             on_segment_end=on_end)
        self.mic.tracker = tr
        if hasattr(vad, "reset_states"):
            vad.reset_states()
        t_start = time.monotonic()
        self.rec.add_array(np.asarray(x, np.float32).reshape(-1), sr0, "user", t_start + 0.5)
        ev("🗣", f"ユーザ: {text}")
        t0 = time.perf_counter()
        for i in range(0, len(d) - chunk + 1, chunk):
            f32 = d[i:i + chunk].astype(np.float32) / 32768.0
            tr.step(d[i:i + chunk].tobytes(), vad(torch.from_numpy(f32), rate).item(),
                    float(np.abs(f32).max()), time.perf_counter())
            wait = t0 + (i + chunk) / rate - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
        return t_start

    def _sidecars_idle(self) -> bool:
        vl = self.vl
        d, r, x = vl.dispatcher, vl.cancel_resolver, vl.task_executor
        return ((d is None or (d._inbox.empty() and d._inbox._unfinished_tasks == 0))
                and (r is None or (r._inbox.empty() and r._inbox._unfinished_tasks == 0))
                and (x is None or x.is_idle()))

    def _quiet(self) -> bool:
        vl = self.vl
        return not (vl.runner.is_busy() or not vl.audio.is_quiet()
                    or (vl.filler is not None and vl.filler.active)
                    or vl.speech_state.stt_pending or vl.queue.qsize() or not self._sidecars_idle())

    def _delivered(self, task_id: str) -> bool:
        return any(dk == f"task:{task_id}" and not c for (_t, _k, dk, _x, c) in self.handles)

    async def settle(self, before: set, timeout: float) -> tuple:
        """応答・タスク・報告がすべて終わって静かになるまで待つ（固定 sleep で判定しない）。"""
        from eve.task.schema import CANCELLED, TERMINAL

        deadline = time.monotonic() + timeout
        seen: dict = {}
        stable = 0
        await asyncio.sleep(1.0)
        new: list = []
        while time.monotonic() < deadline:
            new = [t for t in (self.vl.task_store.list_all() if self.vl.task_store else [])
                   if t.goal and t.task_id not in before]
            for t in new:
                if seen.get(t.task_id) != t.status:
                    seen[t.task_id] = t.status
                    ev("📋", f"タスク {t.task_id} → {t.status}「{t.goal[:60]}」"
                       + (f" 結果: {str(t.result)[:200]}" if t.result and t.status in TERMINAL else ""))
            waiting = [t for t in new if t.status not in TERMINAL
                       or (t.status != CANCELLED and not self._delivered(t.task_id))]
            if self._quiet() and not waiting:
                stable += 1
                if stable >= 8:
                    return new, True
            else:
                stable = 0
            await asyncio.sleep(0.1)
        ev("⚠", f"{timeout:.0f}秒待っても終わらなかった（次へ進む）")
        return new, False

    # ---------------------------------------------------------------- 通し
    async def scenario(self) -> dict:
        from eve.audio_input import _load_silero_vad

        vad = _load_silero_vad()
        await asyncio.sleep(1.0)
        for n, (key, text, crit, must) in enumerate(SCENARIOS[self.args.scenario], 1):
            before = {t.task_id for t in (self.vl.task_store.list_all() if self.vl.task_store else [])}
            self.cur_script, self.cur_must = text, must
            t_start = await self.speak(text, vad)
            tasks, ok = await self.settle(before, self.args.turn_timeout)
            t_end = time.monotonic()
            titles = await self.ui_await(self._sub_titles)
            shot = await self.ui_await(self._grab(f"taskwin_{n}_{key}.png"))
            turn = {"n": n, "key": key, "text": text, "criterion": crit, "t_start": t_start, "t_end": t_end,
                    "settled": ok, "windows": titles, "shot": shot,
                    "tasks": [{"id": t.task_id, "goal": t.goal, "status": t.status, "result": t.result,
                               "created_at": t.created_at, "updated_at": t.updated_at} for t in tasks]}
            if key == "電卓":
                try:
                    res = await self.ui_await(self._run_calc_js, timeout=20)
                    raw = res.get("raw") if isinstance(res, dict) else None
                    turn["calc"] = {"window": res.get("window") if isinstance(res, dict) else None,
                                    "note": res.get("note") if isinstance(res, dict) else None,
                                    "result": json.loads(raw) if isinstance(raw, str) and raw.startswith("{") else raw}
                    turn["calc_shot"] = await self.ui_await(self._grab(f"taskwin_{n}_{key}_after_test.png"))
                except Exception as e:  # noqa: BLE001
                    turn["calc"] = {"error": f"{type(e).__name__}: {e}"}
                ev("🧮", f"電卓の確認: {json.dumps(turn['calc'], ensure_ascii=False)[:400]}")
            self.turns.append(turn)
            await asyncio.sleep(2.0)
        usage = self.vl.usage.format_summary() if getattr(self.vl, "usage", None) else ""
        root = str(self.vl.workspace.paths.root) if self.vl.workspace is not None else ""
        ft = self.vl.filler._t if self.vl.filler is not None else None
        sp = sum(ft.weights.get(c, 0.0) for c in ft.special) if ft is not None else None
        return {"usage": usage, "ws_root": root, "join_gap": getattr(ft, "join_gap_sec", None),
                "special_pct": None if sp is None else round(sp, 2)}


# ==================================================================== 報告
def _rel(t: float, base: float) -> str:
    return f"{t - base:6.2f}"


def write_report(art: Path, h: Harness, extra: dict, task_model: str) -> Path:
    import soundfile as sf

    what = "タスク4種" if h.args.scenario == "tasks" else "通常会話"
    lines = [f"# ルナ × フィラー初期案 × {what}（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"- 応答: {os.getenv('RESPONSE_MODEL', '')}（reasoning_effort={os.getenv('RESPONSE_REASONING_EFFORT', '') or '既定'}）"
             f" / タスク担当: {task_model} / 書き出し: fillers.yaml の opener_prompt",
             f"- フィラーのバンク: {os.getenv('FILLER_VOICE', '')}"
             f"（つなぎの間 {extra.get('join_gap')}秒・特殊フィラーの組 {extra.get('special_pct')}%）",
             "- 本番 UI（タスクウィンドウ）・マイクの代わりに合成音声（Irodori）を実時間で VAD → ストリーミング文字起こしへ",
             "- 画面認識・自発発話は off / 記憶は空から / 時刻は「ユーザが話し始めてからの秒」と「話し終わり（VAD 確定）からの秒」",
             ""]
    summary = []
    allev = sorted([(t, k, x) for (t, k, x) in EVENTS] + [(t, "log", f"{n}: {m}") for (t, n, m) in LOGS],
                   key=lambda e: e[0])
    for turn in h.turns:
        ts, te = turn["t_start"], turn["t_end"]
        vad_end = next((t for (t, k, x) in EVENTS if k == "〈" and ts <= t <= te), None)
        first_filler = next((t for (t, k, x) in EVENTS if k == "🫧" and vad_end and vad_end <= t <= te), None)
        first_voice = next((t for (t, k, x) in EVENTS if k == "🔈" and "USER_UTTERANCE" in x
                            and vad_end and vad_end <= t <= te), None)
        report_voice = next((t for (t, k, x) in EVENTS if k == "🔈" and "CALLFUNCTION_RESULT" in x
                             and ts <= t <= te), None)
        llm_first = next((x for (t, k, x) in EVENTS if k == "🧠" and "最初" in x and vad_end and vad_end <= t <= te), "")
        base = vad_end or ts
        lines += [f"## {turn['n']}. {turn['text']}", "", f"合格条件: {turn['criterion']}", "",
                  "| 話し始めから | 話し終わりから | 出来事 |", "|---:|---:|---|"]
        for (t, k, x) in allev:
            if not (ts - 0.05 <= t <= te):
                continue
            if k == "log":
                if x.startswith("luna_e2e"):
                    continue
                txt = x.split(": ", 1)[1] if ": " in x else x
            else:
                txt = f"{k} {x}"
            txt = txt.replace("|", "／").replace("\n", " ")
            lines.append(f"| {t - ts:.2f} | {t - base:+.2f} | {txt[:300]} |")
        lines.append("")
        for tk in turn["tasks"]:
            lines += [f"- タスク {tk['id']}「{tk['goal']}」→ **{tk['status']}**", f"  - 結果: {tk['result']}"]
        lines.append(f"- タスクウィンドウの窓: {turn['windows']}（画面の写し {Path(turn['shot']).name}）")
        if "calc" in turn:
            lines.append(f"- 電卓の確認（表示された HTML の中でボタンを押した）: "
                         f"```{json.dumps(turn['calc'], ensure_ascii=False)}```")
        if not turn["settled"]:
            lines.append("- ⚠ 時間内に終わらなかった")
        lines.append("")
        # フィラーの1行（「🫧 フィラー #N: … | あー[0.00-1.10] … | 終わり: …」）から鳴った語だけ
        chain = next((m for (t, n, m) in LOGS if "🫧 フィラー" in m and ts <= t <= te), "")
        pieces = chain.split(" | ")[1] if chain.count(" | ") >= 2 else ("（なし）" if not chain else chain[-60:])
        summary.append((turn, vad_end, first_filler, first_voice, report_voice, llm_first, pieces))
    lines += ["## まとめ（秒）", "",
              "| 発話 | 鳴ったフィラー | 話し終わり→フィラーの音 | 話し終わり→応答の声 | 応答LLM 最初の文字 | 話し終わり→報告の声 |",
              "|---|---|---:|---:|---|---:|"]
    for (turn, ve, ff, fv, rv, lf, pc) in summary:
        f = lambda t: f"{t - ve:.2f}" if (t and ve) else "—"  # noqa: E731
        lines.append(f"| {turn['text']} | {pc} | {f(ff)} | {f(fv)} | "
                     f"{lf.replace('応答LLM 最初の文字 ', '') or '—'} | {f(rv)} |")
    lines += ["", "## 使用量", "", "```", extra.get("usage", ""), "```", ""]
    # 作られたファイルの写し
    root = Path(extra.get("ws_root") or ".")
    copied = []
    for (name, args, _r) in h.files:
        p = args.get("path") or ""
        if not p:
            continue
        src = (root / p)
        if src.is_file():
            dst = art / ("ws_" + Path(p).name)
            try:
                shutil.copyfile(src, dst)
                copied.append(dst.name)
            except OSError:
                pass
    lines += ["## 作られたファイル（写し）", "", ", ".join(sorted(set(copied))) or "（なし）", ""]
    # 録音
    if h.turns:
        t_from = h.turns[0]["t_start"]
        t_to = h.turns[-1]["t_end"]
        sf.write(str(art / "conversation.wav"), h.rec.render(t_from, t_to), 24000, subtype="PCM_16")
        for turn in h.turns:
            sf.write(str(art / f"turn{turn['n']}.wav"), h.rec.render(turn["t_start"], turn["t_end"]),
                     24000, subtype="PCM_16")
        lines += ["## 録音", "", "conversation.wav（通し）・turn1〜4.wav（ユーザの声＋フィラー＋応答）", ""]
    path = art / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (art / "events.json").write_text(json.dumps(
        {"events": [(round(t - T0, 3), k, x) for (t, k, x) in EVENTS],
         "logs": [(round(t - T0, 3), n, m) for (t, n, m) in LOGS],
         "turns": [{k: v for k, v in t.items()} for t in h.turns]}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return path


# ==================================================================== 起動
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=1500.0, help="全体の上限[秒]（UI を閉じる）")
    ap.add_argument("--turn-timeout", type=float, default=300.0, help="1発話（タスクと報告まで）の上限[秒]")
    ap.add_argument("--scenario", choices=tuple(SCENARIOS), default="tasks",
                    help="tasks=タスク4件（ユーザ指定の検証）/ chat=通常会話")
    ap.add_argument("--filler-voice", default="", help="フィラーのバンク（例 eve_calm_long）。無ければ .env のまま")
    args = ap.parse_args()
    art = REPO / "e2e_logs" / f"luna_{args.scenario}_{time.strftime('%Y%m%d_%H%M%S')}"
    art.mkdir(parents=True, exist_ok=True)
    setup_env(art, args.filler_voice)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(TOOLS))

    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtWidgets import QApplication

    from eve.config import Config
    from eve.logsetup import configure
    from eve.ui.live2d_view import assets_ready

    configure()
    for noisy in ("httpx", "aiohttp", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    fh = logging.FileHandler(art / "terminal.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(Capture())
    ok, why = assets_ready()
    if not ok:
        print(why)
        return 2
    import live2d.v3 as live2d

    if Config.TASKWIN_HTML_ENABLED:
        from eve.ui import web_view

        web_view.preinit()
    from eve.ui.app import MainWindow

    class Invoker(QObject):
        call = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.call.connect(self._run, Qt.QueuedConnection)

        def _run(self, fn) -> None:
            try:
                fn()
            except Exception:
                log.exception("UI 側の処理で例外")

    live2d.init()
    app = QApplication.instance() or QApplication([])
    inv = Invoker()
    win = MainWindow()
    screen = app.primaryScreen().availableGeometry()
    win.resize(min(1280, int(screen.width() * 0.92)), min(860, int(screen.height() * 0.92)))
    win.show()
    app.processEvents()
    QTimer.singleShot(50, win.apply_capture_guard)
    QTimer.singleShot(80, win.start_loop)
    h = Harness(win, inv, art, args)
    state = {"done": False, "report": None, "closing": False}

    def controller() -> None:
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 600:
                th = win.thread_
                vl = getattr(th, "_vl", None)
                if vl is not None and th._loop is not None and vl.input._ai is not None:
                    break
                time.sleep(1.0)
            else:
                log.error("起動しなかった（600秒）")
                return
            loop = win.thread_._loop
            log.info("起動完了（%.0f秒）", time.monotonic() - t0)
            prep = concurrent.futures.Future()

            def _prep() -> None:
                try:
                    h.prepare(win.thread_._vl)
                    prep.set_result(True)
                except Exception as e:  # noqa: BLE001
                    prep.set_exception(e)

            loop.call_soon_threadsafe(_prep)
            prep.result(timeout=30)
            extra = asyncio.run_coroutine_threadsafe(h.scenario(), loop).result(timeout=args.limit)
            state["report"] = write_report(art, h, extra, os.getenv("TASK_MODEL", "openai/gpt-5.5（既定）"))
        except Exception:
            log.exception("通しが途中で止まった")
            try:
                state["report"] = write_report(art, h, {}, os.getenv("TASK_MODEL", "openai/gpt-5.5（既定）"))
            except Exception:
                log.exception("報告も書けなかった")
        finally:
            state["done"] = True

    def poll() -> None:
        if state["done"] and not state["closing"]:
            state["closing"] = True
            win.close()   # 本番と同じ終わり方（ループを drain → VoiceLoop.stop()）
            QTimer.singleShot(45000, app.quit)
        if state["closing"] and not win.thread_.is_alive():
            QTimer.singleShot(500, app.quit)
            return
        QTimer.singleShot(300, poll)

    def watchdog() -> None:
        log.error("全体の上限 %.0f秒 に達したので閉じる", args.limit)
        state["done"] = True

    threading.Thread(target=controller, daemon=True).start()
    QTimer.singleShot(1000, poll)
    QTimer.singleShot(int(args.limit * 1000), watchdog)
    app.exec()
    live2d.dispose()
    print(f"\n報告: {state['report'] or '（書けなかった）'}\n出力: {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
