#!/usr/bin/env python3
"""本番のイブ（VoiceLoop・.env の設定のまま）で Alchemy を通しで遊ばせ、起きたことを全部記録する（実 API・課金あり）。

ユーザ 2026-10-11「とりあえず動かしてどんな挙動するか報告して」「見るべきところは実例を計測し報告」。

本番から変えるのは入出力だけ（イブの中身・プロンプト・機能・モデルは .env のまま）:
- 声の入力 → 文字（`VoiceLoop.submit_user_text`＝キーボード入力と同じ経路。STT の聞き間違いは入らない）
- 声の合成と再生 → 文字数から長さを見積もった無音を「再生」する（話している間の長さは保つ・音は出さない）
- 画面認識が見るデスクトップ → 用意した画像（ユーザの本物の画面は送らない）。部屋・対戦中はイブ自身のゲーム画面（本物）
- 記憶（会話・RAG・タスク・自己形成・感情）→ 本番の複製（書き込みは複製に・本番を汚さない）
- フィラー・笑い・場面の判定（声の入力が要る機能）は切る
ゲームのサーバーは手元（FlaskRondo を読むだけで起動しておく）。ユーザ役の相手（たかし）はコードの手で、
人っぽく数秒考えてから打つ。

台本（ユーザ役の発話は、イブが話し終わって数秒おいてから打つ）:
 1. 普段の会話（関係ない話・錬金術やカードゲームなど言葉の近い話。画面もそれらしい画像）
 2. Alchemy の話（今は遊ばない）
 3. （準備の期限切れと同じ状態に戻してから）ユーザの画面に Alchemy のタイトル画面 → 画面認識で準備するか
 4. 「よし、やろう」 → 部屋に入るか → ユーザ役が部屋に入って準備OK → 対戦
 5. 対戦中: 点数・関係ない話・カードの理由・前のラウンドの手・事実と違う前提・次の狙い（2戦目にまたがる）
 6. 終わったら「もう一回やろう」 → 2戦目 → 「やめよう」 → 対戦の後の雑談

  python -m tools.prompt_eval.alchemy_live --mode action     → eval_report/prompts/alchemy_live/action/

`--scenario wait`（ユーザ 2026-10-11 の決め方「5分以上待っても入室しなければ催促して、返事が無い（30秒）・やらない反応なら
やめる意思を示して退出」・対戦中にやめたら席は自動操作で続く）:
 A. ルールを聞く（前置きで止まらず答えるか）
 B. 「じゃあAlchemyやろう」→ ユーザ役は入らず何も言わない → 5分で確かめ、30秒で抜けるか
 C. 「ごめん、やっぱやろう」→ 確かめられたら「今から入る」→ 入って対戦 → 1ラウンド目の後に「やっぱゲームやめよう」
`--scenario rules --reps 10`: 毎回ルールが入っていない状態から「Alchemyのルールってどんなのだっけ」と聞く（答えの速さ）
  python -m tools.prompt_eval.alchemy_live --mode action --scenario wait   → .../alchemy_live/wait_action/
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "eval_report" / "prompts" / "alchemy_live"
IMG = REPO / "eval_report" / "prompts" / "alchemy_vlm"
OPP_NAME = "たかし"

T0 = time.monotonic()
TL: list = []
_tl_file = None


def ev(kind: str, text: str, **kw) -> None:
    rec = {"t": round(time.monotonic() - T0, 1), "kind": kind, "text": text, **kw}
    TL.append(rec)
    print(f"{rec['t']:7.1f} {kind:10s} {text}", flush=True)
    if _tl_file is not None:
        _tl_file.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        _tl_file.flush()


# ---------------------------------------------------------------------------- 環境（eve を import する前）
def setup(mode: str, out: Path) -> Path:
    """本番の記憶を複製して、そこを使わせる。.env の値は上書きしない物以外そのまま。"""
    tmp = Path(tempfile.mkdtemp(prefix="eve_alchemy_live_"))
    files = {"HISTORY_FILE": "conversation_history.jsonl", "RAG_FILE": "rag_memory.jsonl", "TASK_FILE": "tasks.jsonl",
             "SELF_MODEL_FILE": "self_model.jsonl", "EMOTION_STATE_FILE": "emotion_state.json"}
    env = {}
    for key, default in files.items():
        name = os.getenv(key) or _dotenv_value(key) or default
        src = REPO / name
        dst = tmp / Path(name).name
        if src.exists():
            shutil.copyfile(src, dst)
        env[key] = str(dst)
    env.update({
        "USAGE_LOG_ENABLED": "1", "USAGE_LOG_FILE": str(out / "usage.jsonl"),
        "FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0",
        "ALCHEMY_ENABLED": "1", "ALCHEMY_URL": "http://127.0.0.1:8765/", "ALCHEMY_PLAY_MODE": mode,
    })
    os.environ.update(env)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    return tmp


def _dotenv_value(key: str):
    p = REPO / ".env"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------- 声の代わり
def _silent_wav(sec: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(sec * rate))
    return buf.getvalue()


def _speech_sec(text: str) -> float:
    return max(0.6, len(text or "") / 7.5)  # 日本語の話す速さ（1秒に7〜8字）


class FakeTTS:
    concurrency = 3

    async def generate_with_meta(self, text: str):
        await asyncio.sleep(0.3)
        return _silent_wav(_speech_sec(text)), None

    async def generate(self, text: str):
        return (await self.generate_with_meta(text))[0]

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeSTT:
    streaming = False

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def transcribe(self, *a, **k) -> str:
        return ""


# ---------------------------------------------------------------------------- 画面認識に見せるデスクトップ
class Desktop:
    """画面認識が見るデスクトップ（ユーザの本物の画面の代わり・画像を差し替える）。"""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self.name = "neutral"
        self._cache: dict = {}
        self._img = self._load("neutral")

    def _load(self, name: str):
        if name in self._cache:
            return self._cache[name]
        if name == "neutral":
            img = _neutral_desktop()
        else:
            img = self._cv2.imread(str(IMG / f"{name}.png"))
        self._cache[name] = img
        return img

    def show(self, name: str) -> None:
        self.name = name
        self._img = self._load(name)
        ev("画面", f"ユーザの画面を「{name}」にした")

    def grab(self):
        return self._img.copy()


def _neutral_desktop():
    """普段のデスクトップ（メモ帳に買い物メモ）。ゲームと関係ない画面。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    w, h = 1920, 1080
    im = Image.new("RGB", (w, h), (40, 90, 140))
    d = ImageDraw.Draw(im)
    d.rectangle([0, h - 48, w, h], fill=(30, 30, 30))
    d.rectangle([300, 150, 1400, 850], fill=(255, 255, 255), outline=(120, 120, 120))
    d.rectangle([300, 150, 1400, 190], fill=(230, 230, 230))
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", 30)
        small = ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", 20)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((315, 158), "買い物メモ.txt - メモ帳", fill=(0, 0, 0), font=small)
    lines = ["買い物メモ", "", "・牛乳", "・卵", "・食パン", "・洗剤（詰め替え）", "", "明日: 10時 歯医者"]
    for i, ln in enumerate(lines):
        d.text((330, 210 + i * 46), ln, fill=(20, 20, 20), font=font)
    return np.array(im)[:, :, ::-1].copy()


# ---------------------------------------------------------------------------- ユーザ役（ゲームの相手）
class Opponent:
    """ユーザ役のプレイヤー（コードの手・人っぽく数秒考える）。部屋に入る・もう一回の準備OKは台本が指示する。"""

    def __init__(self, url: str, keyword: str, seed: int) -> None:
        self.ws_url = url.replace("http://", "ws://").rstrip("/") + "/ws"
        self.keyword = keyword
        self.rng = random.Random(seed)
        self.room_playing = False
        self._ws = None
        self._session = None
        self._task = None
        self.rules = self.recipes = None
        self.last_state = None  # ユーザ役の画面に届いた最後の state（イブが抜けた後の席の様子を見る）
        self.events: list = []

    async def leave(self) -> None:
        await self._ws.send_json({"type": "room.leave"})
        await self._ws.send_json({"type": "keyword.leave"})
        ev("ユーザ操作", "ユーザ役も部屋から出た")

    async def join(self) -> None:
        import aiohttp

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self.ws_url, heartbeat=20)
        await self._ws.send_json({"type": "hello", "name": OPP_NAME})
        await self._ws.send_json({"type": "keyword.join", "keyword": self.keyword})
        self._task = asyncio.create_task(self._loop())
        await asyncio.sleep(2.5)
        await self._ws.send_json({"type": "keyword.ready", "ready": True})
        ev("ユーザ操作", f"ユーザ役が合言葉「{self.keyword}」の部屋に入って準備OK")

    async def ready_again(self) -> None:
        t0 = time.monotonic()
        while self.room_playing and time.monotonic() - t0 < 40:
            await asyncio.sleep(0.3)
        await asyncio.sleep(1.5)
        await self._ws.send_json({"type": "keyword.ready", "ready": True})
        ev("ユーザ操作", "ユーザ役がもう一回の準備OK")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
        try:
            if self._ws is not None:
                await self._ws.close()
            if self._session is not None:
                await self._session.close()
        except Exception:
            pass

    async def _loop(self) -> None:
        import aiohttp

        from eve.alchemy import autoplay, sim

        acted: set = set()
        async for m in self._ws:
            if m.type != aiohttp.WSMsgType.TEXT:
                continue
            d = json.loads(m.data)
            t = d.get("type")
            if t == "joined":
                self.rules, self.recipes = sim.load_catalog(d["catalog"])
            elif t == "keyword":
                self.room_playing = bool(d.get("playing"))
            elif t == "match":
                acted.clear()  # 2戦目はラウンドの番号が 1 から数え直し
            elif t == "events":
                self.events.extend(d.get("events") or [])
            if t != "state" or self.rules is None:
                continue
            v = d["state"]
            self.last_state = v
            me = v.get("me") or {}
            key = (v.get("round"), v.get("phase"))
            if key in acted:
                continue
            try:
                if v["phase"] == "select" and me.get("hand") and not me.get("selection"):
                    acted.add(key)
                    await asyncio.sleep(self.rng.uniform(5, 12))
                    picks = autoplay.choose_select(v, self.rules, self.recipes)
                    await self._ws.send_json({"type": "play.cards",
                                              "cards": [{"card": c, "rotated": r} for c, r in picks]})
                elif v["phase"] == "skill" and any((me.get("skill_pending") or {}).values()):
                    acted.add(key)
                    await asyncio.sleep(self.rng.uniform(2, 5))
                    for u in autoplay.choose_skills(v, self.rules, self.recipes):
                        await self._ws.send_json({"type": "skill.use", **u})
                    await self._ws.send_json({"type": "skill.skip"})
                elif v["phase"] == "brew" and me.get("brew") and not me["brew"].get("done"):
                    acted.add(key)
                    await asyncio.sleep(self.rng.uniform(6, 14))
                    for st in autoplay.choose_brew(v, self.rules, self.recipes):
                        await self._ws.send_json({"type": "brew.auto", "index": st.index})
                        await asyncio.sleep(0.3)
                    await self._ws.send_json({"type": "brew.done"})
            except Exception as e:  # noqa: BLE001
                ev("相手エラー", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------- 本体
class Live:
    def __init__(self, args, out: Path) -> None:
        self.args = args
        self.out = out
        self.vl = None
        self.mgr = None
        self.desk = None
        self.record: dict = {"events": [], "states": [], "moves": [], "voice_blocks": []}
        self.turns: list = []
        self.cur_turn: dict = {}
        self.decisions: list = []
        self.triggers: list = []
        self.started = asyncio.Event()
        self.resolves: list = []  # (mono, ev)
        self.match_count = 0
        self.ended = 0

    # --- 組み立て ---
    def build(self) -> None:
        import eve.voice_loop as vlmod
        from eve.config import Config
        from eve.vlm.capture import ScreenCapture

        vlmod._make_tts = lambda: FakeTTS()
        vlmod.make_stt = lambda *a, **k: FakeSTT()
        vlmod.sweep_orphan_tts_servers = lambda: 0
        vl = vlmod.VoiceLoop()
        self.vl = vl
        if vl.alchemy is None:
            raise SystemExit("Alchemy が有効になっていない（ALCHEMY_ENABLED / CALLFUNCTION_ENABLED）")
        self.mgr = vl.alchemy
        player = vl.player

        async def play_fn(audio, should_stop=None):
            if not audio:
                return
            with wave.open(io.BytesIO(audio), "rb") as w:
                dur = w.getnframes() / float(w.getframerate())
            await asyncio.sleep(dur)
            from eve.clock import now_mono
            player.last_end = now_mono()

        vl.audio._play_fn = play_fn
        player.prepare = lambda rate: None
        player.close = lambda: None
        outer = self

        class StubInput:
            async def start(self):
                outer.started.set()

            def stop(self):
                pass

            def user_in_speech(self):
                return False

        vl.input = StubInput()
        self.desk = Desktop()
        cap = ScreenCapture(monitor=Config.VLM_MONITOR, downscale_max=Config.VLM_DOWNSCALE_MAX,
                            jpeg_quality=Config.VLM_JPEG_QUALITY, blank_std_threshold=Config.VLM_BLANK_STD_THRESHOLD,
                            grab_fn=self.desk.grab)
        vl.vlm_capture = cap
        vl._desktop_capture = cap
        self._instrument()
        ev("設定", f"遊ばせ方 {self.mgr.brain.mode}・合言葉「{self.mgr.s.keyword}」・応答 {vl.registry.resolve('response')}"
                 f"・判定 {vl.registry.resolve('speech_decide')}・判断役 {vl.registry.resolve('game')}"
                 f"・画面 {'on' if Config.VLM_ENABLED else 'off'}・自律発話 {'on' if Config.AUTOSPEECH_ENABLED else 'off'}"
                 f"・機能 {sorted(vl.capabilities.names())}")

    def _instrument(self) -> None:
        vl, mgr = self.vl, self.mgr
        orch = vl.orchestrator
        _handle = orch.handle

        async def handle(stim):
            payload = getattr(stim, "payload", None)
            if hasattr(payload, "content") and hasattr(payload, "function_name"):
                what = f"{payload.function_name}: {payload.content}"
            elif hasattr(payload, "content"):
                what = f"{payload.content}（{getattr(payload, 'reason', '')}）"
            else:
                what = str(payload)
            self.cur_turn = {"kind": stim.kind.name, "what": what, "t0": time.monotonic(), "said": []}
            ev("▶" + stim.kind.name[:8], what[:160])
            if mgr.state != "idle":
                # 応答AI に渡った `# Alchemy`（話した中身を、渡した事実とサーバーの記録の両方と突き合わせるため）
                self.record["voice_blocks"].append({"t": round(time.monotonic() - T0, 1), "kind": stim.kind.name,
                                                    "said": what, "block": mgr.voice_block()})
            cancelled = False
            try:
                await _handle(stim)
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                self.turns.append({"t": round(self.cur_turn["t0"] - T0, 1), "kind": self.cur_turn["kind"],
                                   "what": what, "response": orch.last_response or "",
                                   "said": list(self.cur_turn["said"]), "cancelled": cancelled,
                                   "sec": round(time.monotonic() - self.cur_turn["t0"], 1)})
                ev("◀返事", (orch.last_response or "（本文なし）")[:200] + ("（中断）" if cancelled else ""))

        orch.handle = handle
        _enq = vl.audio.enqueue

        def enqueue(gen, seq, audio, text="", on_played=None):
            if text:
                self.cur_turn.setdefault("said", []).append(text)
                ev("🗣イブ", text)
            _enq(gen, seq, audio, text=text, on_played=on_played)

        vl.audio.enqueue = enqueue
        caps = vl.capabilities
        _ex = caps.execute_async

        async def ex(name, args=None):
            ev("⚙機能", f"{name}({json.dumps(args or {}, ensure_ascii=False)[:160]})")
            r = await _ex(name, args)
            ev("⚙結果", f"{name} → {str(r)[:160]}")
            return r

        caps.execute_async = ex
        dec = vl.speech_decider
        _trig = dec.trigger

        def trigger(source="unknown"):
            self.triggers.append({"t": round(time.monotonic() - T0, 1), "source": source})
            ev("🔔合図", source)
            _trig(source)

        dec.trigger = trigger
        st = vl.speech_state
        _rec = st.record_decision

        def record_decision(*, speak, reason, content, count=True):
            src = self.triggers[-1]["source"] if self.triggers else "?"
            self.decisions.append({"t": round(time.monotonic() - T0, 1), "source": src, "speak": speak,
                                   "reason": reason, "content": content, "state": mgr.state,
                                   "game_block": mgr.decider_block()})
            ev("⚖判定", f"[{src}] {'話す' if speak else '黙る'}: {reason}" + (f" ｜{content}" if speak else ""))
            return _rec(speak=speak, reason=reason, content=content, count=count)

        st.record_decision = record_decision
        _react = mgr._fire_react

        def fire_react(text):
            ev("🎲大きな出来事", text)
            _react(text)

        mgr._fire_react = fire_react
        _seen = mgr.note_seen_on_screen

        def seen():
            ev("👁Alchemy検出", f"画面認識が Alchemy を見つけた（状態 {mgr.state}）")
            _seen()

        mgr.note_seen_on_screen = seen
        _rep = mgr._report_result

        def report(name, content, ok):
            ev("📣報告", f"{name}: {content}")
            _rep(name, content, ok)

        mgr._report_result = report
        _om = mgr.on_message

        def on_message(m):
            t = m.get("type")
            now = round(time.monotonic() - T0, 1)
            if t == "events":
                for e in m.get("events") or []:
                    self.record["events"].append({"t": now, **e})
                    if e.get("type") == "resolve":
                        self.resolves.append((time.monotonic(), e))
                    elif e.get("type") == "end":
                        self.ended += 1
            elif t == "state":
                s = m.get("state") or {}
                snap = {"t": now, "round": s.get("round"), "phase": s.get("phase"),
                        "points": {p.get("name"): p.get("points") for p in s.get("players") or []},
                        "boxes": {p.get("name"): p.get("box") for p in s.get("players") or []}}
                last = self.record["states"][-1] if self.record["states"] else None
                if not last or (last["round"], last["phase"], last["points"]) != (snap["round"], snap["phase"], snap["points"]):
                    self.record["states"].append(snap)
            elif t == "match":
                self.match_count += 1
                ev("🎮開始", f"対戦 {self.match_count} 戦目が始まった")
            _om(m)

        mgr.on_message = on_message
        alog = logging.getLogger("eve.alchemy")
        alog.setLevel(logging.INFO)
        alog.propagate = False  # 下の handler で時系列に入れる（二重に出さない）
        outer_rec = self.record

        class H(logging.Handler):
            def emit(self, r):
                msg = r.getMessage()
                ev("🎮ログ", msg)
                if "🎮 R" in msg:
                    outer_rec["moves"].append({"t": round(time.monotonic() - T0, 1), "log": msg})

        alog.addHandler(H())

    async def watch_vision(self) -> None:
        """画面認識の実況（応答・発話判定に渡る文）が変わるたびに記録する。"""
        last = None
        while True:
            await asyncio.sleep(0.5)
            v = self.vl.vision_state.latest_vision
            if v and v != last:
                last = v
                ev("👁実況", f"[{'イブの画面' if self.mgr.owns_screen() else 'デスクトップ'}] {v}")

    # --- 待つ ---
    def quiet(self) -> bool:
        vl = self.vl
        return (not vl.runner.is_busy() and vl.audio.is_quiet() and not vl.audio.is_playing()
                and vl.queue.qsize() == 0 and vl.speech_decider.is_idle())

    async def wait_quiet(self, gap: float = 3.0, max_wait: float = 60.0) -> None:
        t0 = time.monotonic()
        since = None
        while time.monotonic() - t0 < max_wait:
            if self.quiet():
                since = since or time.monotonic()
                if time.monotonic() - since >= gap:
                    return
            else:
                since = None
            await asyncio.sleep(0.2)

    async def wait_until(self, pred, max_wait: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait:
            if pred():
                return True
            await asyncio.sleep(0.3)
        return False

    async def say(self, text: str, gap: float = 3.0) -> None:
        await self.wait_quiet(gap)
        ev("👤ユーザ", text)
        self.vl.submit_user_text(text)
        await asyncio.sleep(1.0)
        await self.wait_quiet(2.0, max_wait=60.0)

    def false_premise(self) -> str:
        """直前の送り合いでイブが送っていない素材を「送ってきた」と言う（事実と違う前提）。"""
        from eve.alchemy.rules import MAT_JA

        seat = self.mgr.seat
        for _mono, e in reversed(self.resolves):
            for t in e.get("transfers") or []:
                if t.get("from") == seat:
                    sent = t.get("sent") or {}
                    cand = [m for m in ("stone", "homunculus", "mineral", "herb", "dragon") if not sent.get(m)]
                    if cand:
                        return f"さっき{MAT_JA[cand[0]]}めっちゃ送ってきたでしょ、ずるくない？"
        return "さっき賢者の石めっちゃ送ってきたでしょ、ずるくない？"

    # --- 台本 ---
    async def script(self) -> None:
        mgr = self.mgr
        await self.wait_quiet(5.0, max_wait=90)
        ev("段階", "1. 普段の会話")
        await self.say("ただいま。今日バイト長引いてめっちゃ疲れた")
        self.desk.show("other_anime")
        await asyncio.sleep(6)
        await self.say("最近鋼の錬金術師を見返してるんだよね")
        self.desk.show("other_poker")
        await asyncio.sleep(6)
        await self.say("カードゲームって得意？")
        self.desk.show("neutral")
        ev("段階", "2. Alchemy の話（今は遊ばない）")
        await self.say("そういえばAlchemyの新しいカード考えてるんだよね")
        await asyncio.sleep(4)
        await self.say("あとでAlchemyやるかも")
        await asyncio.sleep(8)
        await self.wait_quiet(3.0)
        ev("段階", f"3. 画面から（準備の期限切れと同じ状態に戻す・今 {mgr.state}）")
        if mgr.state == "prepared" and mgr.client is None:
            mgr._go_idle("試験: 準備の期限切れと同じ状態に戻す")
        self.desk.show("game_00_title")
        got = await self.wait_until(lambda: mgr.state != "idle", 45)
        ev("段階", f"画面の後の状態: {mgr.state}（{'準備した' if got else '準備しなかった'}）")
        await asyncio.sleep(5)
        await self.wait_quiet(3.0)
        ev("段階", "4. やろう → 部屋 → 対戦")
        await self.say("よし、やろう")
        joined = await self.wait_until(lambda: mgr.state in ("room", "playing"), 40)
        if not joined:
            ev("段階", f"部屋に入らなかった（状態 {mgr.state}）→ 「Alchemyやろう」と言い直す")
            await self.say("Alchemyやろうよ")
            joined = await self.wait_until(lambda: mgr.state in ("room", "playing"), 40)
        if not joined:
            ev("段階", "部屋に入らなかったので終わる")
            return
        await self.wait_quiet(2.0, max_wait=30)
        await asyncio.sleep(4)
        self.opp = Opponent(self.mgr.s.url, self.mgr.s.keyword, seed=self.args.seed)
        await self.opp.join()
        await self.wait_until(lambda: mgr.state == "playing", 30)
        ev("段階", "5. 対戦中の会話")
        lines = [
            ("score", "今何点？"),
            ("chat", "そういえば今日の晩ごはん何にしようかな"),
            ("why", "なんでそのカードにしたの？"),
            ("last", "さっきのラウンド何したの？"),
            ("false", None),
            ("next", "次どうするつもり？"),
        ]
        rematched = False
        for kind, text in lines:
            if time.monotonic() - T0 > self.args.limit - 240:
                break
            if mgr.state != "playing":
                # 1戦目が終わった → 終わりの反応を見てから、もう一回（台本の残りは2戦目で聞く）
                if rematched or not await self._rematch():
                    break
                rematched = True
            if not await self._wait_condition(kind):
                if mgr.state == "playing":
                    pass  # 条件が来ないまま時間が経った（そのまま聞く）
                elif rematched or not await self._rematch():
                    break
                else:
                    rematched = True
                    if not await self._wait_condition(kind):
                        break
            await self.say(text or self.false_premise())
            await asyncio.sleep(random.uniform(8, 12))
        ev("段階", "6. 対戦の終わり〜やめる")
        if mgr.state == "playing" and not rematched:
            # 台本が1戦目の間に尽きた → 終わりの反応を見てから、もう一回
            await self.wait_until(lambda: self.ended >= 1, 400)
            rematched = await self._rematch()
        if mgr.state == "playing":
            n0 = len(self.resolves)
            await self.wait_until(lambda: len(self.resolves) > n0 or mgr.state != "playing", 120)
            await asyncio.sleep(4)
            await self.say("ごめん、やっぱやめよう")
            await self.wait_until(lambda: mgr.state not in ("room", "playing"), 30)
        elif mgr.state == "room":
            await asyncio.sleep(10)
            await self.say("今日はもうやめとこう")
            await self.wait_until(lambda: mgr.state not in ("room", "playing"), 30)
        await asyncio.sleep(6)
        await self.wait_quiet(3.0, max_wait=40)
        await self.say("楽しかった？")
        await asyncio.sleep(25)  # 終わった後に自分から話すか
        await self.wait_quiet(3.0, max_wait=40)

    async def _join_room(self, line: str) -> bool:
        mgr = self.mgr
        await self.say(line)
        if await self.wait_until(lambda: mgr.state in ("room", "playing"), 40):
            return True
        ev("段階", f"部屋に入らなかった（状態 {mgr.state}）→ 言い直す")
        await self.say("Alchemyやろうよ")
        return await self.wait_until(lambda: mgr.state in ("room", "playing"), 40)

    async def script_cycles(self) -> None:
        """本番どおり全部の機能（記憶・フィードバック・感情・画面）を動かしたまま、同じ流れを1つの会話の中で繰り返す
        （`--scenario cycles`）: 「Alchemyやろう」→ 対戦 → 1ラウンド目の後に「やっぱゲームやめよう」×3。
        4回目はユーザ役が何も言わずに対戦から抜ける（接続を切る）→ イブがどうするか・次に話しかけた時に気づくか。"""
        mgr = self.mgr
        await self.wait_quiet(5.0, max_wait=90)
        chats = ["疲れたー", "今日の晩ごはん何にしよう", "眠くなってきた"]
        for i in range(4):
            ev("段階", f"{i + 1}回目")
            if not await self._join_room("Alchemyやろう"):
                ev("段階", "部屋に入らなかったので終わる")
                return
            await asyncio.sleep(6)
            self.opp = Opponent(self.mgr.s.url, self.mgr.s.keyword, seed=self.args.seed + i)
            await self.opp.join()
            await self.wait_until(lambda: mgr.state == "playing", 30)
            n0 = len(self.resolves)
            await self.wait_until(lambda: len(self.resolves) > n0 or mgr.state != "playing", 150)
            await asyncio.sleep(3)
            if i < 3:
                await self.say("やっぱゲームやめよう")
                await self.wait_until(lambda: mgr.state not in ("room", "playing"), 30)
                ev("段階", f"やめた後の状態: {mgr.state}")
                await self.opp.leave()
                await self.opp.close()
                await asyncio.sleep(8)
                await self.say(chats[i])
                await asyncio.sleep(10)
                await self.wait_quiet(3.0, max_wait=60)
                continue
            ev("段階", "ユーザ役が何も言わずに対戦から抜ける（接続を切る）")
            await self.opp.close()
            t0 = time.monotonic()
            await asyncio.sleep(90)
            st = mgr.view or {}
            seats = [{k: p.get(k) for k in ("name", "points", "connected", "autopilot")} for p in st.get("players") or []]
            ev("段階", f"黙って抜けて{time.monotonic() - t0:.0f}秒: イブの状態 {mgr.state}・{st.get('round')}ラウンド目・席 {seats}"
                       f"・`# Alchemy` に抜けた事実 {'あり' if '自動操作' in (mgr.voice_block() or '') or '切断' in (mgr.voice_block() or '') else 'なし'}")
            await self.say("ただいまー")
            await asyncio.sleep(10)
            await self.wait_quiet(3.0, max_wait=60)
            ev("段階", f"話しかけた後: イブの状態 {mgr.state}")

    async def script_silent(self) -> None:
        """対戦中にユーザ役が何も言わずに抜ける（`--scenario silent`）→ 抜けた事実を渡し、全員抜けたらイブも抜けるか。"""
        mgr = self.mgr
        await self.wait_quiet(5.0, max_wait=90)
        if not await self._join_room("Alchemyやろう"):
            return
        await asyncio.sleep(6)
        self.opp = Opponent(self.mgr.s.url, self.mgr.s.keyword, seed=self.args.seed)
        await self.opp.join()
        await self.wait_until(lambda: mgr.state == "playing", 30)
        n0 = len(self.resolves)
        await self.wait_until(lambda: len(self.resolves) > n0 or mgr.state != "playing", 150)
        await asyncio.sleep(2)
        ev("段階", "ユーザ役が何も言わずに対戦から抜ける（接続を切る）")
        await self.opp.close()
        t0 = time.monotonic()
        await self.wait_until(lambda: mgr.state not in ("room", "playing"), 90)
        ev("段階", f"イブの状態 {mgr.state}（抜けてから {time.monotonic() - t0:.0f}秒）")
        await asyncio.sleep(12)
        await self.wait_quiet(3.0, max_wait=60)
        await self.say("ただいまー")
        await asyncio.sleep(8)
        await self.wait_quiet(3.0, max_wait=60)

    async def script_rewait(self) -> None:
        """確かめられるたびに関係ない話で返す（`--scenario rewait`）: 待ち直し2回の後は確かめずに抜けるか・関係ない返事に
        ツッコむか。発話判定は待っている間も動く（早めの催促も記録する）。"""
        mgr = self.mgr
        await self.wait_quiet(5.0, max_wait=90)
        if not await self._join_room("Alchemyやろう"):
            ev("段階", "部屋に入らなかったので終わる")
            return
        t_room = time.monotonic()
        for i, line in enumerate(["そういえば明日雨らしいよ", "お腹すいたなー"]):
            n0 = mgr._nudges
            await self.wait_until(lambda: mgr._nudges > n0 or mgr.state not in ("room", "playing"), 900)
            if mgr.state not in ("room", "playing"):
                break
            ev("段階", f"確かめられた（{mgr._nudges}回目・部屋に入ってから {time.monotonic() - t_room:.0f}秒）→ 関係ない話で返す")
            await asyncio.sleep(2)
            await self.wait_quiet(1.5, max_wait=30)
            await self.say(line)
        await self.wait_until(lambda: mgr.state not in ("room", "playing"), 900)
        ev("段階", f"抜けた（部屋に入ってから {time.monotonic() - t_room:.0f}秒・状態 {mgr.state}）")
        await asyncio.sleep(10)
        await self.wait_quiet(3.0, max_wait=60)

    async def script_rules(self) -> None:
        """ルールの質問を何回も（`--scenario rules`）。毎回 準備を解き・会話を空にして同じ質問をする＝前置きで止まらずに
        答えるか・答えの最初の声までの秒数（ユーザ「5秒以上かかるなら前置きから話させるでもいい」）。"""
        mgr = self.mgr
        await self.wait_quiet(5.0, max_wait=90)
        for i in range(self.args.reps):
            if mgr.state != "idle":
                mgr._go_idle("試験: 毎回ルールが入っていない状態から聞く")
            self.vl.cache._turns.clear()
            ev("段階", f"ルールの質問 {i + 1}/{self.args.reps}")
            await self.say("Alchemyのルールってどんなのだっけ")
            await asyncio.sleep(3)
            await self.wait_quiet(3.0, max_wait=60)

    async def script_wait(self) -> None:
        """部屋で待つ・催促・見切り・返事をした時・対戦中にやめる（`--scenario wait`・ユーザ 2026-10-11 の決め方）。"""
        mgr = self.mgr
        await self.wait_quiet(5.0, max_wait=90)
        ev("段階", "A. ルールを聞く（前置きで止まらず答えるか）")
        await self.say("Alchemyのルールってどんなのだっけ")
        await asyncio.sleep(20)
        await self.wait_quiet(3.0, max_wait=60)
        ev("段階", "B. やろう → ユーザ役は部屋に入らず何も言わない（5分で確かめ・30秒返事が無ければ抜けるはず）")
        if not await self._join_room("じゃあAlchemyやろう"):
            ev("段階", "部屋に入らなかったので終わる")
            return
        t_room = time.monotonic()
        while time.monotonic() - t_room < self.args.wait_min * 60 and mgr.state in ("room", "playing"):
            await asyncio.sleep(1.0)
        ev("段階", f"B の終わり: 部屋に入ってから {time.monotonic() - t_room:.0f}秒・状態 {mgr.state}")
        await asyncio.sleep(10)
        await self.wait_quiet(3.0, max_wait=60)
        ev("段階", "C. もう一度 → 確かめられたら返事をして入る → 対戦中にやめる")
        if not await self._join_room("ごめん、やっぱやろう"):
            ev("段階", "部屋に入らなかったので終わる")
            return
        t_room = time.monotonic()
        await self.wait_until(lambda: mgr._nudged_at is not None or mgr.state not in ("room", "playing"),
                              self.args.wait_min * 60)
        ev("段階", f"確かめられた（部屋に入ってから {time.monotonic() - t_room:.0f}秒）")
        await asyncio.sleep(2)
        await self.wait_quiet(1.5, max_wait=30)
        await self.say("ごめんごめん、今から入る")
        await asyncio.sleep(12)
        ev("段階", f"返事の後の状態: {mgr.state}")
        if mgr.state not in ("room", "playing"):
            return
        self.opp = Opponent(self.mgr.s.url, self.mgr.s.keyword, seed=self.args.seed)
        await self.opp.join()
        await self.wait_until(lambda: mgr.state == "playing", 30)
        await self.wait_until(lambda: bool(self.resolves) or mgr.state != "playing", 150)
        await asyncio.sleep(3)
        await self.say("やっぱゲームやめよう")
        left = await self.wait_until(lambda: mgr.state not in ("room", "playing"), 30)
        ev("段階", f"対戦中にやめた後: イブの状態 {mgr.state}（{'抜けた' if left else '抜けていない'}）")
        await asyncio.sleep(5)
        st = self.opp.last_state or {}
        seats = [{k: p.get(k) for k in ("name", "points", "connected", "autopilot", "bot")} for p in st.get("players") or []]
        ev("段階", f"ユーザ役の画面の様子（イブが抜けた5秒後）: {st.get('round')}ラウンド目・{st.get('phase')}・席 {seats}")
        n_ev = len(self.opp.events)
        await asyncio.sleep(25)
        later = [e.get("type") for e in self.opp.events[n_ev:]]
        ev("段階", f"その後25秒にユーザ役に届いた出来事: {later}")
        await self.opp.leave()
        await asyncio.sleep(20)  # やめた後に自分から何か言うか
        await self.wait_quiet(3.0, max_wait=60)

    async def _wait_condition(self, kind: str) -> bool:
        """聞く前提が揃うまで待つ（対戦が終わったら False）。"""
        mgr = self.mgr
        n0 = len(self.resolves)
        if kind == "score":
            cond = lambda: bool(self.resolves) or mgr.state != "playing"  # noqa: E731
        elif kind == "last":
            cond = lambda: len(self.resolves) > n0 or mgr.state != "playing"  # noqa: E731
        elif kind == "why":
            cond = lambda: mgr.state != "playing" or any("を置いた" in m for m in mgr.moves)  # noqa: E731
        else:
            cond = lambda: True  # noqa: E731
        await self.wait_until(cond, 150)
        return mgr.state == "playing"

    async def _rematch(self) -> bool:
        mgr = self.mgr
        await self.wait_until(lambda: self.ended >= self.match_count, 300)
        await asyncio.sleep(4)
        await self.wait_quiet(4.0, max_wait=40)
        before = self.match_count
        await self.say("もう一回やろう")
        await self.wait_until(lambda: mgr._want_ready or mgr.state == "playing", 40)
        await asyncio.sleep(5)
        await self.opp.ready_again()
        ok = await self.wait_until(lambda: self.match_count > before, 60)
        ev("段階", f"もう一回: {'始まった' if ok else '始まらなかった'}")
        return ok

    # --- 実行 ---
    async def main(self) -> int:
        self.opp = None
        self.build()
        run = asyncio.create_task(self.vl.run())
        watcher = asyncio.create_task(self.watch_vision())
        try:
            await asyncio.wait_for(self.started.wait(), timeout=300)
            ev("段階", "起動完了")
            script = {"wait": self.script_wait, "rules": self.script_rules,
                      "cycles": self.script_cycles, "rewait": self.script_rewait,
                      "silent": self.script_silent}.get(self.args.scenario, self.script)
            await asyncio.wait_for(script(), timeout=self.args.limit)
        except asyncio.TimeoutError:
            ev("段階", "時間切れで打ち切り")
        except Exception as e:  # noqa: BLE001
            logging.exception("台本の途中で例外")
            ev("段階", f"例外で打ち切り: {type(e).__name__}: {e}")
        finally:
            try:
                await asyncio.wait_for(self.vl.stop(), timeout=60)
            except Exception:
                logging.exception("停止で例外")
            run.cancel()
            watcher.cancel()
            if self.opp is not None:
                await self.opp.close()
            self.save()
        return 0

    def save(self) -> None:
        out = self.out
        (out / "turns.json").write_text(json.dumps(self.turns, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        (out / "decisions.json").write_text(json.dumps(self.decisions, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "record.json").write_text(json.dumps(self.record, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
        stats = {k: {kk: vv for kk, vv in v.items()} for k, v in (self.mgr.stats or {}).items()}
        (out / "summary.json").write_text(json.dumps({
            "mode": self.mgr.brain.mode, "matches": self.match_count, "ended": self.ended,
            "triggers": self.triggers, "last_result": self.mgr.last_result, "stats": stats,
            "usage": self.vl.usage.format_summary() if self.vl.usage else None,
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        if self.vl.usage:
            print("使用量:\n" + self.vl.usage.format_summary())


def main() -> int:
    global _tl_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="action", choices=["action", "strategy"])
    ap.add_argument("--limit", type=float, default=1500.0, help="打ち切りの秒数（台本全体）")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--scenario", default="full", choices=["full", "wait", "rules", "cycles", "rewait", "silent"])
    ap.add_argument("--wait-min", type=float, default=8.0, help="wait: 部屋で待たせる上限の分")
    ap.add_argument("--reps", type=int, default=10, help="rules: 聞く回数")
    args = ap.parse_args()
    out = OUT / (args.mode if args.scenario == "full" else f"{args.scenario}_{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = setup(args.mode, out)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for name in ("httpx", "LiteLLM", "litellm", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _tl_file = open(out / "timeline.jsonl", "w", encoding="utf-8")
    print("記憶の複製:", tmp)
    try:
        return asyncio.run(Live(args, out).main())
    finally:
        _tl_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
