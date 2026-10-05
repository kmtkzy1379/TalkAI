"""S1 Irodori TTS の決定論テスト（API・GPU 不要）。

- 口パク計画（音量方式）の形
- IrodoriTTS: ローカルの偽 HTTP サーバで 合成 / 失敗→None / 口パク計画つき
- IrodoriServer: 既存サーバの再利用・起動・ロード失敗・起動中の終了・打ち切り・止めるのは自分が起動したものだけ
  （起動と health は偽物を注入）
- 取り残しサーバの掃除: 持ち主のいないものだけ止める・手で起動したもの／生きている EVE のものは止めない
- Job Object: 止めたまま作った子（＋その子が作る孫）を Job に入れて再開 → Job を止めると孫まで消える（実プロセス）
- 配線: TTS_BACKEND で選ぶ（自動フォールバックなし）・Irodori は同時合成 2・ウォームアップは tts.warmup

実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_tts_irodori.py
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

from eve.config import Config  # noqa: E402
from eve.response.lipsync import level_at, plan_from_wav  # noqa: E402
from eve.response.tts_irodori import SERVER_MARKER, IrodoriServer, IrodoriTTS  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}  {detail}")


def make_wav(samples: np.ndarray, sr: int = 48000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 口パク計画（音量方式）
# ---------------------------------------------------------------------------

def t_plan() -> None:
    sr = 48000
    t = np.arange(int(0.6 * sr)) / sr
    loud = 0.5 * np.sin(2 * np.pi * 220 * t)
    x = np.concatenate([loud[: sr // 5], np.zeros(sr // 5), loud[: sr // 5]])  # 声・無音・声
    plan = plan_from_wav(make_wav(x, sr))
    check("口パク: 計画ができる（(t0,t1,v0,v1) の並び）", bool(plan) and all(len(p) == 4 for p in plan))
    check("口パク: 声の所で開き、無音の所で閉じる",
          plan is not None and level_at(plan, 0.1) > 0.8 and level_at(plan, 0.3) < 0.05
          and level_at(plan, 0.5) > 0.8,
          f"{level_at(plan, 0.1) if plan else None} {level_at(plan, 0.3) if plan else None}")
    check("口パク: 長さは音声と同じ（±1区間）", plan is not None and abs(plan[-1][1] - 0.6) < 0.03,
          f"{plan[-1][1] if plan else None}")
    check("口パク: 0..1 に収まる", plan is not None and all(0.0 <= v <= 1.0 for p in plan for v in p[2:]))
    check("口パク: 無音だけなら None", plan_from_wav(make_wav(np.zeros(sr // 2), sr)) is None)
    check("口パク: 壊れた WAV は None（音声だけは出す）", plan_from_wav(b"not a wav") is None)


# ---------------------------------------------------------------------------
# IrodoriTTS（ローカルの偽サーバ）
# ---------------------------------------------------------------------------

async def _fake_server(behavior: dict):
    from aiohttp import web

    wav = make_wav(0.3 * np.sin(2 * np.pi * 220 * np.arange(4800) / 48000))
    seen: list = []

    async def synth(req):
        body = await req.json()
        seen.append(body)
        if behavior.get("status", 200) != 200:
            return web.json_response({"detail": "boom"}, status=behavior["status"])
        return web.Response(body=wav, content_type="audio/wav")

    async def health(req):
        return web.json_response({"ok": True, "model": "m", "voice": "v", "steps": 6})

    app = web.Application()
    app.router.add_post("/synthesize", synth)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port, seen, wav


async def t_client() -> None:
    beh: dict = {}
    runner, port, seen, wav = await _fake_server(beh)
    try:
        tts = IrodoriTTS(IrodoriServer(port=port, python="py", repo=".", voice_ref="v.wav"),
                         seed=7, steps=8)
        out = await tts.generate("こんにちは。")
        check("合成: WAV が返る", out == wav)
        # match_plain: 感情の絵文字が付いた文も「絵文字なし」と同じ長さで（ユーザ 2026-09-26「速度は揃えて感情載せる」）
        check("合成: seed と steps を送る（声を揃える）・絵文字の文は長さをそろえる",
              seen and seen[-1] == {"text": "こんにちは。", "seed": 7, "num_steps": 8, "match_plain": True},
              f"{seen[-1:]}")
        w, plan = await tts.generate_with_meta("はい。")
        check("合成: generate_with_meta は (WAV, 口パク計画)", w == wav and bool(plan))
        check("合成: 空文字は送らず None", await tts.generate("  ") is None and len(seen) == 2)
        # 節の終わりの「、」は落として送る（「、」で終わる短い節を1つだけ合成すると、言っていない音で埋める・2026-09-26）
        await tts.generate("そっか、")
        check("合成: 節の終わりの「、」は落として送る", seen[-1]["text"] == "そっか", f"{seen[-1:]}")
        await tts.generate("そっか、😭")
        check("合成: 後ろに感情の絵文字があっても「、」だけ落とす", seen[-1]["text"] == "そっか😭", f"{seen[-1:]}")
        await tts.generate("え、ほんとに？")
        check("合成: 途中の「、」はそのまま", seen[-1]["text"] == "え、ほんとに？", f"{seen[-1:]}")
        n = len(seen)
        check("合成: 「、」だけの節は送らない（None）", await tts.generate("、") is None and len(seen) == n)
        beh["status"] = 500
        check("合成: サーバの失敗は None（落とさない）", await tts.generate("あ。") is None)
        w2, p2 = await tts.generate_with_meta("あ。")
        check("合成: 失敗時の generate_with_meta は (None, None)", w2 is None and p2 is None)
        await tts.close()
    finally:
        await runner.cleanup()
    # つながらない
    tts2 = IrodoriTTS(IrodoriServer(port=1, python="py", repo=".", voice_ref="v.wav"))
    check("合成: つながらない時は None（VOICEVOX に切り替えない）", await tts2.generate("あ。") is None)
    await tts2.close()


# ---------------------------------------------------------------------------
# IrodoriServer（起動と health を偽物に）
# ---------------------------------------------------------------------------

class _Proc:
    def __init__(self) -> None:
        self.rc = None
        self.pid = 12345
        self.killed = False

    def poll(self):
        return self.rc

    @property
    def returncode(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc


def _server(health_seq, *, spawn_log=None, proc=None, **kw):
    seq = list(health_seq)

    async def health():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def spawn(cmd, cwd):
        if spawn_log is not None:
            spawn_log.append((cmd, cwd))
        return proc or _Proc()

    here = os.path.abspath(__file__)
    kw.setdefault("python", sys.executable)
    kw.setdefault("voice_ref", here)  # 存在するファイルなら何でもよい
    return IrodoriServer(repo=".", port=50999, health_fn=health, spawn_fn=spawn, **kw)


async def t_server() -> None:
    ok = {"ok": True, "model": "m", "voice": "v", "steps": 6}
    log: list = []
    s = _server([ok], spawn_log=log)
    check("起動: 既に通るサーバがあれば使う（起動しない）", await s.ensure() and not log and not s.owned)

    log = []
    s = _server([None, None, {"ok": False}, ok], spawn_log=log)
    r = await s.ensure()
    check("起動: 無ければ起動して準備完了を待つ", r and len(log) == 1 and s.owned)
    cmd = log[0][0] if log else []
    check("起動: 目印・持ち主の pid・ポート・話者・steps・seed を渡す",
          SERVER_MARKER in cmd and "--owner-pid" in cmd and str(os.getpid()) in cmd
          and "50999" in cmd and "--voice-ref" in cmd and "--steps" in cmd and "--seed" in cmd, f"{cmd}")

    s = _server([None, {"ok": False, "error": "CUDA out of memory"}])
    check("起動: ロード失敗は False（理由をログに）", await s.ensure() is False)

    p = _Proc()
    p.rc = 1
    s = _server([None], proc=p)
    check("起動: 起動中にサーバが終了したら False", await s.ensure() is False)

    s = _server([None], start_timeout=0.3)
    t0 = time.monotonic()
    r = await s.ensure()
    check("起動: 準備できないまま上限を過ぎたら False", r is False and time.monotonic() - t0 < 2.0)

    log = []
    s = _server([None], spawn_log=log, python=os.path.join(os.path.dirname(__file__), "no_such_python.exe"))
    check("起動: python が無ければ起動しない（False）", await s.ensure() is False and not log)

    # 止めるのは自分が起動したものだけ
    s = _server([ok])
    await s.ensure()
    await s.stop()
    check("停止: 既存のサーバ（自分が起動していない）は止めない", not s.owned)
    killed: list = []
    import eve.response.tts_irodori as mod

    saved = mod._kill_tree
    try:
        mod._kill_tree = lambda pid: killed.append(pid)
        s = _server([None, ok])
        await s.ensure()
        s._job = None  # Job が無い環境（Windows 以外等）の経路
        await s.stop()
        check("停止: 自分が起動したものは止める（Job が無ければプロセスツリーごと）",
              killed == [12345] and not s.owned, f"{killed}")
    finally:
        mod._kill_tree = saved


# ---------------------------------------------------------------------------
# 取り残しサーバの掃除
# ---------------------------------------------------------------------------

class _P:
    def __init__(self, pid, cmd, parent=None) -> None:
        self.info = {"pid": pid, "cmdline": cmd}
        self._parent = parent
        self.killed = False
        self.kids: list = []

    def parent(self):
        return self._parent

    def cmdline(self):
        return self.info["cmdline"]

    def children(self, recursive=False):
        return self.kids

    def kill(self):
        self.killed = True


def t_sweep() -> None:
    from eve.cleanup import sweep_orphan_tts_servers

    orphan = _P(1, ["py", "irodori_server.py", "--owner-pid", "900", SERVER_MARKER])
    orphan_child = _P(2, ["py3", "irodori_server.py", "--owner-pid", "900", SERVER_MARKER], parent=orphan)
    orphan.kids = [orphan_child]
    manual = _P(3, ["py", "irodori_server.py", SERVER_MARKER])  # 持ち主なし＝手で起動
    alive = _P(4, ["py", "irodori_server.py", "--owner-pid", "901", SERVER_MARKER])
    other = _P(5, ["py", "something_else.py", "--owner-pid", "900"])
    procs = [orphan, orphan_child, manual, alive, other]
    n = sweep_orphan_tts_servers(process_iter=lambda: procs, pid_exists=lambda pid: pid == 901)
    check("掃除: 持ち主のいないサーバを止める（ランチャの子まで）", orphan.killed and orphan_child.killed)
    check("掃除: ランチャと子を二重に数えない", n == 1, f"{n}")
    check("掃除: 手で起動したサーバ（--owner-pid なし）は止めない", not manual.killed)
    check("掃除: 生きている EVE のサーバは止めない", not alive.killed)
    check("掃除: 目印の無いプロセスには触らない", not other.killed)


# ---------------------------------------------------------------------------
# Job Object（実プロセス・Windows のみ）
# ---------------------------------------------------------------------------

def t_job() -> None:
    if os.name != "nt":
        print("SKIP Job Object（Windows 以外）")
        return
    import psutil

    from eve.winjob import CREATE_NO_WINDOW, CREATE_SUSPENDED, assign_and_resume, create_kill_on_close_job, terminate_job

    # 子が孫を作る（venv のランチャ → 本体の python と同じ形）
    code = ("import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            "time.sleep(60)")
    job = create_kill_on_close_job()
    check("Job: 作れる", bool(job))
    p = subprocess.Popen([sys.executable, "-c", code], creationflags=CREATE_SUSPENDED | CREATE_NO_WINDOW)
    try:
        in_job = assign_and_resume(job, p.pid)
        check("Job: 止めたまま作った子を Job に入れて再開できる", in_job)
        # 孫（sleep だけするプロセス）ができるまで待つ。venv のランチャを挟むと段数が増えるので、
        # 段数ではなく「孫のコマンドライン」で見分ける。
        deadline = time.time() + 15
        desc: list = []
        grandchild = False
        while time.time() < deadline and not grandchild:
            try:
                desc = psutil.Process(p.pid).children(recursive=True)
                grandchild = any("subprocess" not in " ".join(d.cmdline()) for d in desc)
            except Exception:
                desc = []
            time.sleep(0.2)
        pids = [p.pid] + [d.pid for d in desc]
        check("Job: 子が孫を作った（再開できている）", grandchild, f"{pids}")
        terminate_job(job)
        time.sleep(0.5)
        survivors = [pid for pid in pids if psutil.pid_exists(pid) and psutil.Process(pid).status() != "zombie"]
        check("Job: Job を止めると子も孫も消える", not survivors, f"生存 {survivors}")
    finally:
        try:
            p.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 配線
# ---------------------------------------------------------------------------

def t_wiring() -> None:
    import inspect

    import eve.voice_loop as vlmod
    from eve.response.tts import VoicevoxTTS

    saved = Config.TTS_BACKEND
    try:
        Config.TTS_BACKEND = "irodori"
        check("配線: TTS_BACKEND=irodori → IrodoriTTS", isinstance(vlmod._make_tts(), IrodoriTTS))
        Config.TTS_BACKEND = "voicevox"
        check("配線: TTS_BACKEND=voicevox → VoicevoxTTS", isinstance(vlmod._make_tts(), VoicevoxTTS))
        Config.TTS_BACKEND = "nope"
        try:
            vlmod._make_tts()
            check("配線: 知らない TTS_BACKEND は起動時に止める（黙って別の声にしない）", False)
        except ValueError:
            check("配線: 知らない TTS_BACKEND は起動時に止める（黙って別の声にしない）", True)
        Config.TTS_BACKEND = "irodori"
        vl = vlmod.VoiceLoop()
        check("配線: VoiceLoop の既定 TTS は Irodori（ユーザ裁定 2026-09-25）", isinstance(vl.tts, IrodoriTTS))
        check("配線: Irodori の同時合成は 2", vl.orchestrator._tts_concurrency == 2)
        check("配線: 応答の TTS は generate_with_meta（口パク計画つき）",
              vl.orchestrator._tts_fn == vl.tts.generate_with_meta)
        check("配線: 構築時にはサーバを起動しない（warmup で用意）", vl.tts.server._proc is None)
    finally:
        Config.TTS_BACKEND = saved
    src = inspect.getsource(vlmod.VoiceLoop.warmup)
    check("配線: ウォームアップは tts.warmup（Irodori はここでサーバを用意）", "self.tts.warmup" in src)
    src_run = inspect.getsource(vlmod.VoiceLoop.run)
    check("配線: 取り残しサーバの掃除はウォームアップより前",
          "sweep_orphan_tts_servers" in src_run
          and src_run.index("sweep_orphan_tts_servers") < src_run.index("await self.warmup()"))
    check("配線: VoicevoxTTS にも warmup と concurrency（口を揃える）",
          hasattr(VoicevoxTTS, "warmup") and VoicevoxTTS.concurrency == 3)


async def main() -> None:
    t_plan()
    await t_client()
    await t_server()
    t_sweep()
    t_job()
    t_wiring()


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
