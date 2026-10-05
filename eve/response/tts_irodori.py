"""Irodori-TTS クライアント＋サーバの自動起動（応答の TTS・S1・ユーザ裁定 2026-09-25）。

- `IrodoriTTS`: `VoicevoxTTS` と同じ口（`generate` / `generate_with_meta` / `warmup` / `close`）。
  失敗は None（落とさない）。口パク計画は WAV の音量から作る（`lipsync.plan_from_wav`）。
- `IrodoriServer`: `tools/irodori_server.py`（Irodori の venv で動く）を **EVE が全自動で**起動・再利用・停止する。
  - 起動時: `/health` が通るサーバが既にあれば使う（手で起動しておいたもの等）。無ければ起動し、
    **EVE が落ちたら一緒に落ちるよう Job Object（KILL_ON_JOB_CLOSE だけ）に入れる**（`eve/winjob.py`）。
    サーバ側も `--owner-pid` で EVE を見張り、EVE が消えたら自分で終わる（Job が効かなかった時の保険）
  - 終了時: 自分が起動したものだけ止める（Job ごと＝ランチャの子まで落とす）
  - 起動時の掃除: 前回の EVE が残したサーバ（持ち主の EVE がもういない）を止める（`eve/cleanup.py`）

**VOICEVOX への自動フォールバックはしない**（ユーザ裁定「急に切り替わると不自然だから話者はこっちで選択」）。
Irodori が使えない間は音声が出ない（ログに残る）。切り替えは `.env` の `TTS_BACKEND` で行う。
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from ..config import Config

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
SERVER_SCRIPT = REPO / "tools" / "irodori_server.py"
SERVER_MARKER = "--eve-irodori-server"  # サーバのコマンドラインに付ける目印（掃除で探す）
# 起動〜準備完了の上限。実測 2026-09-25: 初回 約32秒（torch の import・モデルロード 26.7秒・
# ウォームアップ 2.1秒）。HF のキャッシュ確認がネットワークを待つこともあるので余裕を持たせる。
START_TIMEOUT_SEC = 180.0

# 節の終わりの「、」（後ろに感情の絵文字が付いていても）。`clause_text` で落とす
_TRAILING_COMMA = re.compile(r"、(?=[^\w]*$)")


def clause_text(text: str) -> str:
    """Irodori に渡す文字: 節の終わりの「、」を落とす（感情の絵文字は残す）。

    「、」で終わる短い節を1つだけ合成すると、Irodori は「まだ続く文」と受け取って長い時間を予測し（「そっか、」だけで
    約1.7秒）、その時間を**言っていない音で埋める**（「うん、」→「うん入れちねんだ」等）。S7 で応答を「、」でも区切るように
    してから出た（ユーザ 2026-09-26「同じ文章でも音声おかしくなってる部分が多数ある」）。実測（`tools/clause_ending_check.py`・
    短い節12語×seed 4・SenseVoice で聞き取り）: 言っていない音 19/48 → **「、」を取ると 3/48**（残りは「え」「うん」の聞き分け）・
    長さ 1.70→0.88秒。長めの節は変わらない。区切りの間は再生側（SPEECH_GAP_SEC）が作る。画面・記憶の文字は「、」のまま。
    """
    return _TRAILING_COMMA.sub("", text)


def resolve_repo_path(p: str) -> str:
    """`.env` の相対パスは**このリポジトリ基準**で解決する（起動した場所に依らない）。"""
    q = Path(p)
    return str(q if q.is_absolute() else (REPO / q).resolve())


class IrodoriServer:
    def __init__(
        self,
        *,
        python: Optional[str] = None,
        repo: Optional[str] = None,
        port: Optional[int] = None,
        voice_ref: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        host: str = "127.0.0.1",
        start_timeout: float = START_TIMEOUT_SEC,
        health_fn: Optional[Callable[[], Awaitable[Optional[dict]]]] = None,
        spawn_fn: Optional[Callable[[list, str], Any]] = None,
    ) -> None:
        self.repo = resolve_repo_path(repo or Config.IRODORI_REPO)
        self.python = resolve_repo_path(python or Config.IRODORI_PYTHON)
        self.port = int(port or Config.IRODORI_PORT)
        self.voice_ref = resolve_repo_path(voice_ref or Config.IRODORI_VOICE_REF)
        self.steps = int(Config.IRODORI_STEPS if steps is None else steps)
        self.seed = int(Config.IRODORI_SEED if seed is None else seed)
        self.url = f"http://{host}:{self.port}"
        self.start_timeout = start_timeout
        self._health_fn = health_fn or self._http_health
        self._spawn_fn = spawn_fn or self._popen
        self.owned = False       # このランで自分が起動したか（止めるのは自分が起動したものだけ）
        self._proc = None
        self._job = None
        self._lock: Optional[asyncio.Lock] = None
        self.log_path = Path(tempfile.gettempdir()) / "eve_irodori" / "server.log"

    # --- 状態 -----------------------------------------------------------

    async def _http_health(self) -> Optional[dict]:
        """`/health` の中身。つながらなければ None。"""
        import aiohttp

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2.0)) as s:
                async with s.get(f"{self.url}/health") as r:
                    if r.status != 200:
                        return {"ok": False, "error": f"HTTP {r.status}"}
                    return await r.json()
        except Exception:
            return None

    def _proc_exited(self) -> bool:
        return self._proc is not None and self._proc.poll() is not None

    # --- 起動 -----------------------------------------------------------

    def command(self) -> list:
        return [self.python, str(SERVER_SCRIPT),
                "--port", str(self.port), "--voice-ref", self.voice_ref,
                "--steps", str(self.steps), "--seed", str(self.seed),
                "--irodori-repo", self.repo, "--owner-pid", str(os.getpid()), SERVER_MARKER]

    def _popen(self, cmd: list, cwd: str):
        from ..winjob import CREATE_NO_WINDOW, CREATE_SUSPENDED, assign_and_resume, create_kill_on_close_job

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "w", encoding="utf-8")  # PIPE にしない（読まないと詰まって止まる）
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        flags = 0
        if os.name == "nt":
            flags = CREATE_NO_WINDOW | CREATE_SUSPENDED
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        finally:
            logf.close()  # 子が複製を持つので親の分は閉じてよい
        if os.name == "nt":
            self._job = create_kill_on_close_job()
            if not assign_and_resume(self._job, proc.pid):
                logger.warning("Irodori サーバを Job に入れられなかった（EVE の異常終了時は"
                               "サーバ側の親監視と次回起動時の掃除で止まる）")
        return proc

    async def ensure(self) -> bool:
        """使えるサーバを用意する（既にあれば使う・無ければ起動）。準備できたら True。"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            h = await self._health_fn()
            if h and h.get("ok"):
                self._note_ready(h, "既存")
                return True
            if h is None and (self._proc is None or self._proc_exited()):
                if not Path(self.python).is_file():
                    logger.error("Irodori の python が無い: %s（.env の IRODORI_PYTHON）", self.python)
                    return False
                if not Path(self.voice_ref).is_file():
                    logger.error("話者の参照音声が無い: %s（.env の IRODORI_VOICE_REF）", self.voice_ref)
                    return False
                self._release_job()  # 落ちた前のサーバの Job が残っていれば閉じる
                logger.info("Irodori サーバを起動します（ポート %d・ログ %s）", self.port, self.log_path)
                self._proc = self._spawn_fn(self.command(), self.repo)
                self.owned = True
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.start_timeout:
                h = await self._health_fn()
                if h and h.get("ok"):
                    self._note_ready(h, f"起動 {time.monotonic() - t0:.1f}秒")
                    return True
                if h and h.get("error"):
                    logger.error("Irodori サーバのロードに失敗: %s（ログ %s）", h["error"], self.log_path)
                    return False
                if self._proc_exited():
                    logger.error("Irodori サーバが起動中に終了した（rc=%s・ログ %s）",
                                 self._proc.returncode, self.log_path)
                    return False
                await asyncio.sleep(0.5)
            logger.error("Irodori サーバが %.0f 秒で準備できなかった（ログ %s）", self.start_timeout, self.log_path)
            return False

    def _note_ready(self, h: dict, how: str) -> None:
        if not h.get("match_plain"):
            # 手で起動しておいた古いサーバ等。感情の絵文字を付けた文が長く（ゆっくり）なる
            logger.warning("Irodori サーバが長さそろえ（match_plain）に対応していない — 止めて起動し直すと効く")
        logger.info("Irodori サーバ 準備完了（%s・%s・steps %s・話者 %s）",
                    how, h.get("model"), h.get("steps"), Path(str(h.get("voice") or "")).name)
        voice = str(h.get("voice") or "")
        if voice and Path(voice).resolve() != Path(self.voice_ref).resolve():
            # 手で起動しておいたサーバが別の声のまま動いている。**勝手に止めない**（使うのは同じ）が、
            # 気づけるように出しておく。
            logger.warning("既存の Irodori サーバの話者が .env と違う（サーバ %s / .env %s）",
                           voice, self.voice_ref)

    # --- 停止 -----------------------------------------------------------

    def _release_job(self) -> None:
        job, self._job = self._job, None
        if job:
            from ..winjob import terminate_job

            terminate_job(job)

    async def stop(self) -> None:
        """自分が起動したサーバだけ止める（Job ごと＝ランチャの子まで落ちる）。"""
        if not self.owned:
            return
        self.owned = False
        proc, self._proc = self._proc, None
        job, self._job = self._job, None
        if job:
            from ..winjob import terminate_job

            terminate_job(job)
        elif proc is not None and proc.poll() is None:
            _kill_tree(proc.pid)
        if proc is not None:
            try:
                await asyncio.to_thread(proc.wait, 5)
            except Exception:
                pass
        logger.info("Irodori サーバを止めた")


def _kill_tree(pid: int) -> None:
    """Job が無い時の後始末（`Popen.kill()` だけだとランチャの子が残る）。"""
    try:
        import psutil

        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
        p.kill()
    except Exception:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        except Exception:
            logger.debug("サーバの停止に失敗（無視）", exc_info=True)


class IrodoriTTS:
    """応答の TTS（`VoicevoxTTS` と同じ口）。"""

    # 応答の文ごとの合成を同時にいくつ投げるか（`ResponseOrchestrator.tts_concurrency`）。
    # サーバは1本ずつ処理するので、2 あれば次の文が順番待ちに入り HTTP の往復ぶんの隙間が消える。
    concurrency = 2

    def __init__(self, server: Optional[IrodoriServer] = None, *,
                 seed: Optional[int] = None, steps: Optional[int] = None,
                 timeout_sec: float = 30.0) -> None:
        self.server = server or IrodoriServer()
        self.seed = int(Config.IRODORI_SEED if seed is None else seed)
        self.steps = int(Config.IRODORI_STEPS if steps is None else steps)
        self._timeout = timeout_sec
        self._session = None
        self._restarting: Optional[asyncio.Task] = None
        # 感情の絵文字が付いた文でも、絵文字を除いた文と同じ長さで合成させる（サーバの match_plain・S5）。
        # ユーザ 2026-09-26「ゆっくりしゃべられると演技臭いしテンポが悪くなるから速度は揃えて感情載せる」
        self.match_emoji_length = True

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def generate(self, text: str) -> Optional[bytes]:
        """文 → WAV バイト（48kHz mono PCM16）。失敗は None（落とさない）。"""
        text = clause_text(text or "")
        if not text.strip():
            return None
        try:
            s = await self._get_session()
            async with s.post(f"{self.server.url}/synthesize",
                              json={"text": text, "seed": self.seed, "num_steps": self.steps,
                                    "match_plain": self.match_emoji_length}) as r:
                if r.status != 200:
                    detail = (await r.text())[:120]
                    logger.warning("TTS(Irodori) 失敗 HTTP %d: %s（%.20s）", r.status, detail, text)
                    return None
                return await r.read()
        except Exception as e:
            logger.warning("TTS(Irodori) 失敗（None を返す）: %s", e)
            self._maybe_restart()
            return None

    async def generate_with_meta(self, text: str) -> tuple[Optional[bytes], Optional[list]]:
        """文 → `(WAV, 口の開閉計画)`。計画は音量から作る（Irodori は音ごとの時刻を出さない）。"""
        wav = await self.generate(text)
        if wav is None:
            return None, None
        from .lipsync import plan_from_wav

        return wav, plan_from_wav(wav)

    async def warmup(self) -> None:
        """サーバを用意（無ければ起動してロードを待つ）→ 1回合成して HTTP の経路も温める。"""
        if not await self.server.ensure():
            raise RuntimeError("Irodori サーバを用意できない（ログを確認）")
        if await self.generate("あ。") is None:
            raise RuntimeError("Irodori サーバの試し合成に失敗")

    def _maybe_restart(self) -> None:
        """つながらない時: 自分が起動したサーバが落ちていれば背景で起動し直す（1本だけ）。

        その間の文は音声なし（VOICEVOX へは切り替えない＝ユーザ裁定）。
        """
        srv = self.server
        if not srv.owned or not srv._proc_exited():
            return
        if self._restarting is not None and not self._restarting.done():
            return
        logger.warning("Irodori サーバが落ちていたので起動し直す")
        srv.owned = False  # 落ちたものは止め直さない（ensure が新しく起動して owned を立て直す）
        try:
            self._restarting = asyncio.get_running_loop().create_task(srv.ensure())
        except RuntimeError:
            pass

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        await self.server.stop()
