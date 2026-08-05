r"""J-3 inc2 — 専用アカウント + Job Object でのサンドボックス実行（Windows・ctypes）。

`setup_sandbox.py apply` が用意した `eve_sandbox` アカウントで子プロセスを起動する。

【なぜ subprocess ではないのか】
Python の `subprocess` は Windows のトークン/資格情報を渡す引数を持たない。別アカウントでの
起動は `CreateProcessWithLogonW`（特権不要・`runas` の実体）を直接叩くしかない。
`CreateProcessAsUserW` は `SeAssignPrimaryTokenPrivilege` を要求し、通常のユーザは保持しない。

【なぜ runner スタブを噛ませるのか】
`CreateProcessWithLogonW` は `bInheritHandles` 引数を持たず、ハンドルを継承しない。つまり
STARTUPINFO 経由で stdout/stderr のパイプやファイルを子に渡せない。そこで
「job.json を置く → 子の runner.py がそれを読んで実行し result.json を書く → 親が読む」
というファイル受け渡しにして、ハンドル継承への依存を**構造的に無くす**。
内側タイムアウトも runner 側の `subprocess.run(timeout=)` で素直に書ける。

【起動順序が仕様】
CREATE_SUSPENDED で作る → `AssignProcessToJobObject` → `ResumeThread`。
先に走らせると Job に入る前に子孫を撒ける窓ができる。**Job に入れられなければ実行しない**
（`TerminateProcess` して失敗を返す＝安全側に倒す）。

【ctypes の規律】
`restype` だけでなく **`argtypes` も必ず書く**。どちらか欠けると 64bit ハンドルが切り捨て
られ、**例外にならずに**隔離が壊れる（PoC で `IsProcessInJob` が OverflowError を出した）。

【実測で確認済み（PoC 2026-08-05・11/11）】
メモリ上限が効く / `CloseHandle` だけで子と孫が全滅（孤児ゼロ）/ `Popen.kill()` では孫が
生き残る / ActiveProcessLimit で fork bomb が頭打ち / `TerminateJobObject` が 0.1 秒 /
UI 制限で `ExitWindowsEx` が失敗（Users は既定で SeShutdownPrivilege を持つので ACL では防げない）。
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SANDBOX_USER = "eve_sandbox"
_CRED_PATH = Path(os.environ.get("LOCALAPPDATA", "")) / "TalkAI" / "sandbox.cred"

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
adv = ctypes.WinDLL("advapi32", use_last_error=True)

SIZE_T = ctypes.c_size_t
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOB_BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", SIZE_T), ("MaximumWorkingSetSize", SIZE_T),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ULONG_PTR),
                ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _JOB_EXTENDED(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JOB_BASIC), ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", SIZE_T), ("JobMemoryLimit", SIZE_T),
                ("PeakProcessMemoryUsed", SIZE_T), ("PeakJobMemoryUsed", SIZE_T)]


class _JOB_UI(ctypes.Structure):
    _fields_ = [("UIRestrictionsClass", wintypes.DWORD)]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


for _fn, _res, _args in (
    ("CreateJobObjectW", wintypes.HANDLE, [wintypes.LPVOID, wintypes.LPCWSTR]),
    ("SetInformationJobObject", wintypes.BOOL,
     [wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD]),
    ("AssignProcessToJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.HANDLE]),
    ("TerminateJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
    ("TerminateProcess", wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
    ("ResumeThread", wintypes.DWORD, [wintypes.HANDLE]),
    ("CloseHandle", wintypes.BOOL, [wintypes.HANDLE]),
    ("WaitForSingleObject", wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]),
    ("GetExitCodeProcess", wintypes.BOOL,
     [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]),
):
    getattr(k32, _fn).restype = _res
    getattr(k32, _fn).argtypes = _args

adv.CreateProcessWithLogonW.restype = wintypes.BOOL
adv.CreateProcessWithLogonW.argtypes = [
    wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,   # user, domain, password
    wintypes.DWORD,                                         # logon flags
    wintypes.LPCWSTR, wintypes.LPWSTR,                      # app name, command line
    wintypes.DWORD, wintypes.LPVOID, wintypes.LPCWSTR,      # flags, env, cwd
    ctypes.POINTER(_STARTUPINFOW), ctypes.POINTER(_PROCESS_INFORMATION)]

_JobExtended, _JobUI = 9, 4
_LIMIT_ACTIVE_PROCESS = 0x00000008
_LIMIT_PROCESS_MEMORY = 0x00000100
_LIMIT_JOB_MEMORY = 0x00000200
_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_UILIMIT_ALL = 0x000000FF
_LOGON_WITH_PROFILE = 0x00000001
_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_CREATE_NO_WINDOW = 0x08000000
_WAIT_TIMEOUT = 0x00000102


class SandboxUnavailable(RuntimeError):
    """サンドボックスが未セットアップ / 起動不能（`setup_sandbox.py` の案内に繋げる）。"""


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_sec: float


def _cred() -> str:
    """DPAPI で保存したパスワードを復号（復号できるのは保存した本人のセッションだけ）。"""
    if not _CRED_PATH.exists():
        raise SandboxUnavailable(
            f"資格情報が無い（{_CRED_PATH}）。管理者で setup_sandbox.py apply を実行してください")

    class _BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    data = _CRED_PATH.read_bytes()
    src = _BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data),
                                       ctypes.POINTER(ctypes.c_char)))
    dst = _BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        raise SandboxUnavailable(
            "資格情報を復号できない（別ユーザで apply した可能性）。apply をやり直してください")
    try:
        return ctypes.string_at(dst.pbData, dst.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def child_env(runtime_dir: Path, venv_dir: Path, tmp_dir: Path, profile: str = "") -> dict:
    """子に渡す環境変数の**ホワイトリスト**。

    `config.py` が `.env` を `os.environ` に載せるため、`env=` を省略すると子が全 API キーを
    継承する（実測）。ホワイトリスト方式なので**将来 `.env` に鍵が増えても自動的に漏れない**。
    """
    keep = ("SystemRoot", "windir", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE", "SystemDrive")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PATH"] = os.pathsep.join(str(p) for p in (
        venv_dir / "Scripts", runtime_dir / "py",
        Path(env.get("SystemRoot", r"C:\Windows")) / "system32",
        Path(env.get("SystemRoot", r"C:\Windows")),
    ))
    env["TEMP"] = env["TMP"] = str(tmp_dir)
    env["PIP_CACHE_DIR"] = str(tmp_dir / "pipcache")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if profile:
        env["USERPROFILE"] = profile
    return env


def _env_block(env: dict) -> ctypes.Array:
    """CREATE_UNICODE_ENVIRONMENT 用の環境ブロック（`K=V\\0…\\0\\0`）。"""
    s = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
    return ctypes.create_unicode_buffer(s)


def _make_job(*, mem_mb: int, max_procs: int, cpu_sec: int):
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _JOB_EXTENDED()
    b = info.BasicLimitInformation
    b.LimitFlags = (_LIMIT_ACTIVE_PROCESS | _LIMIT_PROCESS_MEMORY | _LIMIT_JOB_MEMORY
                    | _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
    b.ActiveProcessLimit = max_procs
    b.PerJobUserTimeLimit = int(cpu_sec * 10_000_000)  # 100ns 単位
    info.BasicLimitInformation = b
    info.ProcessMemoryLimit = mem_mb * 1024 * 1024
    info.JobMemoryLimit = mem_mb * 1024 * 1024
    if not k32.SetInformationJobObject(job, _JobExtended, ctypes.byref(info),
                                       ctypes.sizeof(info)):
        k32.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    ui = _JOB_UI(_UILIMIT_ALL)  # ExitWindowsEx / 他ウィンドウへの shatter を封じる
    if not k32.SetInformationJobObject(job, _JobUI, ctypes.byref(ui), ctypes.sizeof(ui)):
        k32.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def run_sandboxed(argv: list[str], *, cwd: Path, timeout_sec: float = 60.0,
                  grace_sec: float = 10.0, mem_mb: int = 2048, max_procs: int = 32,
                  runtime_dir: Path | None = None, venv_dir: Path | None = None,
                  max_output_chars: int = 8000) -> RunResult:
    """別アカウント + Job Object でコマンドを実行する（**同期**・呼び出し側が to_thread する）。

    二重タイムアウト: 内側 = runner の `subprocess.run(timeout)`、外側 = `WaitForSingleObject`
    + `TerminateJobObject`。runner 自体が固まっても外側で必ず殺せる（スレッドの自然死を保証）。
    """
    base = Path(cwd).resolve()
    runtime_dir = runtime_dir or Path(r"C:\eve_ws\runtime")
    venv_dir = venv_dir or Path(r"C:\eve_ws\venv")
    runner = runtime_dir / "runner.py"
    if not runner.exists():
        raise SandboxUnavailable(
            f"runner が無い（{runner}）。管理者で setup_sandbox.py apply を実行してください")
    py = venv_dir / "Scripts" / "python.exe"
    if not py.exists():
        py = runtime_dir / "py" / "python.exe"
    if not py.exists():
        raise SandboxUnavailable("サンドボックス用 Python が無い。setup_sandbox.py apply を実行してください")

    jobdir = base / ".eve_run" / uuid.uuid4().hex[:12]
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "job.json").write_text(json.dumps(
        {"argv": [str(a) for a in argv], "cwd": str(base), "timeout": timeout_sec,
         "max_chars": max_output_chars}, ensure_ascii=False), encoding="utf-8")

    tmp_dir = runtime_dir / "tmp"
    env = child_env(runtime_dir, venv_dir, tmp_dir)
    cmdline = f'"{py}" "{runner}" "{jobdir}"'
    si = _STARTUPINFOW()
    si.cb = ctypes.sizeof(si)
    pi = _PROCESS_INFORMATION()
    job = _make_job(mem_mb=mem_mb, max_procs=max_procs,
                    cpu_sec=int(timeout_sec + grace_sec))
    started = time.monotonic()
    try:
        ok = adv.CreateProcessWithLogonW(
            SANDBOX_USER, ".", _cred(), _LOGON_WITH_PROFILE, None,
            ctypes.create_unicode_buffer(cmdline),
            _CREATE_SUSPENDED | _CREATE_UNICODE_ENVIRONMENT | _CREATE_NO_WINDOW,
            ctypes.cast(_env_block(env), wintypes.LPVOID), str(base),
            ctypes.byref(si), ctypes.byref(pi))
        if not ok:
            err = ctypes.get_last_error()
            raise SandboxUnavailable(
                f"サンドボックスを起動できない（WinError {err}）。"
                "setup_sandbox.py check で seclogon とアカウントを確認してください")
        # **Job に入れられないなら実行しない**（安全側に倒す）
        if not k32.AssignProcessToJobObject(job, pi.hProcess):
            err = ctypes.get_last_error()
            k32.TerminateProcess(pi.hProcess, 1)
            raise SandboxUnavailable(f"Job Object に割り当てられない（WinError {err}）")
        k32.ResumeThread(pi.hThread)

        waited = k32.WaitForSingleObject(pi.hProcess, int((timeout_sec + grace_sec) * 1000))
        timed_out = waited == _WAIT_TIMEOUT
        if timed_out:
            k32.TerminateJobObject(job, 1)
            k32.WaitForSingleObject(pi.hProcess, 5000)
        code = wintypes.DWORD()
        k32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
        exit_code = int(code.value)
    finally:
        for h in (getattr(pi, "hThread", None), getattr(pi, "hProcess", None)):
            if h:
                k32.CloseHandle(h)
        k32.CloseHandle(job)  # KILL_ON_JOB_CLOSE: 残った子孫はここで全滅（孤児ゼロ）

    out = err_s = ""
    inner_timeout = False
    try:
        res = json.loads((jobdir / "result.json").read_text(encoding="utf-8"))
        out, err_s = res.get("stdout", ""), res.get("stderr", "")
        inner_timeout = bool(res.get("timed_out"))
        exit_code = int(res.get("exit_code", exit_code))
    except (OSError, ValueError, TypeError):
        if not timed_out:
            err_s = "（実行はしたが結果を取得できなかった）"
    finally:
        shutil.rmtree(jobdir, ignore_errors=True)
    return RunResult(exit_code, out, err_s, timed_out or inner_timeout,
                     time.monotonic() - started)
