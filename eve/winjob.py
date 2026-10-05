"""子プロセスを EVE と一緒に終わらせる Job Object（Windows・KILL_ON_JOB_CLOSE だけ）。

用途: EVE が自動で起動する常駐サーバ（Irodori TTS・S1）。EVE がどんな終わり方をしても
（異常終了・タスクマネージャで強制終了でも）、Job のハンドルが閉じた瞬間に OS が子と孫を落とす。

`eve/workspace/winexec.py` の Job はサンドボックス用（メモリ・プロセス数・UI の制限つき）なので
流用しない。ここは **KILL_ON_JOB_CLOSE だけ**を付ける（サーバは GPU・ウィンドウを普通に使う）。

- venv の `python.exe` はランチャで、本体の python を**子として**起動する。Popen を返した直後に
  Job へ入れても、その間にランチャが子を作っていると子が Job から漏れる → `CREATE_SUSPENDED` で
  止めたまま作り、Job に入れてから再開する（`resume_process`）。
- ctypes は `argtypes` と `restype` を必ず書く（64bit のハンドルが切り捨てられて**例外にならずに**
  壊れる。`winexec.py` の PoC の教訓）。
- Windows 以外・作成失敗時は None を返す（呼び出し側は Job なしで続ける＝サーバ側の親監視が保険）。
"""
from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobExtended = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_SUSPEND_RESUME = 0x0800

_k32 = None
_ntdll = None


def _api():
    """kernel32 / ntdll の関数型を**この モジュール専用の WinDLL** に設定して返す。"""
    global _k32, _ntdll
    if os.name != "nt":
        return None, None
    if _k32 is None:
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for fn, res, args in (
            ("CreateJobObjectW", wintypes.HANDLE, [wintypes.LPVOID, wintypes.LPCWSTR]),
            ("SetInformationJobObject", wintypes.BOOL,
             [wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD]),
            ("AssignProcessToJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.HANDLE]),
            ("TerminateJobObject", wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
            ("OpenProcess", wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
            ("CloseHandle", wintypes.BOOL, [wintypes.HANDLE]),
        ):
            f = getattr(k32, fn)
            f.restype = res
            f.argtypes = args
        nt = ctypes.WinDLL("ntdll")
        nt.NtResumeProcess.restype = ctypes.c_long
        nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
        _k32, _ntdll = k32, nt
    return _k32, _ntdll


def _structs():
    from ctypes import wintypes

    size_t = ctypes.c_size_t
    ulong_ptr = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", size_t), ("MaximumWorkingSetSize", size_t),
                    ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ulong_ptr),
                    ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", size_t), ("JobMemoryLimit", size_t),
                    ("PeakProcessMemoryUsed", size_t), ("PeakJobMemoryUsed", size_t)]

    return Extended


def create_kill_on_close_job() -> Optional[int]:
    """KILL_ON_JOB_CLOSE だけを付けた Job を作る。失敗・Windows 以外は None。"""
    k32, _ = _api()
    if k32 is None:
        return None
    job = k32.CreateJobObjectW(None, None)
    if not job:
        logger.warning("Job Object を作れない（err=%d）", ctypes.get_last_error())
        return None
    info = _structs()()
    info.BasicLimitInformation.LimitFlags = _LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, _JobExtended, ctypes.byref(info), ctypes.sizeof(info)):
        logger.warning("Job Object の設定に失敗（err=%d）", ctypes.get_last_error())
        k32.CloseHandle(job)
        return None
    return job


def assign_and_resume(job: Optional[int], pid: int) -> bool:
    """`CREATE_SUSPENDED` で作ったプロセスを Job に入れてから再開する。

    Job に入れられなくても**必ず再開する**（止まったままだとサーバが永久に起動しない）。
    戻り値 = Job に入ったか。
    """
    k32, nt = _api()
    if k32 is None:
        return False
    h = k32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        logger.warning("プロセスを開けない（pid=%d・err=%d）", pid, ctypes.get_last_error())
        return False
    try:
        ok = bool(job) and bool(k32.AssignProcessToJobObject(job, h))
        if job and not ok:
            logger.warning("Job に入れられない（pid=%d・err=%d）", pid, ctypes.get_last_error())
        status = nt.NtResumeProcess(h)
        if status != 0:
            logger.warning("プロセスを再開できない（pid=%d・NTSTATUS=0x%08x）", pid, status & 0xFFFFFFFF)
        return ok
    finally:
        k32.CloseHandle(h)


def terminate_job(job: Optional[int]) -> None:
    """Job の中のプロセス（子と孫）を全部止めて、ハンドルを閉じる。"""
    k32, _ = _api()
    if k32 is None or not job:
        return
    try:
        k32.TerminateJobObject(job, 0)
    finally:
        k32.CloseHandle(job)
