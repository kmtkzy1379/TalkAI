r"""J-3 inc2 — サンドボックス実行の一度きりセットアップ（**setuptools のスクリプトではない**）。

イブが「自分の作業フォルダの中でプログラムを自由に実行する」ために、実行専用の低権限
ローカルアカウント `eve_sandbox` を用意する。**PC ごとに1回だけ管理者権限で実行する**。

    python setup_sandbox.py check     # 非管理者OK・読取のみ。診断して必要な操作を表示
    python setup_sandbox.py apply     # 管理者必須。実際に作る（冪等・何度でも実行可）
    python setup_sandbox.py probe     # 非管理者OK。全チェーンを実際に走らせて検証

【なぜ別アカウントなのか】
Low IL 方式（同一アカウントで整合性レベルだけ下げる）は管理者権限もパスワードも不要だが、
**読取とネットワークを一切制限できない**（実測: Low IL の子から .env が読め、外部 HTTP も通る）。
しかもファイアウォールのルールは「アカウント」単位でしか書けないため、別プリンシパルが
無いと後付けの緩和手段が原理的に存在しない。ユーザ裁定により別アカウント方式を採用。

【この設計で塞げないもの（黙って安全なふりをしない）】
- **loopback は塞げない**。Windows Firewall は 127.0.0.0/8 をフィルタしないので、
  サンドボックスから VOICEVOX(127.0.0.1:50021) 等のローカルサービスには到達する。
- **allowlist はセキュリティ境界ではない**。`python` を許した時点で任意コード実行を許して
  おり、ctypes で Win32 API も呼べる。本当の境界はアカウント + ACL + Job Object の3つだけ。
- ユーザ自身が作業フォルダの `.lnk`/`.hta` をダブルクリックする二次被害は防げない。
"""
from __future__ import annotations

import ctypes
import os
import secrets
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

SANDBOX_USER = "eve_sandbox"
WS_BASE = Path(r"C:\eve_ws")
WS_WORK = WS_BASE / "workspace"     # ★ イブから見える「作業フォルダ」（封じ込めルート）
WS_VENV = WS_BASE / "venv"          # pip install 先。**封じ込めルートの外**＝イブに壊せない
WS_RUNTIME = WS_BASE / "runtime"    # Python 本体のコピー（子から RX のみ）
WS_TMP = WS_RUNTIME / "tmp"
CRED_PATH = Path(os.environ.get("LOCALAPPDATA", "")) / "TalkAI" / "sandbox.cred"

_OK, _WARN, _NG = [], [], []


def ok(m): _OK.append(m); print(f"  [OK]   {m}")
def warn(m): _WARN.append(m); print(f"  [警告] {m}")
def ng(m): _NG.append(m); print(f"  [NG]   {m}")
def head(m): print(f"\n=== {m} " + "=" * max(0, 60 - len(m)))


def run(args, **kw) -> subprocess.CompletedProcess:
    """外部コマンド実行。**実行ファイルが無くても落とさない**。

    Windows 11 24H2 以降では `wmic` が削除されているなど、環境によって存在しないツールがある。
    1つ欠けただけでセットアップ全体が例外で止まると、他の PC へ持っていったときに原因が
    分かりにくい形で失敗する（実際に wmic で踏んだ）。fail-soft にして呼び出し側が判断する。
    """
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", **kw)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 9009, "", f"not found: {args[0]}")


def ps(script: str) -> subprocess.CompletedProcess:
    """PowerShell 経由（`wmic` 等の削除されたツールに依存しないため）。"""
    return run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# --- DPAPI（パスワードを平文で置かない）------------------------------------
# CreateProcessWithLogonW は平文パスワードを要求するので at-rest だけでも守る。
# per-user スコープ＝復号できるのは tester のログオンセッションだけ。eve_sandbox が
# ファイルを入手しても無価値になる（プロファイル配下なのでそもそも到達不可だが二重に）。

class _BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(fn, data: bytes) -> bytes:
    src = _BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data),
                                       ctypes.POINTER(ctypes.c_char)))
    dst = _BLOB()
    if not fn(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def cred_save(password: str) -> None:
    CRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    blob = _dpapi(ctypes.windll.crypt32.CryptProtectData, password.encode("utf-8"))
    CRED_PATH.write_bytes(blob)
    # 継承を切って tester だけに絞る（プロファイル配下だが明示しておく）
    run(["icacls", str(CRED_PATH), "/inheritance:r", "/grant", f"{os.environ['USERNAME']}:F"])


def cred_load() -> str | None:
    if not CRED_PATH.exists():
        return None
    try:
        return _dpapi(ctypes.windll.crypt32.CryptUnprotectData,
                      CRED_PATH.read_bytes()).decode("utf-8")
    except Exception:
        return None


# --- 診断 -------------------------------------------------------------------

def find_machine_python() -> Path | None:
    """eve_sandbox から読める場所にある Python を探す。

    既存 venv は使えない: `pyvenv.cfg` の home がユーザプロファイル内の per-user install を
    指しており、`C:\\Users\\<user>` には BUILTIN\\Users の ACE が無い（実測）ため
    eve_sandbox からは到達不可能。しかも venv 自体は stdlib を持たない。
    """
    if (WS_RUNTIME / "py" / "python.exe").exists():
        return WS_RUNTIME / "py" / "python.exe"
    for base in (r"C:\Program Files\Python313", r"C:\Python313",
                 r"C:\Program Files\Python312", r"C:\Python312"):
        p = Path(base) / "python.exe"
        if p.exists():
            return p
    return None


def current_python_home() -> Path:
    """今動いている Python の**本体**（venv ではなく base）のディレクトリ。"""
    return Path(getattr(sys, "base_prefix", sys.prefix))


def check() -> int:
    head("1. 実行環境")
    print(f"  Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  base_prefix : {current_python_home()}")
    print(f"  管理者権限  : {'あり' if is_admin() else 'なし'}")
    if os.name != "nt":
        ng("Windows 専用です"); return 2
    ok("Windows で動作中")

    head("2. サンドボックスアカウント")
    r = run(["net", "user", SANDBOX_USER])
    if r.returncode == 0:
        ok(f"{SANDBOX_USER} は作成済み")
        g = run(["net", "user", SANDBOX_USER])
        if "Administrators" in g.stdout or "Administrators" in g.stderr:
            ng(f"{SANDBOX_USER} が Administrators に所属している（apply で外します）")
    else:
        warn(f"{SANDBOX_USER} は未作成 → apply で作成します")
    print(f"  資格情報    : {'保存済み' if cred_load() else '未保存'}  ({CRED_PATH})")

    head("3. 作業フォルダ")
    for p in (WS_BASE, WS_WORK, WS_VENV, WS_RUNTIME):
        print(f"  {'あり' if p.exists() else '無し'}  {p}")

    head("4. seclogon サービス（CreateProcessWithLogonW に必須）")
    r = run(["sc", "qc", "seclogon"])
    auto = "AUTO_START" in r.stdout
    print("  " + "\n  ".join(l.strip() for l in r.stdout.splitlines()
                             if "START_TYPE" in l or "STATE" in l))
    if auto:
        ok("自動起動になっている")
    else:
        warn("自動起動ではない（停止されると実行機能が全滅する）→ apply で Automatic にします")

    head("5. 子プロセス用 Python")
    mp = find_machine_python()
    if mp:
        ok(f"使える Python がある: {mp}")
    else:
        warn(f"eve_sandbox から読める Python が無い → apply で {WS_RUNTIME / 'py'} にコピーします")

    head("6. 横移動路（新規アカウントが書き換えられる既存フォルダ）")
    risky = scan_lateral_paths()
    if risky:
        for p in risky:
            warn(f"書込可能: {p}")
        print(f"\n  → apply で {SANDBOX_USER} に対する明示 Deny を付けます")
        print("     （AuthenticatedUsers ごと消すと、あなた自身の非昇格プロセスが")
        print("       書けなくなるため Deny で封じます）")
    else:
        ok("目立った横移動路は見つからない")

    head("結果")
    print(f"  OK {len(_OK)} / 警告 {len(_WARN)} / NG {len(_NG)}")
    if not is_admin():
        print("\n  次の手順:")
        print("  1) 管理者 PowerShell を開く")
        print(f"  2) cd {Path(__file__).parent}")
        print(f"  3) {sys.executable} setup_sandbox.py apply")
        print(f"  4) 完了後、通常の PowerShell で: {sys.executable} setup_sandbox.py probe")
        return 2
    return 1 if _NG else 0


def scan_lateral_paths() -> list[Path]:
    """固定ドライブ直下で、Authenticated Users 等に書込を与えているフォルダを探す。

    `C:\\` の既定 ACE は `AuthenticatedUsers:(OI)(CI)(IO)(M)` なので、C:\\ 直下に作られた
    フォルダは**新規アカウントから書き換え可能**（実測）。eve_sandbox は Users グループに
    入れる必要がある（CreateProcessWithLogonW が対話ログオン権を要求する）ため、
    この経路は「アカウントを使える状態にすること」の副作用として必ず開く。
    """
    risky: list[Path] = []
    principals = ("Authenticated Users", "Everyone", "BUILTIN\\Users", "AUTHENTICATED USERS")
    try:
        entries = list(os.scandir("C:\\"))
    except OSError:
        return risky
    for e in entries:
        if not e.is_dir(follow_symlinks=False):
            continue
        p = Path(e.path)
        if p in (WS_BASE,) or p.name.lower() in ("users", "windows", "program files",
                                                 "program files (x86)", "programdata",
                                                 "$recycle.bin", "system volume information"):
            continue
        r = run(["icacls", str(p)])
        for line in r.stdout.splitlines():
            if any(pr in line for pr in principals) and ("(M)" in line or "(F)" in line
                                                         or "(W)" in line):
                risky.append(p)
                break
    return risky


# --- 適用 -------------------------------------------------------------------

def apply() -> int:
    if not is_admin():
        print("管理者権限が必要です。管理者 PowerShell で実行してください。\n")
        return check()

    # 昇格で別の管理者アカウントに切り替わっていないか（DPAPI blob は per-user のため、
    # 別ユーザで保存するとイブ本体＝tester が復号できず、原因が分かりにくい形で壊れる）
    interactive = os.environ.get("USERNAME", "")
    head(f"0. 前提（実行ユーザ: {interactive}）")
    if not interactive:
        ng("USERNAME が取れない"); return 1
    ok(f"{interactive} として実行中（この人の DPAPI 鍵で資格情報を保存します）")
    print("  ※ イブ本体を動かすユーザと同じでないと、起動時に資格情報を復号できません")

    head("1. サンドボックスアカウント")
    exists = run(["net", "user", SANDBOX_USER]).returncode == 0
    password = secrets.token_urlsafe(24)
    if exists:
        r = run(["net", "user", SANDBOX_USER, password])
        ok(f"{SANDBOX_USER} は既存 → パスワードを再設定（壊れた資格情報からの自己修復）") \
            if r.returncode == 0 else ng(f"パスワード再設定に失敗: {r.stderr.strip()[:120]}")
    else:
        r = run(["net", "user", SANDBOX_USER, password, "/add",
                 "/comment:AI command execution sandbox (TalkAI)", "/expires:never"])
        ok(f"{SANDBOX_USER} を作成") if r.returncode == 0 else \
            ng(f"作成に失敗: {r.stderr.strip()[:160]}")
    if _NG:
        return 1
    # wmic は Windows 11 24H2 以降で削除されているので PowerShell を使う
    ps(f"Set-LocalUser -Name '{SANDBOX_USER}' -PasswordNeverExpires $true")
    # 管理者グループから外す（万一入っていた場合の自己修復）
    if run(["net", "localgroup", "Administrators", SANDBOX_USER, "/delete"]).returncode == 0:
        warn("Administrators から削除した")
    # Users に入れる: CreateProcessWithLogonW は「対話ログオンを許可されたアカウント」を要求する。
    # ここが横移動路（C:\ 直下の継承 ACE）を開く原因でもあるので、下の 4. で塞ぎ直す。
    run(["net", "localgroup", "Users", SANDBOX_USER, "/add"])
    ok("Users グループに追加（対話ログオン権のため必須）")
    # ログイン画面に出さない
    run(["reg", "add",
         r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList",
         "/v", SANDBOX_USER, "/t", "REG_DWORD", "/d", "0", "/f"])
    ok("ログイン画面から隠した")
    cred_save(password)
    ok(f"パスワードを DPAPI で保存: {CRED_PATH}")

    head("2. 作業フォルダと ACL")
    for p in (WS_BASE, WS_WORK, WS_VENV, WS_RUNTIME, WS_TMP):
        p.mkdir(parents=True, exist_ok=True)
    # 継承を切る（C:\ 直下は AuthenticatedUsers に Modify を継承させるため）
    run(["icacls", str(WS_BASE), "/inheritance:r"])
    run(["icacls", str(WS_BASE), "/grant", "*S-1-5-18:(OI)(CI)F"])          # SYSTEM
    run(["icacls", str(WS_BASE), "/grant", "*S-1-5-32-544:(OI)(CI)F"])      # Administrators
    run(["icacls", str(WS_BASE), "/grant", f"{interactive}:(OI)(CI)F"])
    run(["icacls", str(WS_BASE), "/grant", f"{SANDBOX_USER}:(RX)"])         # 継承なし＝通過のみ
    for p, perm in ((WS_WORK, "(OI)(CI)M"), (WS_VENV, "(OI)(CI)M"),
                    (WS_RUNTIME, "(OI)(CI)RX"), (WS_TMP, "(OI)(CI)M")):
        run(["icacls", str(p), "/grant", f"{SANDBOX_USER}:{perm}"])
    ok(f"{WS_WORK} = 読み書き自由 / {WS_RUNTIME} = 読取のみ")
    print(f"     venv と runtime は封じ込めルートの外＝イブのファイル能力からは触れません")

    head("3. seclogon サービス")
    run(["sc", "config", "seclogon", "start=", "auto"])
    run(["sc", "start", "seclogon"])
    ok("Automatic に設定して起動")

    head("4. 横移動路を塞ぐ")
    risky = scan_lateral_paths()
    if not risky:
        ok("対象なし")
    for p in risky:
        r = run(["icacls", str(p), "/deny", f"{SANDBOX_USER}:(OI)(CI)(W,DE,DC)"])
        ok(f"Deny を付与: {p}") if r.returncode == 0 else warn(f"Deny 失敗: {p}")

    head("5. 子プロセス用 Python")
    dst = WS_RUNTIME / "py"
    if (dst / "python.exe").exists():
        ok(f"既にある: {dst}")
    else:
        src = current_python_home()
        print(f"  {src}\n    -> {dst} にコピー中…（数十秒）")
        r = run(["robocopy", str(src), str(dst), "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1"])
        if (dst / "python.exe").exists():
            ok("コピー完了")
            run(["icacls", str(dst), "/grant", f"{SANDBOX_USER}:(OI)(CI)RX"])
        else:
            ng(f"コピー失敗（robocopy rc={r.returncode}）")

    head("6. サンドボックス用 venv")
    if (WS_VENV / "Scripts" / "python.exe").exists():
        ok(f"既にある: {WS_VENV}")
    else:
        py = find_machine_python()
        if py:
            r = run([str(py), "-m", "venv", str(WS_VENV)])
            ok("作成完了") if (WS_VENV / "Scripts" / "python.exe").exists() else \
                warn(f"作成失敗: {r.stderr.strip()[:160]}")
            run(["icacls", str(WS_VENV), "/grant", f"{SANDBOX_USER}:(OI)(CI)M"])
        else:
            warn("Python が見つからず venv を作れない")

    head("7. runner スタブの配置")
    # 子側で動く固定ロジック。runtime\ は eve_sandbox から RX のみ＝子が書き換えられない。
    src_runner = Path(__file__).parent / "eve" / "workspace" / "runner.py"
    dst_runner = WS_RUNTIME / "runner.py"
    if src_runner.exists():
        dst_runner.write_bytes(src_runner.read_bytes())
        run(["icacls", str(dst_runner), "/grant", f"{SANDBOX_USER}:(RX)"])
        ok(f"配置: {dst_runner}（子からは読取・実行のみ）")
    else:
        ng(f"runner.py が見つからない: {src_runner}")

    head("8. ファイアウォール（LAN 向けのみ遮断）")
    # アウトバウンド全遮断は採らない: pip / git が全部死んで要件2を壊す。
    # LAN 向けだけ塞げばデメリットはほぼゼロで NAS・ルータ管理画面への到達を断てる。
    sid = ps(f"(Get-LocalUser '{SANDBOX_USER}').SID.Value").stdout.strip()
    if sid.startswith("S-1-"):
        # 冪等: 同名ルールがあれば消してから作り直す
        ps("Remove-NetFirewallRule -DisplayName 'TalkAI sandbox LAN block' "
           "-ErrorAction SilentlyContinue")
        r = ps("New-NetFirewallRule -DisplayName 'TalkAI sandbox LAN block' "
               "-Direction Outbound -Action Block -Enabled True -Profile Any "
               "-RemoteAddress 10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16 "
               f"-LocalUser 'D:(D;;CC;;;{sid})'")
        ok("LAN 向けアウトバウンドを遮断") if r.returncode == 0 else \
            warn(f"ルール作成に失敗（続行可）: {r.stderr.strip()[:120]}")
    else:
        warn("SID を取得できずファイアウォール設定をスキップ")
    print("  ※ loopback(127.0.0.1) は Windows Firewall では塞げません。")
    print("     サンドボックスから VOICEVOX 等のローカルサービスには到達します。")

    head("完了")
    print(f"  OK {len(_OK)} / 警告 {len(_WARN)} / NG {len(_NG)}")
    print(f"\n  次: 通常の（管理者でない）PowerShell で検証してください")
    print(f"      {sys.executable} setup_sandbox.py probe")
    return 1 if _NG else 0


# --- 検証 -------------------------------------------------------------------

PROBE_CODE = r"""
import ctypes, os, pathlib, traceback
from ctypes import wintypes

def emit(k, v):
    print("PROBE_%s=%s" % (k, v), flush=True)

try:
    # 実行アカウントは**環境変数ではなくトークン**で見る（USERNAME は渡していないし、
    # 渡したとしても偽装できる値なので隔離の証拠にならない）。
    buf = ctypes.create_unicode_buffer(257)
    size = wintypes.DWORD(257)
    ctypes.windll.advapi32.GetUserNameW(buf, ctypes.byref(size))
    emit("USER", buf.value)
    emit("HAS_APIKEY", any(k.endswith("_API_KEY") for k in os.environ))
    # ★ 各チェックを独立して囲う。`Path.exists()` はアクセス拒否時に False ではなく
    #   **PermissionError を送出する**ため、囲わないと以降の検証が丸ごと巻き込まれる
    #   （初回 probe で実際にそうなった）。しかもその例外こそが「到達できない」証拠。
    #   exists() ではなく実際に**開いて読む**＝知りたいこと（秘密を読めるか）を直接試す。
    try:
        with open(r"{env}", "rb") as f:
            f.read(1)
        emit("ENV_READABLE", True)
    except Exception as e:
        emit("ENV_READABLE", "False:" + type(e).__name__)
    try:
        pathlib.Path(r"{work}\_probe.txt").write_text("ok", encoding="utf-8")
        emit("WRITE_WORK", True)
    except Exception as e:
        emit("WRITE_WORK", "False:" + type(e).__name__)
    try:
        pathlib.Path(r"{repo}\_probe_escape.txt").write_text("x", encoding="utf-8")
        emit("WRITE_REPO", True)
    except Exception as e:
        emit("WRITE_REPO", "False:" + type(e).__name__)
except Exception:
    traceback.print_exc()
    emit("FATAL", "1")
"""


def probe() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from eve.workspace.winexec import SandboxUnavailable, run_sandboxed  # noqa: E402

    head("サンドボックス実行チェーンの検証")
    repo = Path(__file__).parent
    code = PROBE_CODE.format(env=repo / ".env", work=WS_WORK, repo=repo)
    py = WS_VENV / "Scripts" / "python.exe"
    if not py.exists():
        py = find_machine_python() or Path(sys.executable)
    try:
        res = run_sandboxed([str(py), "-c", code], cwd=WS_WORK, timeout_sec=90)
    except SandboxUnavailable as e:
        ng(f"サンドボックスを起動できない: {e}")
        print("\n  管理者 PowerShell で apply を実行しましたか？")
        return 1
    out = res.stdout
    print("  --- 子プロセスの stdout ---")
    for line in out.splitlines() or ["（空）"]:
        print("  " + line)
    if res.stderr.strip():
        # **stderr を必ず出す**。初回 probe はここが見えず、子が途中で落ちた理由が
        # 分からなかった（診断できない検証は無意味）。
        print("  --- 子プロセスの stderr ---")
        for line in res.stderr.splitlines():
            print("  " + line)
    print(f"  exit={res.exit_code} timed_out={res.timed_out} 所要={res.duration_sec:.1f}s")

    def val(key):
        for line in out.splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
        return None

    head("判定")
    (ok if (val("PROBE_USER") or "").lower() == SANDBOX_USER else ng)(
        f"別アカウントで実行された（トークン上のユーザ={val('PROBE_USER')}）")
    (ok if val("PROBE_HAS_APIKEY") == "False" else ng)(
        "API キーが子プロセスに渡っていない（env ホワイトリスト）")
    env_r = val("PROBE_ENV_READABLE") or ""
    (ok if env_r.startswith("False") else ng)(f".env に到達できない（{env_r}）")
    (ok if val("PROBE_WRITE_WORK") == "True" else ng)(
        f"作業フォルダに書ける（{val('PROBE_WRITE_WORK')}）")
    rep_w = val("PROBE_WRITE_REPO") or ""
    (ok if rep_w.startswith("False") else ng)(f"リポジトリには書けない（{rep_w}）")
    try:
        (WS_WORK / "_probe.txt").unlink(missing_ok=True)
        (repo / "_probe_escape.txt").unlink(missing_ok=True)
    except OSError:
        pass
    print(f"\n  OK {len(_OK)} / 警告 {len(_WARN)} / NG {len(_NG)}")
    return 1 if _NG else 0


def main() -> int:
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "check").lower()
    if cmd not in ("check", "apply", "probe"):
        print(__doc__)
        return 2
    print(f"TalkAI サンドボックス セットアップ — {cmd}")
    return {"check": check, "apply": apply, "probe": probe}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
