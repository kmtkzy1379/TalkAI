"""J-3 inc1 作業フォルダの決定論テスト（API 不要・実 FS は一時ディレクトリのみ）。

検証:
- 封じ込め: `..` / 絶対パス / UNC / ADS / 予約名 / 末尾ドット・空白 / NUL / 兄弟prefix / 長すぎ。
- **誤実装の回帰固定**: `abspath+startswith` が兄弟prefix を「通す」ことをテスト側に固定する
  （本実装が拒否することだけを見ると、過剰拒否に退行しても気づけない）。
- リンク: ジャンクション（`islink()` が False・`st_nlink` も 1 なので属性でしか捕まらない）と
  ハードリンク（`resolve()` を通過する）。作れない環境は **SKIP**（PASS/FAIL に環境差を混ぜない）。
- 走査: ジャンクションの中を数えない・降りない（`os.walk` は追跡してしまうので自前実装）。
- クォータ: 境界ちょうど / 超過は拒否 / **既存ファイルを消さない** / 削除で回復 / 外部変更の再同期。
- 失敗マーカ規約: 封じ込め違反は raise（registry がマーカ化）、容量超過・不在は**マーカにしない**
  （TaskAgent が自力回復できる＝要件3の生死）。
- 登録: agent_tool_schemas に出て tool_schemas（応答LLM）に出ない / outward_actions の1文目制約。
- 場所ゲート: リポジトリ配下とドライブ直下を拒否。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_workspace_phase1.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.workspace import (  # noqa: E402
    Workspace, WorkspacePathError, register_workspace_capabilities, resolve_workspace_root,
)
from eve.workspace.paths import iter_files  # noqa: E402

_passed = 0
_failed = 0
_skipped = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


def skip(name, why):
    global _skipped
    _skipped += 1
    print(f"SKIP {name}（{why}）")


def _ws(tmp: Path, *, quota=10 ** 9, write_max=200000, read_max=4000) -> Workspace:
    return Workspace(tmp, quota_bytes=quota, max_write_chars=write_max,
                     max_read_chars=read_max, max_files=20000, rescan_sec=600)


def _rejects(ws: Workspace, rel: str) -> bool:
    try:
        ws.paths.resolve(rel)
        return False
    except WorkspacePathError:
        return True
    except Exception:
        return False  # 別の例外は「意図した拒否」ではない


# --- A. 封じ込め ------------------------------------------------------------

_ATTACKS = [
    ("親へ脱出", "../../Windows/System32/drivers/etc/hosts"),
    ("親へ脱出(単純)", "../secret.txt"),
    ("絶対パス", "C:/Windows/win.ini"),
    ("device path", "//?/C:/Windows/win.ini"),
    ("UNC", "//localhost/C$/Windows/win.ini"),
    ("ADS", "note.txt:hidden"),
    ("ADS($DATA)", "note.txt::$DATA"),
    ("予約名 CON", "CON"),
    ("予約名 NUL", "NUL"),
    ("予約名(拡張子付) com1.txt", "com1.txt"),
    ("末尾ドット", "note.txt."),
    ("末尾空白", "note.txt "),
    ("NUL文字", "a\x00b.txt"),
    ("空", "   "),
]


def t_containment_rejects(ws):
    bad = [n for n, p in _ATTACKS if not _rejects(ws, p)]
    if bad:
        print("   通過してしまった:", bad)
    return not bad


def t_containment_accepts(ws):
    ok1 = ws.paths.resolve("ok.txt").name == "ok.txt"
    ok2 = ws.paths.resolve("sub/dir/a.txt").parent.name == "dir"
    ok3 = ws.paths.resolve("./sub/a.txt").parent.name == "sub"  # 先頭 ./ は無視される
    # `..` は**内側に収まる場合でも拒否**する（正規化の結果に依存した判断をしない＝
    # 「`..` を許すが結果が内側なら OK」より、コンポーネント単位で一律拒否する方が単純で安全）。
    ok4 = _rejects(ws, "sub/../b.txt")
    return ok1 and ok2 and ok3 and ok4


def t_sibling_prefix(ws, tmp: Path):
    """兄弟prefix（`<root>_evil`）。**誤実装が通すこと**も同時に固定する。"""
    evil = tmp.parent / (tmp.name + "_evil")
    naive_passes = os.path.abspath(str(evil / "x.txt")).startswith(str(tmp))
    return _rejects(ws, f"../{evil.name}/x.txt") and naive_passes


# --- B. リンク --------------------------------------------------------------

def _mklink_j(link: Path, target: Path) -> bool:
    try:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, timeout=10)
        return r.returncode == 0 and link.exists()
    except Exception:
        return False


def t_junction(ws, tmp: Path, outside: Path):
    link = tmp / "linkout"
    if not _mklink_j(link, outside):
        return None  # SKIP
    try:
        # islink/st_nlink では捕まらないことも同時に固定（属性判定が必要な根拠）
        not_caught_by_old = (not os.path.islink(link)) and os.lstat(link).st_nlink == 1
        return _rejects(ws, "linkout/loot.txt") and not_caught_by_old
    finally:
        try:
            os.rmdir(link)  # ジャンクションはリンクだけ外れる（中身を消さない）
        except OSError:
            pass


def t_junction_not_scanned(ws, tmp: Path, outside: Path):
    """走査がジャンクションの中を数えない（外部の容量を計上しない・ループしない）。"""
    link = tmp / "linkout2"
    if not _mklink_j(link, outside):
        return None
    try:
        names = [p.name for p, _ in iter_files(tmp)]
        return "loot.txt" not in names
    finally:
        try:
            os.rmdir(link)
        except OSError:
            pass


def t_hardlink(ws, tmp: Path, outside: Path):
    """ハードリンクは `resolve()` を通過するので `st_nlink` でしか捕まらない。"""
    src = outside / "loot.txt"
    link = tmp / "innocent.txt"
    try:
        os.link(src, link)
    except (OSError, NotImplementedError):
        return None
    try:
        # ハードリンクは WS 内の正規の名前なので resolve+is_relative_to は**通る**（＝これだけでは
        # 境界外の中身に届く）。st_nlink での拒否が効いていることを確認する。
        resolved_inside = (ws.paths.root / "innocent.txt").resolve().is_relative_to(ws.paths.root)
        return resolved_inside and os.stat(link).st_nlink > 1 and _rejects(ws, "innocent.txt")
    finally:
        try:
            link.unlink()
        except OSError:
            pass


# --- C. クォータ ------------------------------------------------------------

def t_quota_boundary(tmp: Path):
    ws = _ws(tmp, quota=100)
    r1 = ws.write("a.txt", "x" * 100)          # ちょうど上限 → 通る
    r2 = ws.write("b.txt", "y")                # 1バイト超過 → 断る
    kept = (tmp / "a.txt").read_text(encoding="utf-8")
    return ("書いたよ" in r1 and "空きが足りなくて" in r2
            and not r2.startswith("（")         # ★マーカにしない（自力回復の余地）
            and not (tmp / "b.txt").exists()    # 断ったのに作らない
            and kept == "x" * 100)              # ★既存ファイルを消さない


def t_quota_recovers_after_delete(tmp: Path):
    ws = _ws(tmp, quota=100)
    ws.write("a.txt", "x" * 100)
    blocked = ws.write("b.txt", "y")
    ws.delete("a.txt")
    after = ws.write("b.txt", "y")
    return "空きが足りなくて" in blocked and "書いたよ" in after


def t_quota_resyncs_before_refusing(tmp: Path):
    """外部で消された分を、拒否の直前の再スキャンで拾う（過大計上の固着を防ぐ）。"""
    ws = _ws(tmp, quota=100)
    ws.write("a.txt", "x" * 100)
    os.remove(tmp / "a.txt")            # 台帳の外で削除（inc2 の子プロセスや人手を模す）
    r = ws.write("b.txt", "z" * 50)     # 台帳上は満杯だが、再スキャンで空きが見つかるはず
    return "書いたよ" in r


# --- D. 失敗マーカ規約 ------------------------------------------------------

def t_marker_semantics(tmp: Path):
    ws = _ws(tmp)
    missing = ws.read("nope.txt")
    # 不在はマーカにしない（list_files で確認して直せる）
    ok_missing = not missing.startswith("（") and "見つからなかった" in missing
    # 封じ込め違反は raise → registry がマーカ化する
    reg = CapabilityRegistry()
    register_workspace_capabilities(reg, ws)
    out = reg.execute("read_file", {"path": "../../etc/passwd"})
    return ok_missing and out.startswith("（")


# --- E. 登録スコープ --------------------------------------------------------

def t_registration_scope(tmp: Path):
    reg = CapabilityRegistry()
    register_workspace_capabilities(reg, _ws(tmp))
    agent = {s["function"]["name"] for s in reg.agent_tool_schemas()}
    offered = {s["function"]["name"] for s in reg.tool_schemas()}
    names = {"write_file", "read_file", "list_files", "delete_file"}
    # 同期 handler＝is_blocking False（agent.py の「1step1件」ゲートに乗らない）
    not_blocking = all(not reg.is_blocking(n) for n in names)
    return names <= agent and not (names & offered) and not_blocking


def t_outward_actions_shape(tmp: Path):
    """`outward_actions()` は1文目をユーザ向け発話に載せる＝括弧も「。」も残らないこと。"""
    reg = CapabilityRegistry()
    register_workspace_capabilities(reg, _ws(tmp))
    acts = reg.outward_actions()
    has_ws = any("作業フォルダ" in a for a in acts)
    return has_ws and all("（" not in a and "。" not in a and a.strip() for a in acts)


# --- F. 場所ゲート ----------------------------------------------------------

async def t_exec_registration(tmp: Path):
    """run_program の登録契約（サンドボックス未セットアップでも検証できる範囲）。"""
    from eve.workspace import register_exec_capability
    reg = CapabilityRegistry()
    ws = _ws(tmp)
    register_workspace_capabilities(reg, ws)
    register_exec_capability(reg, ws, allow=["python", "pip"], timeout_sec=5, grace_sec=2,
                             mem_mb=256, max_procs=4, max_output_chars=1000)
    agent = {s["function"]["name"] for s in reg.agent_tool_schemas()}
    offered = {s["function"]["name"] for s in reg.tool_schemas()}
    # ファイル能力と違い**重い能力**＝is_blocking True（agent.py:150 の 1step1件に乗る）
    blocking = reg.is_blocking("run_program")
    # allowlist 外は spawn する前に拒否される（サンドボックス不要でここまで確認できる）
    denied = await reg.execute_async("run_program", {"program": "cmd", "args": ["/c", "dir"]})
    empty = await reg.execute_async("run_program", {"program": "", "args": []})
    return ("run_program" in agent and "run_program" not in offered and blocking
            and denied.startswith("（") and "許可されていない" in denied
            and empty.startswith("（"))


def t_exec_not_registered_without_flag(tmp: Path):
    """WS_EXEC を登録しなければ run_program は存在しない（フラグ分離の構造的保証）。"""
    reg = CapabilityRegistry()
    register_workspace_capabilities(reg, _ws(tmp))
    return "run_program" not in {s["function"]["name"] for s in reg.agent_tool_schemas()}


def t_location_gate():
    repo = Path(__file__).resolve().parent.parent
    inside, why1 = resolve_workspace_root(str(repo / "ws"))
    root, why2 = resolve_workspace_root("C:/")
    good, why3 = resolve_workspace_root(str(Path(tempfile.gettempdir()) / "eve_ws_test"))
    return (inside is None and "リポジトリ" in why1
            and root is None and why2
            and good is not None and not why3)


# --- G. 基本の読み書き ------------------------------------------------------

def t_roundtrip(tmp: Path):
    ws = _ws(tmp)
    ws.write("notes/memo.txt", "こんにちは")
    body = ws.read("notes/memo.txt")
    listing = ws.list()
    appended = ws.write("notes/memo.txt", "！", append=True)
    after = ws.read("notes/memo.txt")
    deleted = ws.delete("notes", recursive=True)
    return ("こんにちは" == body and "notes/" in listing and "書いたよ" in appended
            and after == "こんにちは！" and "消したよ" in deleted
            and not (tmp / "notes").exists())


def t_read_cap(tmp: Path):
    ws = _ws(tmp, read_max=50)
    ws.write("big.txt", "あ" * 500)
    out = ws.read("big.txt")
    return "50文字までにした" in out and len(out) < 200


def t_write_too_long(tmp: Path):
    ws = _ws(tmp, write_max=10)
    out = ws.write("x.txt", "y" * 11)
    return "長すぎる" in out and not out.startswith("（") and not (tmp / "x.txt").exists()


async def main():
    base = Path(tempfile.mkdtemp(prefix="eve_ws_"))
    outside = base / "outside"
    outside.mkdir()
    (outside / "loot.txt").write_text("SECRET", encoding="utf-8")
    try:
        def fresh() -> Path:
            d = Path(tempfile.mkdtemp(prefix="eve_ws_", dir=base))
            return d

        tmp = fresh()
        ws = _ws(tmp)
        check("封じ込め: 攻撃パスを全て拒否", t_containment_rejects(ws))
        check("封じ込め: 正常パスは通す", t_containment_accepts(ws))
        check("封じ込め: 兄弟prefix拒否＋誤実装が通すことの固定", t_sibling_prefix(ws, tmp))

        r = t_junction(ws, tmp, outside)
        if r is None:
            skip("リンク: ジャンクション経由を拒否", "ジャンクションを作れない環境")
        else:
            check("リンク: ジャンクション経由を拒否（islink/st_nlinkでは捕まらない）", r)
        r = t_junction_not_scanned(ws, tmp, outside)
        if r is None:
            skip("走査: ジャンクションの中を数えない", "ジャンクションを作れない環境")
        else:
            check("走査: ジャンクションの中を数えない", r)
        r = t_hardlink(ws, tmp, outside)
        if r is None:
            skip("リンク: ハードリンクを拒否", "ハードリンクを作れない環境")
        else:
            check("リンク: ハードリンクを拒否（resolveは通過する）", r)

        check("クォータ: 境界ちょうど可/超過は拒否/既存を消さない", t_quota_boundary(fresh()))
        check("クォータ: 削除で回復する", t_quota_recovers_after_delete(fresh()))
        check("クォータ: 拒否直前に再同期する", t_quota_resyncs_before_refusing(fresh()))
        check("マーカ規約: 不在は非マーカ/封じ込め違反はマーカ", t_marker_semantics(fresh()))
        check("登録: agent専用・応答LLMに出ない・is_blocking False", t_registration_scope(fresh()))
        check("登録: outward_actions の1文目が発話可能な形", t_outward_actions_shape(fresh()))
        check("実行: agent専用・is_blocking True・allowlist外は拒否", await t_exec_registration(fresh()))
        check("実行: フラグ未登録なら run_program は存在しない", t_exec_not_registered_without_flag(fresh()))
        check("場所ゲート: リポジトリ配下とドライブ直下を拒否", t_location_gate())
        check("読み書き: 作成/読取/一覧/追記/再帰削除", t_roundtrip(fresh()))
        check("読取: LLM に返す文字数を切る", t_read_cap(fresh()))
        check("書込: 長すぎる内容は非マーカで断る", t_write_too_long(fresh()))

        tmp2 = fresh()
        ws2 = _ws(tmp2)
        await ws2.ensure_ready()
        ws2.write("a.txt", "12345")
        check("初期スキャン: 使用量を数える", "5B" in ws2.usage() or "使用" in ws2.usage())
    finally:
        shutil.rmtree(base, ignore_errors=True)


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}" + (f" / SKIP {_skipped}" if _skipped else ""))
sys.exit(1 if _failed else 0)
