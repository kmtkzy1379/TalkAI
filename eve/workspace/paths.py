r"""J-3 inc1 — ワークスペースのパス封じ込め（**唯一の門**）。

ここ以外でワークスペース内のパスを組み立てない。`WorkspacePaths.resolve()` を通ったパスだけが
ファイル I/O に渡る（`store.py`）。

【なぜこの順序か】安い文字列検査で大半を落としてから I/O を伴う検査に進む。

【実測で確認した回避手口（2026-08-05・本機 Windows 11 / Python 3.13）】
- `os.path.abspath()+startswith` は **兄弟 prefix を通す**（`C:\ws` に対し `C:\ws_evil` が True）。
  よって prefix 一致ではなく `Path.resolve()` + `is_relative_to()` を使う。
- **ジャンクション**は非管理者で作成でき、`os.path.islink()` が **False**・`st_nlink` も **1**。
  つまり symlink 判定でもハードリンク判定でも捕まらず、`os.walk` は**追跡してしまう**。
  唯一の確実な検出は `st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT`。
- **ハードリンク**(`os.link`)は `resolve()+is_relative_to()` を**通過**して境界外を読み書きできる
  （リンクではなく同一実体への2つ目の正規名なので resolve では原理的に解決できない）。
  検出は `st_nlink > 1` のみ。
- **ADS**（`note.txt:hidden`）は `resolve()` を通過し、しかも `st_size` に現れないので容量会計も
  すり抜ける。`:` を含むパスを一律拒否すればドライブ指定(`C:\...`)と同時に潰せる。
- 末尾のドット/半角空白は Win32 が暗黙にトリムするため、`foo.` が `foo` に化ける。

【残る穴（塞げないものは正直に書く）】
TOCTOU: `resolve()` してから `open()` するまでの間に、外部（ユーザ・inc2 のサンドボックス子
プロセス）が同名を junction/hardlink に差し替える窓は閉じられない。Windows の Python には
`O_NOFOLLOW` 相当が無い。緩和は「使う直前に必ず resolve し直す」まで。
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

# Win32 の予約デバイス名（拡張子を付けても device に化ける: `com1.txt` も不可）。
_RESERVED = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_MAX_ABS_LEN = 240  # MAX_PATH(260) 手前で切る（中途半端な失敗を予防）


class WorkspacePathError(ValueError):
    """封じ込め違反。メッセージはそのまま能力の失敗マーカに載る（理由が伝わる形にする）。"""


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # eve/workspace/paths.py -> repo
_DEFAULT_ROOT = Path("C:/eve_ws/workspace")


def resolve_workspace_root(configured: str = "") -> tuple[Path | None, str]:
    """設定値 → 実際に使う root。不適切なら `(None, 理由)` を返す（起動時 fail-soft 用）。

    落とすのは**構造的に危険な場所だけ**（緩すぎず厳しすぎず）:
    - リポジトリ配下: `.env`（APIキー）と `*.jsonl`（会話履歴/RAG/タスク）と同じ木に置くと、
      封じ込めの穴1個で鍵と全記憶に届く。128GB を git worktree に置く問題もある。
    - ドライブ直下（`C:\\` 等）: 走査と容量計算がマシン全体に及ぶ。
    """
    root = Path(configured).expanduser() if configured.strip() else _DEFAULT_ROOT
    try:
        root = root.resolve()
    except OSError as e:
        return None, f"パスを解決できない（{e}）"
    if root == root.anchor or root.parent == root:
        return None, f"ドライブ直下は使えない（{root}）"
    if root == _REPO_ROOT or root.is_relative_to(_REPO_ROOT):
        return None, (f"リポジトリ配下は使えない（{root}）＝.env と会話履歴/RAG が同じ木にあるため。"
                      "WS_ROOT にリポジトリ外のパスを指定してください")
    return root, ""


def _has_reparse_point(p: Path) -> bool:
    """ジャンクション/シンボリックリンク/AppExecLink を属性で検出（`islink()` では不十分）。"""
    try:
        return bool(os.lstat(p).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (OSError, AttributeError):
        return False  # 存在しない（これから作る）/ 非 Windows は対象外


class WorkspacePaths:
    """ワークスペース root への封じ込め。root は起動時に1度だけ resolve して保持する。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        """相対パス文字列 → 封じ込め済みの絶対 Path。違反は `WorkspacePathError`。

        **書込/読取の直前に毎回呼ぶ**（結果を使い回さない＝TOCTOU 窓を最小化する）。
        """
        # --- 1) 文字列段階（I/O 不要・ここで大半を落とす）---
        if not isinstance(rel, str) or not rel.strip():
            raise WorkspacePathError("パスが空です")
        if "\x00" in rel:
            raise WorkspacePathError("パスに NUL 文字が含まれています")
        if ":" in rel:
            # ドライブ指定(`C:\...`)と ADS(`a.txt:hidden` / `a.txt::$DATA`)を1発で潰す。
            raise WorkspacePathError("パスに「:」は使えません（ドライブ指定・代替データストリームの禁止）")
        if rel[0] in "/\\":
            raise WorkspacePathError("パスは作業フォルダからの相対で指定してください（先頭の / \\ は不可）")

        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
        if not parts:
            raise WorkspacePathError("パスが空です")
        for part in parts:
            if part == "..":
                raise WorkspacePathError("親フォルダ（..）へは出られません")
            if part != part.rstrip(". "):
                # Win32 が暗黙トリムするので `foo.` を許すと `foo` に化ける（別名衝突の温床）。
                raise WorkspacePathError(f"名前の末尾にドットや空白は使えません: {part!r}")
            if part.split(".")[0].upper() in _RESERVED:
                raise WorkspacePathError(f"予約された名前は使えません: {part}")

        # --- 2) 正規化して封じ込め（junction はここで解決される）---
        target = (self.root / "/".join(parts)).resolve()
        if target != self.root and not target.is_relative_to(self.root):
            # `abspath+startswith` だと兄弟 prefix(`<root>_evil`)を通す。is_relative_to はパス
            # コンポーネント単位で比較するので通さない（実測で確認済）。
            raise WorkspacePathError("作業フォルダの外は触れません")
        if len(str(target)) > _MAX_ABS_LEN:
            raise WorkspacePathError("パスが長すぎます")

        # --- 3) I/O を伴う検査: reparse point は root→leaf の全段で拒否 ---
        # 途中の1段でもジャンクションなら、そこから先は実質どこへでも向けられる。
        rel_parts = target.relative_to(self.root).parts
        for i in range(len(rel_parts)):
            if _has_reparse_point(self.root.joinpath(*rel_parts[: i + 1])):
                raise WorkspacePathError("リンク（ジャンクション/シンボリックリンク）は辿れません")

        # --- 4) ハードリンク（resolve では原理的に見抜けない）---
        try:
            st = os.lstat(target)
            if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                raise WorkspacePathError("ハードリンクされたファイルは扱えません")
        except FileNotFoundError:
            pass  # これから作るファイル
        except OSError as e:
            raise WorkspacePathError(f"パスを確認できませんでした: {e}") from e
        return target

    def relative(self, target: Path) -> str:
        """表示用の相対パス（ユーザ/LLM に絶対パスを見せない）。root 自身は空文字。"""
        try:
            resolved = Path(target).resolve()
            if resolved == self.root:
                return ""  # `.` を表示すると「. の中身:」になって読みにくい（呼び出し側が補う）
            return str(resolved.relative_to(self.root)).replace("\\", "/")
        except (ValueError, OSError):
            return str(target)


def iter_files(root: Path, *, max_entries: int = 200_000):
    """ワークスペース配下のファイルを列挙する（**走査系の唯一の実装**）。

    `os.walk` / `rglob` を使わない理由（実測）: どちらも判定に `islink()` を使うため
    **ジャンクションを剪定せず追跡する**。自己参照ジャンクション1個で 1 ディレクトリが 191
    ディレクトリに膨張することを確認済み（MAX_PATH で止まるので無限ではないが病的）。
    ここでは reparse point に**降りない**ことで外部を数えない・ループしない。

    yield: (Path, st_size)。`max_entries` は暴走に対する bound（超えたら打ち切る）。
    **純粋関数**（呼び出し側の状態に触らない）＝`to_thread` に渡せる（registry.py の契約）。
    """
    seen = 0
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue  # 消えた/権限が無いディレクトリは黙って飛ばす（走査を落とさない）
        for entry in entries:
            if seen >= max_entries:
                return
            try:
                st = entry.stat(follow_symlinks=False)
                if st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    continue  # ジャンクション/symlink には降りない・数えない
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    seen += 1
                    yield Path(entry.path), st.st_size
            except OSError:
                continue
