"""J-3 inc1 — ワークスペースのファイル I/O と容量会計。

【失敗マーカの規約（重要・要件3の生死に関わる）】
`executor.py:110` は「先頭が『（』なら Failed」で verdict を決める。したがって:
- **回復可能な失敗はマーカにしない**（ファイル不在・容量超過）。イブが `delete_file` して
  書き直す、別のパスを試す、という**自力回復の余地を残す**。`client.py:10-13` の
  「0件は正直な空振り＝マーカにしない」と同じ思想。
- **設計上の禁止と内部エラーだけマーカにする**（封じ込め違反）。これは `WorkspacePathError`
  を raise すれば `registry.execute()` が自動でマーカ化してくれる。

【容量会計（128GB 既定・書込ゲート方式）】
- 上限を超えても**削除は一切しない**。新規書込だけを断る（ユーザ裁定）。
- 台帳は int 1個・loop 所有・**永続化しない**。起動時のフルスキャンが唯一の真実源で、
  プロセス再起動＝台帳ズレの自動全治。永続台帳は「ズレたら直せない」問題を発明するだけ。
- **拒否する直前に必ず再スキャン**する。過大計上のまま固着すると、イブが自分では二度と
  書けない詰み状態になる（過小計上は次の定期スキャンで有界に戻るので非対称に扱う）。
- 既知の限界: ADS（`note.txt:hidden`）は `st_size` に出ないので**原理的に数えられない**。
  inc1 は `:` を拒否するのでイブ自身は作れないが、inc2 のサンドボックス実行では作れる。
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path

from .paths import WorkspacePaths, iter_files

logger = logging.getLogger(__name__)

_LIST_MAX = 60  # 一覧に出す最大件数（tool メッセージのトークンを有界にする）


def _human(n: int) -> str:
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


def _scan(root: Path, max_entries: int) -> tuple[int, int]:
    """フルスキャン。返り値 (合計バイト, ファイル数)。**純粋関数**＝`to_thread` に渡せる。"""
    total = count = 0
    for _, size in iter_files(root, max_entries=max_entries):
        total += size
        count += 1
    return total, count


class Workspace:
    """作業フォルダの実体。**loop 単一所有・ロックなし**（state を触るのは loop 側だけ）。"""

    def __init__(self, root: Path, *, quota_bytes: int, max_write_chars: int,
                 max_read_chars: int, max_files: int, rescan_sec: float) -> None:
        self.paths = WorkspacePaths(root)
        self._quota = int(quota_bytes)
        self._max_write = int(max_write_chars)
        self._max_read = int(max_read_chars)
        self._max_files = int(max_files)
        self._rescan_sec = float(rescan_sec)
        self._used = 0
        self._files = 0
        self._scanned_at = 0.0

    # --- ライフサイクル ----------------------------------------------------

    async def ensure_ready(self) -> None:
        """mkdir + 起動時フルスキャン（スキャンは `to_thread`＝loop を塞がない）。"""
        self.paths.root.mkdir(parents=True, exist_ok=True)
        await self._rescan()
        logger.debug("作業フォルダ初期スキャン: %s / %d件", _human(self._used), self._files)

    async def _rescan(self) -> None:
        self._used, self._files = await asyncio.to_thread(
            _scan, self.paths.root, self._max_files)
        self._scanned_at = time.monotonic()

    def usage(self) -> str:
        return f"使用 {_human(self._used)} / 上限 {_human(self._quota)}（{self._files}ファイル）"

    # --- 能力の実体（同期・sub-ms 前提）------------------------------------
    # 同期にしているのは性能でなくセマンティクスの理由: async_handler にすると
    # `registry.is_blocking()` が True になり、`agent.py:150` の「重い能力は1step1件」に
    # ファイル操作まで乗って「書く→動かす」で手数を浪費する。書込は文字数上限で担保する。

    def write(self, rel: str, content: str, append: bool = False) -> str:
        target = self.paths.resolve(rel)  # 違反は raise → registry がマーカ化
        content = content if isinstance(content, str) else str(content)
        if len(content) > self._max_write:
            return (f"内容が長すぎるので書けなかった（{len(content)}文字 / 上限{self._max_write}文字）。"
                    "分割して書いてね。")
        data = content.encode("utf-8")
        old = target.stat().st_size if target.exists() else 0
        delta = len(data) if append else len(data) - old
        if delta > 0 and self._used + delta > self._quota:
            # 拒否の直前に再スキャン（外部/過去の削除を反映しないと誤拒否が固着する）
            self._used, self._files = _scan(self.paths.root, self._max_files)
            self._scanned_at = time.monotonic()
            if self._used + delta > self._quota:
                return (f"作業フォルダの空きが足りなくて書けなかった（{self.usage()}）。"
                        "要らないファイルを消してから書いてね。")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a" if append else "w", encoding="utf-8", newline="") as f:
            f.write(content)
        # 実測サイズで確定（改行変換・エンコードで見積りとズレる）
        new = target.stat().st_size
        self._used += new - old
        if old == 0:
            self._files += 1
        return f"{self.paths.relative(target)} に書いたよ（{_human(new)}）。"

    def read(self, rel: str, max_chars: int | None = None) -> str:
        target = self.paths.resolve(rel)
        if not target.exists():
            return f"{self.paths.relative(target)} は見つからなかった。list_files で確認してみて。"
        if target.is_dir():
            return f"{self.paths.relative(target)} はフォルダだよ。list_files を使ってね。"
        cap = min(int(max_chars or self._max_read), self._max_read)
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"{self.paths.relative(target)} はテキストとして読めなかった（バイナリかも）。"
        if len(text) > cap:
            return text[:cap] + f"\n…（長いので{cap}文字までにした。全体は{len(text)}文字）"
        return text if text else "（中身は空だった）"

    def list(self, rel: str | None = None) -> str:
        target = self.paths.resolve(rel) if rel else self.paths.root
        if not target.exists():
            return f"{self.paths.relative(target)} は見つからなかった。"
        if not target.is_dir():
            return f"{self.paths.relative(target)} はファイルだよ（read_file で読める）。"
        rows: list[str] = []
        try:
            for entry in sorted(os.scandir(target), key=lambda e: (not e.is_dir(), e.name)):
                if len(rows) >= _LIST_MAX:
                    rows.append("・…（多いので省略）")
                    break
                try:
                    st = entry.stat(follow_symlinks=False)
                    if entry.is_dir(follow_symlinks=False):
                        rows.append(f"・{entry.name}/")
                    else:
                        rows.append(f"・{entry.name}（{_human(st.st_size)}）")
                except OSError:
                    continue
        except OSError as e:
            return f"一覧を取れなかった: {e}"
        head = self.paths.relative(target) or "作業フォルダ"
        body = "\n".join(rows) if rows else "（空っぽ）"
        return f"{head} の中身:\n{body}\n{self.usage()}"

    def delete(self, rel: str, recursive: bool = False) -> str:
        target = self.paths.resolve(rel)
        if target == self.paths.root:
            raise ValueError("作業フォルダそのものは消せません")  # registry がマーカ化
        if not target.exists():
            return f"{self.paths.relative(target)} は元々無かったよ。"
        name = self.paths.relative(target)
        if target.is_dir():
            if not recursive:
                return f"{name} はフォルダだよ。中ごと消すなら recursive=true を付けてね。"
            freed, freed_n = _scan(target, self._max_files)
            shutil.rmtree(target)
            self._used = max(0, self._used - freed)
            self._files = max(0, self._files - freed_n)
            return f"{name} をフォルダごと消したよ（{_human(freed)} 空いた）。"
        freed = target.stat().st_size
        target.unlink()
        self._used = max(0, self._used - freed)
        self._files = max(0, self._files - 1)
        return f"{name} を消したよ（{_human(freed)} 空いた）。"

    # --- 定期の再同期（外部要因のドリフト補正）----------------------------

    async def maybe_rescan(self) -> None:
        """一定時間が経っていたら再スキャン（inc2 の子プロセスや人手の変更を吸収）。"""
        if time.monotonic() - self._scanned_at >= self._rescan_sec:
            await self._rescan()
