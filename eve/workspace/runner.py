"""J-3 inc2 — サンドボックス**子側**で動く固定スタブ（`C:\\eve_ws\\runtime\\runner.py`）。

`CreateProcessWithLogonW` はハンドルを継承しないため、パイプで子の出力を取れない。
そこでファイル受け渡しにする:

    親: job.json を書く → runner を起動
    子: job.json を読む → subprocess.run(capture_output) → result.json を書く
    親: プロセス終了を待つ → result.json を読む

これでハンドル継承への依存が構造的に消え、内側タイムアウトも素直に書ける。

【重要】このファイルは `runtime\\` に置かれ、サンドボックスからは **RX（読取・実行のみ）**。
子が自分で書き換えて挙動を変えることはできない。ロジックはここに固定し、
可変部（実行するコマンド）は job.json だけで受け取る。

stdlib のみを使う（サンドボックス用 Python は素の環境なので依存を持たせない）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    jobdir = Path(sys.argv[1])
    try:
        job = json.loads((jobdir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 3

    argv = [str(a) for a in job.get("argv") or []]
    cwd = job.get("cwd") or str(jobdir)
    timeout = float(job.get("timeout") or 60.0)
    max_chars = int(job.get("max_chars") or 8000)
    if not argv:
        return 4

    out = err = ""
    timed_out = False
    code = -1
    try:
        # shell は使わない（argv 配列で渡す）。stdin は塞ぐ＝対話待ちで固まらせない。
        r = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        code = r.returncode
        out = (r.stdout or b"").decode("utf-8", errors="replace")
        err = (r.stderr or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = (e.stdout or b"").decode("utf-8", errors="replace") if e.stdout else ""
        err = (e.stderr or b"").decode("utf-8", errors="replace") if e.stderr else ""
    except FileNotFoundError:
        err = f"コマンドが見つからない: {argv[0]}"
        code = 127
    except OSError as e:
        err = f"起動できなかった: {type(e).__name__}: {e}"
        code = 126

    def cap(s: str) -> str:
        # **戻り値の時点で切り詰める**（ログだけ切っても LLM 文脈とトークンは守れない）
        return s if len(s) <= max_chars else s[:max_chars] + "\n…（以下省略）"

    try:
        (jobdir / "result.json").write_text(json.dumps(
            {"exit_code": code, "stdout": cap(out), "stderr": cap(err),
             "timed_out": timed_out}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
