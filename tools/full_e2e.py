r"""通しE2E ランナー — 23項目 + 挑戦3項目を、実起動と同じ状態で通しで回す。

## 進め方（ユーザ裁定 2026-08-12）
**1周ずつ回して、そのつど Claude Code がログを読んで判定する。** まずければそこで止め、
入力文やシナリオを直してから次の周に進む。機械判定を作り込むより、読んで判断する方が
コードを複雑にせず精度も高い、という判断。よって本スクリプトは**1回の呼び出しで1周だけ**回す。

    python tools\full_e2e.py            # 第1周（A→B）
    python tools\full_e2e.py --pass 2   # 第2周（別ディレクトリに出る）

## 1周 = セッションA（連続）→ セッションB（本物の再起動）
再起動系はプロセスを落とさないと測れない。B は A が書いた記憶（会話・RAG）を
`E2E_KEEP_STATE=1` でそのまま引き継いで起動する＝種まきの模擬ではなく**本物の再起動**。
A の最後のユーザ発話を `PREV_MARK` で渡し、それが B の文脈へ混入していないかを検査する
（`D1_SEED` の固定文字列に依存しない。`D1_SEED` は `_truncate` が A の記憶を消すので併用不可）。

## 既知の限界（許容・ユーザ裁定）
E2E の入力文は**イブの応答を見て合わせていない**（台本を時間で流す）。AI 同士だとラグが出て、
その間をイブが「無言」と解釈してテストがブレるため。したがって会話の噛み合わせは
「不自然」に見えることがあるが、それはハーネス側の制約であってイブの問題ではない。

## 実行前に必要なこと
- VOICEVOX 起動 / `.env` のフラグ / `setup_sandbox.py apply` 済み
- **画面に出したくないものを閉じる**（1時間ほど画面が Gemini に送られる）
- `C:\eve_ws\workspace` は各シナリオが全消去する（要るものは退避）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"

# --- セッションA: 連続で通す（項目1-14, 19-23 と挑戦3項目）-------------------
# **この順で実行される**（`search_e2e_test.py` の main が ONLY の順に回すよう修正済）。
# 順序の意図: きれいな状態の単発から始め、複合タスクを済ませてから画面系、
# 最後に挑戦（先行シナリオと題材が被ると「初見か」を測れなくなるため）。
SESSION_A = [
    # ★挑戦の「初見」を測る2本を最初に置く（2026-08-12）。第1周の実測では、SC1b の時点で
    # 「ハラクラムス」は既に E15G と自発発話で3回出ており、初見を測れていなかった。
    "SC1",      # 挑戦: 誤前提(図鑑番号300)への訂正 + 未知語(ベトムリア)
    "SPUSH",    # ★新: 迎合の二段目（訂正した事実を押し返されても維持できるか）
    "E15A",     # (5) タスク呼出（単発・引受と実行の一致）
    "E15B",     # (10) タスク実行中の雑談
    "E15C",     # (9) タスク取消（+2s/+7s/+15s の3タイミング）
    "S1SLOT",   # (7) 複数タスク＝1件制（断り・現在タスク・取消で解放）
    "E15D",     # (7) 続き: 果たせない約束をしない / 言い直しを頼む / 同時ピーク
    "SREP",     # ★新: 既出用件（同じ依頼を2回。作り直さず答えられるか）
    "SD3",      # (11) 不可能依頼の拒否
    "E15G",     # (12) 存在しない語に正直か
    "SFAIL",    # ★新: タスクの失敗系（無いファイル / 実在しない語の調査）
    "S1",       # (6) 検索タスク呼出
    "S8",       # (8) 発話中の割り込み（barge-in）
    "SJ1",      # (19) 複雑タスク（検索+コード実行+画面表示）
    "SJ3",      # (21) 複雑タスク実行中の取消
    "SJ2",      # (20) 複雑タスク実行中の会話
    "SJ4",      # (22) 複雑タスク2件（1件制の下での挙動）
    "SVLM",     # (4) 画面認識の全場面（複合タスクの窓を閉じた後に置く）
    "SOCR",     # ★新: 作文中の文字ゲート（うるさくないかの実運用検証）
    "SSRC",     # (2)(3)(4) 自発発話の由来切り分け（古い記憶 / 直近 / 画面）
    "SC2",      # 挑戦: 自律タスク実行（PC調子悪い）
    "SC3",      # 挑戦: 意図の汲み取り（飲食店→画像を見せる）
]
# --- セッションB: A の記憶を持ったまま再起動（項目15-18）---------------------
SESSION_B = ["SRB"]


def _run(art: Path, only: list[str], extra_env: dict, timeout_sec: float) -> int:
    art.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "REAL_STATE": "1",
                "E2E_ART": str(art), "ONLY": ",".join(only)})
    env.update(extra_env)
    log = art / "stdout.txt"
    t0 = time.monotonic()
    with open(log, "w", encoding="utf-8") as fh:
        try:
            p = subprocess.run([str(PY), "tools/search_e2e_test.py"], cwd=str(ROOT), env=env,
                               stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               timeout=timeout_sec)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            # 上限を持たせる（パイプラインが死ぬと全 wait が満了して何時間も走り続けるため）
            fh.write("\n[full_e2e] タイムアウトで打ち切り\n")
            rc = 124
    print(f"    → rc={rc} / {time.monotonic() - t0:.0f}秒 / {log}")
    return rc


def _last_user_utterance(history: Path) -> str:
    """A の最後のユーザ発話（B の文脈汚染チェックの目印にする）。"""
    if not history.exists():
        return ""
    last = ""
    for ln in history.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("speaker") == "user" and r.get("text"):
            last = r["text"]
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="n", type=int, default=1, help="第何周か（出力先が変わる）")
    ap.add_argument("--stamp", default="", help="既存の出力ディレクトリに追記したい場合")
    args = ap.parse_args()

    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M")
    base = ROOT / "e2e_logs" / f"full_{stamp}"
    a, b = base / f"pass{args.n}_A", base / f"pass{args.n}_B"
    print(f"■ 通しE2E 第{args.n}周（A {len(SESSION_A)}項目 → B 再起動）")
    print(f"  出力: {base}")
    print("  ⚠ 実 API 課金 / 実画面キャプチャ / 別ウィンドウの自動開閉\n")

    print(f"  A（連続・{len(SESSION_A)}項目）...")
    _run(a, SESSION_A, {}, timeout_sec=2.5 * 3600)

    # B は A の記憶を引き継ぐ（コピー → E2E_KEEP_STATE=1 で上書きさせない）
    b.mkdir(parents=True, exist_ok=True)
    for f in ("history.jsonl", "rag_memory.jsonl", "tasks.jsonl"):
        src = a / f
        if src.exists():
            (b / f).write_bytes(src.read_bytes())
    mark = _last_user_utterance(a / "history.jsonl")[:24]
    print(f"  B（A の記憶を引き継いで再起動 / PREV_MARK={mark!r}）...")
    _run(b, SESSION_B, {"E2E_KEEP_STATE": "1", "PREV_MARK": mark}, timeout_sec=0.75 * 3600)

    print(f"\n■ 第{args.n}周 完了: {base}")
    print("  次はログを読んで判定 → 問題なければ `--pass %d --stamp %s` で次の周へ"
          % (args.n + 1, stamp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
