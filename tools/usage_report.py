r"""API 使用量ログ（`usage.jsonl`）を集計する。**料金指標の読み口。**

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\usage_report.py                # 既定のログ全体
    & .\.venv\Scripts\python.exe tools\usage_report.py usage.jsonl --tag live
    & .\.venv\Scripts\python.exe tools\usage_report.py --recalc       # 今の価格表で計算し直す

## 何が分かるか
- **役割ごと**の回数・トークン・費用（どの機能が高いのか）
- **1時間あたり**の概算（放置運用のコスト感）
- **1応答あたり**の概算（会話の単価）

## 正直に出すこと
- `approx`（ストリーミングで数え直した分）の割合。応答は全部ストリーミングなので
  ここが大半になる。桁は合うが、請求書とぴったりにはならない。
- 価格を引けなかった呼び出し数。**0 円として混ぜない。**
- `--recalc` は保存済みのトークン数に**今の**価格表を当て直す（litellm 更新後や
  `USAGE_PRICE_OVERRIDE` を足した後に使う）。トークンを残しているからできる。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.usage import price_of  # noqa: E402


def load(path: Path, tag: str = "") -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # 書き込み途中の行は捨てる（次回に整う）
            if tag and r.get("tag") != tag:
                continue
            rows.append(r)
    return rows


def main() -> int:
    args = [a for a in sys.argv[1:]]
    recalc = "--recalc" in args
    args = [a for a in args if a != "--recalc"]
    tag = ""
    if "--tag" in args:
        i = args.index("--tag")
        tag = args[i + 1] if i + 1 < len(args) else ""
        args = args[:i] + args[i + 2:]
    path = Path(args[0]) if args else Path("usage.jsonl")

    rows = load(path, tag)
    if not rows:
        print(f"{path} に記録がありません。"
              "`.env` に USAGE_LOG_ENABLED=1 を入れて起動してください。")
        return 1

    by_role = defaultdict(lambda: {"calls": 0, "pin": 0, "pout": 0, "usd": 0.0, "unpriced": 0})
    by_model = defaultdict(lambda: {"calls": 0, "usd": 0.0})
    approx = 0
    for r in rows:
        model = r.get("model", "?")
        cost = r.get("cost_usd")
        if recalc:
            p = price_of(model)
            cost = None if p is None else (r.get("prompt_tokens", 0) * p[0]
                                           + r.get("completion_tokens", 0) * p[1])
        d = by_role[r.get("role", "?")]
        d["calls"] += 1
        d["pin"] += r.get("prompt_tokens", 0)
        d["pout"] += r.get("completion_tokens", 0)
        if cost is None:
            d["unpriced"] += 1
        else:
            d["usd"] += cost
            by_model[model]["usd"] += cost
        by_model[model]["calls"] += 1
        if r.get("accuracy") == "approx":
            approx += 1

    stamps = sorted(r["ts"] for r in rows if r.get("ts"))
    span_h = 0.0
    if len(stamps) >= 2:
        try:
            span_h = (datetime.fromisoformat(stamps[-1])
                      - datetime.fromisoformat(stamps[0])).total_seconds() / 3600.0
        except ValueError:
            span_h = 0.0

    total_usd = sum(d["usd"] for d in by_role.values())
    total_calls = sum(d["calls"] for d in by_role.values())
    unpriced = sum(d["unpriced"] for d in by_role.values())

    print(f"ログ: {path}" + (f"（tag={tag}）" if tag else "")
          + ("　※今の価格表で再計算" if recalc else ""))
    print(f"期間: {stamps[0] if stamps else '?'} 〜 {stamps[-1] if stamps else '?'}"
          f"（{span_h:.2f} 時間）\n")

    print(f"{'役割':<18}{'回数':>6}{'入力tok':>11}{'出力tok':>11}{'USD':>11}{'割合':>7}")
    for role, d in sorted(by_role.items(), key=lambda kv: -kv[1]["usd"]):
        share = (d["usd"] / total_usd * 100) if total_usd else 0.0
        print(f"{role:<18}{d['calls']:>6}{d['pin']:>11}{d['pout']:>11}"
              f"{d['usd']:>11.5f}{share:>6.1f}%")
    print(f"{'合計':<18}{total_calls:>6}"
          f"{sum(d['pin'] for d in by_role.values()):>11}"
          f"{sum(d['pout'] for d in by_role.values()):>11}{total_usd:>11.5f}")

    print(f"\nモデル別: " + " / ".join(
        f"{m}={d['usd']:.5f}USD({d['calls']}回)"
        for m, d in sorted(by_model.items(), key=lambda kv: -kv[1]["usd"])))

    print("\n--- 指標 ---")
    if span_h > 0.01:
        print(f"1時間あたり     : {total_usd / span_h:.4f} USD")
    resp = by_role.get("response", {}).get("calls", 0)
    if resp:
        print(f"1応答あたり     : {total_usd / resp:.5f} USD（応答 {resp} 回で割った全体費用）")
    print(f"概算の精度      : approx {approx}/{len(rows)} 件"
          "（ストリーミングは数え直しのため、請求書とは一致しません）")
    if unpriced:
        print(f"⚠ 価格を引けなかった呼び出し: {unpriced} 件（USD に含まれていません）")
        print("  litellm を更新するか、`.env` の USAGE_PRICE_OVERRIDE で単価を指定してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
