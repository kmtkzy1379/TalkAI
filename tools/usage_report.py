r"""API 使用量ログ（`usage.jsonl`）を集計する。**料金指標の読み口。**

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\usage_report.py                # 既定のログ全体
    & .\.venv\Scripts\python.exe tools\usage_report.py usage.jsonl --tag live
    & .\.venv\Scripts\python.exe tools\usage_report.py --recalc       # 今の価格表で計算し直す
    & .\.venv\Scripts\python.exe tools\usage_report.py --warmup-sec 90  # 起動とみなす秒数

## 短い計測から1時間を見積もるとき
**単純な ×6 はずれる。**起動（ウォームアップ）は1回きりなので、それを含めたまま掛けると
6回起動したことになる。本ツールは起動ぶんを分離して「起動1回 + 定常1時間」で出す。
逆に**過小**になる要因もある（自発発話の同内容抑制の窓が既定600秒＝10分の計測は
その効果を見ていない／RAG は育つのでプロンプトが伸びる）。幅つきで読むこと。

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


def _cost_of(r: dict, recalc: bool):
    # 件数で課金される API（X API・accuracy=fixed）はトークンが無いので当て直さない
    if not recalc or r.get("accuracy") == "fixed":
        return r.get("cost_usd")
    p = price_of(r.get("model", "?"))
    if p is None:
        return None
    return r.get("prompt_tokens", 0) * p[0] + r.get("completion_tokens", 0) * p[1]


def _steady(rows: list, recalc: bool, warmup_sec: float) -> None:
    """**起動ぶんを除いた**定常レートを出す。

    短い計測を単純に×6すると2方向にずれる:
      1. **起動は1回きり**（ウォームアップの STT/LLM/TTS と Ruri ロード）。含めたまま
         掛けると、6回起動したことになって**過大**になる。ここではそれを除く。
      2. 逆に**過小**になる要因もある: 自発発話の同内容抑制の窓は既定 600秒
         （`SPEECH_DUP_WINDOW_SEC`）。10分の計測はちょうどその境目に座るので、
         それ以降に効いてくる抑制（＝発話が減る＝安くなる）を見ていない。
         また RAG は育つのでプロンプトが少しずつ伸びる。
    したがって定常レートも**上下の幅つき**で読むこと。桁を掴む用途には十分。
    """
    stamps = []
    for r in rows:
        try:
            stamps.append((datetime.fromisoformat(r["ts"]), r))
        except (KeyError, ValueError):
            continue
    if len(stamps) < 3:
        return
    stamps.sort(key=lambda x: x[0])
    t0 = stamps[0][0]
    tail = [(t, r) for t, r in stamps if (t - t0).total_seconds() >= warmup_sec]
    if len(tail) < 2:
        print(f"　定常レート     : 起動後（{warmup_sec:.0f}秒以降）の呼び出しが少なく計算できません")
        return
    span = (tail[-1][0] - tail[0][0]).total_seconds() / 3600.0
    if span <= 0:
        return
    usd = sum(c for c in (_cost_of(r, recalc) for _t, r in tail) if c is not None)
    warm = len(stamps) - len(tail)
    warm_usd = sum(c for c in (_cost_of(r, recalc)
                               for t, r in stamps if (t - t0).total_seconds() < warmup_sec)
                   if c is not None)
    print(f"　定常レート     : {usd / span:.4f} USD/時"
          f"（最初の{warmup_sec:.0f}秒＝起動 {warm} 呼び出し・{warm_usd:.5f} USD を除いた {span * 60:.1f} 分から）")
    print(f"　起動1回ぶん    : {warm_usd:.5f} USD（何回起動しても1回だけかかる）")
    print(f"　→ 1時間の目安  : {warm_usd + usd / span:.4f} USD（起動1回 + 定常1時間）")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    recalc = "--recalc" in args
    args = [a for a in args if a != "--recalc"]
    warmup_sec = 60.0
    if "--warmup-sec" in args:
        i = args.index("--warmup-sec")
        try:
            warmup_sec = float(args[i + 1])
        except (IndexError, ValueError):
            pass
        args = args[:i] + args[i + 2:]
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
        if recalc and r.get("accuracy") != "fixed":  # X API 等の件数課金は当て直さない（`_cost_of`）
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
        print(f"1時間あたり     : {total_usd / span_h:.4f} USD（起動ぶんを含む素の割り算）")
        _steady(rows, recalc, warmup_sec)
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
