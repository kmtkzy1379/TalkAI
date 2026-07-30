r"""複数回の E2E アーティファクトを横断集計する（平均/中央値・判定単位の指標）。

CLAUDE.md 進め方7: 1セッションの「件数」ではノイズに埋もれるので、複数ランを束ねて
判定回数を母数にする。ラン毎の生ログは各 E2E_ART に残っているのでここでは読むだけ。

実行:
  $env:PYTHONIOENCODING="utf-8"
  venv\Scripts\python.exe tools\e2e_aggregate.py e2e_logs\rep_*
"""
from __future__ import annotations

import glob
import io
import json
import os
import statistics as st
import sys


def _load(d: str):
    tl, sm = [], {}
    p = os.path.join(d, "timeline.jsonl")
    if os.path.exists(p):
        tl = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    p = os.path.join(d, "summary.json")
    if os.path.exists(p):
        try:
            sm = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            sm = {}
    return tl, sm


def _fmt(vals, unit="s"):
    if not vals:
        return "n=0"
    return (f"n={len(vals)} 中央={st.median(vals):.2f}{unit} 平均={sum(vals)/len(vals):.2f}{unit} "
            f"最小={min(vals):.2f} 最大={max(vals):.2f}")


def main(dirs):
    runs = [(d, *_load(d)) for d in dirs]
    print(f"=== 集計対象 {len(runs)} ラン ===")
    for d, tl, sm in runs:
        print(f"  {os.path.basename(d)}: events={len(tl)}")

    # --- 発話速度（first_audio_sec）: 種別ごと ---
    print("\n■ 発話までの時間（first_audio_sec）")
    by_kind: dict[str, list] = {}
    allv: list = []
    for _d, _tl, sm in runs:
        for r in sm.get("latencies", []):
            v = r.get("first_audio_sec")
            if isinstance(v, (int, float)):
                by_kind.setdefault(r.get("kind", "?"), []).append(v)
                allv.append(v)
    for k, v in sorted(by_kind.items()):
        print(f"  {k:20} {_fmt(v)}")
    print(f"  {'全体':20} {_fmt(allv)}")
    if allv:
        over3 = sum(1 for v in allv if v > 3.0)
        over5 = sum(1 for v in allv if v > 5.0)
        print(f"  3秒超 {over3}/{len(allv)} ({100*over3/len(allv):.0f}%) / "
              f"5秒超 {over5}/{len(allv)} ({100*over5/len(allv):.0f}%)")

    # --- 判定・自発発話・能力呼出の回数 ---
    print("\n■ 判定単位の指標（ラン毎）")
    rows = []
    for d, tl, sm in runs:
        dec = sm.get("decisions") or []
        spoke = sum(1 for x in dec if x.get("speak"))
        autos = sum(1 for r in tl if r.get("tag") == "▶turn" and "AUTONOMOUS" in str(r.get("msg")))
        caps = sm.get("cap_calls") or []
        searches = sm.get("search_calls") or []
        rows.append((os.path.basename(d), len(dec), spoke, autos, len(caps), len(searches)))
    print(f"  {'run':22}{'判定':>6}{'speak':>7}{'自発':>6}{'能力':>6}{'検索':>6}")
    for r in rows:
        print(f"  {r[0][:22]:22}{r[1]:>6}{r[2]:>7}{r[3]:>6}{r[4]:>6}{r[5]:>6}")
    if rows:
        for i, name in ((1, "判定回数"), (2, "speak数"), (3, "自発発話"), (4, "能力呼出"), (5, "検索")):
            v = [float(r[i]) for r in rows]
            print(f"  {name:10} 中央={st.median(v):.1f} 平均={sum(v)/len(v):.1f}")
        sp = [r[2] / r[1] for r in rows if r[1]]
        if sp:
            print(f"  speak率     中央={st.median(sp):.3f} 平均={sum(sp)/len(sp):.3f}（母数=判定回数）")

    # --- ✅/❌ の判定行を横断集計 ---
    print("\n■ シナリオ判定（✅/❌ の出現）")
    verdict: dict[str, list] = {}
    for _d, tl, _sm in runs:
        for r in tl:
            if r.get("tag") in ("✅", "❌", "🎯", "⚠"):
                key = str(r.get("msg", "")).split(":")[0][:44]
                verdict.setdefault(key, []).append(r.get("tag"))
    for k, v in sorted(verdict.items()):
        ok = sum(1 for x in v if x in ("✅", "🎯"))
        print(f"  {'✅' if ok == len(v) else '❌'} {k:46} {ok}/{len(v)}")

    # --- 発話テキスト（自然さの目視用） ---
    print("\n■ 自発発話の全文（重複・単調さの目視）")
    for d, tl, _sm in runs:
        autos = [str(r.get("msg", "")) for r in tl
                 if r.get("tag") == "🤖" and "AUTONOMOUS" in str(r.get("msg"))]
        print(f"  -- {os.path.basename(d)} ({len(autos)}件)")
        for a in autos:
            print(f"     {a[:118]}")


if __name__ == "__main__":
    pats = sys.argv[1:] or ["e2e_logs/rep_*"]
    dirs = sorted({d for p in pats for d in glob.glob(p) if os.path.isdir(d)})
    if not dirs:
        print("対象ディレクトリなし")
        sys.exit(1)
    main(dirs)
