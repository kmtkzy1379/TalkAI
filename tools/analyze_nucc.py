#!/usr/bin/env python3
"""名大会話コーパスの分析（相槌・フィラー・笑い・危険ケース）。

**コーパス本体と抽出した発話例はリポジトリに入れない**（CC BY-NC-ND 4.0。
TalkAI は MIT で GitHub 公開のため）。このスクリプトは読むだけで、
例文の出力先はリポジトリ外（既定 C:/Users/user/Desktop/corpus/nucc_analysis）。
標準出力に出すのは集計値（件数・比率）のみ。

コーパスの記法（data001.txt を読んで確認）:
  F107：発話        … 話者つき。話者なしの行は同じ話者の続き
  （ふーん）        … 他の人の発話中に挟まれた相槌（聞き手の反応）
  ＜笑い＞          … 笑いの位置
  ＜間＞            … 間
  ＊＊＊            … 伏せ字・聞き取り不能
**時刻情報は無い**。よってミリ秒単位のタイミング分析はこのコーパスではできない。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
from pathlib import Path

SPK = re.compile(r"^([A-ZＡ-Ｚ][0-9０-９]*)：")
PAREN = re.compile(r"（([^（）]{1,20})）")
SEG_SPLIT = re.compile(r"[、。？！?!\s　]")

# ユーザ指定のフィラー表（docs/FILLER_DESIGN.md ★表）を数えるための語族。
# ここは「数える」ためだけの定義で、表の中身は変えない。
USER_FAMILIES = {
    "んー":  r"ん[んー]*",
    "あー":  r"あ[あー]+",
    "ふーん": r"ふ[うー]*ん",
    "へー":  r"へ[えー]*",
    "おー":  r"お[おー]+",
    "ほー":  r"ほ[おー]*",
    "いやー": r"いや[あー]*",
    "うー":  r"う[うー]+",
    "え":   r"え",
    "えー":  r"え[えー]+",
    "あ":   r"あ",
    "う":   r"う",
    "お":   r"お",
    "まあ":  r"まあ[あー]*",
    "ま":   r"ま",
    "ね":   r"ね",
    "ねー":  r"ね[えー]+",
    "そ":   r"そ",
    "そう":  r"そう",
}
USER_RE = {k: re.compile(rf"^{v}$") for k, v in USER_FAMILIES.items()}
# 表に無いが頻出しうる語族（**表には足さない**。報告用に数えるだけ）
OTHER_FAMILIES = {
    "うん系": r"(うん[、,]?)+",
    "うーん": r"うー+ん",
    "そう反復": r"(そう[、,]?){2,}",
    "はい系": r"(はい[、,]?)+",
    "ええ系": r"(ええ[、,]?)+",
    "ああ": r"ああ",
}
OTHER_RE = {k: re.compile(rf"^{v}$") for k, v in OTHER_FAMILIES.items()}


def load(corpus: Path):
    """会話ごとに turns=[{spk, text, backchannels}] を返す。"""
    convs = []
    for f in sorted(glob.glob(str(corpus / "data*.txt"))):
        turns, cur = [], None
        for ln in open(f, encoding="utf-8"):
            ln = ln.rstrip("\n").strip("　 ")
            if not ln or ln.startswith("＠"):
                continue
            m = SPK.match(ln)
            if m:
                cur = {"spk": m.group(1), "lines": [ln[m.end():]]}
                turns.append(cur)
            elif cur is not None:
                cur["lines"].append(ln)
        for t in turns:
            raw = "".join(t["lines"])
            t["backchannels"] = PAREN.findall(raw)
            t["text"] = PAREN.sub("", raw)  # 聞き手の相槌を除いた本人の発話
            t["raw"] = raw
        convs.append((Path(f).name, turns))
    return convs


def first_segment(text: str) -> str:
    """ターン冒頭の最初の区切りまで（＜笑い＞等の記号は除く）。"""
    t = re.sub(r"＜[^＞]*＞", "", text).lstrip("　 、")
    return SEG_SPLIT.split(t, maxsplit=1)[0] if t else ""


def family_of(w: str):
    for k, r in USER_RE.items():
        if r.match(w):
            return ("表", k)
    for k, r in OTHER_RE.items():
        if r.match(w):
            return ("表外", k)
    return (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/nucc/nucc"))
    ap.add_argument("--out", type=Path, default=Path(r"C:/Users/user/Desktop/corpus/nucc_analysis"))
    ap.add_argument("--samples", type=int, default=40, help="危険ケースの抽出数（種類ごと）")
    args = ap.parse_args()
    random.seed(0)

    convs = load(args.corpus)
    n_turns = sum(len(t) for _, t in convs)
    print(f"会話 {len(convs)} / ターン {n_turns:,}")

    # ---------- 1. 相槌（聞き手が相手の発話中に挟む）----------
    bc = collections.Counter()
    for _, turns in convs:
        for t in turns:
            bc.update(b.strip() for b in t["backchannels"])
    bc_total = sum(bc.values())
    fam_bc = collections.Counter()
    for w, c in bc.items():
        g, k = family_of(w)
        if g:
            fam_bc[(g, k)] += c
    print(f"\n[1] 相槌（聞き手・割り込み）総数 {bc_total:,} / 種類 {len(bc):,}")
    print("  語族別（上位）:")
    for (g, k), c in fam_bc.most_common(25):
        print(f"    {g:<3} {k:<6} {c:>6,}  ({c/bc_total*100:4.1f}%)")

    # 反復回数の分布（うん・そう）
    rep = collections.Counter()
    for w, c in bc.items():
        for base in ("うん", "そう", "はい"):
            n = len(re.findall(base, w))
            if n and re.fullmatch(rf"({base}[、,]?)+", w):
                rep[(base, n)] += c
    print("  反復回数（1回/2回/3回/4回以上）:")
    for base in ("うん", "そう", "はい"):
        row = [rep[(base, i)] for i in (1, 2, 3)] + [sum(v for (b, i), v in rep.items() if b == base and i >= 4)]
        tot = sum(row) or 1
        print(f"    {base}: " + " / ".join(f"{x:,}({x/tot*100:.0f}%)" for x in row))

    # ---------- 2. ターン冒頭（自分が話し始めるとき＝フィラーの役割）----------
    head = collections.Counter()
    turns_with_filler = 0
    for _, turns in convs:
        for t in turns:
            s = first_segment(t["text"])
            if 0 < len(s) <= 6:
                head[s] += 1
                g, _ = family_of(s)
                if g:
                    turns_with_filler += 1
    print(f"\n[2] ターン冒頭の短い語（6字以下）上位 / 全ターン {n_turns:,}")
    for w, c in head.most_common(30):
        g, k = family_of(w)
        tag = f"{g}:{k}" if g else ""
        print(f"    {w:<8} {c:>6,}  {tag}")
    print(f"  表または表外の語族で始まるターン: {turns_with_filler:,} ({turns_with_filler/n_turns*100:.1f}%)")

    fam_head = collections.Counter()
    for w, c in head.items():
        g, k = family_of(w)
        if g:
            fam_head[(g, k)] += c
    print("  ターン冒頭・語族別:")
    for (g, k), c in fam_head.most_common(20):
        print(f"    {g:<3} {k:<6} {c:>6,}")

    # ---------- 3. ユーザの表の各語の使われ方 ----------
    print("\n[3] ユーザの表の語: 相槌としての回数 / ターン冒頭としての回数")
    for k in USER_FAMILIES:
        print(f"    {k:<5} 相槌 {fam_bc[('表', k)]:>6,}   冒頭 {fam_head[('表', k)]:>6,}")

    # ---------- 4. 笑い ----------
    laugh_bc = sum(c for w, c in bc.items() if "笑" in w)
    laugh_turn_head = sum(1 for _, ts in convs for t in ts if t["raw"].lstrip().startswith("＜笑"))
    laugh_in_turn = sum(t["raw"].count("＜笑") for _, ts in convs for t in ts)
    print(f"\n[4] 笑い")
    print(f"  聞き手が相手の発話中に笑う（（＜笑い＞））: {laugh_bc:,}")
    print(f"  ターンの冒頭で笑う: {laugh_turn_head:,}")
    print(f"  ＜笑い＞総出現（ターン内）: {laugh_in_turn:,}")
    lb = collections.Counter()  # 笑いの直後に続く語
    for _, ts in convs:
        for t in ts:
            for m in re.finditer(r"＜笑[^＞]*＞([^＜、。]{0,6})", t["text"]):
                nxt = m.group(1).strip()
                if nxt:
                    lb[nxt] += 1
    print("  笑いの直後に続く語（上位）:", ", ".join(f"{w}({c})" for w, c in lb.most_common(15)))

    # ---------- 5. 間 ----------
    ma = sum(t["raw"].count("＜間＞") for _, ts in convs for t in ts)
    print(f"\n[5] ＜間＞ {ma:,} 件（時刻が無いので長さは不明）")

    # ---------- 6. 危険ケースの抽出（例文はリポジトリ外へ）----------
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {"iya": [], "sou": [], "laugh": []}
    for fname, ts in convs:
        for i, t in enumerate(ts):
            if i == 0:
                continue
            prev = [ts[j]["text"] for j in range(max(0, i - 2), i)]
            s = first_segment(t["text"])
            rec = {"file": fname, "context": prev, "target": t["raw"][:120]}
            if USER_RE["いやー"].match(s):
                cases["iya"].append(rec)
            if re.fullmatch(r"(そう|そ)[、,]?(そう)*", s):
                cases["sou"].append(rec)
            if t["raw"].lstrip().startswith("＜笑"):
                cases["laugh"].append(rec)
    for k, v in cases.items():
        pick = random.sample(v, min(args.samples, len(v)))
        with open(args.out / f"danger_{k}.jsonl", "w", encoding="utf-8") as f:
            for r in pick:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[6] 危険ケース {k}: 全 {len(v):,} 件 → {len(pick)} 件を抽出")
    print(f"  出力先（リポジトリ外）: {args.out}")

    # 集計値は JSON にも残す（例文を含まない）
    with open(args.out / "stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "conversations": len(convs), "turns": n_turns,
            "backchannel_total": bc_total,
            "backchannel_top": bc.most_common(100),
            "turn_head_top": head.most_common(100),
            "laugh": {"listener": laugh_bc, "turn_head": laugh_turn_head, "total": laugh_in_turn},
            "pause_marker": ma,
        }, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
