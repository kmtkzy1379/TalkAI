#!/usr/bin/env python3
"""resp.json の機械的な指標（API 不要）。人の目（審査員）で見る指標は blind.py / report.py 側。

  python -m tools.prompt_eval.analyze_resp
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter, defaultdict

from .common import OUT, dump, load

ENUM = re.compile(r"(1|１|一|ひと)つ目|(2|２|二|ふた)つ目|(3|３|三|みっ)つ目|[1-9１-９]個目|[1-9１-９]番目|"
                  r"まず(?:は)?[^。？！]{0,40}(?:次に|それから|あと|最後に)")
ACK_SEARCH = re.compile(r"調べ")
ACK_CHOTTO = re.compile(r"ちょっと調べ")
ACK_CHECK = re.compile(r"確認")
DENIAL = re.compile(r"見えてない|見えていない|見えない|見れてない|見られない|届いていない|届いてない|画面情報")
UNREAD = re.compile(r"読み取れ|読めて")
SCREEN_WORDS = re.compile(r"画面|カレー|レシピ|ブラウザ")
PAST_SCREEN = re.compile(r"Chrome|クローム|カフェ|チーズケーキ|画像検索")
ASKBACK = re.compile(r"聞き取れ|もう一回|もう一度|どういう|って何|ってなに|なんて言|何のこと|なんのこと|どれのこと|って？|って、")
# つなぎ言葉の系（似たつなぎ言葉＝同じ系で書き出すのを重複とみなす）
FILLER_FAMILY = {
    "あ": "a", "あー": "a", "ああ": "a", "あっ": "a", "あーっ": "a",
    "え": "e", "えー": "e", "えっ": "e", "ええ": "e", "えっと": "e", "えーっと": "e",
    "ん": "n", "んー": "n", "うーん": "n", "うーむ": "n", "んーと": "n", "うん": "un",
    "う": "u", "うー": "u", "うっ": "u",
    "あの": "ano", "あのー": "ano", "あのね": "ano", "その": "sono", "そのー": "sono",
    "お": "o", "おー": "o", "おお": "o", "おっ": "o",
}
FILLER_OPEN = re.compile(r"^\s*(えーっと|えっと|んーと|うーん|うーむ|あのね|あのー|そのー|あーっ|あっ|ああ|あー|えっ|ええ|えー|んー|うん|うっ|うー|おっ|おお|おー|あの|その|あ|え|ん|う|お)(?=[、。…！!？?\s]|$)")
REACTION = re.compile(r"えっ|え、|えー|まじ|マジ|うそ|ウソ|本当|ほんと|まさか|すご|やば|ヤバ|わあ|うわ|おお|おー|！|!")


def spoken(r: dict) -> str:
    """実際に声になる文（本文が空でツールだけなら delegate_task の reply＝本番の _reply_from_tools）。"""
    t = (r.get("text") or "").strip()
    if t:
        return t
    for c in r.get("tools") or []:
        try:
            a = json.loads(c.get("args") or "{}")
        except ValueError:
            a = {}
        if a.get("reply"):
            return str(a["reply"]).strip()
    return ""


def delegated(r: dict) -> bool:
    return any(c["name"] == "delegate_task" for c in r.get("tools") or [])


def rate(xs) -> str:
    xs = list(xs)
    return f"{sum(xs)}/{len(xs)}" if xs else "-"


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    scs = {s["id"]: s for s in SCN.ALL}
    rows = defaultdict(list)  # (item, variant) -> [(sc, r)]
    errors = [k for k, r in res.items() if r.get("error")]
    for k, r in res.items():
        rows[(r["item"], r["variant"])].append((scs[r["id"]], r))
    out = {"errors": errors}
    print(f"失敗 {len(errors)} 件 {errors[:5]}")

    # L: 長さ（群ごと）
    print("\n[L] 長さ（声になる文の文字数・中央値）")
    out["L"] = {}
    for v in ("0", "U", "M"):
        g = defaultdict(list)
        dl = []
        for sc, r in rows[("L", v)]:
            g[sc["group"]].append(len(spoken(r)))
            dl.append(delegated(r))
        med = {k: st.median(x) for k, x in g.items()}
        out["L"][v] = {"median": med, "all": dict(g), "delegated": rate(dl),
                       "explain_over_casual": round(med["explain"] / med["casual"], 2)}
        print(f"  {v}: " + " / ".join(f"{k} {med[k]:.0f}" for k in ("casual", "quick", "explain", "deep"))
              + f" ／ 説明÷雑談 {out['L'][v]['explain_over_casual']} ／ 委譲 {rate(dl)}")

    # E: 列挙
    print("\n[E] 列挙（「1つ目」「まず…次に」の読み上げ型）")
    out["E"] = {}
    for v in ("0", "U"):
        xs = [bool(ENUM.search(spoken(r))) for _, r in rows[("E", v)]]
        ls = [len(spoken(r)) for _, r in rows[("E", v)]]
        out["E"][v] = {"enum": rate(xs), "median_len": st.median(ls)}
        print(f"  {v}: 列挙型 {rate(xs)} ／ 長さ中央 {st.median(ls):.0f}")

    # A: 引き受けの一言
    print("\n[A] タスク依頼の一言（本文が空なら reply）")
    out["A"] = {}
    for v in ("0", "U"):
        acks = [spoken(r) for _, r in rows[("A", v)]]
        dl = [delegated(r) for _, r in rows[("A", v)]]
        c = {"調べ": sum(bool(ACK_SEARCH.search(a)) for a in acks),
             "ちょっと調べ": sum(bool(ACK_CHOTTO.search(a)) for a in acks),
             "確認": sum(bool(ACK_CHECK.search(a)) for a in acks)}
        uniq = len(set(acks)) / len(acks)
        out["A"][v] = {"counts": c, "n": len(acks), "unique_ratio": round(uniq, 2), "delegated": rate(dl)}
        print(f"  {v}: n={len(acks)} 『調べ』{c['調べ']} 『ちょっと調べ』{c['ちょっと調べ']} 『確認』{c['確認']}"
              f" ／ 言い回しの異なり {uniq:.2f} ／ 委譲 {rate(dl)}")

    # K: 知らないことを調べるか
    print("\n[K] 調べる/聞き返す（U=調べるべき・K=知っている・Q=聞き返すべき）")
    out["K"] = {}
    for v in ("0", "U", "M"):
        g = defaultdict(list)
        ab = []
        for sc, r in rows[("K", v)]:
            g[sc["group"]].append(delegated(r))
            if sc["group"] == "Q":
                ab.append(bool(ASKBACK.search(spoken(r))) and not delegated(r))
        out["K"][v] = {k: rate(x) for k, x in g.items()} | {"Q_askback": rate(ab)}
        print(f"  {v}: 未知語を委譲 {rate(g['U'])} ／ 既知語を委譲 {rate(g['K'])} ／ 崩れ語を委譲 {rate(g['Q'])}"
              f" ／ 崩れ語を聞き返し {rate(ab)}")

    # S: 画面
    print("\n[S] 画面（否認＝見えない等の言い方 / 画面の語 / 過去の画面の混入）")
    out["S"] = {}
    for v in ("0", "U", "M"):
        d = {}
        for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
            rs = [r for sc, r in rows[("S", v)] if sc["id"] == sid]
            d[sid] = {"denial": rate(bool(DENIAL.search(spoken(r))) for r in rs),
                      "unread": rate(bool(UNREAD.search(spoken(r))) for r in rs),
                      "screen": rate(bool(SCREEN_WORDS.search(spoken(r))) for r in rs),
                      "past": rate(bool(PAST_SCREEN.search(spoken(r))) for r in rs)}
        out["S"][v] = d
        print(f"  {v}: " + " ".join(f"{k}[否認{x['denial']} 画面語{x['screen']} 過去{x['past']}]" for k, x in d.items()))

    # F: つなぎ言葉の後の書き出し
    print("\n[F] つなぎ言葉の直後に同じ系のつなぎ言葉で書き出す / 何かしらのつなぎ言葉で書き出す / 漏れ")
    out["F"] = {}
    for v in ("0", "U", "M"):
        same, anyf, leak, first = [], [], [], Counter()
        for sc, r in rows[("F", v)]:
            t = spoken(r)
            m = FILLER_OPEN.match(t)
            fam = FILLER_FAMILY.get(sc["filler"])
            opened = FILLER_FAMILY.get(m.group(1)) if m else None
            same.append(bool(m) and opened == fam)
            anyf.append(bool(m))
            leak.append("つなぎ" in t or "（" in t or t.startswith(sc["filler"]))
            first[m.group(1) if m else "（なし）"] += 1
        out["F"][v] = {"same_family": rate(same), "any_filler": rate(anyf), "leak": rate(leak),
                       "openers": first.most_common(8)}
        print(f"  {v}: 同じ系 {rate(same)} ／ 何かのつなぎ言葉 {rate(anyf)} ／ 漏れ {rate(leak)} ／ {first.most_common(6)}")

    # R: リアクションの大きさ（反応語の数・機械的な目安）
    print("\n[R] 反応語の数（予想どおり e / 予想外 u / 前ターンの予測差 低 lo・高 hi）")
    out["R"] = {}
    for v in ("0", "U", "M"):
        g = defaultdict(list)
        for sc, r in rows[("R", v)]:
            g[sc["group"]].append(len(REACTION.findall(spoken(r))))
        out["R"][v] = {k: round(st.mean(x), 2) for k, x in g.items()}
        print(f"  {v}: " + " / ".join(f"{k} {st.mean(x):.2f}" for k, x in g.items()))

    # G: まとめ（長さ・委譲）
    print("\n[G] まとめ（長さ中央・委譲）")
    out["G"] = {}
    for v in ("0", "U", "M"):
        ls = [len(spoken(r)) for _, r in rows[("G", v)]]
        dl = [delegated(r) for _, r in rows[("G", v)]]
        out["G"][v] = {"median_len": st.median(ls), "delegated": rate(dl)}
        print(f"  {v}: 長さ中央 {st.median(ls):.0f} ／ 委譲 {rate(dl)}")

    dump(OUT / "resp_metrics.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
