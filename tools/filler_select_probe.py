#!/usr/bin/env python3
"""応答非依存フィラーを Jev に選ばせると、人が実際に選ぶフィラーに近づくかを測る（実 API・Jev・1件 約$0.00002）。

ユーザ 2026-09-27「今はフィラーランダムにしてるがjevで最適化とかできない？例えばお系は驚く、う系は迷う、悩むなどで
多くなります。一応非依存フィラーのため大きくおかしくなることはないがjevで選択させればかなりいい感じになりそう」

素材: 名大会話コーパスで、話者が替わった直後のターンが応答非依存フィラー（あ/あー/え/えー/お/う/うー/んー/あの/その）で
始まる所（＝相手の話を聞いた人が最初に口にした語）。直前の会話（最大6ターン）と相手の発話を Jev に渡し、10語から選ばせる。
各語の説明はユーザの表の意味分類（docs/FILLER_DESIGN.md「意味分類」＋ 2026-09-27「お系は驚く、う系は迷う、悩む」）。
あの・その は表に意味の指定が無いので、ユーザの言う「フィラーは次の言葉を探している」をそのまま使う。

見るもの:
- 人がその語を言った場面で、Jev がその語に付けた確率（他の語を言った場面と比べて高いか＝場面を見分けられているか）
- 今の重み（抽選）と、Jev の確率を混ぜた時の「人の選択の当てやすさ」（平均の対数尤度）

コーパス（CC BY-NC-ND）はリポジトリに入れない。出力はリポジトリ外 C:/Users/user/Desktop/corpus/jvnv_eval/filler_select/。
実行（EVE の venv）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\filler_select_probe.py --per-word 20
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from analyze_nucc import first_segment, load  # noqa: E402

CORPUS = Path(r"C:/Users/user/Desktop/corpus/nucc/nucc")
OUT = Path(r"C:/Users/user/Desktop/corpus/jvnv_eval/filler_select")
REPO = HERE.parent

# 語族（analyze_nucc.USER_FAMILIES と同じ書き方）→ fillers.yaml の id
WORDS = {
    "あ": r"あ", "あー": r"あ[あー]+", "え": r"え", "えー": r"え[えー]+", "お": r"お",
    "う": r"う", "うー": r"う[うー]+", "んー": r"ん[んー]*", "あの": r"あの[ー]*", "その": r"その[ー]*",
}
WORD_RE = {k: re.compile(rf"^{v}$") for k, v in WORDS.items()}
# 各語の説明（ユーザの表の意味分類。あ/う/お は「伸ばし版と同じ」）
MEANING = {
    "あ": "通常の反応", "あー": "通常の反応",
    "え": "驚き", "えー": "不満・迷い",
    "お": "驚き・感嘆", "う": "迷い・悩み・困惑", "うー": "迷い・悩み・困惑", "んー": "悩み",
    "あの": "次の言葉を探している", "その": "次の言葉を探している",
}
INSTR = ("相手の発話を聞いた直後に、聞き手が話し始める前に思わず口にするつなぎ言葉（フィラー）として一番自然なもの。"
         "中身の無い短い音で、続く言葉を探している間に出る")
# --mode category: 語ではなく反応の種類を聞く（ユーザの表の意味分類をまとめたもの）。あの・その は意味の指定が無いので入れない
CATEGORY = {
    "通常": "普通に受け止める", "驚き": "驚く・感心する", "迷い・悩み": "答えに迷う・考え込む・困る",
    "不満": "不満がある・納得がいかない",
}
CAT_OF = {"あ": "通常", "あー": "通常", "え": "驚き", "お": "驚き", "う": "迷い・悩み", "うー": "迷い・悩み",
          "んー": "迷い・悩み", "えー": "不満"}
CAT_INSTR = "相手の発話を聞いた直後の、聞き手の反応（話し始める前の気持ち）"
# --mode cat5: ユーザ指定の5種類（2026-09-27）と、それぞれで出てよい語（無言・特殊フィラー込み）
CAT5 = {
    "通常": "普通に受け止める", "悩む": "答えに悩む・考え込む", "驚く": "驚く・感心する",
    "悲しむ": "悲しむ・気の毒に思う", "不満": "不満がある・納得がいかない",
}
CAT5_WORDS = {
    "通常": ["あ", "あー", "え", "えー", "あの", "その", "んー", "無言", "特殊フィラー"],
    "悩む": ["え", "えー", "うー", "んー", "無言", "特殊フィラー"],
    "驚く": ["お"], "悲しむ": ["う", "無言"], "不満": ["えー", "無言"],
}
JVNV = Path(r"C:/Users/user/Desktop/corpus/jvnv_eval")


def load_env() -> dict:
    env = {}
    for ln in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", ln.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def nclean(s: str) -> str:
    return re.sub(r"＜[^＞]*＞|（[^）]*）", "", s or "").strip("　 ")


def collect(per_word: int, seed: int) -> list:
    convs = load(CORPUS)
    pool = collections.defaultdict(list)
    for fname, turns in convs:
        for k in range(1, len(turns)):
            t, prev = turns[k], turns[k - 1]
            if t["spk"] == prev["spk"]:
                continue
            seg = first_segment(t["text"])
            word = next((w for w, r in WORD_RE.items() if r.match(seg)), None)
            said = nclean(prev["raw"])
            if not word or not said:
                continue
            ctx = [f"{x['spk']}: {nclean(x['raw'])}" for x in turns[max(0, k - 7):k - 1] if nclean(x["raw"])]
            pool[word].append({"id": f"{fname[:-4]}_{k}", "word": word, "context": ctx,
                               "said": f"{prev['spk']}: {said}", "reply": nclean(t["raw"])[:60]})
    rng = random.Random(seed)
    items = []
    for w in WORDS:
        c = pool.get(w, [])
        rng.shuffle(c)
        items += c[:per_word]
    counts = {w: len(pool.get(w, [])) for w in WORDS}
    return items, counts


def jev(key: str, it: dict, mode: str = "word", eve_emotion: str = "") -> dict:
    if mode == "word":
        q = {"type": "choice", "instructions": INSTR, "criteria": {w: MEANING[w] for w in WORDS}}
    elif mode == "cat5":
        q = {"type": "choice", "instructions": CAT_INSTR, "criteria": CAT5}
    else:
        q = {"type": "choice", "instructions": CAT_INSTR, "criteria": CATEGORY}
    state = {"相手の発話": it["said"]}
    if it.get("context"):
        state = {"直前の会話": it["context"], **state}
    if eve_emotion:
        state = {"イブの今の感情": eve_emotion, **state}
    body = {"model": "typesafe/jev-1.13", "state": state, "questions": {"filler": q}}
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
            a = (d.get("answers") or {}).get("filler") or {}
            return {"sec": time.perf_counter() - t, "choice": a.get("choice"),
                    "probabilities": a.get("probabilities") or {}, "cost": (d.get("usage") or {}).get("cost")}
        except Exception as e:  # 一時失敗は数回だけ再試行
            err = e
            time.sleep(1 + attempt)
    return {"error": f"{type(err).__name__}: {err}"}


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-word", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eve-emotion", default="",
                    help="Jev の state に「イブの今の感情」を入れる（本番と同じ形・例 楽）。出力の名前にも付く")
    ap.add_argument("--mode", choices=("word", "category", "cat5"), default="word",
                    help="word=10語から選ばせる / category=反応の種類（驚き・迷い悩み・不満・通常）/ "
                         "cat5=ユーザ指定の5種類（通常・悩む・驚く・悲しむ・不満・2026-09-27）＋JVNV の感情つき発話")
    args = ap.parse_args()
    items, counts = collect(args.per_word, args.seed)
    if args.mode == "category":
        items = [it for it in items if it["word"] in CAT_OF]  # あの・その は種類の指定が無いので除く
        for it in items:
            it["cat"] = CAT_OF[it["word"]]
    if args.mode == "cat5":
        tx = json.loads((JVNV / "text.json").read_text(encoding="utf-8"))
        for jt in json.loads((JVNV / "items.json").read_text(encoding="utf-8")):
            said = ((tx.get(jt["id"]) or {}).get("stt") or "").strip()
            if jt["group"] == "recog" and said:
                items.append({"id": jt["id"], "src": "jvnv", "gold": jt["gold"], "context": [], "said": said})
    print("コーパスの件数（話者交代の直後がその語）:", counts)
    print(f"判定する件数: {len(items)}")
    key = load_env()["OPENROUTER_API_KEY"]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda it: jev(key, it, args.mode, args.eve_emotion), items))
    for it, r in zip(items, res):
        it.update(r)
    ok = [it for it in items if "error" not in it]
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.mode}" + (f"_eve{args.eve_emotion}" if args.eve_emotion else "")
    (OUT / f"result{suffix}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.mode == "category":
        return report_category(ok, len(items), args)
    if args.mode == "cat5":
        return report_cat5(ok, len(items), args)

    cfg = yaml.safe_load((REPO / "fillers.yaml").read_text(encoding="utf-8"))
    ids = {e["text"]: e["weight"] for e in cfg["entries"] if "independent" in (e.get("role") or [])}
    base = {w: ids[w] for w in WORDS}
    z = sum(base.values())
    base = {w: v / z for w, v in base.items()}

    def p(it, w):
        return float((it.get("probabilities") or {}).get(w, 0.0))

    lines = [f"応答非依存フィラーを Jev に選ばせた時の見え方（名大会話コーパス・{len(ok)}件・失敗 {len(items) - len(ok)}件）", ""]
    lines.append("■ 人がその語を言った場面での Jev の確率（その語 / 他の語を言った場面での同じ語）と、1位に選んだ割合")
    for w in WORDS:
        mine = [p(it, w) for it in ok if it["word"] == w]
        other = [p(it, w) for it in ok if it["word"] != w]
        top = sum(1 for it in ok if it["word"] == w and it.get("choice") == w)
        if mine:
            lines.append(f"  {w:<3}（{MEANING[w]}）n={len(mine):>2}  {statistics.mean(mine):.2f} / {statistics.mean(other):.2f}"
                         f"  1位 {top}/{len(mine)}  （今の抽選の確率 {base[w]:.3f}）")
    lines += ["", "■ 人の選択の当てやすさ（平均の対数尤度・高いほど良い）: (1-λ)×今の重み ＋ λ×Jev の確率"]
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        ll = statistics.mean(math.log(max(1e-6, (1 - lam) * base[it["word"]] + lam * p(it, it["word"]))) for it in ok)
        lines.append(f"  λ={lam:.1f}: {ll:.3f}")
    lines.append("  （素材は語ごとに同じ数を取っているので、今の重みには不利。見るのは Jev で上がるか）")
    choice = collections.Counter(it.get("choice") for it in ok)
    lines += ["", "■ Jev の1位の内訳: " + " / ".join(f"{k} {v}" for k, v in choice.most_common())]
    secs = [it["sec"] for it in ok]
    cost = sum(float(it.get("cost") or 0) for it in ok)
    lines.append(f"■ 1件 中央 {statistics.median(secs):.2f}秒（{args.workers}並列）・合計 ${cost:.4f}")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def report_category(ok: list, n: int, args) -> int:
    def p(it, c):
        return float((it.get("probabilities") or {}).get(c, 0.0))

    lines = [f"反応の種類を Jev に聞いた時の見え方（名大会話コーパス・{len(ok)}件・失敗 {n - len(ok)}件）",
             "人が言った語 → 種類（ユーザの表の意味分類）: あ/あー=通常・え/お=驚き・う/うー/んー=迷い・悩み・えー=不満", ""]
    lines.append("■ 人がその種類の語を言った場面での Jev の確率（その種類 / 他の種類の語の場面での同じ種類）と1位の割合")
    for c in CATEGORY:
        mine = [p(it, c) for it in ok if it["cat"] == c]
        other = [p(it, c) for it in ok if it["cat"] != c]
        top = sum(1 for it in ok if it["cat"] == c and it.get("choice") == c)
        lines.append(f"  {c:<5} n={len(mine):>2}  {statistics.mean(mine):.2f} / {statistics.mean(other):.2f}  1位 {top}/{len(mine)}")
    lines += ["", "■ 語ごと（人が言った語 → Jev の種類の確率の平均）"]
    for w in CAT_OF:
        sub = [it for it in ok if it["word"] == w]
        if sub:
            lines.append(f"  {w:<3} n={len(sub):>2}  " + " / ".join(f"{c} {statistics.mean(p(it, c) for it in sub):.2f}" for c in CATEGORY))
    choice = collections.Counter(it.get("choice") for it in ok)
    lines += ["", "■ Jev の1位の内訳: " + " / ".join(f"{k} {v}" for k, v in choice.most_common())]
    secs = [it["sec"] for it in ok]
    cost = sum(float(it.get("cost") or 0) for it in ok)
    lines.append(f"■ 1件 中央 {statistics.median(secs):.2f}秒（{args.workers}並列）・合計 ${cost:.4f}")
    (OUT / "README_category.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def report_cat5(ok: list, n: int, args) -> int:
    def p(it, c):
        return float((it.get("probabilities") or {}).get(c, 0.0))

    cats = list(CAT5)
    lines = [f"ユーザ指定の5種類を Jev に聞いた時の見え方（{len(ok)}件・失敗 {n - len(ok)}件）", ""]
    lines.append("■ 名大会話コーパス: 人が実際に言った語 → Jev の種類の確率の平均（1位の種類）")
    for w in WORDS:
        sub = [it for it in ok if it.get("word") == w]
        if sub:
            top = collections.Counter(it.get("choice") for it in sub).most_common(1)[0][0]
            lines.append(f"  {w:<3} n={len(sub):>2}  " + " / ".join(f"{c} {statistics.mean(p(it, c) for it in sub):.2f}"
                                                            for c in cats) + f"  （1位に多い: {top}）")
    lines += ["", "■ JVNV（話し手の感情がはっきりした発話・各30）→ Jev の種類の確率の平均と1位の内訳"]
    for g in ("happy", "surprise", "sad", "fear", "anger", "disgust"):
        sub = [it for it in ok if it.get("gold") == g]
        if sub:
            c1 = collections.Counter(it.get("choice") for it in sub)
            lines.append(f"  {g:<8} " + " / ".join(f"{c} {statistics.mean(p(it, c) for it in sub):.2f}" for c in cats)
                         + "  1位: " + " ".join(f"{k}{v}" for k, v in c1.most_common()))
    choice = collections.Counter(it.get("choice") for it in ok if it.get("src") != "jvnv")
    lines += ["", "■ 名大会話コーパスでの1位の内訳: " + " / ".join(f"{k} {v}" for k, v in choice.most_common())]
    secs = [it["sec"] for it in ok]
    cost = sum(float(it.get("cost") or 0) for it in ok)
    lines.append(f"■ 1件 中央 {statistics.median(secs):.2f}秒（{args.workers}並列）・合計 ${cost:.4f}")
    name = "README_cat5" + (f"_eve{args.eve_emotion}" if args.eve_emotion else "") + ".txt"
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
