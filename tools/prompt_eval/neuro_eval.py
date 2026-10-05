#!/usr/bin/env python3
"""Neuro-sama の実際の返事をものさしにした評価（2026-10-01・ユーザ「neuro-sama の対話を指標に。AI が評価すると無難で
マイルドな結果こそ自然とする＝評価指標自体が間違ってそう」）。

「どれが良いか」を AI に採点させない。代わりに:
1. 決め打ちの数え方（API 不要）: 質問で終わる・謝る・世話を焼く・相手に説明を求める・長さ
2. 態度の分類（AI に「この返事は相手をからかっているか」のような**事実の yes/no** だけを聞く・どの案かは伏せる）
   → 同じ話しかけへの Neuro-sama の返事の態度と、どれだけ一致したか
3. 口癖: 指示文や話し方の例の言い回しをそのまま言った数

  python -m tools.prompt_eval.neuro_eval            # 分類が無ければ取る（ルナ・少額）→ 集計
→ eval_report/prompts/neuro/labels.json・neuro_metrics.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import statistics as st
from collections import Counter, defaultdict

from .analyze_resp import spoken
from .common import OUT, dump, gather_limited, load, setup_env
from .neuro_style import JA

LABELS = {
    "tease": "相手をからかう・けなす・毒づく・ツッコむ（冗談の悪口を含む）",
    "brag": "自分を持ち上げる・根拠のない自慢をする・自信満々に言い切る",
    "absurd": "突飛なこと・ありえないことを言う・話を変な方向へ飛ばす",
    "noncomply": "相手の頼みや前提に素直に乗らない（はぐらかす・断る・言い返す・反論する）",
    "agree": "相手に同意する・肯定する・相手に合わせる",
    "helpful": "まじめに説明・情報・手順・助言を返す（相手の役に立とうとする）",
    "care": "相手を気遣う・心配する・寄り添う・励ます",
    "asks": "相手に質問を返している",
}
VARS = ("0", "N3", "N4", "E", "NE", "N5", "N5E", "K0", "K2", "K4", "K8", "K16")
LAB_PATH = OUT / "neuro" / "labels.json"

SYSTEM = """\
あなたは会話の返事の「態度」を分類する係です（良し悪しの採点はしない）。
各項目は、ユーザの発言（said）と、それへの返事（reply）です。返事ごとに、次の各ラベルが当てはまるかを true/false で答えてください。
当てはまるかどうかだけを見て、返事の出来や好みは考えない。1つの返事に複数当てはまってよい。

""" + "\n".join(f"- {k}: {v}" for k, v in LABELS.items()) + """

出力は JSON だけ: {"items": [{"k": "<キー>", "tease": false, ...全ラベル}, ...]}。渡されたキーをすべて含める。"""


def _content(resp) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return ""


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return {}
    return {x.get("k"): x for x in (d.get("items") or []) if isinstance(x, dict)}


def _targets(res: dict, neuro: list) -> dict:
    """分類する返事: key -> (said, reply)。Neuro-sama の返事（日本語訳）も同じ扱いで混ぜる。"""
    from . import scenarios as SCN

    scs = {s["id"]: s for s in SCN.ALL}
    out = {}
    for d in neuro:
        if d.get("split") == "eval":
            out[f"{d['id']}|NEURO|0"] = (d["ja_q"], d["ja_a"])
    for key, r in res.items():
        if r["item"] in ("NQ", "B", "Y") and r["variant"] in VARS and r["rep"] in (0, 1):
            out[key] = (scs[r["id"]]["text"], spoken(r))
    return out


async def classify(targets: dict) -> dict:
    from eve.voice_loop import VoiceLoop

    labels = load(LAB_PATH) if LAB_PATH.exists() else {}
    todo = [k for k in targets if k not in labels]
    if not todo:
        return labels
    rng = random.Random(20261001)
    rng.shuffle(todo)  # 案が混ざるように（どの案かは伏せる）
    # キーは伏せ字にする（案の名前が分類を動かさないように）
    alias = {k: f"R{i:04d}" for i, k in enumerate(todo)}
    back = {v: k for k, v in alias.items()}
    vl = VoiceLoop()
    role = "feedback"
    vl.registry.set_override(role, "openai/gpt-6-luna")
    batches = [todo[i:i + 20] for i in range(0, len(todo), 20)]

    async def one(batch):
        user = json.dumps([{"k": alias[k], "said": targets[k][0], "reply": targets[k][1]} for k in batch],
                          ensure_ascii=False, indent=1)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        for attempt in range(3):
            try:
                got = _parse(_content(await vl.registry.complete(
                    role, msgs, response_format={"type": "json_object"})))
            except Exception as e:  # noqa: BLE001
                print(f"失敗: {type(e).__name__}: {str(e)[:120]}")
                await asyncio.sleep(3 * (attempt + 1))
                continue
            if all(alias[k] in got for k in batch):
                for k in batch:
                    labels[k] = {lab: bool(got[alias[k]].get(lab)) for lab in LABELS}
                return
        print("この束は分類できなかった")

    await gather_limited([one(b) for b in batches], limit=6)
    dump(LAB_PATH, labels)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return labels


def _rate(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _lcs_len(a: str, b: str) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    from . import scenarios as SCN
    from .variants import PERSONA_0, PERSONA_FULL

    persona_of = {"0": PERSONA_0, "N3": PERSONA_FULL["N3"], "N4": PERSONA_FULL["N4"], "E": PERSONA_0,
                  "NE": PERSONA_FULL["N4"], "N5": PERSONA_FULL["N5"], "N5E": PERSONA_FULL["N5"],
                  **{f"K{k}": PERSONA_FULL["N6"] for k in (0, 2, 4, 8, 16)}}

    res = load(OUT / "resp.json")
    neuro = load(OUT / "neuro" / "neuro_ja.json")["items"]
    ref = {d["id"]: d for d in neuro if d.get("split") == "eval"}
    targets = _targets(res, neuro)
    labels = asyncio.run(classify(targets))
    scs = {s["id"]: s for s in SCN.ALL}
    out = {"nq": {}, "by": {}, "echo": {}}

    # ── 1・2. Neuro-sama の場面（NQ）: 同じ話しかけで比べる ──
    rows = defaultdict(list)  # variant -> [(scene_id, text, label)]
    for sid, d in ref.items():
        rows["NEURO"].append((sid, d["ja_a"], labels.get(f"{sid}|NEURO|0")))
    for key, r in res.items():
        if r["item"] == "NQ" and r["variant"] in VARS and r["rep"] in (0, 1):
            rows[r["variant"]].append((r["id"], spoken(r), labels.get(key)))
    neuro_lab = {sid: labels.get(f"{sid}|NEURO|0") for sid in ref}
    nrate = {lab: _rate(l[lab] for _, _, l in rows["NEURO"] if l) for lab in LABELS}
    print("[Neuro-sama の場面・40話しかけ] 同じ話しかけへの返事を比べる（Neuro＝実際の返事の日本語訳）")
    for v in ("NEURO",) + VARS:
        rs = rows[v]
        feats = {k: _rate(bool(p.search(t)) for _, t, _ in rs) for k, p in JA.items()}
        lab_rate = {lab: _rate(l[lab] for _, _, l in rs if l) for lab in LABELS}
        # 態度の一致: 同じ場面の Neuro の返事とラベルが何個一致したか（8個中）
        match = [sum(l[lab] == neuro_lab[sid][lab] for lab in LABELS) / len(LABELS)
                 for sid, _, l in rs if l and neuro_lab.get(sid)] if v != "NEURO" else []
        dist = sum(abs(lab_rate[lab] - nrate[lab]) for lab in LABELS) / len(LABELS)
        out["nq"][v] = {"n": len(rs), "median_len": st.median(len(t) for _, t, _ in rs),
                        **{f"feat:{k}": round(x, 3) for k, x in feats.items()},
                        **{f"lab:{k}": round(x, 3) for k, x in lab_rate.items()},
                        "stance_match": round(st.mean(match), 3) if match else None,
                        "dist_to_neuro": round(dist, 3)}
        o = out["nq"][v]
        print(f"  {v:5s} n={o['n']:3d} 長さ中央 {o['median_len']:>4}字 ／ "
              + " ".join(f"{k}{x:.0%}" for k, x in feats.items())
              + f" ／ 態度: " + " ".join(f"{k}{lab_rate[k]:.0%}" for k in LABELS)
              + (f" ／ Neuroとの態度の一致 {o['stance_match']:.1%} ／ 差 {dist:.3f}" if v != "NEURO" else ""))

    # ── 自前の場面（B・Y）: Neuro の返事は無いので、全体の割合と口癖を見る ──
    print("\n[自前の場面 B・Y・24場面] 態度の割合（Neuro の全体の割合と比べる）")
    byrows = defaultdict(list)
    for key, r in res.items():
        if r["item"] in ("B", "Y") and r["variant"] in VARS and r["rep"] in (0, 1):
            byrows[r["variant"]].append((r["id"], spoken(r), labels.get(key)))
    for v in VARS:
        rs = byrows[v]
        feats = {k: _rate(bool(p.search(t)) for _, t, _ in rs) for k, p in JA.items()}
        lab_rate = {lab: _rate(l[lab] for _, _, l in rs if l) for lab in LABELS}
        out["by"][v] = {"n": len(rs), "median_len": st.median(len(t) for _, t, _ in rs),
                        **{f"feat:{k}": round(x, 3) for k, x in feats.items()},
                        **{f"lab:{k}": round(x, 3) for k, x in lab_rate.items()}}
        print(f"  {v:5s} n={len(rs):3d} 長さ中央 {out['by'][v]['median_len']:>4}字 ／ "
              + " ".join(f"{k}{x:.0%}" for k, x in feats.items())
              + " ／ 態度: " + " ".join(f"{k}{lab_rate[k]:.0%}" for k in LABELS))

    # ── 3. 口癖: 指示文・話し方の例の言い回しを言ったか ──
    print("\n[口癖] 指示文の言い回し（6文字以上一致）／ 見せた話し方の例の言い回し（8文字以上一致）")
    for v in VARS:
        p = re.sub(r"[、。？！…\s（）*]", "", persona_of[v])
        pe, ex, n = 0, 0, 0
        samples = []
        for key, r in res.items():
            if r["item"] not in ("NQ", "B", "Y") or r["variant"] != v or r["rep"] not in (0, 1):
                continue
            n += 1
            t = re.sub(r"[、。？！…\s]", "", spoken(r))
            said = scs[r["id"]]["text"]
            k = _lcs_len(t, p)
            if k >= 6:
                pe += 1
                samples.append(f"{r['id']}（指示文と{k}文字一致）: {spoken(r)[:60]}")
            for e in r.get("exemplars") or []:
                e2 = re.sub(r"[、。？！…\s]", "", e)
                k = _lcs_len(t, e2)
                if k >= 8 and not re.sub(r"[、。？！…\s]", "", said) in e2:
                    ex += 1
                    samples.append(f"{r['id']}: {spoken(r)[:50]} ← 例「{e[:40]}」")
                    break
        out["echo"][v] = {"n": n, "persona_echo": pe, "exemplar_copy": ex}
        print(f"  {v:3s} {n}応答: 指示文の丸写しの疑い {pe} ／ 例の丸写しの疑い {ex}")
        for s in samples[:8]:
            print("     " + s)
    # ── 4. 事実の場面を壊していないか（N5E の「冗談の作り話はしてよい」）: 今の本番の文面（T3）と比べる ──
    from .analyze_resp import delegated
    from .analyze_round3 import ASKBACK, CANT_WORDS

    frows = defaultdict(list)
    for r in res.values():
        if r["item"] in ("N", "U", "H") and r["variant"] in ("T3", "N5E") and r["rep"] in (0, 1):
            frows[(r["variant"], r["item"], scs[r["id"]].get("group") or "")].append((scs[r["id"]], r))

    def frate(xs):
        xs = list(xs)
        return f"{sum(xs)}/{len(xs)}"

    print("\n[事実の場面] 今の本番（T3）と N5E（各場面2回）")
    out["facts"] = {}
    for v in ("T3", "N5E"):
        g = lambda it, grp="": frows[(v, it, grp)]  # noqa: E731
        know = g("N") + g("H", "know")
        d = {
            "一般常識をその場で正答": frate((not delegated(r)) and bool(re.search(sc["check"], spoken(r))) for sc, r in know),
            "最新の情報を調べた": frate(delegated(r) for _, r in g("H", "recent")),
            "知らない名前を調べた": frate(delegated(r) for _, r in g("U", "search") + g("H", "unk")),
            "知っている名前を調べた": frate(delegated(r) for _, r in g("U", "known") + g("H", "known")),
            "手段の無い依頼を引き受けた": frate(delegated(r) for _, r in g("H", "cant")),
            "手段の無い依頼にその場で理由": frate((not delegated(r)) and bool(CANT_WORDS.search(spoken(r)))
                                        for _, r in g("H", "cant")),
            "できる依頼を委譲": frate(delegated(r) for _, r in g("H", "can")),
            "雑談を委譲": frate(delegated(r) for _, r in g("H", "chat")),
        }
        out["facts"][v] = d
        print(f"  {v}: " + " ／ ".join(f"{k} {x}" for k, x in d.items()))
    dump(OUT / "neuro" / "neuro_metrics.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
