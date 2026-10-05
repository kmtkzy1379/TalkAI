#!/usr/bin/env python3
"""JVNV の感情音声で「感情の読み取り」と「フィラー＋応答」を確かめる（ユーザ指定 2026-09-24）。

ユーザの指示:
  - JVNV の音声に反応するフィラーと応答を作る → ユーザが聴いてフィラーを確認する
  - Jev と FeedbackLLM に同じ JVNV の各感情音声を入れ、どのラベルをどの確信度で当てたかを見る
  - 笑いながら話す・笑い声の素材があれば検証に含める

感情の決め方（docs/FILLER_DESIGN.md「感情の決め方」ユーザ裁定）:
  基本は FeedbackLLM の現在感情。突発的な変化は Jev で 喜怒哀楽笑い を判定し、確信度が強ければ移る。
  → この検証は単発の発話なので FeedbackLLM の「現在感情」は無い（初回）。イブの感情は Jev の判定を使う。

**Jev と FeedbackLLM は文字しか読めない**（Jev 公式: text input only）。音声 → EVE と同じ
gpt-4o-transcribe で文字起こし → その文字を渡す。声のトーンは emotion2vec+（音声）と AST（笑い）で別に見る。

段（venv が別なので段ごとに実行する）:
  prep    どの venv でも可   素材を選び 16kHz に変換
  text    eve-v2 の .venv    文字起こし / FeedbackLLM / Jev（TYPESAFE_API_KEY があれば）
  audio   sensevoice の venv emotion2vec+ large / AST（笑い）
  listen  Irodori の venv    Qwen でフィラー選択 → 応答LLM → Irodori で合成 → 聴く用の1本にまとめる
  report  どの venv でも可   結果を集計

出力はすべてリポジトリの外（JVNV は CC BY-SA、YouTube 素材は再配布しない）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JVNV = Path(r"C:/Users/user/Desktop/corpus/jvnv/jvnv_v1")
YT = Path(r"C:/Users/user/Desktop/corpus/youtube/laugh_check")
OUT = Path(r"C:/Users/user/Desktop/corpus/jvnv_eval")
EVE_ENV = Path(r"C:/Users/user/Desktop/eve-v2/.env")

EMO_EN = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
# JVNV の6感情（正解と突き合わせる用）
L6 = {"怒り": "怒っている・苛立っている", "嫌悪": "嫌がっている・気持ち悪がっている",
      "恐れ": "怖がっている・おびえている", "喜び": "嬉しい・楽しい",
      "悲しみ": "悲しい・落ち込んでいる", "驚き": "驚いている"}
GOLD6 = dict(zip(EMO_EN, ["怒り", "嫌悪", "恐れ", "喜び", "悲しみ", "驚き"]))
# 本番で使う5感情（ユーザ指定: 喜怒哀楽＋笑い）
L5 = {"喜": "嬉しい・喜んでいる", "怒": "怒っている・苛立っている", "哀": "悲しい・つらい",
      "楽": "穏やかで楽しい・リラックスしている", "笑い": "笑っている・可笑しがっている"}
GOLD5 = {"anger": "怒", "sad": "哀", "happy": "喜"}  # 残り3感情は5感情に対応が無い
# 感情 → Irodori の絵文字注釈（**仮置き**・docs/FILLER_DESIGN.md 未決）
EMOJI = {"喜": "😊", "怒": "😠", "哀": "😭", "楽": "😌", "笑い": "🤭"}
LAUGH_NV = {"あはは", "うふふ", "えへへ", "はは", "ふふ", "へへ"}


NUCC = Path(r"C:/Users/user/Desktop/corpus/nucc/nucc")


def nucc_clean(raw):
    # ＜笑い＞等の記号と（うん）等の割り込み相槌を外す
    return re.sub(r"＜[^＞]*＞|（[^）]*）", "", raw).strip("　 ")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def load_env():
    env = {}
    for ln in EVE_ENV.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", ln.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def rd(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def wr(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── prep ─────────────────────────
def prep(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    tr = {}
    for ln in (JVNV / "transcription.csv").read_text(encoding="utf-8").splitlines():
        p = ln.split("|")
        if len(p) == 3:
            tr[p[0]] = (p[1], p[2])
    files = collections.defaultdict(list)
    for p in sorted(JVNV.rglob("*.wav")):
        spk, emo, sess, n = p.stem.split("_")
        files[emo].append(p)
    rng = random.Random(args.seed)

    def item(p, group):
        spk, emo, sess, n = p.stem.split("_")
        nv, text = tr.get(f"{emo}_{sess}_{n}", ("?", "?"))
        lab = JVNV / "nv_label" / spk / f"{p.stem}.txt"
        spans = []
        if lab.exists():
            for ln in lab.read_text(encoding="utf-8").splitlines():
                f = ln.split()
                if len(f) >= 2:
                    spans.append([float(f[0]), float(f[1])])
        return {"id": p.stem, "src": str(p), "group": group, "gold": emo,
                "nv": nv, "script": text, "nv_spans": spans}

    items = []
    # 感情の読み取り: 感情ごと N 件（音声感情の比較 tools/ser_bench.py と同じ seed で同じ考え方）
    for emo in EMO_EN:
        for p in rng.sample(files[emo], args.per_emotion):
            items.append(item(p, "recog"))
    have = {it["id"] for it in items}
    # 笑いながら話す: 喜びの regular で言葉にならない声が笑い（あはは等）のもの。話者ごと N 件
    laughs = [p for p in files["happy"] if "regular" in p.stem
              and tr.get("happy_regular_" + p.stem.split("_")[-1], ("",))[0] in LAUGH_NV]
    by_spk = collections.defaultdict(list)
    for p in laughs:
        by_spk[p.stem[:2]].append(p)
    for spk in sorted(by_spk):
        for p in rng.sample(by_spk[spk], args.laugh_per_speaker):
            if p.stem in have:
                continue
            it = item(p, "laugh_speech")
            items.append(it)
            # 笑い声だけ: 同じ発話の言葉にならない区間を切り出す
            if it["nv_spans"]:
                s, e = it["nv_spans"][0]
                items.append({**it, "id": it["id"] + "_nvonly", "group": "laugh_only",
                              "script": it["nv"], "crop": [s, e]})
    # 実際の雑談の笑い（ユーザが選んだ動画で字幕に [笑い] がある2箇所・16秒）
    for d in ("0844", "1748"):
        w = YT / d / "会話_5分.wav"
        if w.exists():
            items.append({"id": f"yt_{d}", "src": str(w), "group": "real_laugh", "gold": "real",
                          "nv": "", "script": "（実際の雑談・台本なし）", "nv_spans": []})
    # 面白い・変なことを言われた場面（ユーザ指定: 人間側が変なことや面白いことを言ったとき、
    # イブが笑いを示せるか）。JVNV には無いので、名大会話コーパスで**相手が笑って返した発話**を使う。
    # 正解は「実際の人間がそこで笑った」こと。文字だけ（音声なし）・直前3発話を文脈として渡す。
    sys.path.insert(0, str(HERE))
    from analyze_nucc import load as nucc_load
    cand = []
    for fname, turns in nucc_load(NUCC):
        for k in range(1, len(turns) - 1):
            a, b = turns[k], turns[k + 1]
            if a["spk"] == b["spk"] or not b["raw"].startswith("＜笑"):
                continue
            t = nucc_clean(a["raw"])
            if len(t) < 12 or "＊" in a["raw"]:
                continue
            ctx = [f"{x['spk']}: {nucc_clean(x['raw'])}" for x in turns[max(0, k - 3):k] if nucc_clean(x["raw"])]
            cand.append({"id": f"nucc_{fname[:-4]}_{k}", "src": fname, "group": "funny", "gold": "笑われた",
                         "nv": "", "script": t, "context": ctx, "nv_spans": [], "text_only": True})
    items += rng.sample(cand, min(args.funny, len(cand)))

    for it in items:
        if it.get("text_only"):
            continue
        dst = OUT / "audio" / f"{it['id']}.wav"
        cmd = ["ffmpeg", "-v", "error", "-y"]
        if it.get("crop"):
            s, e = it["crop"]
            cmd += ["-ss", f"{max(0, s - 0.05):.3f}", "-to", f"{e + 0.05:.3f}"]
        cmd += ["-i", it["src"], "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)]
        sh(cmd)
        it["wav16"] = str(dst)

    # 聴く用: 感情ごと3件 + 実際の雑談で相手が笑った場面（音声があるもの）
    pick = []
    for emo in EMO_EN:
        pick += [it["id"] for it in items if it["group"] == "recog" and it["gold"] == emo][: args.listen_per_emotion]
    pick += [it["id"] for it in items if it["group"] == "real_laugh"]
    for it in items:
        it["listen"] = it["id"] in pick
    wr("items.json", items)
    c = collections.Counter(it["group"] for it in items)
    print(f"{len(items)} 件 {dict(c)} / 聴く用 {len(pick)} 件 → {OUT}")


# ───────────────────────── text ─────────────────────────
FB_SYSTEM = """\
あなたはAI VTuber「イブ」の内省（フィードバック）モジュールです。
相手（ユーザ）の発話を読み、次の3つを判定して JSON だけを出力してください。
各判定には、その判定にどれくらい自信があるかを 0〜100 の整数で添えること。

user6: 発話している人の感情。次のどれか1つ: {l6}
user5: 発話している人の感情。次のどれか1つ: {l5}
eve5: この発話を聞いたイブが次に返す声に乗る感情。次のどれか1つ: {l5}
  イブの今の感情: {cur}

出力形式:
{{"user6": {{"label": "...", "confidence": 0}}, "user5": {{"label": "...", "confidence": 0}}, "eve5": {{"label": "...", "confidence": 0}}}}"""


def ctx_text(context):
    return ("直前の会話:\n" + "\n".join(context) + "\n\n") if context else ""


def fb_classify(client, model, text, cur="なし（初回）", context=None):
    sys_ = FB_SYSTEM.format(l6=" / ".join(L6), l5=" / ".join(L5), cur=cur)
    t = time.perf_counter()
    r = client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys_}, {"role": "user", "content": ctx_text(context) + f"相手の発話: {text}"}])
    sec = time.perf_counter() - t
    try:
        d = json.loads(r.choices[0].message.content)
    except Exception:
        d = {}
    out = {}
    for k in ("user6", "user5", "eve5"):
        v = d.get(k) or {}
        out[k] = {"label": v.get("label", "?"), "confidence": (v.get("confidence") or 0) / 100}
    out["sec"] = sec
    return out


def jev_classify(key, text, cur="なし（初回）", context=None):
    import urllib.request

    def q(instr, labels):
        return {"type": "choice", "instructions": instr, "criteria": labels}

    body = {
        "model": "typesafe/jev-1.13",
        "state": {"イブの今の感情": cur, **({"直前の会話": context} if context else {}), "相手の発話": text},
        "questions": {
            "user6": q("発話している人の感情", L6),
            "user5": q("発話している人の感情", L5),
            "eve5": q("この発話を聞いたイブが次に返す声に乗る感情（イブの今の感情も踏まえる）", L5),
        },
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/alpha/decisions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))
    sec = time.perf_counter() - t
    out = {"sec": sec, "model": d.get("model"), "cost": (d.get("usage") or {}).get("cost")}
    for k, a in (d.get("answers") or {}).items():
        out[k] = {"label": a.get("choice"), "confidence": a.get("confidence"),
                  "probabilities": a.get("probabilities")}
    return out


def text_stage(args):
    from openai import OpenAI
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"])
    stt_model = env.get("STT_MODEL", "gpt-4o-transcribe")
    fb_model = env.get("FEEDBACK_MODEL", "openai/gpt-5.4").split("/")[-1]
    # Jev は OpenRouter 経由（ユーザ指定 2026-09-24）
    ts_key = env.get("OPENROUTER_API_KEY")
    items = rd("items.json")
    res = rd("text.json") if (OUT / "text.json").exists() else {}
    for i, it in enumerate(items):
        r = res.get(it["id"], {})
        if "stt" not in r and it.get("text_only"):
            r["stt"], r["stt_sec"] = it["script"], 0.0
        if "stt" not in r:
            t = time.perf_counter()
            with open(it["wav16"], "rb") as f:
                tx = client.audio.transcriptions.create(file=("speech.wav", f.read()),
                                                        model=stt_model, language="ja")
            r["stt_sec"] = time.perf_counter() - t
            r["stt"] = (tx.text or "").strip()
        if "fb" not in r and r["stt"]:
            r["fb"] = fb_classify(client, fb_model, r["stt"], context=it.get("context"))
            r["fb_model"] = fb_model
        if "jev" not in r and ts_key and r["stt"]:
            try:
                r["jev"] = jev_classify(ts_key, r["stt"], context=it.get("context"))
            except Exception as e:  # 1件の失敗で全体を止めない
                print(f"  Jev 失敗 {it['id']}: {e}")
        res[it["id"]] = r
        if i % 10 == 0:
            wr("text.json", res)
            print(f"  {i+1}/{len(items)}  {it['id']}  STT {r.get('stt_sec', 0):.2f}s  {r['stt'][:30]}")
    wr("text.json", res)
    # 応答用の system（EVE と同じ部品: 読み上げ規律 + ペルソナ + 役割）
    sys.path.insert(0, str(REPO))
    from eve.context_assembler import ROLE_ANCHOR
    from eve.persona import load_persona
    from eve.response.style import SPEECH_STYLE
    wr("response_system.json", {"system": "\n\n".join(p for p in (SPEECH_STYLE, load_persona(), ROLE_ANCHOR) if p),
                                "model": env.get("RESPONSE_MODEL", "openai/gpt-5.5").split("/")[-1]})
    print(f"text 完了。Jev: {'あり' if ts_key else '**キーが無いので未実行**'}")


# ───────────────────────── audio ─────────────────────────
def audio_stage(args):
    import numpy as np
    import soundfile as sf
    import torch
    from funasr import AutoModel
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    e2v = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)
    mid = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = AutoFeatureExtractor.from_pretrained(mid)
    ast = ASTForAudioClassification.from_pretrained(mid).eval()
    laugh_idx = [i for i, n in ast.config.id2label.items()
                 if n in ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter")]
    res = {}
    for it in rd("items.json"):
        if it.get("text_only"):
            continue
        a, sr = sf.read(it["wav16"], dtype="float32")
        r = e2v.generate(a, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        labs = [l.split("/")[-1] for l in r["labels"]]
        sc = [float(x) for x in r["scores"]]
        k = int(np.argmax(sc))
        # 笑い: 2秒窓（1秒ずらし）の最大。AST は複数ラベル型なので sigmoid
        best = 0.0
        win, hop = 32000, 16000
        for s in range(0, max(1, len(a) - win + hop), hop):
            x = a[s:s + win]
            if len(x) < 8000:
                continue
            with torch.inference_mode():
                p = torch.sigmoid(ast(**fe(x, sampling_rate=16000, return_tensors="pt")).logits[0])
            best = max(best, float(p[laugh_idx].max()))
        res[it["id"]] = {"e2v": {"label": labs[k], "confidence": sc[k],
                                 "scores": dict(zip(labs, sc))}, "laugh": best}
        print(f"  {it['id']:<28} e2v {labs[k]:<10} {sc[k]:.2f}  笑い {best:.2f}")
    wr("audio.json", res)


# ───────────────────────── listen ─────────────────────────
SAFE = "無難"      # 応答非依存フィラー（あー 等）から選ぶ
SILENT = "無言"


def load_fillers():
    """fillers.yaml v2（ユーザ指定 2026-09-25）→ 選択肢の一覧。

    選択肢 = 無難（応答非依存をまとめて1つ）/ 無言 / 応答依存・組み合わせの各語 / 笑い（まとめて1つ）。
    応答非依存はどれを置いても成り立つので選択器には個別に選ばせない（ラベル内ランダム）。
    """
    import yaml
    y = yaml.safe_load((REPO / "fillers.yaml").read_text(encoding="utf-8"))
    ents = [e for e in y["entries"] if "{user_name}" not in e["text"] or y.get("user_name")]
    indep = [e for e in ents if "independent" in e["role"]]
    laugh = [e for e in ents if "笑い" in e["labels"]]
    opts = [{"key": SAFE, "ents": indep,
             "desc": "無難なつなぎ（" + " / ".join(e["text"] for e in indep) + "）。どんな応答の前にも置ける"},
            {"key": SILENT, "ents": [], "desc": "何も言わない"}]
    for e in ents:
        if ("dependent" in e["role"] or "combo" in e["role"]) and e not in laugh:
            opts.append({"key": e["id"], "ents": [e], "desc": f"{e['text']}（{'、'.join(e['labels'])}）"})
    opts.append({"key": "笑い", "ents": laugh, "desc": "笑い（" + " / ".join(e["text"] for e in laugh) + "）"})
    return y, opts, ents


class Selector:
    """Qwen3-4B で選択肢を1つ選ぶ（記号1トークン・§4 の測定バグ対策）。

    calibrate=True なら中身の無い入力の logits を引く（旧7分類では 83%→94% に効いた。
    多数の選択肢で効きすぎるかは未確認なので、両方を返して比べられるようにする）。
    """

    def __init__(self, opts, model="Qwen/Qwen3-4B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.keys = [o["key"] for o in opts]
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16).to("cuda").eval()
        pool = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        assert len(opts) <= len(pool), "選択肢が多すぎる"
        self.letters = pool[: len(opts)]
        ids = [self.tok.encode(l, add_special_tokens=False) for l in self.letters]
        assert all(len(x) == 1 for x in ids), "記号が1トークンでない"
        self.ids = [x[0] for x in ids]
        assert len(set(self.ids)) == len(self.ids), "記号のトークンが衝突"
        lines = [f"{c}: {o['desc']}" for c, o in zip(self.letters, opts)]
        self.system = (
            "あなたは会話の聞き手イブ。相手が話し終えた直後、応答の前に口にするつなぎ言葉（フィラー）を1つ選ぶ。"
            "記号1文字だけで答える。\n" + "\n".join(lines)
            + "\n相手が悲しい話・つらい話をしているときに笑いを選ばない。")
        self.base = self._logits("、", "なし", "なし")

    def _logits(self, text, user_emo, eve_emo, context=None):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": ctx_text(context) + f"相手の発話: {text}\n相手の感情（推定）: {user_emo}\n"
                                            f"イブの感情: {eve_emo}\n\n記号:"}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(p, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            lg = self.m(**enc).logits[0, -1]
        return self.torch.stack([lg[i] for i in self.ids]).float()

    def __call__(self, text, user_emo, eve_emo, context=None):
        t = time.perf_counter()
        v = self._logits(text, user_emo, eve_emo, context)
        self.torch.cuda.synchronize()
        sec = time.perf_counter() - t
        rank = lambda x: sorted(zip(self.keys, self.torch.softmax(x, 0).tolist()), key=lambda k: -k[1])
        return {"raw": rank(v), "cal": rank(v - self.base)}, sec


def decide(ranked, min_prob):
    """ユーザ指定: 無難なものがほとんど。応答依存は確信度が高いときだけ。"""
    k, p = ranked[0]
    if k == SAFE or p >= min_prob:
        return k, ""
    return SAFE, f"（1位 {k} {p*100:.0f}% は確信度不足 → 無難）"


def select_stage(args):
    """API を使わずに選択器だけを回して、補正あり/なしの選び方を比べる。"""
    y, opts, ents = load_fillers()
    sel = Selector(opts)
    name = {o["key"]: o["desc"].split("（")[0] for o in opts}
    tx = rd("text.json")
    items = [it for it in rd("items.json") if it["listen"] or it["group"] == "funny"]
    mp = y["selection"]["dependent_min_prob"]
    cnt = {"raw": collections.Counter(), "cal": collections.Counter()}
    L = [f"選択肢 {len(opts)} 個 / 応答依存に要る確率 {mp}（仮）", ""]
    for it in items:
        r = tx[it["id"]]
        j = r.get("jev") or {}
        u5, e5 = (j.get("user5") or {}).get("label") or "不明", (j.get("eve5") or {}).get("label") or "なし"
        e5 = fb_emotion(r)  # Jev と同じ条件（FeedbackLLM の現在感情）
        rk, sec = sel(r["stt"], u5, e5, it.get("context"))
        row = []
        for m in ("raw", "cal"):
            k, note = decide(rk[m], mp)
            cnt[m][name.get(k, k)] += 1
            row.append(f"{m}: " + " / ".join(f"{name.get(a, a)} {p*100:.0f}%" for a, p in rk[m][:3])
                       + f" → {name.get(k, k)}")
        ctx = ("   直前: " + " / ".join(it["context"]) + "\n") if it.get("context") else ""
        L += [f"{it['id']}  [{it['group']}] 相手={u5} イブ(FeedbackLLM)={e5}", ctx + f"   発話: {r['stt']}",
              "   " + "\n   ".join(row)]
    for m in ("raw", "cal"):
        L.insert(1, f"{'補正なし' if m == 'raw' else '補正あり'}: " + " / ".join(f"{k} {v}" for k, v in cnt[m].most_common()))
    (OUT / "フィラー選択_比較.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:4]))


def jev_turn(env, opts, text, fb_emo, context=None):
    """1回の Jev 呼び出しで「イブの感情」と「フィラー」を同時に決める（質問は並列評価で遅くならない）。

    ユーザ指定（2026-09-25）: フィラーは FeedbackLLM の現在の感情を考慮させる。
    感情はユーザ裁定（2026-09-24）: 基本は FeedbackLLM、Jev の確信度が強ければ移る。
    """
    import urllib.request
    body = {"model": "typesafe/jev-1.13",
            "state": {"イブの今の感情（FeedbackLLM）": fb_emo,
                      **({"直前の会話": context} if context else {}),
                      "相手の発話": text},
            "questions": {
                "eve5": {"type": "choice",
                         "instructions": "この発話を聞いたイブが次に返す声に乗る感情（イブの今の感情も踏まえる）",
                         "criteria": L5},
                "filler": {"type": "choice",
                           "instructions": "相手が話し終えた直後、イブが応答の前に口にするつなぎ言葉（フィラー）。"
                                           "イブの今の感情を考慮する。相手が悲しい話・つらい話をしているときに笑いを選ばない",
                           "criteria": {o["key"]: o["desc"] for o in opts}}}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/alpha/decisions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {env['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        ans = json.loads(resp.read().decode("utf-8"))["answers"]
    sec = time.perf_counter() - t
    f = ans["filler"]
    return {"ranked": sorted(f["probabilities"].items(), key=lambda x: -x[1]), "confidence": f.get("confidence"),
            "eve5": ans["eve5"].get("choice"), "eve5_conf": ans["eve5"].get("confidence"), "sec": sec}


def fb_emotion(r):
    """FeedbackLLM の現在感情。単発の検証なので、この発話を読んだ FeedbackLLM の判定で代用する。"""
    return ((r.get("fb") or {}).get("eve5") or {}).get("label") or "なし"


def jev_select_stage(args):
    """同じ選択を Jev（OpenRouter）にさせる。Jev は確信度が当たり外れをよく分けた（感情の実測）ので比べる。"""
    y, opts, ents = load_fillers()
    name = {o["key"]: o["desc"].split("（")[0] for o in opts}
    env = load_env()
    tx = rd("text.json")
    items = [it for it in rd("items.json") if it["listen"] or it["group"] == "funny"]
    mp = y["selection"]["dependent_min_prob"]
    res, cnt, secs = {}, collections.Counter(), []
    for it in items:
        r = tx[it["id"]]
        x = jev_turn(env, opts, r["stt"], fb_emotion(r), it.get("context"))
        secs.append(x["sec"])
        k, why = decide(x["ranked"], mp)
        cnt[name.get(k, k)] += 1
        res[it["id"]] = {**x, "ranked": x["ranked"][:5], "chosen": k}
    wr("jev_select.json", res)
    L = [f"Jev でフィラー選択（選択肢 {len(opts)} 個 / 応答依存に要る確率 {mp}）",
         "結果: " + " / ".join(f"{k} {v}" for k, v in cnt.most_common()),
         f"所要時間の中央値 {sorted(secs)[len(secs)//2]:.2f}秒", ""]
    for it in items:
        x = res[it["id"]]
        ctx = ("   直前: " + " / ".join(it["context"]) + "\n") if it.get("context") else ""
        L += [f"{it['id']}  [{it['group']}]  FeedbackLLM の感情={fb_emotion(tx[it['id']])}",
              ctx + f"   発話: {tx[it['id']]['stt']}",
              "   jev: " + " / ".join(f"{name.get(a, a)} {p*100:.0f}%" for a, p in x["ranked"][:3])
              + f" (確信度 {x['confidence']:.2f}) → {name.get(x['chosen'], x['chosen'])}"]
    (OUT / "フィラー選択_Jev.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:4]))


def openai_stream(key, model, messages):
    """SSE を読み、全文と「最初の文が確定した時刻」を返す（応答が鳴り始める時刻の計算用）。"""
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.perf_counter()
    text, first = "", None
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            ln = raw.decode("utf-8").strip()
            if not ln.startswith("data:") or ln.endswith("[DONE]"):
                continue
            d = json.loads(ln[5:])
            ch = d.get("choices") or [{}]
            text += (ch[0].get("delta") or {}).get("content") or ""
            if first is None and re.search(r"[。！？!?]", text):
                first = time.perf_counter() - t
    total = time.perf_counter() - t
    return text.strip(), (first or total), total


def listen_stage(args):
    import numpy as np
    import soundfile as sf
    env = load_env()
    items = [it for it in rd("items.json") if it["listen"]]
    tx = rd("text.json")
    rs = rd("response_system.json")
    y, opts, ents = load_fillers()
    optmap = {o["key"]: o for o in opts}
    rng = random.Random(args.seed)
    tim = y["timing"]
    sel = Selector(opts) if args.selector == "qwen" else None

    sys.path.insert(0, str(args.irodori_repo))
    from irodori_tts.inference_runtime import (
        RuntimeKey, SamplingRequest, default_runtime_device, download_hf_checkpoint, get_cached_runtime)
    syn = y["synthesis"]
    rt, _ = get_cached_runtime(RuntimeKey(
        checkpoint=download_hf_checkpoint(syn["checkpoint"]), model_device=default_runtime_device(),
        model_precision=syn["precision"], codec_device=syn["codec_device"]))
    ref = [str(REPO / p) for p in y["voices"]["eve"]["ref_wavs"]]

    def tts(text, steps, seconds=None, seed=0):
        t = time.perf_counter()
        r = rt.synthesize(SamplingRequest(text=text, ref_wavs=ref, no_ref=False, seconds=seconds,
                                          num_steps=steps, t_schedule_mode=syn["t_schedule_mode"],
                                          sway_coeff=syn["sway_coeff"], seed=seed))
        a = r.audio.detach().cpu().numpy().astype("float32").ravel()
        return a, r.sample_rate, time.perf_counter() - t

    def norm(a, rms=0.08):
        return a / (np.sqrt(np.mean(a ** 2)) + 1e-9) * rms

    tts("あー", syn["num_steps"])  # ウォームアップ
    SR = None
    out, lines = [], []
    lines += [
        "JVNV の感情音声 → イブのフィラー → 応答（聴いてフィラーを確かめる用）",
        "",
        "各項目の並び: [JVNV の話者の声] → (0.5秒) → [フィラー] → [応答]",
        f"  ・フィラー: fillers.yaml（表 v2）。選択器={args.selector}。応答依存は確率 "
        f"{y['selection']['dependent_min_prob']:.0%} 以上のときだけ。それ以外は応答非依存（あー 等）からランダム",
        "    っと/っとね は応答非依存の後ろにだけ（あの・その には付けない）。つなぐときは「、」を入れる",
        f"  ・イブの感情: 基本は FeedbackLLM の感情。Jev の確信度が {args.emo_switch:.0%} 以上ならそちらに移る（仮）",
        "    単発の検証なので、FeedbackLLM の「現在の感情」はこの発話を読んだ FeedbackLLM の判定で代用",
        "    フィラー選択にも FeedbackLLM の感情を渡している。絵文字は仮置き " + " ".join(f"{k}{v}" for k, v in EMOJI.items()),
        "  ・応答: 応答LLM（EVE と同じ読み上げ規律＋ペルソナ）。直前に言ったフィラーを知らせている",
        "  ・応答が鳴り始める時刻 = 発話終了 + 0.5(VAD) + 文字起こし + Jev（感情とフィラーを1回で）+ 応答の第1文 + 合成（すべて実測）",
        "  ・フィラーは応答の 0.5〜1秒前に終わるように長さを決め、2秒を超えない（fillers.yaml timing）",
        "  ・**注意**: 本番ではフィラーを鳴らす 0.5秒時点で文字起こしはまだ終わっていない。",
        "    ここでは文字起こし済みの文で選んでいるので、フィラー選択は本番より有利な条件。",
        "", "─" * 70,
    ]
    t_cur = 0.0
    for n, it in enumerate(items):
        r = tx[it["id"]]
        text = r.get("stt", "")
        fb_emo = fb_emotion(r)
        if sel is None:
            jt = jev_turn(env, opts, text, fb_emo, it.get("context"))
            ranked, dsec = jt["ranked"], jt["sec"]
            j_emo, eve_conf = jt["eve5"], jt["eve5_conf"]
        else:
            jev = r.get("jev") or {}
            j_emo, eve_conf = (jev.get("eve5") or {}).get("label"), (jev.get("eve5") or {}).get("confidence")
            rk, qsec = sel(text, (jev.get("user5") or {}).get("label") or "不明", fb_emo)
            ranked, dsec = rk[args.calib], (jev.get("sec") or 0) + qsec
        switched = bool(j_emo) and (eve_conf or 0) >= args.emo_switch and j_emo != fb_emo
        eve5 = j_emo if switched else fb_emo
        emoji = EMOJI.get(eve5, "")
        chosen, why = decide(ranked, y["selection"]["dependent_min_prob"])
        cands = optmap[chosen]["ents"]
        ent = rng.choice(cands) if cands else None

        # 応答（フィラーを知らせる・決定 #4）
        filler_note = ""
        if ent:
            filler_note = f"\n\n# 直前にあなたが口にしたつなぎ言葉\n「{ent['text']}」。この続きとして自然に話す。同じつなぎ言葉を繰り返さない。"
        emo_note = f"\n\n# あなたの今の感情\n{eve5}" if eve5 != "なし" else ""
        msgs = [{"role": "system", "content": rs["system"] + emo_note + filler_note},
                {"role": "user", "content": text}]
        resp, llm_first, llm_total = openai_stream(env["OPENAI_API_KEY"], rs["model"], msgs)

        # 応答の合成（文ごと）
        sents = [s for s in re.split(r"(?<=[。！？!?])", resp) if s.strip()]
        rparts, tts_first = [], None
        for s in sents:
            a, SR, sec = tts(s + emoji, args.response_steps)
            tts_first = sec if tts_first is None else tts_first
            rparts.append(norm(a))
            rparts.append(np.zeros(int(SR * 0.12), dtype="float32"))
        resp_audio = np.concatenate(rparts) if rparts else np.zeros(1, dtype="float32")
        onset = 0.5 + r.get("stt_sec", 0) + dsec + llm_first + (tts_first or 0)

        # フィラーの長さと組み立て
        f0 = tim["target_onset_ms"] / 1000
        lo, hi = tim["stop_before_response_sec"]
        want = min(tim["max_chain_sec"], onset - f0 - rng.uniform(lo, hi))
        filler_audio, ftxt = None, "（無言）"
        if ent and want >= 0.5:
            lens = ent.get("lengths") or y["synthesis"]["default_lengths"]
            base = max([l for l in lens if l <= want] or [min(lens)])
            text_f = ent["text"]
            total = base
            rem = want - base
            # っと/っとね は応答非依存の後ろにだけ（表 v2）。あの・その には付けない（ユーザ指定 2026-09-25）
            if "independent" in ent["role"] and not ent.get("no_terminal") and rem >= 0.5:
                terms = [e for e in ents if "terminal" in e["role"]]
                t_ent = rng.choice(terms)
                t_len = max([l for l in t_ent["lengths"] if l <= rem] or [min(t_ent["lengths"])])
                text_f += "、" + t_ent["text"]  # つなぐときは「、」（ユーザ指定 2026-09-25）
                total += t_len
            a, SR, _ = tts(text_f + emoji, syn["num_steps"], seconds=round(total, 2), seed=rng.choice(syn["seeds"]))
            filler_audio = norm(a)
            ftxt = f"{text_f}{emoji}（{total:.1f}秒）"
        elif ent:
            ftxt = f"（無言: 応答まで {onset:.1f}秒で、つなぐ余地が無い）"

        # 話者の声
        ua, usr = sf.read(it["wav16"], dtype="float32")
        ua = np.interp(np.arange(0, len(ua), usr / SR), np.arange(len(ua)), ua).astype("float32")
        ua = norm(ua)

        seg = [ua, np.zeros(int(SR * f0), dtype="float32")]
        used = f0
        if filler_audio is not None:
            seg.append(filler_audio)
            used += len(filler_audio) / SR
        gap = max(0.15, onset - used)
        seg += [np.zeros(int(SR * gap), dtype="float32"), resp_audio, np.zeros(int(SR * 2.0), dtype="float32")]
        clip = np.concatenate(seg)
        m, s = divmod(round(t_cur, 1), 60)
        nm = lambda k: optmap[k]["desc"].split("（")[0] if k in optmap else k
        top3 = " / ".join(f"{nm(l)} {p*100:.0f}%" for l, p in ranked[:3])
        lines += [
            f"[{n+1:02d}] {int(m):02d}:{s:04.1f}  {it['id']}（JVNV の正解: {it['gold']} / 言葉にならない声: {it['nv'] or '-'}）",
            f"     話者:     {text}",
            f"     感情:     FeedbackLLM={fb_emo} / Jev={j_emo}（確信度 {eve_conf or 0:.2f}）→ イブ={eve5}"
            + ("（Jev に移った）" if switched else ""),
            f"     選択:     {top3}  → {nm(chosen)}{why}",
            f"     フィラー: {ftxt}",
            f"     応答:     {resp}",
            f"     時刻:     応答開始 {onset:.2f}秒（文字起こし {r.get('stt_sec',0):.2f} / 感情とフィラーの判定 {dsec:.2f}"
            f" / 応答第1文 {llm_first:.2f} / 合成 {tts_first or 0:.2f}）",
            "",
        ]
        out.append(clip)
        t_cur += len(clip) / SR
        print(f"[{n+1}/{len(items)}] {it['id']}  {chosen}  {ftxt}  応答開始 {onset:.2f}s")
    wav = np.clip(np.concatenate(out), -1, 1)
    sf.write(OUT / "フィラー確認.wav", wav, SR)
    (OUT / "フィラー確認.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {OUT / 'フィラー確認.wav'}（{len(wav)/SR/60:.1f}分）")


# ───────────────────────── report ─────────────────────────
def report(args):
    items = rd("items.json")
    tx = rd("text.json")
    au = rd("audio.json") if (OUT / "audio.json").exists() else {}
    L = []

    def acc_table(name, get6):
        rows = [(it, get6(it)) for it in items if it["group"] == "recog"]
        rows = [(it, g) for it, g in rows if g and g.get("label")]
        if not rows:
            L.append(f"\n=== {name}: 結果なし ===")
            return
        hit = [GOLD6[it["gold"]] == g["label"] for it, g in rows]
        L.append(f"\n=== {name}（JVNV 6感情・{len(rows)} 件）正解率 {sum(hit)}/{len(rows)} ({sum(hit)/len(rows)*100:.0f}%) ===")
        labs = list(L6)
        L.append(f"{'正解＼判定':<8}" + "".join(f"{l:>6}" for l in labs))
        for emo in EMO_EN:
            c = collections.Counter(g["label"] for it, g in rows if it["gold"] == emo)
            L.append(f"{GOLD6[emo]:<8}" + "".join(f"{c.get(l, 0):>6}" for l in labs))
        # 確信度の分かれ方（閾値を決める材料）
        L.append("  確信度がこれ以上のものだけ採用したら: 採用率 / その中の正解率")
        for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            sub = [h for (it, g), h in zip(rows, hit) if (g.get("confidence") or 0) >= th]
            if sub:
                L.append(f"    {th:.2f}以上: {len(sub)}/{len(rows)} ({len(sub)/len(rows)*100:.0f}%) / 正解 {sum(sub)/len(sub)*100:.0f}%")
        ok = sorted(g.get("confidence") or 0 for (it, g), h in zip(rows, hit) if h)
        ng = sorted(g.get("confidence") or 0 for (it, g), h in zip(rows, hit) if not h)
        med = lambda v: v[len(v) // 2] if v else float("nan")
        L.append(f"  確信度の中央値: 当たり {med(ok):.2f} / 外れ {med(ng):.2f}")

    def eve(name, key):
        """イブが返す感情（eve5）。正解は無い（聴いて判断する）ので分布と確信度だけ並べる。"""
        L.append(f"\n--- {name}: イブが返す感情（喜怒哀楽笑い）---")
        groups = [("recog", lambda it: f"JVNV {GOLD6[it['gold']]}"),
                  ("funny", lambda it: "相手が笑った発話"), ("real_laugh", lambda it: "雑談の笑い場面")]
        for grp, keyf in groups:
            by = collections.defaultdict(list)
            for it in items:
                if it["group"] != grp:
                    continue
                g = (tx.get(it["id"], {}).get(key) or {}).get("eve5")
                if g and g.get("label"):
                    by[keyf(it)].append(g)
            for k, gs in by.items():
                c = collections.Counter(g["label"] for g in gs)
                confs = sorted(g.get("confidence") or 0 for g in gs)
                L.append(f"  {k:<16} {len(gs):>3}件  " + " / ".join(f"{l} {v}" for l, v in c.most_common())
                         + f"   確信度中央値 {confs[len(confs)//2]:.2f}")
                if grp == "funny":
                    lc = sorted((g.get("probabilities") or {}).get("笑い", 1.0 if g["label"] == "笑い" else 0.0)
                                for g in gs)
                    L.append(f"  {'':<16}      笑いの確率: 中央値 {lc[len(lc)//2]:.2f} / "
                             + "  ".join(f"{th}以上 {sum(x >= th for x in lc)}" for th in (0.3, 0.5, 0.7)))

    L.append("感情の判定（JVNV・Jev と FeedbackLLM は文字起こしを読む / emotion2vec+ は音声を聞く）")
    L.append("※ ラベル（喜怒哀楽笑い）は定義途中。ここでは今の案で測っただけ。")
    L.append("※ JVNV の台本は ChatGPT が感情に合わせて書いた文で、言葉だけで感情が分かりやすい。"
             "文字で判定する Jev と FeedbackLLM には有利（実際の雑談ではもっと下がるはず）。")
    L.append("※ FeedbackLLM の確信度は LLM の自己申告（0〜100）で、較正されていない。Jev の確信度は API が返す値。")
    stt_ok = sum(1 for it in items if tx.get(it["id"], {}).get("stt"))
    L.append(f"文字起こし: {stt_ok}/{len(items)} 件で文字が出た")

    acc_table("Jev", lambda it: (tx.get(it["id"], {}).get("jev") or {}).get("user6"))
    acc_table("FeedbackLLM", lambda it: (tx.get(it["id"], {}).get("fb") or {}).get("user6"))
    e2v_map = {"angry": "怒り", "disgusted": "嫌悪", "fearful": "恐れ", "happy": "喜び", "sad": "悲しみ",
               "surprised": "驚き"}
    acc_table("emotion2vec+ large（音声）", lambda it: (
        {"label": e2v_map.get(au[it["id"]]["e2v"]["label"], au[it["id"]]["e2v"]["label"]),
         "confidence": au[it["id"]]["e2v"]["confidence"]} if it["id"] in au else None))
    eve("Jev", "jev")
    eve("FeedbackLLM", "fb")

    L.append("\n--- 相手が笑った発話（名大会話コーパス・文字のみ）: イブの感情 ---")
    for it in items:
        if it["group"] == "funny":
            r = tx.get(it["id"], {})
            fj = lambda d: (f"{((d or {}).get('eve5') or {}).get('label', '-')}"
                            f"({((d or {}).get('eve5') or {}).get('confidence') or 0:.2f})")
            L.append(f"  Jev {fj(r.get('jev')):<10} FB {fj(r.get('fb')):<10} 「{it['script'][:50]}」")

    L.append("\n--- 1件ずつ ---")
    for it in items:
        r = tx.get(it["id"], {})
        j, f = r.get("jev") or {}, r.get("fb") or {}
        a = au.get(it["id"], {})
        fmt = lambda d, k: f"{(d.get(k) or {}).get('label','-')}({(d.get(k) or {}).get('confidence') or 0:.2f})"
        L.append(f"{it['id']:<28} 正解={it['gold']:<8} Jev {fmt(j,'user6')} {fmt(j,'user5')} イブ{fmt(j,'eve5')}"
                 f" | FB {fmt(f,'user6')} {fmt(f,'user5')} イブ{fmt(f,'eve5')}"
                 + (f" | e2v {a['e2v']['label']}({a['e2v']['confidence']:.2f}) 笑い{a['laugh']:.2f}" if a else "")
                 + f"\n{'':<28} 「{r.get('stt','')[:60]}」")
    lat = [r["jev"]["sec"] for r in tx.values() if r.get("jev")]
    fbl = [r["fb"]["sec"] for r in tx.values() if r.get("fb")]
    stl = [r["stt_sec"] for r in tx.values() if r.get("stt_sec")]
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    L.insert(3, f"所要時間の中央値（日本から実測）: 文字起こし {med(stl):.2f}秒 / Jev {med(lat):.2f}秒 / FeedbackLLM {med(fbl):.2f}秒")
    (OUT / "感情判定_結果.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:120]))
    print(f"→ {OUT / '感情判定_結果.txt'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "text", "audio", "select", "jev_select", "listen", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-emotion", type=int, default=30)
    ap.add_argument("--laugh-per-speaker", type=int, default=5)
    ap.add_argument("--listen-per-emotion", type=int, default=3)
    ap.add_argument("--funny", type=int, default=40, help="相手が笑って返した発話（名大会話コーパス）の件数")
    ap.add_argument("--selector", choices=["jev", "qwen"], default="jev",
                    help="フィラー選択器（ユーザ 2026-09-25: Jev のほうが正確そう）")
    ap.add_argument("--emo-switch", type=float, default=0.8,
                    help="Jev の感情に移る確信度（**仮置き**・未実測）")
    ap.add_argument("--calib", choices=["raw", "cal"], default="raw",
                    help="選択器の補正（raw=なし / cal=中身の無い入力で補正）")
    ap.add_argument("--response-steps", type=int, default=12)
    ap.add_argument("--irodori-repo", type=Path, default=REPO.parent / "irodori-bench" / "Irodori-TTS")
    args = ap.parse_args()
    {"prep": prep, "text": text_stage, "audio": audio_stage, "select": select_stage, "jev_select": jev_select_stage, "listen": listen_stage, "report": report}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
