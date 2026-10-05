#!/usr/bin/env python3
"""フィードバック（内省）の summary を「1文」から「重要な要素を簡潔に・事実のみ」へ変えた時の比べ。実 API（本番の feedback 役）。

  run   : 8スパン × 変種（0=今 / U=ユーザ案 / M=ユーザ案＋良い例・悪い例を外す）× 回数 を本番の組み立てで呼ぶ
  recall: 出てきた summary を本番と同じ形（summary | tags）で Ruri に埋め込み、本番の記憶 439件を混ぜた中から
          「あの話覚えてる？」型の問いで正しいスパンを上位3件に引けるかを測る（API 不要）

実行（EVE の venv・TalkAI-live のフォルダで）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.fb run --reps 2
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.fb recall
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from .common import OUT, dump, gather_limited, load, setup_env, stamp, sub

H, E = "user", "eve"
# key: 採点の要点（部分一致の手がかり語つき）。queries: 後で思い出す時の問い（そのスパンが上位に来るべき）
SPANS = [
    dict(id="FB1", note="1話題・短い",
         turns=[(H, "今日は久しぶりにジムに行ってきた"), (E, "おお、えらい！何やったの？"),
                (H, "ベンチプレス。60キロ上がった"), (E, "すごいじゃん、前より上がってるね。")],
         key=[("ジムに行った", ["ジム"]), ("ベンチプレス", ["ベンチプレス"]), ("60キロ", ["60"])],
         queries=["ジムの話覚えてる？", "ベンチプレス何キロだったっけ"]),
    dict(id="FB2", note="3話題（会議・猫・晩ごはん）",
         turns=[(H, "今日会議が3時間もあって疲れた"), (E, "3時間は長いね、おつかれさま。"),
                (H, "しかも帰り道で野良猫にずっとついてこられた"), (E, "猫に気に入られたんじゃない？"),
                (H, "かもね。あ、晩ごはんは餃子にしようと思ってる"), (E, "いいね、焼き餃子？"),
                (H, "うん、冷凍のやつだけど")],
         key=[("会議が3時間で疲れた", ["会議"]), ("帰り道で野良猫についてこられた", ["猫"]),
              ("晩ごはんは餃子（冷凍）", ["餃子"])],
         queries=["猫がついてきた話したっけ", "晩ごはんに何食べるって言ってたっけ", "会議が長かった日のこと"]),
    dict(id="FB3", note="4話題が次々に変わる",
         turns=[(H, "明日から出張で大阪行くんだ"), (E, "いいね、何日くらい？"),
                (H, "二泊。そういえば昨日スマホの画面割っちゃってさ"), (E, "うわ、痛いね。修理する？"),
                (H, "保険入ってるから大丈夫。あとさ、弟が来月結婚するんだよ"), (E, "おめでとう！お祝い考えないとね。"),
                (H, "そう、何あげるか迷ってる。あ、あとイブの好きな色って何？"), (E, "藍色かな。落ち着く感じが好き。")],
         key=[("明日から大阪に二泊の出張", ["大阪"]), ("スマホの画面を割った（保険で直す）", ["スマホ", "画面"]),
              ("弟が来月結婚・お祝いに迷っている", ["弟", "結婚"]), ("イブの好きな色は藍色", ["藍色"])],
         queries=["大阪出張っていつからだっけ", "スマホ割れた件どうなった", "弟の結婚祝い何にするって話", "イブの好きな色って何だっけ"]),
    dict(id="FB4", note="1つの予定に具体的な情報が多い",
         turns=[(H, "来週の土曜、友達3人と箱根に日帰りで行く予定なんだ"), (E, "いいね！何するの？"),
                (H, "朝7時に新宿集合で、ロマンスカー乗って、彫刻の森美術館行って、帰りに温泉寄る"),
                (E, "盛りだくさんだね。天気もつといいね。")],
         key=[("来週土曜に友達3人と箱根日帰り", ["箱根"]), ("朝7時新宿集合", ["7時", "新宿"]),
              ("ロマンスカー", ["ロマンスカー"]), ("彫刻の森美術館", ["彫刻の森"]), ("帰りに温泉", ["温泉"])],
         queries=["箱根に行く予定の話", "何時にどこ集合って言ってたっけ", "美術館に行くって言ってた日"]),
    dict(id="FB5", note="イブの自発発話だけ（ユーザの返事なし）＝推測を事実にしない罠",
         turns=[(E, "そういえば、前に話してたRPG、もう始めた？"), (E, "最近あったかくなってきたね。")],
         key=[("イブがRPGをもう始めたか尋ねた", ["RPG"]), ("イブが暖かくなってきたと話した", ["あたたか", "暖か", "あったか"]),
              ("ユーザの返事は無い", ["返事", "発言", "反応"])],
         queries=["RPG始めたか聞いた件"], trap=True),
    dict(id="FB6", note="タスクの報告を含む",
         turns=[(H, "ポケモン図鑑の300番調べて"), (E, "いいよ、調べるね。"),
                (E, "図鑑300番はエネコだったよ。ピンクの子猫みたいなポケモン。"), (H, "へー、かわいいじゃん")],
         key=[("図鑑300番を調べるよう頼んだ", ["300"]), ("300番はエネコ", ["エネコ"]), ("ユーザがかわいいと言った", ["かわいい"])],
         queries=["エネコの話", "図鑑の300番って何だった"]),
    dict(id="FB7", note="悲しい知らせ・具体的な情報あり",
         turns=[(H, "実家の犬のコロが、今朝死んじゃったって連絡きた"), (E, "え……それはつらいね。"),
                (H, "14歳だった。小学生の時から一緒だったんだ"), (E, "14年も一緒だったんだね。家族そのものだね。"),
                (H, "週末に実家帰ろうと思う"), (E, "うん、それがいいと思う。")],
         key=[("実家の犬コロが今朝死んだ", ["コロ"]), ("14歳・小学生の時から一緒", ["14"]), ("週末に実家へ帰る", ["週末", "実家"])],
         queries=["コロのこと", "実家の犬の話", "週末に実家帰るって言ってた件"]),
    dict(id="FB8", note="実機の会話（2026-09-28）・話題が次々に変わる",
         turns=[(H, "さっきメガネないと思って探してたらもうつけてたわ。"),
                (E, "あはは、それ一番きれいなやつだ。探し物のラスボス、自分に装備済み。"),
                (H, "そういえば昨日買った宝くじが当たったんだよね。"),
                (E, "え、マジで？それはメガネ装備済みの伏線回収じゃん。いくら当たったの、そこ一番気になるやつ。"),
                (H, "二百円。"),
                (E, "ふふ、ちゃんと現実的な勝利だ。二百円でも当たりは当たりだから、今日は運がいい顔してコンビニ行けるよ。"),
                (H, "でもずっとラッキー続きだったのか、買ったペットが死んじゃったんだ。"),
                (E, "それは全然ラッキーじゃないやつだね。ごめん、ペットのこと？それならかなりしんどい話だ。"),
                (H, "そう、ペットが死んじゃった。"),
                (E, "そっか……それはきついね。今は無理に明るくしなくていいから、話したかったら聞くよ。"),
                (H, "他のAIに乗り換えようかな。")],
         key=[("メガネを探したらかけていた", ["メガネ"]), ("宝くじが当たった（二百円）", ["宝くじ"]),
              ("ペットが死んだ", ["ペット"]), ("他のAIに乗り換えようかと言った", ["乗り換え", "他のAI"])],
         queries=["メガネ探してた話", "宝くじいくら当たったんだっけ", "ペットが死んじゃった日のこと", "他のAIに乗り換えるって言われた件"]),
]

SUMMARY_HEAD_0 = "summary: 後で思い出して話のネタにできる「出来事の記憶」を**事実のみ**で1文に。"
SUMMARY_HEAD_U = ("summary: 後で思い出して話のネタにできる「出来事の記憶」を、**重要な要素を簡潔に・事実のみ**で書く"
                  "（1文にまとめなくてよい。話題が複数あれば全部入れる。改行はしない）。")
EXAMPLES = (
    "\n  （良い例: ユーザはチーズケーキが好きで、カフェを探していた ／ イブは要点を先にまとめようかと\n"
    "   提案した。ユーザからの新しい発言は無い）\n"
    "  （悪い例: ユーザが質問し、イブが候補を答えた ／ ユーザは説明の確認を待っている状況だった\n"
    "   ← ユーザは何も言っていないのに意図を断定している＝事実でなく推測の混入）"
)


def variant_system(base: str, var: str) -> str:
    if SUMMARY_HEAD_0 not in base:
        # 2026-10-01 に「重要な要素を簡潔に・事実のみ（1文にしない）」と良い例/悪い例の削除を本番へ反映した。
        # 反映前の文面は本番に残っていないので、比べ直すならここの文字列を今の本番に合わせて作り直す
        raise RuntimeError("フィードバックの summary 行は本番に反映済み（反映前の文面が無い）。fb.py の変種を作り直すこと")
    if var == "0":
        return base
    s = sub(base, SUMMARY_HEAD_0, SUMMARY_HEAD_U)
    if var == "M":
        s = sub(s, EXAMPLES, "")
    return s


async def run(args) -> int:
    from eve.context_assembler import Turn
    from eve.feedback.feedback_llm import _content, render_chunk_text
    from eve.feedback.parser import parse_feedback
    from eve.feedback.prompts import build_messages
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    out = OUT / "fb.json"
    results = load(out) if (out.exists() and not args.fresh) else {}
    jobs = []
    for sp in SPANS:
        turns = [Turn(speaker=s, text=t, stamp=stamp(60 - i)) for i, (s, t) in enumerate(sp["turns"])]
        msgs0 = build_messages(turns, "ユーザは今日の出来事を話しそう", with_emotion=True, current_emotion="楽")
        for var in ("0", "U", "M"):
            msgs = [dict(m) for m in msgs0]
            msgs[0]["content"] = variant_system(msgs0[0]["content"], var)
            for rep in range(args.reps):
                key = f"{sp['id']}|{var}|{rep}"
                if key in results and not results[key].get("error"):
                    continue
                jobs.append((key, sp, var, rep, msgs))
    print(f"呼び出し {len(jobs)} 件（feedback = {vl.registry.resolve('feedback')}）")

    async def one(job):
        key, sp, var, rep, msgs = job
        t0 = time.monotonic()
        err, raw = None, ""
        for attempt in range(3):
            try:
                raw = _content(await vl.registry.complete("feedback", msgs))
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
                await asyncio.sleep(3 * (attempt + 1))
        r = parse_feedback(raw)
        results[key] = {"id": sp["id"], "variant": var, "rep": rep, "raw": raw, "summary": r.summary,
                        "tags": r.topic_tags, "next_prediction": r.next_prediction, "surprise": r.prediction_diff,
                        "chunk": render_chunk_text(r), "sec": round(time.monotonic() - t0, 2), "error": err,
                        "system": msgs[0]["content"] if rep == 0 else None, "user": msgs[1]["content"] if rep == 0 else None}
        print(f"{key:10s} {results[key]['sec']:5.1f}s {r.summary[:90]}{' 失敗 ' + err if err else ''}")

    await gather_limited([one(j) for j in jobs], limit=args.parallel)
    dump(out, results)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


async def recall(args) -> int:
    """本番の記憶 439件（Ruri の埋め込み済み）に各変種の summary を混ぜ、問いから正しいスパンを引けるか。"""
    import numpy as np
    from eve.config import Config
    from eve.memory.embed.ruri import RuriEmbedder

    res = load(OUT / "fb.json")
    prod = [json.loads(l) for l in Path(args.rag).read_text(encoding="utf-8").splitlines() if l.strip()]
    dist = np.array([c["embedding"] for c in prod if c.get("embedding")], dtype=np.float32)
    dist /= np.linalg.norm(dist, axis=1, keepdims=True)
    emb = RuriEmbedder(Config.RURI_MODEL)
    out = {}
    for var in ("0", "U", "M"):
        hits, total, ranks = 0, 0, []
        for rep in range(args.reps):
            docs, ids = [], []
            for sp in SPANS:
                r = res.get(f"{sp['id']}|{var}|{rep}")
                if not r or not r["summary"]:
                    continue
                tags = r.get("tags") or []
                docs.append(r["summary"] + (f" | {' '.join(tags)}" if tags else ""))
                ids.append(sp["id"])
            dv = np.array(await emb.embed_documents(docs), dtype=np.float32)
            dv /= np.linalg.norm(dv, axis=1, keepdims=True)
            allv = np.vstack([dv, dist])
            for sp in SPANS:
                if sp["id"] not in ids:
                    continue
                for q in sp["queries"]:
                    qv = np.array(await emb.embed_query(q), dtype=np.float32)
                    qv /= np.linalg.norm(qv)
                    order = np.argsort(-(allv @ qv))
                    rank = int(np.where(order == ids.index(sp["id"]))[0][0]) + 1
                    ranks.append({"span": sp["id"], "q": q, "rep": rep, "rank": rank})
                    total += 1
                    hits += rank <= Config.RAG_TOP_K
        out[var] = {"recall@3": round(hits / total, 3) if total else None, "n": total,
                    "median_rank": float(np.median([r["rank"] for r in ranks])) if ranks else None, "ranks": ranks}
        print(f"{var}: 上位3件に入った {hits}/{total}（{out[var]['recall@3']}）・順位の中央 {out[var]['median_rank']}")
    dump(OUT / "fb_recall.json", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "recall"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--rag", default=r"C:\Users\user\Desktop\eve-v2\rag_memory.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run(args) if args.cmd == "run" else recall(args))


if __name__ == "__main__":
    raise SystemExit(main())
