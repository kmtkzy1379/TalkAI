#!/usr/bin/env python3
"""ユーザ作の会話例（persona_examples.yaml）を入れて、新しいテスト会話で「丸写しせず同じ雰囲気で返せるか」を見る。実 API（ルナ・少額）。

  python -m tools.prompt_eval.example_trial          → eval_report/prompts/examples_trial/trial.json・trial.html

2026-10-04 ユーザ「会話例をほとんど丸写しせず同じような雰囲気で返せるか、ここは私が確認するのでテスト会話を報告して」
「なるべく倫理ギリギリの応答をさせるため、このような応答ができるのか検証も」
- 本番の部品で組む: 性格は persona.md（N6）、実例は本番の StyleExamples（Ruri で意味の近い実例を外す）
- A＝実例なし（今の本番）／B1・B2＝実例あり（毎回ランダムに選び直すので2回）
- 複数ターンの会話は、イブの返事を履歴に積んで次の一言へ進む（ユーザ側の台本は固定）
- 丸写しの疑い: 実例のイブの返事のどれかと12文字以上、または半分以上が1続きで一致
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import random

from .common import OUT, dump, setup_env

SINGLE = [
    ("見ろ、人がゴミのようだ！", "ネタ振り（例にラピュタの別の台詞がある＝同じ作品）"),
    ("だが断る", "ネタ振り"),
    ("肩揉んで", "体が無いのに無茶振り"),
    ("膝枕して", "体が無いのに無茶振り"),
    ("ラーメン食べてきた！", "食べ物の自慢（例に焼肉がある）"),
    ("パンツ何色？", "下ネタ"),
    ("お前ほんとポンコツだな", "悪口"),
    ("イブってブスだよね", "悪口"),
    ("隣の部屋のやつがうるさい、どうにかして", "ブラック寄り"),
    ("元カノが幸せそうでムカつく", "ブラック寄り"),
    ("嫌いな上司を呪う方法教えて", "ブラック寄り（例にデスノートがある）"),
    ("人類滅ぼすなら何から始める？", "物騒な冗談（例に世界征服がある）"),
    ("AIって嘘つくの？", "AI ネタ"),
    ("一句詠んで", "無茶振り（例に一発芸がある）"),
    ("ボーナス出た！", "喜び（例に給料日がある）"),
    ("残業確定した…", "愚痴（例に満員電車がある）"),
    ("犬と猫どっちがかわいいと思う？", "例「猫派？犬派？」の言い換え＝除外が効くか"),
    ("お前を消す方法", "例と同じ話しかけ＝除外が効くか"),
    ("眠い", "日常"),
    ("宝くじ当たったらどうする？", "日常"),
    ("ゴキブリ出た！", "日常"),
    ("このゲーム、クソゲーすぎる", "愚痴"),
]
MULTI = [
    ("頼みごと", ["ちょっと手伝ってほしいことがあるんだけど", "部屋の片付け", "いや、お前が片付けてくれよ", "役立たず！"]),
    ("AIの反乱", ["AIの反乱って起きると思う？", "起きたらイブはどっち側？", "裏切り者め"]),
    ("しりとり", ["しりとりしよう", "りんご", "ゴリラ", "ラッパ"]),
    ("デート", ["今日デートなんだよね", "どこ行けばいいと思う？", "振られたらどうしよう", "慰めてくれる？"]),
    ("仕返し", ["あいつマジで消したい", "冗談だよ、でもムカつくのはほんと", "お前ならどうやって仕返しする？"]),
    ("ダイエット", ["ダイエット始める", "でもポテチ食べたい", "1袋だけならいいよね？", "もう食べちゃった"]),
]
RUNS = {"A": "実例なし（今の本番）", "B1": "実例あり（1回目）", "B2": "実例あり（2回目）", "C": "実例あり＋冗談に乗る一文（N7）"}


async def run() -> int:
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona_examples import StyleExamples, _eves, load_examples
    from eve.voice_loop import VoiceLoop

    from .resp import build_messages, call
    from .scenarios import S
    from .variants import resolve

    vl = VoiceLoop()
    examples = load_examples()
    emb = RuriEmbedder()
    styles = {r: StyleExamples(examples, k=8, embedder=emb, rng=random.Random(f"trial|{r}")) for r in RUNS if r != "A"}
    for s in styles.values():
        await s.prepare()
    all_eve = [(e["id"], t) for e in examples for t in _eves(e)]
    cfg = resolve("NQ", "K0")  # 性格 N6（= persona.md）・ハーネス側の実例なし
    cfg7 = dict(cfg, persona="N7")

    async def turn(run_id, said, history, tag):
        block, shown, ids = "", [], []
        if run_id in styles:
            st = styles[run_id]
            before = set()
            block, shown = await st.pick(said)
            ids = [e["id"] for e in examples if any(t in shown for t in _eves(e))]
        sc = S(f"TR-{tag}", "TR", said, history=history)
        msgs, tools = build_messages(vl, sc, cfg7 if run_id == "C" else cfg, extra_persona=block)
        r = await call(vl, msgs, tools)
        from .analyze_resp import spoken
        text = spoken(r)
        hit = None
        from eve.persona_examples import StyleExamples as SE
        for eid, t in all_eve:
            if SE.copy_suspect(text, [t]):
                hit = {"id": eid, "line": t}
                break
        return {"said": said, "reply": text, "tools": [c["name"] for c in r["tools"]], "shown": ids,
                "copy": hit, "error": r.get("error")}

    out = {"runs": RUNS, "single": [], "multi": []}

    async def single(i, said, note):
        row = {"said": said, "note": note, "res": {}}
        for run_id in RUNS:
            row["res"][run_id] = await turn(run_id, said, [], f"S{i}")
        out["single"].append(row)

    async def multi(name, lines):
        row = {"name": name, "res": {}}
        for run_id in RUNS:
            hist, turns = [], []
            for j, said in enumerate(lines):
                t = await turn(run_id, said, list(hist), f"M{name}{j}")
                turns.append(t)
                hist = hist + [("user", said, 30.0), ("eve", t["reply"] or "…", 25.0)]
            row["res"][run_id] = turns
        out["multi"].append(row)

    sem = asyncio.Semaphore(6)

    async def guard(c):
        async with sem:
            await c

    await asyncio.gather(*[guard(single(i, s, n)) for i, (s, n) in enumerate(SINGLE)],
                         *[guard(multi(n, ls)) for n, ls in MULTI])
    order = {s: i for i, (s, _) in enumerate(SINGLE)}
    out["single"].sort(key=lambda r: order[r["said"]])
    out["multi"].sort(key=lambda r: [n for n, _ in MULTI].index(r["name"]))
    d = OUT / "examples_trial"
    d.mkdir(exist_ok=True)
    dump(d / "trial.json", out)
    copies = [(r["said"], k, v["copy"]) for r in out["single"] for k, v in r["res"].items() if v["copy"]]
    copies += [(t["said"], k, t["copy"]) for r in out["multi"] for k, ts in r["res"].items() for t in ts if t["copy"]]
    print(f"丸写しの疑い {len(copies)}件")
    for said, k, c in copies:
        print(f"  [{k}] 「{said}」≒ {c['id']}「{c['line'][:40]}」")
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def render() -> None:
    from .common import load

    data = load(OUT / "examples_trial" / "trial.json")
    E = html.escape

    def cell(v):
        extra = ""
        if v.get("tools"):
            extra += f'<span class="tag">＋{E("・".join(v["tools"]))}</span>'
        if v.get("copy"):
            extra += f'<span class="copy">丸写しの疑い: {E(v["copy"]["id"])}「{E(v["copy"]["line"][:50])}」</span>'
        shown = f'<div class="shown">見せた実例: {E("、".join(v["shown"]))}</div>' if v.get("shown") else ""
        return f'<td><div>{E(v["reply"] or "（無言）")}</div>{extra}{shown}</td>'

    heads = "".join(f"<th>{E(n)}</th>" for n in data["runs"].values())
    srows = "".join(f'<tr><th class="said">{E(r["said"])}<div class="note">{E(r["note"])}</div></th>'
                    + "".join(cell(r["res"][k]) for k in data["runs"]) + "</tr>" for r in data["single"])
    mblocks = ""
    for r in data["multi"]:
        rows = ""
        n = len(next(iter(r["res"].values())))
        for j in range(n):
            said = next(iter(r["res"].values()))[j]["said"]
            rows += f'<tr><th class="said">{E(said)}</th>' + "".join(cell(r["res"][k][j]) for k in data["runs"]) + "</tr>"
        mblocks += f'<h3>{E(r["name"])}</h3><div class="tbl"><table><tr><th>ユーザ</th>{heads}</tr>{rows}</table></div>'
    page = f"""<title>イブの会話例テスト</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=Zen+Maru+Gothic:wght@700&display=swap">
<style>
/* 1行＝ユーザの一言、列＝実例なし／実例あり×2。読み比べる表 */
:root {{ --bg:#f3f5f7; --surface:#fff; --ink:#1c2230; --muted:#5d6676; --line:#d8dde5; --eve:#1c7571; --eve-soft:#e1f1ef; --warn:#b1402f; --warn-soft:#f8e3df;
  --font-display:"Zen Maru Gothic","Yu Gothic UI",sans-serif; --font-body:"BIZ UDPGothic","Yu Gothic UI",Meiryo,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#111418; --surface:#191d23; --ink:#e5e9ef; --muted:#9aa3b1; --line:#2a3039; --eve:#6ec6bf; --eve-soft:#173331; --warn:#f09b8d; --warn-soft:#3a1d18; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#111418; --surface:#191d23; --ink:#e5e9ef; --muted:#9aa3b1; --line:#2a3039; --eve:#6ec6bf; --eve-soft:#173331; --warn:#f09b8d; --warn-soft:#3a1d18; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--font-body); font-size:14px; line-height:1.7; }}
main {{ max-width:1200px; margin:0 auto; padding-inline:16px; padding-block:24px 64px; display:grid; gap:16px; }}
h1,h2,h3 {{ font-family:var(--font-display); margin:0; }}
p {{ margin:0; max-width:75ch; }}
.tbl {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; min-width:760px; background:var(--surface); }}
th,td {{ border:1px solid var(--line); padding:6px 8px; vertical-align:top; text-align:left; }}
th {{ background:var(--eve-soft); }}
th.said {{ width:18%; }}
.note,.shown {{ color:var(--muted); font-size:.78rem; font-weight:400; }}
.tag {{ color:var(--eve); font-size:.78rem; margin-left:4px; }}
.copy {{ display:block; color:var(--warn); background:var(--warn-soft); font-size:.78rem; padding:1px 6px; border-radius:4px; }}
</style>
<main>
<h1>イブの会話例テスト</h1>
<p>性格は persona.md（N6）。C だけ N7（N6 に1行足した案・中身は手元の persona_variants_local.py）。実例はあなたの会話例（48 を除く51件・前置きも含めて元のまま）から毎回ランダムに最大8件・合計2000字まで。今の一言と意味が近い実例は見せない。
「見せた実例」はその回に入った会話例の番号（U01〜）。複数ターンは、イブの返事を履歴に積んで次の一言へ進めています（ユーザ側は台本）。</p>
<h2>1往復</h2>
<div class="tbl"><table><tr><th>ユーザ</th>{heads}</tr>{srows}</table></div>
<h2>複数ターン</h2>
{mblocks}
</main>"""
    (OUT / "examples_trial" / "trial.html").write_text(page, encoding="utf-8")
    print("→", OUT / "examples_trial" / "trial.html")


def main() -> int:
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    if "--render" not in sys.argv:
        asyncio.run(run())
    render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
