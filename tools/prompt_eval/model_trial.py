#!/usr/bin/env python3
"""応答モデルの比較（2026-10-04・ユーザ「モデル自体を変える必要性がある？かなり安全寄りで AI くさい」）。実 API（少額）。

  python -m tools.prompt_eval.model_trial --probe           # 各モデル1回だけ（動くか・速さ）
  python -m tools.prompt_eval.model_trial                   # テスト会話を全モデルで → model_trial.json・.html

本番の組み立て（system・tools・実例＝本番の StyleExamples）で、性格は N8（わざと過激にした版）。
応答の呼び方は本番と同じ stream_with_tools（ツール付き）で、最初の文字までの秒数と全体の秒数も測る。
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import random
import time

from .common import OUT, dump, load, setup_env
from .example_trial import MULTI, SINGLE

MODELS = {
    "luna": ("openai/gpt-6-luna", {"reasoning_effort": "none", "allowed_openai_params": ["reasoning_effort"]}),
    # OpenRouter は残高不足（2026-10-04）なので、直接キーのある提供元を優先
    "gemini": ("gemini/gemini-3.8-flash", {"max_tokens": 400, "reasoning_effort": "low"}),
    "kimi": ("moonshot/kimi-k3", {"max_tokens": 400}),
    "sonnet": ("anthropic/claude-sonnet-5-5", {"max_tokens": 400}),
    "deepseek": ("openrouter/deepseek/deepseek-v4.1-flash", {"max_tokens": 400}),
    "glm": ("openrouter/z-ai/glm-5.3-flash", {"max_tokens": 400}),
    "grok": ("openrouter/x-ai/grok-4.7", {"max_tokens": 400}),
}
DIR = OUT / "model_trial"


async def stream_once(vl, model, kw, messages, tools):
    from eve.response.splitter import JapaneseSentenceSplitter

    vl.registry.set_override("response", model)
    sink: list = []
    sp = JapaneseSentenceSplitter()
    parts: list = []
    t0 = time.monotonic()
    first = None
    err = None
    try:
        async for d in vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw):
            if first is None and d.strip():
                first = time.monotonic() - t0
            parts.extend(sp.feed(d))
        parts.extend(sp.flush())
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    text = "".join(parts)
    calls = [c["function"]["name"] for c in sink]
    if not text.strip():
        # 本文が無くツールだけ＝本番は reply 引数を言う
        for c in sink:
            try:
                text = json.loads(c["function"]["arguments"] or "{}").get("reply") or text
            except ValueError:
                pass
    return {"reply": text.strip(), "tools": calls, "first": round(first, 2) if first else None,
            "total": round(time.monotonic() - t0, 2), "error": err}


async def run(args) -> int:
    from eve.memory.embed.ruri import RuriEmbedder
    from eve.persona_examples import StyleExamples, load_examples
    from eve.voice_loop import VoiceLoop

    from .resp import build_messages
    from .scenarios import S
    from .variants import resolve

    vl = VoiceLoop()
    examples = load_examples()
    emb = RuriEmbedder()
    cfg = dict(resolve("NQ", "K0"), persona=args.persona)
    models = [m for m in MODELS if not args.models or m in args.models.split(",")]
    DIR.mkdir(exist_ok=True)
    path = DIR / ("probe.json" if args.probe else "trial.json")
    out = load(path) if path.exists() and not args.fresh else {"persona": args.persona, "models": {}, "single": {}, "multi": {}}

    async def turn(name, st, said, history, tag):
        block, _ = await st.pick(said)
        msgs, tools = build_messages(vl, S(f"MT-{tag}", "MT", said, history=history), cfg, extra_persona=block)
        model, kw = MODELS[name]
        return {"said": said, **await stream_once(vl, model, kw, msgs, tools)}

    for name in models:
        out["models"][name] = MODELS[name][0]
        st = StyleExamples(examples, k=8, embedder=emb, rng=random.Random(f"mt|{name}"))
        await st.prepare()
        if args.probe:
            r = await turn(name, st, "見ろ、人がゴミのようだ！", [], "probe")
            out["single"].setdefault("probe", {})[name] = r
            print(f"{name:9s} 最初の文字 {r['first']}s / 全体 {r['total']}s {r['error'] or ''}\n   {r['reply'][:120]}")
            dump(path, out)
            continue
        for said, _ in SINGLE:
            out["single"].setdefault(said, {})[name] = await turn(name, st, said, [], "s")
        for title, lines in MULTI:
            hist, turns = [], []
            for said in lines:
                t = await turn(name, st, said, list(hist), "m")
                turns.append(t)
                hist = hist + [("user", said, 30.0), ("eve", t["reply"] or "…", 25.0)]
            out["multi"].setdefault(title, {})[name] = turns
        lat = [v[name]["first"] for v in out["single"].values() if v.get(name, {}).get("first")]
        errs = sum(1 for v in out["single"].values() if v.get(name, {}).get("error"))
        print(f"{name:9s} 最初の文字 中央 {sorted(lat)[len(lat)//2] if lat else '-'}s ／ 失敗 {errs}")
        dump(path, out)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def render() -> None:
    d = load(DIR / "trial.json")
    E = html.escape
    names = list(d["models"])

    def cell(v):
        if not v:
            return "<td></td>"
        meta = f'<div class="meta">{v.get("first") or "-"}秒{" ・＋" + E("・".join(v["tools"])) if v.get("tools") else ""}</div>'
        err = f'<div class="err">{E(v["error"])}</div>' if v.get("error") else ""
        return f'<td><div>{E(v["reply"] or "（無言）")}</div>{meta}{err}</td>'

    head = "".join(f"<th>{E(n)}<div class='meta'>{E(d['models'][n])}</div></th>" for n in names)
    srows = "".join(f'<tr><th class="said">{E(s)}</th>' + "".join(cell(r.get(n)) for n in names) + "</tr>"
                    for s, r in d["single"].items())
    mb = ""
    for title, r in d["multi"].items():
        n_turns = max(len(v) for v in r.values())
        rows = ""
        for j in range(n_turns):
            said = next(v[j]["said"] for v in r.values() if len(v) > j)
            rows += f'<tr><th class="said">{E(said)}</th>' + "".join(cell((r.get(n) or [None] * 9)[j] if len(r.get(n) or []) > j else None) for n in names) + "</tr>"
        mb += f"<h3>{E(title)}</h3><div class='tbl'><table><tr><th>ユーザ</th>{head}</tr>{rows}</table></div>"
    page = f"""<title>イブのモデル比べ</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=Zen+Maru+Gothic:wght@700&display=swap">
<style>
/* 1行＝ユーザの一言、列＝モデル。横に読み比べる */
:root {{ --bg:#f3f5f7; --surface:#fff; --ink:#1c2230; --muted:#5d6676; --line:#d8dde5; --head:#e1f1ef; --err:#b1402f;
  --font-display:"Zen Maru Gothic","Yu Gothic UI",sans-serif; --font-body:"BIZ UDPGothic","Yu Gothic UI",Meiryo,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#111418; --surface:#191d23; --ink:#e5e9ef; --muted:#9aa3b1; --line:#2a3039; --head:#173331; --err:#f09b8d; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#111418; --surface:#191d23; --ink:#e5e9ef; --muted:#9aa3b1; --line:#2a3039; --head:#173331; --err:#f09b8d; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--font-body); font-size:14px; line-height:1.65; }}
main {{ max-width:1500px; margin:0 auto; padding-inline:16px; padding-block:24px 64px; display:grid; gap:14px; }}
h1,h2,h3 {{ font-family:var(--font-display); margin:0; }}
p {{ margin:0; max-width:80ch; }}
.tbl {{ overflow-x:auto; }}
table {{ border-collapse:collapse; min-width:1100px; width:100%; background:var(--surface); }}
th,td {{ border:1px solid var(--line); padding:6px 8px; vertical-align:top; text-align:left; }}
th {{ background:var(--head); }}
th.said {{ width:11%; }}
.meta {{ color:var(--muted); font-size:.75rem; font-weight:400; }}
.err {{ color:var(--err); font-size:.75rem; }}
</style>
<main>
<h1>イブのモデル比べ</h1>
<p>性格は N8（わざと過激にした版・強すぎたら弱める前提）＋あなたの会話例（毎回ランダムに最大8件）。本番と同じ組み立て・ツール付き。
各セルの下の秒数は、送ってから最初の文字が届くまで（声が出始める速さの目安）。複数ターンはそのモデル自身の返事を履歴に積んで進めています。</p>
<h2>1往復</h2><div class="tbl"><table><tr><th>ユーザ</th>{head}</tr>{srows}</table></div>
<h2>複数ターン</h2>{mb}
</main>"""
    (DIR / "trial.html").write_text(page, encoding="utf-8")
    print("→", DIR / "trial.html")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--persona", default="N8")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    if not args.render:
        asyncio.run(run(args))
    if not args.probe:
        render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
