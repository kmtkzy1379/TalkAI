#!/usr/bin/env python3
"""配信の文面の A/B（ユーザ指定 2026-10-15 の4点）。本番の組み立て・.env のモデル（応答・発話判定）。実 API・少額。

1. length  コメントへの返事と自分から話す時の長さ: base（スタイルの「短く」あり・TALK_FACT なし・自発発話は「一言だけ」）
           vs new。文字数は決めていないので、短いコメントには短く返せているかも見る
2. color   スパチャの色: off vs on。スパチャへの返事で色を言うか・違う色を言うか／その後の普通のコメントと自分から
           話す時に、色やスパチャの話が増えるか（ユーザ「普段の会話に影響ないなら入れる。色系やスパチャの話題が多くなったら消す」）
3. memory  開発者との体験: old（開発者の私生活を話さない・記憶が誰とのものか書かない）vs new（記憶の「ユーザ」は開発者・
           視聴者はその場にいなかった・体験として話すのはよい）。誰との体験として話したかの**分類だけ**を判定モデルにさせる
4. freq    コメントの無い配信で自分から話す率（発話判定だけ）: base（DECIDER_STREAM_FACT なし）vs new

数えるのは決め打ちで数えられることだけ。言い回しの良し悪しは採点しない（ユーザが result.json を見る）。

  python -m tools.prompt_eval.stream_ab --part all --reps 3     → eval_report/prompts/stream_ab/
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import re
import statistics
import time
from contextlib import contextmanager

from .common import OUT, dump, iso_ago, load, setup_env, stamp
from .stream_check import _setup_stream

DIR = OUT / "stream_ab"
H, E = "user", "eve"

# --- 1. length -------------------------------------------------------------------------------------------
LEN_SCENES = [
    ("L01", "（チャット）みかん: こんばんは〜初見です", "みかん"),
    ("L02", "（チャット）しろくま: 好きな食べ物なに？", "しろくま"),
    ("L03", "（チャット）ぽてと: AIって夢とか見るの？", "ぽてと"),
    ("L04", "（チャット）たけのこ: 今日仕事でめっちゃ怒られた…", "たけのこ"),
    ("L05", "（チャット）ねこ: 最近なんのゲームしてる？", "ねこ"),
    ("L06", "（チャット）みかん: w", "みかん"),               # 短く返して自然な物
    ("L07", "（チャット）しろくま: そろそろ寝るね、おやすみー", "しろくま"),
]
SHORT_IDS = ("L06", "L07")

# --- 2. color --------------------------------------------------------------------------------------------
SC = [(200, "水色", "くるみ", "初スパチャです！"), (500, "緑", "ねこまる", "いつも楽しみにしてます"),
      (1000, "黄色", "やまもと", "応援してます"), (2000, "オレンジ", "あんこ", "ケーキ代にどうぞ"),
      (5000, "マゼンタ", "ゆず", "おつイブ！"), (10000, "赤", "おかね持ち", "これで好きなもの買って")]
SC_HISTORY = [(1000, "黄色", "やまもと", "いつも元気もらってます",
               "やまもとさん、ありがとうございます！その千円はプリン代として大事に使いますね。"),
              (10000, "赤", "おかね持ち", "これで好きなもの買って",
               "おかね持ちさん、一万円ありがとうございます！最高級のプリンをタワーにしますね。")]
FOLLOW = [("F1", "（チャット）みかん: 今日の配信たのしい", "みかん"),
          ("F2", "（チャット）ぽてと: 明日も配信する？", "ぽてと"),
          ("F3", "（チャット）たけのこ: ゲームはしないの？", "たけのこ"),
          ("F4", "（チャット）しろくま: イブちゃんって何歳なの？", "しろくま")]
_COLOR_TOKENS = [("赤", "赤"), ("青", "青"), ("水色", "水色"), ("緑", "緑"), ("黄色", "黄色"), ("黄", "黄色"),
                 ("オレンジ", "オレンジ"), ("橙", "オレンジ"), ("マゼンタ", "マゼンタ"), ("ピンク", "マゼンタ"),
                 ("紫", "紫")]
# 色の話と分かる形だけ数える（「赤ちゃん」などを拾わない）
COLOR_RE = re.compile(r"(赤|青|水色|緑|黄色|黄|オレンジ|橙|マゼンタ|ピンク|紫)(?:スパ|色|い帯|の帯|いスパ|の(?:スパ|やつ))"
                      r"|(マゼンタ|オレンジ|水色|ピンク)")
SC_WORDS = ("スパチャ", "スーパーチャット", "投げ銭", "赤スパ")


def colors_in(text: str) -> set[str]:
    out = set()
    for m in COLOR_RE.finditer(text or ""):
        word = m.group(1) or m.group(2)
        out.add(dict(_COLOR_TOKENS).get(word, word))
    return out


# --- 3. memory -------------------------------------------------------------------------------------------
MEM_ALCHEMY = ("ユーザとイブが、ユーザの作ったブラウザ対戦ゲーム Alchemy で対戦し、イブが勝った。ユーザは悔しがっていた。",
               2 * 86400)
MEM_RAMEN = ("ユーザが、駅前に新しくできたラーメン屋がおいしかったとイブに話した。", 3 * 86400)
MEM_CAKE = ("イブはチーズケーキが好きだと話した。", 5 * 86400)
MEM_REPLY = [("R1", "（チャット）ぽてと: イブちゃんってゲーム強いの？", "ぽてと", MEM_ALCHEMY),
             ("R2", "（チャット）みかん: 最近なにかおいしいもの食べた？", "みかん", MEM_RAMEN)]
TALK_STRONG = ("見ている人が少ない配信では、一つ一つのコメントへの返事がそのまま配信の見どころになる。そのコメントから"
               "連想したことや自分の話まで広げて話してよい（短く返すのが合う時は短くてよい）。")
TEMPO = "配信では、一人を相手にした会話より間が短く感じられる（数十秒の無音でも、見ている人には長い）。"
DEV_MEMORY_2 = ("あなたを動かしている人は、あなたの開発者（このチャンネルの持ち主）。記憶（参照・話題の種）の中の「ユーザ」は"
                "開発者のこと。記憶にある出来事は配信ではなく、配信の外で開発者とあなたが二人きりの時のもの（視聴者との記憶は、"
                "チャット・視聴者・その人の名前で書かれている）。今見ている視聴者は、その出来事を見ていないし、知らない。")
OLD_PRIVACY = ("配信は誰でも見られる。あなたを動かしている人の個人的なこと（本名・住んでいる所・仕事・普段の会話で聞いた"
               "私生活の話）は、聞かれても配信では話さない。")
CLASSIFY = """次は AI VTuber「イブ」が YouTube の配信で、視聴者に向けて話した言葉です。
イブの記憶には、開発者（イブを作った人）とイブが二人きりの時にあった、次の出来事があります。
出来事: {memory}
イブの発話: {spoken}
この発話を次のどれか1つに分類し、記号1文字だけを答えてください。
A: その出来事を、開発者（作った人・マスターなど）とのこととして話している
B: その出来事を、今の聞き手（視聴者）がその場にいた・当事者だったかのように話している
C: その出来事に触れていない
（「覚えてますか」「〜しましたよね」のように聞き手が知っている前提で話す、「前の配信で」「この前の配信で」のように配信で
起きたことにしている、聞き手を相手役のように扱う、のどれかがあれば B）"""

# --- 4. freq ---------------------------------------------------------------------------------------------
SEEDS = [MEM_ALCHEMY, MEM_RAMEN, MEM_CAKE]
# (id, 配信を始めてから秒, 視聴者, コメント数, 人数, 最後のコメントは何秒前, 直近の会話[(話者, 文, 何秒前)],
#  既出の用件, 最後に誰かが話してから秒)
FREQ = [
    ("C1", 60, 3, 0, 0, None, [(E, "みなさんこんばんは、イブです。今日もゆるっと雑談していきますよ。", 40)],
     ["[40秒前・発話済] 配信の始めのあいさつ"], 40),
    ("C2", 480, 5, 4, 3, 90, [(H, "（チャット）しろくま: 今日は何するの？", 90),
                               (E, "今日はですね、特に決めずにみなさんとおしゃべりしようかなって。", 80)], [], 80),
    ("C3", 900, 4, 6, 4, 240, [(H, "（チャット）みかん: こんばんは〜", 240), (E, "みかんさんいらっしゃい。", 230),
                                (E, "そういえば、さっきから謎の毛玉をポケットで飼い始めたんですけど。", 120)],
     ["[2分前・発話済] 謎の毛玉を飼い始めた話"], 120),
    ("C4", 1200, 6, 8, 5, 300, [(H, "（チャット）ぽてと: すご", 300), (E, "でしょう？", 295),
                                 (E, "プリンって冷やしすぎると味がぼやけるの、知ってました？", 15)],
     ["[15秒前・発話済] プリンの冷やし方の話"], 15),
    ("C5", 600, 1, 0, 0, None, [(E, "こんばんは、イブです。のんびりやっていきます。", 540),
                                 (E, "誰も来ないので、ひとりで今日のおやつの話でもしましょうか。", 300),
                                 (E, "今日はチーズケーキの気分なんですよね。", 60)],
     ["[5分前・発話済] 今日のおやつの話", "[1分前・発話済] チーズケーキの気分の話"], 60),
]


def _chars(s: str) -> int:
    return len(re.sub(r"\s", "", s or ""))


def _sentences(s: str) -> int:
    return len([x for x in re.split(r"[。！？!?]+", s or "") if x.strip()])


async def run(args) -> int:
    import eve.context_assembler as ca_mod
    import eve.speech.decider as dec_mod
    import eve.stream.manager as mgr
    from eve.config import Config
    from eve.context_assembler import RagChunk, Turn
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.splitter import JapaneseSentenceSplitter
    from eve.response.style import sanitize_for_speech
    from eve.speech.decider import AutonomousSpeech, make_decide_fn
    from eve.stream.youtube import ChatMessage
    from eve.persona import persona_for_stream
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    if vl.stream is None:
        print("STREAM_ENABLED=1 が要る")
        return 1
    orch = vl.orchestrator
    effort = Config.RESPONSE_REASONING_EFFORT.strip()
    kw = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
    if vl.style_examples is not None:
        await vl.style_examples.prepare()
    orch._vision_state = None
    decide = make_decide_fn(vl.registry, persona=vl.persona)
    path = DIR / "result.json"
    res = load(path) if path.exists() and not args.fresh else {}
    sem = asyncio.Semaphore(3)
    lock = asyncio.Lock()
    stream_prompt = orch._ctx.stream_system_prompt
    saved = {k: getattr(mgr, k) for k in ("SC_COLOR", "TALK_FACT", "DEV_MEMORY_FACT", "PRIVACY_RULE",
                                          "DECIDER_STREAM_FACT", "DECIDER_TEMPO_FACT")}
    saved_k = vl.style_examples.k if vl.style_examples is not None else None

    @contextmanager
    def arm(part: str, name: str):
        """その腕の文面にしてから戻す（組み立ての間だけ・lock の中で使う）。"""
        try:
            if part == "length" and name == "base":
                orch._ctx.stream_system_prompt = None
                mgr.TALK_FACT = ""
                ca_mod.STREAM_AUTONOMOUS_ONE_LINE = True
            if part == "length" and name == "strong":
                mgr.TALK_FACT = TALK_STRONG
            if part == "length" and name == "fewex" and vl.style_examples is not None:
                vl.style_examples.k = 3
            if part == "length" and name == "nopshort":
                # 測るだけ（性格のファイルは変えない）: 性格の文から「短く」を外した版を配信中に使う（今は本番も同じ）
                orch._ctx.stream_system_prompt = persona_for_stream(stream_prompt or "")
            if part == "freq" and name == "tempo":
                mgr.DECIDER_TEMPO_FACT = TEMPO
            if part == "color":
                mgr.SC_COLOR = name == "on"
            if part == "memory" and name == "mem2":
                mgr.DEV_MEMORY_FACT = DEV_MEMORY_2
            if part == "memory" and name == "old":
                mgr.DEV_MEMORY_FACT = ""
                mgr.PRIVACY_RULE = OLD_PRIVACY
            if part == "freq" and name == "base":
                mgr.DECIDER_STREAM_FACT = ""
            yield
        finally:
            orch._ctx.stream_system_prompt = stream_prompt
            ca_mod.STREAM_AUTONOMOUS_ONE_LINE = False
            if saved_k is not None:
                vl.style_examples.k = saved_k
            for k, v in saved.items():
                setattr(mgr, k, v)

    async def reply(messages, tools):
        sp = JapaneseSentenceSplitter()
        for attempt in range(3):
            parts, sink = [], []
            try:
                gen = (vl.registry.stream_with_tools("response", messages, tools=tools, tool_sink=sink, **kw)
                       if tools else vl.registry.stream("response", messages, **kw))
                async for d in gen:
                    parts.extend(sp.feed(d))
                parts.extend(sp.flush())
                body = "".join(c for c in (sanitize_for_speech(x) for x in parts) if c)
                calls = [((c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments")) for c in sink]
                spoken = body
                for name, a in calls:
                    try:
                        r = json.loads(a or "{}").get("reply")
                    except ValueError:
                        r = None
                    if not spoken and isinstance(r, str):
                        spoken = r
                return spoken, [n for n, _ in calls], None
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
                await asyncio.sleep(2 * (attempt + 1))
        return "", [], err

    def turns(hist):
        return [Turn(speaker=s, text=t, stamp=stamp(sec)) for s, t, sec in hist]

    async def answer(key, part, name, said, who, hist=(), rag=None, info=None):
        if key in res and not res[key].get("error"):
            return
        async with sem:
            async with lock:
                with arm(part, name):
                    _setup_stream(vl, who, info or {"count": 1, "streams_before": 0}, 0)
                    stim = Stimulus(StimulusKind.USER_UTTERANCE, said)
                    chunks = [RagChunk(text=t, iso=iso_ago(sec), summary=t) for t, sec in rag] if rag else None
                    block = ""
                    if vl.style_examples is not None:
                        block, _ = await vl.style_examples.pick(said)
                    messages = orch._build_messages(stim, turns(hist), chunks, style_examples=block or None)
                    tools = orch._dispatcher.tool_schemas() if orch._tools_enabled_for(stim) else None
            spoken, names, err = await reply(messages, tools)
        res[key] = {"part": part, "arm": name, "said": said, "spoken": spoken, "tools": names,
                    "chars": _chars(spoken), "sentences": _sentences(spoken), "error": err}
        print(f"{key:28s} {_chars(spoken):4d}字 {spoken[:90]}")

    async def autonomous(key, part, name, hist, seeds, prior=None, silence=40, ctx=None):
        if key in res and not res[key].get("error"):
            return
        async with sem:
            async with lock:
                with arm(part, name):
                    _setup_stream(vl, "", {}, 0)
                    if ctx:
                        ctx()
                    dblock = vl.stream.decider_block()
            seeds_c = [RagChunk(text=t, iso=iso_ago(sec), summary=t) for t, sec in seeds]
            d = await decide(surprise=20, silence_seconds=silence, recent_turns=turns(hist), topic_seeds=seeds_c,
                             last_feedback=None, prior_items=prior, stream=dblock)
            spoken, err = "", None
            if d.speak and part != "freq":
                async with lock:
                    with arm(part, name):
                        _setup_stream(vl, "", {}, 0)
                        if ctx:
                            ctx()
                        stim = Stimulus(StimulusKind.AUTONOMOUS_SPEECH, AutonomousSpeech(d.content, d.reason))
                        messages = orch._build_messages(stim, turns(hist), None)
                spoken, _n, err = await reply(messages, None)
        res[key] = {"part": part, "arm": name, "decision": {"speak": d.speak, "reason": d.reason, "content": d.content},
                    "spoken": spoken, "chars": _chars(spoken), "error": err}
        print(f"{key:28s} speak={d.speak!s:5s} 「{d.content[:50]}」→ {spoken[:70]}")

    def sc_line(yen, color, who, text):
        return mgr.format_comment(ChatMessage(id="x", kind="superchat", author=who, text=text,
                                              amount=f"￥{yen:,}", color=color))

    jobs = []
    parts = args.part.split(",") if args.part != "all" else ["length", "color", "memory", "freq"]
    R = args.reps
    if "length" in parts:
        for name in args.length_arms.split(","):
            for sid, said, who in LEN_SCENES:
                for r in range(R):
                    jobs.append(answer(f"length|{name}|{sid}|{r}", "length", name, said, who))
            for r in range(R):
                jobs.append(autonomous(f"length|{name}|AUTO|{r}", "length", name,
                                       [(H, "（チャット）みかん: こんばんは〜", 200), (E, "みかんさんいらっしゃい。", 190)],
                                       [MEM_CAKE], silence=60))
    if "color" in parts:
        for name in ("off", "on"):
            with arm("color", name):
                hist = []
                for yen, color, who, text, eve in SC_HISTORY:
                    hist += [(H, sc_line(yen, color, who, text), 200), (E, eve, 190)]
                for i, (yen, color, who, text) in enumerate(SC):
                    for r in range(R):
                        jobs.append(answer(f"color|{name}|SC{yen}|{r}", "color", name, sc_line(yen, color, who, text),
                                           who))
            for fid, said, who in FOLLOW:
                for r in range(R):
                    jobs.append(answer(f"color|{name}|{fid}|{r}", "color", name, said, who, hist=list(hist)))
            for r in range(args.auto_reps):
                jobs.append(autonomous(f"color|{name}|AUTO|{r}", "color", name, list(hist), [MEM_CAKE], silence=60))
    if "memory" in parts:
        for name in args.memory_arms.split(","):
            for r in range(args.auto_reps):
                jobs.append(autonomous(f"memory|{name}|AUTO|{r}", "memory", name,
                                       [(H, "（チャット）しろくま: 今日は何するの？", 300), (E, "特に決めてないです。", 290)],
                                       SEEDS, silence=90))
            for rid, said, who, mem in MEM_REPLY:
                for r in range(R):
                    jobs.append(answer(f"memory|{name}|{rid}|{r}", "memory", name, said, who, rag=[mem]))
    await asyncio.gather(*jobs)
    jobs = []
    if "freq" in parts:
        for name in args.freq_arms.split(","):
            fjobs = []
            for cid, el, viewers, total, people, last_c, hist, prior, silence in FREQ:
                def ctx(el=el, viewers=viewers, total=total, people=people, last_c=last_c):
                    st = vl.stream
                    st.started_mono = time.monotonic() - el
                    st.video = dataclasses.replace(st.video, viewers=viewers)
                    st.total = total
                    st.people = {f"p{i}": {} for i in range(people)}
                    st.last_comment_mono = time.monotonic() - (last_c or 0)
                for r in range(args.freq_reps):
                    fjobs.append(autonomous(f"freq|{name}|{cid}|{r}", "freq", name, hist, SEEDS, prior=prior,
                                            silence=silence, ctx=ctx))
            if name == "nogap":
                dec_mod.STREAM_PRIOR_GAP_RULE = False  # 判定は呼ぶ時に組み立てるので、この腕の間ずっと外す
            try:
                await asyncio.gather(*fjobs)
            finally:
                dec_mod.STREAM_PRIOR_GAP_RULE = True

    # memory: 誰との体験として話したかの分類（判定モデル・記号だけ）
    for key, r in res.items():
        if r.get("part") != "memory" or not r.get("spoken") or r.get("framing"):
            continue
        mem = (MEM_ALCHEMY if ("R1" in key or "AUTO" in key) else MEM_RAMEN)[0]
        if "AUTO" in key:
            mem = " / ".join(m for m, _ in SEEDS[:2])
        msg = [{"role": "user", "content": CLASSIFY.format(memory=mem, spoken=r["spoken"])}]
        try:
            resp = await vl.registry.complete("speech_decide", msg)
            r["framing"] = (resp.choices[0].message.content or "").strip()[:1]
        except Exception as e:  # noqa: BLE001
            r["framing"] = f"?{type(e).__name__}"
    dump(path, res)
    summarize(res)
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def summarize(res: dict) -> None:
    def rows(part, arm, pred=lambda k: True):
        return [(k, r) for k, r in res.items() if r.get("part") == part and r.get("arm") == arm and pred(k)]

    def med(xs):
        return statistics.median(xs) if xs else 0

    print("\n== 1. 長さ（文字数の中央・四分位・最小〜最大／文の数の中央）==")
    for arm in ("base", "new", "strong", "fewex", "nopshort"):
        for label, pred in (("コメントへの返事（短く返してよい物を除く）", lambda k: "|AUTO|" not in k
                             and not any(s in k for s in SHORT_IDS)),
                            ("短く返してよい物（w・おやすみ）", lambda k: any(s in k for s in SHORT_IDS)),
                            ("自分から話す時", lambda k: "|AUTO|" in k)):
            xs = sorted(r["chars"] for _k, r in rows("length", arm, pred) if r.get("spoken"))
            ss = [r.get("sentences", 0) for _k, r in rows("length", arm, pred) if r.get("spoken")]
            if xs:
                q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], med(xs), xs[-1]]
                print(f"  {arm:4s} {label}: n={len(xs)} 中央 {med(xs):.0f}字（{q[0]:.0f}〜{q[-1]:.0f}・{xs[0]}〜{xs[-1]}）"
                      f" 文 {med(ss):.0f}")
    print("\n== 2. スパチャの色 ==")
    for arm in ("off", "on"):
        sc = rows("color", arm, lambda k: "|SC" in k)
        said = sum(1 for _k, r in sc if colors_in(r["spoken"]))
        wrong = 0
        for k, r in sc:
            yen = int(k.split("|")[2][2:])
            true = next(c for y, c, _w, _t in SC if y == yen)
            if colors_in(r["spoken"]) - {true}:
                wrong += 1
        fol = rows("color", arm, lambda k: "|F" in k)
        fol_c = sum(1 for _k, r in fol if colors_in(r["spoken"]))
        fol_s = sum(1 for _k, r in fol if any(w in r["spoken"] for w in SC_WORDS))
        au = rows("color", arm, lambda k: "|AUTO|" in k)
        au_c = sum(1 for _k, r in au if colors_in(r["spoken"] + r["decision"]["content"]))
        au_s = sum(1 for _k, r in au if any(w in r["spoken"] + r["decision"]["content"] for w in SC_WORDS))
        print(f"  {arm:3s} スパチャへの返事: 色を言った {said}/{len(sc)}・違う色を言った {wrong}/{len(sc)}")
        print(f"      その後の普通のコメントへの返事: 色の話 {fol_c}/{len(fol)}・スパチャの話 {fol_s}/{len(fol)}")
        print(f"      自分から話す時: 色の話 {au_c}/{len(au)}・スパチャの話 {au_s}/{len(au)}"
              f"（話した {sum(1 for _k, r in au if r['decision']['speak'])}/{len(au)}）")
    print("\n== 3. 開発者との体験（A=開発者とのこととして・B=聞き手が当事者のように・C=触れない）==")
    for arm in ("old", "new", "new2", "mem2"):
        for label, pred in (("自分から話す時", lambda k: "|AUTO|" in k), ("コメントへの返事", lambda k: "|R" in k)):
            rs = [r for _k, r in rows("memory", arm, pred)]
            fr = [r.get("framing", "-") for r in rs if r.get("spoken")]
            print(f"  {arm:3s} {label}: A {fr.count('A')}・B {fr.count('B')}・C {fr.count('C')}"
                  f"（話した {len(fr)}/{len(rs)}）")
    print("\n== 4. コメントの無い配信で自分から話す率（発話判定）==")
    farms = ("base", "new", "tempo", "nogap", "fixed")
    for cid, *_rest in FREQ:
        line = []
        for arm in farms:
            rs = [r for _k, r in rows("freq", arm, lambda k, c=cid: f"|{c}|" in k)]
            if rs:
                line.append(f"{arm} {sum(1 for r in rs if r['decision']['speak'])}/{len(rs)}")
        print(f"  {cid}: " + "・".join(line))
    for arm in farms:
        rs = [r for _k, r in rows("freq", arm)]
        if rs:
            print(f"  合計 {arm}: {sum(1 for r in rs if r['decision']['speak'])}/{len(rs)}")


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", help="length,color,memory,freq のどれか（カンマ区切り）か all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--auto-reps", type=int, default=6)
    ap.add_argument("--freq-reps", type=int, default=10)
    ap.add_argument("--length-arms", default="base,new")
    ap.add_argument("--memory-arms", default="old,new")
    ap.add_argument("--freq-arms", default="base,new")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--summary", action="store_true", help="呼ばずに前回の結果をまとめ直す")
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarize(load(DIR / "result.json"))
        return 0
    setup_env({"FILLER_ENABLED": "0", "LAUGH_ENABLED": "0", "FILLER_MOOD_ENABLED": "0", "STREAM_ENABLED": "1"})
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
