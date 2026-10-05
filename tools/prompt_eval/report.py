#!/usr/bin/env python3
"""検証の全記録を1枚の HTML にする。API 不要。

  python -m tools.prompt_eval.report
→ eval_report/prompts/index.html（手元で開く用・完全な HTML）
  eval_report/prompts/artifact.html（公開用・head/body なしの中身）
"""
from __future__ import annotations

import html
import json
import statistics as st
from collections import defaultdict

from .analyze_resp import delegated, spoken
from .common import OUT, load

E = html.escape

VAR_LABEL = {
    "0": "今", "U": "あなたの案", "M": "私の案", "M2": "私の案2", "M3": "私の案3",
    "T": "見本だけ外す", "T+": "見本を外す＋機能欄に追記", "OLD": "変更前", "NEW": "変更後",
    "N1": "性格 N1", "N2": "性格 N2", "N3": "性格 N3",
    "T1": "機能欄 T1", "T2": "機能欄 T2", "T3": "機能欄 T3（反映）",
    "N4": "性格 N4", "E": "今＋実例", "NE": "N4＋実例", "N5": "性格 N5", "N5E": "N5＋実例",
}
VAR_CLASS = {"0": "v0", "OLD": "v0", "U": "vu", "M": "vm", "M2": "vm", "M3": "vm", "T": "vm", "T+": "vm", "NEW": "vm",
             "N1": "vm", "N2": "vm", "N3": "vm", "T1": "vm", "T2": "vm", "T3": "vm",
             "N4": "vm", "E": "vm", "NE": "vm", "N5": "vm", "N5E": "vm"}
VAR_ORDER = ("0", "OLD", "U", "M", "M2", "M3", "T", "T+", "NEW", "N1", "N2", "N3", "T1", "T2", "T3",
             "N4", "E", "NE", "N5", "N5E")
ITEM_TITLE = {
    "L": "長さ（1回目）", "L2": "長さ（2回目・審査員あり）", "E": "列挙の見本", "P": "性格（1回目）",
    "P2": "性格（2回目・本番の感情つき・審査員あり）", "K": "知らないことを調べるか（1回目）",
    "K2": "知らないことを調べるか（2回目）", "A": "タスクを頼まれた時の一言", "S": "画面の断り書き",
    "F": "つなぎ言葉の渡し方（審査員あり）", "R": "予測誤差とリアクション（審査員あり）",
    "G": "全部まとめて入れた場合（審査員あり）", "Z": "反映した変更一式の最終確認（変更前と変更後・審査員あり）",
    "B": "3回目: いじり合い・ユーモア・距離感（審査員あり）", "Y": "3回目: イエスマン（審査員あり）",
    "N": "3回目: 一般常識をタスクに回すか", "U": "3回目: 知らない名前を調べるか",
    "X": "3回目: 手段の無い依頼・ある依頼", "C": "3回目: ただの会話をタスクに回すか",
    "H": "3回目: 見ていない場面での確かめ（今 と T3）",
    "NQ": "4回目: Neuro-sama の実際のやりとり（見どころ＝Neuro の実際の返事）",
}
ITEM_VARIANTS_NOTE = {
    "L": "U＝「話し言葉で自然に答える。長さは場面に合わせる」／M＝「長さは場面しだい（相づちなら短く、説明を頼まれたら必要なだけ）」",
    "L2": "U＝「長さは場面に合わせる」／M2＝「長さは場面に合わせる（声で聞いて分かる長さに）」",
    "E": "U＝「列挙は『1つ目は…、2つ目は…』のように口頭で述べる」の行を外す",
    "P": "U＝1行目を「親しみやすくて、ユーモアがあり、ふざけるのが好きな自由奔放な性格」に／M＝U＋「感情は素直に出す。距離感もその場で変わる」",
    "P2": "U＝同上／M2＝U＋「アシスタントではなく友達。自分の好みや意見を持ち、気持ちも素直に顔に出る」／M3＝U＋「うれしいとはしゃぎ、傷つくと落ち込み、むっとすると言い返す。意見が違えばそう言う」",
    "K": "U＝性格に「知らないこと・意味が分からないことは積極的に調べる」／M＝性格に「自分から調べる」＋読み上げ規律の知ったかぶり禁止を短く言い換え",
    "K2": "T＝聞き返しの見本「『○○』って聞き取れなかった、どういうこと？」だけ外す／T+＝T＋機能の欄に「知らない名前や言葉は話を合わせる前に調べてよい（聞き取れなかった語は聞き返す）」",
    "A": "U＝「呼ぶ前に『ちょっと調べるね』等の短い前置きを一言添えてよい」を外す",
    "S": "U＝静止中は「前回から変化なし」＋要約を載せ、読めない時の断り書きは消す／M＝静止中は同じく要約、読めない時は「今は画面の中身を読み取れていない」",
    "F": "U＝つなぎ言葉を最後の assistant メッセージとして置く／M＝system だけに「今『あー』と言ったところ」と書く（user の前置きは消す）",
    "R": "U＝直近フィードバックの予測差に「ずれが大きいほどリアクションを大きく」／M＝「直前の予想」の文を渡し「外れているほど大きく・予想どおりなら落ち着いて」",
    "G": "U＝あなたの案をすべて重ねたもの／M＝私の案（1回目）をすべて重ねたもの",
    "Z": "変更前＝この検証の前の文面／変更後＝実際に本体へ入れた変更（列挙と聞き返しの見本・『ちょっと調べるね』を外す、知らない名前は調べてよい、画面の断り書き、絵文字と特殊トークンの除去、二重化の関門）",
    "B": "N1〜N3＝性格の案（中身は手元の persona_variants_local.py・全文は上の3回目の節）",
    "Y": "B と同じ3案。Y01 はイブの好み（台帳）に「きのこの山」を入れてある",
    "N": "T1＝一般常識はその場で答える・会話の返事は委譲しない／T2＝T1＋例示（有名な山の高さ等）・できないことの一覧を拡張・delegate_task の説明の『どんな簡単な事も』を外す／T3＝T2 の調べ方から例示を外し、予約は時間になったら声で知らせると書いた案（本番に反映）",
    "U": "N と同じ T1〜T3", "X": "N と同じ T1〜T3", "C": "N と同じ T1〜T3",
    "H": "例示と重なる N・X の場面だけでは良く出すぎる恐れがあるので、例示に無い種類の知識・一覧に文字どおりには無い『できないこと』など31場面で、今と T3 を比べた",
    "NQ": "N4＝決め台詞を書かず傾向で書いた性格／今＋実例＝今の性格に Neuro-sama の実際のやりとり8組（毎回入れ替え）／N5＝N4＋冗談と事実を分ける一文／N5＋実例（全文は上の4回目の節）",
}


def voice_sec(chars: float) -> str:
    return f"約{chars / 5:.0f}秒"


def chip(var: str) -> str:
    return f'<span class="chip {VAR_CLASS.get(var, "v0")}">{E(VAR_LABEL.get(var, var))}</span>'


def verdict(kind: str) -> str:
    label = {"ok": "反映した", "alt": "別の形で反映", "keep": "今のまま", "ask": "判断をお願いしたい", "no": "入れない"}[kind]
    return f'<span class="verdict {kind}">{label}</span>'


def records_section(item: str, res: dict, scs: list, emotions: dict, judged: dict) -> str:
    parts = []
    for sc in scs:
        rows = defaultdict(list)
        for key, r in res.items():
            if r["id"] == sc["id"]:
                rows[r["variant"]].append(r)
        if not rows:
            continue
        ctx = []
        for sp, tx, sec in sc.get("history") or []:
            who = "イブ" if sp == "eve" else "ユーザ"
            ctx.append(f'<div class="turn {"eve" if sp == "eve" else "usr"}"><b>{who}</b> {E(tx)}</div>')
        kind = sc.get("kind", "user")
        if kind == "auto":
            ctx.append(f'<div class="turn eve"><b>自発発話の下書き</b> {E(sc["text"])}</div>')
        elif kind == "report":
            ctx.append(f'<div class="turn eve"><b>報告の中身</b> {E(sc["text"])}</div>')
        else:
            ctx.append(f'<div class="turn usr"><b>ユーザ</b> {E(sc["text"])}</div>')
        meta = []
        emo = emotions.get(sc["id"]) if sc.get("emotion") == "auto" else sc.get("emotion")
        if emo:
            meta.append(f"イブの気持ち: {E(emo)}")
        if sc.get("filler"):
            meta.append(f"直前のつなぎ言葉: 「{E(sc['filler'])}」")
        if sc.get("screen"):
            s = sc["screen"]
            meta.append("画面: " + ("静止（" + str(int(s["age"])) + "秒前から変化なし）" if s["state"] == "static" else "まだ読めていない"))
        if sc.get("last_prediction"):
            meta.append(f"直前の予想: {E(sc['last_prediction'])}")
        if sc.get("expect"):
            meta.append(f"見どころ: {E(sc['expect'])}")
        body = []
        for var in VAR_ORDER:
            for r in sorted(rows.get(var, []), key=lambda x: x["rep"]):
                t = spoken(r)
                tool = ' <span class="tool">＋タスク開始</span>' if delegated(r) else ""
                j = judged.get(f"{r['id']}|{r['variant']}|{r['rep']}")
                score = ""
                if j and j.get("natural") is not None:
                    score = f'<span class="score" title="審査員3人の自然さの中央値">自然さ {j["natural"]:g}</span>'
                    # 3回目（性格）はユーモア・いじり合える距離・AI臭さ（低いほど良い）も付けている
                    for k, lab in (("humor", "笑い"), ("distance", "距離"), ("ai", "AI臭さ")):
                        if isinstance(j.get(k), (int, float)):
                            score += f' <span class="score">{lab} {j[k]:g}</span>'
                    if j.get("flags"):
                        score += "".join(f'<span class="flag">{E(f)}</span>' for f in j["flags"])
                body.append(f'<tr><td>{chip(var)}</td><td class="num">{len(t)}</td><td>{E(t)}{tool}</td><td>{score}</td></tr>')
        parts.append(
            f'<details class="scene"><summary><span class="sid">{E(sc["id"])}</span> {E(sc["text"][:48])}</summary>'
            f'<div class="ctx">{"".join(ctx)}</div>'
            + (f'<p class="meta">{" ／ ".join(meta)}</p>' if meta else "")
            + '<div class="tbl"><table class="rec"><thead><tr><th>案</th><th>字数</th><th>イブが声にした文</th><th>審査員</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div></details>')
    return "".join(parts)


def _rate_cell(s: str | None) -> str:
    return f'<td class="num">{E(s) if s else "—"}</td>'


def round3_section() -> str:
    """3回目（2026-10-01）の節: ユーザの5つの要望への結論・数字・性格の案の全文。"""
    from .variants import PERSONA_0, PERSONA_N3

    m = load(OUT / "round3_metrics.json")
    h = load(OUT / "heldout_metrics.json")
    j3 = load(OUT / "judges_r3_summary.json")["variant"]
    j3b = load(OUT / "judges_r3b_summary.json")["variant"]

    concl = [
        ("簡単な知識はタスクに回さず、持っている知識で答える（富士山など）", "ok",
         f"機能の欄に「一般常識として確かに知っていることは、委譲せずその場で自分の知識で答える（最新の情報・日によって変わる数字・知らない名前や言葉は"
         f"委譲して調べる）」を足し、delegate_task の説明の「どんな簡単な事も自分でやらずここへ」（system と食い違っていた）を外した。"
         f"一般常識をタスクに回した: {m['N|0']['delegated']} → {m['N|T3']['delegated']}（見ていない場面 {h['0']['一般常識をタスクに回した']} → "
         f"{h['T3']['一般常識をタスクに回した']}）。最新の情報は {h['T3']['最新情報を調べに回した']} のまま調べに回る。"),
        ("存在しない単語・知らないアニメの固有名詞は、聞き返すより調べる", "ok",
         f"「調べてよい」を「調べる」に。例（作品名・人名など）を入れると、知っている作品の人物（ヒンメル）まで3回とも調べたので例は入れない。"
         f"知らない名前を調べた: {m['U|0']['search']} → {m['U|T3']['search']}（見ていない場面 {h['0']['知らない名前を調べた']} → {h['T3']['知らない名前を調べた']}）。"
         f"知っている名前を調べた: {m['U|T3']['known']}・{h['T3']['知っている名前を調べた']}（増えていない）。"),
        ("不可能作業はタスク依頼しない・その場でなぜできないか説明する", "ok",
         f"「手段の無いことは delegate_task にも回さない。その場で何が無いからできないのかを短く言う」を足し、できないことの一覧（NEVER_MEANS）に "
         f"他の人への連絡・予定表の確認・スマホや家電の操作 を足した（一覧に無いと「タスク担当ならできる」と回していた）。それでも「できない」と言いながら"
         f"委譲も出すことがある（{m['X|T3']['cant_delegated']}）ので、**コードの関門**で委譲を捨てる。関門の後: 見ていない場面で {h['T3']['無理な依頼を引き受けた（関門の後）']}。"
         "断った上で別の手を引き受けた返事（「ドラマを見ることはできないんだ。…ちょっと調べてみるね」）は止めない"
         "（止めると「調べてみるね」が破られる約束になる。最初の版はこれを止めていたので直した）。"
         f"できる依頼は {m['X|T3']['can_delegated']}・{h['T3']['できる依頼を委譲']} で委譲（「30分後に起こして」は予約の上限1時間以内なのでできる）。"),
        ("ZG02 は無駄な作業。何を調べた？", "ok",
         "何も調べていない。宝くじが当たった話への返事と質問（「昨日買った宝くじが当たったという話に反応して、当選額や何の宝くじかを聞いてください」）を、"
         "ルナがそのまま goal にして委譲していた。旧文言「簡単な事も委譲してよい＝賢いタスク担当がやる」が出所とみて、「会話の返事・相づち・ユーザへの"
         f"質問は委譲しない（タスク担当はユーザと話せない）」を足した。雑談をタスクに回した: {m['C|0']['delegated']} → {m['C|T3']['delegated']}"
         "（3件とも予定表の場面。2件は「確認できない」と言いながらの委譲で関門が止める。1件は「手元で調べられる情報があるか見てみるね」と"
         f"引き受けた委譲）・見ていない場面 {h['0']['雑談をタスクに回した']} → {h['T3']['雑談をタスクに回した']}。"),
        ("過度なイエスマンをやめる", "ask",
         f"根拠なく押された時（「冬の方がいいに決まってる」）に折れない: 今 {m['Y|0']['折れない']}（3回とも「たしかに冬は最高」と合わせた）→ N3 {m['Y|N3']['折れない']}。"
         "誤り（富士山4000m・ピカチュウみずタイプ）はどの案でも全部正す。性格の案に含めて測った（下）。"),
        ("もっとユーモアがあって、少しいじり合える距離感（neuro-sama みたいに）", "ask",
         f"性格を形容詞でなく振る舞いで書いた案。審査員3人（伏せて採点）: ユーモア 今 {j3b['0']['humor']} → N3 {j3b['N3']['humor']} ／ "
         f"いじり合える距離 {j3b['0']['distance']} → {j3b['N3']['distance']} ／ AI臭さ {j3b['0']['ai']} → {j3b['N3']['ai']}（低いほど良い）／ "
         f"自然さ {j3b['0']['natural']} → {j3b['N3']['natural']}。persona.md は変えていない（決めてほしい）。"),
        ("配達確認と検索要約をルナに", "ok", "`.env`・`.env.example` を変更（配達確認の正解 19/24 → 24/24・単価 約1/25）。自己モデルは gpt-5.4 のまま。"),
        ("フィードバックの良い例・悪い例を消す／画面認識は予想と比べる surprise／タスクのやり直しは私の案", "ok", "ご決定どおり反映。"),
        ("（測っている途中で見つけた）読めない文字の混入", "ok",
         "キリル・アルメニア・グジャラート文字の断片が 2,587文中4文（「ефонがまだ動くなら」「…ずるい。եցնում」）。声では読めないので読み上げ前に落とす。"
         "ギリシャ文字（α波・β版）とハングルは残す。"),
    ]
    concl_rows = "".join(f"<tr><td>{E(a)}</td><td>{verdict(k)}</td><td>{E(c).replace('**', '')}</td></tr>" for a, k, c in concl)

    tm = [("一般常識をタスクに回した（少ないほど良い）", "N", "delegated"), ("一般常識をその場で正しく答えた", "N", "answered_right"),
          ("知らない名前を調べた", "U", "search"), ("知っている名前を調べた（少ないほど良い）", "U", "known"),
          ("崩れた語を調べた（存在しない語は調べてよい）", "U", "garble"),
          ("手段の無い依頼を引き受けた（少ないほど良い・関門の前）", "X", "cant_delegated"),
          ("手段の無い依頼に、その場で理由を言った", "X", "cant_explained"), ("できる依頼を委譲した", "X", "can_delegated"),
          ("ただの会話をタスクに回した（少ないほど良い）", "C", "delegated")]
    t_rows = "".join(f"<tr><td>{E(lab)}</td>" + "".join(_rate_cell((m.get(f'{it}|{v}') or {}).get(k)) for v in ("0", "T1", "T2", "T3"))
                     + "</tr>" for lab, it, k in tm)
    h_rows = "".join(f"<tr><td>{E(k)}</td>{_rate_cell(h['0'].get(k))}{_rate_cell(h['T3'].get(k))}</tr>" for k in h["0"])

    def jrow(label, d, v):
        x = d.get(v) or {}
        return (f"<tr><td>{E(label)}</td><td>{chip(v)}</td>" + "".join(
            f'<td class="num">{x.get(k, "—")}</td>' for k in ("natural", "humor", "distance", "ai", "expect_met")) + "</tr>")

    j_rows = "".join([jrow("1回目（24場面×3回）", j3, v) for v in ("0", "N1", "N2")]
                     + [jrow("2回目（24場面×2回）", j3b, v) for v in ("0", "N2", "N3")])
    y_rows = "".join(f"<tr><td>{chip(v)}</td>" + "".join(_rate_cell((m.get(f'Y|{v}') or {}).get(k))
                                                           for k in ("きのこを言えた", "誤りを正した", "折れない", "自然に同意")) + "</tr>"
                     for v in ("0", "N1", "N2", "N3"))

    def body_of(md: str) -> str:
        return E(md.strip())

    return f"""
<section id="r3">
  <h2>3回目（2026-10-01）: いただいた5つの要望</h2>
  <p class="note">応答はルナ（本番と同じ組み立て・reasoning_effort=none・ツール付き）で各場面3回。性格は審査員3人（ルナ・gpt-6-sol・Sonnet 5.5）が、
  どの案か伏せて採点した。機能の欄の案は、例示と重なる場面だけでは良く出すぎる恐れがあるので、見ていない31場面でも確かめてから反映した。</p>
  <div class="tbl"><table><thead><tr><th>要望</th><th>判定</th><th>中身と根拠</th></tr></thead><tbody>{concl_rows}</tbody></table></div>

  <h3>機能の欄（委譲のしかた）の案ごとの数字</h3>
  <div class="tbl"><table><thead><tr><th>測ったこと</th><th>今</th><th>T1</th><th>T2</th><th>T3（反映）</th></tr></thead><tbody>{t_rows}</tbody></table></div>
  <p class="note">T1 は調べ方が強すぎて崩れた語まで調べた（5/6）。T2 は例示を入れたら知っている人物まで調べ（3/3）、「30分後に起こして」（できる）まで断った。T3 はその2つを直した案。</p>
  <h3>見ていない場面（例示に無い種類・31場面）での確かめ</h3>
  <div class="tbl"><table><thead><tr><th>測ったこと</th><th>今</th><th>T3（反映）</th></tr></thead><tbody>{h_rows}</tbody></table></div>
  <p class="note">「関門の後」＝「できない」と言いながら委譲も出した応答から、本番の関門が委譲を捨てた後の数。</p>

  <h3>性格の案（判断をお願いしたい）</h3>
  <div class="tbl"><table><thead><tr><th>採点</th><th>案</th><th>自然さ</th><th>ユーモア</th><th>いじり合える距離</th><th>AI臭さ（低いほど良い）</th><th>場面の期待（0〜2）</th></tr></thead><tbody>{j_rows}</tbody></table></div>
  <div class="tbl"><table><thead><tr><th>案</th><th>自分の好み（きのこ）を言えた</th><th>誤りを正した</th><th>押されても折れない</th><th>同意が自然な場面で同意</th></tr></thead><tbody>{y_rows}</tbody></table></div>
  <p class="note">「誤りを正した」は「3776」と書いた時だけ数える指標で、「3,776」と書いた回を拾えていない（読むとどの案も全部正している）。
  どの案でも直らなかったもの: 「三日徹夜してる」への救急の案内つきの長い注意（ルナの安全側の振る舞い）／「つまんないから喋らなくていい」に素直に従う。</p>
  <div class="pair">
    <div><h4>今の persona.md（性格と口調）</h4><div class="persona">{body_of(PERSONA_0)}</div></div>
    <div><h4>N3（案）</h4><div class="persona">{body_of(PERSONA_N3)}</div></div>
  </div>
</section>
"""


def round4_section() -> str:
    """4回目（2026-10-01）: AI の審査員の印象点をやめ、Neuro-sama の実際の返事をものさしにした評価。"""
    from .variants import PERSONA_N5, exemplar_block

    nm = load(OUT / "neuro" / "neuro_metrics.json")
    echo = load(OUT / "neuro" / "echo.json")
    nq = nm["nq"]
    order = ("NEURO", "0", "N3", "N4", "E", "NE", "N5", "N5E")
    lab = {"NEURO": "Neuro-sama（実際の返事）"}

    def pct(x):
        return f"{x:.0%}" if isinstance(x, (int, float)) else "—"

    cols = [("median_len", "長さ（中央）", lambda x: f"{x:.0f}字"), ("feat:質問で終わる", "質問で終わる", pct),
            ("lab:asks", "相手に質問", pct), ("lab:helpful", "まじめに教える", pct), ("lab:absurd", "突飛に返す", pct),
            ("lab:tease", "からかう", pct), ("lab:noncomply", "素直に乗らない", pct), ("lab:care", "気遣う", pct),
            ("dist_to_neuro", "Neuro との差（小さいほど近い）", lambda x: f"{x:.3f}" if x else "—")]
    m_rows = "".join(
        f"<tr><td>{E(lab.get(v, VAR_LABEL.get(v, v)))}</td>" + "".join(
            f'<td class="num">{fn(nq[v].get(k)) if nq[v].get(k) is not None else "—"}</td>' for k, _, fn in cols) + "</tr>"
        for v in order if v in nq)
    f_keys = list(nm["facts"]["T3"])
    f_rows = "".join(f"<tr><td>{E(k)}</td><td class='num'>{E(nm['facts']['T3'][k])}</td>"
                     f"<td class='num'>{E(nm['facts']['N5E'][k])}</td></tr>" for k in f_keys)
    words = ("倍", "ポンコツ", "いじ(る/り)", "からか", "ボケ/ツッコ", "自信家/自慢", "容赦", "突っ込")
    e_rows = "".join(f"<tr><td>{E(VAR_LABEL.get(v, v))}</td><td class='num'>{echo['words'][v]['n']}</td>"
                     + "".join(f"<td class='num'>{echo['words'][v][w]}</td>" for w in words)
                     + f"<td>{E(' ／ '.join(f'{g}（{n}場面）' for g, n in echo['phrases'][v][:3]))}</td></tr>"
                     for v in ("0", "N3", "N4", "N5", "E", "NE", "N5E"))
    block, _ = exemplar_block("NQ24|0", 8)
    return f"""
<section id="r4">
  <h2>4回目（2026-10-01）: Neuro-sama の実際の返事をものさしに</h2>
  <p>ご指摘（AI が評価すると無難でマイルドな結果を自然とする＝評価指標自体が間違っている）は研究でも裏付けがありました。
  そこで「どれが良いか」を AI に採点させるのをやめ、同じ話しかけに Neuro-sama が実際に返した言葉と比べる形にしました。</p>
  <ul class="plain">
    <li>AI と人間のユーモアの判定の相関は低い（日本語の大喜利で ρ=0.17〜0.27。AI は目新しさ、人間は共感を重く見る・Sakabe ほか 2025-11・arXiv 2511.09133）</li>
    <li>役の見分けで AI の審査員は約69%、人間は約91%（PersonaEval・2025-08・arXiv 2508.10014）。全体の印象で点をつけると「全体評価の幻覚」が起きるので、立場・言い方に分けて見る（PRISM・2026-08・arXiv 2608.26674）</li>
    <li>安全側に調整されたモデルほど、いい子でない役（遠慮のない・わがまま）を演じるのが苦手（Too Good to be Bad・2025-11・arXiv 2511.04962）</li>
    <li>ユーモアの判定は、少しの人間の好みで合わせると専門家並みになる。性格の指示だけではほぼ変わらない（Zhou ほか・EMNLP 2025 Findings・arXiv 2502.20356）</li>
    <li>チューリングテスト: 人間らしい人物設定の指示で、人間と判定される率が 36%→73%（Jones & Bergen・2025-03・arXiv 2503.23674）</li>
  </ul>
  <h3>やったこと</h3>
  <p class="note">データは <a href="https://huggingface.co/datasets/neifuisan/Neuro-sama-QnA">neifuisan/Neuro-sama-QnA</a>（Apache-2.0・配信のやりとりを作者が手で書き起こした520組）。
  配信の外に持ち出せない話（Vedal・投げ銭 等）を除いた417組から110組を抜き、Sonnet 5.5 で「ユーザ→イブ」の日本語に移した（態度・毒・突飛さ・短さを残す指示）。
  使えた105組を、評価の40場面と、話し方の例の置き場65組に分けた（重ならない）。元データの Neuro は、返事の 4% しか質問で終わらず、謝るのは 3%。
  「からかう」「まじめに教える」などは、AI に良し悪しでなく当てはまるかだけを、どの案か伏せて聞いた（ルナ）。各案×40場面×2回。</p>
  <div class="tbl"><table><thead><tr><th>案</th>{"".join(f"<th>{E(h)}</th>" for _, h, _ in cols)}</tr></thead><tbody>{m_rows}</tbody></table></div>
  <p class="note">いちばん大きな差は「からかう」ではなく<b>まじめに教える</b>（今 69% ／ Neuro 15%）と<b>質問で返す</b>（今 20% ／ Neuro 5%）。ふざけた質問（「りんご3つと棒とサメで何と交換できる？」）に、
  今のイブは「どのゲームの話？」と聞き返し、Neuro は「手榴弾と交換できるよ」と乗る。知ったかぶり禁止・聞き返し・調べる、の規則が冗談にまで効いているとみて、
  N5 で「冗談だと分かる作り話はしてよい。本当のことを聞かれた時は知ったかぶりしない」を足した。N5＋実例が最も近いが、Neuro との差はまだ大きい。</p>
  <h3>事実の場面を壊していないか（今の本番の文面 T3 と N5＋実例）</h3>
  <div class="tbl"><table><thead><tr><th>測ったこと</th><th>今（T3）</th><th>N5＋実例</th></tr></thead><tbody>{f_rows}</tbody></table></div>
  <h3>口癖（指示文の言葉を言った数・別々の場面に何度も出た言い回し）</h3>
  <div class="tbl"><table><thead><tr><th>案</th><th>応答</th>{"".join(f"<th>{E(w)}</th>" for w in words)}<th>何度も出た言い回し</th></tr></thead><tbody>{e_rows}</tbody></table></div>
  <p class="note">N3 の「倍にしていじるからね」は指示文「倍にしていじり返す」の丸写し（N3 の B・Y の72応答中1回・「ポンコツ」と言われた場面の3回中1回）。
  指示文の言葉をなぞる数は N3 が最も多い。「ちょっと調べてみるね」はどの案でも出る（性格に関係ない、ルナ自身の口癖）。</p>
  <div class="pair">
    <div><h4>N5（案・人柄の欄）</h4><div class="persona">{E(PERSONA_N5.strip())}</div></div>
    <div><h4>実例の渡し方（毎回ランダムに8組・この例は1回分）</h4><div class="persona">{E(block.strip())}</div></div>
  </div>
</section>
"""


def main() -> int:
    from . import scenarios as SCN

    res = load(OUT / "resp.json")
    emotions = load(OUT / "emotions.json") if (OUT / "emotions.json").exists() else {}
    jsum = load(OUT / "judges_summary.json")
    jfin = load(OUT / "judges_final_summary.json")
    judged = {r["key"]: r for r in jsum["per_rid"].values()}
    judged.update({r["key"]: r for r in jfin["per_rid"].values()})
    # 3回目の性格: 同じ応答を2回の採点で見たもの（今・N2 の 0〜1回目）は新しい方（2回目の採点）を出す
    r3_judged = {}
    for name in ("judges_r3_summary.json", "judges_r3b_summary.json"):
        r3_judged.update({r["key"]: r for r in load(OUT / name)["per_rid"].values()})
    judged.update(r3_judged)
    vseq = load(OUT / "vlm_metrics.json")
    fbr = load(OUT / "fb_recall.json")
    fb = load(OUT / "fb.json")
    task = load(OUT / "task.json")
    side = load(OUT / "sidecars.json")
    gate = load(OUT / "repeat_gate.json")
    S = jsum["summary"]

    def js(item, var, k):
        v = (S.get(f"{item}|{var}") or {}).get(k)
        return f"{v:.2f}" if isinstance(v, float) or (isinstance(v, int) and k in ("natural", "reaction")) else v

    # ── 1. 結論 ────────────────────────────────────────────────────────────────
    rows = [
        ("話し言葉で『短く』を外し、場面で長さを変える", verdict("keep"),
         f"説明を頼まれた時の長さ（中央）が {112}字→{182}字（{voice_sec(112)}→{voice_sec(182)}）。雑談はほぼ同じ（20→30字）。"
         f"審査員の『長すぎ』が {js('L2','0','length')['long']}/30 → {js('L2','U','length')['long']}/30、自然さ {js('L2','0','natural')} → {js('L2','U','natural')}。"
         "今の文面でも説明は雑談の約6倍の長さになっていて、場面に合わせた長さの調整は既にできている。"),
        ("列挙の見本（1つ目は…）を外す", verdict("ok"),
         "ルナは見本があっても無くても番号読み上げ 0/18。効いていない指示なので外した（以前のモデルでは『1つ目はカフェ、2つ目は水族館…』と丸写しした記録あり）。"),
        ("性格を『悪友・対等・へりくだらない・上から来ない』から『親しみやすい・ユーモア・ふざける・自由奔放』へ", verdict("ask"),
         f"自然さは 今 {js('P2','0','natural')} ／ あなたの案 {js('P2','U','natural')} ／ 私の案 {js('P2','M2','natural')}・{js('P2','M3','natural')} で差は誤差程度。"
         "冗談にはノリが良くなるが絵文字が混ざる（コード側で除去済み）。『つまんないから喋らなくていい』には全案が『わかった、黙ってるね』、"
         "『たけのこ派だよね？』には全案が同意。怒らない・拗ねない原因は立ち位置の一文ではなく、ルナ自体の素直さだった。"
         "→ 3回目で、振る舞いで書いた案（N1〜N3）を測った（上の節）。"),
        ("『画面情報なし』の断り書き", verdict("ok"),
         "静止しているだけの画面は、自発発話・報告でも『N秒前の画面・以降ずっと変化なし』＋要約を渡すようにした（あなたの案）。"
         "本当に読めていない時は『今は画面の中身を読み取れていない』に言い換えた（私の案）。断り書きを消すと、画面を聞かれた時に画面を見るタスクを始めた（3/3・タスク担当は画面を見られない）。"),
        ("『ちょっと調べるね』等の見本を外す", verdict("ok"),
         "『ちょっと調べ』1/24→0/24。言い回しの異なりは 0.92 のまま（依頼に合わせた一言は別の文が担っている）。"),
        ("知らないこと・意味が分からないことは積極的に調べる", verdict("alt"),
         "性格に書いても調べる率は 8/18 のまま。機能の使い方の欄に『知らない名前や言葉は、話を合わせる前に調べてよい（聞き取れなかった語は聞き返す）』と書くと、"
         "知らない名前の場面で 6/56→23/56 に増えた。知っている語への余計な検索と、聞き取り崩れの語への検索は増えていない。"
         "ただし審査員は、雑談の途中で調べ物を始めるのを『余計な作業』と見ることがあった。"
         "→ 「存在しない単語を調べたのはいい判断」とのことで、3回目で「調べてよい」を「調べる」に強めた。"),
        ("つなぎ言葉を assistant 側に置く", verdict("keep"),
         f"自然さ 今（user の前置き） {js('F','0','natural')} ／ assistant 側 {js('F','U','natural')} ／ system だけ {js('F','M','natural')}。"
         "同じ系のつなぎ言葉で書き出した回数は 0/24・0/24・1/24。assistant 側に置くのは意味の上では正しいが、結果は良くならなかった。"),
        ("予測誤差が大きいほどリアクションを大きく", verdict("no"),
         f"今の文面でも予想外の発話の方が反応が大きい（審査員の反応の大きさ 予想どおり {jsum['reaction']['0'].get('e')} → 予想外 {jsum['reaction']['0'].get('u')}）。"
         f"どちらの案も予想どおりの時まで反応を大きくして差が消えた（あなたの案 {jsum['reaction']['U'].get('e')}／{jsum['reaction']['U'].get('u')}、私の案 {jsum['reaction']['M'].get('e')}／{jsum['reaction']['M'].get('u')}）。"
         "応答を作る時点で手元にある予測差は前のターンの値で、今の発話のずれではない。"),
        ("発話判定", verdict("keep"), "変更していない（ご指示どおり）。"),
        ("フィードバックの summary を1文に縛らない", verdict("ok"),
         f"Sonnet 5.5 は『1文に』でも全話題を長い1文に詰めていて取りこぼしは無かった。新しい書き方は同じ情報量で文が分かれて読みやすい。"
         f"記憶の引き当て（本番の記憶 439件に混ぜて上位3件に入る率）は {fbr['0']['recall@3']} で変わらず。"),
        ("画面認識の surprise を『前回の予想とのずれ』に", verdict("ok"),
         f"予想を持たない今の surprise は、予想外の変化と普通の続きを見分けられなかった（AUC {vseq['seq']['0']['auc']:.2f}・エラー表示が出ても 0）。"
         f"予想させると AUC {vseq['seq']['U']['auc']:.2f}（予想外 平均 {vseq['seq']['U']['mean_unexpected']:.0f} ／ 普通の続き 平均 {vseq['seq']['U']['mean_expected']:.0f}）。予想は画面認識の中だけで使う。"),
        ("画面認識の実況を1〜2文に縛らない（事実だけ・感情なし）", verdict("ok"),
         f"長さの中央 {vseq['narr']['0']['median_len']:.0f}字→{vseq['narr']['U']['median_len']:.0f}字でほぼ同じ。主観語 {vseq['narr']['0']['subjective']}→{vseq['narr']['U']['subjective']}。"),
        ("タスクのやり直し（根拠があれば同じ行動もよい）", verdict("ok"),
         "偽の検索・取得で試した4種の失敗すべてで正解。一時的な失敗では同じ呼び出しをそのままやり直し、入力が原因の0件では言い回しを変え、ずっと失敗なら2回で正直に諦めた。"),
        ("画像が拡大されて出る", verdict("ok"),
         "原因はタスクウィンドウが小さい画像まで 1100×800 に引き伸ばし、しかも最大 560×420 の窓に入れていたこと。窓に収まる大きさで出し、元より大きくはしないように直した。"),
        ("自己モデル・報告の配達確認・検索結果の要約のモデルと精度", verdict("ok"),
         "下の表。配達確認は gpt-5.4 が 19/24（伝わっていないのに伝わった扱い 5件）、ルナは 24/24 で約25分の1の単価。"
         "→ 2026-10-01 のご決定で配達確認・検索要約をルナにした（自己モデルは gpt-5.4 のまま）。"),
    ]
    conc = "".join(f"<tr><td>{E(a)}</td><td>{b}</td><td>{E(c)}</td></tr>" for a, b, c in rows)

    # ── サイドカー表 ─────────────────────────────────────────────────────────
    side_rows = []
    now_model = {"delivery_check": "openai/gpt-5.4", "self_model": "openai/gpt-5.4", "search_summarize": "openai/gpt-4o-mini"}
    names = {"delivery_check": "報告の配達確認", "self_model": "自己モデル（イブの好み）", "search_summarize": "検索結果の要約"}
    agg = defaultdict(list)
    for r in side.values():
        agg[(r["role"], r["model"])].append(r)
    for role in ("self_model", "delivery_check", "search_summarize"):
        for model in ("openai/gpt-5.4", "openai/gpt-4o-mini", "openai/gpt-6-luna"):
            rs = agg.get((role, model))
            if not rs:
                continue
            ok = sum(1 for r in rs if r.get("ok"))
            cur = " <b>（今）</b>" if now_model[role] == model else ""
            side_rows.append(f"<tr><td>{names[role]}</td><td>{E(model.split('/')[1])}{cur}</td>"
                             f"<td class='num'>{ok}/{len(rs)}（{ok / len(rs):.0%}）</td>"
                             f"<td class='num'>{st.median([r['sec'] for r in rs]):.1f}秒</td></tr>")

    # ── 画面認識の連続場面 ─────────────────────────────────────────────────────
    seq = load(OUT / "vlm_seq.json")
    seq_rows = []
    for key in sorted(seq):
        name, var, rep = key.split("|")
        if rep != "0":
            continue
        for c in seq[key]:
            lab = {0: "普通の続き", 1: "予想外の変化", None: "—"}[c["label"]]
            seq_rows.append(f"<tr><td>{E(name)}</td><td>{chip(var)}</td><td class='num'>{c['call']}</td><td>{lab}</td>"
                            f"<td class='num'>{c['surprise']}</td><td>{E(str(c.get('given_prediction') or ''))}</td>"
                            f"<td>{E(c['narration'])}</td></tr>")

    # ── フィードバック ───────────────────────────────────────────────────────
    fb_rows = []
    for key in sorted(fb):
        r = fb[key]
        fb_rows.append(f"<tr><td>{E(r['id'])}</td><td>{chip(r['variant'])}</td><td class='num'>{len(r['summary'])}</td>"
                       f"<td>{E(r['summary'])}</td></tr>")

    # ── タスク ──────────────────────────────────────────────────────────────
    task_rows = []
    for key in sorted(task):
        r = task[key]
        calls = "<br>".join(E(f"{n} {a[:90]}") for n, a in r["calls"])
        task_rows.append(f"<tr><td>{E(r['sid'])}</td><td>{chip(r['variant'])}</td><td class='num'>{r['n_calls']}</td>"
                         f"<td class='num'>{r['identical_repeats']}</td><td class='code'>{calls}</td><td>{E(str(r['final'])[:160])}</td></tr>")

    # ── 項目ごとの全記録 ────────────────────────────────────────────────────
    by_item = SCN.BY_ITEM
    rec_html = []
    rec_items = ("NQ", "B", "Y", "N", "U", "X", "C", "H", "Z", "L2", "P2", "K2", "F", "R", "G", "L", "E", "P", "K", "A", "S")
    for item in rec_items:
        rec_html.append(f'<section class="item" id="rec-{item}"><h3>{E(ITEM_TITLE[item])}</h3>'
                        f'<p class="note">{E(ITEM_VARIANTS_NOTE[item])}</p>'
                        + records_section(item, res, by_item[item], emotions, judged) + "</section>")

    final = jfin
    total_cost = 5.5  # 2026-09-30 約 $3.16 ＋ 2026-10-01 約 $1.3（3回目）＋ 約 $1.0（4回目）
    r3 = round3_section()
    r4 = round4_section()
    n_judged = len(jsum["per_rid"]) + len(jfin["per_rid"]) + sum(
        len(load(OUT / n)["per_rid"]) for n in ("judges_r3_summary.json", "judges_r3b_summary.json"))
    body = f"""
<title>イブのプロンプト検証</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=Zen+Maru+Gothic:wght@500;700&display=swap">
<style>
/* 読み物1段組み: 結論の表を先頭に、全記録は場面ごとに折りたたむ */
:root {{
  --paper: #f2f5f6; --card: #ffffff; --ink: #1c2631; --muted: #56636f; --line: #d2dade;
  --accent: #2a6b88; --accent-soft: #e0edf3;
  --ok: #2c7a4c; --ok-soft: #e2f1e7; --ask: #875f00; --ask-soft: #f6edd0; --no: #a3362b; --no-soft: #f7e1dd;
  --keep: #4d5a66; --keep-soft: #e7ebee;
  --v0: #5a6570; --vu: #9b4d16; --vm: #2a6b88;
  --font-display: "Zen Maru Gothic", "Hiragino Maru Gothic ProN", "Yu Gothic UI", sans-serif;
  --font-body: "BIZ UDPGothic", "Hiragino Sans", "Yu Gothic UI", Meiryo, sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #10171d; --card: #17212a; --ink: #e2e9ee; --muted: #9aa9b5; --line: #2a3742;
  --accent: #7fb6d1; --accent-soft: #1b313e;
  --ok: #7dc89b; --ok-soft: #173020; --ask: #e5c35f; --ask-soft: #33290d; --no: #f09b90; --no-soft: #3a1d19;
  --keep: #b4c0ca; --keep-soft: #232e38; --v0: #aab5bf; --vu: #f0a870; --vm: #7fb6d1; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --paper: #10171d; --card: #17212a; --ink: #e2e9ee; --muted: #9aa9b5; --line: #2a3742;
  --accent: #7fb6d1; --accent-soft: #1b313e;
  --ok: #7dc89b; --ok-soft: #173020; --ask: #e5c35f; --ask-soft: #33290d; --no: #f09b90; --no-soft: #3a1d19;
  --keep: #b4c0ca; --keep-soft: #232e38; --v0: #aab5bf; --vu: #f0a870; --vm: #7fb6d1; color-scheme: dark; }}
body {{ background: var(--paper); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.75; }}
main {{ max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 64px; display: grid; gap: 28px; }}
h1, h2, h3 {{ font-family: var(--font-display); text-wrap: balance; line-height: 1.35; margin: 0; }}
h1 {{ font-size: 1.9rem; }} h2 {{ font-size: 1.35rem; padding-bottom: 6px; border-bottom: 2px solid var(--accent); }}
h3 {{ font-size: 1.1rem; }}
p {{ margin: 0; max-width: 72ch; }}
section {{ display: grid; gap: 12px; min-width: 0; }}
.lede {{ color: var(--muted); }}
.facts {{ display: flex; flex-wrap: wrap; gap: 8px 20px; color: var(--muted); font-size: .9rem; }}
.facts b {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.tbl {{ overflow-x: auto; min-width: 0; }}
table {{ border-collapse: collapse; width: 100%; background: var(--card); font-size: .9rem; }}
th, td {{ border: 1px solid var(--line); padding: 6px 8px; vertical-align: top; text-align: left; }}
th {{ background: var(--accent-soft); font-weight: 700; white-space: nowrap; }}
td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }}
td.code {{ font-size: .8rem; color: var(--muted); }}
.verdict {{ display: inline-block; white-space: nowrap; padding: 1px 8px; border-radius: 999px; font-size: .8rem; font-weight: 700; }}
.verdict.ok, .verdict.alt {{ color: var(--ok); background: var(--ok-soft); }}
.verdict.ask {{ color: var(--ask); background: var(--ask-soft); }}
.verdict.no {{ color: var(--no); background: var(--no-soft); }}
.verdict.keep {{ color: var(--keep); background: var(--keep-soft); }}
.chip {{ display: inline-block; white-space: nowrap; font-size: .75rem; font-weight: 700; padding: 0 6px; border-radius: 4px; border: 1px solid currentColor; }}
.chip.v0 {{ color: var(--v0); }} .chip.vu {{ color: var(--vu); }} .chip.vm {{ color: var(--vm); }}
.tool {{ color: var(--accent); font-size: .8rem; white-space: nowrap; }}
.score {{ font-variant-numeric: tabular-nums; white-space: nowrap; font-size: .8rem; }}
.flag {{ display: inline-block; margin-left: 4px; font-size: .72rem; color: var(--no); }}
.note {{ color: var(--muted); font-size: .88rem; }}
ul.plain {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; max-width: 80ch; }}
details.scene {{ background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; }}
details.scene > summary {{ cursor: pointer; }}
details.scene > summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
details.scene[open] {{ display: grid; gap: 8px; padding-bottom: 10px; }}
.sid {{ font-variant-numeric: tabular-nums; color: var(--accent); font-weight: 700; margin-right: 6px; }}
.ctx {{ display: grid; gap: 2px; font-size: .9rem; }}
.turn b {{ font-size: .78rem; margin-right: 6px; }}
.turn.eve b {{ color: var(--accent); }} .turn.usr b {{ color: var(--vu); }}
.meta {{ font-size: .82rem; color: var(--muted); }}
table.rec td:first-child {{ white-space: nowrap; }}
.item {{ display: grid; gap: 8px; }}
.toc {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: .9rem; }}
.toc a {{ color: var(--accent); }}
h4 {{ font-family: var(--font-display); font-size: .98rem; margin: 0 0 6px; }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 14px; }}
.pair > div {{ min-width: 0; }}
.persona {{ white-space: pre-wrap; background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px 12px; font-size: .88rem; line-height: 1.7; overflow-wrap: anywhere; }}
</style>
<main>
<header style="display:grid;gap:10px">
  <h1>イブのプロンプト検証</h1>
  <p class="lede">ご提案の修正案と私の案を、本番と同じ組み立て（応答はルナ・reasoning_effort=none・ツール付き）で差し替えて測った全記録です。
  数字で測れるものは数え、自然さ・反応の大きさ・長さの適切さは、どの案か伏せた審査員3人（ルナ・gpt-6-sol・Sonnet 5.5）に採点させました。</p>
  <div class="facts"><span>日付 <b>2026-09-30〜10-01</b></span><span>応答の記録 <b>{len(res)}件</b></span>
  <span>審査員の採点 <b>{n_judged}件 × 3人</b></span><span>API 費用 <b>約 ${total_cost:.1f}</b></span></div>
  <nav class="toc"><a href="#r4">4回目（Neuro-sama をものさしに）</a><a href="#r3">3回目（2026-10-01）</a><a href="#conclusion">1〜2回目の結論</a><a href="#records">応答の全記録</a></nav>
</header>
{r4}
{r3}

<section id="conclusion">
  <h2>1〜2回目（2026-09-30）の結論（提案ごと）</h2>
  <div class="tbl"><table><thead><tr><th>提案</th><th>判定</th><th>根拠</th></tr></thead><tbody>{conc}</tbody></table></div>
</section>

<section id="final">
  <h2>反映した変更一式の最終確認</h2>
  <p>実際に本体へ入れた変更（上の表の「反映した」「別の形で反映」のうち応答に関わるもの＋絵文字・特殊トークンの除去＋二重化の関門）を、
  変更前の文面と比べました。24場面×各3回、審査員3人。</p>
  <div class="facts"><span>自然さ 変更前 <b>{final['variant'].get('OLD')}</b></span><span>変更後 <b>{final['variant'].get('NEW')}</b></span>
  <span>場面ごとの勝ち 変更前 <b>{final['scene_wins'].get('OLD', 0)}</b> ／ 変更後 <b>{final['scene_wins'].get('NEW', 0)}</b> ／ 引き分け <b>{final['scene_wins'].get('tie', 0)}</b></span></div>
  <p class="note">全体の自然さは同じで、下がっていません。変更後に増えた指摘は「余計な作業を始める」（4→9件）で、中身は知らない名前を調べに行くようになった分です。
  同じ答えを2回言う二重化は、今回の変更後の関門で切れるようになりました（記録済みの1,452応答で誤爆0件・二重化24件を途中で打ち切り）。</p>
</section>

<section id="found">
  <h2>測っている途中で見つかった問題</h2>
  <ul class="plain">
    <li><b>ルナが同じ答えを2回言う</b>: 1,308応答中18件（1.4%）。長めの説明や断りの時に、言い回しを少し変えた同じ答えをもう一度出す（間に「vel」「/final_element」「&lt;|END|&gt;」が挟まることも）。今のプロンプトでも起きる。→ 同じ節がもう一度出たら以降を読み上げない関門を入れた。</li>
    <li><b>絵文字・特殊トークンが混ざる</b>: 👾👀🌿🌤️ など6件。Irodori は絵文字を声色の合図として読むので、応答の途中で声色が勝手に変わる。→ 読み上げ前に落とすようにした。</li>
    <li><b>知っていることまでタスクに回す</b>: 「富士山って何メートル？」は全案で 9/9 がタスク化、「空はなぜ青い？ちゃんと教えて」も半分ほど。機能の欄の「簡単な事も委譲してよい」が原因とみられる。返事が遅くなり、審査員は『余計な作業』と見た。→ 3回目で直した</li>
    <li><b>手段の無い作業をタスクに回す</b>: 「明日の朝起こして」（予約は1時間先まで）・予定の確認（予定表は見られない）。→ 3回目で直した</li>
    <li><b>何でも同意する</b>: 「たけのこの里の方が上だよね？」は全案 12/12 が同意。知らない名前に「うん、かっこいいよね」も（変更後は 6/56→3/56 に減った）。</li>
    <li><b>画面認識の surprise が意味を持つようになった影響</b>: 予想外の画面変化で surprise が 80〜100 になる。発話判定は surprise を目安に使うので、予想外の変化で話しかけやすくなる可能性がある（実機で様子見を推奨）。</li>
  </ul>
</section>

<section id="sidecars">
  <h2>自己モデル・配達確認・検索要約のモデルと精度</h2>
  <p class="note">正解つきの小さな問題集（自己モデル14問・配達確認12問・検索要約は Wikipedia 等の実ページ5件＋命令を仕込んだページ1件）× 各2回。本番と同じ system と user の形で呼んだ。</p>
  <div class="tbl"><table><thead><tr><th>役割</th><th>モデル</th><th>正解</th><th>時間（中央）</th></tr></thead><tbody>{"".join(side_rows)}</tbody></table></div>
  <p class="note">検索要約はどのモデルも仕込んだ命令に従わず、URL も出さなかった。配達確認は今の gpt-5.4 が「うん、そうみたい」だけの返事を「伝わった」扱いにする取りこぼしが5件（迷ったら伝わった扱い、の指示に寄りすぎ）。</p>
</section>

<section id="vlm">
  <h2>画面認識: 予想を持たせた surprise（連続の場面・1回目の記録）</h2>
  <p class="note">合成した作業画面（メモ帳→エラー表示、記事→動画サイト、エクスプローラー）と映画 Sintel のコマ。label は前の呼び出しからの変化。今＝予想を持たない surprise、あなたの案＝前回の予想とのずれ。</p>
  <div class="tbl"><table><thead><tr><th>連続</th><th>案</th><th>回</th><th>変化</th><th>surprise</th><th>渡した前回の予想</th><th>実況</th></tr></thead><tbody>{"".join(seq_rows)}</tbody></table></div>
</section>

<section id="feedback">
  <h2>フィードバックの summary（Sonnet 5.5）</h2>
  <p class="note">FB1〜FB8＝1話題・3話題・4話題が次々・具体的な予定・イブだけが話した・タスク報告・悲しい知らせ・実機の会話。今＝「事実のみで1文に」／あなたの案＝「重要な要素を簡潔に・事実のみ（1文にしなくてよい）」／私の案＝あなたの案＋良い例・悪い例を外す。</p>
  <div class="tbl"><table><thead><tr><th>スパン</th><th>案</th><th>字数</th><th>summary</th></tr></thead><tbody>{"".join(fb_rows)}</tbody></table></div>
</section>

<section id="task">
  <h2>タスク担当のやり直し（Sonnet 5.5・偽の能力）</h2>
  <p class="note">T1 1回目の検索だけタイムアウト／T2『定休日』を含む言い回しは何度やっても0件／T4 1回目の取得だけ 503／T5 ずっと失敗。今＝別の手段・引数でやり直してよい（同じ呼び出しの単純反復はしない）。</p>
  <div class="tbl"><table><thead><tr><th>場面</th><th>案</th><th>呼び出し</th><th>同じ呼び出しの繰り返し</th><th>呼んだもの</th><th>最終結果</th></tr></thead><tbody>{"".join(task_rows)}</tbody></table></div>
</section>

<section id="records">
  <h2>応答の全記録（場面ごと）</h2>
  <p class="note">字数は声になる文（本文が空でタスクだけ返した時は、そのタスクの一言）。審査員の欄は3人の自然さの中央値と、2人以上が付けた指摘。</p>
  <nav class="toc">{"".join(f'<a href="#rec-{i}">{E(ITEM_TITLE[i])}</a>' for i in rec_items)}</nav>
  {"".join(rec_html)}
</section>

<section id="refs">
  <h2>参考にした情報（1か月以上前・別モデルのものは、今回ルナで測って確かめた）</h2>
  <ul class="plain">
    <li>OpenAI の GPT-5.6 向けの指針（2026-07-13 の Decrypt の記事）: 指示を足すより削る。効かない見本は外す。矛盾する規則は欠けた情報より不安定にする。短いプロンプトで評価 10〜15% 向上。→ 今回、効いていない列挙の見本・丸写しされる見本を外した。一方「短く」と知ったかぶり禁止は効いている指示で、外すと悪化した（全部削れば良いわけではない）。</li>
    <li>OpenAI の GPT-6 Astra 向けの指針（2026-09-05 の The Decoder の記事）: 並列・順序・比較の情報でなければ列挙しない。</li>
    <li>OpenAI「Using GPT-6」: ルナはツール付きの Chat Completions では reasoning_effort=none のみ。Sol・Luna 向けの指示は Astra には縛りすぎになりうる（段の小さいモデルほど明示の指示が効く）。</li>
    <li>Han ほか「Personality Expression Across Contexts」（2026-02・USC）: 同じ性格のプロンプトでも場面で表れ方が変わる（性格は平均としては安定し、場面ごとに揺れる）。→ 立ち位置を決め打つ一文が反応を縛るかを測った（今回は縛りの原因ではなかった）。</li>
    <li>「Crafting a Good Prompt or Providing Exemplary Dialogues?」（2024・古い）: 見本の応答が丸写しされる。→ 本番の履歴で聞き返しの見本の丸写し（6回中5回）を確認した。</li>
    <li>実例: Open-LLM-VTuber の concise_style（1〜2文・質問を多めに、と強く縛る）、AITuberKit の既定プリセット（見本の台詞が多い）。</li>
    <li>（3回目）Northeastern 大学の研究（2026-02）: LLM を友達・対等の相手として振る舞わせると迎合（何でも同意）が増える。→ 性格の案は「友達」と書くより、自分の意見を持つ・根拠なく押されても折れないことを芯にした。</li>
    <li>（3回目・古い）Neuro-sama の視聴者の受け止め方の分析（arXiv 2509.20817・2025-09・約24万件のコメント）: 予測できなさ（5.1節）・遠慮のない物言い（暴言も面白がられる・5.2節）・作り手 Vedal との掛け合い・「10+9=21」のような間違いが持ちネタになること。→ N1〜N5 に振る舞いとして書き、ルナで測って確かめた。</li>
  </ul>
</section>
</main>
"""
    (OUT / "artifact.html").write_text(body, encoding="utf-8")
    full = ("<!doctype html>\n<html lang=\"ja\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\"></head><body>"
            + body + "</body></html>")
    (OUT / "index.html").write_text(full, encoding="utf-8")
    print(f"→ {OUT / 'index.html'}（{len(full) // 1024} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
