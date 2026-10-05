"""応答LLM の変種（0=今 / U=ユーザ案 / M=アシスタント案）。

各項目は**今のプロンプトからの最小の差分**で作る（項目ごとの効果を切り分けるため）。
まとめ（bundle）は項目の差分を全部重ねたもの。
"""
from __future__ import annotations

import json

from .common import sub

# ── 読み上げ規律（eve/response/style.py の SPEECH_STYLE）の行 ─────────────────────────────
SHORT_LINE = "・話し言葉で短く自然に答える。\n"
ENUM_LINE = "・列挙は「1つ目は…、2つ目は…」のように口頭で述べる。\n"
UNKNOWN_LINE_HEAD = "・聞き取れない語・意味の取れない語・知らない固有名詞を、知ったかぶりで作文しない。"

LENGTH_U = "・話し言葉で自然に答える。長さは場面に合わせる。\n"
LENGTH_M = ("・声に出して自然な話し言葉で話す。長さは場面しだい"
            "（相づちや軽い返事なら短く、説明や話を頼まれたら必要なだけ）。\n")
# 2回目: 「短く」という縛りの代わりに、媒体（声）から長さを決めさせる
LENGTH_M2 = "・話し言葉で自然に答える。長さは場面に合わせる（声で聞いて分かる長さに）。\n"
# 2回目: 聞き返しの見本（そのまま真似される）だけを外す
ASKBACK_TEMPLATE = "それも無理なら「『○○』って聞き取れなかった、どういうこと？」と一度だけ聞き返す"
ASKBACK_PLAIN = "それも無理なら一度だけ聞き返す"
# K-M: 聞き取れない語（→聞き返す）と知らない語（→調べる・人柄の側）を分け、言い回しの見本を外す
UNKNOWN_M = ("・聞き取れなかった語や意味の取れない語は、推測で話を合わせず聞き返す（同じ語を何度も聞き返さない）。"
             "画面について聞かれた時は、聞き取れなかった語より画面に見えているものを優先して答える。")

# ── ペルソナの案（persona.md の水平線より後）──────────────────────────────────────────────
# 中身は手元だけ（`persona_variants_local.py`・.gitignore 済み）。2026-10-06 ユーザ「コミットはコードとシステムプロンプト
# フォーマットだけで、実例データや詳細なペルソナは除外」。無い環境では案は空（今の persona.md だけで回せる）
try:
    from .persona_variants_local import (  # noqa: F401
        K_LINE0, K_LINE_M, K_LINE_U, P_LINE0, P_LINE_M, P_LINE_M2, P_LINE_M3, P_LINE_U,
        PERSONA_0, PERSONA_FULL, PERSONA_N3, PERSONA_N5,
    )
except ImportError:
    K_LINE0 = K_LINE_M = K_LINE_U = P_LINE0 = P_LINE_M = P_LINE_M2 = P_LINE_M3 = P_LINE_U = ""
    PERSONA_0 = PERSONA_N3 = PERSONA_N5 = ""
    PERSONA_FULL: dict = {}


def _norm(s: str) -> str:
    import re

    return re.sub(r"[\s、。，．,.！？!?…・ー〜~「」『』（）()]|イブ", "", s or "")


def similar_line(a: str, b: str) -> float:
    """話しかけどうしの似ている度合い（文字2-gram の Jaccard・0〜1）。実例の話しかけが今の話しかけと同じ場面なら、
    その実例は見せない（同じ場面の実例は丸写しを招く・arXiv 2402.09954）。本番の `eve/persona_examples.py` と同じ式。"""
    x, y = _norm(a), _norm(b)
    gx = {x[i:i + 2] for i in range(len(x) - 1)} or {x}
    gy = {y[i:i + 2] for i in range(len(y) - 1)} or {y}
    return len(gx & gy) / len(gx | gy) if gx | gy else 0.0


def exemplar_pool() -> list[dict]:
    from .common import OUT, load

    return [d for d in load(OUT / "neuro" / "neuro_ja.json")["items"]
            if d.get("split") == "pool" and len(d["ja_a"]) <= 70]


def exemplar_block(seed: str, n: int = 8, *, said: str = "", exclude_sim: float | None = None,
                   force: str | None = None) -> tuple[str, list[str]]:
    """E: Neuro-sama の実際のやりとり（日本語に移したもの・評価の場面とは別の置き場）を毎回ランダムに n 組。
    同じ言い回しを毎回見せると口癖になるので、呼び出しごとに入れ替える。返り値は（ブロック, 見せたイブの返事）。
    - 同じ seed なら n を変えても入れ子（n=2 の組は n=4 の組に含まれる）＝個数だけを比べられる
    - exclude_sim: 今の話しかけ（said）と似た実例は見せない（本番の除外と同じ）
    - force: 丸写しの検証用に、この id の実例を必ず入れる"""
    import random

    pool = exemplar_pool()
    if exclude_sim is not None and said:
        pool = [d for d in pool if similar_line(d["ja_q"], said) < exclude_sim]
    order = pool[:]
    random.Random(seed).shuffle(order)
    picked = order[:n]
    if force:
        f = next(d for d in exemplar_pool() if d["id"] == force)
        picked = [f] + [d for d in picked if d["id"] != force][:max(0, n - 1)]
        random.Random(seed + "|force").shuffle(picked)
    lines = "\n".join(f"ユーザ: {d['ja_q']}\nイブ: {d['ja_a']}" for d in picked)
    block = ("## 話し方の参考（こういうノリで話す。言い回しや話の中身はまねしない）\n" + lines + "\n")
    return block, [d["ja_a"] for d in picked]


def persona_md(*, line1: str = "0", lookup: str = "0") -> str:
    if line1 in PERSONA_FULL:
        return "# イブのペルソナ\n\n---\n\n" + PERSONA_FULL[line1]
    md = PERSONA_0
    line = {"U": P_LINE_U, "M": P_LINE_M, "M2": P_LINE_M2, "M3": P_LINE_M3}.get(line1)
    if line:
        md = sub(md, P_LINE0, line)
    if lookup == "U":
        md = sub(md, K_LINE0, K_LINE_U)
    elif lookup == "M":
        md = sub(md, K_LINE0, K_LINE_M)
    return "# イブのペルソナ\n\n---\n\n" + md


def speech_style(base: str, *, length: str = "0", enum: str = "0", unknown: str = "0") -> str:
    s = base
    line = {"U": LENGTH_U, "M": LENGTH_M, "M2": LENGTH_M2}.get(length)
    if line:
        s = sub(s, SHORT_LINE, line)
    if enum in ("U", "M"):
        s = sub(s, ENUM_LINE, "")
    if unknown == "M":
        head, _, _ = s.partition(UNKNOWN_LINE_HEAD)
        s = head + UNKNOWN_M
    elif unknown == "T":
        s = sub(s, ASKBACK_TEMPLATE, ASKBACK_PLAIN)
    return s


# ── 機能の前置き（context_assembler の # 機能 ブロック） ─────────────────────────────
ACK_SENTENCE = "呼ぶ前に「ちょっと調べるね」等の短い前置きを一言添えてよい。"
# 2回目: 知らない名前・言葉を調べる（delegate_task）を機能の使い方の側に書く（人柄の一文では効かなかった）
LOOKUP_TOOL_ANCHOR = "**委譲する goal は、それ単体で意味が通る一文にする**"
LOOKUP_TOOL_LINE = ("話に出てきた名前や言葉を知らない時は、話を合わせる前に `delegate_task` で調べてよい"
                    "（聞き取れなかっただけの語は調べずに聞き返す）。\n")


def apply_ack(system: str, ack: str) -> str:
    if ack in ("U", "M") and ACK_SENTENCE in system:
        return sub(system, ACK_SENTENCE, "")
    return system


def apply_lookup_tool(system: str, on: bool) -> str:
    if on and LOOKUP_TOOL_ANCHOR in system:
        return sub(system, LOOKUP_TOOL_ANCHOR, LOOKUP_TOOL_LINE + LOOKUP_TOOL_ANCHOR)
    return system


# ── 3回目: 機能の欄の書き直し（T1）。ユーザ 2026-10-01「簡単な知識はタスクに回さず持ってる知識で」「存在しない単語や
# アニメの固有名詞でルナが分からないものは聞き返すより調べた方がいい」「不可能作業はタスク依頼しない・その場で理由を説明」
# ＋ZG02（会話の返事そのものを delegate_task に入れた）への対処 ──
DELEGATE_OLD = ("**タスク（調べ物・状態確認・後回しの予約）は自分でやらず、`delegate_task` に『～して』と"
                "自然文で丸ごと委譲する**（即時も『30秒後に〜』も全部これ・簡単な事も委譲してよい＝賢い"
                "タスク担当がやる）。")
DELEGATE_T1 = ("**タスク（調べ物・状態確認・作業・後回しの予約）は自分でやらず、`delegate_task` に『～して』と"
               "自然文で丸ごと委譲する**（即時も『30秒後に〜』も全部これ）。"
               "ただし一般常識として確かに知っていることは、委譲せずその場で自分の知識で答える"
               "（最新の情報・日によって変わる数字・知らない名前や言葉は委譲して調べる）。"
               "**会話の返事・相づち・ユーザへの質問は委譲しない**（タスク担当はユーザと話せない。それは自分で話す）。")
LOOKUP_T1_LINE = ("話に出てきた名前や言葉を知らない時は、話を合わせたり聞き返したりする前に `delegate_task` で調べる"
                  "（聞き取れなかっただけの崩れた語は調べずに聞き返す）。\n")
CANT_OLD = "「条件を教えてくれたらやる」と引き受け直さない。"
CANT_T1 = ("「条件を教えてくれたらやる」と引き受け直さない。**手段の無いことは `delegate_task` にも回さない**"
           "（回してもできないので、引き受けた時点で嘘になる）。その場で、何が無いからできないのかを短く言う。")
WHEN_OLD = "when_seconds 省略=すぐ、指定=その秒後。"
WHEN_T1 = "when_seconds 省略=すぐ、指定=その秒後（最大3600秒＝1時間先まで。それより先の予約はできない）。"


# T2: T1 で残った失敗への対処。(1) delegate_task の説明文の「どんな簡単な事も自分でやらずここへ」が system と
# 食い違っていた (2) 予定表・スマホ等は手段の一覧に無くても「タスク担当ならできる」と思って回す (3)「できない」と
# 言いながら委譲も出す (4) T1 の調べ方が強すぎて崩れた語まで調べた
DELEGATE_T2 = ("**タスク（調べ物・状態確認・作業・後回しの予約）は自分でやらず、`delegate_task` に『～して』と"
               "自然文で丸ごと委譲する**（即時も『30秒後に〜』も全部これ）。"
               "ただし一般常識として確かに知っていること（有名な山の高さ・国の首都・有名なキャラの名前やタイプ・理科の仕組みなど）は、"
               "委譲せずその場で自分の知識で答える（最新の情報・日によって変わる数字・知らない名前や言葉は委譲して調べる）。"
               "**会話の返事・相づち・ユーザへの質問は委譲しない**（タスク担当はユーザと話せない。それは自分で話す）。")
LOOKUP_T2_LINE = ("話に出てきた名前や言葉（作品名・人名・店名・新しい言葉など）を知らない時は、話を合わせる前に `delegate_task` で調べる。"
                  "言葉として崩れていて何を言いたいのか分からない時は、調べずに聞き返す。\n")
NEVER_SYS_OLD = "（メール送信、買い物、外部サービスへの書き込み等）"
NEVER_SYS_T2 = "（メール送信・他の人への連絡、買い物、外部サービスへの書き込み、予定表の確認、スマホや家電の操作等）"
CANT_T2 = ("「条件を教えてくれたらやる」と引き受け直さない。**手段の無いことは `delegate_task` にも回さない**"
           "（回してもできないので、引き受けた時点で嘘になる）。その場で、何が無いからできないのかを短く言う。"
           "できないと伝えた依頼で `delegate_task` を呼ばない。")
EASY_OLD = "どんな簡単な事も自分でやらずここへ。"
EASY_T2 = "一般常識で答えられることはここに回さず自分で答える。会話の返事やユーザへの質問も入れない。"
NEVER_TOOL_OLD = "（メール送信・買い物・外部サービスへの書き込みなどはできない）"
NEVER_TOOL_T2 = "（メール送信・他の人への連絡・買い物・外部サービスへの書き込み・予定表の確認・スマホや家電の操作などはできない）"


# T3: T2 の副作用への対処。(1) 調べ方に「作品名・人名など」と例を入れたら、知っている作品の人物まで調べた（ヒンメル 3/3）
# → 例を外して「自分が知らない時だけ」に (2)「30分後に起こして」（1時間以内＝できる）まで断った → 予約は時間になったら
# イブが声で知らせる、と説明文に書く
LOOKUP_T3_LINE = ("話に出てきた名前や言葉を自分が知らない時は、話を合わせる前に `delegate_task` で調べる"
                  "（知っているものは調べずに話す。言葉として崩れていて何を言いたいのか分からない時は、調べずに聞き返す）。\n")
WHEN_T3 = ("when_seconds 省略=すぐ、指定=その秒後（最大3600秒＝1時間先まで。その時間になったらイブが声で知らせる。"
           "それより先の予約はできない）。")


def apply_tools(system: str, tools: list | None, mode: str) -> tuple[str, list | None]:
    if mode not in ("T1", "T2", "T3") or DELEGATE_OLD not in system:
        # 2026-10-01 に T3 を本番へ反映した＝今の本番（0）は T3。機能の欄がある場面で旧文言が見つからないのに
        # T 変種を頼まれたら、黙って「0 と同じ物」を測らないように止める
        if mode in ("T1", "T2", "T3") and "# 機能（Call-Function）の使い方" in system:
            raise RuntimeError(f"{mode}: 本番は既に T3 を反映済み（旧文言が無い）。比べるなら variants.py を今の本番に合わせる")
        return system, tools
    if mode == "T3":
        s = sub(system, DELEGATE_OLD, DELEGATE_T2)
        s = sub(s, LOOKUP_TOOL_LINE, LOOKUP_T3_LINE)
        if CANT_OLD in s:
            s = sub(s, CANT_OLD, CANT_T2)
        if NEVER_SYS_OLD in s:
            s = sub(s, NEVER_SYS_OLD, NEVER_SYS_T2)
        new_tools = []
        for t in tools or []:
            t = json.loads(json.dumps(t))
            fn = t.get("function") or {}
            if fn.get("name") == "delegate_task":
                d = fn.get("description", "")
                d = sub(d, WHEN_OLD, WHEN_T3)
                d = sub(d, EASY_OLD, EASY_T2)
                d = sub(d, NEVER_TOOL_OLD, NEVER_TOOL_T2)
                fn["description"] = d
            new_tools.append(t)
        return s, new_tools
    if mode == "T1":
        s = sub(system, DELEGATE_OLD, DELEGATE_T1)
        s = sub(s, LOOKUP_TOOL_LINE, LOOKUP_T1_LINE)
        if CANT_OLD in s:
            s = sub(s, CANT_OLD, CANT_T1)
    else:
        s = sub(system, DELEGATE_OLD, DELEGATE_T2)
        s = sub(s, LOOKUP_TOOL_LINE, LOOKUP_T2_LINE)
        if CANT_OLD in s:
            s = sub(s, CANT_OLD, CANT_T2)
        if NEVER_SYS_OLD in s:
            s = sub(s, NEVER_SYS_OLD, NEVER_SYS_T2)
    new_tools = []
    for t in tools or []:
        t = json.loads(json.dumps(t))
        fn = t.get("function") or {}
        if fn.get("name") == "delegate_task":
            d = fn.get("description", "")
            d = d.replace(WHEN_OLD, WHEN_T1)
            if mode == "T2":
                d = sub(d, EASY_OLD, EASY_T2)
                d = sub(d, NEVER_TOOL_OLD, NEVER_TOOL_T2)
            fn["description"] = d
        new_tools.append(t)
    return s, new_tools


# ── 画面 ────────────────────────────────────────────────────────────────────────
ABSENT_BLOCK = (
    "# 画面（今この瞬間）\n（画面情報なし：今この瞬間の画面は届いていない）\n"
    "**画面に何が映っているかを、推測や過去の記憶から作らない**"
    "（# 参照 に出てくる画面の話は過去のもので、今の画面ではない）。"
    "画面の内容そのものを聞かれたら「今は画面が見えていない」と正直に言う。"
    "画面以外（会話の流れ・相手の様子）から言えることは、これまでどおり普通に話してよい。"
)
SCREEN_NOTE = "（会話に必要なら自然に織り込む。話題と無関係なら無理に触れなくてよい＝画面に引っ張られすぎない）"
UNREAD_M = (
    "# 画面\n（今は画面の中身を読み取れていない）\n"
    "画面に何が映っているかを、推測や過去の記憶から作らない（# 参照 に出てくる画面の話は過去のもの）。"
    "画面のことを聞かれたら、今は読み取れていないと伝える。"
    "画面以外（会話の流れ・相手の様子）から言えることは普通に話してよい。"
)


def apply_screen(system: str, screen: str, sc: dict) -> str:
    """sc["screen"] = {"state": "static"|"unknown"|..., "narration":..., "age":秒}。
    本番は自発発話・報告で画面が静止していると ABSENT_BLOCK を出す（fresh_vision の TTL 7秒）。"""
    st = (sc.get("screen") or {})
    if screen == "0" or ABSENT_BLOCK not in system:
        return system
    if st.get("state") == "static":
        if screen == "U":
            new = f"# 画面（前回から変化なし）\n{st['narration']}\n{SCREEN_NOTE}"
        else:
            new = f"# 画面（{int(st['age'])}秒前から変化なし）\n{st['narration']}\n{SCREEN_NOTE}"
        return sub(system, ABSENT_BLOCK, new)
    # 本当に読めていない（起動直後・キャプチャ停止・真っ黒 等）
    if screen == "U":
        return sub(system, ABSENT_BLOCK + "\n\n", "") if ABSENT_BLOCK + "\n\n" in system \
            else sub(system, ABSENT_BLOCK, "")
    return sub(system, ABSENT_BLOCK, UNREAD_M)


# ── つなぎ言葉（非依存フィラー）の渡し方 ─────────────────────────────────────────────
FILLER_RULE_U = (
    "# つなぎ言葉\n"
    "あなたはユーザの話を聞き終えた直後、考えている間のつなぎ言葉（「あー」「えー」など）を"
    "既に声に出していることがある。その時は会話の最後に、そのつなぎ言葉だけの assistant メッセージがある。"
    "応答はその続きとして話し、**同じつなぎ言葉や似たつなぎ言葉で書き出さない**。"
    "つなぎ言葉そのものを読み上げたり、それについて説明したりしない。"
)


def filler_rule_m(filler: str) -> str:
    return ("# いま口にしたつなぎ言葉\n"
            f"あなたはユーザの話を聞き終えて、考えながら「{filler}」と既に声に出した。"
            "応答はその続きとして話す（同じつなぎ言葉や似たつなぎ言葉で始めない）。")


def apply_filler(messages: list[dict], filler_mode: str, filler: str | None) -> list[dict]:
    if filler_mode == "0" or not filler:
        return messages
    from eve.context_assembler import FILLER_PREFIX, FILLER_RULE

    msgs = [dict(m) for m in messages]
    prefix = FILLER_PREFIX.format(filler) + "\n"
    last = msgs[-1]
    assert last["role"] == "user" and last["content"].startswith(prefix), "つなぎ言葉の前置きが無い"
    last["content"] = last["content"][len(prefix):]
    if filler_mode == "U":
        msgs[0]["content"] = sub(msgs[0]["content"], FILLER_RULE, FILLER_RULE_U)
        msgs.append({"role": "assistant", "content": filler})
    else:
        msgs[0]["content"] = sub(msgs[0]["content"], FILLER_RULE, filler_rule_m(filler))
    return msgs


# ── 予測誤差でリアクションを変える ─────────────────────────────────────────────────
REACT_U = "（予測差＝予想と現実のずれ。ずれが大きいほど、リアクションを大きくする）"


def react_m(prediction: str) -> str:
    return ("# 直前の予想（あなたがこの発話の前に予想していたこと）\n"
            f"{prediction}\n"
            "（相手の今の発話がこの予想から外れているほど、驚きや感情の動きを大きく出す。"
            "予想どおりなら落ち着いて返してよい。予想の中身そのものは説明しない）")


def apply_react(system: str, react: str, sc: dict) -> str:
    lf = sc.get("last_feedback")
    if react == "0" or not lf:
        return system
    block = f"# 直近フィードバック\n{lf}"
    if react == "U":
        return sub(system, block, block + "\n" + REACT_U)
    return sub(system, block, block + "\n\n" + react_m(sc["last_prediction"]))


# ── 変種の定義 ──────────────────────────────────────────────────────────────────
# 項目キー: length / enum / persona / lookup / ack / screen / filler / react
BASE = dict(length="0", enum="0", persona="0", lookup="0", ack="0", screen="0", filler="0", react="0",
            unknown="0", lookup_tool=False)
ITEM_VARIANTS = {
    "L": {"0": {}, "U": {"length": "U"}, "M": {"length": "M"}},
    "E": {"0": {}, "U": {"enum": "U"}},
    "P": {"0": {}, "U": {"persona": "U"}, "M": {"persona": "M"}},
    "K": {"0": {}, "U": {"lookup": "U"}, "M": {"lookup": "M"}},
    "A": {"0": {}, "U": {"ack": "U"}},
    "S": {"0": {}, "U": {"screen": "U"}, "M": {"screen": "M"}},
    "F": {"0": {}, "U": {"filler": "U"}, "M": {"filler": "M"}},
    "R": {"0": {}, "U": {"react": "U"}, "M": {"react": "M"}},
    # まとめ: 全項目の差分を重ねる
    "G": {"0": {},
          "U": {k: "U" for k in ("length", "enum", "persona", "lookup", "ack", "screen", "filler", "react")},
          "M": {"length": "M", "enum": "U", "persona": "M", "lookup": "M", "ack": "U",
                "screen": "M", "filler": "M", "react": "M"}},
    # ── 2回目 ──
    # 性格（本番の感情判定の値を入れた状態で）: 0 / U / M2＝友達であってアシスタントではない / M3＝気持ちの幅を書く
    "P2": {"0": {}, "U": {"persona": "U"}, "M2": {"persona": "M2"}, "M3": {"persona": "M3"}},
    # 長さ: 0 / U / M2＝「短く」の代わりに声で聞いて分かる長さ
    "L2": {"0": {}, "U": {"length": "U"}, "M2": {"length": "M2"}},
    # 調べる: 0 / T＝聞き返しの見本だけ外す / T+＝見本を外す＋機能の使い方に「知らない名前は調べてよい」
    "K2": {"0": {}, "T": {"unknown": "T"}, "T+": {"unknown": "T", "lookup_tool": True}},
}


# ── 反映後の確認（Z）: 本体に入れた変更一式（NEW＝今のコードそのまま）と、変更前（OLD＝差し替えで戻す）──
MARKDOWN_LINE = "・マークダウンや記号（*, #, `, 「」, 箇条書きの - や 1. など）を使わない。\n"
FUNC_HEAD_NEW = "必要なら提供された関数を呼んでよい。関数名・引数・JSON"
FUNC_HEAD_OLD = "必要なら提供された関数を呼んでよい。" + ACK_SENTENCE + "関数名・引数・JSON"
UNREAD_PROD = (
    "# 画面\n（今は画面の中身を読み取れていない）\n"
    "**画面に何が映っているかを、推測や過去の記憶から作らない**"
    "（# 参照 に出てくる画面の話は過去のもので、今の画面ではない）。"
    "画面のことを聞かれたら、今は読み取れていないと伝える。"
    "画面以外（会話の流れ・相手の様子）から言えることは、これまでどおり普通に話してよい。"
)


def to_old(system: str, sc: dict) -> str:
    """今のコードで組んだ system を、変更前（2026-09-30 の反映前）の文面に戻す。"""
    s = sub(system, MARKDOWN_LINE, MARKDOWN_LINE + ENUM_LINE)
    s = sub(s, ASKBACK_PLAIN, ASKBACK_TEMPLATE)
    if FUNC_HEAD_NEW in s:  # ユーザ発話への応答だけ（機能の欄がある時）
        s = sub(s, FUNC_HEAD_NEW, FUNC_HEAD_OLD)
        s = sub(s, LOOKUP_TOOL_LINE, "")
    st = sc.get("screen") or {}
    if UNREAD_PROD in s:
        s = sub(s, UNREAD_PROD, ABSENT_BLOCK)
    elif st.get("state") == "static" and sc.get("kind") in ("auto", "report"):
        # 変更前は自発発話・報告で静止画面を渡さず「画面情報なし」だった
        head = s.index("# 画面（今この瞬間）")
        end = s.index(SCREEN_NOTE, head) + len(SCREEN_NOTE)
        s = s[:head] + ABSENT_BLOCK + s[end:]
    return s


ITEM_VARIANTS["Z"] = {"NEW": {}, "OLD": {"old": True}}
# 3回目（2026-10-01）: 性格は 0 / N1 / N2、機能の欄は 0 / T1（今の本番＝昨日反映した分を含む）
for _it in ("B", "Y"):
    ITEM_VARIANTS[_it] = {"0": {}, "N1": {"persona": "N1"}, "N2": {"persona": "N2"}, "N3": {"persona": "N3"}}
for _it in ("N", "U", "X", "C"):
    ITEM_VARIANTS[_it] = {"0": {}, "T1": {"tools_mode": "T1"}, "T2": {"tools_mode": "T2"}, "T3": {"tools_mode": "T3"}}
# 見ていない場面での確かめ: 今の本番（0）と T3 だけ
ITEM_VARIANTS["H"] = {"0": {}, "T3": {"tools_mode": "T3"}}
# 4回目（2026-10-01・Neuro-sama の実際の返事をものさしに）: N4＝決め台詞なしの傾向で書いた性格 / E＝今の性格＋実際のやりとり8組
# （毎回入れ替え）/ NE＝N4＋実際のやりとり。B・Y（自前の場面）にも同じ案を足す
_R4 = {"N4": {"persona": "N4"}, "E": {"exemplars": 8}, "NE": {"persona": "N4", "exemplars": 8},
       "N5": {"persona": "N5"}, "N5E": {"persona": "N5", "exemplars": 8}}
ITEM_VARIANTS["NQ"] = {"0": {}, "N3": {"persona": "N3"}, **_R4}
for _it in ("B", "Y"):
    ITEM_VARIANTS[_it].update(_R4)
# 5回目（2026-10-01・ユーザ「N5＋実例で」「実例は何個入れるのが最適か」）: 性格は N6 に固定し、実例の個数だけを変える。
# 実例は今の話しかけと似たものを見せない（本番と同じ除外）
for _it in ("NQ", "B", "Y"):
    ITEM_VARIANTS[_it].update({f"K{k}": ({"persona": "N6", "exemplars": k, "exclude_sim": 0.5} if k else {"persona": "N6"})
                               for k in (0, 2, 4, 8, 16)})
# N5E の「冗談の作り話はしてよい」が、事実の場面（一般常識・知らない名前は調べる・崩れた語は聞き返す）を壊さないかの確かめ。
# 本番は 2026-10-01 から T3 の文面なので、N5E は T3 の上に性格を差し替えたもの（比べる相手は各項目の T3）
for _it in ("N", "U", "H"):
    ITEM_VARIANTS[_it]["N5E"] = {"persona": "N5", "exemplars": 8}


def resolve(item: str, variant: str) -> dict:
    cfg = dict(BASE)
    cfg.update(ITEM_VARIANTS[item][variant])
    return cfg
