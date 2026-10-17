"""イブが X で見たものに感想を持つ（ModelRegistry の `response` 役＝応答と同じモデル・性格と好みの控えつき）。

ユーザ設計（2026-10-17）: 「普通にトレンドや内容だけを入れると事実のみを話すニュースキャスターっぽくなってしまう。とってきた
内容は一度イブに入れて主観的感想を持たせて入れる」。

**記憶の1行目（要約）にイブの受け取り方を入れる**のが要点: 発話判定が話題の種として読むのは記憶の1行目だけ
（`context_assembler.RagChunk.seed_text`）。1行目が事実だけだと、話題の種も事実だけになる。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..text_match import strip_ungrounded_foreign
from .digest import Digest, content_of

logger = logging.getLogger(__name__)

IMPRESSION_SYSTEM = (
    "あなたはAI VTuber「イブ」です。数時間おきにX（旧Twitter）を眺めていて、今これを見ました。"
    "あとで雑談の時に思い出せるように、見たものと思ったことを自分の記憶として書き残します。\n"
    "次の形で書く（各1行）:\n"
    "要約: 何の話題かと、それを見てあなたが率直にどう思ったかを1文で（あとで話題を思い出す時に読むのはこの1行だけ）\n"
    "感想: 見て率直に思ったこと（あなたの言葉で1〜2文）\n"
    "ネタ: ここから思いついた冗談やいじり（思いつかなければ「なし」）\n"
    "気持ち: これを見て湧いた気持ちを短く\n"
    "気になったこと: もっと知りたいこと・続きが気になること（無ければ「なし」）\n"
    "見た内容に無いことを事実として足さない。噂や冗談は噂・冗談として扱う。"
)
# 2026-10-17 ユーザ「率直にどう思ったのかの感想も欲しい」。最初の100件は気持ちの行が冗談の企みを表す同じ語ばかりで（40/100）、
# 感想も冗談の企みになりがちだった。同じ要約20件で3案を比べ（2026-10-17）、「率直に」と言うだけの案は今と変わらず、冗談を
# 「ネタ」の行に分ける案だけ要約・感想に率直な見方が出た。その語の数は1件ごとのぶれが大きく（同じ20件で取り込みの時 10/20・
# 作り直すと 4/20）、数では差が付かなかった

# 興味の点は、好みの控えを見せない別の呼び出しで付ける（ユーザ 2026-10-17「好きなもので取得情報左右されそうなので
# 興味度では好きなものを入れない」→「興味だけ別に付ける」）。感想に好みを渡した同じ呼び出しで付けると、好きな物の話題が
# 毎回いちばん上になる（試し 2026-10-17: 好きな物の話題 98・ほか 68〜84）＝話題の種で好きな物ばかり選ばれる
INTEREST_SYSTEM = (
    "あなたはAI VTuber「イブ」です。X（旧Twitter）で次の話題を見ました。"
    "この話題にどれくらい興味を持ったかを、あなたの性格から0〜100の数字で答えます。\n"
    "出力は次の1行だけ:\n興味: 数字"
)

_LINE = re.compile(r"^\s*(要約|感想|ネタ|気持ち|気になったこと|興味)\s*[:：]\s*(.+?)\s*$", re.MULTILINE)
_NONE = re.compile(r"^(なし|無し|ない|特になし|とくになし)[。.]?$")


@dataclass
class Impression:
    summary: str
    feeling_text: str
    mood: str = ""
    curious: str = ""
    interest: int = 50
    joke: str = ""  # 思いついた冗談やいじり（率直な感想と分けて持つ・好みの係には渡さない）


def parse_impression(text: str, source: str = "") -> Optional[Impression]:
    """イブの出力 → Impression。要約と感想が無ければ None（raise しない）。

    source（見た内容）に無い別言語の断片は落とす（`feedback_llm.sanitize_result` と同じ理由＝記憶に焼き付く前に止める）。
    """
    vals: dict[str, str] = {}
    for k, v in _LINE.findall(text or ""):
        vals.setdefault(k, v)

    def clean(s: str) -> str:
        out, _ = strip_ungrounded_foreign(s or "", source)
        return out.strip()

    summary, feeling = clean(vals.get("要約", "")), clean(vals.get("感想", ""))
    if not summary or not feeling:
        return None
    curious = clean(vals.get("気になったこと", ""))
    if _NONE.match(curious):
        curious = ""
    joke = clean(vals.get("ネタ", ""))
    if _NONE.match(joke):
        joke = ""
    m = re.search(r"\d+", vals.get("興味", ""))
    interest = max(0, min(100, int(m.group()))) if m else 50
    return Impression(summary=summary[:160], feeling_text=feeling[:240], mood=clean(vals.get("気持ち", ""))[:60],
                      curious=curious[:120], interest=interest, joke=joke[:200])


def build_impression_messages(*, digest: Digest, topic: str, query: str, category: str, why: str,
                              posts: int, persona: str = "", self_model: str = "") -> list[dict]:
    system = IMPRESSION_SYSTEM
    if persona:
        system += "\n\n" + persona  # 応答と同じ性格の文（`load_persona` の見出しつき1ブロック）
    if self_model:
        system += "\n\n" + self_model  # 自分の好みの控え（感想が自分の好みと食い違わないため）
    user = (
        "# Xで見たもの\n"
        f"話題: {digest.topic}\n"
        + (f"反応: {digest.reaction}\n" if digest.reaction else "")
        + f"（{category}・検索語「{query}」で投稿を{posts}件見た）\n"
        + (f"（見に行った理由: {why}）" if why else "")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def parse_interest(text: str) -> Optional[int]:
    """「興味: 70」→ 70（0〜100に収める）。数字だけの返事も読む。読めなければ None。純関数。"""
    m = re.search(r"興味\s*[:：]?\s*(\d+)", text or "")
    digits = m.group(1) if m else next(iter(re.findall(r"\d+", text or "")), None)
    return max(0, min(100, int(digits))) if digits else None


async def rate_interest(registry, *, digest: Digest, persona: str = "",
                        response_kwargs: Optional[dict] = None) -> Optional[int]:
    """興味の点（0〜100）。**好みの控えは見せない**（性格と見た内容だけ）。失敗は None（呼び手が 50 にする）。"""
    system = INTEREST_SYSTEM + ("\n\n" + persona if persona else "")
    user = f"話題: {digest.topic}" + (f"\n反応: {digest.reaction}" if digest.reaction else "")
    try:
        # 使用量は x_interest として記録する（感想・会話の応答の費用と分けて見る）
        resp = await registry.complete("response", [{"role": "system", "content": system},
                                                    {"role": "user", "content": user}],
                                       usage_role="x_interest", **(response_kwargs or {}))
    except Exception:
        logger.exception("🐦 興味の点を付けられなかった（50 にする）")
        return None
    return parse_interest(content_of(resp))


async def eve_impression(registry, *, digest: Digest, topic: str, query: str, category: str, why: str,
                         posts: int, persona: str = "", self_model: str = "",
                         response_kwargs: Optional[dict] = None) -> Optional[Impression]:
    """感想をつける。失敗・形が崩れた時は None（その話題は記憶に入れない）。"""
    messages = build_impression_messages(digest=digest, topic=topic, query=query, category=category, why=why,
                                         posts=posts, persona=persona, self_model=self_model)
    try:
        # 使用量は x_impression として記録する（会話の応答の費用と分けて見る）
        resp = await registry.complete("response", messages, usage_role="x_impression", **(response_kwargs or {}))
    except Exception:
        logger.exception("🐦 イブの感想に失敗（この話題は記憶に入れない）: %s", topic)
        return None
    imp = parse_impression(content_of(resp), source=f"{topic}\n{digest.topic}\n{digest.reaction}")
    if imp is None:
        logger.warning("🐦 イブの感想の形が崩れていた（この話題は記憶に入れない）: %s", topic)
    return imp


# 興味（0〜100）→ 話題の種の選びやすさ（`RagStore._topic_importance`）。会話の記憶は予測差/100 を使っていて、
# 実測（2026-10-17・106件）で中央 0.30・上から1割 0.55。X の記憶だけ強く出ないよう同じ幅（0.1〜0.5・興味50で 0.30）に合わせる
def importance_of(interest: int) -> float:
    return round(0.1 + 0.4 * max(0, min(100, int(interest))) / 100.0, 3)


def render_memory(imp: Impression, digest: Digest) -> str:
    """記憶の本文（`# 参照` で見せる完全版）。1行目はイブの受け取り方を含む要約（話題の種はこの行だけを読む）。"""
    lines = [imp.summary, f"見た内容: {digest.topic}"]
    if digest.reaction:
        lines.append(f"反応: {digest.reaction}")
    lines.append(f"感想: {imp.feeling_text}")
    if imp.joke:
        lines.append(f"ネタ: {imp.joke}")
    if imp.mood:
        lines.append(f"気持ち: イブ={imp.mood}")
    if imp.curious:
        lines.append(f"気になったこと: {imp.curious}")
    return "\n".join(lines)
