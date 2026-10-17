"""情報収集AI — 今回イブが X で見に行く話題を選ぶ（ModelRegistry の `x_curate` 役）。

ユーザ設計（2026-10-17）: 「直近・過去の記憶からトレンドを調べたいことがあればそのことについて、特に無ければユーザの趣味や
イブの趣向、最新のトレンドランキングから持ってくる。バランス良く」。

**バランスは件数の決まりにしない**（イブに決めさせる・事実を渡す方針）。代わりに、最近仕入れた話題とその種類の内訳、
今の記憶に多い話題（会話の記憶のタグの件数）を事実として渡す。記憶がゲームに偏っている間はそれが見える。
ユーザのタイムラインは「ユーザが何に興味があるか」を知る材料にだけ使う（投稿の本文は記憶に入れない・ユーザ決定 2026-10-17）。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from ..clock import elapsed_wall, humanize, now_iso
from ..persona import persona_for
from ..search.injection import sanitize_lines

logger = logging.getLogger(__name__)

# 話題の種類（ユーザの挙げた4つ）。最初の取り込み（`tools/x_bootstrap.py`）だけ「いろいろ」も使う
CATEGORIES = ("トレンド", "記憶の続き", "ユーザの趣味", "イブの好み")
CATEGORY_MISC = "いろいろ"
CATEGORY_TREND = "トレンド"

CURATE_SYSTEM = (
    "あなたはAI VTuber「イブ」の情報収集係です。イブは数時間おきにX（旧Twitter）を眺めて、雑談や配信で話せる話題を"
    "仕入れています。今回イブが見に行く話題を選びます。選んだ話題ごとにXで投稿を検索して読み、イブが感想を持って"
    "記憶に残します。\n"
    "\n"
    "話題の種類:\n"
    "- トレンド: 今の日本のトレンドから\n"
    "- 記憶の続き: 最近の会話に出たことや、前にXで見て「もっと知りたい」と思ったことの、その後や反応\n"
    "- ユーザの趣味: ユーザが興味を持っていそうなこと（ユーザのタイムラインや会話から分かること）\n"
    "- イブの好み: イブの好みや性格から、イブが食いつきそうなこと\n"
    "{misc}"
    "\n"
    "選び方:\n"
    "- 「最近仕入れた話題」の種類の内訳を見て、種類が偏らないように選ぶ。\n"
    "- 最近仕入れた話題と同じ話題は選ばない（続報を追う時は、何の続きかを理由に書く）。\n"
    "- 「今の記憶に多い話題」は、イブがもうよく話している話題。そこに寄せすぎない。\n"
    "- ユーザのタイムラインは、ユーザが何に興味を持っているかを知る材料。投稿した人の個人的な話や私生活は話題にしない。\n"
    "- Xで今まさに話されていて、検索すれば投稿が読める話題を選ぶ。トレンドの名前だけでは何の話か分からないものも、"
    "検索すれば分かるので選んでよい。\n"
    "\n"
    "出力は次の形の行だけを{n}行（良い順。上から使い、投稿が見つからない時は次の行を使う）:\n"
    "pick: 種類 | 話題（短い名詞句） | 検索語 | 選んだ理由（1文）\n"
    "検索語はXの検索にそのまま入れる1〜3語（空白区切り）。トレンドから選ぶ時はトレンドの名前をそのまま入れる。"
)
MISC_LINE = "- いろいろ: 上のどれにも当てはまらない分野（分野は自由。今の記憶に多い話題とは別の分野）\n"
PERSONA_HOW = "イブが食いつきそうな話題を考える材料"

_PICK = re.compile(r"^\s*pick\s*[:：]\s*(.+?)\s*[|｜]\s*(.+?)\s*[|｜]\s*(.+?)\s*[|｜]\s*(.+?)\s*$",
                   re.IGNORECASE | re.MULTILINE)
_NORM_DROP = re.compile(r"[\s#＃・「」『』（）()\[\]【】、。,.!！?？]")


@dataclass
class Pick:
    category: str
    topic: str
    query: str
    why: str = ""

    @property
    def is_trend(self) -> bool:
        return self.category == CATEGORY_TREND


def normalize_topic(s: str) -> str:
    """同じ話題かどうかを比べるための形（全角半角・大文字小文字・空白と記号をそろえる）。純関数。"""
    return _NORM_DROP.sub("", unicodedata.normalize("NFKC", s or "").lower())


def _category(raw: str, allowed: tuple[str, ...]) -> Optional[str]:
    raw = unicodedata.normalize("NFKC", raw or "").strip()
    for c in allowed:  # 書き出しが種類の名前なら、それ（「イブの好み（トレンド寄り）」をトレンドにしない）
        if raw.startswith(c):
            return c
    for c in allowed:
        if c in raw:
            return c
    return None


def parse_picks(text: str, *, bootstrap: bool = False) -> list[Pick]:
    """LLM 出力 → 候補の並び（**raise しない**・崩れた行と種類の分からない行と同じ話題の2回目は捨てる）。"""
    allowed = CATEGORIES + ((CATEGORY_MISC,) if bootstrap else ())
    out: list[Pick] = []
    seen: set[str] = set()
    for m in _PICK.finditer(text or ""):
        cat = _category(m.group(1), allowed)
        topic, query, why = (m.group(i).strip() for i in (2, 3, 4))
        key = normalize_topic(topic)
        if cat is None or not key or not query.strip() or key in seen:
            continue
        seen.add(key)
        out.append(Pick(cat, topic[:40], " ".join(query.split())[:80], why[:120]))
    return out


def _when(iso: str, now: str) -> str:
    return humanize(elapsed_wall(iso, now)) if iso else "いつか不明"


def _lines(rows: list[str], empty: str) -> str:
    return "\n".join(rows) if rows else empty


def build_curate_messages(*, trends, timeline, recent, older, curious, history, bias, self_model: str = "",
                          persona: str = "", count: int = 3, candidates: int = 5, bootstrap: bool = False,
                          now: Optional[str] = None) -> list[dict]:
    """情報収集AI への messages。材料はすべて事実として渡す（どれを選ぶかは書かない）。

    trends: [Trend] / timeline: [Post]（本文だけ・投稿者は伏せる） / recent・older: [(iso, 要約)] /
    curious: [(iso, 話題, 気になったこと)] / history: [dict(at, category, topic)] 新しい順 / bias: [(タグ, 件数)]
    """
    n = now or now_iso()
    trend_rows = [f"{i}. {t.name}" + (f"（{t.count:,}件）" if t.count else "") for i, t in enumerate(trends or [], 1)]
    # タイムラインは外部の文（本人がフォローしている人の投稿）。指示文の形の行は落とし、URL も消す
    tl_raw = [f"・{' '.join(p.text.split())[:140]}" for p in timeline or []]
    tl_rows, dropped = sanitize_lines(tl_raw)
    if dropped:
        logger.info("🐦 タイムラインの投稿 %d 件を指示文の形で除いた", dropped)
    recent_rows = [f"・[{_when(i, n)}] {s}" for i, s in recent or []]
    older_rows = [f"・[{_when(i, n)}] {s}" for i, s in older or []]
    curious_rows = [f"・[{_when(i, n)}] {t}: {c}" for i, t, c in curious or []]
    hist = list(history or [])
    hist_rows = [f"・[{_when(h.get('at', ''), n)}・{h.get('category', '')}] {h.get('topic', '')}" for h in hist[:20]]
    tally: dict[str, int] = {}
    for h in hist[:20]:
        tally[h.get("category", "")] = tally.get(h.get("category", ""), 0) + 1
    kinds = CATEGORIES + ((CATEGORY_MISC,) if bootstrap else ())
    tally_line = "・".join(f"{c} {tally.get(c, 0)}" for c in kinds)
    bias_line = "・".join(f"{t} {c}" for t, c in bias or []) or "（まだ少ない）"
    # 好みの控え（`SelfModelStore.render`）は応答向けの見出しと注記つきなので、項目の行だけを渡す
    likes = "\n".join(ln for ln in (self_model or "").splitlines() if ln.startswith("- "))
    user = (
        f"# 今の日本のトレンド（順位・投稿数）\n{_lines(trend_rows, '（取れなかった）')}\n\n"
        f"# ユーザのタイムラインに最近流れていた投稿（{len(tl_rows)}件・投稿者は伏せている）\n"
        f"{_lines(tl_rows, '（使っていない）')}\n\n"
        f"# 最近の会話（ユーザとイブ・要約）\n{_lines(recent_rows, '（なし）')}\n\n"
        f"# 前の会話から（無作為にいくつか）\n{_lines(older_rows, '（なし）')}\n\n"
        f"# 前にXで見て、もっと知りたいと思ったこと\n{_lines(curious_rows, '（なし）')}\n\n"
        f"# 最近仕入れた話題（新しい順）\n{_lines(hist_rows, '（まだ無い）')}\n"
        f"種類の内訳（直近{min(len(hist), 20)}件）: {tally_line}\n\n"
        f"# 今の記憶に多い話題（会話の記憶のタグと件数）\n{bias_line}\n\n"
        f"# イブの好み（イブが自分で言ったこと）\n{likes or '（まだ無い）'}\n\n"
        f"# 今回\n{count}件を見に行く。候補を{candidates}行出す。"
        + ("今回は最初のまとめての取り込みなので、いろいろな分野から幅広く選ぶ。" if bootstrap else "")
    )
    system = CURATE_SYSTEM.format(misc=MISC_LINE if bootstrap else "", n=candidates)
    system += persona_for(persona, PERSONA_HOW)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
