"""話し方の実例（few-shot）。性格の文（`persona.md`）だけでは出ない「ノリ」を、実際のやりとりで見せる。

2026-10-01 ユーザ「N5＋実例で」「Neuro-sama の実際の会話データをプロンプトに混ぜてもいいけど、その場面になったら
ほぼその実際の会話データの丸写しにならないかよく確認」「入れる会話データの吟味は人間の感性がよさそう」。

## 決めたこと（実測 `tools/prompt_eval/`・gpt-6-luna・性格 N6）
- **個数は 8**: 0/2/4/8/16 個で比べ、Neuro-sama の実際の返事との差（小さいほど近い）が 0.162/0.167/0.159/0.134/0.133
  （8 で下がり 16 で横ばい）。人物設定の対話の研究（arXiv 2402.09954）でも 7〜9 個で頭打ち。
- **毎ターン、ランダムに選び直す**。似た場面の実例を選ぶより、ランダムの方が良い（同研究・似た場面の実例が最悪だった）。
- **今の話しかけと意味が近い実例は見せない**（丸写しの元）。その実例を見せると、同じ話しかけで 33/100・
  言い換えで 16/100 が実例の返事の丸写しになった（見せなければ 0/50）。意味の近さは記憶の検索と同じ Ruri で測り、
  しきい値 0.88 で言い換え 50/50 を捕まえる（普通の話しかけで外れる実例は1回あたり平均 1.2個）。
  埋め込みが使えない時は文字の2-gram で代用する（言い換えは 50件中 19件しか捕まえられない＝次善）。
- 実例は**人が選ぶ**（`persona_examples.yaml`）。ファイルが無い・空なら実例なし（性格の文だけで動く・壊れない）。
- 返事が実例をほぼそのまま言っていたら**ログに残す**（止めはしない＝もう声に出ている）。
"""
from __future__ import annotations

import difflib
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HEADING = "## 話し方の参考（こういうノリで話す。言い回しや話の中身はまねしない）"
# 実例を除外する意味の近さ（Ruri の余弦）。言い換えは 0.883〜0.994（中央 0.974）、普通の話しかけと候補の最大は中央 0.828
SIM_EXCLUDE = 0.88
# 埋め込みが無い時の代用（文字2-gram の Jaccard）。同じ話しかけは 1.0・無関係な候補どうしで 0.3 を超えたのは 1225組中2組
BIGRAM_EXCLUDE = 0.3
# 丸写しの疑い: 実例の返事の半分以上、または12文字以上がそのまま出た
COPY_MIN_CHARS = 12
# 1ターンに入れる実例の合計の上限（文字）。ユーザ作の実例は複数ターンで長いものがある（最長 約1000字）
MAX_BLOCK_CHARS = 2000
COPY_MIN_COVER = 0.5

_NORM = re.compile(r"[\s、。，．,.！？!?…・ー〜~「」『』（）()]")


def _default_path() -> Path:
    """既定は**リポジトリ直下の persona_examples.yaml**（`.env` の `PERSONA_EXAMPLES_FILE` で差し替え可）。"""
    return Path(__file__).resolve().parent.parent / "persona_examples.yaml"


def examples_path() -> str:
    return str(os.getenv("PERSONA_EXAMPLES_FILE") or _default_path())


def _turns_of(d: dict) -> list[tuple[str, str]]:
    """1件の実例を [(話者, 文)] に。`turns: [{user: …}, {eve: …}, …]`（複数ターン）か `user`/`eve`（1往復）。"""
    out: list[tuple[str, str]] = []
    if isinstance(d.get("turns"), list):
        for t in d["turns"]:
            if isinstance(t, dict) and len(t) == 1:
                (who, text), = t.items()
                text = str(text or "").strip()
                if who in ("user", "eve") and text:
                    out.append((who, text))
    elif str(d.get("user") or "").strip() and str(d.get("eve") or "").strip():
        out = [("user", str(d["user"]).strip()), ("eve", str(d["eve"]).strip())]
    # ユーザの話しかけとイブの返事が両方ある時だけ実例にする
    return out if any(w == "user" for w, _ in out) and any(w == "eve" for w, _ in out) else []


def load_examples(path: Optional[str] = None) -> list[dict]:
    """`[{id, turns: [(話者, 文)]}]` を返す。無い・壊れている・空なら []（実例なしで動く）。"""
    p = Path(path) if path else Path(examples_path())
    try:
        if not p.exists():
            return []
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    except Exception:
        logger.exception("話し方の実例の読み込みに失敗（実例なしで継続）: %s", p)
        return []
    items = data.get("examples") if isinstance(data, dict) else data
    out = []
    for d in items or []:
        turns = _turns_of(d) if isinstance(d, dict) else []
        if turns:
            out.append({"id": str(d.get("id") or ""), "turns": turns})
    return out


def _users(e: dict) -> list[str]:
    return [t for w, t in e["turns"] if w == "user"]


def _eves(e: dict) -> list[str]:
    return [t for w, t in e["turns"] if w == "eve"]


def _render(e: dict) -> str:
    return "\n".join(f"{'ユーザ' if w == 'user' else 'イブ'}: {t}" for w, t in e["turns"])


def _norm(s: str) -> str:
    return _NORM.sub("", s or "").replace("イブ", "")


def similar_line(a: str, b: str) -> float:
    """話しかけどうしの似かた（文字2-gram の Jaccard・0〜1）。埋め込みが無い時の代用。"""
    x, y = _norm(a), _norm(b)
    gx = {x[i:i + 2] for i in range(len(x) - 1)} or {x}
    gy = {y[i:i + 2] for i in range(len(y) - 1)} or {y}
    union = gx | gy
    return len(gx & gy) / len(union) if union else 0.0


def _cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def _lcs(a: str, b: str) -> str:
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return a[m.a:m.a + m.size]


def _lcs_len(a: str, b: str) -> int:
    return len(_lcs(a, b))


# ひらがなだけの重なり（「じゃないですか」等の語尾）は丸写しに数えない（2026-10-04 の試しで誤検知 6/6 がこれ）
_KANA_ONLY = re.compile(r"^[ぁ-ゖー]+$")


class StyleExamples:
    """実例の置き場。`pick()` で毎ターン k 組を選んでブロックにする（loop 上で使う・状態は持たない）。"""

    def __init__(self, examples: list[dict], *, k: int = 8, embedder=None,
                 rng: Optional[random.Random] = None, max_chars: int = MAX_BLOCK_CHARS) -> None:
        self.examples = list(examples)
        self.k = max(0, int(k))
        self.max_chars = max_chars
        self._embedder = embedder
        self._vecs: Optional[list] = None  # 実例ごとの、ユーザの話しかけの埋め込みのリスト（prepare で作る）
        self._rng = rng or random.Random()

    @property
    def active(self) -> bool:
        return bool(self.examples) and self.k > 0

    async def prepare(self) -> None:
        """起動時（埋め込みモデルのウォームアップの後）に、実例の話しかけを埋め込んでおく。失敗したら2-gram で代用。"""
        if not self.active or self._embedder is None:
            return
        try:
            self._vecs = [[await self._embedder.embed_query(u) for u in _users(e)] for e in self.examples]
        except Exception:
            logger.exception("話し方の実例の埋め込みに失敗（文字の似かたで代用）")
            self._vecs = None

    async def _keep(self, said: str) -> list[dict]:
        """今の話しかけと同じ場面の実例を外した残り（複数ターンの実例は、どれか1つの話しかけが近ければ外す）。"""
        if not said:
            return self.examples
        if self._vecs is not None and self._embedder is not None:
            try:
                # 記憶の検索が同じ文を直前に埋め込んでいる（埋め込み側の直近キャッシュで二度計算しない）
                q = await self._embedder.embed_query(said)
                return [e for e, vs in zip(self.examples, self._vecs)
                        if max((_cos(q, v) for v in vs), default=0.0) < SIM_EXCLUDE]
            except Exception:
                logger.exception("話しかけの埋め込みに失敗（文字の似かたで代用）")
        return [e for e in self.examples
                if max((similar_line(u, said) for u in _users(e)), default=0.0) < BIGRAM_EXCLUDE]

    async def pick(self, said: str) -> tuple[str, list[str]]:
        """（ブロック, 見せた実例のイブの返事）。実例が無ければ ("", [])。
        最大 k 件を、ブロックが max_chars を超えない範囲で選ぶ（長い複数ターンの実例で system が膨らまないように）。"""
        if not self.active:
            return "", []
        keep = await self._keep(said)
        order = self._rng.sample(keep, len(keep))
        picked, size = [], 0
        for e in order:
            if len(picked) >= self.k:
                break
            n = len(_render(e))
            if picked and size + n > self.max_chars:
                continue
            picked.append(e)
            size += n
        if not picked:
            return "", []
        body = "\n".join(f"（{i}）\n{_render(e)}" for i, e in enumerate(picked, 1))
        return f"{HEADING}\n{body}", [t for e in picked for t in _eves(e)]

    @staticmethod
    def copy_suspect(reply: str, shown: list[str]) -> Optional[str]:
        """返事が見せた実例の返事をほぼそのまま言っていたら、その実例の返事を返す（ログ用）。"""
        r = _NORM.sub("", reply or "")
        for e in shown or []:
            x = _NORM.sub("", e)
            common = _lcs(r, x)
            n = len(common)
            if not x or _KANA_ONLY.match(common or "あ"):
                continue
            if n >= COPY_MIN_CHARS or (n / len(x) >= COPY_MIN_COVER and n >= 6):
                return e
        return None
