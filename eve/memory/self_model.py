"""イブ自身の好み・趣向のメモ（自己モデル）。

## なぜ要るか（実機事故 2026-08-15）
イブの「好きな食べ物」がチーズケーキ→ラーメン→りんご→おかか/ツナマヨ/生姜焼きと毎回
変わった。原因は**嗜好を保持する層がどこにも無く**、毎ターン RAG の top-3 に何が入ったかで
即興していたこと（02:41 と 02:45 で top-3 が総入れ替えになっていた）。
さらに、ユーザの好み（`conversation_history.jsonl:32`「俺はおかかが好き」）が2か月かけて
イブの自己申告（`:515`）に転写される**ロンダリング**が起きていた。

## 役割分担（ユーザ裁定 2026-08-15）
- **性格は人が書く**（ペルソナプロンプト）。遺伝的なもので、経験で変わる前提を置かない。
- **好み・趣向は AI が書く**。経験要因が大きいので会話から育つのが自然。
  書き込み・削除は自己形成サイドカー（`eve/self_model/worker.py`）が行う。

## キーの形 = (極性, 対象)
- **極性は閉じた2値**（`好き` / `苦手`）。ここを開くと「好きな食べ物」と「苦手な食べ物」が
  混ざる。埋め込みでは分離できないことを実測済み（2026-08-15・実 Ruri）:
      好きな食べ物 / 苦手な食べ物 = 0.931  ← 絶対に統合してはいけない
      好きな食べ物 / 好きな食材   = 0.928  ← 統合したい
  **潰したい組より残したい組の方が似ている**ので、閾値による自動統合は原理的に無理。
  よって極性は意味で測らず、構造で分ける。
- **対象は自由**（食べ物・ゲーム・ポケモン・色…）。ユーザ指摘 2026-08-15: 好きなジャンルは
  実質無限にあり、閉じた一覧にすると不自然。

## 無限に増えないための不変条件（**コードで強制**・プロンプトに委ねない）
1. `MAX_TOPICS` を超えたら、**最も長く言及されていない項目を落とす**（LRU）。
   → 対象名が自由でも、注入ブロックの長さは有界。
2. 1項目の値は最大 `MAX_PER_SLOT`(=3) 件で、**好きな順（苦手な物は苦手な順）の順位つき**。新しい値が入って4位に
   なった値は落ちる（「好きな食べ物が5つ」を構造的に作れない）。
   2026-10-17 ユーザ「好きなものや嫌いなものは1つに絞らず3位までランキングつけることでばらけそう」で、
   2026-08-15 の「各項目1〜2ぐらい」（2件並び＝どちらも好きで迷う）から変えた。順位は自己形成係（LLM）がつける。
3. 対象名・値の長さにも上限。

**言い換えによる突破（好きな食べ物 / 好きな食材 / 好物）は完全には防げない**（上記の実測の
とおり自動判別できない）。プロンプトで「既存項目に入るなら新設するな」と第一線を張るが、
保証は 1 の総数上限が担う＝突破されても枠を1つ無駄にするだけで、context は膨らまない。

## 永続化
追記のみの JSONL（`set` / `remove` イベント）を起動時に再生する。会話履歴・RAG と同じ流儀で、
スナップショットを書き換えないので部分書き込みで全損しない。
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..clock import Stamp

logger = logging.getLogger(__name__)

# 相対パスはリポジトリ基準（会話・記憶・タスクと同じ。起動した場所に依らない）
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 極性（**閉じた2値**）。表示は「{極性}な{対象}」＝「好きな食べ物」「苦手な食べ物」。
POLARITIES: tuple[str, ...] = ("好き", "苦手")
_POLARITY_SET = frozenset(POLARITIES)

MAX_TOPICS = 12       # 台帳に載る項目数の上限（超えたら LRU で1つ落とす）
MAX_PER_SLOT = 3      # 1項目の値の上限＝順位の数（ユーザ 2026-10-17「3位までランキング」・以前は「各項目1〜2ぐらい」）
MAX_TOPIC_CHARS = 12  # 対象名の長さ上限
MAX_VALUE_CHARS = 24  # 値の長さ上限（読み上げに載る長さ）

# 返事に入れるのは、今の話しかけに近い項目だけ（ユーザ 2026-10-17「好みの種類が近いときに読み込ませる。好きな食べ物は？で
# 好きな食べ物はプリンを入れる。好きな映画とかは同時に読み込ませない。RAG に似た感じ。0件でもいい」）。毎ターン全部入れて
# いた時は、項目が少ないうちはその1つ（好きな食べ物: プリン）を聞かれていない所でも持ち出していた。
# 近さは記憶の検索と同じ埋め込み（Ruri）で「好きな食べ物: プリン」と話しかけを両方クエリとして測る（実測 2026-10-17・
# 仮の項目を足した12項目×46の話しかけ: 入れるべき項目は全部いちばん近く 0.859〜0.943、要らない話しかけの最大は 0.823）。
# 配信のコメントは印（（チャット）名前: ）を外して測る（付けたままだと 0.02〜0.03 下がり「好きな映画ある？」を落とした）
RECALL = True          # False なら従来どおり毎ターン全部入れる（測る口）
RECALL_MIN = 0.85      # これ以上近い項目だけ入れる
RECALL_MARGIN = 0.02   # いちばん近い項目からこれ以上離れた項目は入れない（食べ物を聞かれて飲み物・苦手な虫まで入れない）
RECALL_TOP = 3
# 置いている訳（聞かれた時に答えがぶれないため）を書くか（測る口）。毎回全部入れていた時に足した物で、近い時だけ入れる形では
# 違いが出なかった（好きな食べ物・SNS・おやつを聞かれて答えがそろうのは訳の有無によらず 5/5・甘いものを聞かれて
# プリンの話 5/5・聞かれていない所のプリン 4/25 と 3/25＝`tools/prompt_eval/stream_variety.py` の gate / gate+note）ので、
# 10-16 までの注記に戻した（ユーザ 2026-10-17「プロフィール注釈必要かは検証が必要そう」）
PURPOSE_NOTE = False
# 3位までの並びの見せ方（2026-10-17 ユーザ「好みは最大3点入ってるけど3点すべて使わなくてもいい、これをいい感じに伝えられない？」）。
# 「1位 / 2位」と書くと、好きな物を聞かれるたびに「第1位は…第2位は…」と読み上げた。測った結果（好きな食べ物・動物を
# 通常と配信で各2回・本番の組み立て）: 順位を読み上げた 今 8/8・「1位/2位」のまま「全部挙げなくてよい」を添える 6/8・
# 「位」を書かずに好きな順に並べて同じ一言を添える 0/8（いちばん好きな物は 8/8 で言い、2つ目は言ったり言わなかったり）。測る口
RANK_LABELS = False    # 並びに「1位」「2位」を付けるか
ALL_OPTIONAL = True    # 「聞かれても全部を挙げなくてよい」を添えるか
# 配信のコメントの印（`stream.manager.format_comment` の「（チャット）名前: 」「（スーパーチャット ￥500・緑）名前: 」）
_COMMENT_HEAD = re.compile(r"^（[^）\n]{1,40}）[^\n]{0,80}?: ")


def comment_body(text: str) -> str:
    """配信のコメントなら印と名前を外した本文（それ以外はそのまま）。近さを測る時だけに使う。純関数。"""
    return _COMMENT_HEAD.sub("", text or "", count=1)


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

# 対象名の正規化で剥がす接頭辞。「好きな食べ物」と言われても対象は「食べ物」。
# 完全な言い換え吸収は狙わない（できないと実測済み）。**表記ゆれだけ**を潰す。
_PREFIX = re.compile(r"^(?:大?好きな?|嫌いな?|苦手な?|得意な?)")
_SUFFIX = re.compile(r"(?:のこと|について)$")


def normalize_topic(topic: str) -> str:
    """対象名の表記ゆれを潰す（全角半角・空白・「好きな」等の接頭辞）。純関数。"""
    t = unicodedata.normalize("NFKC", (topic or "").strip())
    t = re.sub(r"\s+", "", t)
    t = _PREFIX.sub("", t)
    t = _SUFFIX.sub("", t)
    return t.strip()


@dataclass
class SelfFact:
    """1項目内の1つの値。`iso` は**最後に言った時刻**（記録用）。順位は並び順（`Slot.facts` の前ほど上）。"""

    value: str
    iso: str


@dataclass
class Slot:
    """(極性, 対象) 1つ分。facts は順位の順（0番目が1位）。`last_iso` は LRU 追い出しの基準（最後に言及された時刻）。"""

    polarity: str
    topic: str
    facts: list[SelfFact] = field(default_factory=list)
    last_iso: str = ""

    @property
    def label(self) -> str:
        return f"{self.polarity}な{self.topic}"


def place(slot: Slot, value: str, iso: str, rank: Optional[int]) -> str:
    """値を順位つきで置く（slot を書き換える）。押し出された値（4位になった値）を返す・無ければ ""。

    rank（1〜MAX_PER_SLOT）があればその位置へ入れる（既にある値なら動かす・下の値は1つずつ下がる）。
    rank が無ければ、既にある値はそのまま（時刻だけ）・新しい値はいちばん下（満杯ならいちばん下と入れ替わる）。
    順位をつけるのは自己形成係（LLM）の役目で、無い時はこの逃げ方にする。
    """
    existing = next((f for f in slot.facts if f.value == value), None)
    if existing is not None:
        existing.iso = iso
        if rank is None:
            return ""
        slot.facts.remove(existing)
        fact = existing
    else:
        fact = SelfFact(value, iso)
        if rank is None:
            rank = min(len(slot.facts) + 1, MAX_PER_SLOT)
    slot.facts.insert(max(0, min(int(rank), MAX_PER_SLOT) - 1), fact)
    dropped = ""
    while len(slot.facts) > MAX_PER_SLOT:
        dropped = slot.facts.pop().value  # いちばん下（4位）
    return dropped


class SelfModelStore:
    """(極性, 対象) → 順位つきの値（最大3件）。loop 単一所有・ロックなし（他サイドカーと同じ前提）。"""

    def __init__(self, path: Optional[str] = None) -> None:
        # 相対パスはリポジトリ基準で解決する（以前は起動した場所＝cwd 基準で、別のフォルダから起動すると
        # 好みの台帳だけ別の場所に新しく作られ、前回の好みを忘れたように見えた）
        if path and not Path(path).is_absolute():
            path = str(_REPO_ROOT / path)
        self._path = path
        self._slots: dict[tuple[str, str], Slot] = {}
        if path:
            self._load()

    # --- 読み取り ---------------------------------------------------------

    @property
    def path(self) -> Optional[str]:
        """保存先（リポジトリ基準で解決済み・保存しない時は None）。"""
        return self._path

    def __len__(self) -> int:
        return sum(len(s.facts) for s in self._slots.values())

    @property
    def topics(self) -> int:
        return len(self._slots)

    def values(self, polarity: str, topic: str) -> list[str]:
        s = self._slots.get((polarity, normalize_topic(topic)))
        return [f.value for f in s.facts] if s else []

    def as_dict(self) -> dict[str, list[str]]:
        return {s.label: [f.value for f in s.facts] for s in self._ordered()}

    def _ordered(self) -> list[Slot]:
        """表示順は「最近言及した順」ではなく安定順（極性→対象）＝毎ターン揺れないため。"""
        return sorted(self._slots.values(), key=lambda s: (POLARITIES.index(s.polarity), s.topic))

    def entries(self) -> list[tuple[tuple[str, str], str]]:
        """（キー, 1行）の一覧。1行は「好きな食べ物: A / B」の形（項目ごとに1行・好きな順）。"""
        return [((s.polarity, s.topic), f"{s.label}: {' / '.join(f.value for f in s.facts)}")
                for s in self._ordered() if s.facts]

    def value_lines(self) -> list[tuple[tuple[str, str], str]]:
        """（キー, 1行）の一覧を**値ごと**に。1行は「好きな食べ物: A」の形（話しかけとの近さを測る文・`SelfRecall`）。

        項目ごとの1行（`entries`）で測ると、3位まで並べた時に行が長くなって近さが下がり、聞かれているのに入らなかった
        （2026-10-17 実測: 配信のコメント「イブちゃんの好きな食べ物は？」と、値が1つの行 0.865／値を2つ並べた行 0.844
        ＝しきい値 0.85 を割った）。しきい値は1つだけの行で測った物なので、その形で測る。
        """
        return [((s.polarity, s.topic), f"{s.label}: {f.value}") for s in self._ordered() for f in s.facts]

    def render(self, only=None) -> str:
        """system に入れる1ブロック。**何も無ければ空文字**（空の見出しを出さない）。only はキーの集まり（その項目だけ）。

        文言の制約（実測に基づく・緩めると壊れる）:
        - 「今の会話に絡める」とは**書かない**。`# 参照`(RAG) にはその指示があり、それが
          52日前の「チーズケーキが美味しそう」を“イブの好物”に格上げした引力の正体だった
          （実機事故 2026-08-15 02:41）。ここは**聞かれた時に答えるための台帳**に徹する。
        - 「ここに無いことは決まっていない」を明示する。書いていない項目まで自信満々に
          即興されると、この層を作った意味が無くなる。
        - 毎ターン全部入れていた時は、項目が少ないうちはその1つを聞かれていない所でも持ち出していた（2026-10-16 の
          配信リハーサルで「好きな食べ物: プリン」がスパチャの使い道・貢ぎ物の話に毎回出た・21回中9回）。今は話しかけに
          近い項目だけを入れる（`SelfRecall`）。置いている訳を書く案（`PURPOSE_NOTE`）は、近い時だけ入れる形では違いが無かった
        """
        slots = [s for s in self._ordered() if s.facts and (only is None or (s.polarity, s.topic) in only)]
        if not slots:
            return ""

        def ranked(s: Slot) -> str:
            if len(s.facts) == 1 or not RANK_LABELS:
                return " / ".join(f.value for f in s.facts)
            return " / ".join(f"{i}位 {f.value}" for i, f in enumerate(s.facts, 1))

        lines = [f"- {s.label}: {ranked(s)}" for s in slots]
        purpose = "聞かれた時に答えがぶれないように控えている。" if PURPOSE_NOTE else ""
        asked = "" if PURPOSE_NOTE else "ここにある事は聞かれたら同じように答える。"
        # 2026-10-17: 「2つ並んでいるのは『どちらも好きで迷う』という意味」から順位の説明に変えた（3位までのランキング）。
        # どれを言うかは書かない（イブが決める）
        order = ("並んでいる時は、好きな順（苦手な物は苦手な順）の順位。" if RANK_LABELS
                 else "並んでいる時は好きな順（苦手な物は苦手な順）。")
        optional = "聞かれても全部を挙げなくてよい。" if ALL_OPTIONAL else ""
        return (
            "# イブ自身のこと（自分で言ったこと。理由なく変えない）\n"
            + "\n".join(lines)
            + f"\n（{purpose}{order}{optional}"
              "ここに無いことはまだ決まっていない＝聞かれたら決めつけず、"
              f"その場で思ったことを言ってよい。{asked}そのまま読み上げない。）"
        )

    # --- 書き込み ---------------------------------------------------------

    def set(self, polarity: str, topic: str, value: str, iso: Optional[str] = None,
            rank: Optional[int] = None) -> bool:
        """値を追加/更新する。受け付けたら True。

        rank は好きな（苦手な）順の順位（1〜MAX_PER_SLOT・`place`）。4位になった値と、`MAX_TOPICS` 超過で
        最も長く言及されていない項目を落とす。
        """
        polarity = (polarity or "").strip()
        topic = normalize_topic(topic)
        value = (value or "").strip()
        if polarity not in _POLARITY_SET or not topic or not value:
            return False
        if len(topic) > MAX_TOPIC_CHARS or len(value) > MAX_VALUE_CHARS:
            logger.info("自己モデル: 長すぎるので却下 %s/%.20s = %.30s", polarity, topic, value)
            return False
        iso = iso or Stamp.now().iso
        key = (polarity, topic)
        slot = self._slots.get(key)
        if slot is None:
            slot = Slot(polarity, topic)
            self._slots[key] = slot
            self._evict_topics_if_needed(protect=key)
            if key not in self._slots:      # 自分自身が落ちた（理論上は起きない）
                return False
        slot.last_iso = max(slot.last_iso, iso)
        if rank is not None:
            try:
                rank = max(1, min(MAX_PER_SLOT, int(rank)))
            except (TypeError, ValueError):
                rank = None
        dropped = place(slot, value, iso, rank)
        rec = {"op": "set", "polarity": polarity, "topic": topic, "value": value, "iso": iso}
        if rank is not None:
            rec["rank"] = rank
        self._append(rec)
        if dropped:
            logger.info("自己モデル: %s は%d位までなので「%s」が外れた（→ %s）",
                        slot.label, MAX_PER_SLOT, dropped, " / ".join(f.value for f in slot.facts))
        return True

    def _evict_topics_if_needed(self, *, protect: tuple[str, str]) -> None:
        """項目数の上限を守る（最も長く言及されていないものから落とす）。

        言い換え（好きな食べ物 / 好きな食材 / 好物）は**自動判別できない**ことを実測済みなので、
        「言い換えられない」ではなく「言い換えられても有界」で守る。ここがその要。
        """
        while len(self._slots) > MAX_TOPICS:
            victim = min(
                (k for k in self._slots if k != protect),
                key=lambda k: (self._slots[k].last_iso, self._slots[k].topic),
                default=None,
            )
            if victim is None:
                return
            s = self._slots.pop(victim)
            self._append({"op": "remove_slot", "polarity": s.polarity, "topic": s.topic})
            logger.info("自己モデル: 項目上限(%d)のため「%s」を落とした", MAX_TOPICS, s.label)

    def remove(self, polarity: str, topic: str, value: str) -> bool:
        """値を消す（本人が否定した / ユーザに違うと言われた時）。空になった項目も消える。"""
        key = ((polarity or "").strip(), normalize_topic(topic))
        slot = self._slots.get(key)
        if slot is None:
            return False
        before = len(slot.facts)
        slot.facts = [f for f in slot.facts if f.value != value]
        if len(slot.facts) == before:
            return False
        self._append({"op": "remove", "polarity": key[0], "topic": key[1], "value": value})
        if not slot.facts:
            self._slots.pop(key, None)  # 空の項目は残さない（注入ブロックを汚さない）
        return True

    # --- 永続化（追記のみ・再生で復元）------------------------------------

    def _append(self, rec: dict) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("自己モデルの保存に失敗（メモリ上は保持して継続）")

    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        applied = 0
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 壊れた1行で全損させない
                    op = rec.get("op")
                    pol, top = rec.get("polarity"), normalize_topic(rec.get("topic") or "")
                    if pol not in _POLARITY_SET or not top:
                        continue
                    if op == "remove_slot":
                        self._slots.pop((pol, top), None)
                    elif op == "remove":
                        self._load_remove(pol, top, rec)
                    else:
                        self._load_set(pol, top, rec)
                    applied += 1
        except OSError:
            logger.exception("自己モデルの読み込みに失敗（空で継続）")
            return
        # 再生の途中では上限を超えうる（後で落とされた項目も再生されるため）→ 最後に整える。
        self._evict_topics_if_needed(protect=("", ""))
        if applied:
            logger.info("自己モデルを復元: %d 項目 / %d 件", self.topics, len(self))

    def _load_set(self, pol: str, top: str, rec: dict) -> None:
        value = (rec.get("value") or "").strip()
        if not value:
            return
        slot = self._slots.setdefault((pol, top), Slot(pol, top))
        iso = rec.get("iso") or ""
        slot.last_iso = max(slot.last_iso, iso)
        rank = rec.get("rank")
        # 順位の無い昔の行（2026-10-17 より前）は、書かれた順にいちばん下へ足す（＝今の `set` の順位なしと同じ）
        place(slot, value, iso, int(rank) if isinstance(rank, int) else None)

    def _load_remove(self, pol: str, top: str, rec: dict) -> None:
        slot = self._slots.get((pol, top))
        if slot is None:
            return
        slot.facts = [f for f in slot.facts if f.value != rec.get("value")]
        if not slot.facts:
            self._slots.pop((pol, top), None)


class SelfRecall:
    """控えのうち、今の話しかけに近い項目だけを返す（記憶の検索と同じ埋め込み・0件もある）。

    応答の system に入れる `# イブ自身のこと` を作る（`ResponseOrchestrator` が刺激ごとに呼ぶ）。タスク担当には従来どおり
    全部を渡す（goal の「イブの好きな〜」を解決するため・`task/agent.py`）。埋め込みが使えない時は全部を返す
    （聞かれた時の答えがぶれない方に倒す）。
    """

    def __init__(self, store: SelfModelStore, embedder) -> None:
        self.store = store
        self._embedder = embedder
        self._vecs: dict[str, list[float]] = {}  # 項目の1行 → 埋め込み（値が変わった項目だけ作り直す）

    async def block_for(self, text: str) -> str:
        if not RECALL or self._embedder is None:
            return self.store.render()
        lines = self.store.value_lines()  # 値ごとの行（項目の近さ＝その項目の値の行のうちいちばん近いもの）
        query = comment_body(text).strip()
        if not lines or not query:
            return ""
        try:
            q = await self._embedder.embed_query(query)
            for _key, line in lines:
                if line not in self._vecs:
                    self._vecs[line] = await self._embedder.embed_query(line)
        except Exception:
            logger.exception("自分のことの控えの近さを測れなかった（全部入れる）")
            return self.store.render()
        by_key: dict = {}
        for key, line in lines:
            by_key[key] = max(by_key.get(key, -1.0), _cos(q, self._vecs[line]))
        scored = sorted(((s, key) for key, s in by_key.items()), key=lambda x: -x[0])
        best = scored[0][0]
        keep = {key for s, key in scored[:RECALL_TOP] if s >= RECALL_MIN and s >= best - RECALL_MARGIN}
        if keep:
            logger.info("🙋 自分のことの控えから入れた: %s（%.3f）",
                        "・".join(ln for k, ln in self.store.entries() if k in keep), best)
        return self.store.render(only=keep) if keep else ""
