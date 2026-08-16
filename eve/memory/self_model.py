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
2. 1項目の値は最大 `MAX_PER_SLOT`(=2) 件。3件目で最も古い値が落ちる
   （「好きな食べ物が5つ」を構造的に作れない＝ユーザ指示）。
3. 2件並んだ状態が、そのまま「どちらも好きで迷う」という正直な表現になる
   （＝矛盾を黙って上書きしない、の実装）。
4. 対象名・値の長さにも上限。

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
from typing import Optional

from ..clock import Stamp

logger = logging.getLogger(__name__)

# 極性（**閉じた2値**）。表示は「{極性}な{対象}」＝「好きな食べ物」「苦手な食べ物」。
POLARITIES: tuple[str, ...] = ("好き", "苦手")
_POLARITY_SET = frozenset(POLARITIES)

MAX_TOPICS = 12       # 台帳に載る項目数の上限（超えたら LRU で1つ落とす）
MAX_PER_SLOT = 2      # 1項目の値の上限（ユーザ指示「各項目1〜2ぐらい」）
MAX_TOPIC_CHARS = 12  # 対象名の長さ上限
MAX_VALUE_CHARS = 24  # 値の長さ上限（読み上げに載る長さ）

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
    """1項目内の1つの値。`iso` は**その発話の時刻**（古い順に追い出すため）。"""

    value: str
    iso: str


@dataclass
class Slot:
    """(極性, 対象) 1つ分。`last_iso` は LRU 追い出しの基準（最後に言及された時刻）。"""

    polarity: str
    topic: str
    facts: list[SelfFact] = field(default_factory=list)
    last_iso: str = ""

    @property
    def label(self) -> str:
        return f"{self.polarity}な{self.topic}"


class SelfModelStore:
    """(極性, 対象) → 値（最大2件）。loop 単一所有・ロックなし（他サイドカーと同じ前提）。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._slots: dict[tuple[str, str], Slot] = {}
        if path:
            self._load()

    # --- 読み取り ---------------------------------------------------------

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

    def render(self) -> str:
        """system に入れる1ブロック。**何も無ければ空文字**（空の見出しを出さない）。

        文言の制約（実測に基づく・緩めると壊れる）:
        - 「今の会話に絡める」とは**書かない**。`# 参照`(RAG) にはその指示があり、それが
          52日前の「チーズケーキが美味しそう」を“イブの好物”に格上げした引力の正体だった
          （実機事故 2026-08-15 02:41）。ここは**聞かれた時に答えるための台帳**に徹する。
        - 「ここに無いことは決まっていない」を明示する。書いていない項目まで自信満々に
          即興されると、この層を作った意味が無くなる。
        """
        slots = [s for s in self._ordered() if s.facts]
        if not slots:
            return ""
        lines = [f"- {s.label}: {' / '.join(f.value for f in s.facts)}" for s in slots]
        return (
            "# イブ自身のこと（自分で言ったこと。理由なく変えない）\n"
            + "\n".join(lines)
            + "\n（2つ並んでいるのは「どちらも好きで迷う」という意味。"
              "ここに無いことはまだ決まっていない＝聞かれたら決めつけず、"
              "その場で思ったことを言ってよい。ここにある事は聞かれたら同じように答える。"
              "そのまま読み上げない。）"
        )

    # --- 書き込み ---------------------------------------------------------

    def set(self, polarity: str, topic: str, value: str, iso: Optional[str] = None) -> bool:
        """値を追加/更新する。受け付けたら True。

        既にある値なら時刻だけ更新（言い直した＝現役）。`MAX_PER_SLOT` 超過で最も古い値を、
        `MAX_TOPICS` 超過で最も長く言及されていない項目を落とす。
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
        for f in slot.facts:
            if f.value == value:
                f.iso = iso
                self._append({"op": "set", "polarity": polarity, "topic": topic,
                              "value": value, "iso": iso})
                return True
        slot.facts.append(SelfFact(value, iso))
        slot.facts.sort(key=lambda f: f.iso)
        dropped = ""
        while len(slot.facts) > MAX_PER_SLOT:
            dropped = slot.facts.pop(0).value  # 最も古い＝いちばん前
        self._append({"op": "set", "polarity": polarity, "topic": topic,
                      "value": value, "iso": iso})
        if dropped:
            logger.info("自己モデル: %s の上限で「%s」を押し出した（→ %s）",
                        slot.label, dropped, " / ".join(f.value for f in slot.facts))
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
        slot.facts = [f for f in slot.facts if f.value != value]
        slot.facts.append(SelfFact(value, iso))
        slot.facts.sort(key=lambda f: f.iso)
        del slot.facts[:-MAX_PER_SLOT]

    def _load_remove(self, pol: str, top: str, rec: dict) -> None:
        slot = self._slots.get((pol, top))
        if slot is None:
            return
        slot.facts = [f for f in slot.facts if f.value != rec.get("value")]
        if not slot.facts:
            self._slots.pop((pol, top), None)
