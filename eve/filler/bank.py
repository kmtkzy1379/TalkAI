"""フィラーの表・抽選・バンク（S3・ユーザ指定 2026-09-25・`docs/FILLER_IMPL_PLAN.md` §1.4 / §6）。

## 抽選（毎回同じ表・1段目も2段目以降も）
- 無言 ＋ 応答非依存10語 ＋ 特殊フィラー（語＋っと・っとね をまとめて合成した組・例 えーっと）を weight の比率で。
  特殊フィラーは合計 5%（系ごとに按分・系の中は等分）で、その分だけ無言と語を同じ比率で縮める（ユーザ 2026-09-29・
  fillers.yaml の special）＝無言 19%・語 76%・特殊 5%
- 2段目以降（無言以外のうち）: 伸ばし棒 1%（直前の語が伸ばせる語で、無言を挟んでいない時だけ・採用が決まっている時だけ）/
  残り＝別のフィラー（直前と同じものは除く・比率は weight のまま）。伸ばし棒が使えない時はその分を別のフィラーに回す
- 「直前の語」は無言を挟んでも保つ（あー、〔無言〕、あー にならない）
- 1段目は場面（S8）で選べるものが変わる（`draw_first`）。悩む・悲しむ には え系の特殊フィラーを足す（ユーザ 2026-09-29）

`draw()` は乱数を注入する純関数（決定論テストで分布を数万回確かめる）。
"""
from __future__ import annotations

import dataclasses
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import audio as fa

logger = logging.getLogger(__name__)


SILENCE = "silence"
MOOD_DEFAULT = "通常"  # S8: 場面の判定が無い・弱い時の種類


@dataclass(frozen=True)
class Piece:
    """抽選で出た1つ。kind = "silence" | "word"（応答非依存の語・特殊フィラーの組）| "stretch"。"""

    kind: str
    word_id: str = ""
    sec: float = 0.0  # 無言の長さ（語の長さは音声で決まる）


@dataclass
class FillerTable:
    """`fillers.yaml` のうち、鳴らし方と抽選に要る部分。"""

    weights: dict                     # 抽選に出る id → 重み（%）。応答非依存の語＋特殊フィラーの組（id 例 "e+tto"）
    texts: dict                       # id → 文字（応答非依存の語・特殊フィラーの組 例 えっと・笑いの語）
    silence_weight: float = 20.0
    silence_sec: tuple = (0.3, 0.5)
    stretch_prob: float = 0.01
    stretchable: dict = field(default_factory=dict)  # 語 id → 継ぐ母音
    # 特殊フィラーの組 id → 系（例 "e+tto" → "え"）。語＋っと・っとね をまとめて合成した音で、独立したフィラーとして
    # 抽選する（ユーザ 2026-09-29「確率0.2は1段目にくっつく確率であり、今回の方針で独立することが決まったため5％にする」）
    special: dict = field(default_factory=dict)
    max_chain_sec: float = 4.0
    min_utterance_sec: float = 0.6
    join_gap_sec: Optional[float] = None   # 未決なら None（フィラーは鳴らさない）
    stretch_adopted: Optional[bool] = None  # 未決なら None（伸ばし棒は使わない）
    stretch_sources: dict = field(default_factory=dict)
    stretch_add_sec: float = 0.4
    repeat_factor: float = 1.0  # 同じチェーンで使った語の重みに使った回数だけ掛ける（1.0=補正なし）
    # S6 笑い（ユーザ裁定 2026-09-26）。笑いの語＝表で label に「笑い」を持つ語（あはは・ふふ …）
    laugh_ids: tuple = ()
    laugh_immediate: float = 0.9
    laugh_filler: float = 0.75
    laugh_bank_suffix: str = "_laugh"
    # 確率 ≥ laugh_filler の時の笑い方。laugh = フィラーの中で普通に1回笑う（ユーザ 2026-09-26「ただ普通に
    # 笑うほうがよさそう」）/ tone = フィラーの語を笑いの声色で鳴らす（旧方式・「採用してもしなくてもいい」）
    laugh_filler_style: str = "laugh"
    # S8 1段目を場面に寄せる（ユーザ指定 2026-09-27・fillers.yaml の mood）。無ければ今までどおり
    mood_threshold: float = 0.9
    mood_with_emotion: bool = False
    mood_desc: dict = field(default_factory=dict)    # 種類 → Jev に渡す説明
    mood_first: dict = field(default_factory=dict)   # 種類 → 1段目に出してよいもの（語 id / 組 id / "silence"）
    mood_word_bank: dict = field(default_factory=dict)  # (種類, 語 id) → その音を取るバンクの名前
    # 種類 → {組 id: その種類の1段目での %}。その種類の他の選択肢は残りを按分（ユーザ 2026-09-29
    # 「悩むと悲しむ両方に入れる、え系特殊フィラー確率は悩む5％、悲しむ10％」）
    mood_special: dict = field(default_factory=dict)

    def without(self, ids: set) -> "FillerTable":
        """その id（バンクに音が無い特殊フィラーの組）を抽選から外した表。語だけでも鳴らせるようにする。"""
        ids = set(ids)
        return dataclasses.replace(
            self,
            weights={k: v for k, v in self.weights.items() if k not in ids},
            special={k: v for k, v in self.special.items() if k not in ids},
            mood_first={n: tuple(o for o in opts if o not in ids) for n, opts in self.mood_first.items()},
            mood_special={n: {k: v for k, v in d.items() if k not in ids} for n, d in self.mood_special.items()},
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "FillerTable":
        import yaml

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        ents = cfg.get("entries") or []
        words = [e for e in ents if "independent" in (e.get("role") or [])]
        terms = [e for e in ents if "terminal" in (e.get("role") or [])]
        laughs = [e for e in ents if "笑い" in (e.get("labels") or [])]
        lg = cfg.get("laugh") or {}
        tm = cfg.get("timing") or {}
        sel = cfg.get("selection") or {}
        st = cfg.get("stretch") or {}
        missing = [e["id"] for e in words if e.get("weight") is None]
        if missing:
            raise ValueError(f"fillers.yaml: weight の無い応答非依存フィラー {missing}")
        gap = tm.get("join_gap_sec")
        lo, hi = (sel.get("silence_sec") or [0.3, 0.5])[:2]
        weights = {e["id"]: float(e["weight"]) for e in words}
        silence_weight = float(sel.get("silence_weight", 20.0))
        special, sp_weights, sp_texts = _special(cfg.get("special") or {}, words, terms)
        if sp_weights:
            # 特殊フィラーの分だけ、無言と語を同じ比率で縮める（ユーザ 2026-09-29「按分するから変わらない」＝割合は変えない）
            whole = silence_weight + sum(weights.values())
            f = (whole - sum(sp_weights.values())) / whole
            silence_weight *= f
            weights = {k: v * f for k, v in weights.items()}
            weights.update(sp_weights)
        return cls(
            weights=weights,
            texts={**{e["id"]: e["text"] for e in words + laughs}, **sp_texts},
            silence_weight=silence_weight,
            silence_sec=(float(lo), float(hi)),
            stretch_prob=float(sel.get("stretch_prob", 0.01)),
            stretchable={e["id"]: e["stretch"] for e in words if e.get("stretch")},
            special=special,
            max_chain_sec=float(tm.get("max_chain_sec", 4.0)),
            min_utterance_sec=float(tm.get("min_utterance_sec", 0.6)),
            join_gap_sec=None if gap is None else float(gap),
            stretch_adopted=st.get("adopted"),
            stretch_sources=dict(st.get("sources") or {}),
            stretch_add_sec=float(st.get("add_sec") or 0.4),
            repeat_factor=float(sel.get("repeat_factor", 1.0)),
            laugh_ids=tuple(e["id"] for e in laughs),
            laugh_immediate=float(lg.get("immediate", 0.9)),
            laugh_filler=float(lg.get("filler", 0.75)),
            laugh_bank_suffix=str(lg.get("bank_suffix") or "_laugh"),
            laugh_filler_style=_filler_style(lg.get("filler_style")),
            **_mood(cfg.get("mood") or {}, words, special),
        )


def _special(sp: dict, words: list, terms: list) -> tuple[dict, dict, dict]:
    """fillers.yaml の special（特殊フィラーを独立したフィラーとして抽選・ユーザ 2026-09-29）。

    返り値 = (組 id → 系, 組 id → 重み[%], 組 id → 文字)。組 = 系の語 × 特別フィラー（っと・っとね）で、id は
    `tools/build_filler_bank.py --combos` と同じ「語 id+特別フィラー id」。系の合計は total を share で按分し、系の中は等分
    （「１は按分でいい」「４は均等でいい」）。組の文字は語の combo_text（例 う → うん）＋特別フィラー。
    """
    if not sp:
        return {}, {}, {}
    by_text = {e["text"]: e for e in words}
    fams = sp.get("families") or {}
    total = float(sp.get("total", 0.0))
    share_sum = sum(float((f or {}).get("share", 0.0)) for f in fams.values())
    fam_of: dict = {}
    weight: dict = {}
    text: dict = {}
    for fam, f in fams.items():
        members = []
        for w in (f or {}).get("words") or []:
            e = by_text.get(w)
            if e is None:
                raise ValueError(f"fillers.yaml: special の {fam} に応答非依存フィラーに無い語 {w!r}")
            if e.get("no_terminal"):
                # ユーザ指定で っと・っとね を付けない語（あの・その・お）は組にしない（黙って組を作らない）
                raise ValueError(f"fillers.yaml: special の {fam} に っと・っとね を付けない語 {w!r}（no_terminal）")
            for tmn in terms:
                cid = f"{e['id']}+{tmn['id']}"
                members.append(cid)
                text[cid] = (e.get("combo_text") or e["text"]) + tmn["text"]
        if not members or share_sum <= 0:
            continue
        each = total * float((f or {}).get("share", 0.0)) / share_sum / len(members)
        for cid in members:
            fam_of[cid] = str(fam)
            weight[cid] = each
    return fam_of, weight, text


def _mood(md: dict, words: list, special: Optional[dict] = None) -> dict:
    """fillers.yaml の mood（S8）を読む。書いた語が表に無ければ止める（黙って読み替えない）。

    「特殊フィラー」は、その種類に special（系 → %）が無ければ全部の組（重みは表のまま・通常）。special があれば
    その系の組だけを、その % で（残りの選択肢は按分・ユーザ 2026-09-29 の 悩む・悲しむ）。
    """
    if not md:
        return {}
    special = special or {}
    ids = {e["text"]: e["id"] for e in words}
    first: dict = {}
    desc: dict = {}
    mood_special: dict = {}
    for name, c in (md.get("categories") or {}).items():
        opts = []
        sp_pct = (c or {}).get("special") or {}
        for w in (c or {}).get("words") or []:
            if w == "無言":
                opts.append(SILENCE)
            elif w == "特殊フィラー":
                if not sp_pct:
                    opts.extend(special)  # 全部の組（1段目の通常は表の重みのまま）
            elif w in ids:
                opts.append(ids[w])
            else:
                raise ValueError(f"fillers.yaml: mood の {name} に応答非依存フィラーに無い語 {w!r}")
        per = {}
        for fam, pct in sp_pct.items():
            members = [cid for cid, f in special.items() if f == str(fam)]
            if not members:
                raise ValueError(f"fillers.yaml: mood の {name} の special に無い系 {fam!r}（special.families）")
            per.update({cid: float(pct) / len(members) for cid in members})
        if not opts and not per:
            raise ValueError(f"fillers.yaml: mood の {name} に1段目に出せる語が無い")
        first[name] = tuple(opts)
        if per:
            mood_special[name] = per
        desc[name] = str((c or {}).get("desc") or name)
    if MOOD_DEFAULT not in first:
        raise ValueError(f"fillers.yaml: mood に「{MOOD_DEFAULT}」が無い")
    wb: dict = {}
    for name, m in (md.get("word_bank") or {}).items():
        if name not in first:
            raise ValueError(f"fillers.yaml: mood.word_bank の {name!r} が categories に無い")
        for w, bank in (m or {}).items():
            if w not in ids:
                raise ValueError(f"fillers.yaml: mood.word_bank の {w!r} が応答非依存フィラーに無い")
            wb[(name, ids[w])] = str(bank)
    return {"mood_threshold": float(md.get("threshold", 0.9)), "mood_with_emotion": bool(md.get("with_eve_emotion")),
            "mood_desc": desc, "mood_first": first, "mood_word_bank": wb, "mood_special": mood_special}


def _filler_style(v) -> str:
    s = str(v or "laugh").strip().lower()
    if s not in ("laugh", "tone"):
        raise ValueError(f"fillers.yaml: laugh.filler_style は laugh か tone（{v!r}）")
    return s


def _load_emotion_bank(table: "FillerTable", d: Path, want: set, sr: int, emoji: Optional[str]) -> dict:
    """感情のバンク（語 id → [Clip]・特殊フィラーの組も id で同じく）。無い・絵文字が emotion.yaml と違う・
    レートが違う時は空＝普通の声。"""
    man_path = d / "manifest.json"
    if not man_path.is_file():
        logger.warning("感情のフィラーのバンクが無い（%s）— その感情でも普通の声で鳴らす"
                       "（tools/build_filler_bank.py --emoji で作る）", d.name)
        return {}
    man = json.loads(man_path.read_text(encoding="utf-8"))
    pool: dict = {}
    for c in man.get("clips") or []:
        cid = c.get("id")
        if cid not in want:
            continue
        synth = str(c.get("synth_text") or "")
        if emoji and not synth.endswith(emoji):
            logger.warning("感情のフィラーのバンク %s の絵文字が emotion.yaml（%s）と違う（%s）— 作り直すまで使わない",
                           d.name, emoji, synth)
            return {}
        x, sr_c = fa.read_wav((d / c["file"]).read_bytes())
        if sr and sr_c != sr:
            logger.warning("感情のフィラーのバンクのサンプルレートが違う（%s）— 使わない", c["file"])
            return {}
        t = fa.trim(x, sr_c)
        if len(t) == 0:
            continue
        pool.setdefault(cid, []).append(Clip(cid, table.texts.get(cid, c.get("text", "")), t, sr_c))
    return pool


def _pick_word(table: FillerTable, rng: random.Random, exclude: Optional[str],
               used: Optional[dict] = None) -> str:
    ids = [i for i in table.weights if i != exclude] or list(table.weights)
    f = table.repeat_factor
    ws = [table.weights[i] * (f ** (used or {}).get(i, 0)) for i in ids]
    return rng.choices(ids, weights=ws, k=1)[0]


def draw(table: FillerTable, rng: random.Random, *, first: bool, last_word: Optional[str],
         last_kind: Optional[str], stretch_ok: bool, used: Optional[dict] = None) -> Piece:
    """次の1つを抽選する（上の規則どおり）。

    last_word = 直前の語（特殊フィラーの組も含む・無言を挟んでも保つ）/ last_kind = 直前に鳴らしたものの種類。
    stretch_ok = 伸ばし棒が使える状態か（採用済み・素材あり）。
    used = このチェーンで既に鳴らした語 id → 回数（`repeat_factor` の補正に使う。1段目は空）。
    """
    total = table.silence_weight + sum(table.weights.values())
    if rng.random() * total < table.silence_weight:
        lo, hi = table.silence_sec
        return Piece("silence", sec=rng.uniform(lo, hi))
    if first:
        return Piece("word", _pick_word(table, rng, None))
    if rng.random() < table.stretch_prob:
        if stretch_ok and last_kind == "word" and last_word in table.stretchable:
            return Piece("stretch", last_word)
    return Piece("word", _pick_word(table, rng, last_word, used))


def draw_first(table: FillerTable, rng: random.Random, mood: Optional[str]) -> Piece:
    """S8 の1段目: その場面の種類に出してよいものから選ぶ（ユーザ指定 2026-09-27）。

    「通常」は今の重み（無言も今の割合・特殊フィラーの組も表の重み）でその中から＝う・うー・お は出ない。
    それ以外の種類は等分（例 驚く＝お・悲しむ＝う/無言 を半々）。その種類に mood_special（組 → %）があれば
    組はその % で、残りを等分（ユーザ 2026-09-29: 悩む＝え系 5%・悲しむ＝え系 10%）。種類が表に無ければ「通常」。
    """
    name = mood if mood in table.mood_first else MOOD_DEFAULT
    opts = list(table.mood_first.get(name) or ())
    sp = table.mood_special.get(name) or {}
    if not opts and not sp:
        return draw(table, rng, first=True, last_word=None, last_kind=None, stretch_ok=False)
    if name == MOOD_DEFAULT:
        ws = [table.silence_weight if o == SILENCE else table.weights.get(o, 0.0) for o in opts]
        o = rng.choices(opts, weights=ws, k=1)[0] if sum(ws) > 0 else rng.choice(opts)
    elif sp:
        rest = max(0.0, 100.0 - sum(sp.values()))
        items = opts + list(sp)
        ws = [rest / len(opts)] * len(opts) + list(sp.values()) if opts else list(sp.values())
        o = rng.choices(items, weights=ws, k=1)[0]
    else:
        o = rng.choice(opts)
    if o == SILENCE:
        lo, hi = table.silence_sec
        return Piece("silence", sec=rng.uniform(lo, hi))
    return Piece("word", o)


@dataclass
class Clip:
    """鳴らす1本（前後の無音を刈った後）。"""

    word_id: str
    text: str
    samples: np.ndarray
    sr: int

    @property
    def sec(self) -> float:
        return len(self.samples) / self.sr


class FillerBank:
    """`bank/<voice>/manifest.json` の音声を起動時にメモリへ読む（鳴らす瞬間にディスクを触らない）。"""

    def __init__(self, table: FillerTable, clips: dict, stretched: dict, sr: int,
                 voice_ref: Optional[str] = None, laugh_words: Optional[dict] = None,
                 laughs: Optional[dict] = None, by_emotion: Optional[dict] = None,
                 mood_clips: Optional[dict] = None) -> None:
        self.table = table
        # 語 id → [Clip, ...]（seed 違い）。特殊フィラーの組（id 例 "e+tto"＝えっと）もここ。組は語と っと・っとね を
        # **まとめて合成した**1つの音（ユーザ 2026-09-29「普通に繋げてすべて作成してもいいかも…現在のほぼランダム
        # フィラーならすべて作成したほうがクオリティ高い」・語と っと を間を空けて繋いだ音は「少し不自然」だった）
        self.clips = clips
        self.stretched = stretched  # (語 id, 何本目) → Clip（伸ばし棒を継いだもの）
        self.sr = sr
        self.voice_ref = voice_ref
        # S6 笑い: 笑いの声色の応答非依存フィラー（語 id → [Clip]）と、笑いの語（あはは 等・id → [Clip]）
        self.laugh_words = laugh_words or {}
        self.laughs = laughs or {}
        # S5 感情ごとの声の語（感情 → 語 id → [Clip]）。無い感情・語は普通の声を使う
        self.by_emotion = by_emotion or {}
        # S8 場面の種類の1段目だけ別のバンクの音（(種類, 語 id) → [Clip]・例 不満の えー＝語尾が上がる高い えー）
        self.mood_clips = mood_clips or {}
        self._last_variant: dict = {}

    @classmethod
    def load(cls, table: FillerTable, bank_dir: Path, laugh_dir: Optional[Path] = None,
             emotion_dirs: Optional[dict] = None, emotion_emoji: Optional[dict] = None) -> "FillerBank":
        """バンクを読む。emotion_dirs = 感情 → 感情のバンクのフォルダ（emotion_emoji = その感情の絵文字）。

        感情のバンクは、合成した文字の後ろの絵文字が emotion.yaml と同じ時だけ使う（違えば作り直すまで普通の声）。
        特殊フィラーの組がバンクに無ければ、その組を抽選から外した表で鳴らす（返す bank.table がその表）。
        """
        man = json.loads((Path(bank_dir) / "manifest.json").read_text(encoding="utf-8"))
        # 特別フィラー（っと・っとね）は単独では鳴らさない＝語とまとめて合成した組（id 例 "e+tto"）だけ（2026-09-29）
        want = set(table.weights)
        clips: dict = {}
        sources: dict = {}
        sr = 0
        for c in man.get("clips") or []:
            cid = c.get("id")
            role = c.get("role") or []
            if cid not in want and "stretch_src" not in role:
                continue
            x, sr_c = fa.read_wav((Path(bank_dir) / c["file"]).read_bytes())
            if sr and sr_c != sr:
                raise ValueError(f"バンクのサンプルレートが揃っていない（{c['file']}: {sr_c} / {sr}）")
            sr = sr_c
            if "stretch_src" in role:
                sources.setdefault(c.get("vowel") or cid.replace("stretch_", ""), []).append(x)
                continue
            t = fa.trim(x, sr)
            if len(t) == 0:
                logger.warning("フィラーのクリップが無音（除外）: %s", c["file"])
                continue
            clips.setdefault(cid, []).append(Clip(cid, table.texts.get(cid, c.get("text", "")), t, sr))
        lack_sp = {i for i in want - set(clips) if i in table.special}
        if lack_sp:
            logger.warning("バンクに無い特殊フィラー %s — 抽選から外して鳴らす"
                           "（tools/build_filler_bank.py --combos で作る）", sorted(lack_sp))
            table = table.without(lack_sp)
            want = set(table.weights)
        missing = sorted(want - set(clips))
        if missing:
            raise ValueError(f"バンクに無い語 {missing}（tools/build_filler_bank.py で作る）")
        stretched: dict = {}
        if table.stretch_adopted:
            for wid, vowel in table.stretchable.items():
                src = sources.get(vowel)
                if not src:
                    continue
                for k, cl in enumerate(clips.get(wid, [])):
                    s = fa.stretch(cl.samples, src, sr, add_sec=table.stretch_add_sec)
                    if s is not None:
                        stretched[(wid, k)] = Clip(wid, cl.text + "ー", s, sr)
        refs = (man.get("voice_config") or {}).get("ref_wavs") or []
        laugh_words: dict = {}
        laughs: dict = {}
        if laugh_dir is not None and (Path(laugh_dir) / "manifest.json").is_file():
            lman = json.loads((Path(laugh_dir) / "manifest.json").read_text(encoding="utf-8"))
            for c in lman.get("clips") or []:
                cid = c.get("id")
                if cid not in table.weights and cid not in table.laugh_ids:
                    continue
                x, sr_c = fa.read_wav((Path(laugh_dir) / c["file"]).read_bytes())
                if sr and sr_c != sr:
                    logger.warning("笑いのバンクのサンプルレートが違う（%s）— 使わない", c["file"])
                    continue
                t = fa.trim(x, sr_c)
                if len(t) == 0:
                    continue
                dest = laugh_words if cid in table.weights else laughs
                dest.setdefault(cid, []).append(Clip(cid, table.texts.get(cid, c.get("text", "")), t, sr_c))
        by_emotion: dict = {}
        for label, d in (emotion_dirs or {}).items():
            pool = _load_emotion_bank(table, Path(d), want, sr, (emotion_emoji or {}).get(label))
            if pool:
                by_emotion[label] = pool
        mood_clips: dict = {}
        for (name, wid), bank_name in table.mood_word_bank.items():
            pool = _load_emotion_bank(table, Path(bank_dir).parent / bank_name, {wid}, sr, None)
            if pool.get(wid):
                mood_clips[(name, wid)] = pool[wid]
        return cls(table, clips, stretched, sr, voice_ref=refs[0] if refs else None,
                   laugh_words=laugh_words, laughs=laughs, by_emotion=by_emotion, mood_clips=mood_clips)

    def stretch_ok(self) -> bool:
        return bool(self.table.stretch_adopted) and bool(self.stretched)

    def has_emotion(self, emotion: Optional[str], word_id: str) -> bool:
        """その感情の声の語があるか（無ければ普通の声で鳴らす）。"""
        return bool(emotion) and word_id in self.by_emotion.get(emotion, {})

    def pick(self, word_id: str, rng: random.Random, *, laughing: bool = False,
             emotion: Optional[str] = None, mood: Optional[str] = None) -> tuple[int, Clip]:
        """その語の1本を選ぶ（**直前に鳴らした同じ語の同じ1本は避ける**＝同じ音が続かない）。

        laughing=True で笑いの声色の方から選ぶ（S6 の tone）。mood（S8 の1段目の種類）でその語の別の音が決まって
        いればそれ（例 不満の えー）。emotion があればその感情の声から（S5）。どれも無ければ普通の方。
        """
        if laughing and word_id in self.laugh_words:
            pool, kind = self.laugh_words, "laugh"
        elif mood and (mood, word_id) in self.mood_clips:
            pool, kind = {word_id: self.mood_clips[(mood, word_id)]}, "mood:" + str(mood)
        elif self.has_emotion(emotion, word_id):
            pool, kind = self.by_emotion[emotion], "emo:" + str(emotion)
        else:
            pool, kind = self.clips, ""
        key = (word_id, kind)
        vs = pool[word_id]
        last = self._last_variant.get(key)
        idx = [i for i in range(len(vs)) if i != last] or list(range(len(vs)))
        k = rng.choice(idx)
        self._last_variant[key] = k
        return k, vs[k]

    def pick_laugh(self, rng: random.Random) -> Optional[Clip]:
        """笑いの語（あはは・ふふ …）から1本。表の笑いの語をそのまま等確率で使う（直前と同じ1本は避ける）。"""
        ids = [i for i in self.table.laugh_ids if self.laughs.get(i)]
        if not ids:
            return None
        items = [(i, k) for i in ids for k in range(len(self.laughs[i]))]
        last = self._last_variant.get(("__laugh__", True))
        cand = [x for x in items if x != last] or items
        i, k = rng.choice(cand)
        self._last_variant[("__laugh__", True)] = (i, k)
        return self.laughs[i][k]
