"""Alchemy（錬金輪舞・ユーザ作のブラウザ対戦ゲーム）の名前とルールの文面。

**数値は書き写さない**: 素材の上限・勝ち点・手札の枚数などは、サーバーが部屋に入った時に送ってくる
catalog（`joined.catalog.rules`）から毎回作る。ゲームは開発中なので、ここに数字を書くと静かに嘘になる。
catalog が届く前（部屋に入る前の準備の段階）は、手元の既定値（`DEFAULT_RULES`）で説明する。
この既定値はゲームの `server/data/rules.json`（2026-10-08 時点）と同じ。

席の向き（ゲームのエンジン `server/engine/game.py` と同じ）:
- フラスコの中身は**右隣（送り先）**の素材箱へ送る
- 残りの手札は**左隣**へ渡る。**左隣（送り元）**が自分の素材箱へ送ってくる
- 2人なら右隣も左隣も同じ相手
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

# 素材の呼び名と色（ゲームの `client/src/names.js` と同じ）。色はユーザが「赤の素材」のように呼ぶため
MAT_JA = {
    "dragon": "龍の鱗", "herb": "薬草", "mineral": "鉱物", "homunculus": "ホムンクルス", "stone": "賢者の石",
}
MAT_COLOR_JA = {"dragon": "赤", "herb": "緑", "mineral": "青", "homunculus": "紫", "stone": "金"}
SKILL_JA = {"A": "送り込み", "B": "浄化", "C": "召喚"}
SKILL_DESC = {
    "A": "好きな素材を1個フラスコに入れる（送るだけで、レシピの材料には使えない）",
    "B": "自分の素材箱から素材を2個まで取りのぞく",
    "C": "素材を1種類えらび、このラウンドはその素材を材料として何個でも使える",
}
PHASE_JA = {"select": "レシピをえらぶ", "skill": "スキル", "brew": "錬成", "end": "対戦終了"}

# 部屋に入る前の説明用（catalog が届いたらそちらを使う）
DEFAULT_RULES: dict[str, Any] = {
    "materials": ["dragon", "herb", "mineral", "homunculus", "stone"],
    "basic_materials": ["dragon", "herb", "mineral"],
    "capacity": {"dragon": 9, "herb": 9, "mineral": 9, "homunculus": 4, "stone": 3},
    "hand_size": 4,
    "max_place": 3,
    "win_points": 5,
    "max_rounds": 20,
    "chain_bonus": True,
    "skill_step": 2,
    "timers": {"select": 45, "skill": 45, "brew": 180},
}


def mat(m: str) -> str:
    """素材の呼び名（知らない素材はそのまま）。"""
    return MAT_JA.get(m, m)


def mat_c(m: str) -> str:
    """色つきの呼び名「龍の鱗（赤）」。"""
    c = MAT_COLOR_JA.get(m)
    return f"{mat(m)}（{c}）" if c else mat(m)


def counts_text(counts: dict, materials: Optional[Iterable[str]] = None, *, zero: bool = True,
                capacity: Optional[dict] = None) -> str:
    """{"dragon": 3, ...} → 「龍の鱗3 薬草0 …」（capacity があれば「龍の鱗3/9」）。空なら「なし」。"""
    order = list(materials) if materials is not None else list(MAT_JA)
    parts = []
    for m in order:
        n = int(counts.get(m, 0) or 0)
        if not n and not zero:
            continue
        parts.append(f"{mat(m)}{n}/{capacity[m]}" if capacity and m in capacity else f"{mat(m)}{n}")
    return " ".join(parts) or "なし"


def _slot_text(slot) -> str:
    return "か".join(mat(m) for m in slot)


def sides_text(inputs, outputs) -> str:
    """入力と出力 → 「龍の鱗＋鉱物 → 龍の鱗＋ホムンクルス」（2択の枠は「龍の鱗か薬草」）。"""
    left = "＋".join(_slot_text(s) for s in inputs) or "（材料なし）"
    right = "＋".join(_slot_text(s) for s in outputs) or "（なし）"
    return f"{left} → {right}"


def recipe_text(recipe: dict, rotated: bool = False, *, number: bool = True, emblems: bool = True,
                rotatable_note: bool = False) -> str:
    """catalog のレシピ1枚の説明。rotated なら入力と出力を入れ替えた向き。"""
    ins, outs = (recipe.get("outputs") or [], recipe.get("inputs") or []) if rotated else (
        recipe.get("inputs") or [], recipe.get("outputs") or [])
    head = f"《{recipe.get('name', recipe.get('id', '?'))}》"
    if number:
        head += f"(番号{recipe.get('number', '?')})"
    if rotated:
        head += "（逆向き）"
    text = f"{head} {sides_text(ins, outs)}"
    skills = recipe.get("skills") or []
    if emblems and skills:
        text += f"［紋章 {'・'.join(SKILL_JA.get(s, s) for s in skills)}］"
    if rotatable_note and recipe.get("rotatable") and not rotated:
        text += "（逆向きにも置ける）"
    return text


def rules_brief(rules: Optional[dict] = None) -> str:
    """会話（応答・発話判定）に入れる短いルール。遊び方を話せる程度（作戦の細部は入れない）。"""
    r = rules or DEFAULT_RULES
    cap = r.get("capacity", {})
    # 上限が同じ素材はまとめて言う（「龍の鱗（赤）・薬草（緑）・鉱物（青）は9個まで」）。数は catalog から
    groups: dict = {}
    for m in r.get("materials", []):
        groups.setdefault(cap.get(m, "?"), []).append(m)
    caps = "、".join(f"{'・'.join(mat_c(m) for m in ms)}は{n}個まで" for n, ms in groups.items())
    return (
        "ユーザが作った、錬金術がテーマのブラウザ対戦ゲーム（2〜4人・全員が同時に進める）。\n"
        "・目的: レシピで素材を作り変えてフラスコに入れ、ラウンドの最後に右隣（送り先）の素材箱へ送る。"
        f"箱に入りきらずにあふれた素材1個につき、送った人に1点。先に{r.get('win_points', '?')}点で勝ち"
        f"（同点なら素材箱の中身が少ない人。{r.get('max_rounds', '?')}ラウンドで打ち切り）。\n"
        f"・素材箱に入る数: {caps}。自分の箱には左隣（送り元）が送ってくる＝自分の箱があふれると相手の点。\n"
        f"・1ラウンド: ①手札{r.get('hand_size', '?')}枚から1〜{r.get('max_place', '?')}枚のレシピを工房に置く（残りは左隣へ回る）"
        f" ②置いたカードの紋章が{r.get('skill_step', '?')}つたまるとスキル（送り込み・浄化・召喚）"
        " ③錬成: レシピで箱の素材を作り変える。フラスコの素材を別のレシピの材料にすると連鎖でボーナス"
        " ④フラスコの中身を右隣へ送る。\n"
        "・2人で遊ぶ時は、右隣も左隣も同じ相手。"
    )


def rules_full(rules: Optional[dict] = None) -> str:
    """ゲームの判断役に渡す詳しいルール（作戦を立てられる程度）。"""
    r = rules or DEFAULT_RULES
    cap = r.get("capacity", {})
    t = r.get("timers", {})
    caps = "、".join(f"{mat_c(m)} {cap.get(m, '?')}個" for m in r.get("materials", []))
    basic = "・".join(mat(m) for m in r.get("basic_materials", []))
    step = r.get("skill_step", "?")
    return (
        "# ゲーム「Alchemy」のルール\n"
        "2〜4人が同時に進める対戦ゲーム。席は輪になっている。\n"
        "・**フラスコの中身はラウンドの最後に、まるごと右隣（送り先）の素材箱へ送られる**。"
        "箱の上限を超えてあふれた素材1個につき、送った人に1点。"
        f"先に{r.get('win_points', '?')}点以上で勝ち（同じラウンドに複数人なら点の多い人→素材箱の中身が少ない人。"
        f"{r.get('max_rounds', '?')}ラウンドで決着しなければ同じ決め方）。\n"
        f"・素材箱の上限: {caps}。自分の箱には左隣（送り元）が送ってくるので、自分の箱が満杯に近いほど相手に点を取られやすい。"
        "素材箱の中身を減らしておくのは守りになる。\n"
        "・2人で遊ぶ時は、右隣と左隣は同じ相手（お互いに送り合う）。\n"
        "## 1ラウンドの流れ\n"
        f"1. えらぶ: 手札{r.get('hand_size', '?')}枚から1〜{r.get('max_place', '?')}枚のレシピを自分の工房に置く"
        f"（制限 {t.get('select', '?')}秒）。残りの手札は左隣へ渡り、右隣から手札が回ってくる。"
        "工房はラウンドごとに空になる。逆向きOKのレシピは、置く時に入力と出力を入れ替えられる。"
        "カードの番号が小さい人ほどスキルの処理が先。\n"
        f"2. スキル: 置いたカードの紋章が種類ごとにたまり、{step}つたまるごとに1回発動（制限 {t.get('skill', '?')}秒）。"
        + "".join(f"{SKILL_JA[k]}＝{SKILL_DESC[k]}。" for k in ("A", "B", "C")) + "\n"
        f"3. 錬成（制限 {t.get('brew', '?')}秒）: 工房のレシピを1回ずつ使える。材料を素材箱（召喚した素材も可）から"
        "取って変換し、できた素材はフラスコに入る。フラスコの素材を別のレシピの材料にすると連鎖になり、"
        f"2連鎖で{basic}のどれか1個、3連鎖で2個のボーナスがフラスコに増える。"
        "出力が「AかB」のレシピは、どちらを作るか選ぶ。\n"
        "4. 送る: 全員のフラスコの中身が右隣へ送られ、あふれた分が得点になる。\n"
        "## 作戦の基本\n"
        "・送り先の箱で満杯に近い素材を送るとあふれて点になる（上限の小さいホムンクルス・賢者の石はあふれやすい）。\n"
        "・自分の箱の素材をレシピで使うと、箱が減って守りになる。\n"
        "・置かなかったカードは左隣に渡るので、多く置けば相手に良いカードを渡しにくい。"
    )


def seat_names(players: list[dict], me: int) -> tuple[str, str]:
    """(右隣＝送り先の名前, 左隣＝送り元の名前)。"""
    n = len(players)
    if n < 2:
        return "", ""
    right = players[(me - 1) % n].get("name", "?")
    left = players[(me + 1) % n].get("name", "?")
    return right, left
