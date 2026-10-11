"""Alchemy の錬成の計算と自動プレイの見積もり（**通信も LLM も持たない純粋な計算**）。

錬成の規則はゲームのエンジン（FlaskRondo `server/engine/core.py`）の移植。見積もりの点数は同じく
`server/engine/bots.py` の GreedyBot（サーバーが切断した人の代わりに打つ Bot）と同じ式に、
作戦（どの素材を多く送りたいか・減らしたいか）の重みを足したもの。

**移植にした理由**: ゲームは別リポジトリで、イブはサーバーの公開情報（部屋に入った時の catalog と
自分の state）しか持たない。エンジンを import すると別リポジトリの内部構造に依存する。
ずれた時の安全網: 実際の錬成はサーバーが検証する（`brew.place` が通らなければ `brew.auto` →
それも通らなければそのレシピは諦める）。エンジンとの突き合わせは `tools/alchemy_sim_check.py`。

数値（素材の上限・ボーナスの有無など）はすべて catalog の Rules から読む（書き写さない）。
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

BOX, PRODUCT, SUPPLY = "box", "product", "supply"
SOURCES = (BOX, PRODUCT, SUPPLY)
RARITY = {"stone": 2.0, "homunculus": 1.5}  # bots.py と同じ（上限が小さい素材ほど送る価値が高い）
SEND_W = 0.6        # 送った素材1個（bots.py と同じ）
OVERFLOW_W = 3.0    # あふれ見込み1個（＝1点）
DRAIN_W = 0.3       # 自分の箱から減らした素材1個の基本（＋その素材の埋まり具合）
FOCUS_BONUS = 1.0   # 作戦で「多く送りたい／減らしたい」素材に足す重み


# ---------------------------------------------------------------------------- データ
@dataclass(frozen=True)
class Rules:
    materials: tuple
    basic: tuple
    capacity: dict
    hand_size: int = 4
    max_place: int = 3
    win_points: int = 5
    max_rounds: int = 20
    chain_bonus: bool = True
    skill_step: int = 2
    timers: dict = field(default_factory=dict)

    @classmethod
    def from_catalog(cls, d: dict) -> "Rules":
        return cls(
            materials=tuple(d["materials"]),
            basic=tuple(d.get("basic_materials", ())),
            capacity={k: int(v) for k, v in d["capacity"].items()},
            hand_size=int(d.get("hand_size", 4)),
            max_place=int(d.get("max_place", 3)),
            win_points=int(d.get("win_points", 5)),
            max_rounds=int(d.get("max_rounds", 20)),
            chain_bonus=bool(d.get("chain_bonus", True)),
            skill_step=max(1, int(d.get("skill_step", 2))),
            timers=dict(d.get("timers", {})),
        )


Slot = tuple


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    number: int
    rotatable: bool
    skills: tuple
    inputs: tuple
    outputs: tuple

    def sides(self, rotated: bool) -> tuple:
        """向きを反映した (入力, 出力)。回転すると入力と出力が入れ替わる。"""
        return (self.outputs, self.inputs) if rotated else (self.inputs, self.outputs)

    @classmethod
    def from_catalog(cls, d: dict) -> "Recipe":
        return cls(
            id=str(d["id"]), name=str(d.get("name", d["id"])), number=int(d.get("number", 0)),
            rotatable=bool(d.get("rotatable", False)), skills=tuple(d.get("skills", ())),
            inputs=tuple(tuple(s) for s in d.get("inputs", ())),
            outputs=tuple(tuple(s) for s in d.get("outputs", ())),
        )


def load_catalog(catalog: dict) -> tuple:
    """`joined.catalog` → (Rules, {id: Recipe})。"""
    rules = Rules.from_catalog(catalog["rules"])
    recipes = {r["id"]: Recipe.from_catalog(r) for r in catalog.get("recipes", [])}
    return rules, recipes


@dataclass(frozen=True)
class Placement:
    material: str
    source: str
    depth: int = 0


@dataclass
class BrewCore:
    """1人分の錬成状態（エンジンの BrewCore と同じ）。産物は (素材, 段数) で数える。"""

    box: Counter
    products: Counter = field(default_factory=Counter)
    sealed: Counter = field(default_factory=Counter)
    supply: frozenset = frozenset()

    def clone(self) -> "BrewCore":
        return BrewCore(Counter(self.box), Counter(self.products), Counter(self.sealed), self.supply)

    def product_counts(self) -> Counter:
        counts: Counter = Counter()
        for (m, _d), n in self.products.items():
            if n > 0:
                counts[m] += n
        return counts

    def flask(self) -> Counter:
        """送る素材（産物＋送り込み）。"""
        total = self.product_counts()
        total.update(self.sealed)
        return total

    def max_product_depth(self, material: str) -> Optional[int]:
        depths = [d for (m, d), n in self.products.items() if m == material and n > 0]
        return max(depths) if depths else None

    def can_take(self, material: str, source: str) -> bool:
        if source == BOX:
            return self.box.get(material, 0) > 0
        if source == PRODUCT:
            return self.max_product_depth(material) is not None
        if source == SUPPLY:
            return material in self.supply
        return False

    def take(self, material: str, source: str) -> Placement:
        if source == BOX:
            if self.box.get(material, 0) <= 0:
                raise ValueError("box_empty")
            self.box[material] -= 1
            return Placement(material, BOX)
        if source == PRODUCT:
            depth = self.max_product_depth(material)
            if depth is None:
                raise ValueError("no_product")
            key = (material, depth)
            self.products[key] -= 1
            if self.products[key] <= 0:
                del self.products[key]
            return Placement(material, PRODUCT, depth)
        if source == SUPPLY:
            if material not in self.supply:
                raise ValueError("no_supply")
            return Placement(material, SUPPLY)
        raise ValueError("bad_source")

    def produce(self, materials: Iterable[str], depth: int) -> None:
        for m in materials:
            self.products[(m, depth)] += 1


def chain_depth(placements: Iterable[Placement]) -> int:
    """何段目の連鎖か。産物を使わなければ1段目。"""
    return 1 + max((p.depth for p in placements if p.source == PRODUCT), default=0)


def bonus_count(depth: int, rules: Rules) -> int:
    return depth - 1 if rules.chain_bonus else 0


def auto_bonus(core: BrewCore, rules: Rules, target_box: dict, extra: list) -> str:
    """連鎖ボーナスの自動選択（エンジンと同じ: 送り先の箱で空きが一番少ない基本素材）。"""
    planned = core.flask()
    planned.update(extra)

    def free(m: str) -> int:
        return rules.capacity[m] - target_box.get(m, 0) - planned[m]

    return min(rules.basic, key=lambda m: (free(m), rules.basic.index(m)))


def _assign(combo: tuple, core: BrewCore, prefer_products: bool):
    """入力の素材の組 combo に出どころを割り当てる → [(素材, 出どころ)] か None（足りない）。

    出どころの優先はエンジンの plan_sources と同じ: 既定は「素材箱 → 召喚 → フラスコの産物」
    （箱を減らして送る量を最大にする）、prefer_products は連鎖ボーナスを狙う「産物 → 箱 → 召喚」。
    """
    products = core.product_counts()
    order = (PRODUCT, BOX, SUPPLY) if prefer_products else (BOX, SUPPLY, PRODUCT)
    assign: dict = {}
    for m, n in Counter(combo).items():
        sources: list = []
        for src in order:
            if src == BOX:
                k = min(n - len(sources), core.box.get(m, 0))
            elif src == PRODUCT:
                k = min(n - len(sources), products.get(m, 0))
            else:
                k = (n - len(sources)) if m in core.supply else 0
            sources += [src] * k
        if len(sources) < n:
            return None
        assign[m] = sources
    cursor = {m: 0 for m in assign}
    plan = []
    for m in combo:
        plan.append((m, assign[m][cursor[m]]))
        cursor[m] += 1
    return plan


def source_plans(slots, core: BrewCore) -> list:
    """入力枠の埋め方の候補すべて（素材の組 × 出どころの優先2通り・重複なし）。足りなければ空。

    エンジンの Bot は「費用が最小の1通り」しか試さないが、イブは全部試す（`brew_plans`）＝Bot より悪くならない。
    """
    slots = list(slots)
    if not slots:
        return [[]]
    plans: list = []
    for combo in sorted(set(itertools.product(*slots))):
        for prefer in (False, True):
            plan = _assign(combo, core, prefer)
            if plan is not None and plan not in plans:
                plans.append(plan)
    return plans


# ---------------------------------------------------------------------------- 作戦と点数
@dataclass(frozen=True)
class Strategy:
    """作戦（ユーザ案2「赤の素材を多く送りたい・青の素材を消費したい」）。素材のキーで持つ。"""

    send: tuple = ()
    consume: tuple = ()
    note: str = ""

    def is_empty(self) -> bool:
        return not self.send and not self.consume


NO_STRATEGY = Strategy()


def score(core: BrewCore, start_box: Counter, target_box: dict, rules: Rules,
          strategy: Strategy = NO_STRATEGY) -> float:
    """錬成後の状態の点数（bots.py GreedyBot._score と同じ式＋作戦の重み）。"""
    flask = core.flask()
    total = 0.0
    for m in rules.materials:
        cap = rules.capacity[m]
        sent = flask.get(m, 0)
        free = cap - target_box.get(m, 0)
        send_w = SEND_W * RARITY.get(m, 1.0) + (FOCUS_BONUS if m in strategy.send else 0.0)
        total += sent * send_w + max(0, sent - free) * OVERFLOW_W
        drained = start_box.get(m, 0) - core.box.get(m, 0)
        drain_w = DRAIN_W + start_box.get(m, 0) / cap + (FOCUS_BONUS if m in strategy.consume else 0.0)
        total += drained * drain_w
    return total


@dataclass
class Step:
    """錬成の1手: レシピ index を plan の出どころで作り、outputs を選ぶ。"""

    score: float
    index: int
    plan: list
    outputs: tuple
    depth: int = 1
    bonus: tuple = ()


def _apply_step(core: BrewCore, rules: Rules, target_box: dict, plan, outputs) -> tuple:
    placements = [core.take(m, src) for m, src in plan]
    depth = chain_depth(placements)
    bonus: list = []
    for _b in range(bonus_count(depth, rules)):
        bonus.append(auto_bonus(core, rules, target_box, list(outputs) + bonus))
    core.produce(list(outputs) + bonus, depth)
    return depth, tuple(bonus)


def run_steps(rules: Rules, start: BrewCore, target_box: dict, steps: list) -> BrewCore:
    """手順を実際に当てはめた後の状態（見積もりの説明用）。"""
    core = start.clone()
    for s in steps:
        _apply_step(core, rules, target_box, s.plan, s.outputs)
    return core


@dataclass
class Outcome:
    """手順の結果の見積もり（イブの判断役・会話に渡す事実）。"""

    score: float
    sent: dict
    overflow: dict
    drained: dict
    max_depth: int
    used: tuple

    @property
    def points(self) -> int:
        return sum(self.overflow.values())


def outcome(rules: Rules, start: BrewCore, target_box: dict, steps: list,
            strategy: Strategy = NO_STRATEGY) -> Outcome:
    end = run_steps(rules, start, target_box, steps)
    flask = end.flask()
    sent = {m: flask[m] for m in rules.materials if flask.get(m, 0) > 0}
    overflow = {}
    for m, k in sent.items():
        over = k - max(0, rules.capacity[m] - target_box.get(m, 0))
        if over > 0:
            overflow[m] = over
    drained = {m: start.box.get(m, 0) - end.box.get(m, 0) for m in rules.materials
               if start.box.get(m, 0) - end.box.get(m, 0) > 0}
    return Outcome(
        score=score(end, Counter(start.box), target_box, rules, strategy),
        sent=sent, overflow=overflow, drained=drained,
        max_depth=max((s.depth for s in steps), default=0),
        used=tuple(s.index for s in steps),
    )


# ---------------------------------------------------------------------------- えらぶ
def card_options(hand: list, recipes: dict, max_place: int) -> list:
    """置き方の候補すべて: 手札から1〜max_place枚 × 逆OKのカードの向き（bots.py card_options と同じ）。"""
    options: list = []
    hand = [c for c in hand if c in recipes]
    for k in range(1, min(max_place, len(hand)) + 1):
        for combo in itertools.combinations(hand, k):
            ways = [[(c, False)] + ([(c, True)] if recipes[c].rotatable else []) for c in combo]
            options.extend(list(p) for p in itertools.product(*ways))
    return options


def skill_triggers(skill_counts: dict, picks: list, recipes: dict, step: int) -> dict:
    """その置き方で発動するスキルの回数（紋章が step つたまるごとに1回・エンジンの finish_select と同じ）。"""
    gained: Counter = Counter()
    for card, _rot in picks:
        gained.update(recipes[card].skills)
    out = {}
    for s, g in gained.items():
        before = int(skill_counts.get(s, 0))
        t = (before + g) // step - before // step
        if t:
            out[s] = t
    return out


@dataclass
class SelectOption:
    picks: list
    value: float
    steps: list
    outcome: Outcome
    triggers: dict


def selection_options(rules: Rules, recipes: dict, hand: list, box: dict, target_box: dict,
                      skill_counts: dict, strategy: Strategy = NO_STRATEGY) -> list:
    """置き方の全候補を、錬成まで見積もった点数の高い順に返す（同点なら多く置く方＝左隣へ良い札を渡さない）。"""
    out: list = []
    for picks in card_options(hand, recipes, rules.max_place):
        sides = [recipes[c].sides(r) for c, r in picks]
        start = BrewCore(box=Counter(box))
        steps = best_plan(rules, sides, set(), start, target_box, strategy)
        oc = outcome(rules, start, target_box, steps, strategy)
        out.append(SelectOption(picks, oc.score, steps, oc,
                                skill_triggers(skill_counts, picks, recipes, rules.skill_step)))
    out.sort(key=lambda o: (-round(o.value, 6), -len(o.picks), [(c, r) for c, r in o.picks]))
    return out


# ---------------------------------------------------------------------------- スキル
def best_supply(rules: Rules, sides: list, box: dict, target_box: dict, supply: Iterable[str],
                strategy: Strategy = NO_STRATEGY) -> list:
    """召喚（C）の候補: 素材ごとに「その素材を使い放題にした時の錬成の見積もり」を高い順に。"""
    ranked = []
    for m in rules.materials:
        start = BrewCore(box=Counter(box), supply=frozenset(set(supply) | {m}))
        steps = best_plan(rules, sides, set(), start, target_box, strategy)
        oc = outcome(rules, start, target_box, steps, strategy)
        ranked.append((m, oc))
    ranked.sort(key=lambda mo: (-round(mo[1].score, 6), rules.materials.index(mo[0])))
    return ranked


def needed_from_box(rules: Rules, sides: list, box: dict, target_box: dict, supply: Iterable[str],
                    strategy: Strategy = NO_STRATEGY) -> Counter:
    """今の工房の見積もりの手順で、素材箱から使う予定の素材（浄化で捨ててはいけない分）。"""
    start = BrewCore(box=Counter(box), supply=frozenset(supply))
    steps = best_plan(rules, sides, set(), start, target_box, strategy)
    need: Counter = Counter()
    for s in steps:
        for m, src in s.plan:
            if src == BOX:
                need[m] += 1
    return need


def purge_choice(rules: Rules, box: dict, needed: Counter, strategy: Strategy = NO_STRATEGY,
                 limit: int = 2) -> list:
    """浄化（B）で取りのぞく素材（最大 limit 個）。錬成で使う予定の分は残し、埋まっている素材から。

    減らしたい素材（作戦）を優先する。余りが無ければ何も捨てない（使う予定の素材を捨てるのは自爆）。
    """
    spare = Counter({m: max(0, int(box.get(m, 0)) - needed.get(m, 0)) for m in rules.materials})
    picks: list = []
    for _ in range(limit):
        cands = [m for m in rules.materials if spare[m] > 0]
        if not cands:
            break
        m = max(cands, key=lambda x: (x in strategy.consume, (box.get(x, 0) - picks.count(x)) / rules.capacity[x],
                                      -rules.materials.index(x)))
        picks.append(m)
        spare[m] -= 1
    return picks


def seal_ranking(rules: Rules, target_box: dict, planned: Counter, sealed: Counter,
                 strategy: Strategy = NO_STRATEGY) -> list:
    """送り込み（A）の候補を良い順に。錬成で送る予定の分も足して、あふれるか（＝1点）を最優先にする。"""
    def value(m: str) -> tuple:
        free = rules.capacity[m] - target_box.get(m, 0) - planned.get(m, 0) - sealed.get(m, 0)
        return (free <= 0, m in strategy.send, RARITY.get(m, 1.0), -free, -rules.materials.index(m))

    return sorted(rules.materials, key=value, reverse=True)


def free_after(rules: Rules, target_box: dict, planned: Counter, m: str) -> int:
    """送り先の箱の、予定どおり送った後の空き（0 以下ならあふれる）。"""
    return rules.capacity[m] - target_box.get(m, 0) - planned.get(m, 0)


# ---------------------------------------------------------------------------- 錬成の候補（1行動ごとの判断用）
def brew_plans(rules: Rules, sides: list, used: Iterable[int], start: BrewCore, target_box: dict,
               strategy: Strategy = NO_STRATEGY, limit: int = 5) -> list:
    """錬成の手順の候補（順番・入力の素材の組・出どころの優先・出力の選び方を全部試す）を見積もりの高い順に最大 limit 件。

    工房は多くても数枚なので全部試しても一瞬。1手も作らない手順は含めない（作れるのに何もしないのは自爆）。
    1件目は全部試した中の最良＝コードの手（`best_plan`）と同じなので、自動プレイの手は必ず候補に入る。
    """
    used = set(used)
    free_idx = [i for i in range(len(sides)) if i not in used]
    start_box = Counter(start.box)
    seen: dict = {}

    def explore(core: BrewCore, done: list, left: list) -> None:
        if done:
            key = tuple((s.index, s.outputs, tuple(s.plan)) for s in done)
            if key not in seen:
                seen[key] = list(done)
        for i in left:
            ins, outs = sides[i]
            for plan in source_plans(list(ins), core):
                for out_combo in sorted(set(itertools.product(*outs))):
                    nxt = core.clone()
                    depth, bonus = _apply_step(nxt, rules, target_box, plan, out_combo)
                    sc = score(nxt, start_box, target_box, rules, strategy)
                    step = Step(sc, i, plan, tuple(out_combo), depth, bonus)
                    explore(nxt, done + [step], [j for j in left if j != i])

    explore(start.clone(), [], free_idx)
    ranked = []
    for steps in seen.values():
        oc = outcome(rules, start, target_box, steps, strategy)
        ranked.append((oc.score, steps, oc))
    ranked.sort(key=lambda x: (-round(x[0], 6), -len(x[1])))
    # 結果が同じ手順（作る順番を入れ替えただけ等）は1つにまとめる＝候補として見せる意味のある違いだけ残す
    out: list = []
    seen_results: set = set()
    for _sc, steps, oc in ranked:
        sig = (tuple(sorted(oc.sent.items())), tuple(sorted(oc.drained.items())), oc.max_depth)
        if sig in seen_results:
            continue
        seen_results.add(sig)
        out.append((steps, oc))
        if len(out) >= limit:
            break
    return out


def best_plan(rules: Rules, sides: list, used: Iterable[int], start: BrewCore, target_box: dict,
              strategy: Strategy = NO_STRATEGY) -> list:
    """いちばん点数の高い錬成の手順（全部試した中の最良。作らない方が良ければ空）。

    Bot（GreedyBot）は1手ずつの貪欲法で、出力の選び方も4通りまでしか見ない。工房は3枚までなので
    全部試しても軽く、貪欲法より悪くならない（`tools/alchemy_sim_check.py` で確認）。
    """
    plans = brew_plans(rules, sides, used, start, target_box, strategy, limit=1)
    if not plans:
        return []
    steps, oc = plans[0]
    base = score(start, Counter(start.box), target_box, rules, strategy)
    return steps if oc.score > base + 1e-9 else []


def core_from_view(me_brew: dict, box: dict) -> BrewCore:
    """state の me.brew（と自分の箱）から錬成状態を作る。"""
    products: Counter = Counter()
    for m, d, n in me_brew.get("products_by_depth") or []:
        products[(str(m), int(d))] += int(n)
    return BrewCore(
        box=Counter({k: int(v) for k, v in (box or {}).items()}),
        products=products,
        sealed=Counter({k: int(v) for k, v in (me_brew.get("sealed") or {}).items()}),
        supply=frozenset(me_brew.get("supply") or []),
    )
