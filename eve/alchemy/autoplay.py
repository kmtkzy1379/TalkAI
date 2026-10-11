"""コードだけで手を決める（LLM を使わない）。

2つの役目:
- 遊ばせ方 strategy（ユーザ案2）の実行役: LLM が決めた作戦（多く送りたい素材・減らしたい素材）の重みで、
  えらぶ・スキル・錬成をここが決める。
- 遊ばせ方 action（ユーザ案1）の安全網: LLM が締切までに答えない・形式が崩れた・合法でない時は、ここの手で打つ。

入力はサーバーの state（`view`）と catalog から作った Rules / Recipe だけ（純粋な関数・通信しない）。
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from . import sim
from .sim import NO_STRATEGY, Strategy


def me_player(view: dict) -> dict:
    seat = (view.get("me") or {}).get("seat", 0)
    return view["players"][seat]


def target_player(view: dict) -> dict:
    """送り先（右隣＝(自分-1)%n）。"""
    seat = (view.get("me") or {}).get("seat", 0)
    n = len(view["players"])
    return view["players"][(seat - 1) % n]


def tableau_sides(view: dict, recipes: dict) -> list:
    """自分の工房のレシピの (入力, 出力)（置いた向きを反映）。"""
    me = me_player(view)
    return [recipes[t["card"]].sides(bool(t.get("rotated"))) for t in me.get("tableau") or [] if t["card"] in recipes]


def select_options(view: dict, rules: sim.Rules, recipes: dict, strategy: Strategy = NO_STRATEGY) -> list:
    me = me_player(view)
    hand = list((view.get("me") or {}).get("hand") or [])
    return sim.selection_options(rules, recipes, hand, me.get("box") or {}, target_player(view).get("box") or {},
                                 me.get("skills") or {}, strategy)


def choose_select(view: dict, rules: sim.Rules, recipes: dict, strategy: Strategy = NO_STRATEGY) -> list:
    """置くカード [(card, rotated)]。候補が無ければ手札の先頭1枚（サーバーの時間切れと同じ扱い）。"""
    opts = select_options(view, rules, recipes, strategy)
    if opts:
        return opts[0].picks
    hand = list((view.get("me") or {}).get("hand") or [])
    return [(hand[0], False)] if hand else []


def planned_flask(view: dict, rules: sim.Rules, recipes: dict, supply=(),
                  strategy: Strategy = NO_STRATEGY) -> Counter:
    """今の工房で錬成した時に送る見込みの素材（送り込みの選び方に使う）。"""
    me = me_player(view)
    start = sim.BrewCore(box=Counter(me.get("box") or {}), supply=frozenset(supply))
    steps = sim.best_plan(rules, tableau_sides(view, recipes), set(), start, target_player(view).get("box") or {},
                          strategy)
    return sim.run_steps(rules, start, target_player(view).get("box") or {}, steps).flask()


def choose_skills(view: dict, rules: sim.Rules, recipes: dict, strategy: Strategy = NO_STRATEGY,
                  only: Optional[set] = None) -> list:
    """発動中のスキルの使い方 [{"skill", "materials"}]。召喚 → 浄化 → 送り込みの順に決める。

    - 召喚(C): その素材を使い放題にした時の錬成の見積もりが最良の素材
    - 浄化(B): 錬成で使う予定の分を残し、埋まっている素材から最大2個（余りが無ければ何も捨てない）
    - 送り込み(A): 錬成で送る予定の分も足して、送り先であふれる素材を最優先（無ければ空きの少ない素材）
    `only` を渡すとその種類だけ決める（LLM の答えに抜けていた分の補い）。
    """
    me_view = view.get("me") or {}
    pending = {k: int(v) for k, v in (me_view.get("skill_pending") or {}).items() if int(v) > 0}
    if only is not None:
        pending = {k: v for k, v in pending.items() if k in only}
    if not pending:
        return []
    me = me_player(view)
    box = dict(me.get("box") or {})
    target = target_player(view).get("box") or {}
    sides = tableau_sides(view, recipes)
    supply = set((me_view.get("skill_choices") or {}).get("supply") or [])
    sealed = Counter((me_view.get("skill_choices") or {}).get("sealed") or {})
    uses: list = []
    for _ in range(pending.get("C", 0)):
        # 召喚済みの素材をもう一度選ぶのは無駄打ち（使い放題は1回で足りる）なので、まだの素材から選ぶ
        ranked = [mo for mo in sim.best_supply(rules, sides, box, target, supply, strategy) if mo[0] not in supply]
        m = ranked[0][0] if ranked else rules.materials[0]
        supply.add(m)
        uses.append({"skill": "C", "materials": [m]})
    for _ in range(pending.get("B", 0)):
        need = sim.needed_from_box(rules, sides, box, target, supply, strategy)
        picks = sim.purge_choice(rules, box, need, strategy)
        for m in picks:
            box[m] = box.get(m, 0) - 1
        uses.append({"skill": "B", "materials": picks})
    if pending.get("A", 0):
        start = sim.BrewCore(box=Counter(box), supply=frozenset(supply))
        steps = sim.best_plan(rules, sides, set(), start, target, strategy)
        planned = sim.run_steps(rules, start, target, steps).flask()
        for _ in range(pending["A"]):
            m = sim.seal_ranking(rules, target, planned, sealed, strategy)[0]
            sealed[m] += 1
            uses.append({"skill": "A", "materials": [m]})
    return uses


def brew_start(view: dict) -> Optional[sim.BrewCore]:
    """state の自分の錬成状態（錬成の段階でなければ None）。"""
    me_view = view.get("me") or {}
    b = me_view.get("brew")
    if not b:
        return None
    return sim.core_from_view(b, me_player(view).get("box") or {})


def brew_target(view: dict) -> dict:
    """錬成の見積もりに使う送り先の箱（錬成開始時の値・エンジンの自動ボーナスもこれを使う）。"""
    b = (view.get("me") or {}).get("brew") or {}
    return dict(b.get("target_box") or target_player(view).get("box") or {})


def choose_brew(view: dict, rules: sim.Rules, recipes: dict, strategy: Strategy = NO_STRATEGY) -> list:
    """錬成の手順（sim.Step のリスト）。"""
    start = brew_start(view)
    if start is None:
        return []
    used = set((view.get("me") or {}).get("brew", {}).get("used") or [])
    return sim.best_plan(rules, tableau_sides(view, recipes), used, start, brew_target(view), strategy)
