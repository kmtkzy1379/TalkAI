r"""イブの錬成の計算（`eve/alchemy/sim.py`）を、ゲーム本体のエンジンと突き合わせる（API 不要）。

ゲームのエンジン（FlaskRondo の `server/engine`）を**この道具の中でだけ** import する（イブ本体は import しない）。
ランダムな素材箱・手札で、次の3つが一致するかを数える:
1. イブの錬成の手順（全部試した最良）の点数が、Bot（GreedyBot.plan_brew・貪欲法）**以上**か
   （Bot は出力の選び方を4通りまでしか見ず、同点の時の選び方も集合の順番で決まるので一致はしない）
2. イブが決めた手順をエンジンに実際に打ち込んだ後のフラスコ・素材箱が、イブの見積もりと**同じ**か
3. 置き方の候補の数が Bot と同じで、いちばん良い置き方の点数が Bot（choose_cards の評価）**以上**か

使い方:
  $env:PYTHONIOENCODING="utf-8"
  & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\alchemy_sim_check.py [--n 300] [--engine C:\Users\user\Desktop\FlaskRondo]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.alchemy import sim  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--engine", default=os.getenv("ALCHEMY_ENGINE_DIR", r"C:\Users\user\Desktop\FlaskRondo"))
    args = ap.parse_args()
    sys.path.insert(0, args.engine)
    from server.engine import Game, load_recipes, load_rules  # noqa: E402
    from server.engine import core as ecore  # noqa: E402
    from server.engine.bots import GreedyBot, card_options  # noqa: E402

    erules = load_rules()
    erecipes = load_recipes(erules)
    catalog = {"rules": erules.to_public(), "recipes": [r.to_public() for r in erecipes.values()]}
    rules, recipes = sim.load_catalog(catalog)
    rng = random.Random(args.seed)
    bot = GreedyBot(seed=1, epsilon=0.0)

    plan_same = 0
    exec_same = 0
    exec_total = 0
    sel_same = 0
    mism: list = []
    for trial in range(args.n):
        box = {m: rng.randint(0, rules.capacity[m]) for m in rules.materials}
        target = {m: rng.randint(0, rules.capacity[m]) for m in rules.materials}
        hand = rng.sample(list(recipes), 4)
        k = rng.randint(1, 3)
        picks = [(c, recipes[c].rotatable and rng.random() < 0.5) for c in hand[:k]]
        supply = frozenset(rng.sample(list(rules.materials), rng.choice([0, 0, 1])))
        # 1. 貪欲法の最終点数
        sides = [recipes[c].sides(r) for c, r in picks]
        start = sim.BrewCore(box=Counter(box), supply=supply)
        mine = sim.best_plan(rules, sides, set(), start, target)
        g = Game(erules, erecipes, [("a", "blue"), ("b", "red")], seed=trial)
        esides = [erecipes[c].sides(r) for c, r in picks]
        theirs = bot.plan_brew(g, esides, set(), ecore.BrewCore(box=Counter(box), supply=supply), target)
        base = sim.score(start, Counter(box), target, rules)
        ms = round(sim.outcome(rules, start, target, mine).score if mine else base, 6)
        ts = round(theirs[-1].score if theirs else base, 6)
        if ms >= ts - 1e-6:
            plan_same += 1
        else:
            mism.append({"kind": "plan", "box": box, "target": target, "picks": picks, "mine": ms, "bot": ts})
        # 2. イブの手順をエンジンに打ち込む（召喚なしの時だけ・スキルの段階を通さず錬成に入る）
        if not supply and mine:
            g = Game(erules, erecipes, [("a", "blue"), ("b", "red")], seed=trial)
            p0, p1 = g.players
            p0.box = Counter(box)
            p1.box = Counter(target)
            p0.hand = [c for c, _r in picks]
            p1.hand = [hand[-1]]
            g.select(0, [[c, r] for c, r in picks])
            g.select(1, [[hand[-1], False]])
            g.finish_select()
            if g.phase != "brew":
                # スキルが発動した時は全部使わずに進める
                g.skill_skip(0)
                g.skill_skip(1)
                g.finish_skill()
            exec_total += 1
            ok = True
            for st in mine:
                ins, outs = sides[st.index]
                choice = {kk: m for kk, m in enumerate(st.outputs) if len(outs[kk]) > 1}
                try:
                    if choice:
                        g.brew_choose(0, st.index, outputs=choice)
                    if not ins:
                        g.brew_auto(0, st.index)
                    else:
                        for slot, (m, src) in enumerate(st.plan):
                            g.brew_place(0, st.index, slot, m, src)
                except Exception as e:  # noqa: BLE001
                    ok = False
                    mism.append({"kind": "exec_error", "err": repr(e), "picks": picks, "box": box})
                    break
            if ok:
                predicted = sim.run_steps(rules, sim.BrewCore(box=Counter(box)), target, mine)
                real_flask = dict(p0.brew.core.flask())
                pf = {m: v for m, v in predicted.flask().items() if v}
                rf = {m: v for m, v in real_flask.items() if v}
                pb = {m: v for m, v in predicted.box.items() if v}
                rb = {m: v for m, v in p0.brew.core.box.items() if v}
                if pf == rf and pb == rb:
                    exec_same += 1
                else:
                    mism.append({"kind": "exec", "picks": picks, "pred_flask": pf, "real_flask": rf,
                                 "pred_box": pb, "real_box": rb})
        # 3. 置き方の評価
        g = Game(erules, erecipes, [("a", "blue"), ("b", "red")], seed=trial)
        g.players[0].box = Counter(box)
        g.players[1].box = Counter(target)
        g.players[0].hand = list(hand)
        theirs_opts = card_options(g, 0)
        best_t = max(round(bot._plan_value(g, 0, [erecipes[c].sides(r) for c, r in o]), 6) for o in theirs_opts)
        mine_opts = sim.selection_options(rules, recipes, hand, box, target, {"A": 0, "B": 0, "C": 0})
        if len(theirs_opts) == len(mine_opts) and round(mine_opts[0].value, 6) >= best_t - 1e-6:
            sel_same += 1
        else:
            mism.append({"kind": "select", "n_bot": len(theirs_opts), "n_mine": len(mine_opts),
                         "best_bot": best_t, "best_mine": round(mine_opts[0].value, 6), "hand": hand, "box": box})
    print(f"1. 錬成の手順の点数が Bot 以上: {plan_same}/{args.n}")
    print(f"2. イブの手順をエンジンに打ち込んだ結果が見積もりと一致: {exec_same}/{exec_total}")
    print(f"3. 置き方の候補数が同じで最良点が Bot 以上: {sel_same}/{args.n}")
    for m in mism[:8]:
        print("  不一致:", json.dumps(m, ensure_ascii=False, default=list)[:400])
    return 0 if plan_same == args.n and exec_same == exec_total and sel_same == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
