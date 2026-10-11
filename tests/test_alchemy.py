"""Alchemy（ユーザ作のブラウザ対戦ゲーム）をイブが遊ぶ部分の決定論テスト（API・ブラウザ・ゲームのサーバー不要）。

守るもの:
- 錬成の計算（ゲームのエンジンの移植）と見積もり・自動の手が**合法**で、明らかな自爆をしない
- 判断役（LLM）の答えが崩れても・遅れても・合法でなくても、コードの手で締切までに打つ
- 出来事の文はサーバーの数だけから作り、「大きく動いた」時だけ話しかけの合図を出す
- **普段（idle）はルールを会話に入れない**・部屋に入るのは alchemy_join だけ・画面からは準備だけ
- 部屋・対戦中は画面認識の見る先がイブのゲーム画面になり、画面の変化では話しかけを起こさない

ゲームの本物のエンジンとの突き合わせは `tools/alchemy_sim_check.py`（ゲームのフォルダが要る）。
実行: $env:PYTHONIOENCODING="utf-8"; python tests\\test_alchemy.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.alchemy import autoplay, events, sim  # noqa: E402
from eve.alchemy.brain import Brain, parse_json, to_material  # noqa: E402
from eve.alchemy.capability import NAMES, register_alchemy_capabilities  # noqa: E402
from eve.alchemy.manager import IDLE, PLAYING, PREPARED, ROOM, AlchemyManager, AlchemySettings  # noqa: E402
from eve.alchemy.rules import DEFAULT_RULES, recipe_text, rules_brief, rules_full  # noqa: E402
from eve.alchemy.screen import GameScreenFeed  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


# ---------------------------------------------------------------- 試験用のカード（作り物・ゲームの実データではない）
RULES_D = dict(DEFAULT_RULES)
RECIPES_D = [
    {"id": "R1", "name": "赤の増殖", "number": 1, "rotatable": False, "skills": ["A"],
     "inputs": [["herb"]], "outputs": [["dragon"], ["herb"]]},
    {"id": "R2", "name": "赤の転化", "number": 2, "rotatable": True, "skills": ["A", "A"],
     "inputs": [["mineral"]], "outputs": [["dragon"]]},
    {"id": "R3", "name": "星の変成", "number": 3, "rotatable": False, "skills": ["B"],
     "inputs": [["dragon"]], "outputs": [["mineral"]]},
    {"id": "R4", "name": "小人の分解", "number": 4, "rotatable": False, "skills": ["C"],
     "inputs": [["homunculus"]], "outputs": [["dragon"], ["dragon"]]},
    {"id": "R5", "name": "石の大分解", "number": 5, "rotatable": False, "skills": [],
     "inputs": [["stone"]], "outputs": [["herb"], ["herb", "mineral"]]},
    {"id": "R6", "name": "井戸", "number": 6, "rotatable": False, "skills": ["B", "C"],
     "inputs": [], "outputs": [["herb"]]},
]
CATALOG = {"rules": RULES_D, "recipes": RECIPES_D}
RULES, RECIPES = sim.load_catalog(CATALOG)
MATS = RULES.materials


def box(**kw) -> dict:
    return {m: kw.get(m, 0) for m in MATS}


def view_for(*, phase="select", hand=None, my_box=None, their_box=None, tableau=None, pending=None,
             brew=None, rnd=1, skills=None, selection=None, deadline=True) -> dict:
    me = {"seat": 0, "name": "イブ", "box": my_box or box(dragon=5, herb=5, mineral=5, homunculus=1),
          "points": 0, "tableau": tableau or [], "skills": skills or {"A": 0, "B": 0, "C": 0}, "flask": {}}
    other = {"seat": 1, "name": "たかし", "box": their_box or box(dragon=8, herb=5, mineral=5, homunculus=1),
             "points": 0, "tableau": [], "skills": {"A": 0, "B": 0, "C": 0}, "flask": {}}
    return {
        "round": rnd, "phase": phase, "players": [me, other],
        "deadline": 1000.0 if deadline else None, "server_time": 960.0,
        "me": {"seat": 0, "hand": hand if hand is not None else ["R1", "R2", "R3", "R4"],
               "selection": selection, "skill_pending": pending or {"A": 0, "B": 0, "C": 0},
               "skill_choices": {"sealed": {}, "supply": []}, "brew": brew},
    }


def run(coro):
    return asyncio.run(coro)


# ================================================================ 1. 錬成の計算
print("\n[1] 錬成の計算（エンジンと同じ規則）")
core = sim.BrewCore(box=Counter(box(herb=2, mineral=1)))
p = core.take("herb", sim.BOX)
check("素材箱から取ると1減る", core.box["herb"] == 1 and p.source == sim.BOX)
core.produce(["dragon", "herb"], 1)
check("産物はフラスコに入る（送る素材）", core.flask() == Counter({"dragon": 1, "herb": 1}))
p2 = core.take("dragon", sim.PRODUCT)
check("産物から取ると段数を覚える", p2.depth == 1 and core.flask()["dragon"] == 0)
check("産物を使うと2連鎖", sim.chain_depth([p2]) == 2 and sim.bonus_count(2, RULES) == 1)
check("箱だけなら1段目（ボーナスなし）", sim.chain_depth([p]) == 1 and sim.bonus_count(1, RULES) == 0)

c2 = sim.BrewCore(box=Counter(box(herb=1)), products=Counter({("herb", 1): 1}))
check("埋め方の候補: 素材箱からと、フラスコの産物から（連鎖）の両方",
      sim.source_plans([("herb",)], c2) == [[("herb", sim.BOX)], [("herb", sim.PRODUCT)]])
check("足りなければ候補なし（作れない）", sim.source_plans([("stone",)], c2) == [])
c3 = sim.BrewCore(box=Counter(box()), supply=frozenset({"stone"}))
check("召喚した素材は箱に無くても使える", sim.source_plans([("stone",)], c3) == [[("stone", sim.SUPPLY)]])
check("2択の入力枠はある方で埋める",
      sim.source_plans([("dragon", "mineral")], sim.BrewCore(box=Counter(box(mineral=1)))) == [[("mineral", sim.BOX)]])
check("入力の埋め方の候補は全部の組み合わせ",
      len(sim.source_plans([("dragon", "mineral")], sim.BrewCore(box=Counter(box(dragon=1, mineral=1))))) == 2)
# 箱にも産物にもある時、最良の手は箱から取る（箱が減り、産物はそのまま送れる）
st_hp = sim.best_plan(RULES, [RECIPES["R1"].sides(False)], set(), c2, box())
check("最良の手は素材箱から取る（箱を減らして送る量を最大に）", st_hp and st_hp[0].plan == [("herb", sim.BOX)],
      f"{[s.plan for s in st_hp]}")

# 送り先の龍の鱗が満杯 → 龍の鱗を送ればあふれて点になる
start = sim.BrewCore(box=Counter(box(herb=3)))
sides = [RECIPES["R1"].sides(False)]
steps = sim.best_plan(RULES, sides, set(), start, box(dragon=9))
oc = sim.outcome(RULES, start, box(dragon=9), steps)
check("満杯の素材を送るとあふれ見込み1点", oc.points == 1 and oc.overflow == {"dragon": 1}, f"{oc}")
check("材料の分だけ自分の箱が減る", oc.drained == {"herb": 1})

# 2連鎖: R2（鉱物→龍の鱗）の産物の龍の鱗で R3（龍の鱗→鉱物）を作るとボーナス
start = sim.BrewCore(box=Counter(box(mineral=1)))
sides = [RECIPES["R2"].sides(False), RECIPES["R3"].sides(False)]
steps = sim.best_plan(RULES, sides, set(), start, box())
check("箱に無い材料はフラスコの産物で作る（連鎖）", [s.index for s in steps] == [0, 1] and steps[1].depth == 2,
      f"{[(s.index, s.depth) for s in steps]}")
check("2連鎖のボーナスが1個増える", len(steps[1].bonus) == 1)

# 全部試す探索は「並んだ順に作れるものを作る」素朴な手より悪くならない
# （ゲームの Bot＝貪欲法との比較は、本物のエンジンを使う tools/alchemy_sim_check.py で 1000場面確かめた）
def _naive(sd, st, t):
    core_n = st.clone()
    steps_n = []
    for i, (ins, outs) in enumerate(sd):
        plans_n = sim.source_plans(list(ins), core_n)
        if not plans_n:
            continue
        out_n = tuple(slot[0] for slot in outs)
        d, bn = sim._apply_step(core_n, RULES, t, plans_n[0], out_n)
        steps_n.append(sim.Step(0.0, i, plans_n[0], out_n, d, bn))
    return steps_n


rng = random.Random(5)
worse = 0
for _ in range(150):
    b = {m: rng.randint(0, RULES.capacity[m]) for m in MATS}
    t = {m: rng.randint(0, RULES.capacity[m]) for m in MATS}
    picks = [(c, RECIPES[c].rotatable and rng.random() < 0.5) for c in rng.sample(list(RECIPES), rng.randint(1, 3))]
    sd = [RECIPES[c].sides(r) for c, r in picks]
    st = sim.BrewCore(box=Counter(b))
    base = sim.score(st, Counter(b), t, RULES)
    nv = _naive(sd, st, t)
    best = sim.best_plan(RULES, sd, set(), st, t)
    ns = sim.outcome(RULES, st, t, nv).score if nv else base
    bs = sim.outcome(RULES, st, t, best).score if best else base
    worse += bs < max(ns, base) - 1e-9
check("全部試した手は素朴な手・何もしないより悪くならない（150場面）", worse == 0, f"{worse}")

plans = sim.brew_plans(RULES, [RECIPES["R1"].sides(False), RECIPES["R2"].sides(False)], set(),
                       sim.BrewCore(box=Counter(box(herb=2, mineral=2))), box(), limit=5)
sigs = [(tuple(sorted(o.sent.items())), tuple(sorted(o.drained.items())), o.max_depth) for _s, o in plans]
check("錬成の候補は結果の違うものだけ（順番を入れ替えただけの重複なし）", len(sigs) == len(set(sigs)))
check("錬成の候補の1件目はいちばん良い手（コードの手と同じ）",
      plans and abs(plans[0][1].score - sim.outcome(
          RULES, sim.BrewCore(box=Counter(box(herb=2, mineral=2))), box(),
          sim.best_plan(RULES, [RECIPES["R1"].sides(False), RECIPES["R2"].sides(False)], set(),
                        sim.BrewCore(box=Counter(box(herb=2, mineral=2))), box())).score) < 1e-9)

opts = sim.card_options(["R1", "R2"], RECIPES, 3)
check("置き方の候補: 逆OKのカードは向きも数える（1枚×向き + 2枚の組）",
      len(opts) == 1 + 2 + 2, f"{len(opts)}")
check("スキルの発動: 紋章が2つたまるごとに1回",
      sim.skill_triggers({"A": 1, "B": 0, "C": 0}, [("R1", False)], RECIPES, 2) == {"A": 1}
      and sim.skill_triggers({"A": 0}, [("R2", False)], RECIPES, 2) == {"A": 1}
      and sim.skill_triggers({"A": 0}, [("R1", False)], RECIPES, 2) == {})

need = Counter({"herb": 2})
purge = sim.purge_choice(RULES, box(herb=2, dragon=8), need)
check("浄化: 錬成で使う予定の素材は捨てない", "herb" not in purge and purge == ["dragon", "dragon"], f"{purge}")
check("浄化: 余りが無ければ何も捨てない", sim.purge_choice(RULES, box(herb=2), need) == [])
purge2 = sim.purge_choice(RULES, box(dragon=8, mineral=3), Counter(), sim.Strategy(consume=("mineral",)))
check("浄化: 作戦で減らしたい素材を優先", purge2[0] == "mineral", f"{purge2}")
rank = sim.seal_ranking(RULES, box(stone=3, dragon=2), Counter(), Counter())
check("送り込み: 送ればあふれる素材（満杯の賢者の石）を最優先", rank[0] == "stone", f"{rank}")
rank2 = sim.seal_ranking(RULES, box(), Counter(), Counter(), sim.Strategy(send=("herb",)))
check("送り込み: あふれない時は作戦で送りたい素材", rank2[0] == "herb", f"{rank2}")

# 作戦の重みで選ぶ手が変わる（赤を送りたい／青を送りたい）
hand = ["R1", "R3"]
red = sim.selection_options(RULES, RECIPES, hand, box(herb=3, dragon=3), box(), {}, sim.Strategy(send=("dragon",)))
blue = sim.selection_options(RULES, RECIPES, hand, box(herb=3, dragon=3), box(), {}, sim.Strategy(send=("mineral",)))
check("作戦「赤を多く送る」なら龍の鱗を送る手が先頭", red[0].outcome.sent.get("dragon", 0) >= 1, f"{red[0].outcome}")
check("作戦「青を多く送る」なら鉱物を送る手が先頭", blue[0].outcome.sent.get("mineral", 0) >= 1, f"{blue[0].outcome}")

core_v = sim.core_from_view({"products_by_depth": [["dragon", 1, 2]], "sealed": {"stone": 1}, "supply": ["herb"]},
                            box(herb=1))
check("state の錬成状態を読める", core_v.flask() == Counter({"dragon": 2, "stone": 1}) and "herb" in core_v.supply)

# ================================================================ 2. コードの手（自動プレイ・安全網）
print("\n[2] コードの手は合法")
v = view_for(hand=["R1", "R2", "R5", "R6"], my_box=box(herb=4, mineral=2))
picks = autoplay.choose_select(v, RULES, RECIPES)
check("置くカードは手札から1〜3枚", 1 <= len(picks) <= 3 and all(c in v["me"]["hand"] for c, _ in picks))
check("逆向きは逆OKのカードだけ", all((not r) or RECIPES[c].rotatable for c, r in picks))
check("作れないカード（箱に無い賢者の石が要る）は置かない", "R5" not in [c for c, _ in picks], f"{picks}")
v = view_for(phase="skill", tableau=[{"card": "R1", "rotated": False}], my_box=box(herb=2, dragon=8),
             pending={"A": 1, "B": 1, "C": 1})
uses = autoplay.choose_skills(v, RULES, RECIPES)
kinds = Counter(u["skill"] for u in uses)
check("発動した回数だけスキルを使う", kinds == Counter({"A": 1, "B": 1, "C": 1}), f"{uses}")
b_use = next(u for u in uses if u["skill"] == "B")
check("浄化は錬成で使う薬草を残す", "herb" not in b_use["materials"], f"{b_use}")
check("送り込み・召喚は素材1個", all(len(u["materials"]) == 1 for u in uses if u["skill"] in ("A", "C")))
v = view_for(phase="brew", tableau=[{"card": "R2", "rotated": False}, {"card": "R3", "rotated": False}],
             my_box=box(mineral=1),
             brew={"products_by_depth": [], "sealed": {}, "supply": [], "used": [], "done": False,
                   "target_box": box()})
st = autoplay.choose_brew(v, RULES, RECIPES)
check("錬成の手順は工房のレシピで連鎖まで組む", [s.index for s in st] == [0, 1], f"{[s.index for s in st]}")

# ================================================================ 3. 判断役（LLM）
print("\n[3] 判断役: 崩れても・遅れても合法な手で打つ")
check("JSON を前後の文やコードの囲みから取り出す",
      parse_json('はい\n```json\n{"choice": 2, "reason": "x"}\n```') == {"choice": 2, "reason": "x"})
check("JSON が無ければ None", parse_json("わかりません") is None)
check("素材はキー・呼び名・色で書いてよい",
      [to_material(x, MATS) for x in ("dragon", "龍の鱗", "赤", "青", "金", "なにか")]
      == ["dragon", "dragon", "dragon", "mineral", "stone", None])


class FakeRegistry:
    def __init__(self, replies=None, delay=0.0, error=None):
        self.replies = list(replies or [])
        self.delay = delay
        self.error = error
        self.calls = []

    async def complete(self, role, messages, **kw):
        self.calls.append((role, messages, kw))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        text = self.replies.pop(0) if self.replies else "{}"
        return {"choices": [{"message": {"content": text}}]}


v_sel = view_for(hand=["R1", "R2", "R3", "R4"], my_box=box(herb=3, mineral=2, homunculus=1))


async def _sel(reg, mode="action", budget=10.0):
    b = Brain(reg, mode)
    return b, await b.decide_select(v_sel, RULES, RULES_D, RECIPES, budget=budget)


_b, d = run(_sel(FakeRegistry(['{"choice": 2, "reason": "龍の鱗を送りたい"}'])))
check("action: 候補番号で選ぶ（LLM の手）", d.source == "llm" and d.detail["rank"] == 2 and d.reason == "龍の鱗を送りたい")
check("action: 選んだ手は手札から", all(c in v_sel["me"]["hand"] for c, _ in d.payload))
_b, d = run(_sel(FakeRegistry(['{"choice": 99}'])))
check("action: 候補に無い番号 → コードの手", d.source.startswith("fallback") and d.payload)
_b, d = run(_sel(FakeRegistry(["考え中…"])))
check("action: 形式が崩れた → コードの手", d.source.startswith("fallback") and d.payload)
_b, d = run(_sel(FakeRegistry(error=RuntimeError("down"))))
check("action: 呼び出し失敗 → コードの手", d.source.startswith("fallback") and d.payload)
_b, d = run(_sel(FakeRegistry(['{"choice": 1}'], delay=2.0), budget=0.6))
check("action: 締切の予算を超えたら待たずにコードの手", d.source.startswith("fallback") and "時間切れ" in d.source,
      d.source)
reg = FakeRegistry(['{"send": ["赤", "dragon", "鉱物", "金"], "consume": ["青"], "reason": "赤を溢れさせる"}'])
b, d = run(_sel(reg, mode="strategy"))
check("strategy: 作戦は素材のキーにそろえて2つまで（重複は1つ）",
      b.strategy.send == ("dragon", "mineral") and b.strategy.consume == ("mineral",), f"{b.strategy}")
check("strategy: 手はコードが作戦で決める（合法）", d.payload and all(c in v_sel["me"]["hand"] for c, _ in d.payload))
check("判断役の role は game", reg.calls and reg.calls[0][0] == "game")
check("判断役にルールを渡す", "フラスコの中身" in reg.calls[0][1][0]["content"])
check("判断役には手札と状況を渡す（サーバーの数）", "手札" in reg.calls[0][1][1]["content"]
      and "龍の鱗0/9" in reg.calls[0][1][1]["content"])


async def _strategy_keep():
    b = Brain(FakeRegistry(['{"send": ["dragon"], "consume": [], "reason": "a"}']), "strategy")
    await b.decide_strategy(v_sel, RULES, RULES_D, RECIPES, budget=5)
    b._registry = FakeRegistry(["崩れた"])
    d2 = await b.decide_strategy(v_sel, RULES, RULES_D, RECIPES, budget=5)
    return b, d2


b, d2 = run(_strategy_keep())
check("strategy: 作戦が決まらなければ前の作戦のまま", b.strategy.send == ("dragon",) and d2.source.startswith("fallback"))

v_sk = view_for(phase="skill", tableau=[{"card": "R1", "rotated": False}], my_box=box(herb=2, dragon=8),
                pending={"A": 1, "B": 1, "C": 0})


async def _skills(text):
    return await Brain(FakeRegistry([text]), "action").decide_skills(v_sk, RULES, RULES_D, RECIPES, budget=5)


d = run(_skills('{"uses": [{"skill": "A", "materials": ["stone"]}, {"skill": "B", "materials": ["dragon", "dragon"]}]}'))
check("スキル: 合法な答えはそのまま", d.source == "llm" and Counter(u["skill"] for u in d.payload) == Counter("AB"))
d = run(_skills('{"uses": [{"skill": "A", "materials": ["stone"]}, {"skill": "B", "materials": ["stone"]}]}'))
check("スキル: 箱に無い素材の浄化は捨てて、コードで補う",
      Counter(u["skill"] for u in d.payload) == Counter("AB")
      and all(m != "stone" for u in d.payload if u["skill"] == "B" for m in u["materials"]), f"{d.payload}")
d = run(_skills('{"uses": [{"skill": "C", "materials": ["herb"]}]}'))
check("スキル: 発動していないスキルは使わない・足りない分はコード",
      Counter(u["skill"] for u in d.payload) == Counter("AB") and "llm+code" in d.source, f"{d.source} {d.payload}")
v_need = view_for(phase="skill", tableau=[{"card": "R1", "rotated": False}], my_box=box(herb=1, dragon=8),
                  pending={"A": 0, "B": 1, "C": 0})
d = run(Brain(FakeRegistry(['{"uses": [{"skill": "B", "materials": ["herb"]}]}']), "action").decide_skills(
    v_need, RULES, RULES_D, RECIPES, budget=5))
check("スキル: 錬成で使う唯一の薬草（工房の赤の増殖の材料）は浄化させず、コードが余りから選ぶ",
      [u["materials"] for u in d.payload] == [["dragon", "dragon"]], f"{d.payload}")
v_spare = view_for(phase="skill", tableau=[{"card": "R1", "rotated": False}], my_box=box(herb=2, dragon=8),
                   pending={"A": 0, "B": 1, "C": 0})
d = run(Brain(FakeRegistry(['{"uses": [{"skill": "B", "materials": ["herb", "herb"]}]}']), "action").decide_skills(
    v_spare, RULES, RULES_D, RECIPES, budget=5))
check("スキル: 余りの薬草は浄化してよい（使う1個は残す）", [u["materials"] for u in d.payload] == [["herb"]],
      f"{d.payload}")

v_c2 = view_for(phase="skill", tableau=[{"card": "R1", "rotated": False}, {"card": "R5", "rotated": False}],
                my_box=box(herb=2, dragon=3), pending={"A": 0, "B": 0, "C": 2})
d = run(Brain(FakeRegistry(['{"uses": [{"skill": "C", "materials": ["stone"]}, {"skill": "C", "materials": ["stone"]}]}']),
              "action").decide_skills(v_c2, RULES, RULES_D, RECIPES, budget=5))
cs = [u["materials"][0] for u in d.payload if u["skill"] == "C"]
check("スキル: 同じ素材を2回召喚しない（2回目はコードが別の素材で補う）",
      len(cs) == 2 and len(set(cs)) == 2 and "stone" in cs, f"{d.payload}")
cs_code = [u["materials"][0] for u in autoplay.choose_skills(v_c2, RULES, RECIPES) if u["skill"] == "C"]
check("コードの手も同じ素材を2回召喚しない", len(cs_code) == 2 and len(set(cs_code)) == 2, f"{cs_code}")

v_br = view_for(phase="brew", tableau=[{"card": "R2", "rotated": False}, {"card": "R3", "rotated": False}],
                my_box=box(mineral=1, dragon=1),
                brew={"products_by_depth": [], "sealed": {}, "supply": [], "used": [], "done": False,
                      "target_box": box()})
d = run(Brain(FakeRegistry(['{"choice": 1, "reason": "連鎖"}']), "action").decide_brew(
    v_br, RULES, RULES_D, RECIPES, budget=5))
check("錬成: 候補から選んだ手順", d.source == "llm" and d.payload)

# ================================================================ 4. 出来事の文
print("\n[4] 出来事の文（サーバーの数だけ）と、大きく動いたか")
players = [{"seat": 0, "name": "イブ"}, {"seat": 1, "name": "たかし"}]
ev = {"type": "resolve", "round": 3, "transfers": [
    {"from": 0, "to": 1, "sent": {"dragon": 3}, "stored": {"dragon": 1}, "overflow": {"dragon": 2}, "points": 2,
     "total_points": 4},
    {"from": 1, "to": 0, "sent": {"herb": 1}, "stored": {"herb": 1}, "overflow": {}, "points": 0, "total_points": 1}],
    "executions": {"0": [{"depth": 1}, {"depth": 3}], "1": []}}
text, big, why = events.resolve_summary(ev, players, 0, 5)
check("送った数・あふれた数・点がそのまま入る", "龍の鱗3" in text and "龍の鱗2があふれて" in text and "+2点・計4点" in text, text)
check("点が入れば大きく動いた", big and "得点" in why)
check("イブの3連鎖も大きく動いた", "イブの大連鎖" in why)
check("あと1点を書く", "あと1点で勝ち: イブ" in text)
quiet = {"type": "resolve", "round": 1, "transfers": [
    {"from": 0, "to": 1, "sent": {"herb": 1}, "overflow": {}, "points": 0, "total_points": 0}], "executions": {}}
check("点も連鎖も無いラウンドは大きく動いていない（話しかけない）", events.resolve_summary(quiet, players, 0, 5)[1] is False)
end = {"type": "end", "round": 9, "reason": "points", "winners": [1],
       "ranking": [{"seat": 1, "points": 5, "box_total": 20}, {"seat": 0, "points": 3, "box_total": 12}]}
check("勝ち負けの文", "たかしの勝ち（イブの負け）" in events.end_text(end, players, 0))
rv = {"type": "reveal", "round": 2, "plays": [{"seat": 0, "cards": [{"card": "R2", "rotated": True}],
                                                "triggers": {"A": 1}}]}
check("置いたカードの文", "《赤の転化》（逆向き）" in events.reveal_text(rv, players, 0, RECIPES))
sk = {"type": "skills", "round": 2, "uses": [{"seat": 1, "skill": "A", "materials": ["stone"], "to": 0},
                                             {"seat": 0, "skill": "B", "skipped": True}]}
check("スキルの文（使わなかった分は書かない）", events.skills_text(sk, players, 0) == "2ラウンド目のスキル: たかしが送り込みで賢者の石をイブへ")

# ================================================================ 5. 流れ（準備・部屋・対戦・やめる）
print("\n[5] 流れ: 普段はルールを入れない・部屋に入るのは join だけ・画面はイブの画面")


class FakeClient:
    """ゲームのページの代わり。送られた操作に、サーバーと同じ形の返事（keyword / state）を返す。"""

    def __init__(self, mgr):
        self.mgr = mgr
        self.sent = []
        self.alive = True
        self.view = None

    async def start(self):
        pass

    async def close(self):
        self.alive = False

    def _later(self, msg):
        asyncio.get_running_loop().call_soon(self.mgr.on_message, msg)

    async def join_keyword(self, kw):
        self.sent.append({"type": "keyword.join", "keyword": kw})
        self._later({"type": "keyword", "keyword": kw.lower(), "playing": False,
                     "members": [{"name": "イブ", "me": True, "ready": False}]})
        return True

    async def set_ready(self, ready=True):
        self.sent.append({"type": "keyword.ready", "ready": ready})
        self._later({"type": "keyword", "keyword": "x", "playing": False,
                     "members": [{"name": "イブ", "me": True, "ready": ready}]})
        return True

    async def back_to_room(self):
        return False

    async def leave(self):
        self.sent.append({"type": "leave"})

    async def screenshot(self, quality=85):
        return b"jpeg"

    async def send(self, msg):
        self.sent.append(msg)
        v = json.loads(json.dumps(self.view))
        me = v["me"]
        t = msg["type"]
        if t == "play.cards":
            me["selection"] = msg["cards"]
        elif t == "skill.use":
            me["skill_pending"][msg["skill"]] -= 1
        elif t == "skill.skip":
            me["skill_pending"] = {"A": 0, "B": 0, "C": 0}
        elif t == "brew.choose":
            me["brew"].setdefault("pending", {})[str(msg["index"])] = {"slots": {}}
        elif t == "brew.place":
            pend = me["brew"].setdefault("pending", {}).setdefault(str(msg["index"]), {"slots": {}})
            pend["slots"][str(msg["slot"])] = {}
            card = v["players"][0]["tableau"][msg["index"]]
            ins, _outs = RECIPES[card["card"]].sides(card.get("rotated", False))
            if len(pend["slots"]) == len(ins):
                me["brew"]["used"].append(msg["index"])
        elif t == "brew.auto":
            me["brew"]["used"].append(msg["index"])
        elif t == "brew.done":
            me["brew"]["done"] = True
        self.view = v
        self._later({"type": "state", "state": v})
        return True


async def _flow():
    reacts, reports, screens = [], [], []

    async def hook(feed):
        screens.append(feed)

    t = [0.0]
    mgr = AlchemyManager(FakeRegistry(error=RuntimeError("no llm")),
                         settings=AlchemySettings(url="http://x/", keyword="AIバトル", mode="action",
                                                  prepare_ttl_sec=600),
                         react=reacts.append, report=lambda *a: reports.append(a), screen_hook=hook,
                         now_fn=lambda: t[0])
    mgr._client_factory = lambda: FakeClient(mgr)
    out = {}
    out["idle_voice"] = mgr.voice_block()
    out["idle_decider"] = mgr.decider_block()
    mgr.note_seen_on_screen()
    out["screen_state"] = mgr.state
    out["prepared_voice"] = mgr.voice_block()
    out["prepared_decider"] = mgr.decider_block()
    out["prepared_no_room"] = mgr.client is None
    t[0] += 601
    out["ttl"] = mgr.check_ttl()
    out["after_ttl"] = (mgr.state, mgr.voice_block())
    # 部屋に入る（会話で遊ぶと言われた）
    out["join_ack"] = mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state == ROOM:
            break
    await asyncio.sleep(0.05)
    out["room_state"] = mgr.state
    out["report"] = reports[-1] if reports else None
    out["room_voice"] = mgr.voice_block()
    out["screen_on"] = (mgr.owns_screen(), screens and isinstance(screens[-1], GameScreenFeed))
    out["note"] = mgr.screen_note()
    out["ready_sent"] = any(m.get("type") == "keyword.ready" and m.get("ready") for m in mgr.client.sent)
    # 相手が部屋に来た → 合図
    mgr.on_message({"type": "keyword", "keyword": "aiバトル", "playing": False,
                    "members": [{"name": "イブ", "me": True, "ready": True}, {"name": "たかし", "ready": False}]})
    out["react_join"] = len(reacts)
    # 対戦が始まる
    mgr.on_message({"type": "match", "room": "k-1", "mode": "keyword", "keyword": "aiバトル"})
    mgr.on_message({"type": "joined", "seat": 0, "catalog": CATALOG})
    v = view_for(hand=["R1", "R2", "R3", "R4"], my_box=box(herb=3, mineral=2, homunculus=1))
    mgr.client.view = v
    mgr.on_message({"type": "state", "state": v})
    out["playing"] = mgr.state
    out["react_start"] = len(reacts)
    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(m.get("type") == "play.cards" for m in mgr.client.sent):
            break
    await asyncio.sleep(0.05)
    sel = next((m for m in mgr.client.sent if m.get("type") == "play.cards"), None)
    out["select_sent"] = sel
    out["moves"] = list(mgr.moves)
    # スキル → 錬成
    picked = [c["card"] for c in sel["cards"]] if sel else ["R1"]
    tab = [{"card": c, "rotated": False} for c in picked]
    v2 = view_for(phase="skill", tableau=tab, my_box=box(herb=3, mineral=2, homunculus=1),
                  pending={"A": 1, "B": 0, "C": 0})
    mgr.client.view = v2
    mgr.on_message({"type": "state", "state": v2})
    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(m.get("type") == "skill.use" for m in mgr.client.sent):
            break
    await asyncio.sleep(0.05)
    v3 = view_for(phase="brew", tableau=tab, my_box=box(herb=3, mineral=2, homunculus=1),
                  brew={"products_by_depth": [], "sealed": {}, "supply": [], "used": [], "done": False,
                        "target_box": box(dragon=8)})
    mgr.client.view = v3
    mgr.on_message({"type": "state", "state": v3})
    for _ in range(200):
        await asyncio.sleep(0.01)
        if any(m.get("type") == "brew.done" for m in mgr.client.sent):
            break
    out["brew_msgs"] = [m["type"] for m in mgr.client.sent if m["type"].startswith("brew.")]
    # 静かなラウンドの送り合い → 合図なし / 点が入った → 合図
    n = len(reacts)
    mgr.on_message({"type": "events", "events": [{"type": "resolve", "round": 1, "transfers": [
        {"from": 0, "to": 1, "sent": {"dragon": 1}, "overflow": {}, "points": 0, "total_points": 0}],
        "executions": {}}]})
    out["react_quiet"] = len(reacts) - n
    mgr.on_message({"type": "events", "events": [{"type": "resolve", "round": 2, "transfers": [
        {"from": 0, "to": 1, "sent": {"dragon": 3}, "overflow": {"dragon": 2}, "points": 2, "total_points": 2}],
        "executions": {}}]})
    out["react_points"] = len(reacts) - n
    out["play_voice"] = mgr.voice_block()
    out["play_decider"] = mgr.decider_block()
    # 終わり → 部屋に戻る
    mgr.on_message({"type": "events", "events": [{"type": "end", "round": 2, "reason": "points", "winners": [0],
                                                   "ranking": [{"seat": 0, "points": 5, "box_total": 3},
                                                               {"seat": 1, "points": 0, "box_total": 9}]}]})
    out["end_state"] = mgr.state
    out["last_result"] = mgr.last_result
    out["react_end"] = reacts[-1] if reacts else ""
    # やめる
    mgr.tool_leave({})
    for _ in range(100):
        await asyncio.sleep(0.01)
        if mgr.client is None:
            break
    await asyncio.sleep(0.4)
    out["left"] = (mgr.state, mgr.client is None, mgr.owns_screen(), screens[-1] is None)
    out["after_leave_voice"] = mgr.voice_block()
    await mgr.stop()
    return out


o = run(_flow())
check("idle: 会話にルールを入れない（応答も発話判定も）", o["idle_voice"] is None and o["idle_decider"] is None)
check("画面に映った → 準備（ルールを入れる）", o["screen_state"] == PREPARED and "フラスコ" in (o["prepared_voice"] or ""))
check("画面に映っただけではゲームを始めない（ブラウザも開かない）", o["prepared_no_room"])
check("準備の段階では発話判定にゲームを入れない", o["prepared_decider"] is None)
check("しばらく話が無ければ準備を解く（ルールを外す）", o["ttl"] and o["after_ttl"] == (IDLE, None))
check("join はすぐ返し、結果は後で報告する", "準備を始めた" in o["join_ack"])
check("join: 合言葉の部屋に入って準備OK", o["room_state"] == ROOM and o["ready_sent"])
# ユーザ 2026-10-11「入室退出の報告はなくしていい」（受付の一言と重なっていた）。部屋の様子は会話の事実に入る
check("部屋に入れた時は報告しない", o["report"] is None, f"{o['report']}")
check("部屋の様子（合言葉・メンバー）は会話の事実に入る", "AIバトル" in (o["room_voice"] or "")
      and "イブ（準備OK）" in (o["room_voice"] or ""))
check("部屋に入ると画面認識はイブのゲーム画面を見る", o["screen_on"] == (True, True))
check("画面認識にはユーザの画面ではないと断る", o["note"] and "ユーザのPC画面ではなく" in o["note"])
check("相手が部屋に来たら合図", o["react_join"] == 1)
check("対戦が始まった", o["playing"] == PLAYING and o["react_start"] == 2)
check("えらぶ: 判断役が落ちていてもコードの手で置く", o["select_sent"] is not None and o["select_sent"]["cards"])
check("イブの手を会話用に覚える", o["moves"] and "を置いた" in o["moves"][0], f"{o['moves']}")
check("錬成: 作って最後に錬成おわり", o["brew_msgs"] and o["brew_msgs"][-1] == "brew.done", f"{o['brew_msgs']}")
check("静かなラウンドは合図を出さない", o["react_quiet"] == 0)
check("点が入ったラウンドは合図を出す", o["react_points"] == 1)
check("対戦中の # Alchemy に得点・箱・最近の出来事・正の記録の注記",
      o["play_voice"] and "得点:" in o["play_voice"] and "イブの素材箱" in o["play_voice"]
      and "最近の出来事" in o["play_voice"] and "正確な記録" in o["play_voice"], (o["play_voice"] or "")[:300])
check("対戦中の # Alchemy にイブの手札（イブにしか見えない）", "イブの手札" in (o["play_voice"] or ""))
check("発話判定に直前の大きな出来事と、細かい動きに反応しない注記",
      "直前の大きな出来事" in (o["play_decider"] or "") and "いちいち反応しない" in (o["play_decider"] or ""))
check("終わったら合言葉の部屋に戻る（もう一回は join）", o["end_state"] == ROOM and "イブの勝ち" in (o["last_result"] or ""))
check("勝ち負けで合図", "対戦終了" in o["react_end"])
check("やめる: 部屋を出て、画面認識はデスクトップへ", o["left"] == (PREPARED, True, False, True), f"{o['left']}")
check("やめた後もしばらくは結果を話せる（ルールと結果は残る）", "さっきの対戦" in (o["after_leave_voice"] or ""))


async def _join_fail():
    reports = []
    mgr = AlchemyManager(FakeRegistry(), settings=AlchemySettings(url="http://x/"), report=lambda *a: reports.append(a))

    class Broken(FakeClient):
        async def start(self):
            from eve.alchemy.client import ClientError
            raise ClientError("ゲームのページ（http://x/）を開けなかった: TimeoutError")

    mgr._client_factory = lambda: Broken(mgr)
    mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if reports:
            break
    await mgr.stop()
    return mgr, reports


mgr, reps = run(_join_fail())
check("部屋に入れなかったら理由つきで報告（できたふりをしない）",
      reps and reps[0][2] is False and reps[0][1].startswith("（") and "開けなかった" in reps[0][1], f"{reps}")
check("入れなかったら部屋の状態にしない", mgr.state == PREPARED and mgr.client is None)

async def _reconnect():
    reports = []
    screens = []

    async def hook(feed):
        screens.append(feed)

    mgr = AlchemyManager(FakeRegistry(), settings=AlchemySettings(), report=lambda *a: reports.append(a),
                         screen_hook=hook)
    mgr._client_factory = lambda: FakeClient(mgr)
    mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state == ROOM:
            break
    await asyncio.sleep(0.05)
    out = {"room": mgr.state}
    # つなぎ直し: ページが部屋に入り直すと、サーバーは準備OKを外す → 付け直す
    n = sum(1 for m in mgr.client.sent if m.get("type") == "keyword.ready")
    mgr.on_message({"type": "welcome", "token": "t", "resume": None})
    mgr.on_message({"type": "keyword", "keyword": "aiバトル", "playing": False,
                    "members": [{"name": "イブ", "me": True, "ready": False}]})
    await asyncio.sleep(0.05)
    out["reready"] = sum(1 for m in mgr.client.sent if m.get("type") == "keyword.ready") - n
    # 対戦中につなぎ直したら対戦が無かった → 外れたと報告して準備の状態へ
    mgr.on_message({"type": "match"})
    mgr.on_message({"type": "welcome", "token": "t", "resume": None})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if reports:
            break
    out["lost"] = (mgr.state, mgr.client is None, mgr.owns_screen(), reports[-1] if reports else None)
    await mgr.stop()
    return out


o2 = run(_reconnect())
check("つなぎ直しで準備OKが外れたら付け直す", o2["room"] == ROOM and o2["reready"] == 1, f"{o2}")
check("つなぎ直したら対戦が無かった → 外れたと報告・画面認識はデスクトップへ",
      o2["lost"][:3] == (PREPARED, True, False) and o2["lost"][3] and "外れた" in o2["lost"][3][1]
      and o2["lost"][3][2] is False, f"{o2['lost']}")



async def _wait_room():
    """部屋で1人で待つ: 5分で確かめる → 30秒返事が無ければやめると伝えて抜ける／返事があれば待ち直す（ユーザ 2026-10-11）。"""
    t = [0.0]
    said: list = []
    seq = [0]
    busy = [False]
    mgr = AlchemyManager(FakeRegistry(), settings=AlchemySettings(keyword="AIバトル", nudge_sec=300, giveup_sec=30),
                         now_fn=lambda: t[0], say=lambda c, r: said.append(c), user_seq=lambda: seq[0],
                         busy=lambda: busy[0])
    mgr._client_factory = lambda: FakeClient(mgr)
    mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state == ROOM:
            break
    await asyncio.sleep(0.05)
    out = {}
    mgr._wait_tick()
    t[0] = 299
    mgr._wait_tick()
    out["before5"] = list(said)
    out["wait_fact"] = mgr.voice_block()
    t[0] = 301
    mgr._wait_tick()
    out["nudge"] = list(said)
    busy[0] = True       # 催促を話している間は数えない
    t[0] = 310
    mgr._wait_tick()
    busy[0] = False
    t[0] = 335           # 話し終えて25秒
    mgr._wait_tick()
    out["not_yet"] = (len(said), mgr.state)
    t[0] = 341           # 話し終えて31秒・返事なし → 抜ける
    mgr._wait_tick()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state != ROOM:
            break
    out["gave_up"] = (len(said), mgr.state, mgr.client is None)
    out["giveup_text"] = said[-1] if said else ""
    # 返事があった場合: 待ち直す（抜けない）
    said.clear()
    t[0] = 1000
    mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state == ROOM:
            break
    await asyncio.sleep(0.05)
    mgr._wait_tick()
    t[0] = 1301
    mgr._wait_tick()
    seq[0] += 1          # ユーザが返事をした
    t[0] = 1400
    mgr._wait_tick()
    t[0] = 1500
    mgr._wait_tick()
    out["replied"] = (len(said), mgr.state)
    # 待ち直し2回目（もう一度確かめて返事あり）→ 次の5分では確かめずに抜ける（ユーザ「2回以上待ち直したら抜けていい」）
    t[0] = 1701
    mgr._wait_tick()
    out["second_fact"] = mgr.voice_block()
    seq[0] += 1
    t[0] = 1710
    mgr._wait_tick()
    t[0] = 2011
    mgr._wait_tick()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state != ROOM:
            break
    out["capped"] = (len(said), mgr.state, said[-1] if said else "")
    mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state == ROOM:
            break
    await asyncio.sleep(0.05)
    # 相手が部屋に来たら数えない
    mgr.on_message({"type": "keyword", "keyword": "aiバトル", "playing": False,
                    "members": [{"name": "イブ", "me": True, "ready": True}, {"name": "たかし", "ready": False}]})
    t[0] = 5000
    mgr._wait_tick()
    out["with_user"] = (len(said), mgr.waiting_sec())
    await mgr.stop()
    return out


ow = run(_wait_room())
check("部屋で5分たつまでは確かめない", ow["before5"] == [], f"{ow['before5']}")
check("待っている時間は会話の事実に入る", "イブだけが部屋で待って" in (ow["wait_fact"] or ""))
check("5分たったら確かめる（言い回しは応答AI・例文を渡さない）", len(ow["nudge"]) == 1 and "確かめる" in ow["nudge"][0]
      and "「" not in ow["nudge"][0].replace("「AIバトル」", ""), f"{ow['nudge']}")
check("催促を話し終えて30秒たつまでは抜けない", ow["not_yet"] == (1, ROOM), f"{ow['not_yet']}")
check("返事が無ければ、やめると伝えて部屋から抜ける", ow["gave_up"] == (2, PREPARED, True) and "抜ける" in ow["giveup_text"],
      f"{ow['gave_up']}")
check("返事があれば抜けずに待ち直す（どうするかは応答AI）", ow["replied"] == (1, ROOM), f"{ow['replied']}")
check("2回目の確かめには回数の事実が付く（関係ない返事への反応は応答AI）", "2回確かめた" in (ow["second_fact"] or ""),
      f"{ow['second_fact']}")
check("2回待ち直しても来なければ、確かめずにやめると伝えて抜ける",
      ow["capped"][0] == 3 and ow["capped"][1] == PREPARED and "抜ける" in ow["capped"][2], f"{ow['capped']}")
check("相手が部屋に来たら待ちの見張りは止まる", ow["with_user"] == (3, None), f"{ow['with_user']}")


async def _seats():
    """対戦中に相手が抜けた: 事実と合図 → イブ以外が全員抜けたら少し待ってイブも抜ける（ユーザ 2026-10-12）。"""
    t = [0.0]
    said, reacts = [], []
    mgr = AlchemyManager(FakeRegistry(), settings=AlchemySettings(keyword="AIバトル", all_left_grace_sec=10),
                         now_fn=lambda: t[0], say=lambda c, r: said.append(c), react=reacts.append)
    mgr._client_factory = lambda: FakeClient(mgr)
    mgr.tool_join({})
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state == ROOM:
            break
    mgr.on_message({"type": "match"})

    def view(conn, auto):
        v = view_for()
        v["players"][1]["connected"], v["players"][1]["autopilot"] = conn, auto
        return v
    out = {}
    mgr.on_message({"type": "state", "state": view(True, False)})
    out["facts_in"] = mgr.voice_block()
    mgr.on_message({"type": "state", "state": view(False, True)})
    out["react"] = reacts[-1] if reacts else ""
    out["facts_left"] = mgr.voice_block()
    t[0] = 5
    mgr._left_tick()
    out["before_grace"] = mgr.state
    t[0] = 11
    mgr._left_tick()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if mgr.state != PLAYING:
            break
    out["after"] = (mgr.state, said[-1] if said else "")
    # 戻ってきたら抜けない
    mgr2 = AlchemyManager(FakeRegistry(), settings=AlchemySettings(all_left_grace_sec=10), now_fn=lambda: t[0],
                          react=reacts.append)
    mgr2.state = PLAYING
    mgr2.on_message({"type": "state", "state": view(True, False)})
    t[0] = 100
    mgr2.on_message({"type": "state", "state": view(False, True)})
    t[0] = 105
    mgr2.on_message({"type": "state", "state": view(True, False)})
    out["back_react"] = reacts[-1]
    t[0] = 200
    mgr2._left_tick()
    out["back_state"] = mgr2.state
    await mgr.stop()
    return out


os_ = run(_seats())
check("誰が参加しているかを会話の事実に入れる", "参加者: イブ（自分）・" in (os_["facts_in"] or "") and "（参加中）" in os_["facts_in"],
      os_["facts_in"][:300] if os_["facts_in"] else "")
check("相手が抜けたら合図し、事実に出す", "抜けた" in os_["react"] and "抜けた（自動操作が代わりに打っている）" in (os_["facts_left"] or ""),
      os_["react"])
check("全員抜けても、つなぎ直しを待つ間は抜けない", os_["before_grace"] == PLAYING)
check("イブ以外が全員抜けたら、イブも抜けてそう伝える", os_["after"][0] == PREPARED and "抜けた" in os_["after"][1], f"{os_['after']}")
check("戻ってきたら合図し、抜けない", "戻ってきた" in os_["back_react"] and os_["back_state"] == PLAYING, f"{os_['back_react']}")


async def _reanswer():
    """ルール準備だけを呼んで本文ゼロ → 一言は言わず、ルールを入れて同じ発話への返事を作り直す。"""
    from eve.capability.registry import CapabilityRegistry
    from eve.config import Config
    from eve.pipeline.audio_play_queue import AudioPlayQueue
    from eve.pipeline.stimulus import Stimulus, StimulusKind
    from eve.response.function_dispatcher import FunctionDispatcher
    from eve.response.orchestrator import ResponseOrchestrator

    async def noop_play(a):
        pass

    async def tts(s):
        return b"x"

    reg = CapabilityRegistry()
    mgr = AlchemyManager(FakeRegistry(), settings=AlchemySettings(keyword="AIバトル"))
    register_alchemy_capabilities(reg, mgr)
    disp = FunctionDispatcher(registry=reg, queue=asyncio.Queue())
    seen: list = []

    def tc(name, reply):
        return {"id": "c1", "function": {"name": name, "arguments": json.dumps({"reply": reply}, ensure_ascii=False)}}

    async def stream_fn(messages, *, tools=None, tool_sink=None):
        sysmsg = "\n".join(m["content"] for m in messages if m["role"] == "system")
        seen.append(("# Alchemy" in sysmsg, sorted(t["function"]["name"] for t in tools or [])))
        if "# Alchemy" in sysmsg:
            yield "先に5点取った方が勝ちですよ。"
        elif tool_sink is not None:
            tool_sink.append(tc("alchemy_prepare", "ちょっと待ってくださいね。"))

    old = Config.CALLFUNCTION_ENABLED
    Config.CALLFUNCTION_ENABLED = True
    try:
        audio = AudioPlayQueue(play_fn=noop_play)
        orch = ResponseOrchestrator(audio, stream_fn, tts, dispatcher=disp, game_provider=mgr.voice_block)
        w = asyncio.create_task(audio.play_worker())
        await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "Alchemyのルールってどんなのだっけ"))
        w.cancel()
    finally:
        Config.CALLFUNCTION_ENABLED = old
    return orch.last_response, seen, mgr.state


resp, seen_r, st_r = run(_reanswer())
check("ルールを聞かれたら、前置きで止まらずルールを入れて答え直す", resp == "先に5点取った方が勝ちですよ。" and st_r == PREPARED,
      f"{resp} / {seen_r}")
check("作り直しの時はルール準備を出さない（もう入っている・2回目は1回だけ）",
      len(seen_r) == 2 and seen_r[0][0] is False and seen_r[1][0] is True and "alchemy_prepare" not in seen_r[1][1]
      and "alchemy_join" in seen_r[1][1], f"{seen_r}")

m2 = AlchemyManager(FakeRegistry(), settings=AlchemySettings())
check("やっていないのにやめる → 断り（報告される）", m2.tool_leave({}).startswith("（"))
m2.prepare("tool")
check("準備だけの時にやめる → idle に戻す", m2.tool_leave({}) == "Alchemy の準備をやめた" and m2.state == IDLE)
m2.state = PLAYING
check("対戦中の join は断る", m2.tool_join({}).startswith("（"))
m2.state = IDLE
v_bud = view_for()
m2.view = v_bud
m2._view_mono = m2._now()
check("締切の予算 = 残り時間 − 余裕", abs(m2._budget("select", v_bud) - (40 - 6)) < 0.5, f"{m2._budget('select', v_bud)}")
check("締切が無い時はルールの制限時間から", m2._budget("brew", view_for(deadline=False)) >= 150)
check("打つべき段階の判定", m2._needed(view_for()) == "select"
      and m2._needed(view_for(selection=[{"card": "R1"}])) is None
      and m2._needed(view_for(phase="skill", pending={"A": 1})) == "skill"
      and m2._needed(view_for(phase="brew", brew={"done": False})) == "brew"
      and m2._needed(view_for(phase="brew", brew={"done": True})) is None)

# ================================================================ 6. 機能（応答AI）
print("\n[6] 応答AI の機能")
from eve.capability.registry import CapabilityRegistry  # noqa: E402

reg = CapabilityRegistry()
m3 = AlchemyManager(FakeRegistry(), settings=AlchemySettings(keyword="AIバトル"))
register_alchemy_capabilities(reg, m3)
schemas = {s["function"]["name"]: s for s in reg.tool_schemas()}
check("3つとも応答AI に出す", all(n in schemas for n in NAMES))
check("タスク担当には渡さない", not any(s["function"]["name"] in NAMES for s in reg.agent_tool_schemas()))
check("今すぐ返す一言（reply）は必須", all("reply" in schemas[n]["function"]["parameters"]["required"] for n in NAMES))
check("join の説明に合言葉", "AIバトル" in schemas["alchemy_join"]["function"]["description"])
# ユーザ 2026-10-11「例を入れるとその言い回しのフォーマットだらけになる」: 説明に言い回しの例（「」や例示の括弧）を書かない
check("説明にユーザの言い回しの例が無い",
      not any(w in schemas[n]["function"]["description"]
              for n in NAMES for w in ("遊ぼう", "やろう", "やめよう", "終わりにしよう", "かも等", "例:", "例：")),
      " / ".join(schemas[n]["function"]["description"] for n in NAMES))
# ユーザ 2026-10-11「この報告が単純に要らない」: 報告ターンで結果の文を毎回同じ型で言い直していた
check("prepare の結果は報告しない（ルールは次のターンから会話に入る）", not reg.report_result("alchemy_prepare"))
check("join・leave の受付は報告しない（結果は後で届く・reply が受付の一言）",
      not reg.report_result("alchemy_join") and not reg.report_result("alchemy_leave"))
check("prepare を呼ぶと準備に入る", reg.execute("alchemy_prepare", {"reply": "x"}) and m3.state == PREPARED
      and m3.voice_block() is not None)
check("「外の世界に手を出す手段」の一覧には出さない（会話の機能）",
      not any("Alchemy" in a for a in reg.outward_actions()))

# ================================================================ 7. ルールの文
print("\n[7] ルールの文（数は catalog から）")
custom = dict(RULES_D, capacity=dict(RULES_D["capacity"], stone=5), win_points=7)
check("勝ち点は catalog の値", "7点で勝ち" in rules_brief(custom) and "7点以上で勝ち" in rules_full(custom))
check("上限は catalog の値", "賢者の石（金）は5個まで" in rules_brief(custom))
check("2人なら右隣も左隣も同じ相手", "同じ相手" in rules_brief() and "同じ相手" in rules_full())
check("レシピの説明（逆向きは入出力が入れ替わる）",
      recipe_text(RECIPES_D[1], rotated=True).endswith("龍の鱗 → 鉱物［紋章 送り込み・送り込み］"),
      recipe_text(RECIPES_D[1], rotated=True))

# ================================================================ 8. 応答・発話判定・画面認識への組み込み
print("\n[8] 応答・発話判定・画面認識への組み込み")
from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.speech.decider import build_decide_messages, should_speak, SpeechDecision  # noqa: E402
from eve.vlm.narrator import ALCHEMY_TAG, GAME_SCREEN_PREFIX, build_messages, make_narrate_fn  # noqa: E402
from eve.vlm.parser import parse_vision  # noqa: E402
from eve.vlm.types import Frame  # noqa: E402

ca = ContextAssembler(system_prompt="S")
sys_no = ca.assemble(user_text="やあ")[0]["content"]
sys_yes = ca.assemble(user_text="やあ", game="# Alchemy（テスト）\n事実")[0]["content"]
check("応答: game が無ければ何も足さない", "Alchemy" not in sys_no)
check("応答: game があれば system に入る", "# Alchemy（テスト）" in sys_yes)
dm_no = build_decide_messages(surprise=10, silence_seconds=5, recent_turns=[], topic_seeds=[])
dm_yes = build_decide_messages(surprise=10, silence_seconds=5, recent_turns=[], topic_seeds=[],
                               game="# 今イブが遊んでいるゲーム（Alchemy）\n得点")
check("発話判定: 普段は入力が変わらない", "Alchemy" not in dm_no[1]["content"])
check("発話判定: 部屋・対戦中はゲームの事実が入る", "# 今イブが遊んでいるゲーム" in dm_yes[1]["content"])


async def _speak():
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return SpeechDecision(False, "x")

    await should_speak(surprise=1, silence_seconds=1, recent_turns=[], topic_seeds=[], decide_fn=fake)
    a = "game" in seen
    seen.clear()
    await should_speak(surprise=1, silence_seconds=1, recent_turns=[], topic_seeds=[], decide_fn=fake, game="g")
    return a, seen.get("game")


a, g = run(_speak())
check("should_speak: game は渡された時だけ転送（受けない古い decide_fn を壊さない）", a is False and g == "g")

r = parse_vision("narration: 画面\nnotable: no\nvisible: yes\nalchemy: yes")
check("画面認識: alchemy タグを読む", r.alchemy is True)
check("画面認識: タグが無ければ映っていない扱い", parse_vision("narration: 画面\nvisible: yes").alchemy is False)
fr = [Frame(0, 0.0, "AAAA")]
check("画面認識: Alchemy の検出は有効な時だけ足す",
      ALCHEMY_TAG not in build_messages(fr)[0]["content"]
      and ALCHEMY_TAG in build_messages(fr, detect_alchemy=True)[0]["content"])
gm = build_messages(fr, screen_note="ユーザのPC画面ではなくイブの画面", detect_alchemy=True)
check("画面認識: イブのゲーム画面の時は断り書きを入れ、検出の1行は付けない",
      "イブの画面" in gm[1]["content"][0]["text"] and ALCHEMY_TAG not in gm[0]["content"])


class _VReg:
    async def complete(self, role, messages, **kw):
        return {"choices": [{"message": {"content": "narration: 手札が4枚\nnotable: yes\nvisible: yes"}}]}


class _Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class _VReg2:
    async def complete(self, role, messages, **kw):
        return _Resp("narration: 手札が4枚\nnotable: yes\nvisible: yes")


res = run(make_narrate_fn(_VReg2(), screen_note_fn=lambda: "イブの画面")(fr))
check("画面認識: イブのゲーム画面の実況には頭に断りを付ける", res.narration.startswith(GAME_SCREEN_PREFIX), res.narration)
res2 = run(make_narrate_fn(_VReg2(), screen_note_fn=lambda: None)(fr))
check("画面認識: 普段の実況はそのまま", res2.narration == "手札が4枚")

from eve.vlm.state import VisionState  # noqa: E402
from eve.vlm.types import VisionResult  # noqa: E402
from eve.vlm.worker import VlmWorker  # noqa: E402


class _Pred:
    def note_vlm_surprise(self, v):
        pass


hits = []
w = VlmWorker(vision_state=VisionState(), prediction_state=_Pred(), narrate_fn=None,
              on_alchemy_seen=lambda: hits.append(1))
w._apply(VisionResult(narration="ブラウザに Alchemy のタイトル", alchemy=True))
w._apply(VisionResult(narration="ブラウザに Alchemy のタイトル", alchemy=True))  # 同じ実況（重複）でも数える
w._apply(VisionResult(narration="メモ帳", alchemy=False))
check("画面認識: Alchemy が映っていたら準備の合図（重複の実況でも）", len(hits) == 2)

feed = GameScreenFeed(client=None)
img = feed.grab()
check("ゲーム画面がまだ無い時は黒（画面認識は中身を作らない）", img.shape == (576, 1024, 3) and int(img.max()) == 0)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ok, buf = cv2.imencode(".jpg", np.full((10, 20, 3), 200, dtype=np.uint8))
feed._latest = buf.tobytes()
check("ゲーム画面の写しを画像にして渡す", feed.grab().shape == (10, 20, 3))

# ================================================================ 9. VoiceLoop の配線（ALCHEMY_ENABLED=1）
print("\n[9] VoiceLoop の配線")
from eve.config import Config  # noqa: E402

_saved = (Config.ALCHEMY_ENABLED, Config.CALLFUNCTION_ENABLED)
Config.ALCHEMY_ENABLED = True
Config.CALLFUNCTION_ENABLED = True
try:
    from eve.voice_loop import VoiceLoop  # noqa: E402

    vl = VoiceLoop()
    check("Alchemy の流れを作る", vl.alchemy is not None)
    check("応答AI に3つの機能", all(vl.capabilities.has(n) for n in NAMES))
    check("UI のタスク ON/OFF の束には入れない", not any(n in vl._task_cap_names for n in NAMES))
    check("応答に # Alchemy を渡す配線（普段は None）", vl.orchestrator._game_provider() is None)
    vl.alchemy.prepare("test")
    check("準備すると応答に # Alchemy", "# Alchemy" in (vl.orchestrator._game_provider() or ""))
    check("発話判定は部屋・対戦中だけ", vl.speech_decider._game_provider() is None)
    vl.alchemy._screen_on = True
    check("イブのゲーム画面を見ている間は、画面の変化で話しかけを起こさない", vl._vision_can_speak() is False)
    check("ゲーム画面の断り書きを画面認識へ", "ユーザのPC画面ではなく" in (vl.alchemy.screen_note() or ""))
    vl.alchemy._screen_on = False

    async def _swap():
        feed = GameScreenFeed(client=None)
        await vl._alchemy_screen(feed)
        a = vl.vlm_capture is not vl._desktop_capture and vl.vlm_capture._grab_fn == feed.grab
        await vl._alchemy_screen(None)
        return a, vl.vlm_capture is vl._desktop_capture

    a, b2 = run(_swap())
    check("対戦中は画面認識の写し先をイブのゲーム画面へ差し替え、終わったらデスクトップへ戻す", a and b2)

    from eve.pipeline.stimulus import Stimulus as _Stim, StimulusKind as _SK  # noqa: E402

    _s = _Stim(_SK.USER_UTTERANCE, "やあ")
    if vl.capabilities.has("delegate_task"):
        _sys_on = vl.orchestrator._build_messages(_s)[0]["content"]
        check("タスクがある時は委譲の使い方を出す", "delegate_task" in _sys_on)
        _removed = vl.capabilities.unregister(*vl._task_cap_names)  # UI でタスクを切った状態
        _sys_off = vl.orchestrator._build_messages(_s)[0]["content"]
        check("タスクを切っても Alchemy の機能は渡る", vl.orchestrator._tools_enabled_for(_s))
        check("タスクを切ったら委譲の使い方は出さない（無い delegate_task を使うつもりで喋らない）",
              "delegate_task" not in _sys_off and "# 実際に実行できる手段" not in _sys_off)
    else:
        check("（この .env ではタスクが無効なので委譲のブロックの試験は飛ばす）", True)
finally:
    Config.ALCHEMY_ENABLED, Config.CALLFUNCTION_ENABLED = _saved

Config.ALCHEMY_ENABLED = False
vl2 = VoiceLoop()
check("ALCHEMY_ENABLED=0 なら何も作らない・機能も出さない",
      vl2.alchemy is None and not any(vl2.capabilities.has(n) for n in NAMES))
Config.ALCHEMY_ENABLED = _saved[0]

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
