"""Alchemy の判断役（LLM）。遊ばせ方はユーザ案の2つを `ALCHEMY_PLAY_MODE` で切り替える。

- `action`（案1「構造化で1行動ごとに何をさせるか決めさせる」）: えらぶ・スキル・錬成のたびに、コードが作った
  **合法な候補（見積もりつき）**を見せて JSON で選ばせる。
- `strategy`（案2「戦略的な部分だけ決めさせてあとはその戦略に合わせ自動処理」）: ラウンドの初めに
  「多く送りたい素材・減らしたい素材」だけを JSON で決めさせ、手はコード（`autoplay.py`）がその重みで決める。

**ハルシネーションを手に持ち込まない**: LLM には画面ではなくサーバーの state から作った事実を渡し、
答えは候補番号・素材のキーだけを受け付ける（合法かはコードが確かめる）。時間内に答えない・崩れた・
合法でない時はコードの手で打つ（締切までに必ず合法手を出す）。どちらで打ったかは `Decision.source` に残す。

reason（狙い）はイブが後で「何を狙ったか」を話すための材料（会話の `# Alchemy` に入る）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from . import autoplay, sim
from .rules import (
    MAT_COLOR_JA, MAT_JA, PHASE_JA, SKILL_DESC, SKILL_JA, counts_text, mat, mat_c, recipe_text, rules_full,
    sides_text,
)
from .sim import NO_STRATEGY, Strategy

logger = logging.getLogger(__name__)

MODES = ("action", "strategy")

BRAIN_SYSTEM = (
    "あなたはAI VTuber「イブ」の頭の中で、ゲーム「Alchemy」の手を決める役です。イブ（あなた）はユーザたちと対戦しています。\n"
    "{rules}\n"
    "## 決め方\n"
    "・状況と候補の数字は、サーバーの情報からコードが計算した事実と見積もり。書いていないことを推測で足さない。\n"
    "・あふれ見込みは目安（送り先も同じラウンドに錬成で自分の箱を減らすので、実際は少なくなることがある）。\n"
    "・明らかに損な手（作れないカードばかり置く・錬成で使う素材を捨てる等）は選ばない。\n"
    "・出力は指示された JSON だけ。reason は、イブが後で「何を狙ったか」を話せるような短い日本語"
    "（手の説明の繰り返しではなく狙い。1文）。"
)

MAT_KEYS_NOTE = "素材の書き方: " + " / ".join(f"{k}={MAT_JA[k]}（{MAT_COLOR_JA[k]}）" for k in MAT_JA)


@dataclass
class Decision:
    kind: str                 # select / skill / brew / strategy
    payload: Any              # picks / uses / steps / Strategy
    reason: str = ""
    source: str = "code"      # llm / code / fallback:<理由>
    latency: float = 0.0
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- 状況の文
def _player_line(p: dict, rules: sim.Rules, me_seat: int) -> str:
    who = "あなた（イブ）" if p.get("seat") == me_seat else p.get("name", "?")
    extra = ""
    if p.get("bot"):
        extra = "（Bot）"
    elif p.get("autopilot"):
        extra = "（今は自動操作）"
    return (f"{who}{extra}: {p.get('points', 0)}点 / 素材箱 "
            f"{counts_text(p.get('box') or {}, rules.materials, capacity=rules.capacity)}")


def situation_text(view: dict, rules: sim.Rules, recipes: dict, *, recent: Optional[list] = None,
                   strategy: Optional[Strategy] = None) -> str:
    """判断役に渡す今の状況（サーバーの state から作る事実だけ）。"""
    me_seat = (view.get("me") or {}).get("seat", 0)
    players = view.get("players") or []
    n = len(players)
    lines = [f"# 今の状況（{view.get('round', '?')}ラウンド目／最大{rules.max_rounds}・"
             f"{PHASE_JA.get(view.get('phase', ''), view.get('phase', ''))}）",
             f"{rules.win_points}点以上で勝ち。"]
    for p in players:
        lines.append("・" + _player_line(p, rules, me_seat))
    if n >= 2:
        right = players[(me_seat - 1) % n].get("name", "?")
        left = players[(me_seat + 1) % n].get("name", "?")
        if n == 2:
            lines.append(f"送り先も送り元も {right}（2人対戦なので、お互いに送り合う）。")
        else:
            lines.append(f"送り先（右隣）: {right} / 送り元（左隣・あなたの箱に送ってくる）: {left}")
    me = players[me_seat] if me_seat < n else {}
    skills = me.get("skills") or {}
    lines.append("あなたの紋章のたまり具合: " + " ".join(f"{SKILL_JA[k]}{skills.get(k, 0)}" for k in ("A", "B", "C"))
                 + f"（同じ種類が{rules.skill_step}つたまるごとに1回発動）")
    if strategy is not None and not strategy.is_empty():
        lines.append("今の作戦: " + strategy_text(strategy))
    if recent:
        lines.append("最近の出来事:")
        lines += [f"- {r}" for r in recent[-6:]]
    return "\n".join(lines)


def strategy_text(st: Strategy) -> str:
    parts = []
    if st.send:
        parts.append("多く送る＝" + "・".join(mat_c(m) for m in st.send))
    if st.consume:
        parts.append("減らす＝" + "・".join(mat_c(m) for m in st.consume))
    text = " / ".join(parts) or "特になし（見積もりのまま）"
    return text


def hand_text(view: dict, recipes: dict) -> str:
    hand = (view.get("me") or {}).get("hand") or []
    rows = [f"{i + 1}. " + recipe_text(_cat(recipes[c]), rotatable_note=True) for i, c in enumerate(hand) if c in recipes]
    return "# 手札（あなたにしか見えない）\n" + ("\n".join(rows) or "（なし）")


def _cat(r: sim.Recipe) -> dict:
    return {"id": r.id, "name": r.name, "number": r.number, "rotatable": r.rotatable, "skills": list(r.skills),
            "inputs": [list(s) for s in r.inputs], "outputs": [list(s) for s in r.outputs]}


def _outcome_text(oc: sim.Outcome) -> str:
    sent = counts_text(oc.sent, zero=False) if oc.sent else "なし"
    over = ("あふれ見込み " + "・".join(f"{mat(m)}{k}" for m, k in oc.overflow.items()) + f"（{oc.points}点）"
            if oc.overflow else "あふれ見込み なし")
    drained = counts_text(oc.drained, zero=False) if oc.drained else "なし"
    chain = f" / 最大{oc.max_depth}連鎖" if oc.max_depth >= 2 else ""
    return f"送る見込み {sent} / {over} / 箱から減る {drained}{chain}"


def steps_text(steps: list, sides: list, names: list) -> str:
    """錬成の手順を「《名》箱の薬草→龍の鱗＋薬草（連鎖2）」の並びに。"""
    src_ja = {sim.BOX: "箱", sim.PRODUCT: "フラスコ", sim.SUPPLY: "召喚"}
    rows = []
    for s in steps:
        ins = "＋".join(f"{src_ja.get(src, src)}の{mat(m)}" for m, src in s.plan) or "材料なし"
        outs = "＋".join(mat(m) for m in s.outputs)
        if s.bonus:
            outs += "＋ボーナス" + "・".join(mat(m) for m in s.bonus)
        chain = f"（{s.depth}連鎖）" if s.depth >= 2 else ""
        rows.append(f"《{names[s.index]}》{ins}→{outs}{chain}")
    return " → ".join(rows) or "（何も作らない）"


# ---------------------------------------------------------------------------- JSON の読み取り
_JSON = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> Optional[dict]:
    """LLM の出力から最初の JSON オブジェクトを取り出す（崩れていたら None・raise しない）。"""
    if not text:
        return None
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except (ValueError, TypeError):
        pass
    m = _JSON.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except (ValueError, TypeError):
        return None


_MAT_ALIASES: dict = {}
for _k, _ja in MAT_JA.items():
    _MAT_ALIASES[_k] = _k
    _MAT_ALIASES[_ja] = _k
    _MAT_ALIASES[MAT_COLOR_JA[_k]] = _k
_MAT_ALIASES.update({"鱗": "dragon", "龍": "dragon", "石": "stone", "金色": "stone", "黄": "stone", "青色": "mineral",
                     "赤色": "dragon", "緑色": "herb", "紫色": "homunculus"})


def to_material(x: Any, materials: tuple) -> Optional[str]:
    """素材の書き方の揺れ（キー・呼び名・色）をキーにそろえる。知らない語は None。"""
    if not isinstance(x, str):
        return None
    k = x.strip().lower()
    m = _MAT_ALIASES.get(k) or _MAT_ALIASES.get(x.strip())
    return m if m in materials else None


def _materials(xs: Any, materials: tuple) -> list:
    if isinstance(xs, str):
        xs = [xs]
    if not isinstance(xs, list):
        return []
    out = []
    for x in xs:
        m = to_material(x, materials)
        if m is not None:
            out.append(m)
    return out


def _content(resp: Any) -> str:
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""
    return resp if isinstance(resp, str) else ""


# ---------------------------------------------------------------------------- 判断役
class Brain:
    def __init__(self, registry, mode: str = "action", *, role: str = "game", reasoning_effort: str = "",
                 max_wait_sec: float = 20.0) -> None:
        self.mode = mode if mode in MODES else "action"
        self._registry = registry
        self._role = role
        self._effort = (reasoning_effort or "").strip()
        self._max_wait = max_wait_sec
        self.strategy: Strategy = NO_STRATEGY

    # --- LLM 呼び出し ---
    async def _ask(self, rules_d: dict, user: str, budget: float) -> tuple:
        """(dict か None, 秒, 失敗理由)。budget 秒で打ち切る（締切を守るのが最優先）。"""
        wait = max(0.5, min(self._max_wait, budget))
        kw: dict = {}
        if self._effort:
            kw = {"reasoning_effort": self._effort, "allowed_openai_params": ["reasoning_effort"]}
        messages = [{"role": "system", "content": BRAIN_SYSTEM.format(rules=rules_full(rules_d))},
                    {"role": "user", "content": user}]
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(self._registry.complete(self._role, messages, **kw), timeout=wait)
        except asyncio.TimeoutError:
            return None, time.monotonic() - t0, f"時間切れ{wait:.0f}秒"
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001（判断役の失敗でゲームを止めない）
            logger.warning("🎮 判断役の呼び出しに失敗（コードの手で打つ）: %s", e)
            return None, time.monotonic() - t0, f"呼び出し失敗 {type(e).__name__}"
        d = parse_json(_content(resp))
        if d is None:
            return None, time.monotonic() - t0, "形式が崩れた"
        return d, time.monotonic() - t0, ""

    # --- えらぶ ---
    async def decide_select(self, view: dict, rules: sim.Rules, rules_d: dict, recipes: dict, *,
                            budget: float, recent: Optional[list] = None) -> Decision:
        """えらぶ。strategy は作戦を決めてから重みで、action は候補から選ばせる。"""
        if self.mode == "strategy":
            st = await self.decide_strategy(view, rules, rules_d, recipes, budget=budget, recent=recent)
            options = await asyncio.to_thread(autoplay.select_options, view, rules, recipes, self.strategy)
            picks = options[0].picks if options else autoplay.choose_select(view, rules, recipes, self.strategy)
            return Decision("select", picks, st.reason, st.source, st.latency,
                            {"strategy": self.strategy, "option": options[0] if options else None})
        options = await asyncio.to_thread(autoplay.select_options, view, rules, recipes, NO_STRATEGY)
        if not options:
            return Decision("select", autoplay.choose_select(view, rules, recipes), "", "fallback:候補なし")
        shown = _shortlist(options, 6)
        user = "\n\n".join([
            situation_text(view, rules, recipes, recent=recent),
            hand_text(view, recipes),
            "# 置き方の候補（コードの見積もり・見積もりの高い順。置かなかったカードは左隣へ渡る）\n"
            + "\n".join(_select_option_text(i + 1, o, recipes, view) for i, o in enumerate(shown)),
            "# 出力\nどの候補で置くか決めて、JSON だけを出力: {\"choice\": 候補番号, \"reason\": \"狙い（短く）\"}",
        ])
        d, dt, why = await self._ask(rules_d, user, budget)
        if d is not None:
            try:
                k = int(d.get("choice"))
            except (TypeError, ValueError):
                k = 0
            if 1 <= k <= len(shown):
                return Decision("select", shown[k - 1].picks, str(d.get("reason") or "")[:120], "llm", dt,
                                {"option": shown[k - 1], "rank": k})
            why = f"候補番号が不正({d.get('choice')!r})"
        return Decision("select", options[0].picks, "", f"fallback:{why}", dt, {"option": options[0]})

    # --- 作戦（strategy） ---
    async def decide_strategy(self, view: dict, rules: sim.Rules, rules_d: dict, recipes: dict, *,
                              budget: float, recent: Optional[list] = None) -> Decision:
        user = "\n\n".join([
            situation_text(view, rules, recipes, recent=recent, strategy=self.strategy),
            hand_text(view, recipes),
            "# 作戦を決める\n"
            "このラウンドの作戦を決める。カードの選び方・スキル・錬成は、この作戦に合わせてコードが自動で決める。\n"
            "・send: 多く送りたい素材（送り先の箱で満杯に近い素材を送るとあふれて点になる）\n"
            "・consume: 自分の箱から減らしたい素材（自分の箱が満杯に近い素材は相手に点を取られやすい）\n"
            "それぞれ0〜2個。前のラウンドの作戦を続けてもよい。\n" + MAT_KEYS_NOTE,
            "# 出力\nJSON だけを出力: {\"send\": [素材], \"consume\": [素材], \"reason\": \"狙い（短く）\"}",
        ])
        d, dt, why = await self._ask(rules_d, user, budget)
        if d is not None:
            send = list(dict.fromkeys(_materials(d.get("send"), rules.materials)))[:2]
            consume = list(dict.fromkeys(_materials(d.get("consume"), rules.materials)))[:2]
            reason = str(d.get("reason") or "")[:120]
            self.strategy = Strategy(tuple(send), tuple(consume), reason)
            return Decision("strategy", self.strategy, reason, "llm", dt)
        # 作戦が決まらなければ前の作戦のまま（無ければ見積もりのまま＝Bot と同じ重み）
        return Decision("strategy", self.strategy, self.strategy.note, f"fallback:{why}", dt)

    # --- スキル ---
    async def decide_skills(self, view: dict, rules: sim.Rules, rules_d: dict, recipes: dict, *,
                            budget: float, recent: Optional[list] = None) -> Decision:
        if self.mode == "strategy":
            uses = await asyncio.to_thread(autoplay.choose_skills, view, rules, recipes, self.strategy)
            return Decision("skill", uses, self.strategy.note, "code")
        pending = {k: int(v) for k, v in ((view.get("me") or {}).get("skill_pending") or {}).items() if int(v) > 0}
        code_uses = await asyncio.to_thread(autoplay.choose_skills, view, rules, recipes, NO_STRATEGY)
        hints = await asyncio.to_thread(_skill_hints, view, rules, recipes, pending)
        user = "\n\n".join([
            situation_text(view, rules, recipes, recent=recent),
            _tableau_text(view, recipes),
            "# 発動したスキル\n" + "\n".join(
                f"・{SKILL_JA[k]}（{k}）×{v}: {SKILL_DESC[k]}" for k, v in pending.items()),
            "# 見積もり（コード）\n" + hints,
            "# 出力\n発動した回数だけ使い方を決める。JSON だけを出力: "
            "{\"uses\": [{\"skill\": \"A\", \"materials\": [素材]}, ...], \"reason\": \"狙い（短く）\"}\n"
            "A と C は素材1個、B は0〜2個（自分の素材箱にある物だけ）。" + MAT_KEYS_NOTE,
        ])
        d, dt, why = await self._ask(rules_d, user, budget)
        if d is not None:
            uses, bad = await asyncio.to_thread(_validate_uses, d.get("uses"), pending, view, rules, recipes)
            left = {k: v - sum(1 for u in uses if u["skill"] == k) for k, v in pending.items()}
            if any(n > 0 for n in left.values()):
                # 足りない分（抜け・合法でなかった分）だけコードで補う。答えの合法な分は生かし、
                # 補う手はそれを使った後の状態から決める（同じ素材の召喚を重ねない・捨てた後の箱で考える）
                uses += await asyncio.to_thread(
                    autoplay.choose_skills, _after_uses(view, uses, left), rules, recipes, NO_STRATEGY)
            src = "llm" if not bad and not any(n > 0 for n in left.values()) else \
                f"llm+code:{'・'.join(bad) or '不足を補った'}"
            return Decision("skill", uses, str(d.get("reason") or "")[:120], src, dt)
        return Decision("skill", code_uses, "", f"fallback:{why}", dt)

    # --- 錬成 ---
    async def decide_brew(self, view: dict, rules: sim.Rules, rules_d: dict, recipes: dict, *,
                          budget: float, recent: Optional[list] = None) -> Decision:
        if self.mode == "strategy":
            steps = await asyncio.to_thread(autoplay.choose_brew, view, rules, recipes, self.strategy)
            return Decision("brew", steps, self.strategy.note, "code")
        start = autoplay.brew_start(view)
        if start is None:
            return Decision("brew", [], "", "code")
        sides = autoplay.tableau_sides(view, recipes)
        used = set(((view.get("me") or {}).get("brew") or {}).get("used") or [])
        target = autoplay.brew_target(view)
        plans = await asyncio.to_thread(sim.brew_plans, rules, sides, used, start, target, NO_STRATEGY, 5)
        if not plans:
            return Decision("brew", [], "", "code")  # 作れるレシピが無い
        names = [recipes[t["card"]].name for t in autoplay.me_player(view).get("tableau") or [] if t["card"] in recipes]
        user = "\n\n".join([
            situation_text(view, rules, recipes, recent=recent),
            _tableau_text(view, recipes),
            "# 錬成の手順の候補（コードの見積もり・見積もりの高い順）\n" + "\n".join(
                f"候補{i + 1}: {steps_text(s, sides, names)}\n  → {_outcome_text(oc)}" for i, (s, oc) in enumerate(plans)),
            "# 出力\nどの手順で錬成するか決めて、JSON だけを出力: {\"choice\": 候補番号, \"reason\": \"狙い（短く）\"}",
        ])
        d, dt, why = await self._ask(rules_d, user, budget)
        if d is not None:
            try:
                k = int(d.get("choice"))
            except (TypeError, ValueError):
                k = 0
            if 1 <= k <= len(plans):
                return Decision("brew", plans[k - 1][0], str(d.get("reason") or "")[:120], "llm", dt,
                                {"outcome": plans[k - 1][1], "rank": k})
            why = f"候補番号が不正({d.get('choice')!r})"
        return Decision("brew", plans[0][0], "", f"fallback:{why}", dt, {"outcome": plans[0][1]})


# ---------------------------------------------------------------------------- 候補の文
def _shortlist(options: list, k: int) -> list:
    """上位 k 件＋置く枚数ごとの最良（1枚だけ・2枚・3枚の選択肢を必ず見せる）。"""
    shown = list(options[:k])
    for n in (1, 2, 3):
        best = next((o for o in options if len(o.picks) == n), None)
        if best is not None and best not in shown:
            shown.append(best)
    return shown


def _select_option_text(i: int, o: sim.SelectOption, recipes: dict, view: dict) -> str:
    names = [recipes[c].name + ("（逆向き）" if r else "") for c, r in o.picks]
    sides = [recipes[c].sides(r) for c, r in o.picks]
    made = set(o.outcome.used)
    unused = [names[j] for j in range(len(names)) if j not in made]
    hand = (view.get("me") or {}).get("hand") or []
    passed = [recipes[c].name for c in hand if c not in {c2 for c2, _r in o.picks} and c in recipes]
    trig = "・".join(f"{SKILL_JA[s]}{t}回" for s, t in o.triggers.items()) or "なし"
    plan = steps_text(o.steps, sides, names)
    return (f"候補{i}: " + "＋".join(f"《{n}》" for n in names) + "\n"
            f"  錬成の見込み: {plan}" + (f"（作れないカード: {'・'.join(unused)}）" if unused else "") + "\n"
            f"  {_outcome_text(o.outcome)} / 発動するスキル: {trig}"
            + (f" / 左隣へ渡す札: {'・'.join('《' + p + '》' for p in passed)}" if passed else ""))


def _tableau_text(view: dict, recipes: dict) -> str:
    me = autoplay.me_player(view)
    rows = []
    for t in me.get("tableau") or []:
        if t["card"] in recipes:
            r = recipes[t["card"]]
            ins, outs = r.sides(bool(t.get("rotated")))
            rows.append(f"・《{r.name}》{'（逆向き）' if t.get('rotated') else ''} {sides_text(ins, outs)}")
    return "# あなたの工房（このラウンドに置いたレシピ）\n" + ("\n".join(rows) or "（なし）")


def _skill_hints(view: dict, rules: sim.Rules, recipes: dict, pending: dict) -> str:
    me = autoplay.me_player(view)
    box = dict(me.get("box") or {})
    target = autoplay.target_player(view).get("box") or {}
    sides = autoplay.tableau_sides(view, recipes)
    lines = []
    if "C" in pending:
        ranked = sim.best_supply(rules, sides, box, target, set())
        lines.append("召喚の見積もり（素材ごとの錬成の結果・良い順）: " + " / ".join(
            f"{mat(m)}→{_outcome_text(oc)}" for m, oc in ranked[:3]))
    if "B" in pending:
        need = sim.needed_from_box(rules, sides, box, target, set())
        spare = {m: max(0, box.get(m, 0) - need.get(m, 0)) for m in rules.materials}
        lines.append("浄化: 錬成で箱から使う予定 " + (counts_text(dict(need), zero=False) if need else "なし")
                     + " / 捨てても困らない余り " + counts_text(spare, rules.materials, zero=False))
    if "A" in pending:
        planned = autoplay.planned_flask(view, rules, recipes)
        frees = [(m, sim.free_after(rules, target, planned, m)) for m in rules.materials]
        lines.append("送り込み: 錬成で送る予定の分を足した後の、送り先の箱の空き（0以下ならあふれて1点）: "
                     + " ".join(f"{mat(m)}{f}" for m, f in frees))
    return "\n".join(lines) or "（なし）"


def _validate_uses(raw: Any, pending: dict, view: dict, rules: sim.Rules, recipes: dict) -> tuple:
    """LLM の答えのスキルの使い方から、合法で明らかな無駄・自爆でない分だけ残す → (uses, 捨てた理由の一覧)。

    コードで止めるもの（ユーザ「明らか自爆やアホみたいな行動しなければ許容」・実測 2026-10-10 で
    召喚を同じ素材に2回使った）:
    - 召喚(C): もう使い放題にしてある素材をもう一度選ぶ（2回目は何も起きない）
    - 浄化(B): 箱に無い素材・**工房の錬成で箱から使う予定の素材**を捨てる（作れるレシピが作れなくなる）
    決める順は 召喚 → 浄化 → 送り込み（召喚で使い放題になる素材は箱から使わないので、浄化の余りが変わる）。
    """
    if not isinstance(raw, list):
        return [], ["usesが無い"]
    me_view = view.get("me") or {}
    box = Counter({k: int(v) for k, v in (autoplay.me_player(view).get("box") or {}).items()})
    supply = set((me_view.get("skill_choices") or {}).get("supply") or [])
    left = dict(pending)
    parsed: list = []
    bad: list = []
    for u in raw:
        if not isinstance(u, dict):
            bad.append("形式")
            continue
        k = str(u.get("skill", "")).strip().upper()[:1]
        if k not in ("A", "B", "C") or pending.get(k, 0) <= 0:
            bad.append(f"{k or '?'}は発動していない")
            continue
        mats = _materials(u.get("materials"), rules.materials)
        if k in ("A", "C") and len(mats) != 1:
            bad.append(f"{k}の素材の数")
            continue
        parsed.append((k, mats[:2]))
    uses: list = []
    for k, mats in parsed:
        if k != "C" or left["C"] <= 0:
            continue
        if mats[0] in supply:
            bad.append("同じ素材をもう一度召喚")
            continue
        supply.add(mats[0])
        uses.append({"skill": "C", "materials": mats})
        left["C"] -= 1
    if any(k == "B" for k, _m in parsed):
        need = sim.needed_from_box(rules, autoplay.tableau_sides(view, recipes), dict(box),
                                   autoplay.target_player(view).get("box") or {}, supply)
        for k, mats in parsed:
            if k != "B" or left["B"] <= 0:
                continue
            keep = []
            for m in mats:
                if box.get(m, 0) - need.get(m, 0) > 0:
                    keep.append(m)
                    box[m] -= 1
                else:
                    bad.append("錬成で使う素材・箱に無い素材の浄化")
            if mats and not keep:
                continue  # 全部だめだった → この回はコードが余りから選ぶ（何も捨てない、とは答えていない）
            uses.append({"skill": "B", "materials": keep})
            left["B"] -= 1
    for k, mats in parsed:
        if k != "A" or left["A"] <= 0:
            continue
        uses.append({"skill": "A", "materials": mats})
        left["A"] -= 1
    return uses, bad


def _after_uses(view: dict, uses: list, left: dict) -> dict:
    """uses を使った後の state（残りの発動回数＝left）。足りない分をコードで決める時に使う。"""
    import copy

    v = copy.deepcopy(view)
    me_view = v.setdefault("me", {})
    me_view["skill_pending"] = {k: max(0, int(n)) for k, n in left.items()}
    choices = me_view.setdefault("skill_choices", {})
    supply = set(choices.get("supply") or [])
    sealed = Counter(choices.get("sealed") or {})
    box = autoplay.me_player(v).setdefault("box", {})
    for u in uses:
        for m in u["materials"]:
            if u["skill"] == "C":
                supply.add(m)
            elif u["skill"] == "A":
                sealed[m] += 1
            elif u["skill"] == "B":
                box[m] = int(box.get(m, 0)) - 1
    choices["supply"] = sorted(supply)
    choices["sealed"] = dict(sealed)
    return v
