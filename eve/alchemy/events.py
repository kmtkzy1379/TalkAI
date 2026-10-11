"""サーバーの出来事（events）を、イブの会話と判断役に渡す短い事実の文にする（純粋な関数）。

ここで作る文は**サーバーが送ってきた数だけ**から作る（推測を足さない＝プレイ内容のハルシネーション防止）。
「大きく動いた」かどうかもここで決める（ユーザ指定「毎行動うるさく反応するのはうざいから、ゲームが
大きく動いた瞬間などを話しかけられると良い」）: 点が入った・3連鎖以上・勝ち負けが決まった・対戦の始まり。
大きく動いた時だけ発話判定を起こし、話すかどうかは発話判定（LLM）が決める（ここは話させない）。
"""
from __future__ import annotations

from typing import Optional

from .rules import SKILL_JA, counts_text, mat

ME_NAME = "イブ"


def who(players: list, seat: Optional[int], me_seat: Optional[int]) -> str:
    if seat is None:
        return "?"
    if seat == me_seat:
        return ME_NAME
    if 0 <= seat < len(players):
        return str(players[seat].get("name", f"席{seat}"))
    return f"席{seat}"


def reveal_text(ev: dict, players: list, me_seat: Optional[int], recipes: dict) -> str:
    parts = []
    for p in ev.get("plays") or []:
        cards = "".join(
            f"《{recipes[c['card']].name if c.get('card') in recipes else c.get('card')}》"
            + ("（逆向き）" if c.get("rotated") else "") for c in p.get("cards") or [])
        note = "（時間切れで自動）" if p.get("auto") else ""
        trig = "・".join(f"{SKILL_JA.get(s, s)}{n}回" for s, n in (p.get("triggers") or {}).items())
        parts.append(f"{who(players, p.get('seat'), me_seat)}{cards}{note}" + (f"［スキル発動 {trig}］" if trig else ""))
    return f"{ev.get('round', '?')}ラウンド目に置いたカード: " + " / ".join(parts)


def skills_text(ev: dict, players: list, me_seat: Optional[int]) -> Optional[str]:
    rows = []
    for u in ev.get("uses") or []:
        if u.get("skipped"):
            continue
        name = who(players, u.get("seat"), me_seat)
        k = u.get("skill")
        ms = u.get("materials") or []
        if k == "A":
            rows.append(f"{name}が送り込みで{mat(ms[0]) if ms else '?'}を{who(players, u.get('to'), me_seat)}へ")
        elif k == "B":
            rows.append(f"{name}が浄化で{'・'.join(mat(m) for m in ms) or '何も'}を捨てた")
        elif k == "C":
            rows.append(f"{name}が召喚で{mat(ms[0]) if ms else '?'}を使い放題に")
    if not rows:
        return None
    return f"{ev.get('round', '?')}ラウンド目のスキル: " + " / ".join(rows)


def resolve_summary(ev: dict, players: list, me_seat: Optional[int], win_points: int) -> tuple:
    """(文, 大きく動いたか, 理由の短い語)。送った・あふれた・点・連鎖はサーバーの数そのまま。"""
    rows = []
    scorers = []
    match_point = []
    for t in ev.get("transfers") or []:
        frm, to = t.get("from"), t.get("to")
        sent = t.get("sent") or {}
        over = t.get("overflow") or {}
        pts = int(t.get("points") or 0)
        total = int(t.get("total_points") or 0)
        sent_text = counts_text(sent, zero=False) if sent else "何も送らず"
        line = f"{who(players, frm, me_seat)}→{who(players, to, me_seat)}: {sent_text}"
        if pts:
            line += (f"（{'・'.join(f'{mat(m)}{k}' for m, k in over.items())}があふれて"
                     f"{who(players, frm, me_seat)}が+{pts}点・計{total}点）")
            scorers.append(who(players, frm, me_seat))
        elif sent:
            line += "（あふれなし）"
        rows.append(line)
        if 0 < win_points - total <= 1:
            match_point.append(who(players, frm, me_seat))
    chains = []
    for seat, exes in (ev.get("executions") or {}).items():
        try:
            s = int(seat)
        except (TypeError, ValueError):
            continue
        depth = max((int(e.get("depth") or 1) for e in exes or []), default=0)
        if depth >= 2:
            chains.append((who(players, s, me_seat), depth))
    text = f"{ev.get('round', '?')}ラウンド目の送り合い: " + " / ".join(rows)
    if chains:
        text += " / 連鎖: " + "・".join(f"{n}が{d}連鎖" for n, d in chains)
    if match_point:
        text += f" / あと1点で勝ち: {'・'.join(dict.fromkeys(match_point))}"
    why = []
    if scorers:
        why.append("得点")
    if any(n == ME_NAME and d >= 3 for n, d in chains):
        why.append("イブの大連鎖")
    if match_point:
        why.append("あと1点")
    return text, bool(why), "・".join(why)


def end_text(ev: dict, players: list, me_seat: Optional[int]) -> str:
    winners = [who(players, s, me_seat) for s in ev.get("winners") or []]
    rank = " / ".join(
        f"{i + 1}位 {who(players, r.get('seat'), me_seat)} {r.get('points', 0)}点（箱{r.get('box_total', 0)}個）"
        for i, r in enumerate(ev.get("ranking") or []))
    reason = "勝ち点に到達" if ev.get("reason") == "points" else "最終ラウンドで打ち切り"
    me_won = me_seat is not None and me_seat in (ev.get("winners") or [])
    head = "イブの勝ち" if me_won and len(winners) == 1 else (
        "イブも勝ち（同点）" if me_won else f"{'・'.join(winners)}の勝ち（イブの負け）")
    return f"対戦終了（{ev.get('round', '?')}ラウンド目・{reason}）: {head}。{rank}"
