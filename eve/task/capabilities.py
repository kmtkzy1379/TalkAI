"""タスク管理の能力を CapabilityRegistry へ登録（**応答LLM は委譲/取消の意図を渡すだけ**）。

- `delegate_task(goal, when_seconds?)`: 自然文ゴールを TaskAgent に委譲。即時も後回しも状態確認も全部これ。
- `cancel_task(reference?)`: 取消の意図をそのまま渡す（どのタスクかはタスク側が解決。Part B で CancelResolver 化）。
- `list_tasks`: **agent 専用**(agent_tool)＝「予約教えて」も委譲で答える。
実行系能力(self_status/pc_status/将来の検索・画面操作)は agent_tool=True で TaskAgent 専用。
handler は同期・非ブロッキング（store へ即追加 / 状態は store がコード一本化で持つ）。

**1件制（2026-08-07 ユーザ裁定）**: 「依頼を受けてから報告を発話し終えるまで」が常に1件。
アクティブ(Pending/Running)か未配達の完了報告が1つでもあれば新規委譲を**コードで断る**。
狙いは取消の曖昧さの構造的除去（候補が常に高々1件＝どれを指しているか迷う余地が無い）。
統合・優先度付けは**やらない**（複雑化がバグの起点になるという裁定）。自発タスクが枠を
埋めてユーザ依頼が断られる場合は、ユーザが取り消すか待つ。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..capability.registry import MEANS_BOUNDARY, NEVER_MEANS, Capability, CapabilityRegistry
from ..clock import humanize_eta
from .cancel_resolver import undelivered_reports
from .schema import CANCELLED, PENDING, RUNNING, TERMINAL, Task, new_task_id

_DEDUP_TOL_SEC = 20.0  # 同一 goal で when がこの秒数以内なら重複予約とみなす

# 1件制の前提: 予約で枠を無限に押さえられると機能が止まる（`_iso_in` に上限が無く、Pending は
# 再起動しても age-out されない＝`store._age_out_orphans` は Running のみ）。「夕方に教えて」
# 1件で半日ロックするのを防ぐ。
MAX_WHEN_SEC = 3600.0

# 応答LLM が同一ターンで delegate_task → cancel_task を続けて撃つ事故（2026-07-13 実機・
# `tools/task_incident_replay.py:752` が 65μ秒の自己取消として再現）への決定論ガード。
# tool_calls は逐次実行なのでこの間隔はμ秒オーダー。一方ユーザの本物の取消は、応答音声を
# 再生し終えてから tool が走る構造（`response/orchestrator.py:453,460`）上、必ず次ターン＝
# 数秒後になる。よって短い窓で両者は綺麗に分離できる。
SELF_CANCEL_GUARD_SEC = 1.0


def _span_ja(sec: float) -> str:
    """予約の上限を delegate_task の説明に書くための言い方（3600→「1時間」・1800→「30分」）。"""
    s = int(sec)
    return f"{s // 3600}時間" if s % 3600 == 0 else f"{s // 60}分"


def _display_name(t) -> str:
    """一覧/取消の表示名。goal タスクは what="" なので goal を優先（RC4 空名対策）。"""
    return t.goal or t.what or "(無題)"


def _iso_in(seconds) -> Optional[str]:
    if isinstance(seconds, (int, float)) and seconds > 0:
        return (datetime.now(timezone.utc) + timedelta(seconds=float(seconds))).isoformat()
    return None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _near(a: Optional[datetime], b: Optional[datetime], tol: float = _DEDUP_TOL_SEC) -> bool:
    if a is None and b is None:
        return True  # 両方 即時（when None）は同一窓
    if a is None or b is None:
        return False
    return abs((a - b).total_seconds()) <= tol


def active_tasks_for_context(store, *, now: Optional[datetime] = None, limit: int = 5,
                             undelivered=()) -> list[str]:
    """応答LLM の system 注入用の予約タスク行（Fix#2・2026-07-13 実機事故対応）。

    Pending/Running のみ・**task_id は出さない**（読み上げ leak の唯一の確実な防御は
    そもそも渡さないこと。cancel は reference 自然文で解決するので ID 不要）。
    残り時間は丸め表現（復唱されても自然）。ゼロ件は呼び出し側が「（予約タスクは無い）」を出す。

    `undelivered` は「終わったが結果をまだ喋っていない」タスク（1件制ではこれも枠を握る）。
    **ETA を付けない**: `humanize_eta` は負値を 0 に丸めるので、完了済みに時刻表現を付けると
    「まもなく実行」＝これから実行すると読める嘘になる（Fix#2 が潰した事故クラスそのもの）。
    """
    now = now or datetime.now(timezone.utc)
    act = [t for t in store.list_all() if t.status in (PENDING, RUNNING)]
    act.sort(key=lambda t: t.when or "")  # when は UTC ISO 固定形式（_iso_in 産）＝辞書順で時刻順
    lines: list[str] = []
    for t in act[:limit]:
        if t.status == RUNNING:
            eta = "実行中"
        else:
            w = _parse_iso(t.when)
            eta = "まもなく実行" if w is None else humanize_eta((w - now).total_seconds()) + "で実行"
        lines.append(f"・「{_display_name(t)}」（{eta}）")
    if len(act) > limit:
        lines.append(f"・ほか{len(act) - limit}件")
    for t in undelivered:
        lines.append(f"・「{_display_name(t)}」（終わった・結果をまだ伝えていない）")
    return lines


def register_task_capabilities(registry: CapabilityRegistry, store, cancel_resolver=None,
                               queue=None) -> None:
    """`queue` は1件制の「未配達の完了報告」を数えるために使う（None なら未配達は数えない＝
    アクティブだけで判定する。決定論テストや queue を持たない tools 向けの縮退動作）。"""
    # 同一ターンの自己取消ガード用（`SELF_CANCEL_GUARD_SEC`）。closure でだけ共有する。
    last_add = {"mono": 0.0}

    def _slot_holder():
        """いま枠を握っているタスクと、その状態語（無ければ (None, "")）。"""
        for t in store.list_all():
            if t.status == RUNNING:
                return t, "やってる"
        for t in store.list_all():
            if t.status == PENDING:
                return t, "予約してる"
        if queue is not None:
            for t in undelivered_reports(store, queue):
                return t, "終えたけどまだ結果を伝えてない"
        return None, ""

    def _find_duplicate(goal: str, when: Optional[str]):
        # 同じ goal かつ when が近接の PENDING/Running があれば重複（混乱ターンの再作成を弾く＝RC2。
        # Running も見るのは、実行中に応答LLMが同じゴールを再委譲する事故＝二重検索/二重報告を
        # 構造的に防ぐため。claim_due() は when を変更しないので when 近接判定は Running でも成立）。
        target = _parse_iso(when)
        for t in store.list_all():
            if (t.status in (PENDING, RUNNING) and (t.goal or "").strip() == goal
                    and _near(_parse_iso(t.when), target)):
                return t
        return None

    def _delegate_task(args: dict) -> str:
        goal = (args.get("goal") or "").strip()
        if not goal:
            return "（ゴールが空でした）"
        secs = args.get("when_seconds")
        if isinstance(secs, (int, float)) and secs > MAX_WHEN_SEC:
            # 断る文は必ず「（」始まり: dispatcher は report_result=False の能力の戻り値を
            # `content.startswith("（")` でしか報告しない（`function_dispatcher.py:100,111`）。
            # 付け忘れると「やっとくね」と言った直後に依頼だけ黙って消える。
            return f"（{int(MAX_WHEN_SEC // 60)}分より先の予約はまだ受けられないよ）"
        when = _iso_in(secs)
        # 重複判定を先に置く: 同じ goal の再委譲には専用の文言がある（RC2）。1件制の一般文より
        # 正確で、かつ**そのタスクは実在して報告される**ので黙っていても実害が無い。
        dup = _find_duplicate(goal, when)
        if dup is not None:
            if dup.status == RUNNING:
                return f"それ、今ちょうど調べてるところだよ（ID:{dup.task_id}）。"
            return f"それはもう予約してるよ（ID:{dup.task_id}）。"
        # 1件制ゲート（2026-08-07 ユーザ裁定）。ここで作らないので**必ず声に出す**。
        #
        # E15 の2重構造（ユーザ裁定 2026-08-07）: 1段目はプロンプト（`context_assembler.py`）で
        # 「引き受ける言い方をしない」。それをすり抜けて呼んでしまった場合の2段目がここで、
        # **訂正の指示そのものを戻り値に埋める**。応答本文は tool より先に喋り終わるので
        # （`response/orchestrator.py:453,460`）、この時点で既に「やっとくね」と言った後の
        # ことがある＝黙って断ると依頼が蒸発する。だから「訂正しろ」まで書く。
        holder, state = _slot_holder()
        if holder is not None:
            name = _display_name(holder)
            return (f"（受け付けられなかった。今は「{name[:40]}」を{state}。"
                    f"**さっき引き受けるように言っていたら、ここで必ず訂正する**"
                    f"（例:「ごめん、今『{name[:20]}』をやってて、それはまだ受けられないんだ」）。"
                    # この依頼はどこにも保存されていない。「あとでやる」と言わせると必ず破る
                    # （実測 2026-08-07 e15_run3 で 2/2 が「終わったら調べるね」と言った）。
                    # ユーザに言い直してもらうのが唯一の確実な引き継ぎ手段。
                    "**「あとでやる」と約束しない**（覚えておく仕組みが無い）。"
                    "終わったらもう一度言ってほしいと頼むか、今のを取り消して先にやるかを聞くこと。"
                    "**この依頼の中身を自分の記憶で答えて済ませない**）")
        task = Task(task_id=new_task_id(), what="", goal=goal, when=when)
        store.add(task)
        last_add["mono"] = time.monotonic()
        eta = f"{int(args['when_seconds'])}秒後" if when else "すぐ"
        return f"タスクを引き受けたよ（{goal[:24]} / {eta} / ID:{task.task_id}）。"

    def _list_tasks(args: dict) -> str:
        tasks = store.list_all()
        if not tasks:
            return "今は登録されているタスクは無いよ。"
        lines = [f"・{_display_name(t)}（{t.status}{'・' + t.when if t.when else ''}）ID:{t.task_id}" for t in tasks[-10:]]
        return "登録中のタスク:\n" + "\n".join(lines)

    def _cancel_task(args: dict) -> str:
        reference = (args.get("reference") or args.get("task_id") or "").strip()
        # 自己取消ガード: いま作ったばかりのタスクへの取消は応答LLM の撃ち直し（2026-07-13）。
        # 1件制では候補が常に1件＝この cancel は必ず生まれたてのタスクに当たるので、照合ではなく
        # **時間差**で弾く。ユーザの本物の取消は音声再生後＝必ず次ターンなので巻き込まない。
        if last_add["mono"] and time.monotonic() - last_add["mono"] < SELF_CANCEL_GUARD_SEC:
            return "（今引き受けたばかりだから、取り消しはしないでおくね）"
        if cancel_resolver is not None:
            # 本番: タスク側(CancelResolver)がファジー解決＋報告する（非同期）。ここは受付けるだけ。
            cancel_resolver.submit(reference)
            return "取消を確認するね。"  # report_result=False で抑制・実結果は resolver が報告
        # フォールバック(resolver 未接続・主にテスト): 直近未終了を即キャンセル。
        actives = [t for t in store.list_all() if t.status in (PENDING, RUNNING)]
        if not actives:
            return "（今は取り消せる予約タスクは無いよ）"
        t = actives[-1]
        store.set_status(t.task_id, CANCELLED)
        return f"予約していた「{_display_name(t)}」を取り消したよ。"

    registry.register(Capability(
        name="delegate_task",
        description="やりたいこと・調べ物・状態確認・後回しの予約を、賢いタスク担当(TaskAgent)に任せる。"
                    "goal に自然文で『何を達成/報告したいか』を、**ユーザの言い回しをそのまま**書く"
                    "（例: 30秒後に今の時刻を教えて / PCとイブの状態を調べてまとめて）。"
                    # 2026-10-01: 旧文言「どんな簡単な事も自分でやらずここへ」が system の「一般常識はその場で
                    # 答える」と食い違い、富士山の高さまで委譲していた（ユーザ「簡単な知識はタスクに回さず」）
                    "一般常識で答えられることはここに回さず自分で答える。会話の返事やユーザへの質問も入れない。"
                    # 上限を書かないと「明日の朝7時に起こして」を引き受けて、受付で断られる。「声で知らせる」まで
                    # 書かないと、今度は「30分後に起こして」（できる）まで断った
                    f"when_seconds 省略=すぐ、指定=その秒後（最大{int(MAX_WHEN_SEC)}秒＝{_span_ja(MAX_WHEN_SEC)}先まで。"
                    "その時間になったらイブが声で知らせる。それより先の予約はできない）。"
                    # ユーザ 2026-09-29 採用: 旧文言「呼んだら『やっとくね』とだけ言う」は定型化して機械的だった
                    # （ユーザ「｢やっとくね｣が定型文化してて機械的。出来ればその文脈にあった言い回しとかして欲しい」）
                    "呼ぶ前に、依頼の中身に合った短い一言で引き受ける（毎回同じ言い方にしない）。"
                    # D3: 万能感の打ち消し。**available 側を列挙しない**（search は SEARCH_ENABLED ∧ ddgs の
                    # 内側でしか登録されず、静的な文字列では構成差を表現できないため必ず嘘になる）。
                    # J-3: 否定側も同じ理由で列挙をやめた。旧文言「外の世界を操作する手段は無い」は
                    # 作業フォルダ・プログラム実行の追加でそれ自体が嘘になるため、**構成に依らず恒真な**
                    # 言い方（MEANS_BOUNDARY）に置換し、永久に持たない物だけ（NEVER_MEANS）を例示する。
                    f"タスク担当が外の世界にできることにも境界がある: **{MEANS_BOUNDARY}**"
                    f"（{'・'.join(NEVER_MEANS)}などはできない）。手段に対応しない操作は goal にしない。"
                    "調べ物や、考えて答えるだけの依頼は goal にしてよい。",
        params_schema={
            "goal": {"type": "string", "description": "達成/報告したいこと（自然文・ユーザの言い回しを保つ）"},
            "when_seconds": {"type": "integer", "description": "何秒後に始めるか（省略=すぐ）"},
            # 応答LLM が本文を書かずにツールだけ返したターンは、定型の「やっとくね。」の代わりにこれを言う
            # （`response/orchestrator.py` の `_ack_from_tools`・ユーザ採用 2026-09-29）
            "reply": {"type": "string",
                      "description": "ユーザに今すぐ返す短い一言（依頼の中身に合わせた自然な話し言葉・毎回同じ言い方にしない）"},
        },
        handler=_delegate_task, mutates_state=True, report_result=False,  # ack は応答本文が担う
        required=("goal", "reply"),
    ))
    registry.register(Capability(
        name="list_tasks", description="登録中の予約タスク一覧を見る。引数なし。", params_schema={},
        handler=_list_tasks, offered=False, agent_tool=True,  # agent 専用（「予約教えて」も委譲で答える）
    ))
    registry.register(Capability(
        name="cancel_task",
        description="タスクの取り消し。『キャンセルして』『やっぱりいい』等と言われたら、ユーザの言い回しを"
                    "reference にそのまま入れて1回呼ぶ（どのタスクかはタスク側が判断する）。",
        params_schema={"reference": {"type": "string", "description": "取り消したいタスクのユーザの言い回し（省略可）"}},
        handler=_cancel_task, mutates_state=True, report_result=False,  # 実結果は CancelResolver が報告
    ))
