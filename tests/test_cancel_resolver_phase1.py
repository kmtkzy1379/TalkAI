"""CancelResolver の決定論テスト（API 不要・LLM を一切使わない）。

**1件制（2026-08-07 ユーザ裁定）**: `task/capabilities.py` の受付ゲートでアクティブは高々1件。
resolver は reference を照合せず件数だけで解決する:
- 0件 → 未配達の報告があれば抑止（D2）、無ければ「無い」報告。
- 1件 → reference の中身に関係なく即キャンセル（照合の余地が構造的に無い）。

旧実装はタスクLLM でファジー照合していた。撤去した理由は `cancel_resolver.py` の docstring。
応答LLM の cancel 撃ち直し（2026-07-13 実機事故の実体）は照合ではなく時間差で弾く
＝`task/capabilities.py` SELF_CANCEL_GUARD_SEC（テストは test_task_phase1.py にある）。
- TOCTOU: 書込直前に対象が terminal 化 → set_status False → 「もう終わってた」。
報告は CALLFUNCTION_RESULT・dedup_key=cancel:...。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_cancel_resolver_phase1.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.pipeline.stimulus import StimulusKind  # noqa: E402
from eve.pipeline.stimulus import CallFunctionResult, Stimulus  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.task import CANCELLED, DONE, PENDING, CancelResolver, Task, TaskStore  # noqa: E402
from eve.task.cancel_resolver import CancelRequest  # noqa: E402
from eve.task.schema import FAILED  # noqa: E402

_passed = 0
_failed = 0
_n = 0
_dir = tempfile.mkdtemp(prefix="eve_cancel_")
BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


def _tmp():
    global _n
    _n += 1
    return os.path.join(_dir, f"c{_n}.jsonl")


async def _run_once(store, reference):
    q = StimulusQueue()
    r = CancelResolver(store=store, queue=q)
    r.start()
    r.submit(reference)
    await asyncio.wait_for(r._inbox.join(), timeout=2.0)
    await r.stop()
    res = [s for s in q.snapshot() if s.kind == StimulusKind.CALLFUNCTION_RESULT]
    return res


async def _store(*tasks):
    s = TaskStore(task_file=_tmp(), now_fn=lambda: BASE)
    await s.initialize()
    for t in tasks:
        s.add(t)
    return s


async def t_zero():
    s = await _store()
    res = await _run_once(s, "さっきのやつ")
    await s.shutdown()
    return (len(res) == 1 and res[0].payload.ok is False
            and "無い" in res[0].payload.content and (res[0].dedup_key or "").startswith("cancel:"))


async def t_one_vague_code():
    # 1件×曖昧 reference → 即キャンセル（06:09:05 実機の正当ケースを維持）。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻を教えて", when=None))
    res = await _run_once(s, "やっぱりいいや")
    ok = (s.get("a").status == CANCELLED
          and len(res) == 1 and res[0].payload.ok is True
          and "30秒後に時刻を教えて" in res[0].payload.content and "止めた" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_one_empty_ref_code():
    # 1件×reference 省略（空）= 「今のをやめて」→ 即キャンセル。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻を教えて", when=None))
    res = await _run_once(s, "")
    ok = (s.get("a").status == CANCELLED and res[0].payload.ok is True)
    await s.shutdown()
    return ok


async def t_one_specific_also_cancels():
    """1件制（2026-08-07 ユーザ裁定）: reference に実質語があっても照合せず取り消す。

    旧実装はここでタスクLLM 照合に倒し、「30秒のやつ」vs「50秒後に…」の食い違いを none にして
    温存していた（2026-07-13 実機事故対応）。1件制ではその事故の前提が変わる:
    - タスクは同時に1つしか存在しない＝ユーザが対象を取り違えようがない
    - 事故の実体はユーザの勘違いではなく**応答LLM の cancel 撃ち直し**で、
      それは `task/capabilities.py` の SELF_CANCEL_GUARD_SEC（時間差）が弾く
    - 照合ロジック自体が実害を出した実績がある（E4 差し戻し `6cb6f25`）
    """
    s = await _store(Task(task_id="b", what="", goal="50秒後に今の気持ちを伝えて", when=None))
    res = await _run_once(s, "30秒のやつ")
    ok = (s.get("b").status == CANCELLED and res[0].payload.ok is True
          and "50秒後に今の気持ちを伝えて" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_no_llm_dependency():
    # 構造ガード: resolver は取消判断に LLM を使わない（model_registry を受け取らない）。
    # 往復が復活すると取消の不感時間（E3）が戻るので、シグネチャで固定する。
    import inspect
    params = set(inspect.signature(CancelResolver.__init__).parameters)
    return "model_registry" not in params and not hasattr(CancelResolver, "_resolve_with_llm")


async def t_multi_residual_cancels_latest():
    # 1件制ゲート導入前に作られた残留（旧バージョンの状態ファイル等）で actives≥2 になっても、
    # 落ちずに最新を1件だけ取り消す（他は温存）。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None),
                     Task(task_id="b", what="", goal="5分後にPC状態", when=None))
    res = await _run_once(s, "やめて")
    ok = (s.get("b").status == CANCELLED and s.get("a").status == PENDING
          and res[0].payload.ok is True and "5分後にPC状態" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_toctou_terminal_before_write():
    """TOCTOU: 判断から書込までの間に対象が terminal 化 → 「取り消した」と誤報しない。

    HANDOFF E3（取消の不感時間中に完了が滑り込む）の唯一の防御。LLM を外しても
    resolver は別コルーチン＝inbox の await をまたぐので、この窓は残る。
    """
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None))
    q = StimulusQueue()
    await q.put(Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT,
                         payload=CallFunctionResult("goal", "できた", True), dedup_key="task:a"))
    r = CancelResolver(store=s, queue=q)
    s.set_status("a", DONE, result="done")  # 書込直前に完了した状況
    await r._cancel_and_report(CancelRequest(reference="やめて"), "a")
    rep = [x for x in q.snapshot() if (x.dedup_key or "").startswith("cancel:")]
    left = [x for x in q.snapshot() if (x.dedup_key or "").startswith("task:")]
    ok = (s.get("a").status == DONE  # Cancelled に上書きされない（terminal-stays-terminal）
          and rep and rep[0].payload.ok is False and "もう終わってた" in rep[0].payload.content
          and left == [])  # 再配達は止める
    await s.shutdown()
    return ok


# --- D2: 完了が取消要求を追い越すレース（2026-07-29 実機 E2E）--------------------
# 候補は「時計で直近の terminal」ではなく **キューに未配達の報告が残っている terminal**。
# 実機 D2 時点の90秒窓には3件（東京タワー56.2s前/琵琶湖46.5s前/川1.9s前）あり、
# 時計だけだと「2件以上→抑止しない」に落ちて D2 が直らなかった。

def _terminal(task_id, goal, *, status=DONE, updated_at=None):
    """terminal 済みタスクを直接構築（store.add は updated_at を上書きしない）。"""
    t = Task(task_id=task_id, what="", goal=goal, when=None)
    t.status = status
    t.updated_at = BASE.isoformat() if updated_at is None else updated_at  # "" は空のまま渡す
    return t


async def _run_with_queue(store, reference, queued_keys=()):
    """キューに未配達の報告を並べた状態で resolver を回し、(報告, 残キー) を返す。"""
    q = StimulusQueue()
    for k in queued_keys:
        await q.put(Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT,
                             payload=CallFunctionResult("goal", f"結果:{k}", True), dedup_key=k))
    r = CancelResolver(store=store, queue=q, now_fn=lambda: BASE)
    r.start()
    r.submit(reference)
    await asyncio.wait_for(r._inbox.join(), timeout=2.0)
    await r.stop()
    reports = [s for s in q.snapshot() if (s.dedup_key or "").startswith("cancel:")]
    left = sorted((s.dedup_key or "") for s in q.snapshot() if (s.dedup_key or "").startswith("task:"))
    return reports, left, q


async def t_d2_single_undelivered_suppressed():
    # 実機 D2 の再現: terminal 3件だが未配達の報告は1件だけ → その1件だけ抑止する。
    s = await _store(_terminal("t_tower", "東京タワーの高さを調べて"),
                     _terminal("t_biwa", "琵琶湖の面積を調べて"),
                     _terminal("t_river", "世界で一番長い川を調べて"))
    rep, left, q = await _run_with_queue(s, "さっきの川の調べもの、やっぱりやめていいよ",
                                         queued_keys=["task:t_river"])
    ok = (left == []                                   # 川の報告は配達されない
          and len(rep) == 1 and rep[0].payload.ok is True
          and "世界で一番長い川" in rep[0].payload.content
          and "流さないでおくね" in rep[0].payload.content
          and "止めた" not in rep[0].payload.content    # status は Done＝「止めた」は嘘
          and q.is_suppressed("task:t_river")           # 遅れて来る再配達も遮断される
          and s.get("t_river").status == DONE)          # terminal-stays-terminal を壊さない
    await s.shutdown()
    return ok


async def t_d2_two_undelivered_not_suppressed():
    # 未配達が2件＝どれか特定できない →【ユーザ裁定】抑止せずそのまま流す。
    s = await _store(_terminal("t_a", "東京タワーの高さを調べて"),
                     _terminal("t_b", "琵琶湖の面積を調べて"))
    rep, left, q = await _run_with_queue(s, "やっぱりやめて",
                                         queued_keys=["task:t_a", "task:t_b"])
    ok = (left == ["task:t_a", "task:t_b"]  # 両方そのまま配達される（誤破棄しない）
          and len(rep) == 1 and rep[0].payload.ok is False
          and "特定できない" in rep[0].payload.content
          and not q.is_suppressed("task:t_a") and not q.is_suppressed("task:t_b"))
    await s.shutdown()
    return ok


async def t_d2_delivered_already_blocks_redelivery():
    # 未配達ゼロ＋直近 terminal 1件＝配達済み/配達中/再配達待ち。
    # 未来の約束はせず、再配達だけを止める。
    s = await _store(_terminal("t_river", "世界で一番長い川を調べて"))
    rep, left, q = await _run_with_queue(s, "やっぱりやめて")
    ok = (len(rep) == 1 and rep[0].payload.ok is False
          and "もう終わってた" in rep[0].payload.content
          and "かも" in rep[0].payload.content
          and "流さないでおくね" not in rep[0].payload.content  # 守れない約束をしない
          and q.is_suppressed("task:t_river"))
    await s.shutdown()
    return ok


async def t_d2_stale_terminal_not_candidate():
    # 90秒より古い terminal（再起動で復元された過去のタスク）は候補にしない＝現行文言のまま。
    old = (BASE - timedelta(days=3)).isoformat()
    s = await _store(_terminal("t_old", "3日前のタスク", updated_at=old))
    rep, left, q = await _run_with_queue(s, "やっぱりやめて")
    ok = (len(rep) == 1 and rep[0].payload.ok is False
          and "無い" in rep[0].payload.content
          and not q.is_suppressed("task:t_old"))
    await s.shutdown()
    return ok


async def t_d2_broken_updated_at_not_candidate():
    # updated_at が壊れている/空 → clock.elapsed_wall は 0.0(=たった今) に倒すが、
    # ここで「直近」に含めると誤破棄になるので明示的に弾く（配達側に倒す）。
    s = await _store(_terminal("t_broken", "壊れた時刻のタスク", updated_at=""),
                     _terminal("t_junk", "壊れた時刻のタスク2", updated_at="not-a-date"))
    rep, left, q = await _run_with_queue(s, "やっぱりやめて")
    ok = (len(rep) == 1 and "無い" in rep[0].payload.content
          and not q.is_suppressed("task:t_broken") and not q.is_suppressed("task:t_junk"))
    await s.shutdown()
    return ok


async def t_d2_cancelled_not_candidate():
    # Cancelled は executor が結果を破棄する（報告が発生しない）＝候補外。
    s = await _store(_terminal("t_c", "取消済みタスク", status=CANCELLED))
    rep, left, q = await _run_with_queue(s, "やっぱりやめて")
    ok = ("無い" in rep[0].payload.content and not q.is_suppressed("task:t_c"))
    await s.shutdown()
    return ok


async def t_d2_failed_is_candidate():
    # Failed も報告を put する（executor._finish）ので候補に含める。
    s = await _store(_terminal("t_f", "失敗したタスク", status=FAILED))
    rep, left, q = await _run_with_queue(s, "やっぱりやめて",
                                         queued_keys=["task:t_f"])
    ok = (left == [] and rep[0].payload.ok is True and q.is_suppressed("task:t_f"))
    await s.shutdown()
    return ok


async def t_d2_suppress_before_first_await():
    """【death-detection】suppress は `_handle_one` の最初の await より前に済むこと。

    resolver 起床→runner が次の刺激を pop するまでの猶予は call_soon 1スロットしかない。
    ここに await が1つ入ると報告が先に配達されて D2 が再発する（実測プローブで確認済）。
    """
    s = await _store(_terminal("t_river", "世界で一番長い川を調べて"))
    q = StimulusQueue()
    await q.put(Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT,
                         payload=CallFunctionResult("goal", "ナイル川", True),
                         dedup_key="task:t_river"))
    r = CancelResolver(store=s, queue=q, now_fn=lambda: BASE)
    coro = r._handle_one(CancelRequest(reference="やっぱりやめて"))
    # コルーチンを最初の中断点まで**1ステップだけ**進める＝この時点で除去済みなら await 前に済んでいる
    try:
        coro.send(None)
    except StopIteration:
        pass
    removed_before_await = not any((x.dedup_key or "").startswith("task:") for x in q.snapshot())
    coro.close()
    await s.shutdown()
    return removed_before_await


async def main():
    check("0件→止めるタスク無い(LLM 呼ばない)", await t_zero())
    check("1件×曖昧ref→コード即時キャンセル(LLM 呼ばない)", await t_one_vague_code())
    check("1件×空ref→コード即時キャンセル(LLM 呼ばない)", await t_one_empty_ref_code())
    check("1件制: 実質語refでも照合せずキャンセル", await t_one_specific_also_cancels())
    check("1件制: 取消判断に LLM を使わない(構造ガード)", await t_no_llm_dependency())
    check("1件制: 残留で≥2件でも落ちず最新のみ取消", await t_multi_residual_cancels_latest())
    check("TOCTOU: 書込直前に完了→もう終わってた(E3 防御)", await t_toctou_terminal_before_write())

    check("D2 未配達1件のみ抑止(terminal 3件でも一意)", await t_d2_single_undelivered_suppressed())
    check("D2 未配達2件は抑止せずそのまま流す", await t_d2_two_undelivered_not_suppressed())
    check("D2 配達済みは再配達だけ止め約束しない", await t_d2_delivered_already_blocks_redelivery())
    check("D2 古い terminal は候補外(回帰)", await t_d2_stale_terminal_not_candidate())
    check("D2 updated_at 破損は候補外(配達側に倒す)", await t_d2_broken_updated_at_not_candidate())
    check("D2 Cancelled は候補外", await t_d2_cancelled_not_candidate())
    check("D2 Failed は候補に含む", await t_d2_failed_is_candidate())
    check("D2 death: suppress は最初の await より前", await t_d2_suppress_before_first_await())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
