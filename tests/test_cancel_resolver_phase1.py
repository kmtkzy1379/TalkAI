"""CancelResolver inc2 の決定論テスト（API 不要・fake task LLM）。

ハイブリッド（2026-07-13 実機事故対応で改定）:
- 0件 → 「無い」報告（LLM 呼ばない）。
- 1件×曖昧 reference（空/「やっぱりいいや」等）→ コード即時キャンセル（LLM 呼ばない・速く確実）。
- 1件×実質語あり reference（「30秒のやつ」等）→ LLM 照合（無照合で唯一のアクティブを
  誤取消した実機事故の回帰ガード）。none なら温存。
- 2件以上 → fake task LLM の match/none/ambiguous に従う。
- LLM の幻覚 task_id（スナップショット外）は拒否。LLM 例外時は取り消さず名前つき報告。
- TOCTOU: LLM 決定後に対象が terminal 化 → set_status False → 「もう終わってた」。
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


class FakeModel:
    def __init__(self, content, on_call=None, raise_exc=None):
        self.content = content
        self.calls = 0
        self._on_call = on_call
        self._raise_exc = raise_exc

    async def complete(self, role, messages, **kw):
        self.calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._on_call:
            self._on_call()
        return {"choices": [{"message": {"content": self.content}}]}


async def _run_once(store, model, reference):
    q = StimulusQueue()
    r = CancelResolver(store=store, model_registry=model, queue=q)
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
    m = FakeModel(None)
    res = await _run_once(s, m, "さっきのやつ")
    await s.shutdown()
    return (m.calls == 0 and len(res) == 1 and res[0].payload.ok is False
            and "無い" in res[0].payload.content and (res[0].dedup_key or "").startswith("cancel:"))


async def t_one_vague_code():
    # 1件×曖昧 reference → LLM を呼ばず即キャンセル（06:09:05 実機の正当ケースを維持）。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻を教えて", when=None))
    m = FakeModel(None)
    res = await _run_once(s, m, "やっぱりいいや")
    ok = (m.calls == 0 and s.get("a").status == CANCELLED
          and len(res) == 1 and res[0].payload.ok is True
          and "30秒後に時刻を教えて" in res[0].payload.content and "止めた" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_one_empty_ref_code():
    # 1件×reference 省略（空）= 「今のをやめて」→ LLM を呼ばず即キャンセル。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻を教えて", when=None))
    m = FakeModel(None)
    res = await _run_once(s, m, "")
    ok = (m.calls == 0 and s.get("a").status == CANCELLED and res[0].payload.ok is True)
    await s.shutdown()
    return ok


async def t_one_specific_none():
    # 実機事故 2026-07-13 の回帰: 1件×不一致 reference「30秒のやつ」（アクティブは50秒）
    # → LLM 照合で none → 温存（旧実装は無照合で即取消していた）。
    s = await _store(Task(task_id="b", what="", goal="50秒後に今の気持ちを伝えて", when=None))
    m = FakeModel('{"decision":"none","message":"そんなタスクは入ってないよ。今あるのは50秒後のだよ"}')
    res = await _run_once(s, m, "30秒のやつ")
    ok = (m.calls == 1 and s.get("b").status == PENDING
          and res[0].payload.ok is False and "入ってない" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_one_specific_match():
    # 1件×内容一致 reference → LLM 照合経由で取消（旧 t_one_code の意図を LLM 経由で継承）。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻を教えて", when=None))
    m = FakeModel('{"decision":"match","task_id":"a","message":"「30秒後に時刻を教えて」を止めるね"}')
    res = await _run_once(s, m, "時刻のやつ")
    ok = (m.calls == 1 and s.get("a").status == CANCELLED and res[0].payload.ok is True)
    await s.shutdown()
    return ok


async def t_one_llm_error_keeps_task():
    # 1件×実質語 reference×LLM 例外 → 取り消さず、名前つきで「そのまま」報告（fail-safe）。
    s = await _store(Task(task_id="b", what="", goal="50秒後に今の気持ちを伝えて", when=None))
    m = FakeModel(None, raise_exc=RuntimeError("api down"))
    res = await _run_once(s, m, "30秒のやつ")
    ok = (m.calls == 1 and s.get("b").status == PENDING
          and res[0].payload.ok is False and "50秒後に今の気持ちを伝えて" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_llm_bad_task_id():
    # LLM が幻覚 task_id で match → スナップショット外 ID は拒否し、両タスク温存。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None),
                     Task(task_id="b", what="", goal="5分後にPC状態", when=None))
    m = FakeModel('{"decision":"match","task_id":"zzz","message":"止めるね"}')
    res = await _run_once(s, m, "時刻のやつ")
    ok = (s.get("a").status == PENDING and s.get("b").status == PENDING
          and res[0].payload.ok is False)
    await s.shutdown()
    return ok


async def t_multi_match():
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None),
                     Task(task_id="b", what="", goal="5分後にPC状態", when=None))
    m = FakeModel('{"decision":"match","task_id":"a","message":"「30秒後に時刻」を止めるね"}')
    res = await _run_once(s, m, "時刻のやつ")
    ok = (m.calls == 1 and s.get("a").status == CANCELLED and s.get("b").status == PENDING
          and res[0].payload.ok is True and "止める" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_multi_none():
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None),
                     Task(task_id="b", what="", goal="5分後にPC状態", when=None))
    m = FakeModel('{"decision":"none","message":"そんなタスクは入ってないよ"}')
    res = await _run_once(s, m, "天気のやつ")
    ok = (s.get("a").status == PENDING and s.get("b").status == PENDING
          and res[0].payload.ok is False and "入ってない" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_multi_ambiguous():
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None),
                     Task(task_id="b", what="", goal="30秒後にPC状態", when=None))
    m = FakeModel('{"decision":"ambiguous","message":"時刻とPC状態、どっち?"}')
    res = await _run_once(s, m, "30秒後のやつ")
    ok = (s.get("a").status == PENDING and s.get("b").status == PENDING
          and res[0].payload.ok is False and "どっち" in res[0].payload.content)
    await s.shutdown()
    return ok


async def t_toctou_finished_during_llm():
    # 2件→LLM が a を指すが、LLM 中に a が Done 化 → set_status False → 「もう終わってた」。
    s = await _store(Task(task_id="a", what="", goal="30秒後に時刻", when=None),
                     Task(task_id="b", what="", goal="5分後にPC状態", when=None))
    m = FakeModel('{"decision":"match","task_id":"a","message":"止めるね"}',
                  on_call=lambda: s.set_status("a", DONE, result="done"))
    res = await _run_once(s, m, "時刻のやつ")
    ok = (s.get("a").status == DONE  # Cancelled に上書きされない（terminal-stays-terminal）
          and res[0].payload.ok is False and "もう終わってた" in res[0].payload.content)
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


async def _run_with_queue(store, model, reference, queued_keys=()):
    """キューに未配達の報告を並べた状態で resolver を回し、(報告, 残キー) を返す。"""
    q = StimulusQueue()
    for k in queued_keys:
        await q.put(Stimulus(kind=StimulusKind.CALLFUNCTION_RESULT,
                             payload=CallFunctionResult("goal", f"結果:{k}", True), dedup_key=k))
    r = CancelResolver(store=store, model_registry=model, queue=q, now_fn=lambda: BASE)
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
    rep, left, q = await _run_with_queue(s, FakeModel(None), "さっきの川の調べもの、やっぱりやめていいよ",
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
    rep, left, q = await _run_with_queue(s, FakeModel(None), "やっぱりやめて",
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
    rep, left, q = await _run_with_queue(s, FakeModel(None), "やっぱりやめて")
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
    rep, left, q = await _run_with_queue(s, FakeModel(None), "やっぱりやめて")
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
    rep, left, q = await _run_with_queue(s, FakeModel(None), "やっぱりやめて")
    ok = (len(rep) == 1 and "無い" in rep[0].payload.content
          and not q.is_suppressed("task:t_broken") and not q.is_suppressed("task:t_junk"))
    await s.shutdown()
    return ok


async def t_d2_cancelled_not_candidate():
    # Cancelled は executor が結果を破棄する（報告が発生しない）＝候補外。
    s = await _store(_terminal("t_c", "取消済みタスク", status=CANCELLED))
    rep, left, q = await _run_with_queue(s, FakeModel(None), "やっぱりやめて")
    ok = ("無い" in rep[0].payload.content and not q.is_suppressed("task:t_c"))
    await s.shutdown()
    return ok


async def t_d2_failed_is_candidate():
    # Failed も報告を put する（executor._finish）ので候補に含める。
    s = await _store(_terminal("t_f", "失敗したタスク", status=FAILED))
    rep, left, q = await _run_with_queue(s, FakeModel(None), "やっぱりやめて",
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
    r = CancelResolver(store=s, model_registry=FakeModel(None), queue=q, now_fn=lambda: BASE)
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
    check("1件×不一致ref→LLM none で温存(実機事故回帰)", await t_one_specific_none())
    check("1件×一致ref→LLM match でキャンセル", await t_one_specific_match())
    check("1件×LLM例外→取り消さず名前つき報告(fail-safe)", await t_one_llm_error_keeps_task())
    check("LLM 幻覚 task_id→拒否して温存", await t_llm_bad_task_id())
    check("≥2件→LLM match でその1件をキャンセル", await t_multi_match())
    check("≥2件→LLM none は status 不変で報告", await t_multi_none())
    check("≥2件→LLM ambiguous は確認質問", await t_multi_ambiguous())
    check("TOCTOU: LLM 中に完了→set_status False→もう終わってた", await t_toctou_finished_during_llm())

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
