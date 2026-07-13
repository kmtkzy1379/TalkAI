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
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.pipeline.stimulus import StimulusKind  # noqa: E402
from eve.pipeline.stimulus_queue import StimulusQueue  # noqa: E402
from eve.task import CANCELLED, DONE, PENDING, CancelResolver, Task, TaskStore  # noqa: E402

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


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
