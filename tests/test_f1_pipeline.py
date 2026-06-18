"""F1 パイプライン骨格の決定論テスト（API 不要・純 stdlib）。

T1 配線レイテンシ / T3 順序+世代 / T7 E2E 破綻なし。
実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f1_pipeline.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.pipeline import (  # noqa: E402
    AudioPlayQueue,
    PipelineRunner,
    Stimulus,
    StimulusKind,
    StimulusQueue,
    StubOrchestrator,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


def _recorder():
    played: list = []

    async def play_fn(audio):
        played.append(audio)

    return played, play_fn


# ---------- T1 配線レイテンシ ----------
async def t1a_nonblocking_sidecar() -> bool:
    """2.0s の sidecar 実行中でも応答経路（runner）が即座に流れる。"""
    _, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    q = StimulusQueue()
    orch = StubOrchestrator(audio, sentences=1)
    runner = PipelineRunner(q, orch, audio)

    async def long_capability():
        await asyncio.sleep(2.0)  # DL/検索などの長時間 op を模す（サイドカー）

    side = asyncio.create_task(long_capability())
    await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "hi"))
    t0 = time.monotonic()
    await runner.run_once()
    dt = time.monotonic() - t0
    # 2.0s の sidecar は当然まだ終わっていない（=応答経路を塞いでいない証拠）
    sidecar_still_running = not side.done()
    side.cancel()
    return dt < 0.1 and len(orch.handled) == 1 and sidecar_still_running


async def t1b_parallel_compose() -> bool:
    """独立 prep は並列合成され、総時間は直列和より有意に短い。"""
    async def prep(delay: float) -> float:
        await asyncio.sleep(delay)
        return delay

    t0 = time.monotonic()
    await asyncio.gather(prep(0.10), prep(0.10))
    par = time.monotonic() - t0
    return par < 0.18  # 直列なら 0.20。並列なら ~0.10


async def t1b2_enqueue_nonblocking() -> bool:
    """put はノンブロッキングで、バースト投入が応答ループを塞がない。"""
    q = StimulusQueue()
    t0 = time.monotonic()
    for i in range(200):
        await q.put(Stimulus(StimulusKind.VISION_UPDATE, i, merge_key=f"v{i}"))
    dt = time.monotonic() - t0
    return dt < 0.1 and q.qsize() == 200


async def t1c_aging_no_starvation() -> bool:
    """高優先が流入し続けても、低優先(VISION)が aging で有限回内に drain される。"""
    now = [0.0]
    q = StimulusQueue(aging_threshold_s=2.0, aging_step_s=1.0, clock_fn=lambda: now[0])
    await q.put(Stimulus(StimulusKind.VISION_UPDATE, "v"))  # t=0 に投入（最低優先）
    drained_at = None
    for i in range(1, 11):
        now[0] = float(i)
        await q.put(Stimulus(StimulusKind.CALLFUNCTION_RESULT, f"cf{i}"))  # 毎回 fresh で高優先
        s = await q.get()
        if s.kind == StimulusKind.VISION_UPDATE:
            drained_at = i
            break
    return drained_at is not None and drained_at <= 10


# ---------- T3 順序 + 世代 ----------
async def t3_seq_order() -> bool:
    played, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())
    gen = audio.current_generation()
    audio.enqueue(gen, 2, "s2")  # 順不同で投入
    audio.enqueue(gen, 0, "s0")
    audio.enqueue(gen, 1, "s1")
    await audio.join()
    worker.cancel()
    return played == ["s0", "s1", "s2"]


async def t3_generation_drop() -> bool:
    played, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())
    g0 = audio.current_generation()
    audio.enqueue(g0, 0, "old0")
    audio.enqueue(g0, 2, "old2")  # seq1 が来ないので buffer に滞留（未再生）
    await audio.join()
    # barge-in: 世代を進め、滞留中の old2 は破棄されるべき
    g1 = audio.bump_generation()
    audio.enqueue(g0, 1, "stale")  # 旧世代 → 破棄
    audio.enqueue(g1, 0, "new0")
    audio.enqueue(g1, 1, "new1")
    await audio.join()
    worker.cancel()
    return played == ["old0", "new0", "new1"]  # old2 と stale は流れない


async def t3_midsentence_stop() -> bool:
    """B3: 再生中に世代が変わったら文の途中でも停止する（should_stop）。"""
    chunks: list = []
    started = asyncio.Event()

    async def play_fn(audio, should_stop):  # 2引数=should_stop を受ける
        for i in range(100):  # 長い文を模擬（100チャンク）
            if should_stop():
                break
            chunks.append(i)
            if i == 0:
                started.set()
            await asyncio.sleep(0.001)

    audio = AudioPlayQueue(play_fn=play_fn)
    worker = asyncio.create_task(audio.play_worker())
    audio.enqueue(audio.current_generation(), 0, "S")
    await started.wait()  # 再生中
    audio.bump_generation()  # barge-in（世代を進める）
    await asyncio.sleep(0.05)
    worker.cancel()
    return 0 < len(chunks) < 100  # 全チャンクは鳴らない＝途中で停止した


# ---------- T7 E2E パイプライン破綻なし ----------
async def t7_e2e() -> dict:
    played, play_fn = _recorder()
    audio = AudioPlayQueue(play_fn=play_fn)
    q = StimulusQueue()
    orch = StubOrchestrator(audio, sentences=1)
    runner = PipelineRunner(q, orch, audio)
    worker = asyncio.create_task(audio.play_worker())

    raised = None
    qsize_after_puts = 0
    try:
        # 台本: USER → VISION×3(merge) → CALLFUNCTION×2(dedup) → USER（coalesce 対象）
        await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "u1"))
        await q.put(Stimulus(StimulusKind.VISION_UPDATE, "v1", merge_key="vision"))
        await q.put(Stimulus(StimulusKind.VISION_UPDATE, "v2", merge_key="vision"))
        await q.put(Stimulus(StimulusKind.VISION_UPDATE, "v3", merge_key="vision"))
        await q.put(Stimulus(StimulusKind.CALLFUNCTION_RESULT, "r1", dedup_key="r1"))
        await q.put(Stimulus(StimulusKind.CALLFUNCTION_RESULT, "r1", dedup_key="r1"))
        await q.put(Stimulus(StimulusKind.USER_UTTERANCE, "u2"))
        qsize_after_puts = q.qsize()  # merge/dedup 後の件数（USER2 はまだ別々）
        while q.qsize() > 0:  # coalesce で USER は1回に畳まれるので件数固定でなく空まで
            await runner.run_once()
            await audio.join()
    except Exception as e:  # 破綻=例外を捕捉して報告
        raised = repr(e)
    finally:
        worker.cancel()

    kinds = [s.kind for s in orch.handled]
    user_payloads = [str(s.payload) for s in orch.handled if s.kind == StimulusKind.USER_UTTERANCE]
    return {
        "raised": raised,
        "qsize_after_puts": qsize_after_puts,
        "qsize_end": q.qsize(),
        "vision_count": kinds.count(StimulusKind.VISION_UPDATE),
        "cf_count": kinds.count(StimulusKind.CALLFUNCTION_RESULT),
        "user_count": kinds.count(StimulusKind.USER_UTTERANCE),
        "user_merged": user_payloads[0] if user_payloads else "",
        "played_count": len(played),
    }


async def main() -> None:
    check("T1a 非ブロッキング sidecar", await t1a_nonblocking_sidecar())
    check("T1b 並列合成 < 直列和", await t1b_parallel_compose())
    check("T1b2 put ノンブロッキング", await t1b2_enqueue_nonblocking())
    check("T1c aging で starvation 無し", await t1c_aging_no_starvation())

    check("T3 seq 順再生", await t3_seq_order())
    check("T3 世代で古い音声を破棄", await t3_generation_drop())
    check("T3 barge-inで文の途中でも停止", await t3_midsentence_stop())

    r = await t7_e2e()
    check("T7 例外なし", r["raised"] is None)
    check("T7 vision は1件に畳まれる", r["vision_count"] == 1)
    check("T7 callfunction は dedup される", r["cf_count"] == 1)
    check("T7 USER は coalesce で1件に", r["user_count"] == 1)
    check("T7 USER は u1+u2 を結合", "u1" in r["user_merged"] and "u2" in r["user_merged"])
    check("T7 put 後の件数 = 4(USER2+VISION1+CF1)", r["qsize_after_puts"] == 4)
    check("T7 最終キューは空", r["qsize_end"] == 0)
    check("T7 再生は3件(USER結合+CF+VISION)", r["played_count"] == 3)


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
