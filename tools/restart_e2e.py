r"""実起動で各機能の ON/OFF（停止→再開）を実際に叩いて確かめる検証ハーネス。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\restart_e2e.py

**本番の `VoiceLoop` をそのまま起動する**（`tools/voice_chat.py` と同じ経路・同じ `.env`）。
UI のメニュー（停止/再開・自律発話・画面認識・タスク）が将来やる操作を、UI 抜きで
先に叩いて「本当に止まって、本当に再開する」を観測で確かめるのが目的。

判定は**発話やログの文字列ではなく、実際に回っている回数**で行う:
  - 自律発話 = 発話判定トリガの発火回数
  - 画面認識 = capture スレッドからループへ橋渡しされたフレーム数
  - タスク   = 応答LLM に見えている tool 一覧と、サイドカーのタスクハンドル
  - マイク   = AudioInput の再生成と稼働フラグ

**実 API を使う**（起動時のウォームアップ1回 + 自律発話の判定LLM 数回）。所要 約4分。
発話頻度を上げて短時間で観測するため `SILENCE_THRESHOLD_SEC` を下げて起動する
（`.env` は書き換えない。**この1点だけ実起動と違う**ので、判定はトリガの有無だけに使う）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `eve.config` を import する**前**に環境変数を上書きする（落とし穴 #3）。
os.environ.setdefault("SILENCE_THRESHOLD_SEC", "3.0")

from eve.logsetup import configure  # noqa: E402
from eve.voice_loop import VoiceLoop  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}  {extra}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {extra}")


async def wait_ready(vl: VoiceLoop, timeout: float = 300.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if vl.input._ai is not None:  # mic 起動＝run() が最後まで到達した
            return True
        await asyncio.sleep(0.5)
    return False


async def main() -> int:
    configure()
    vl = VoiceLoop()

    # --- 観測点（run() より前に差し込む: CaptureThread は run() 内で属性を読む）---
    trig: list[str] = []
    frames: list[int] = []
    _orig_trigger = vl.speech_decider.trigger
    _orig_on_frame = vl.vlm_worker.on_frame

    def trigger_wrap(source: str = "?") -> None:
        trig.append(source)
        return _orig_trigger(source)

    def on_frame_wrap(frame, changed) -> None:
        frames.append(1)
        return _orig_on_frame(frame, changed)

    vl.speech_decider.trigger = trigger_wrap      # type: ignore[method-assign]
    vl.vlm_worker.on_frame = on_frame_wrap        # type: ignore[method-assign]

    run_task = asyncio.create_task(vl.run())
    print("起動中（ウォームアップに数分かかる）…")
    if not await wait_ready(vl):
        print("❌ 起動しなかった")
        run_task.cancel()
        return 1
    print("起動完了。ON/OFF の検証を始める\n")
    await asyncio.sleep(2.0)

    # ---------------------------------------------------------------- 自律発話
    print("[1] 自律発話 OFF → ON")
    n0 = len(trig)
    await asyncio.sleep(10.0)
    base = len(trig) - n0
    check("OFF にする前は発火している", base > 0, f"10秒で {base} 回")

    await vl.silence_monitor.stop()
    await vl.speech_decider.stop()
    n1 = len(trig)
    await asyncio.sleep(10.0)
    off = len(trig) - n1
    check("OFF の間は1回も発火しない", off == 0, f"10秒で {off} 回")

    vl.speech_decider.start()
    vl.silence_monitor.start()
    n2 = len(trig)
    # 窓は再評価カデンス（SILENCE_THRESHOLD_SEC + tick）の数周ぶん取る。
    # 初回の実測では 10秒窓が短すぎて「再開していないように見えた」（実際は11秒後に発火）。
    t_on = time.monotonic()
    while len(trig) == n2 and time.monotonic() - t_on < 20.0:
        await asyncio.sleep(0.5)
    delay = time.monotonic() - t_on
    on = len(trig) - n2
    check("ON に戻すと再び発火する", on > 0, f"{delay:.1f}秒後に {on} 回")
    # `_event` の消し忘れがあると無駄な判定LLM が1回走り、その間 is_idle() が False で
    # 発火できない＝初回が大きく遅れる。カデンス2周ぶん以内なら健全とみなす。
    check("再開の初回が遅れない（無駄な判定が走っていない）", delay < 9.0, f"{delay:.1f}秒")
    check("再開後も判定worker が生きている（トリガが処理されている）",
          vl.speech_decider._task is not None and not vl.speech_decider._stopping)

    # ---------------------------------------------------------------- 画面認識
    print("\n[2] 画面認識 OFF → ON")
    if vl.capture_thread is None:
        check("VLM_ENABLED=1 で起動している", False, "→ .env を確認（この項目は飛ばす）")
    else:
        f0 = len(frames)
        await asyncio.sleep(4.0)
        base_f = len(frames) - f0
        check("OFF にする前はフレームが届いている", base_f > 0, f"4秒で {base_f} 枚")

        await asyncio.to_thread(vl.capture_thread.stop)   # join はループ外で
        await vl.vlm_worker.stop()
        f1 = len(frames)
        await asyncio.sleep(4.0)
        off_f = len(frames) - f1
        check("OFF の間はフレームが1枚も来ない", off_f == 0, f"4秒で {off_f} 枚")

        # **インスタンスを作り直さずに**再 start できるか（`_stop.clear()` の効果）
        vl.vlm_worker.start()
        vl.capture_thread.start()
        f2 = len(frames)
        await asyncio.sleep(4.0)
        on_f = len(frames) - f2
        check("ON に戻すとフレームが再開する（同じインスタンスのまま）", on_f > 0,
              f"4秒で {on_f} 枚")
        check("capture スレッドが生きている", vl.capture_thread._thread is not None)

    # ---------------------------------------------------------------- タスク
    print("\n[3] タスク OFF → ON")
    caps = vl.capabilities
    if not vl.capabilities.has("delegate_task"):
        check("TASK_ENABLED=1 で起動している", False, "→ .env を確認（この項目は飛ばす）")
    else:
        before = set(caps.names())
        task_bundle = sorted(n for n in before if n in (
            "delegate_task", "cancel_task", "list_tasks", "search_web",
            "write_file", "read_file", "list_files", "delete_file",
            "run_program", "show_on_screen", "close_window"))
        check("タスク束を特定できた", len(task_bundle) >= 3, f"{task_bundle}")
        check("OFF 前は応答LLM に delegate_task が見えている",
              any(s["function"]["name"] == "delegate_task" for s in caps.tool_schemas()))
        n_actions = len(caps.outward_actions())

        # サイドカー停止 + 能力の取り外し
        await vl.task_executor.stop()
        await vl.reconcile_timer.stop()
        await vl.cancel_resolver.stop()
        removed = caps.unregister(*task_bundle)
        check("能力を取り外せた", sorted(removed) == task_bundle, f"{removed}")
        check("応答LLM から delegate_task が消えた",
              all(s["function"]["name"] != "delegate_task" for s in caps.tool_schemas()))
        check("「できること」一覧も自動で縮んだ",
              len(caps.outward_actions()) < n_actions,
              f"{n_actions} → {len(caps.outward_actions())}")
        check("タスクのサイドカーが止まっている",
              vl.task_executor._task is None and vl.reconcile_timer._task is None
              and vl.cancel_resolver._task is None)

        # 再登録（本番 `voice_loop.__init__` と**同じ順序・同じ組み合わせ**で組み直す）。
        # 登録関数は5本あり、フラグごとに分かれている。1本でも忘れると能力が欠けたまま
        # 復帰する（初回の実測でまさに run_program / show_on_screen / close_window を
        # 取りこぼした）。P4 の ControlPlane は「登録手順そのもの」を持つ必要がある。
        from eve.config import Config
        from eve.task import register_task_capabilities
        register_task_capabilities(caps, vl.task_store,
                                   cancel_resolver=vl.cancel_resolver, queue=vl.queue)
        if vl.search_client is not None:
            from eve.search import register_search_capability
            register_search_capability(caps, vl.search_client)
        if vl.workspace is not None:
            from eve.workspace import (
                register_exec_capability, register_viewer_capability,
                register_workspace_capabilities,
            )
            register_workspace_capabilities(caps, vl.workspace)
            if Config.WS_EXEC_ENABLED:
                register_exec_capability(
                    caps, vl.workspace, allow=Config.WS_EXEC_ALLOW.split(","),
                    timeout_sec=Config.WS_RUN_TIMEOUT_SEC, grace_sec=Config.WS_RUN_GRACE_SEC,
                    mem_mb=Config.WS_RUN_MEM_MB, max_procs=Config.WS_RUN_MAX_PROCS,
                    max_output_chars=Config.WS_MAX_OUTPUT_CHARS)
            if Config.WS_VIEWER_ENABLED:
                register_viewer_capability(caps, vl.workspace,
                                           max_windows=Config.WS_VIEWER_MAX_WINDOWS)
        vl.task_executor.start()
        vl.reconcile_timer.start()
        vl.cancel_resolver.start()
        await asyncio.sleep(1.0)
        check("ON に戻すと delegate_task が復活する",
              any(s["function"]["name"] == "delegate_task" for s in caps.tool_schemas()))
        check("サイドカーが再び動いている",
              vl.task_executor._task is not None and vl.reconcile_timer._task is not None
              and vl.cancel_resolver._task is not None)
        check("取り外した能力が全部戻った", set(caps.names()) >= before,
              f"欠け: {before - set(caps.names())}")

    # ---------------------------------------------------------------- 停止/再開（マイク）
    print("\n[4] 停止/再開（マイク）")
    ai_before = vl.input._ai
    check("マイクが動いている", ai_before is not None and ai_before.running)
    vl.input.stop()
    await asyncio.sleep(1.0)
    check("停止でマイクが止まる", not ai_before.running, f"running={ai_before.running}")
    await vl.input.start()
    await asyncio.sleep(1.0)
    check("再開でマイクが動く", vl.input._ai is not None and vl.input._ai.running)
    check("AudioInput は作り直されている（terminate 済みを使い回さない）",
          vl.input._ai is not ai_before)

    # ---------------------------------------------------------------- 後片付け
    print("\n終了処理（drain）…")
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    await vl.stop()

    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
