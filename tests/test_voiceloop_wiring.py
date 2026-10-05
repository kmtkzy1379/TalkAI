"""VoiceLoop の組み立て(配線)スモークテスト（API/モデル/音声デバイス不要）。

VoiceLoop() の __init__ は全 backend を遅延 import で構築する（torch/pyaudio/network なしで構築可）。
これは「ロジックテストが全緑でも VoiceLoop のコンストラクタ引数ドリフトで“実起動だけ”壊れる」
（監査 Gap1）を防ぐ最小ガード。run() は呼ばない＝構築と結線のみ検証する。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_voiceloop_wiring.py
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.voice_loop import VoiceLoop  # noqa: E402

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


try:
    vl = VoiceLoop()
    check("VoiceLoop 構築成功(遅延importのみ)", True)
    check("orchestrator に短期記憶を配線", vl.orchestrator._cache is vl.cache)
    check("orchestrator に長期記憶(RAG)を配線", vl.orchestrator._rag is vl.rag)
    # 応答プロンプト leak 修正: system プロンプトが空でなく SPEECH_STYLE を含む（"システム応答" leak 防止）
    from eve.response.style import SPEECH_STYLE as _SS  # noqa: E402
    check("orchestrator の system プロンプトが非空", bool(vl.orchestrator._ctx.system_prompt))
    # 2026-08-15: ペルソナ層（`persona.md`）を後付けするようになったので完全一致ではなくなった。
    # 守りたいのは「SPEECH_STYLE が先頭にある（leak 防止）」＋「ペルソナが載る／無くても壊れない」。
    _sp = vl.orchestrator._ctx.system_prompt
    check("orchestrator の system が SPEECH_STYLE で始まる", _sp.startswith(_SS))
    from eve.persona import PERSONA_HEADING, load_persona  # noqa: E402
    _persona = load_persona()
    check("ペルソナがあれば後ろに載る／無ければ SPEECH_STYLE のまま",
          (PERSONA_HEADING in _sp) if _persona else (_sp == _SS))
    # F4: 内分泌系（PredictionState / FeedbackLLM / FeedbackWorker）の配線
    check("orchestrator に PredictionState を配線", vl.orchestrator._state is vl.prediction)
    check("orchestrator 完了トリガ=_on_response_complete(feedback+沈黙リセット)", vl.orchestrator._on_complete == vl._on_response_complete)
    check("FeedbackLLM に RAG/PredictionState を配線", vl.feedback._rag is vl.rag and vl.feedback._state is vl.prediction)
    check("FeedbackWorker に feedback/cache を配線", vl.feedback_worker._fb is vl.feedback and vl.feedback_worker._cache is vl.cache)
    # F5: 発話判定（SpeechState / SpeechDecider / SilenceMonitor）の配線
    check("SpeechDecider に state/queue/prediction を配線",
          vl.speech_decider._state is vl.speech_state and vl.speech_decider._queue is vl.queue and vl.speech_decider._pred is vl.prediction)
    check("SilenceMonitor に state/decider を配線",
          vl.silence_monitor._state is vl.speech_state and vl.silence_monitor._decider is vl.speech_decider)
    # S3: ガードは「応答中 or フィラー中」（フィラー既定 off なら runner.is_busy と同じ値）
    check("SilenceMonitor の busy ガード=応答中 or フィラー中",
          vl.silence_monitor._is_busy == vl._busy_for_autospeech
          and vl._busy_for_autospeech() == vl.runner.is_busy())
    check("input に発話到着フック(沈黙時計リセット)を配線", vl.input._on_utterance == vl.speech_state.mark_user_utterance)
    check("runner に orchestrator を配線", vl.runner._orch is vl.orchestrator)
    check("runner と input が同じ queue を共有", vl.runner._queue is vl.queue and vl.input._queue is vl.queue)
    check("input に barge-in callback を結線", callable(vl.input._on_speech_start))
    # F6: 画面認識(VLM)の配線
    check("orchestrator に VisionState を配線", vl.orchestrator._vision_state is vl.vision_state)
    check("SpeechDecider に VisionState を配線", vl.speech_decider._vision_state is vl.vision_state)
    check("VlmWorker に vision_state/prediction を配線",
          vl.vlm_worker._vs is vl.vision_state and vl.vlm_worker._pred is vl.prediction)
    # J-2 ③: trigger は source ラベル付き lambda 経由（同一性でなく挙動＝decider の event が立つこと）
    vl.speech_decider._event.clear()
    vl.vlm_worker._speak_trigger()
    check("VlmWorker の発話トリガが speech_decider を起こす", vl.speech_decider._event.is_set())
    vl.speech_decider._event.clear()  # 後続チェックへの汚染防止
    check("VLM 既定 off では capture スレッド未生成(構築は可)", vl.capture_thread is None)
    # 話している途中にイブの音が鳴り始めたら最初の節で譲る（割り込みのかけ直し・2026-09-26）
    check("再生開始の通知に『話している途中なら譲る』を配線",
          vl._yield_if_user_speaking in vl.audio._start_listeners)
    # リップシンク: 実起動では (WAV, 口の開閉計画) を返す TTS が刺さっていること。
    # ここが `tts.generate` に戻ると、UI は口を動かせないのに Tier-1 は全部 PASS する。
    check("tts_fn が generate_with_meta（リップシンク計画つき）",
          vl.orchestrator._tts_fn == vl.tts.generate_with_meta)
    check("再生開始/終了フックは既定 None（UI 未接続では既存挙動が不変）",
          vl.audio.on_sentence_start is None and vl.audio.on_sentence_end is None)
    # J-1/Fix#2: 予約タスク状態注入の配線（TASK_ENABLED に連動）
    from eve.config import Config as _Cfg  # noqa: E402
    check("tasks_provider の配線が TASK_ENABLED に連動",
          (vl.orchestrator._tasks_provider is not None) == bool(_Cfg.TASK_ENABLED))
    _prev_task_enabled = _Cfg.TASK_ENABLED
    _prev_search_enabled = _Cfg.SEARCH_ENABLED
    _prev_ws_enabled = _Cfg.WS_ENABLED
    _prev_ws_root = _Cfg.WS_ROOT
    _Cfg.TASK_ENABLED = True
    _Cfg.SEARCH_ENABLED = True
    # フラグは**明示的に落とす**。既定値を .env から読むと、ユーザが WS_ENABLED=1 を書いた
    # 瞬間にこのテストが落ちる（実際に落ちた）。テストは環境ではなく自分で条件を作る。
    _Cfg.WS_ENABLED = False
    try:
        vl2 = VoiceLoop()
        check("TASK_ENABLED=True で tasks_provider 配線(空 store→[])",
              vl2.orchestrator._tasks_provider is not None and vl2.orchestrator._tasks_provider() == [])
        # J-2: search_web は TaskAgent 専用（agent_tool に出て offered に出ない）
        _agent_names = {s["function"]["name"] for s in vl2.capabilities.agent_tool_schemas()}
        _offered_names = {s["function"]["name"] for s in vl2.capabilities.tool_schemas()}
        check("SEARCH_ENABLED で search_web 配線(agent専用・応答LLM不可視)",
              vl2.search_client is not None and "search_web" in _agent_names
              and "search_web" not in _offered_names)
        check("WS_ENABLED=0 では作業フォルダ未配線", vl2.workspace is None
              and "write_file" not in _agent_names)

        # J-3: 作業フォルダ（agent 専用・**構築時にディレクトリを作らない**＝テストが開発機を汚さない）
        import tempfile as _tf
        _Cfg.WS_ENABLED = True
        _Cfg.WS_ROOT = os.path.join(_tf.mkdtemp(prefix="eve_wsw_"), "workspace")
        vl3 = VoiceLoop()
        _a3 = {s["function"]["name"] for s in vl3.capabilities.agent_tool_schemas()}
        _o3 = {s["function"]["name"] for s in vl3.capabilities.tool_schemas()}
        _ws_names = {"write_file", "read_file", "list_files", "delete_file"}
        check("WS_ENABLED で作業フォルダ配線(agent専用・応答LLM不可視)",
              vl3.workspace is not None and _ws_names <= _a3 and not (_ws_names & _o3))
        check("WS は構築時にディレクトリを作らない（起動時 ensure_ready で作る）",
              not os.path.exists(_Cfg.WS_ROOT))

        # 場所ゲート: リポジトリ配下は fail-soft で無効化（起動は続行する）
        _Cfg.WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vl4 = VoiceLoop()
        check("WS: リポジトリ配下は無効化して起動継続（fail-soft）",
              vl4.workspace is None
              and "write_file" not in {s["function"]["name"] for s in vl4.capabilities.agent_tool_schemas()})

        # TASK_ENABLED=0 では構造的に登録されない（誰も呼べない状態を作らない＝search と同じ規律）
        _Cfg.TASK_ENABLED = False
        _Cfg.WS_ROOT = os.path.join(_tf.mkdtemp(prefix="eve_wsw_"), "workspace")
        vl5 = VoiceLoop()
        check("WS_ENABLED=1 でも TASK_ENABLED=0 なら未登録",
              vl5.workspace is None
              and "write_file" not in {s["function"]["name"] for s in vl5.capabilities.agent_tool_schemas()})
    finally:
        _Cfg.TASK_ENABLED = _prev_task_enabled
        _Cfg.SEARCH_ENABLED = _prev_search_enabled
        _Cfg.WS_ENABLED = _prev_ws_enabled
        _Cfg.WS_ROOT = _prev_ws_root
    # 再配達（barge-in で潰れた機能報告の救済）の配線と WHEN ポリシー実挙動
    check("orchestrator に再配達 fn を配線", vl.orchestrator._redeliver_fn == vl._redeliver_stimulus)
    # D2: 抑止判定は同一 queue の is_suppressed でなければならない。resolver 側に hasattr
    # フォールバックを置かず直接呼ぶ設計なので、配線切れはここで落として気づく。
    check("orchestrator に抑止判定(is_suppressed)を配線",
          vl.orchestrator._is_suppressed == vl.queue.is_suppressed)
    check("CancelResolver が runner と同じ queue を持つ",
          vl.cancel_resolver is None or vl.cancel_resolver._queue is vl.queue)

    # 応答LLM の reasoning_effort（2026-09-29 ルナ採用・gpt-6-luna はツール付きだと none 必須＝実測）が設定どおり渡る
    import asyncio as _aio0  # noqa: E402

    _prev_eff = _Cfg.RESPONSE_REASONING_EFFORT
    _got: list = []

    async def _fake_stream(**kw):
        _got.append(kw)
        yield "はい。"

    async def _drive(v, tools) -> list:
        msgs = [{"role": "user", "content": "x"}]
        gen = (v.orchestrator._stream_fn(msgs, tools=tools, tool_sink=[]) if tools
               else v.orchestrator._stream_fn(msgs))
        return [d async for d in gen]

    try:
        _Cfg.RESPONSE_REASONING_EFFORT = "none"
        v6 = VoiceLoop()
        v6.registry._stream_fn = _fake_stream
        _aio0.run(_drive(v6, [{"type": "function", "function": {"name": "delegate_task"}}]))
        _aio0.run(_drive(v6, None))
        check("応答LLM: RESPONSE_REASONING_EFFORT=none をツール有り/無しの両方で渡す",
              len(_got) == 2 and all(k.get("reasoning_effort") == "none"
                                     and k.get("allowed_openai_params") == ["reasoning_effort"] for k in _got))
        _got.clear()
        _Cfg.RESPONSE_REASONING_EFFORT = ""
        v7 = VoiceLoop()
        v7.registry._stream_fn = _fake_stream
        _aio0.run(_drive(v7, None))
        check("応答LLM: 空なら reasoning_effort を渡さない（モデルの既定）",
              len(_got) == 1 and "reasoning_effort" not in _got[0])
    finally:
        _Cfg.RESPONSE_REASONING_EFFORT = _prev_eff

    import asyncio as _aio  # noqa: E402
    from eve.pipeline.stimulus import CallFunctionResult as _CFR, Stimulus as _St, StimulusKind as _SK  # noqa: E402

    async def _w2() -> bool:
        vl._redeliver_grace_sec = 0.3  # テスト短縮（インスタンス属性）
        stim = _St(_SK.CALLFUNCTION_RESULT, _CFR("goal", "x", True, 1), dedup_key="task:w2")
        vl.speech_state.mark_user_speech_start()
        vl._redeliver_stimulus(stim, lambda: False)
        await _aio.sleep(0.5)
        during_speech = vl.queue.qsize()  # 発話中は投入されない
        vl.speech_state.mark_user_utterance()
        await _aio.sleep(0.6)  # 猶予 0.3s 経過後に投入
        after = vl.queue.qsize()
        # abort=True の再配達は投入されない（put 直前の最終判定）
        vl._redeliver_stimulus(_St(_SK.CALLFUNCTION_RESULT, _CFR("goal", "y", True, 1), dedup_key="task:w3"),
                               lambda: True)
        await _aio.sleep(0.5)
        aborted = vl.queue.qsize()
        return during_speech == 0 and after == 1 and aborted == 1 and len(vl._redeliver_waiters) == 0

    check("再配達WHEN: 発話中は待機→解除+猶予後に投入 / abort で中止 / waiter 自己除去", _aio.run(_w2()))
except Exception as e:  # 構築自体が落ちたら配線ドリフト＝即失敗
    check(f"VoiceLoop 構築で例外: {e!r}", False)

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
