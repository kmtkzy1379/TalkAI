"""ControlPlane — 機能の ON/OFF を**安全な地点で**切り替える（UI メニューの実体）。

## なぜ独立した層にするか
UI から直接サイドカーを止めると、止める順序・待ち・巻き戻しが UI に散る。散ると
「タスクだけ止めて検索が残る」「停止したのに喋る」のような組み合わせ事故が、UI の
コードを読んでも見つけられなくなる。ここに1本化して、**UI は「押された」を伝えるだけ**、
**表示は loop が確定させた状態だけ**にする（嘘の状態を作らない）。ヘッドレス
（`tools/voice_chat.py`）は本モジュールを一切使わない＝既存の挙動は不変。

## スレッド契約（`docs/PIPELINE_DESIGN.md` §9.1/§9.3）
`request()` を含め**すべて loop 上**で動く。UI スレッドからは
`loop.call_soon_threadsafe(control.request, key, on)` で入る。`on_state` は loop 上から
同期で呼ばれるので、UI 側は Qt signal を emit するだけで戻ること。

## 3つの規律
1. **作り直さない**: VoiceLoop / StimulusQueue / AudioPlayQueue / PipelineRunner は
   絶対に再生成しない。止めるのは「入口」と「サイドカー」だけ。runner を作り直すと
   consumer が二重化し、刺激の取りこぼしと二重発話が起きる。
2. **直列化**: 遷移は同時に1つ。stop の最中に start が来ると、サイドカーの
   `start()` は `_task is None` を見て黙って何もしない＝「ON にしたのに動かない」に
   なる。意図は latest-wins で覚えておき、1つずつ順に適用する。
3. **押した瞬間には切り替えない**: 各機能が正常終了した時点ではじめて確定を通知する
   （ユーザ要件）。待っている間は `pending` を出し、**何を待っているか**も出す。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from .config import Config
from .pipeline.stimulus import StimulusKind
from .task.schema import CANCELLED, PENDING, RUNNING

logger = logging.getLogger(__name__)

KEYS = ("pause", "mute", "autospeech", "vision", "task")

# 応答LLM に見せる「今はオフ」の説明。**機能の不在ではなく一時的な停止**として書く
# （「画面を見る機能がない」はハルシネーションに見える＝ユーザ裁定 2026-08-14）。
FEATURE_OFF_TEXT = {
    "vision": "画面を見る機能（今はユーザがオフにしている。オンにすれば今の画面が見える）",
    "task": "調べ物や作業を引き受ける機能（今はユーザがオフにしている。オンにすれば頼まれごとができる）",
    "autospeech": "自分から話しかける機能（今はユーザがオフにしている。聞かれたら答えるのは普通にできる）",
    "mute": "声を出す機能（今はミュート。話した内容は画面のログには出ている）",
}

# 待ちの上限[秒]。超えても**強制終了はしない**（正常終了させる方針）。
# 上限まで来たら「まだ終わっていないが切り替えは確定させる」に倒し、何が残っているかを出す。
DEFAULT_QUIESCE_SEC = 90.0
_POLL_SEC = 0.2


class ControlPlane:
    def __init__(self, vl, *, on_state: Optional[Callable[[str, str, bool], None]] = None,
                 quiesce_sec: Optional[float] = None) -> None:
        self._vl = vl
        # fn(key, state, on): state は "on" / "off" / "pending" / "disabled"
        self.on_state = on_state
        # fn(text): 待ちの途中経過（UI のログ用・任意）
        self.on_note: Optional[Callable[[str], None]] = None
        self._quiesce = float(quiesce_sec if quiesce_sec is not None else DEFAULT_QUIESCE_SEC)
        self._on: dict[str, bool] = {k: True for k in KEYS}
        self._on["mute"] = False  # ミュートだけは「オンにする＝黙らせる」なので初期は解除
        self._want: dict[str, bool] = {}
        self._wake = asyncio.Event()
        # 「積まれた要求が全部片付いた」印（他サイドカーの `_idle` と同じ流儀）。
        # **適用中も clear のまま**にする＝`_want` が空でも「まだ切り替わっていない」を表せる。
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker: Optional[asyncio.Task] = None
        # 構成で最初から使えない機能は押させない（`disabled`）。
        self._available: dict[str, bool] = {
            "pause": True,
            "mute": True,
            "autospeech": True,
            "vision": bool(Config.VLM_ENABLED),
            "task": bool(Config.TASK_ENABLED and vl.task_store is not None),
        }
        for k in KEYS:
            if not self._available[k]:
                self._on[k] = False

    # --- 状態の公開 -------------------------------------------------------

    def is_on(self, key: str) -> bool:
        return bool(self._on.get(key, False))

    def features_off(self) -> list[str]:
        """応答LLM の system に入れる「今オフの機能」。ミュートは pause と重複しないよう単独で見る。"""
        off = []
        for key in ("vision", "task", "autospeech"):
            if self._available[key] and not self._on[key]:
                off.append(FEATURE_OFF_TEXT[key])
        if self._on["mute"]:
            off.append(FEATURE_OFF_TEXT["mute"])
        return off

    def snapshot(self) -> dict[str, tuple[str, bool]]:
        """UI の初期表示用（key -> (state, on)）。"""
        return {k: ("disabled" if not self._available[k] else ("on" if self._on[k] else "off"),
                    self._on[k]) for k in KEYS}

    # --- 入口 -------------------------------------------------------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            w.cancel()
            try:
                await w
            except asyncio.CancelledError:
                pass

    def request(self, key: str, on: bool) -> None:
        """UI から「これを ON/OFF にしたい」。**loop 上で呼ぶこと**（O(1)・待たない）。"""
        if key not in KEYS:
            return
        if not self._available[key]:
            self._emit(key, "disabled")
            return
        self._want[key] = bool(on)
        self._idle.clear()
        self._wake.set()
        self._emit(key, "pending")

    def is_settling(self) -> bool:
        """まだ切り替えの途中か（表示や終了処理から見る）。"""
        return not self._idle.is_set()

    async def wait_idle(self) -> None:
        """積まれた切り替えが全部確定するまで待つ。"""
        await self._idle.wait()

    # --- 遷移の直列実行 ---------------------------------------------------

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            # latest-wins: 待っている間に何度押されても、最後の意図だけを適用する。
            try:
                await self._drain()
            finally:
                self._idle.set()

    async def _drain(self) -> None:
        """積まれた意図を**1つずつ順に**適用する（同時に2つ走らせない）。"""
        while self._want:
            key = next(iter(self._want))
            want = self._want.pop(key)
            if want == self._on[key]:
                self._emit(key, "on" if self._on[key] else "off")
                continue
            try:
                await self._apply(key, want)
                self._on[key] = want
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("機能の切り替えに失敗: %s -> %s", key, want)
                self._note(f"{key} の切り替えに失敗しました（元の状態のまま）")
            self._emit(key, "on" if self._on[key] else "off")

    async def _apply(self, key: str, on: bool) -> None:
        handler = {
            "mute": self._apply_mute,
            "pause": self._apply_pause,
            "autospeech": self._apply_autospeech,
            "vision": self._apply_vision,
            "task": self._apply_task,
        }[key]
        await handler(on)

    # --- 各機能 -----------------------------------------------------------

    async def _apply_mute(self, on: bool) -> None:
        """即座に黙る/戻す。**待たない**（これが「速攻黙る」手段そのもの）。

        止めるのは音だけ。世代を進めない＝進行中ターンの残りの文も普通に流れ、記憶にも
        ログにも残る（ユーザは読める）。generation を進めると barge-in と同じになり、
        「正常終了させてから」の約束を壊す。
        """
        self._vl.player.muted = bool(on)
        self._note("ミュートしました（喋った内容はログに出ます）" if on else "ミュートを解除しました")

    async def _apply_pause(self, on: bool) -> None:
        """on=True が「稼働」。OFF にする＝停止。

        停止の順序（音の出どころを上流から順に断つ→進行中を待つ）:
          1. runner の入口を閉じる（新しい刺激を取り出さない）
          2. マイクを閉じる（新しいユーザ発話が入らない）
          3. 自発発話の駆動を止める（沈黙監視 + 判定 worker）
          4. 画面認識を止める（判定 worker を叩く経路なので、3 の後に止める）
          5. **進行中の応答と再生が終わるまで待つ**（殺さない）
        タスクは止めない: 走っているものは正常に終わらせ、完了報告は入口が閉じているので
        キューで待つ（再開時に配達される）。殺すと「正常終了させてから」を破る。
        """
        vl = self._vl
        if on:
            # 再開
            vl.runner.resume()
            if self._on["vision"] and self._available["vision"]:
                vl.start_vision()
            if self._on["autospeech"]:
                # 沈黙時計をリセットしてから起こす。停止していた時間ぶん「沈黙が続いた」
                # ことになっており、再開の瞬間に独り言が飛び出す。
                vl.speech_state.mark_eve_activity()
                vl.speech_decider.start()
                vl.silence_monitor.start()
            await vl.input.start()
            self._note("再開しました")
            return

        vl.runner.pause()
        vl.input.stop()
        await self._stop_autospeech()
        if self._on["vision"] and self._available["vision"]:
            await vl.stop_vision()
        left = await self._wait_quiet(response=True, audio=True, task=False)
        self._note("停止しました" if not left else f"停止しました（まだ動いているもの: {left}）")

    async def _apply_autospeech(self, on: bool) -> None:
        vl = self._vl
        if on:
            vl.speech_state.mark_eve_activity()  # OFF 中の沈黙で即発火させない
            vl.speech_decider.start()
            vl.silence_monitor.start()
            self._note("自律発話を ON にしました")
            return
        await self._stop_autospeech()
        # 判定は通ったが**まだ喋っていない**下書きを捨てる。捨てないと OFF 直後に1回喋る。
        n = vl.queue.discard_kind(StimulusKind.AUTONOMOUS_SPEECH)
        self._note(f"自律発話を OFF にしました（未発話の下書き {n} 件を破棄）" if n
                   else "自律発話を OFF にしました")

    async def _stop_autospeech(self) -> None:
        """沈黙監視 → 判定 worker の順で止める（新規トリガを断ってから drain）。

        **両方止める**のが要点。判定 worker を残すと、画面認識の `speak_trigger`
        （`voice_loop.py` の `speak_trigger=lambda: self.speech_decider.trigger("vlm")`）
        経由で自発発話が出続ける。止めた後の `trigger()` は Event を立てるだけで
        何も起こさず、`start()` が clear するので副作用も残らない。
        """
        vl = self._vl
        try:
            await vl.silence_monitor.stop()
        except Exception:
            logger.exception("沈黙監視の停止で例外（続行）")
        try:
            await vl.speech_decider.stop()
        except Exception:
            logger.exception("発話判定 worker の停止で例外（続行）")

    async def _apply_vision(self, on: bool) -> None:
        vl = self._vl
        if on:
            vl.start_vision()
            self._note("画面認識を ON にしました")
            return
        await vl.stop_vision()
        self._note("画面認識を OFF にしました（今の画面の記憶も消しました）")

    async def _apply_task(self, on: bool) -> None:
        """タスク OFF は**キャンセルを走らせて正常終了させてから**止める（ユーザ裁定）。

        順序:
          1. 能力を外す（`delegate_task` が消える＝新しい依頼を受け付けなくなる）
          2. 残っているタスクを Cancelled にする（TaskAgent は毎 step / 各 tool 呼び出しの
             前後で Cancelled を見て中断し、結果を破棄する＝`task/agent.py`）
          3. executor が idle になるまで待つ
          4. 取消解決 → 再照合タイマ → executor の順に停止（`voice_loop.stop()` と同順）
          5. 取り消したタスクの完了報告がキューに残っていれば捨てる（もう報告する意味がない）
          6. 開いた表示ウィンドウ等を掃除する
        """
        vl = self._vl
        if on:
            for fn in vl._task_cap_registrars:
                try:
                    fn()
                except Exception:
                    logger.exception("タスク能力の再登録で例外（続行）")
            if vl.task_executor is not None:
                vl.task_executor.start()
            if vl.reconcile_timer is not None:
                vl.reconcile_timer.start()
            if vl.cancel_resolver is not None:
                vl.cancel_resolver.start()
            self._note("タスクを ON にしました")
            return

        removed = vl.capabilities.unregister(*vl._task_cap_names)
        logger.info("タスク能力を外した: %s", removed)
        n = self._cancel_actives()
        if n:
            self._note(f"実行中/予約中のタスク {n} 件を取り消しています…")
        # 応答と再生も待つ。1件制の定義は「依頼を受けてから**報告を発話し終えるまで**」で、
        # 報告は `PipelineRunner` が pop した瞬間にキューから消える＝喋っている最中は
        # キューを見ても分からない。ここで待たないと、報告の途中でタスクを OFF にできてしまう。
        left = await self._wait_quiet(response=True, audio=True, task=True)
        if vl.cancel_resolver is not None:
            try:
                await vl.cancel_resolver.stop()
            except Exception:
                logger.exception("取消解決の停止で例外（続行）")
        if vl.reconcile_timer is not None:
            try:
                await vl.reconcile_timer.stop()
            except Exception:
                logger.exception("再照合タイマの停止で例外（続行）")
        if vl.task_executor is not None:
            try:
                await vl.task_executor.stop()
            except Exception:
                logger.exception("タスク実行の停止で例外（続行）")
        # 取り消したタスクの報告は口に出さない（「終わったよ」と言われても中身は破棄済み）。
        dropped = vl.queue.discard_kind(StimulusKind.CALLFUNCTION_RESULT)
        # 掃除は "shutdown" 相当（`.eve_run` の残骸とタスクの取りこぼし）。**startup は使わない**:
        # あちらは「マーカを持つビューアを全部殺す」なので、稼働中に呼ぶと無関係のプロセスまで畳む。
        try:
            from .cleanup import sweep as cleanup_sweep

            swept = cleanup_sweep(task_store=vl.task_store, workspace=vl.workspace, phase="shutdown")
            if any(swept.values()):
                logger.info("🧹 タスク OFF の掃除: %s", swept)
        except Exception:
            logger.exception("タスク OFF の掃除で例外（続行）")
        # イブが開いた表示ウィンドウは畳む。残すと E2 の再発（自分の出力を「ユーザの画面」
        # として画面認識が拾い、自分の過去の出力を相手の作業画面として実況する）。
        if vl.workspace is not None:
            try:
                closed = vl.workspace.close_viewers()
                if closed:
                    logger.info("表示ウィンドウを %d 個閉じた（タスク OFF）", closed)
            except Exception:
                logger.exception("表示ウィンドウを閉じるのに失敗（続行）")
        msg = "タスクを OFF にしました"
        if dropped:
            msg += f"（未配達の報告 {dropped} 件を破棄）"
        if left:
            msg += f"（まだ終わっていないもの: {left}）"
        self._note(msg)

    def _cancel_actives(self) -> int:
        """アクティブ（Pending/Running）を Cancelled にする。件数を返す。

        Running を Cancelled にできるのが要点（`task/store.py` の `set_status` は
        **今が terminal の時だけ** no-op なので Running→Cancelled は通る）。TaskAgent は
        毎 step と各 tool 呼び出しの前後で Cancelled を見て中断し、結果を破棄する。
        """
        store = self._vl.task_store
        if store is None:
            return 0
        n = 0
        for t in list(store.list_all()):
            if t.status in (PENDING, RUNNING) and store.set_status(t.task_id, CANCELLED):
                n += 1
        return n

    # --- 待ち --------------------------------------------------------------

    async def _wait_quiet(self, *, response: bool, audio: bool, task: bool) -> str:
        """指定したものが落ち着くまで待つ。上限で諦め、**残っていたものの名前**を返す。

        殺さないのが要点（「正常終了させてから」）。上限まで待っても終わらないものは、
        そのまま走らせたうえで切り替えだけ確定させ、何が残っているかを正直に出す。
        """
        vl = self._vl
        deadline = asyncio.get_running_loop().time() + self._quiesce
        last_note = 0.0
        while True:
            busy: list[str] = []
            if response and vl.runner.is_busy():
                busy.append("応答")
            if audio and not self._audio_quiet():
                busy.append("再生")
            if task and vl.task_executor is not None and not vl.task_executor.is_idle():
                busy.append("タスク")
            if not busy:
                return ""
            now = asyncio.get_running_loop().time()
            if now >= deadline:
                return "・".join(busy)
            if now - last_note >= 5.0:  # 待ちが長い時だけ、何を待っているかを出す
                last_note = now
                self._note(f"{'・'.join(busy)}が終わるのを待っています…")
            await asyncio.sleep(_POLL_SEC)

    def _audio_quiet(self) -> bool:
        """再生待ちが残っていないか（`join()` は待ち切りなので使わない）。"""
        try:
            return self._vl.audio.is_quiet()
        except Exception:
            return True

    # --- 通知 --------------------------------------------------------------

    def _emit(self, key: str, state: str) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(key, state, self._on[key])
        except Exception:
            logger.exception("状態通知で例外（切り替え自体は続行）")

    def _note(self, text: str) -> None:
        logger.info("🎛 %s", text)
        if self.on_note is None:
            return
        try:
            self.on_note(text)
        except Exception:
            logger.exception("経過通知で例外（切り替え自体は続行）")
