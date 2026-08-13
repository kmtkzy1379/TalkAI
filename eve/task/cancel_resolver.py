"""CancelResolver — タスク取消を**タスク側**で解決するサイドカー（dispatcher の近クローン・別コルーチン）。

応答LLM は `cancel_task(reference)` で取消意図（ユーザの言い回し）を渡すだけ。**1件制
（2026-08-07 ユーザ裁定・`task/capabilities.py` の受付ゲート）でアクティブは高々1件**なので、
ここは reference を照合せず件数だけで解決する（LLM を呼ばない＝速く確実）:
- **アクティブ0件** → 未配達の報告があればそれを抑止、無ければ「今止めるタスクは無いよ」。
- **アクティブ1件** → 即キャンセル。reference がどれを指すか迷う余地が構造的に無い。

**なぜ照合をやめたか**（旧実装はタスクLLM でファジー照合していた）:
1件しか無い以上ユーザが対象を取り違えようがなく、言い間違えても「消す意図」は同じ。
2026-07-13 実機事故はユーザの勘違いではなく**応答LLM が完了済みターンを再実行して cancel を
撃ち直した**もので、これは照合ではなく時間差で弾ける（`capabilities.py` SELF_CANCEL_GUARD_SEC）。
照合ロジックは実害を出した実績もある（E4 差し戻し `6cb6f25`: 日常的な取消表現 9/9 をブロック）。

**executor(single-flight) 経由にしない**＝取消が実行中タスクの後ろで待つのを防ぐ（別コルーチンで並行）。
実行中 goal は agent の毎step CANCELLED チェックが中断・破棄する（agent/executor 側で既存）。

**TOCTOU**: await の前後で状態が変わり得る→**書込直前に store を読み直し**、`set_status` の戻り値 bool で
分岐（False=既に終了→「もう終わってた」）。スナップショットから「取り消した」と誤報しない。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from ..pipeline.stimulus import CallFunctionResult, Stimulus, StimulusKind
from .schema import CANCELLED, DONE, FAILED, PENDING, RUNNING

logger = logging.getLogger(__name__)

# 「ついさっき終わった」とみなす窓（キューに実体が無い＝配達済み/再配達待ちの場合の保険用）。
# 未配達の報告はキュー実体で一意に決まるのでこの窓に依存しない（_undelivered_candidates）。
RECENT_TERMINAL_SEC = 90.0


@dataclass
class CancelRequest:
    reference: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


def _display_name(t) -> str:
    return t.goal or t.what or "(無題)"


def undelivered_reports(store, queue) -> list:
    """**まだ発話されていない完了報告**を持つ terminal タスク（この定義の唯一の実装）。

    時計だけで「直近の terminal」を選ぶと実機で誤る: 2026-07-29 E2E の D2 時点では 90秒窓に
    3件（東京タワー56.2s前 / 琵琶湖46.5s前 / 川1.9s前）あったが、先の2件は既に発話済みで、
    未配達はユーザが指した川の1件だけだった。キューの実体と突き合わせれば候補は一意に決まる。
    起動時 age-out の孤児（store が updated_at を打ち直す）も put を出さないので自動的に候補外。

    1件制の受付ゲート（`task/capabilities.py`）もこれを使う＝「未配達」の定義を二重に持たない。
    """
    queued = {s.dedup_key for s in queue.snapshot() if (s.dedup_key or "").startswith("task:")}
    return [t for t in store.list_all()
            if t.status in (DONE, FAILED) and f"task:{t.task_id}" in queued]


class CancelResolver:
    def __init__(
        self, *, store, queue,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._queue = queue
        self._now = now_fn  # TaskStore と同じシグネチャ（決定論テストで注入する）
        self._inbox: "asyncio.Queue[CancelRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # --- ライフサイクル（dispatcher と同形）---
    def start(self) -> None:
        if self._task is None:
            self._stopping = False  # 再開可能にする（stop() は立てたままにする）
            self._task = asyncio.create_task(self._run())

    async def stop(self, drain_timeout: float = 3.0) -> None:
        self._stopping = True
        try:
            await asyncio.wait_for(self._inbox.join(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def submit(self, reference: str) -> None:
        """取消意図(ユーザの言い回し)を積むだけ（O(1)・非ブロッキング）。"""
        self._inbox.put_nowait(CancelRequest(reference=reference or ""))

    async def _run(self) -> None:
        while not self._stopping:
            req = await self._inbox.get()
            try:
                await self._handle_one(req)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("キャンセル解決で例外（フォールバック報告）")
                await self._report(req, "ごめん、取り消しの確認でエラーが起きたよ。", ok=False)
            finally:
                self._inbox.task_done()

    def _actives(self):
        return [t for t in self._store.list_all() if t.status in (PENDING, RUNNING)]

    # --- terminal 直後の取消（D2: 完了が取消要求を追い越すレース）-------------

    def _terminal_age_sec(self, t) -> Optional[float]:
        """terminal タスクの経過秒。updated_at が空/壊れていれば None（＝候補に含めない）。

        `clock.elapsed_wall` は壊れた ISO を 0.0（＝たった今）へ倒す設計（応答を落とさないため）
        だが、ここでその既定を継ぐと「timestamp が壊れたタスクは必ず抑止する」ことになり、
        『迷ったら配達側に倒す』方針と真逆になるので明示的に弾く。
        """
        raw = getattr(t, "updated_at", None)
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (self._now() - dt).total_seconds()

    def _undelivered_candidates(self) -> list:
        return undelivered_reports(self._store, self._queue)

    def _recent_terminals(self) -> list:
        """キューに実体が無い terminal のうち「ついさっき終わった」もの（再配達の抑止用）。"""
        out = []
        for t in self._store.list_all():
            if t.status not in (DONE, FAILED):
                continue
            age = self._terminal_age_sec(t)
            if age is not None and 0 <= age <= RECENT_TERMINAL_SEC:
                out.append(t)
        return out

    async def _handle_terminal_race(self, req: CancelRequest) -> None:
        """アクティブが無い時に「ついさっき終わったタスクの報告」を止める（D2）。

        **この関数は最初の `await` までに suppress を終える**こと。resolver が起きてから
        runner が次の刺激を pop するまでの猶予は call_soon 1スロットしかない
        （実測プローブ: dispatcher→resolver 起床→suppress→runner pop の順）。
        ここに await を1つ挟むと報告が先に配達され、D2 が再発する。
        """
        queued = self._undelivered_candidates()  # ← 同期。await 禁止区間ここから
        if len(queued) == 1:
            t = queued[0]
            name = _display_name(t)
            removed = self._queue.suppress(f"task:{t.task_id}")
            logger.info("⚰ 報告を抑止: task=%s 「%.40s」 除去%d件", t.task_id, name, removed)
            # 「取り消した」「止めた」とは言わない: status は Done のままで、止めたのは報告だけ。
            await self._report(req, f"「{name}」はもう終わってたよ。結果はまだ言ってないから、"
                                    "このまま流さないでおくね。", ok=True)
            return
        if len(queued) >= 2:
            # 【ユーザ裁定】どれを止めるべきか特定できない → 抑止しない（誤破棄より誤配達に倒す）。
            await self._report(req, "ごめん、ちょうど終わったのがいくつかあって、どの報告のことか"
                                    "特定できないから、そのまま伝えるね。", ok=False)
            return
        recent = self._recent_terminals()  # キューに実体なし＝配達済み or 再配達待ち
        if len(recent) == 1:
            t = recent[0]
            name = _display_name(t)
            self._queue.suppress(f"task:{t.task_id}")  # 再配達（barge-in/配達確認）だけは止まる
            logger.info("⚰ 報告を抑止(再配達のみ): task=%s 「%.40s」", t.task_id, name)
            # 未来の約束をしない: 既に喋った/今喋っている可能性を排除できない（suppress の 0 は3義）。
            await self._report(req, f"「{name}」はもう終わってたよ。結果、もう伝えちゃってたかも。", ok=False)
            return
        if len(recent) >= 2:
            await self._report(req, "ごめん、それもう終わったあとだったみたい。", ok=False)
            return
        ref = req.reference.strip()
        msg = f"「{ref}」ってタスクは入ってないよ。今は止められる予約も無いよ。" if ref else "今は止められるタスクは無いよ。"
        await self._report(req, msg, ok=False)

    async def _handle_one(self, req: CancelRequest) -> None:
        actives = self._actives()
        # 観測性: 取消要求の入口を必ず1行残す（2026-07-13 実機事故はここの判断が無ログで追跡困難だった）。
        logger.info("🚫 取消要求 ref=%.60r actives=%d", req.reference, len(actives))
        if not actives:
            await self._handle_terminal_race(req)
            return
        # 1件制（2026-08-07 ユーザ裁定）: アクティブは高々1件＝reference がどれを指すか迷う余地が
        # 構造的に無いので、照合はしない。**タスクLLM 照合は撤去した**。理由:
        # - 1件しか無い以上ユーザが対象を取り違えようがなく、言い間違えても消す意図は同じ
        # - 2026-07-13 事故はユーザの勘違いではなく**応答LLM の cancel 撃ち直し**だった。
        #   これは照合ではなく時間差で弾ける（`task/capabilities.py` の SELF_CANCEL_GUARD_SEC）
        # - 照合の副作用が実害を出した実績がある（E4 差し戻し `6cb6f25`: 日常的な取消表現 9/9 を
        #   ブロックした）。往復が消えることで取消の不感時間（E3）も縮む
        if len(actives) > 1:
            # 1件制ゲートより前に作られた残留（旧バージョンの状態ファイル等）。落とさず最新を止める。
            logger.warning("1件制の不変条件違反: actives=%d（旧状態の残留?）→ 最新を取り消す", len(actives))
        await self._cancel_and_report(req, actives[-1].task_id)

    async def _cancel_and_report(self, req: CancelRequest, task_id: str, llm_message: str = "") -> None:
        # TOCTOU: 書込直前に読み直し、set_status の bool で成否判定（スナップショットから誤報しない）。
        t = self._store.get(task_id)
        if t is None:
            await self._report(req, "そのタスクはもう無くなってたよ。", ok=False)
            return
        name = _display_name(t)
        advanced = self._store.set_status(task_id, CANCELLED)
        if not advanced:
            # LLM 照合の **後** なので初回配達には間に合わない（数秒〜十数秒遅れる）。
            # ここでの抑止が保証するのは「2回目以降（barge-in 再配達 / 配達確認の forgotten 再送）」だけ。
            removed = self._queue.suppress(f"task:{task_id}")
            logger.info("⚰ 報告を抑止(TOCTOU): task=%s 「%.40s」 除去%d件", task_id, name, removed)
            await self._report(req, f"「{name}」はもう終わってたよ。", ok=False)
            return
        logger.info("🚫 タスク取消: %s 「%.40s」(ref=%.40r)", task_id, name, req.reference)
        await self._report(req, llm_message or f"「{name}」を止めたよ。", ok=True)

    async def _report(self, req: CancelRequest, message: str, *, ok: bool) -> None:
        await self._queue.put(Stimulus(
            kind=StimulusKind.CALLFUNCTION_RESULT,
            payload=CallFunctionResult(function_name="cancel_task", content=message, ok=ok),
            dedup_key=f"cancel:{req.id}",
        ))
