"""CancelResolver — タスク取消を**タスク側**で解決するサイドカー（dispatcher の近クローン・別コルーチン）。

応答LLM は `cancel_task(reference)` で取消意図（ユーザの言い回し）を渡すだけ。ここが件数で解決:
- **0件** → コード即時「今止めるタスクは無いよ」（LLM 不要）。
- **1件 かつ reference が曖昧**（空/「やっぱりいいや」等の常套句のみ）→ コード即時キャンセル
  （LLM 不要・停止は速く確実に）。
- **1件 かつ reference に実質語**（「30秒のやつ」等）→ LLM 照合。2026-07-13 実機事故:
  応答LLMが完了済み変更を再実行して cancel を再発行し、無照合 fast path が唯一のアクティブ
  （ユーザが残したい方）を削除した。1件でも「その1件が reference の指す対象」とは限らない。
- **2件以上** → タスクLLM(gpt-5.5) が store を見て reference をファジー照合（match/none/ambiguous）。

**executor(single-flight) 経由にしない**＝取消が実行中タスクの後ろで待つのを防ぐ（別コルーチンで並行）。
実行中 goal は agent の毎step CANCELLED チェックが中断・破棄する（agent/executor 側で既存）。

**TOCTOU**: LLM await の前後で状態が変わり得る→**書込直前に store を読み直し**、`set_status` の戻り値 bool で
分岐（False=既に終了→「もう終わってた」）。スナップショットから「取り消した」と誤報しない。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
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


CANCEL_SYSTEM = (
    "あなたはAI VTuber「イブ」のタスク取消担当です。ユーザが取り消したい対象(reference)と、現在の"
    "未終了タスク一覧が与えられます。どれを取り消すべきか判断し、次の JSON だけを返してください:\n"
    '{"decision":"match|none|ambiguous","task_id":"<一致したIDまたは空>","message":"<ユーザに話す一言>"}\n'
    "・match: reference が1つのタスクに対応。task_id を入れ、message は「『〜』を止めるね」等。\n"
    "・none: 対応するタスクが無い。message は「そんなタスクは入ってないよ。今あるのは〜」等。\n"
    "・ambiguous: 複数該当で決められない。message は候補を挙げ「〜と〜、どっち?」と聞く。\n"
    "・一覧が1件だけでも安易に match にしない。reference の数字・時刻・内容がタスクと食い違うなら"
    "必ず none（例: reference『30秒後のやつ』・一覧『50秒後に気持ちを伝えて』→ none）。\n"
    "・reference が対象を特定しない取消の言い回しだけ（『やっぱりいい』『さっきの』等）で"
    "一覧が1件なら、その1件を match としてよい。\n"
    "message は簡潔に事実だけ。JSON 以外は書かない。"
)

# --- 曖昧語ゲート（1件 fast path の可否判定）------------------------------------
# 「取消の常套句・指示語・フィラー・助詞」だけで構成される reference は対象を特定していない
# → 1件なら即キャンセルしてよい。実質語（数字・内容語）が残れば LLM 照合へ倒す（安全方向:
# 誤って「具体的」と判定しても遅延が増えるだけで、誤取消にはならない）。
_PUNCT = re.compile(r"[\s、。,.!?！？〜ー・…「」『』()（）]+")
_VAGUE_TOKENS = re.compile(
    "|".join((
        # 長い語を先に（re の選択は左から先勝ちのため、接頭辞関係は長い順に並べる）
        "やっぱりいいや", "やっぱりいい", "やっぱり", "やっぱ", "やはり",
        "取り消して", "取り消し", "取消", "きゃんせる", "キャンセル", "cancel", "stop", "ストップ", "中止",
        "やめといて", "やめとく", "やめて", "やめる", "やめた", "止めといて", "止めて", "止める",
        "いらない", "いらん", "いいや", "いいよ", "いいです", "いい", "なしにして", "なしで", "なし",
        "さっきの", "さっき", "例の", "それも", "それ", "あれ", "その", "あの", "この", "今の",
        "お願い", "ごめん", "うん", "もう", "全部", "タスク", "予約", "やつ", "奴", "にして", "して",
        "を", "は", "が", "の", "で", "に", "も", "ね", "よ",
    ))
)


def _is_vague_reference(ref: str) -> bool:
    """対象を特定しない取消か（True=1件ならコード即取消してよい / False=LLM 照合へ）。"""
    s = _PUNCT.sub("", unicodedata.normalize("NFKC", ref or "").strip().lower())
    if not s:
        return True  # 空指定 = 「今のをやめて」
    return not _VAGUE_TOKENS.sub("", s)  # 実質語（数字・内容語）が残れば False


def _g(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _content(resp) -> str:
    choices = _g(resp, "choices") or []
    if not choices:
        return ""
    msg = _g(choices[0], "message")
    return (_g(msg, "content") or "") if msg is not None else ""


def _display_name(t) -> str:
    return t.goal or t.what or "(無題)"


class CancelResolver:
    def __init__(
        self, *, store, model_registry, queue,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._model = model_registry
        self._queue = queue
        self._now = now_fn  # TaskStore と同じシグネチャ（決定論テストで注入する）
        self._inbox: "asyncio.Queue[CancelRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # --- ライフサイクル（dispatcher と同形）---
    def start(self) -> None:
        if self._task is None:
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
        """**まだ発話されていない報告**を持つ terminal タスク（＝抑止してよい対象）。

        時計だけで「直近の terminal」を選ぶと実機で誤る: 2026-07-29 E2E の D2 時点では
        90秒窓に3件（東京タワー56.2s前 / 琵琶湖46.5s前 / 川1.9s前）あったが、先の2件は
        既に発話済みで、未配達はユーザが指した川の1件だけだった。キューの実体と突き合わせれば
        候補は一意に決まる。起動時 age-out の孤児（store が updated_at を打ち直す）も
        put を出さないので自動的に候補外になる。
        """
        queued = {s.dedup_key for s in self._queue.snapshot() if (s.dedup_key or "").startswith("task:")}
        return [t for t in self._store.list_all()
                if t.status in (DONE, FAILED) and f"task:{t.task_id}" in queued]

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
        elif len(actives) == 1 and _is_vague_reference(req.reference):
            # 1件×曖昧・空 reference（「やっぱりいいや」等）＝対象はその1件→即キャンセル（速く確実に）。
            await self._cancel_and_report(req, actives[0].task_id)
        else:
            # 2件以上、または1件でも実質語のある reference → タスクLLM がファジー照合。
            # （1件無照合は 2026-07-13 実機事故: 「30秒の…」で唯一の50秒タスクを誤取消）
            await self._resolve_with_llm(req, actives)

    async def _resolve_with_llm(self, req: CancelRequest, actives) -> None:
        listing = "\n".join(f'- id={t.task_id} 内容="{_display_name(t)}" 状態={t.status}' for t in actives)
        messages = [
            {"role": "system", "content": CANCEL_SYSTEM},
            {"role": "user", "content": f"取り消したい対象: {req.reference or '(指定なし)'}\n現在の未終了タスク:\n{listing}"},
        ]
        try:
            resp = await self._model.complete("task", messages)
        except Exception:
            logger.exception("キャンセル解決 LLM 失敗")
            if len(actives) == 1:
                # fail-safe: 取り消していないことを名前つきで明示（ユーザが「やっぱりいい」と
                # 言い直せば曖昧ゲート経由で LLM 無しでも止められる）。
                await self._report(req, f"ごめん、確認できなかったよ。「{_display_name(actives[0])}」はそのままにしてるよ。", ok=False)
            else:
                await self._report(req, "ごめん、どのタスクか確認できなかったよ。", ok=False)
            return
        decision, task_id, message = self._parse(resp)
        logger.info("🚫 取消解決(LLM): decision=%s task_id=%s", decision or "(空)", task_id or "(空)")
        if decision == "match" and task_id in {t.task_id for t in actives}:
            # task_id はスナップショット内のものだけ許可（LLM の幻覚 ID で誤取消/誤報しない）。
            await self._cancel_and_report(req, task_id, llm_message=message)
        else:
            # none / ambiguous / 幻覚ID / 解析失敗 → status は変えず message を報告。
            await self._report(req, message or "どのタスクのことか分からなかったよ。", ok=False)

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

    @staticmethod
    def _parse(resp) -> tuple[str, str, str]:
        raw = _content(resp).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)  # フェンスや前後テキストを許容
        if m:
            try:
                d = json.loads(m.group(0))
                if isinstance(d, dict):
                    return (str(d.get("decision", "")), str(d.get("task_id") or "").strip(), str(d.get("message", "")))
            except (ValueError, TypeError):
                pass
        return ("none", "", "")
