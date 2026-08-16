"""配達確認サイドカー（**決定論の関門 + 曖昧な時だけ賢い判定LLM**）。

発端の実機事故(2026-07-18 E2E S2): 検索完了の報告ターンが、直前の未応答ユーザ発話への返事を
優先し、報告内容(payload.content)を一度も発話せずに正常完了した（barge-in ではないため既存の
`_maybe_redeliver`(世代変化/CancelledError 専用)は対象外＝検出装置が無かった）。

**なぜ LLM 判定をやめたか（裁定 2026-08-15・実機事故 02:44）**:
報告「（任せてくれたタスク）表示したよ。」に対し実発話も「表示したよ。」＝**ほぼ完全一致**なのに
判定LLM(既定 gpt-4o-mini)が `forgotten` を返し、再配達→応答LLM が謝罪ごと作り直して**二重回答**に
なった。同経路の成功2件（「チーズケーキの画像を表示したよ。」等）との唯一の差は**照合できる
内容語の有無**だった。判定は毎回 LLM を呼ぶ構造で、完全一致でも確率的に falseに倒れ得た。
パイプライン中で**最も弱いモデルが最も強い権限（再依頼）を持つ**構図でもあった。

**なぜ全部を決定論にしなかったか（ユーザ指摘 2026-08-15）**:
語の一致だけで見ると、報告「チーズケーキの画像を表示したよ」に対して実発話「やったよ」が
「未配達」に倒れ、**不要な再配達＝直そうとしていた二重発話そのもの**を生む。しかも同じ
「語が重ならない」形でも、報告「東京の明日の天気は雨だよ」×発話「うん、そうみたい」は
**中身が本当に届いていない**ので再配達が正しい。差は「文脈から要点を復元できるか」であり、
語の一致では原理的に判別できない。よってそこだけ判定LLM に渡す。

構成:
  1. 一文字も喋っていない              → not_conveyed（決定論・確実）
  2. 報告文（定型接頭辞を除く）を丸ごと含む → delivered（決定論）
  3. 報告に**内容語が無い**             → delivered（決定論。照合材料が無い。ここで再配達すると
     応答LLM は文脈で埋めるしかなく＝**捏造を強制する**。02:44 の事故がまさにこれ）
  4. 内容語が1つでも重なる             → delivered（決定論・緩め＝ユーザ裁定）
  5. 内容語が1つも重ならない           → **判定LLM に聞く**（迷ったら delivered）

この形なら 02:44 の事故は規則3で**確率に依らず**消え、LLM が呼ばれるのは稀な5だけになる。
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..pipeline.stimulus import (
    REDELIVER_NOT_CONVEYED,
    CallFunctionResult,
    Stimulus,
    StimulusKind,
)
from ..text_match import content_tokens, strip_report_prefix

logger = logging.getLogger(__name__)


def _g(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _content(resp) -> str:
    choices = _g(resp, "choices") or []
    if not choices:
        return ""
    msg = _g(choices[0], "message")
    return (_g(msg, "content") or "") if msg is not None else ""


VERDICT_DELIVERED = "delivered"
VERDICT_UNSURE = "unsure"  # 決定論では決められない＝判定LLM に聞く


def decide_delivery_fast(report: str, spoken: str) -> str:
    """決定論で確定できる分だけ判定する。決められなければ `VERDICT_UNSURE`。

    ここで確定するのは「LLM に聞くまでもない」ケースだけ。**02:44 の事故（報告と発話が
    ほぼ一致しているのに forgotten）はここで確定するので、確率的に揺れる余地が無い**。
    """
    core = strip_report_prefix(report)
    said = (spoken or "").strip()
    if not said:
        return REDELIVER_NOT_CONVEYED  # 1) 一文字も喋っていない＝確実に未配達
    if core and core in said:
        return VERDICT_DELIVERED       # 2) 報告文を丸ごと言った
    want = content_tokens(core)
    if not want:
        # 3) 報告に照合できる内容語が無い（例「表示したよ。」）。ここで再配達しても応答LLM は
        #    文脈で埋めるしかない＝**捏造を強制する**。伝えるべき中身が無い以上、済み扱いにする。
        return VERDICT_DELIVERED
    if want & content_tokens(said):
        return VERDICT_DELIVERED       # 4) 要点の語が発話に出ている（言い換え可）
    # 5) 語が一つも重ならない。ここは**二分できない**:
    #    「東京の明日の天気は雨だよ」×「うん、そうみたい」＝中身が届いておらず再配達が正しい。
    #    「チーズケーキの画像を表示したよ」×「やったよ」＝直前にユーザ自身が頼んでおり文脈で
    #    通じている＝再配達は不要（ユーザ指摘 2026-08-15）。
    #    差は「文脈から要点を復元できるか」で、語の一致では原理的に判別できない → LLM に聞く。
    return VERDICT_UNSURE


# 判定LLM は**この分岐でしか呼ばれない**（＝実測ではほぼ発火しない）ので、安さより賢さを取る。
# 既定を旧 gpt-4o-mini から上げた理由: 02:44 の誤判定を出したのがその世代で、しかも
# 「パイプラインで最も弱いモデルが最も強い権限（再依頼）を持つ」構図になっていたため。
DELIVERY_CHECK_SYSTEM = (
    "あなたはAI VTuber「イブ」の報告が、ユーザに伝わったかを見る確認係です。\n"
    "イブには機能実行結果（報告すべき内容）が渡され、直後にイブが実際に発話しました。\n"
    "**判定は緩めに。迷ったら conveyed（伝わった）にしてください。**\n"
    "- conveyed: 報告の要点が伝わっている。**言い換え・要約・指示語でもよい**。"
    "ユーザが直前に自分で頼んでいて、短い返事（「やったよ」「出しといたよ」等）でも"
    "何のことか文脈で分かるなら conveyed。\n"
    "- not_conveyed: 報告に含まれる**具体的な中身（調べた結果・数値・事実）がユーザに一切"
    "届いていない**場合だけ。例: 天気を調べた報告なのに天気に触れず別の話だけをした。\n"
    "出力は conveyed / not_conveyed のどちらか1語のみ。"
)


def parse_delivery_verdict(text: Optional[str]) -> str:
    """判定LLM出力→verdict。解析失敗/不明瞭は delivered（安全側＝過剰な再配達をしない）。"""
    t = (text or "").strip().lower()
    # "not_conveyed" は "conveyed" を部分文字列に含むので**先に**見る。
    if "not_conveyed" in t or "not conveyed" in t:
        return REDELIVER_NOT_CONVEYED
    return VERDICT_DELIVERED


def _summarize(text: str, limit: int = 24) -> str:
    """ログ用の短縮（改行を潰す）。診断のために報告/発話の頭を残す。"""
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:limit] + ("…" if len(t) > limit else "")


@dataclass
class DeliveryCheckRequest:
    stimulus: Stimulus
    recent_text: str
    spoken_text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class DeliveryChecker:
    """single-flight サイドカー（FeedbackWorker/CancelResolver と同形）。既定 no-op（未 start）。

    LLM を持たないので判定自体は同期で終わるが、**サイドカーの形は維持する**:
    `orchestrator.handle` を1マイクロ秒もブロックしない契約と、再配達の put が
    `voice_loop._redeliver_stimulus`（STT 猶予つき）経由であることを変えないため。
    """

    def __init__(self, *, model_registry, redeliver_fn, max_attempts: int) -> None:
        self._model = model_registry
        self._redeliver_fn = redeliver_fn
        self._max_attempts = max_attempts
        self._inbox: "asyncio.Queue[DeliveryCheckRequest]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    # --- ライフサイクル（dispatcher/cancel_resolver と同形）---

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

    def submit(self, stimulus: Stimulus, *, recent_text: str, spoken_text: str) -> None:
        """判定要求を積むだけ（O(1)・非ブロッキング・orchestrator.handle をブロックしない）。"""
        self._inbox.put_nowait(DeliveryCheckRequest(stimulus, recent_text, spoken_text))

    async def _run(self) -> None:
        while not self._stopping:
            req = await self._inbox.get()
            try:
                await self._handle_one(req)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("配達確認で例外（継続）")
            finally:
                self._inbox.task_done()

    async def _ask_llm(self, req: DeliveryCheckRequest, payload: CallFunctionResult) -> str:
        """曖昧な分岐だけ判定LLM に聞く。失敗・不明瞭は delivered（安全側）。"""
        if self._model is None:
            return VERDICT_DELIVERED  # 未配線＝過剰な再配達に倒さない
        messages = [
            {"role": "system", "content": DELIVERY_CHECK_SYSTEM},
            {"role": "user", "content": (
                f"# 直近の会話\n{req.recent_text or '（なし）'}\n\n"
                f"# 機能実行結果（報告すべき内容）\n{payload.content}\n\n"
                f"# イブの実際の発話\n{req.spoken_text or '（発話なし）'}"
            )},
        ]
        try:
            resp = await self._model.complete("delivery_check", messages)
        except Exception:
            logger.exception("配達確認 LLM 失敗（再配達しない・安全側）")
            return VERDICT_DELIVERED
        return parse_delivery_verdict(_content(resp))

    async def _handle_one(self, req: DeliveryCheckRequest) -> None:
        payload = req.stimulus.payload
        if not isinstance(payload, CallFunctionResult):
            return
        verdict = decide_delivery_fast(payload.content, req.spoken_text)
        asked = verdict == VERDICT_UNSURE
        if asked:
            verdict = await self._ask_llm(req, payload)
        logger.info(
            "📮 配達確認: %s%s (dedup=%s) 報告=%r 発話=%r",
            verdict, "(LLM判定)" if asked else "", req.stimulus.dedup_key,
            _summarize(payload.content), _summarize(req.spoken_text),
        )
        if verdict != REDELIVER_NOT_CONVEYED:
            return
        if payload.attempts >= self._max_attempts:
            logger.warning("⚠ 配達確認で未配達だが再配達上限のため断念: %.40s", payload.content)
            return
        if self._redeliver_fn is None:
            return
        retry = Stimulus(
            kind=StimulusKind.CALLFUNCTION_RESULT,
            payload=CallFunctionResult(
                payload.function_name, payload.content, payload.ok, payload.attempts + 1,
                # barge-in ではない＝「遮られた」と言ってはいけない（02:44 の二重回答の原因）。
                REDELIVER_NOT_CONVEYED,
            ),
            dedup_key=req.stimulus.dedup_key,
        )
        logger.info("🔁 報告を再配達予約（未配達・%d回目）: %.40s", payload.attempts + 1, payload.content)
        # ターンは既に完了済み（audio.join 後）＝put 直前に再考すべきレースは無い
        # （barge-in 用の delivered() クロージャと違い、ここでの spoken は既に確定している）。
        self._redeliver_fn(retry, lambda: False)
