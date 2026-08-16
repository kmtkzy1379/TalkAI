"""TaskAgent — ゴールを完遂する境界つき native ツール呼出ループ（inc2）。

応答LLM は `delegate_task(goal=自然文)` で**何をしたいか**だけ委譲する。TaskAgent（賢い `task`
ロールの LLM・非同期）がゴールを受け、次のループを回して**最終結果を1つ**返す:

    LLM(tools=能力) → tool_calls → コードが能力を実行 → 結果を messages へ戻す → LLM が次の手 or 完了

失敗した tool 結果を見て**その場で別アプローチ**を試す（暗黙の Reflexion）。達成したら関数を呼ばず
**短い最終結果（文章）**を返す＝それで完了。**コードが境界を所有**する（V1 修復の核）:
- `max_steps`（OpenAI Agents SDK 既定=5 相当）/ `timeout`（壁時計でなく monotonic）で打ち切り。
- 毎 step `store.get().status==Cancelled` を見て**中断**（結果は破棄＝None を返す）。
- status 確定・終端不変は executor/store 側。LLM は Done を主張できない（最終文を返すだけ）。

eve.task を eve.response から独立に保つため tool_call パースは本ファイルに内製（function_dispatcher
の parse_tool_call と同等）。non-streaming `complete()` は完全な tool_calls を返す（断片結合は不要）。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..clock import now_mono
from ..text_match import has_substance
from .schema import CANCELLED

logger = logging.getLogger(__name__)


TASK_AGENT_SYSTEM = (
    "あなたはAI VTuber「イブ」のタスク実行担当です。与えられたゴールを、提供された関数(tools)を"
    "使って達成してください。各関数の結果を見て次の手を決め、達成できたら**短い最終結果を一言**で"
    "返してください（その時は関数を呼ばず、文章だけを返す）。\n"
    "・ゴールに『N秒後に/N分後に/あとで』等の時間指定があっても、それは予約の時刻で**もう到来している**。"
    "待とうとせず（時間を待つ手段は無い・同じ関数を繰り返し呼ばない）、時間の語は無視して**今すぐ内容を1回実行**する。\n"
    "・失敗した結果が返ってきたら、別の手段・引数で**やり直してよい**（同じ呼び出しの単純反復はしない）。\n"
    "・どうしても達成できない/できない依頼なら、**正直に理由を述べて**終えてください（憶測で成功と言わない）。\n"
    "・**画面で触れるもの（電卓・カウンタ・ミニゲーム等）を頼まれたら、1枚のHTMLファイルを書いて"
    "show_on_screen で出す**。tkinter 等のデスクトップGUIは窓を出せないので使わない"
    "（プログラム実行は画面を持たない environment で動く）。\n"
    "・最終結果はイブがユーザに報告する一言になります。簡潔に、事実だけを。"
    "**必ず「何を」したかが分かる語を入れる**（例:「表示したよ」ではなく「ラーメンの画像を表示したよ」）。"
    "ユーザは実行の経過を見ていないので、対象を言わないと何の報告か伝わらない。"
)

# 実質チェックで直せなかった時に返る「せめて何のタスクか」を示す接頭辞（決定論フォールバック）。
_SUBSTANCE_REPAIR_SYSTEM = (
    "タスクの実行結果を、ユーザに口頭で報告する**1文**に直してください。"
    "**何をしたか（対象）が分かる語を必ず含める**こと。事実だけを書き、実行していないことを足さない。"
    "分からない対象を推測で補わない。話し言葉で短く。記号・箇条書き・かっこ書きは使わない。"
)


def _g(o, k):
    """dict / オブジェクト両対応の属性取り出し（litellm resp は object・fake は dict）。"""
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _content(resp) -> str:
    """resp から choices[0].message.content を頑健に取り出す（要約呼び出し用）。"""
    choices = _g(resp, "choices") or []
    if not choices:
        return ""
    msg = _g(choices[0], "message")
    return (_g(msg, "content") or "") if msg is not None else ""


def _extract(message) -> tuple[str, list[tuple[str, str, str]]]:
    """message から (content, [(call_id, name, raw_args_json)]) を頑健に取り出す。raise しない。"""
    content = _g(message, "content") or ""
    calls: list[tuple[str, str, str]] = []
    for tc in _g(message, "tool_calls") or []:
        cid = _g(tc, "id") or ""
        fn = _g(tc, "function")
        name = (_g(fn, "name") or "") if fn is not None else ""
        raw = (_g(fn, "arguments") if fn is not None else None) or "{}"
        calls.append((cid, name, raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)))
    return content, calls


def _parse_args(raw: str) -> dict:
    try:
        a = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        a = {}
    return a if isinstance(a, dict) else {}


class TaskAgent:
    """ゴール → 境界つきツール呼出ループ → 最終結果1つ。コードが境界(max_steps/timeout/cancel)を所有。"""

    def __init__(
        self, *, registry, model_registry, store,
        max_steps: int = 6, timeout_sec: float = 30.0, now_fn=now_mono,
        self_model_fn=None,
    ) -> None:
        self._registry = registry          # CapabilityRegistry（agent_tool_schemas + execute_async）
        self._model = model_registry        # ModelRegistry（complete("task", ...)）
        self._store = store                  # TaskStore（cancel 検知に get()）
        self._max_steps = max(1, int(max_steps))
        self._timeout = float(timeout_sec)
        self._now = now_fn
        # イブ自身の好み（`memory/self_model.py` の render）を返す関数。既定 None＝従来どおり。
        # **会話履歴は渡さない**ので「タスク担当は今の会話を見られない」契約は無傷。
        # これが無いと、goal が「イブの好きな食べ物の画像」のままの時に解決できない
        # （実機事故 2026-08-15 02:44: TaskAgent が Web 検索で“りんご”を拾ってきた）。
        self._self_model_fn = self_model_fn

    def _self_model_block(self) -> str:
        """goal 中の「イブの好きな〜」を解決するための台帳（無ければ空文字）。

        goal は応答LLM が書くが、**自己参照を具体名に書き換えてくれるとは限らない**
        （実測 2026-08-15: 4回中3回「好きな食べ物の画像出してくれない？」のまま投げた）。
        goal の書き方に期待せず、受け手側が解決できるようにする。
        """
        if self._self_model_fn is None:
            return ""
        try:
            block = self._self_model_fn() or ""
        except Exception:
            logger.exception("自己モデルの取得に失敗（無しで継続）")
            return ""
        if not block:
            return ""
        return ("\n\n" + block
                + "\nゴールに「イブの好きな〜」のような自己参照があれば、上の台帳で解決してから動く。"
                  "台帳に無いことは**調べて決めつけない**（イブ本人が決めていないだけなので、"
                  "無難なものを選ぶか、選べないなら正直にそう言う）。")

    def _cancelled(self, task_id: str) -> bool:
        t = self._store.get(task_id)
        return t is None or t.status == CANCELLED

    async def run(self, goal: str, task_id: str) -> Optional[str]:
        """ゴールを完遂し最終結果文字列を返す。実行中に取り消されたら None（executor が破棄）。"""
        deadline = self._now() + self._timeout
        messages: list[dict] = [
            {"role": "system", "content": TASK_AGENT_SYSTEM + self._self_model_block()},
            {"role": "user", "content": f"ゴール: {goal}"},
        ]
        last_content = ""
        notes: list[str] = []  # 各 step の手順（限界離脱時に「なぜ終わらなかったか」を要約する材料）
        for step in range(self._max_steps):
            if self._cancelled(task_id):
                logger.info("🗒 TaskAgent: 実行中に取消（破棄） task=%s", task_id)
                return None
            if self._now() > deadline:
                logger.info("🗒 TaskAgent: timeout task=%s step=%d", task_id, step)
                # 文言は実際の設定値から作る（固定文字列だと config を変えた時に嘘になる。
                # 実際 J-3 で 180→600 に変えた後も「3分」と表示し続けていた）。
                limit = (f"{self._timeout / 60:.0f}分" if self._timeout >= 60
                         else f"{self._timeout:.0f}秒")
                return await self._summarize_failure(goal, notes, f"時間切れ({limit})")
            try:
                resp = await self._model.complete(
                    "task", messages, tools=self._registry.agent_tool_schemas(),
                )
            except Exception:
                logger.exception("TaskAgent LLM 呼び出し失敗 task=%s", task_id)
                notes.append(f"step{step}: LLM 呼び出しエラー")
                return await self._summarize_failure(goal, notes, "エラー")

            choices = _g(resp, "choices") or []
            message = _g(choices[0], "message") if choices else None
            if message is None:
                return last_content or "（応答が空で完了できなかった）"
            content, calls = _extract(message)
            if content:
                last_content = content
            if not calls:
                # 関数を呼ばない＝完了（達成 or 正直な失敗報告）。
                return await self._ensure_substance(
                    goal, content or last_content or "（結果が空でした）", notes)

            # tool_calls あり → assistant メッセージ + 各 tool 結果を積んで次の step へ。
            messages.append({
                "role": "assistant",
                "content": content or None,  # tool_calls 同伴時は content 省略可
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": raw}}
                    for (cid, name, raw) in calls
                ],
            })
            blocking_done = 0  # 1 step 内で実行した重い能力（検索等）の数
            for (cid, name, raw) in calls:
                # per-call 取消チェック: async 能力（検索等）は call 実行中/実行間に取消が入り得る。
                # step 冒頭のみのチェックだと「止めたよ」の発話後に残り call が実行され続け、
                # single-flight の executor を占有し他タスクも遅延する（レッドチーム S-1）。
                if self._cancelled(task_id):
                    logger.info("🗒 TaskAgent: 実行中に取消（破棄） task=%s", task_id)
                    return None
                # 重い能力は 1 step 1 件まで（J-2 レッドチーム S1: 1 step の一括 tool_calls で
                # executor を長時間占有し他タスクを遅延させる穴を、能力の種類によらず塞ぐ）。
                # 実行せず結果を見てから次を決めさせる＝Reflexion ループの本来の形にも合う。
                if blocking_done >= 1 and self._registry.is_blocking(name):
                    result = "（時間のかかる操作は一度に1つまで。この前の結果を見てから、必要なら次の手で呼んで）"
                else:
                    # J-2 前提工事: 統一経路 execute_async（同期能力は同値・ブロッキングは loop を塞がない）。
                    result = await self._registry.execute_async(name, _parse_args(raw))
                    if self._registry.is_blocking(name):
                        blocking_done += 1
                logger.info("🗒 TaskAgent step=%d %s -> %s", step, name, result[:50])
                notes.append(f"step{step}: {name} -> {result[:60]}")
                messages.append({"role": "tool", "tool_call_id": cid, "content": result})

        logger.info("🗒 TaskAgent: max_steps 到達 task=%s", task_id)
        return await self._summarize_failure(goal, notes, f"手数上限({self._max_steps}手)")

    async def _ensure_substance(self, goal: str, result: str, notes: list[str]) -> str:
        """最終結果に「何を」が入っていなければ、goal と実行手順から一度だけ言い直させる。

        実機事故 2026-08-15 02:44: ネットから拾った画像を出したタスクの最終結果が
        **「表示したよ。」**（内容語ゼロ）になり、ユーザには何を出したのか伝わらなかった。
        TaskAgent 自身も「何の画像か」を知らなかった（検索結果の URL を curl しただけ）ので、
        これは**正直な振る舞いの結果**であり、プロンプトだけでは防ぎ切れない。

        規律（CLAUDE.md「規律はコードで強制」）:
        - 検出は決定論（`text_match.has_substance`）。**内容語がある通常ケースは LLM を呼ばない**
          ＝コスト増ゼロ。呼ぶのは稀な空報告のときだけ。
        - 「（…」で始まる**正直な失敗報告は触らない**（executor がこれで Failed を判定するため、
          言い直すと成否が反転する）。
        - 直せなかった/直しても空のままなら**元の文字列をそのまま返す**（no-op 規律）。
        """
        if not result or result.startswith("（") or has_substance(result):
            return result
        detail = " / ".join(notes[-4:]) if notes else "（手がかりなし）"
        messages = [
            {"role": "system", "content": _SUBSTANCE_REPAIR_SYSTEM},
            {"role": "user", "content": (
                f"ゴール: {goal}\n実際に行った手順:\n{detail}\n"
                f"今の報告文（対象が抜けている）: {result}"
            )},
        ]
        try:
            fixed = _content(await self._model.complete("task", messages)).strip()
        except Exception:
            logger.exception("報告の実質チェックで LLM 失敗（元の報告のまま続行）")
            return result
        # 直った時だけ差し替える。「（」始まりは成否判定を反転させるので**必ず弾く**。
        if not fixed or fixed.startswith("（") or not has_substance(fixed):
            logger.info("🗒 報告の実質チェック: 直せず元のまま: %.40s", result)
            return result
        logger.info("🗒 報告に対象を補った: %.40s → %.40s", result, fixed)
        return fixed

    async def _summarize_failure(self, goal: str, notes: list[str], reason: str) -> str:
        """限界(タイムアウト/手数上限/エラー)で離脱した時、**なぜ終わらなかったか**を一言にまとめて返す。
        返り値は「（…」始まり＝executor が Failed と判定（コードが verdict 所有）。要約 LLM 失敗時は
        notes を決定論連結（no-op 規律）。稀な離脱時のみ 1回だけ呼ぶ。"""
        detail = " / ".join(notes) if notes else "（手がかりなし）"
        messages = [
            {"role": "system", "content": "タスクが完了できませんでした。試した手順から、なぜ時間がかかった/"
                                          "何が原因で終わらなかったかを日本語で**1文**に簡潔にまとめて（言い訳せず事実だけ）。"},
            {"role": "user", "content": f"ゴール: {goal}\n打ち切り理由: {reason}\n試した手順:\n{detail}"},
        ]
        summary = ""
        try:
            resp = await self._model.complete("summarize", messages)
            summary = (_content(resp) or "").strip()
        except Exception:
            logger.exception("限界要約 LLM 失敗（notes で代替）")
        if not summary:
            summary = detail[:120]
        return f"（{reason}: {summary}）"
