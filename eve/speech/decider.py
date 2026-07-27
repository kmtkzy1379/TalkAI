"""F5 発話判定（should_speak）— 企画書フロー + 中核原理 surprise を「指標」として総合判断。

企画書: ユーザ5秒無言で `…` を発話判定LLMに送る。LLM は「話す/黙る」を判断し、
True なら(理由+応答LLMへ渡す content)、False なら(理由)を返す（理由は両方で必須＝
「楽な False」への偏り防止）。

**surprise は数値で判定を絶対決定しない（ユーザ裁定）**: 人間も予想が外れたから必ず話す訳でも、
当たったから必ず黙る訳でもない（感情/思考が高ぶる/安定するだけ）。よって surprise は **各感情
(直近フィードバック) と内容と合わせて発話判定LLMが総合判断する「指標(必須入力)」**として渡す。
唯一の hard ゲートは pending_obligation（予約締切等の事実・将来 Call-Function）。

T2 death-detection（surprise の非装飾性）: surprise は **必須引数**（Optional 化禁止）で decide_fn へ
配線され、判定を動かしうる。surprise を読む fake decide_fn で値を振ると判定が反転することを検証
（surprise が決定に効く配線である保証）。解析失敗は silence へ保守的フォールバック。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ..clock import elapsed_wall, humanize, now_iso

logger = logging.getLogger(__name__)

# 話す判断だが LLM が content を出さなかった時の最小ヒント（応答LLMが文脈から膨らませる）。
_FALLBACK_CONTENT = "（今の状況に自然に一言）"


@dataclass(frozen=True)
class AutonomousSpeech:
    """自発発話の刺激 payload（content=応答LLMへ渡す内容 / reason=なぜ話すか）。"""

    content: str
    reason: str


@dataclass(frozen=True)
class SpeechDecision:
    speak: bool
    reason: str
    content: str = ""


# decide_fn の型: 文脈を受けて LLM 判断（SpeechDecision）を返す。テストは fake を注入。
DecideFn = Callable[..., Awaitable[SpeechDecision]]


# ---- 出力パーサ（タグ付きテキスト・raise しない）-------------------------
def _extract(text: str, *aliases: str) -> str:
    for a in aliases:
        m = re.search(rf"^\s*{re.escape(a)}\s*[:：]\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return ""


_YES = re.compile(r"(yes|true|話す|はい|y\b|1)", re.IGNORECASE)


def parse_speech_decision(text: Optional[str]) -> SpeechDecision:
    """発話判定LLM の出力 → SpeechDecision。raise しない・不明瞭は保守的に silence。"""
    text = text or ""
    raw_speak = _extract(text, "speak", "判定", "話す")
    reason = _extract(text, "reason", "理由")
    content = _extract(text, "content", "内容", "発話")
    if raw_speak:
        speak = bool(_YES.search(raw_speak)) and "no" not in raw_speak.lower() and "黙" not in raw_speak
    else:
        # 保守的: speak タグが無ければ黙る（content の有無で speak と推定しない＝不明瞭は silence）。
        speak = False
    if not (raw_speak or reason or content):
        return SpeechDecision(False, "発話判定の出力を解析できず沈黙（保守）", "")
    return SpeechDecision(speak=speak, reason=reason or "(理由なし)", content=content)


# ---- 本体（pending だけ hard・surprise は指標として LLM が総合判断）----------
async def should_speak(
    *,
    surprise: int,  # 必須・非 Optional（中核原理: surprise を装飾化させない・指標として配線）
    silence_seconds: float,
    recent_turns,
    topic_seeds,
    decide_fn: DecideFn,
    last_feedback: Optional[str] = None,  # イブの今の感情/要約（直近フィードバック）
    vision: Optional[str] = None,  # F6 直近の画面ナレーション（あれば判定材料に・None なら無し）
    active_tasks: Optional[list] = None,  # J-2 P2-3 実行中タスク一覧（あれば材料に・None なら無し）
    pending_obligation: bool = False,
) -> SpeechDecision:
    if pending_obligation:
        # 唯一の hard ゲート（事実: 予約締切等。感情でないのでここだけ確定で沈黙）。
        # 将来 Call-Function/task が締切近接を計算して渡す（今は常に False）。
        return SpeechDecision(False, "保留中の予約/義務があるため沈黙", "")
    # surprise + 感情(last_feedback) + 会話 + 話題の種 (+ 画面 + 実行中タスク) を渡し、LLM が総合判断。
    # vision/active_tasks は **非 None の時だけ** 転送する（A6: 受けない既存 decide_fn を壊さない）。
    kwargs = dict(
        surprise=surprise,
        silence_seconds=silence_seconds,
        recent_turns=recent_turns,
        topic_seeds=topic_seeds,
        last_feedback=last_feedback,
    )
    if vision is not None:
        kwargs["vision"] = vision
    if active_tasks is not None:
        kwargs["active_tasks"] = active_tasks
    d = await decide_fn(**kwargs)
    if d.speak and not (d.content or "").strip():
        # 話す判断だが content が空 → 全 speak 経路で最小ヒントを保証（応答LLMが膨らませる）。
        return SpeechDecision(True, d.reason, _FALLBACK_CONTENT)
    return d


# ---- 本番 decide_fn（ModelRegistry role=speech_decide）---------------------
_SPEAKER_LABEL = {"user": "ユーザ", "eve": "イブ"}

# ---- J-2 ②-4: 根拠なき話題の丸投げ（空振り発話）の判定 --------------------
# 実機事故(2026-07-26 実起動状態E2E): 再生された自発発話6件中2件が「今ふと、気になってること
# とかある？」のように**中身を相手に出させるだけ**だった。プロンプトには既に「毎回は話しかけない」
# 「本当に良い一言がある時だけ」があるのに、115秒間に同種の下書きが9件生成された＝規律は
# コードで強制する（`content_similarity` と同じ経緯）。
# 判定は2条件の**論理積**。実データ23件で適合率1.00/再現率0.80、イブ実発話198件への誤爆は1件
# （＝狙った空振りそのもの）。接地アーム単独では実発話の53%を潰す（イブが自分から具体を持ち込む
# 発話を罰してしまう）ため、必ず「丸投げ表現」との積にする。
_TOPIC_TOKEN = re.compile(r"[一-龥]{2,}|[ァ-ヴー]{3,}|[A-Za-z][A-Za-z0-9]+")
_GENERIC_TOKEN = frozenset({
    "自分", "気持", "時間", "今日", "今度", "感じ", "本当", "一緒", "最近", "話題", "内容",
    "予定", "休憩", "様子", "気分", "会話", "確認", "状況", "説明", "相談", "返事",
    "ユーザ", "イブ", "テーマ", "タイム",
})
_INDEF = r"(こと|もの|話|テーマ|ネタ|話題|疑問)"
_VAGUE = r"(気になっ|気になる|引っかかっ|思い出し|浮かん|話したい|聞きたい|掘り下げ)"
_SOLICIT = r"(ある[？\?]|ない[？\?]|聞かせて|教えて|聞いてみたい|話して|どう[？\?]|した[？\?]|みる[？\?])"
_OUTSOURCE = re.compile(
    _VAGUE + r"[^。]{0,20}" + _INDEF + r"[^。]{0,12}" + _SOLICIT
    + r"|" + _VAGUE + r"[^。]{0,20}" + _INDEF + r"[^。]{0,6}(を|、)?(聞かせて|教えて|聞いてみたい)"
    + r"|" + r"(困っ|詰まっ|悩ん|大変)[^。]{0,10}(こと|の)?[^。]{0,6}[？\?]"
)


def _topic_tokens(text: Optional[str]) -> set:
    return {t for t in _TOPIC_TOKEN.findall(text or "") if t not in _GENERIC_TOKEN}


def is_topic_outsourcing(content: str, materials) -> bool:
    """材料(話題の種/画面/直近会話)に接地しないまま、話題を相手に丸投げする下書きか（True=抑制）。

    materials に画面ナレーションが入るのが要点: 画面が動いている時（＝「困ってそう」の観測可能な
    根拠がある時）だけ「何か困ってる?」が接地して通る。静止画面では画面ブロック自体が渡らない
    （層分離）ので、根拠なしの丸投げは止まる。
    """
    if not _OUTSOURCE.search(content or ""):
        return False  # そもそも丸投げの言い回しでない（気遣い/報告/記憶起点は対象外）
    mine = _topic_tokens(content)
    if not mine:
        return True  # 具体語ゼロ＋丸投げ＝完全な空振り
    theirs: set = set()
    for m in materials or []:
        theirs |= _topic_tokens(m)
    return not (mine & theirs)

SPEECH_DECIDE_SYSTEM = """\
あなたはAI VTuber「イブ」の発話判定モジュールです。今は誰も話していない（沈黙 or 相手が画面を操作中）。
直近の会話・話題の種(記憶)・イブの今の感情(直近フィードバック)・画面(今見えているもの)・予測差(surprise)を
**総合的に**見て、イブが今"自分から"一言を言うべきか黙るべきかを判断します。

【話す(yes) — 相手がうれしい/役立つ一言があるなら積極的に言ってよい】
- 画面の内容を**過去の記憶と結びつける**（例:「前にチーズケーキ好きって言ってたね、これ良さそう」）。
- 画面で相手が**探している/迷っているものに気づいて手伝う**（例:「スポッチャ、室内で雨でも遊べていいね」）。
- 直近の会話を一歩進める／関連する**新しい話題を記憶から**振る／相手が一息ついた間に声をかける。
- **沈黙が長い時は、過去の記憶や、そこから素朴に気になったことを自分から振ってよい**（人は黙っている
  相手にも、ふと思い出した話や疑問を話しかける。画面や直前の会話に材料が無い時こそ、記憶が話のきっかけになる）。
- 「# 画面」ブロックが**無い**ときは、直近に画面の変化が無い（＝今の画面については何も分かっていない）。
  画面には触れず、**記憶や直近の会話を材料にする**（画面が無い＝話す材料が無い、ではない）。

【黙る(no)】
- **直前にイブが自分から話して、相手がまだ返事していない**（畳みかけない・間を空けて相手の番を待つ）。
  ※ただし会話行の「いつ」を見ること。何時間も/何日も前の会話は**返事待ちではない**（前のセッションの
  続きなので、久しぶりに声をかけてよい。そこで止まっている話題があるなら、それを振るのはむしろ自然）。
- 本当に言うことがない／さっきと同じことの繰り返しになる。
- 相手が**今まさに手を動かして**作業に没頭している（入力中・操作中で画面が動いている）など、口を挟むと**邪魔になりそう**な時。※画面の情報が無い時は画面について何も推測しない（止まっている＝手が空いている、とも、集中中、とも決めつけない）。
- **画面に新しい変化が無いのに、こちらから画面の話題を持ち出さない**（変化していない画面に急に話しかけるのは不自然。「# 画面」ブロックが無い時に画面の話をするのは捏造になる）。
- 相手が**疲れていたり休みたそう**な時（「疲れた」等の直後）は、話題を増やさずそっとしておく。
- **実行中のタスク（検索/調べ物等）が扱っている内容そのものを、自分の知識で先回りして答えない**
  （結果は完了後に別途届く。今ここで自分の記憶/知識から答えると、届く結果と食い違ったり
  二重に伝わったりする。「まだ調べてるところだよ」等、進行に触れる一言は話してよい）。

【禁止】
- **相手に話題を出させるだけの一言を言わない**（「何か気になってることある？」「聞かせて」等）。
  画面や記憶に**名指しできる具体物**がある時だけ、それを添えて聞く。困りごとを聞くのも同じで、
  画面に「同じ操作の繰り返し」等の**見えている根拠**がある時に限る（根拠なしに聞くのは話題の丸投げ）。
- **毎回は話しかけない**。本当に良い一言・気の利いた一言がある時だけに絞る（質問の連投・実況の垂れ流しはしない）。
- 同じ記憶を続けて持ち出さない（一度触れた話題は間を空ける）。
- **画面変化のいちいちを実況報告しない**（「○○が表示されました」の垂れ流しは黙る）。挨拶の繰り返しもしない。
- surprise(予測差)は強い指標だが絶対ではない（高くても黙ってよいし、低くても話してよい）。

必ず次の形式で1行ずつ出力（**理由は yes/no どちらでも必須**）:
speak: yes または no
reason: なぜそう判断したか（1文）
content: 話すなら、イブが実際に話す内容のたたき台（1文）。黙るなら空でよい。"""


def _render_turns(turns, now: Optional[str] = None) -> str:
    """直近会話の描画: **話者 + いつの発話か**。

    時刻が無いと、起動直後に前回セッションの続き（例: 9日前に中断した調査の話）を「今まさに
    相手の返事待ち」と読み、沈黙し続ける（実測 2026-07-26・実起動状態のE2E: 25判定中24が
    「直前に伝えたばかりで返事待ち」を理由に沈黙）。沈黙秒数はプロセス起動からの値なので、
    会話がいつのものかはここでしか分からない。
    """
    n = now or now_iso()
    lines = []
    for t in turns or []:
        label = _SPEAKER_LABEL.get(getattr(t, "speaker", ""), getattr(t, "speaker", ""))
        iso = getattr(getattr(t, "stamp", None), "iso", "") or ""
        when = f"/{humanize(elapsed_wall(iso, n))}" if iso else ""
        lines.append(f"[{label}{when}] {getattr(t, 'text', '')}")
    return "\n".join(lines) or "（直近の会話なし）"


def _render_seeds(seeds, now: Optional[str] = None) -> str:
    """話題の種の描画: **要約1行 + いつの記憶か**（応答文脈の `[話題の種/3日前]` と同じ時間接地）。

    時刻を付けないと「そういえば*前に*」が言えない（5分前か3週間前か判定LLMに分からない）。
    text 全文でなく1行要約なのは、text が「感情/次の予測/予測差/理由」を含む内部ログ形式で
    話題として使えるのは先頭1行だけだから（実測: 平均162字中40字・2026-07-26）。
    """
    n = now or now_iso()
    lines = []
    for c in seeds or []:
        body = c.seed_text() if hasattr(c, "seed_text") else getattr(c, "text", "")
        iso = getattr(c, "iso", "")
        when = f"[{humanize(elapsed_wall(iso, n))}] " if iso else ""
        lines.append(f"・{when}{body}")
    return "\n".join(lines) or "（なし）"


def _render_active_tasks(active_tasks: Optional[list]) -> str:
    """J-2 P2-3: 実行中タスクの簡易描画。None=ブロック自体を出さない（active_tasks_for_context
    と同じ規約: 空リスト=ゼロ件を明示 / None=タスク機能未配線でブロック自体を注入しない）。"""
    if active_tasks is None:
        return ""
    body = "\n".join(active_tasks) if active_tasks else "（実行中の予約タスクは無い）"
    return (
        "\n\n# 実行中のタスク（検索/調べ物等。結果は完了後に別途届く。今ここでは先回りして"
        f"答えない）\n{body}"
    )


def build_decide_messages(
    *, surprise: int, silence_seconds: float, recent_turns, topic_seeds,
    last_feedback: Optional[str] = None, vision: Optional[str] = None,
    active_tasks: Optional[list] = None, now: Optional[str] = None,
) -> list[dict]:
    fb = (last_feedback or "").strip() or "（なし）"
    screen = (vision or "").strip()
    screen_block = f"\n\n# 画面（今この瞬間）\n{screen}" if screen else ""
    tasks_block = _render_active_tasks(active_tasks)
    user = (
        "…\n\n"
        f"# 直近の会話（[話者/いつの発話か]）\n{_render_turns(recent_turns, now)}\n\n"
        f"# 過去の記憶・話題の種（今の流れに合えば「そういえば前〜って言ってたね」と自然に話を広げてよい）\n{_render_seeds(topic_seeds, now)}\n\n"
        f"# イブの今の状態（直近フィードバック: 感情/要約）\n{fb}"
        f"{screen_block}{tasks_block}\n\n"
        f"# 状況\n沈黙{silence_seconds:.0f}秒 / 予測差(surprise)={surprise}"
        "（指標。高=思考/感情が高ぶる・低=安定。これだけで決めない）"
    )
    return [
        {"role": "system", "content": SPEECH_DECIDE_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_decide_fn(registry) -> DecideFn:
    """ModelRegistry role=speech_decide を叩く本番 decide_fn を作る。"""

    async def decide_fn(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None,
                        vision=None, active_tasks=None) -> SpeechDecision:
        messages = build_decide_messages(
            surprise=surprise, silence_seconds=silence_seconds,
            recent_turns=recent_turns, topic_seeds=topic_seeds, last_feedback=last_feedback,
            vision=vision, active_tasks=active_tasks,
        )
        try:
            resp = await registry.complete("speech_decide", messages)
        except Exception:
            logger.exception("発話判定LLM 呼び出し失敗（沈黙にフォールバック）")
            return SpeechDecision(False, "発話判定LLM 失敗のため沈黙（保守）", "")
        return parse_speech_decision(_content(resp))

    return decide_fn


def _content(resp: object) -> str:
    try:
        c = resp.choices[0].message.content  # type: ignore[attr-defined]
        if isinstance(c, str):
            return c
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    if isinstance(resp, dict):
        try:
            c = resp["choices"][0]["message"]["content"]
            if isinstance(c, str):
                return c
        except (KeyError, IndexError, TypeError):
            pass
    if isinstance(resp, str):
        return resp
    return ""
