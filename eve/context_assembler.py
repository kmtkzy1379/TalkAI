"""応答LLM への文脈を組み立てる器（native チャットロール構成）。

過去参照防止(T6): RAG チャンク・直近会話に相対時刻を付け現在に接地。無言時の random RAG は
「話題の種」と明示し「過去の記憶(思い出話)」と峻別（v1 の過去逸れ抑止）。

**会話は native な user/assistant ロールで構成する（Fix4）**:
1個の `role:"user"` ブロブに会話を全部詰めると、モデルが話者を取り違える（自分=イブの直前の
発話に自分で返事する／自分の質問に自答する）。assistant=イブ自身 / user=相手、をモデル本来の
ロール構造で示し、取り違えを構造的に防ぐ。RAG/直近フィードバック等の文脈は system に置く。
実際の LLM 呼び出しは ResponseOrchestrator 側。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .clock import Stamp, elapsed_wall, humanize

logger = logging.getLogger(__name__)

# 注入セレクション（ConversationCache.recent_for_injection）が差し込む省略マーカの話者名。
OMITTED_SPEAKER = "__omitted__"
SPEAKER_EVE = "eve"

# D1: 「会話の間隔」ブロックを出す閾値。
# **`decider.py` の STALE_RECENCY_SEC(900) と値は同じだが根拠は別タスク**なので import しない
# （あちらは「イブの下書きの指示語が誤りか」の校正値・母数はイブ発話266件中の指示語12件）。
# こちらは「前の依頼がまだ生きているとみなすか」で、本番履歴426ターンでの実測は:
#   閾値 300s→15回 / 900s→14回 / 1800s→14回 / 3600s→12回 発火（user ターン187件が母数）
# ＝300〜3600秒の広い平坦域にあり、値の微調整に敏感でない。最小の実発火は 38.9分。
SESSION_GAP_SEC = 900.0

# system に置く役割アンカー（assistant=イブ自身を明示し、自分の発話への自答を防ぐ）。
ROLE_ANCHOR = (
    "あなたはAI VTuber「イブ」です。この会話では、assistant のメッセージは“あなた自身(イブ)の発話”、"
    "user のメッセージは“相手(ユーザ)の発話”を表します。自分(assistant)の発話に返事をしたり、"
    "自分の質問に自分で答えたりしないでください。"
)


@dataclass
class Turn:
    """会話の1ターン（話者単位の1発話）。壁時計(stamp.iso)で経過を測り現在に接地する。"""

    speaker: str
    text: str
    stamp: Stamp


@dataclass
class RagChunk:
    """RAG チャンク（永続化済み、壁時計 ISO で経過を測る）。"""

    text: str
    iso: str
    as_topic_seed: bool = False  # 無言時 random は True（思い出話と区別）
    # 1行の要約（FeedbackLLM が付ける summary）。発話判定の「話題の種」は**これだけ**を渡す:
    # text は「感情/次の予測/予測差/理由」を含む内部ログ形式で平均162字あり、話題として使える
    # のは先頭1行(平均40字)だけ＝残りがノイズになる（実測 2026-07-26）。応答文脈は従来どおり
    # text 全文（感情の色が応答の質に効く）。
    summary: str = ""

    def seed_text(self) -> str:
        """話題の種としての1行表現（summary が無い古い記録は text の1行目で代替）。"""
        s = (self.summary or "").strip()
        if s:
            return s
        lines = (self.text or "").splitlines()
        return lines[0].strip() if lines else ""


def _parse_iso(raw: str | None) -> datetime | None:
    """Turn.stamp.iso を datetime に。空/破損は None（`elapsed_wall` は 0.0＝「たった今」に
    倒す設計だが、ここでその既定を継ぐと**壊れた時刻が「間隔なし」を意味してしまい
    ガードが自分で自分を切る**ため、明示的に分ける（D1 R6）。"""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def session_gap_seconds(recent_turns, now: Stamp, threshold: float | None = None) -> float | None:
    """会話のタイムラインに閾値超の空白があればその秒数を返す（無ければ None）。

    D1（2026-07-29 実機）: 起動直後、5時間前の未応答依頼「PCの状態を教えてくれる?」を
    「さっきの」と呼んで勝手に実行した。会話ターンだけが時刻接地を持たないことが土台。

    測り方の根拠:
    - **最後の"ユーザ"発話**からの経過で測る。直前ターン（話者不問）で測ると、イブ自身の
      自発発話でタイムラインが新しくなり取り逃す（`decider.py:154-155` が同じ判断を
      「実データで1/3を取り逃す」と実測済み）。
    - 注入窓の**内部**にある空白も見る（セッション2ターン目以降は境界が窓の中に来る）。
    - **省略マーカ(OMITTED_SPEAKER)が挟まる窓では出さない**。マーカは「間に実在するターンを
      省略した」印であって「会話が途切れた」印ではない。ここで空白を主張すると、連続稼働中の
      セッションでユーザの直前の依頼を失効と宣言してしまう。
    """
    # 既定は呼び出し時にモジュール変数を引く（def 時に束縛しない＝対照実験や設定変更で
    # 差し替えられる）。
    threshold = SESSION_GAP_SEC if threshold is None else threshold
    turns = list(recent_turns or [])
    if not turns:
        return None
    if any(getattr(t, "speaker", "") == OMITTED_SPEAKER for t in turns):
        return None  # 省略された実ターンがある＝空白ではない
    now_dt = _parse_iso(now.iso)
    if now_dt is None:
        return None
    stamps = [(t, _parse_iso(getattr(getattr(t, "stamp", None), "iso", None))) for t in turns]
    if any(dt is None for _, dt in stamps):
        # 壊れた/欠損した時刻が混ざる窓は判定を諦める。**黙って 0 にしない**（無言で
        # ガードが外れるのを防ぐ＝観測できる形で降りる）。
        logger.info("⏳ 会話ギャップ判定を見送り（時刻が欠損/破損したターンを含む）")
        return None
    gaps: list[float] = []
    last_user = next((dt for t, dt in reversed(stamps) if getattr(t, "speaker", "") != SPEAKER_EVE), None)
    if last_user is not None:
        gaps.append((now_dt - last_user).total_seconds())
    for (_, a), (_, b) in zip(stamps, stamps[1:]):
        gaps.append((b - a).total_seconds())
    widest = max(gaps, default=0.0)
    return widest if widest > threshold else None


def messages_to_text(messages: list[dict]) -> str:
    """messages を1本のテキストに（テスト/デバッグ用の可読化）。"""
    return "\n\n".join(f"<{m.get('role')}>\n{m.get('content', '')}" for m in messages)


class ContextAssembler:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt

    def _build_system(self, *, rag_chunks, last_feedback, vision, speech_decision_reason, now,
                      callfunction_result=None, tools_active=False, active_tasks=None,
                      last_feedback_iso=None, session_gap_sec=None) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.append(ROLE_ANCHOR)
        if rag_chunks:
            lines = [
                f"[{'話題の種' if c.as_topic_seed else '過去の記憶'}/"
                f"{humanize(elapsed_wall(c.iso, now.iso))}] {c.text}"
                for c in rag_chunks
            ]
            parts.append("# 参照（今の会話に絡める。思い出話に逸れない）\n" + "\n".join(lines))
        if vision:
            parts.append(
                f"# 画面（今この瞬間）\n{vision}\n"
                "（会話に必要なら自然に織り込む。話題と無関係なら無理に触れなくてよい＝画面に引っ張られすぎない）"
            )
        elif vision == "":
            # 3値規約（active_tasks と同じ）: None=VLM 未配線でブロック自体を出さない /
            # ""=画面情報が今は無いことの**明示**。実機事故(2026-07-27 camp_src VLM-1):
            # 画面情報が渡っていないのに「今はブラウザで近くのカフェの画像検索が見えてるよ」と
            # 答えた。文言の出所は `# 参照` に入っていた33日前の記憶チャンクで、**過去の画面を
            # 現在として語った**。不在が伝わらないと、RAG にある過去の画面描写が現在の答えに
            # なる（判定LLM側には同じ規則があるのに応答側だけ無かった非対称）。
            # 最後の1文は必須: これが無いと画面を匂わせる質問すべてを「見えないから分からない」で
            # 潰し、会話から推測して寄り添う応答が消える（実測）。
            parts.append(
                "# 画面（今この瞬間）\n（画面情報なし：今この瞬間の画面は届いていない）\n"
                "**画面に何が映っているかを、推測や過去の記憶から作らない**"
                "（# 参照 に出てくる画面の話は過去のもので、今の画面ではない）。"
                "画面の内容そのものを聞かれたら「今は画面が見えていない」と正直に言う。"
                "画面以外（会話の流れ・相手の様子）から言えることは、これまでどおり普通に話してよい。"
            )
        if last_feedback:
            # D1 の主経路: 起動時 catch-up の内省が**前セッション末尾**を要約し、それが
            # 時刻ラベル無しで「# 直近フィードバック」として出ていた。実機(2026-07-29)では
            # 挨拶の5.5秒前に書かれた内省が「ユーザは…PCの状態を教えてくれるかと発言した /
            # 次の予測: ユーザはPCの状態確認…を求めるかもしれない」と述べ、応答LLM に
            # 「未処理の依頼が今ある」と読ませていた。
            # 接地は**内省が書かれた時刻ではなく、内省が対象にした会話スパンの末尾**
            # (`PredictionState.watermark`＝`worker.py:92` の snapshot[-1].stamp.iso) で行う。
            # 書かれた時刻で測ると D1 では「たった今」となり、古い内容を新鮮だと**追認**してしまう。
            age = elapsed_wall(last_feedback_iso, now.iso) if last_feedback_iso else None
            if age is not None and age > SESSION_GAP_SEC:
                head = (f"# 直近フィードバック（{humanize(age)}の会話についての内省。"
                        "今この場で出ている依頼ではない）")
            else:
                head = "# 直近フィードバック"
            parts.append(f"{head}\n{last_feedback}")
        if session_gap_sec is not None:
            # D1: 会話ターンだけが時刻接地を持たないので、空白の存在を system で1度だけ述べる。
            # **語彙は指定しない**（「さっき」「前に」等の逐語トークンを渡すと、それが採用されて
            # 発話の入り方が単調になる＝J-2 ①-b の実測: 見本があると先頭69%が「そういえば」系。
            # 逐語見本の再導入を止めるコードゲートが tests/test_f5_speech.py:650 にある）。
            # 実際に生きている予約は `# 予約タスク` が正なので、そちらへ明示的に道を譲る。
            parts.append(
                f"# 会話の間隔（状況把握用・そのまま読み上げない）\n"
                f"直前のやりとりから約{humanize(session_gap_sec).replace('前', '')}空いている。"
                "ここから先はその続きではなく、新しく始まった会話。\n"
                "空白より前に出ていた依頼・質問・約束は、まだ生きているとは限らない。"
                "ユーザーが自分から触れたときだけその話として扱い、こちらから実行し直したり"
                "続きを始めたりしない。実際に残っている予約は「# 予約タスク」が正。"
            )
        if speech_decision_reason:
            parts.append(f"# 発話判定理由\n{speech_decision_reason}")
        if callfunction_result:
            # 再投入された機能実行結果（ユーザ発話ではない＝user 枠に入れない）。会話が無くても
            # この結果を必ず一言で報告させる（予約タスクが沈黙中に完了した時の挨拶化を防ぐ）。
            parts.append(f"# 機能実行結果（この結果をユーザに一言で報告して）\n{callfunction_result}")
        if active_tasks is not None:
            # Fix#2（2026-07-13 実機事故対応）: 予約タスクの現在状態を毎ターン注入。
            # None=タスク機能未配線（ブロック自体を出さない）/ []=ゼロ件を明示（「止めたよ」の
            # 直後に「伝えるね」と約束する矛盾発話は、ゼロ件だと知らないことが原因だった）。
            body = "\n".join(active_tasks) if active_tasks else "（予約タスクは無い）"
            parts.append(
                "# 予約タスク（現在の状態・この一覧だけが正）\n" + body + "\n"
                "この一覧に無い予約は存在しない（完了・取消済みを含む）。直近の会話と食い違うときは"
                "一覧を信じ、既に済んだ登録・変更・取消をもう一度実行しない。一覧に無いタスクの実行を"
                "約束しない。一覧は状況把握用で、そのまま読み上げない。"
            )
        if tools_active:
            parts.append(
                "# 機能（Call-Function）の使い方\n"
                "必要なら提供された関数を呼んでよい。呼ぶ前に「ちょっと調べるね」等の短い前置きを一言"
                "添えてよい。関数名・引数・JSON は読み上げない（呼び出しは自動で発話されない）。\n"
                "**タスク（調べ物・状態確認・後回しの予約）は自分でやらず、`delegate_task` に『～して』と"
                "自然文で丸ごと委譲する**（即時も『30秒後に〜』も全部これ・簡単な事も委譲してよい＝賢い"
                "タスク担当がやる）。任せたら「やっとくね」と一言添えれば、達成後に結果が後で届く。\n"
                "**委譲する goal は、それ単体で意味が通る一文にする**（タスク担当は今の会話を見られない）。"
                "『それと〜も』『あと〜』のような追加依頼のときは、**新しく頼む部分だけ**を自己完結の一文に"
                "して渡し、既に予約タスク一覧にある実行中/予約済みの内容を goal に混ぜて再依頼しない"
                "（例: 実行中に『それと沖縄の人口も』→ goal=『沖縄県の人口を調べて教えて』。"
                "実行中の『北海道の面積』は含めない）。\n"
                "取り消しは `cancel_task` に、ユーザの言い回し（例『さっきの時間のやつ』）をそのまま渡す。"
            )
        return "\n\n".join(parts)

    def _build_conversation(self, recent_turns) -> list[dict]:
        """直近会話を native ロール列に。連続同 role はマージ（provider 互換）。"""
        convo: list[dict] = []
        pending_note: str | None = None
        for t in recent_turns or []:
            if t.speaker == OMITTED_SPEAKER:
                # 非連続なタイムラインを正直に伝える（次の発話に注記を前置）。
                pending_note = f"（中略: ユーザー沈黙中の発話 {t.text}件を省略）"
                continue
            role = "assistant" if t.speaker == SPEAKER_EVE else "user"
            # ターン本文はそのまま（相対時刻の前置きは付けない＝応答LLMが「（たった今）」等を
            # 復唱してしまう leak を防ぐ。直近会話は本来"直近"で、新旧の接地は RAG 側で行う）。
            seg = t.text
            if pending_note:
                seg = pending_note + "\n" + seg
                pending_note = None
            if convo and convo[-1]["role"] == role:
                convo[-1]["content"] += "\n" + seg
            else:
                convo.append({"role": role, "content": seg})
        return convo

    def assemble(
        self,
        *,
        user_text: str | None = None,
        autonomous_content: str | None = None,
        recent_turns: list[Turn] | None = None,
        rag_chunks: list[RagChunk] | None = None,
        last_feedback: str | None = None,
        vision: str | None = None,
        speech_decision_reason: str | None = None,
        stale_reference: bool = False,  # ②-6: 下書きの時間表現が経過と食い違う（訂正許可を出す）
        callfunction_result: str | None = None,
        tools_active: bool = False,
        active_tasks: list[str] | None = None,
        last_feedback_iso: str | None = None,  # D1: 内省が対象にした会話スパンの末尾(watermark)
        now: Stamp | None = None,
    ) -> list[dict]:
        now = now or Stamp.now()
        # D1: 自発発話経路には出さない。判定LLM は「何時間も/何日も前の会話は返事待ちではない…
        # そこで止まっている話題を振るのはむしろ自然」と**逆を明示的に許可**されており
        # (`decider.py:218-220`)、しかも「間隔ブロックが出る条件」＝「古い話題を振ってよい条件」
        # なので衝突は常態になる。ここで蒸し返しを禁じると T2（古い記憶の自律発話）を壊す。
        gap = None if autonomous_content is not None else session_gap_seconds(recent_turns, now)
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._build_system(
                    rag_chunks=rag_chunks, last_feedback=last_feedback, vision=vision,
                    speech_decision_reason=speech_decision_reason, now=now,
                    callfunction_result=callfunction_result, tools_active=tools_active,
                    active_tasks=active_tasks, last_feedback_iso=last_feedback_iso,
                    session_gap_sec=gap,
                ),
            }
        ]
        messages.extend(self._build_conversation(recent_turns))
        if autonomous_content is not None:
            # 自発発話: ユーザ発話でなく“イブが自分から言う一言”の指示（話者取り違え防止）。
            messages.append({
                "role": "user",
                "content": (
                    "（指示）今は沈黙が続いています。直前の会話を踏まえ、ユーザ発話への返事ではなく、"
                    "あなた（イブ）から自分の言葉で自然に一言だけ話してください。\n"
                    f"下書き（これを自分の発話にする）: {autonomous_content}"
                    + (
                        # ②-6: コード側のゲートが立った時だけ出す（常時出すと新しい会話でも
                        # 不要に「前に」と言い出す）。下書きは判定LLMが「10日前」と正しく
                        # 認識しながら「さっきの件」と書くことがある（実測）。
                        "\n（注意）下書きの時間表現が実際の経過と食い違っています。何日も前の話題を"
                        "「さっき」「今の」「直前」とは呼ばず、「前に」「この前」に直して話してください。"
                        if stale_reference else ""
                    )
                ),
            })
        elif user_text is not None:
            messages.append({"role": "user", "content": user_text})
        return messages
