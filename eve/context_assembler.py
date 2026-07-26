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

from dataclasses import dataclass

from .clock import Stamp, elapsed_wall, humanize

# 注入セレクション（ConversationCache.recent_for_injection）が差し込む省略マーカの話者名。
OMITTED_SPEAKER = "__omitted__"
SPEAKER_EVE = "eve"

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


def messages_to_text(messages: list[dict]) -> str:
    """messages を1本のテキストに（テスト/デバッグ用の可読化）。"""
    return "\n\n".join(f"<{m.get('role')}>\n{m.get('content', '')}" for m in messages)


class ContextAssembler:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt

    def _build_system(self, *, rag_chunks, last_feedback, vision, speech_decision_reason, now,
                      callfunction_result=None, tools_active=False, active_tasks=None) -> str:
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
        if last_feedback:
            parts.append(f"# 直近フィードバック\n{last_feedback}")
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
        callfunction_result: str | None = None,
        tools_active: bool = False,
        active_tasks: list[str] | None = None,
        now: Stamp | None = None,
    ) -> list[dict]:
        now = now or Stamp.now()
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._build_system(
                    rag_chunks=rag_chunks, last_feedback=last_feedback, vision=vision,
                    speech_decision_reason=speech_decision_reason, now=now,
                    callfunction_result=callfunction_result, tools_active=tools_active,
                    active_tasks=active_tasks,
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
                ),
            })
        elif user_text is not None:
            messages.append({"role": "user", "content": user_text})
        return messages
