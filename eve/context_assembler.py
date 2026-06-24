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

    def _build_system(self, *, rag_chunks, last_feedback, vision, speech_decision_reason, now) -> str:
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
        now: Stamp | None = None,
    ) -> list[dict]:
        now = now or Stamp.now()
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._build_system(
                    rag_chunks=rag_chunks, last_feedback=last_feedback, vision=vision,
                    speech_decision_reason=speech_decision_reason, now=now,
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
