"""応答LLM への文脈を組み立てる器（F0 雛形）。

過去参照防止(T6)の責任点:
- 全要素に時刻を持たせ、組立時に相対時刻(「3分前」)を注入して現在に接地する。
- 無言時の random RAG は「話題の種」と明示ラベルし、「過去の記憶(思い出話)」と峻別する。
  ＝ v1 の「過去の記憶から話が逸れる」を構造的に抑止。

ここでは描画(文字列化)までを担う。実際の LLM 呼び出しは ResponseOrchestrator 側。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clock import Stamp, elapsed_mono, elapsed_wall, humanize


@dataclass
class Turn:
    """直近会話の1ターン（in-session、monotonic で経過を測る）。"""

    speaker: str
    text: str
    stamp: Stamp


@dataclass
class RagChunk:
    """RAG チャンク（永続化済み、壁時計 ISO で経過を測る）。"""

    text: str
    iso: str
    as_topic_seed: bool = False  # 無言時 random は True（思い出話と区別）


@dataclass
class AssembledContext:
    system: str
    blocks: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n\n".join(self.blocks)


class ContextAssembler:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt

    def assemble(
        self,
        *,
        user_text: str | None = None,
        recent_turns: list[Turn] | None = None,
        rag_chunks: list[RagChunk] | None = None,
        last_feedback: str | None = None,
        vision: str | None = None,
        speech_decision_reason: str | None = None,
        now: Stamp | None = None,
    ) -> AssembledContext:
        now = now or Stamp.now()
        blocks: list[str] = []

        if recent_turns:
            lines = [
                f"[{humanize(elapsed_mono(t.stamp, now))}] {t.speaker}: {t.text}"
                for t in recent_turns
            ]
            blocks.append("# 直近の会話\n" + "\n".join(lines))

        if rag_chunks:
            lines = []
            for c in rag_chunks:
                label = "話題の種" if c.as_topic_seed else "過去の記憶"
                lines.append(f"[{label}/{humanize(elapsed_wall(c.iso, now.iso))}] {c.text}")
            blocks.append(
                "# 参照（今の会話に絡める。思い出話に逸れない）\n" + "\n".join(lines)
            )

        if vision:
            blocks.append(f"# 画面（今この瞬間）\n{vision}")
        if last_feedback:
            blocks.append(f"# 直近フィードバック\n{last_feedback}")
        if speech_decision_reason:
            blocks.append(f"# 発話判定理由\n{speech_decision_reason}")
        if user_text is not None:
            blocks.append(f"# ユーザ発話（今）\n{user_text}")

        return AssembledContext(system=self.system_prompt, blocks=blocks)
