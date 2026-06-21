"""応答LLM への文脈を組み立てる器（F0 雛形）。

過去参照防止(T6)の責任点:
- 全要素に時刻を持たせ、組立時に相対時刻(「3分前」)を注入して現在に接地する。
- 無言時の random RAG は「話題の種」と明示ラベルし、「過去の記憶(思い出話)」と峻別する。
  ＝ v1 の「過去の記憶から話が逸れる」を構造的に抑止。

ここでは描画(文字列化)までを担う。実際の LLM 呼び出しは ResponseOrchestrator 側。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clock import Stamp, elapsed_wall, humanize

# 注入セレクション（ConversationCache.recent_for_injection）が差し込む省略マーカの話者名。
# ユーザー沈黙中の自律発話などで直近窓とアンカー往復が非連続になった時、その間に
# 何件省略したかを伝えるセンチネル Turn（text=省略件数）。描画は assemble 内で特別扱い。
OMITTED_SPEAKER = "__omitted__"


@dataclass
class Turn:
    """会話の1ターン（話者単位の1発話）。

    壁時計(stamp.iso)で経過を測り「3分前」等に接地する。再起動でディスクから復元した
    Turn の monotonic は前プロセスの値で無意味なので、表示用の経過は壁時計を正とする
    （monotonic は deadline 計算など in-session 期間用途に限定）。
    """

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
        autonomous_content: str | None = None,
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
            lines = []
            for t in recent_turns:
                if t.speaker == OMITTED_SPEAKER:
                    # 非連続なタイムラインを AI に正直に伝える（地続きと誤解させない）
                    lines.append(f"（中略: ユーザー沈黙中の発話 {t.text}件を省略）")
                    continue
                lines.append(
                    f"[{humanize(elapsed_wall(t.stamp.iso, now.iso))}] {t.speaker}: {t.text}"
                )
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
        # 自発発話: イブ自身が"自分から"言う下書き。**ユーザ発話ではない**ので、これに返事を
        # するのでなく、直前の会話に合うイブ自身の自然な一言にして言う（話者ロール取り違え防止）。
        if autonomous_content is not None:
            blocks.append(
                "# 自分から話す（イブ自身の発話・ユーザの発話ではない）\n"
                "次の下書きを、直前の会話に合うイブ自身の自然な話し言葉にして"
                "“あなた（イブ）が自分から”言う。ユーザ発話への返事にはしない。\n"
                f"下書き: {autonomous_content}"
            )
        elif user_text is not None:
            blocks.append(f"# ユーザ発話（今）\n{user_text}")

        return AssembledContext(system=self.system_prompt, blocks=blocks)
