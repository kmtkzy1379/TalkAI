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

# D1: 「前回の会話は古い」とみなす閾値。`# 直近フィードバック` の時刻ラベルと
# `# 前回の会話` ブロックの発火に使う。実測で 300〜3600秒の広い平坦域にあり値に敏感でない。
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


def messages_to_text(messages: list[dict]) -> str:
    """messages を1本のテキストに（テスト/デバッグ用の可読化）。"""
    return "\n\n".join(f"<{m.get('role')}>\n{m.get('content', '')}" for m in messages)


class ContextAssembler:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt

    def _build_system(self, *, rag_chunks, last_feedback, vision, speech_decision_reason, now,
                      callfunction_result=None, tools_active=False, active_tasks=None,
                      last_feedback_iso=None, session_gap_sec=None, capabilities=None) -> str:
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
            # D1（案2）: 前セッションの会話は**注入していない**（`recent_for_injection` が
            # 復元分を落とす）。旧実装は復元会話を注入したまま「間が空いた」と注記していたが、
            # 実測（オフライン n=8）で 8/8 蒸し返して**無効**だった。注入しなければ 0/8。
            # よってここは「駆動源を止める装置」ではなく、**前回いつまで話したかを1度述べる
            # 事実の提示**に徹する（命令を減らす＝過剰適用と読み上げ leak を避ける）。
            # 語彙は指定しない（「さっき」「前に」等の逐語トークンを渡すと採用されて入り方が
            # 単調になる実測＝J-2 ①-b・コードゲートは tests/test_f5_speech.py:650）。
            parts.append(
                f"# 前回の会話（状況把握用・そのまま読み上げない）\n"
                f"前回このユーザと話したのは約{humanize(session_gap_sec).replace('前', '')}前。"
                "その会話の逐語は残っていない（覚えているのは「# 参照」の記憶だけ）。"
                "今は新しく始まった会話で、その続きではない。"
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
            if capabilities:
                # D3（2026-07-29 実機）: 能力の無い依頼（メール送信）を「いいよ」と承諾した。
                # 危険系（ファイル削除）は断れるので「危険だから断る」は効き「**手段が無いから
                # 断る**」だけが無かった。しかも断った側も「指定してくれたら手伝う」と実在しない
                # 手段を約束していた。
                #
                # 文面の制約（すべて実測に基づく・緩めると壊れる）:
                # - 一覧は registry 導出（`outward_actions`）。手書きは構成差と J-3 で必ず腐る。
                # - 「一覧」という語を使わない。`# 予約タスク` が既に同じ語で別の意味
                #   （その場の予約状態）を支配しており、混同すると Fix#2 の抑止が緩む。
                # - **制限対象は「外の世界への操作」だけ**と明示する。ここを「できること全部」に
                #   広げると会話・知識・記憶まで拒否に倒れる。実データ: ユーザ188発話のうち
                #   会話/知識/記憶で答えるべき依頼は49件(26.1%)、真に不能なのは4件(2.1%)＝1:12。
                # - **考えて答えるだけの予約は引き受けてよい**と明示する。実データで
                #   「30秒後に好きな焼肉の部位を教えて」等3件が、どの手段にも対応しないまま
                #   TaskAgent の知識で Done している（これを潰すと正当な委譲が死ぬ）。
                # - 逃がしの1文（最後の段落）は `# 画面` の不在マーカと同じ役割。あちらは
                #   最後の1文が無いと画面関連の質問を全部潰した実測があり、回帰テストで固定
                #   されている。ここも同様にテストで固定する（削除禁止）。
                parts.append(
                    "# 実際に実行できる手段（行動の話）\n"
                    "実行できるのは次だけ: " + " / ".join(capabilities) + "。\n"
                    "これ以外の操作（メール送信、ファイルの作成・移動・削除、アプリの起動や操作、"
                    "買い物、外部サービスへの書き込み等）は**手段を持っていない**。頼まれたら"
                    "「その手段は持っていない」と正直に短く言い、できるふりや「あとでやっておく」"
                    "「条件を教えてくれたらやる」と引き受け直さない。危ないから断るときは危ない理由も言う。\n"
                    "**制限しているのは「外の世界を操作すること」だけ。** 会話・意見・おすすめ・感想・"
                    "気持ち・知っていることの説明・昔のやりとりの思い出しは、この制限と関係なく"
                    "これまでどおり普通に答える。『30秒後におすすめの晩ごはん教えて』のような"
                    "考えて答えるだけの頼まれ事も普通に引き受けてよい。"
                    "「知らない」と「手段が無い」は別のこと。"
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
        capabilities: list[str] | None = None,  # D3: 外の世界に手を出せる手段（registry 導出）
        prev_session_gap_sec: float | None = None,  # D1: 前セッション最終発話からの経過秒
        now: Stamp | None = None,
    ) -> list[dict]:
        now = now or Stamp.now()
        # D1: 自発発話経路には出さない。判定LLM は「何時間も/何日も前の会話は返事待ちではない…
        # そこで止まっている話題を振るのはむしろ自然」と**逆を明示的に許可**されており
        # (`decider.py:218-220`)、しかも「間隔ブロックが出る条件」＝「古い話題を振ってよい条件」
        # なので衝突は常態になる。ここで蒸し返しを禁じると T2（古い記憶の自律発話）を壊す。
        # D1: 前セッション境界からの経過（呼び手＝orchestrator が cache から渡す）。
        # 自発経路には出さない（判定LLM は decider.py:218-220 で「古い話題を振ってよい」と
        # 逆を明示指示されており、発火条件が重なるので衝突が常態になる＝T2 を壊す）。
        gap = None if autonomous_content is not None else prev_session_gap_sec
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._build_system(
                    rag_chunks=rag_chunks, last_feedback=last_feedback, vision=vision,
                    speech_decision_reason=speech_decision_reason, now=now,
                    callfunction_result=callfunction_result, tools_active=tools_active,
                    active_tasks=active_tasks, last_feedback_iso=last_feedback_iso,
                    session_gap_sec=gap, capabilities=capabilities,
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
