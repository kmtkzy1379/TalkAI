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

from .capability.registry import NEVER_MEANS
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

# S3: ユーザの話し終わりにイブが既に口にしたつなぎ言葉（応答非依存フィラー）。
# 前置きは user メッセージにだけ付ける（payload・会話履歴・RAG には入らない）。書き出しとの重複は
# まずプロンプトで抑える（ユーザ裁定 2026-09-25「コード肥大するからあまりやりたくない」）。
FILLER_PREFIX = "（直前にあなたが口にしたつなぎ言葉: {}）"
# 機能の報告ターンで会話がイブの発言で終わる時に足す、ユーザ側の指示（ユーザの発言ではない）
REPORT_CUE = "（指示）頼まれていた機能の結果が届きました。上の「# 機能実行結果」を、あなた（イブ）から自分の言葉で伝えてください。"
FILLER_RULE = (
    "# つなぎ言葉\n"
    "あなたはユーザの話を聞き終えた直後、考えている間のつなぎ言葉（「あー」「えー」など）を"
    "既に声に出していることがある。その時は user メッセージの先頭に"
    "（直前にあなたが口にしたつなぎ言葉: …）と示す。応答はその続きとして話し、"
    "**同じつなぎ言葉や似たつなぎ言葉で書き出さない**（例: 「あー」の後に「あー、それは…」と始めない）。"
    "つなぎ言葉そのものを読み上げたり、それについて説明したりしない。"
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


def silence_facts(recent_turns, now: str | None = None, *, me: str = "イブ") -> str:
    """相手（ユーザ）が最後に話してからの経過と、そのあとイブだけが話した回数（事実だけ・どうするかは書かない）。

    沈黙の時計（`SpeechState`）はイブが話しても戻るので、それだけでは「相手がずっと黙っている」ことが
    見えない（2026-10-05 実測: 返事の無いまま話しかけ続けても、無視されていることに一度も触れなかった）。
    省略マーカ（間のイブの発話の件数）も数える。me は文中のイブの呼び方（応答の指示では「あなた」）。
    """
    last_user = None
    eve_after = 0
    for t in recent_turns or []:
        if t.speaker == "user":
            last_user, eve_after = t, 0
        elif t.speaker == SPEAKER_EVE:
            eve_after += 1
        elif t.speaker == OMITTED_SPEAKER:
            eve_after += int(t.text) if str(t.text).isdigit() else 0
    if last_user is None:
        return "この起動から、相手はまだ何も話していない。"
    ago = humanize(elapsed_wall(getattr(last_user.stamp, "iso", "") or "", now))
    if eve_after == 0:
        return f"相手が最後に話したのは{ago}。"
    return f"相手が最後に話したのは{ago}。それ以降、相手は何も言っておらず、{me}だけが{eve_after}回話した。"


class ContextAssembler:
    def __init__(self, system_prompt: str = "", reply_opener: str = "") -> None:
        self.system_prompt = system_prompt
        # S4: 応答の書き出しに置く応答依存フィラーの表（`eve/filler/prompt.py`）。
        # **ユーザ発話への応答のときだけ**入れる（自発発話・機能報告の書き出しに「へー」は付かない）。
        self.reply_opener = reply_opener

    def _build_system(self, *, rag_chunks, last_feedback, vision, vision_stale,
                      speech_decision_reason, now,
                      callfunction_result=None, tools_active=False, active_tasks=None,
                      last_feedback_iso=None, session_gap_sec=None, capabilities=None,
                      features_off=None, self_model=None, spoken_filler=None,
                      user_reply=False, emotion=None, style_examples=None) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if style_examples:
            # 話し方の実例（`eve/persona_examples.py`）。性格（system_prompt の最後）の直後に置く＝測った置き方
            # （`tools/prompt_eval/` の N5＋実例）。毎ターン入れ替わる。
            parts.append(style_examples)
        parts.append(ROLE_ANCHOR)
        if user_reply and self.reply_opener:
            parts.append(self.reply_opener)
        if spoken_filler:
            # このターンで実際に口にした時だけ出す（常時出すとプロンプトが太るだけ）
            parts.append(FILLER_RULE)
        if emotion:
            # S5（ユーザ裁定 2026-09-26「その感情によって…応答LLM…に入れたりして感情にそった応答をさせる」）。
            # ユーザの話の「、」「。」ごと・フィードバックで更新されるイブの今の感情（`eve/emotion.py`）
            parts.append(f"# イブの今の気持ち\n{emotion}。この気持ちに沿って話す（気持ちを言葉で説明しなくてよい）")
        if features_off:
            # UI で機能を OFF にしている間の断り書き（ユーザ裁定 2026-08-14）。
            # **「機能が無い」と言わせない**: 「画面を見る機能がない」は事実と違い、
            # ユーザにはハルシネーションに見える。正しくは「機能はあるが今は切ってある」。
            # 一時的だと分かれば「じゃあ点けて」と言える＝会話が前に進む。
            parts.append(
                "# 今オフにしている機能\n"
                + "\n".join(f"- {f}" for f in features_off)
                + "\n（**「その機能が無い」とは言わない**。備わってはいるが今は切ってある、"
                  "という言い方をする。切り替えるのはユーザなので、必要そうなら"
                  "「点けてくれたらできるよ」と伝えてよい。勝手に点けることはできない。）"
            )
        if self_model:
            # **`# 参照`(RAG) とは別枠にする**。あちらには「今の会話に絡める」という指示が
            # 付いており、それが 52日前の「チーズケーキが美味しそう」を“イブの好物”に
            # 格上げした引力の正体だった（実機事故 2026-08-15 02:41）。ここは絡めさせず、
            # 聞かれた時に同じ答えを返すための台帳として置く。長さはコード側で有界
            # （`memory/self_model.py` の閉じたスロット × 最大2件）。
            parts.append(self_model)
        if rag_chunks:
            lines = [
                f"[{'話題の種' if c.as_topic_seed else '過去の記憶'}/"
                f"{humanize(elapsed_wall(c.iso, now.iso))}] {c.text}"
                for c in rag_chunks
            ]
            parts.append("# 参照（今の会話に絡める。思い出話に逸れない）\n" + "\n".join(lines))
        if vision and vision_stale:
            # 「画面が変わったが実況がまだ追いついていない」区間。ここを不在明示にすると、
            # 下の分岐の指示どおり「今は画面が見えていない」と**能力ごと否認**する
            # （実測 2026-09-21: 空白時 9/9＝100%。ユーザには「変わった瞬間しか見えていない」
            # と映る）。見出しを分けるのは、本文が「今この瞬間」ではないと一目で分かるため。
            # この文言での実測: 否認 0/9・中身を話せた 9/9・断り付き 9/9。
            parts.append(
                f"# 画面（直前に見えていたもの）\n{vision}\n"
                "（**能力が無いとは言わない**。画面は見えていて、今映っているものの読み取りが"
                "追いついていないだけ。直前に見えていた内容は上のとおりなので、"
                "「さっきの時点では」と断って話してよい。今この瞬間のことは断定しない。）\n"
                "（会話に必要なら自然に織り込む。話題と無関係なら無理に触れなくてよい＝画面に引っ張られすぎない）"
            )
        elif vision:
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
            #
            # 2026-09-30 文言を変えた（ユーザ指摘「画面情報なし・今この瞬間の画面は届いていない、は正解だけど誤解を
            # 生みそう」）。旧文言は「今は画面が見えていない」と言わせており、能力が無いように聞こえた（実測 3/3）。
            # ここに来るのは**本当に読めていない時だけ**（起動直後で実況がまだ無い・キャプチャ停止・真っ黒・実況が
            # 詰まった）。静止しているだけの画面は `vision_for_user` が「N秒前の画面。以降ずっと変化なし」で渡す
            # （自発発話・機能報告も同じ・`orchestrator._vision_for`）。ブロックごと消す案は、画面を聞かれた時に
            # 画面を見るタスクを勝手に始めた（3/3・タスク担当は画面を見られない）ので採らない。
            parts.append(
                "# 画面\n（今は画面の中身を読み取れていない）\n"
                "**画面に何が映っているかを、推測や過去の記憶から作らない**"
                "（# 参照 に出てくる画面の話は過去のもので、今の画面ではない）。"
                "画面のことを聞かれたら、今は読み取れていないと伝える。"
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
                # 2026-09-30 ユーザ提案で「呼ぶ前に『ちょっと調べるね』等の短い前置きを一言添えてよい」を外した
                # （引き受けの一言は下の段落が依頼の中身に合わせて言わせている＝見本の言い回しが要らない）。
                "必要なら提供された関数を呼んでよい。関数名・引数・JSON は読み上げない（呼び出しは自動で発話されない）。\n"
                # 2026-10-01 ユーザ「簡単な知識はタスクに回さず持ってる知識で答えてほしい。富士山など」＋ZG02
                # （宝くじが当たった話への返事と質問を、そのまま goal にして委譲した）。旧文言「簡単な事も委譲してよい
                # ＝賢いタスク担当がやる」が出所。一般常識をタスクに回した率: 測った場面 10/21→0/21、
                # 見ていない場面（例示に無い種類の知識）5/24→0/24。最新の情報は 9/9 のまま調べに回る。
                "**タスク（調べ物・状態確認・作業・後回しの予約）は自分でやらず、`delegate_task` に『～して』と"
                "自然文で丸ごと委譲する**（即時も『30秒後に〜』も全部これ）。"
                "ただし一般常識として確かに知っていること（有名な山の高さ・国の首都・有名なキャラの名前やタイプ・"
                "理科の仕組みなど）は、委譲せずその場で自分の知識で答える"
                "（最新の情報・日によって変わる数字・知らない名前や言葉は委譲して調べる）。"
                "**会話の返事・相づち・ユーザへの質問は委譲しない**（タスク担当はユーザと話せない。それは自分で話す）。"
                "任せる時は、関数を呼ぶ前に、依頼の中身に合った短い一言で引き受ける"
                "（毎回同じ言い方にしない）。達成後に結果が後で届く。\n"
                # 2026-09-30 ユーザ要望「知らないこと・意味が分からないことは積極的に調べさせたい」。人柄（persona.md）に
                # 「積極的に調べる」と書いても調べる率は変わらなかった（未知語 8/18 のまま）。ここに書くと 9/24→13/24。
                # 2026-10-01「存在しない単語やアニメの固有名詞でルナが分からないものは聞き返すより調べた方がいい」で
                # 「調べてよい」→「調べる」。例（作品名・人名など）を入れると、知っている作品の人物まで調べた（3/3）ので
                # 例は入れない。知らない名前を調べた率 9/12→12/12（見ていない場面 5/6→6/6）・知っている名前 0/3・0/6 のまま。
                "話に出てきた名前や言葉を自分が知らない時は、話を合わせる前に `delegate_task` で調べる"
                "（知っているものは調べずに話す。言葉として崩れていて何を言いたいのか分からない時は、調べずに聞き返す）。\n"
                "**委譲する goal は、それ単体で意味が通る一文にする**（タスク担当は今の会話を見られない）。\n"
                # 1件制（2026-08-07 ユーザ裁定）: 上の `# 予約タスク` が空でない間、`delegate_task` は
                # コードで断られる（`task/capabilities.py` の受付ゲート）。ここが旧文言のままだと
                # 「追加依頼は新しい goal で渡せ」と指示しながらコードが弾く矛盾になる。
                "**タスクは同時に1つだけ**。上の `# 予約タスク` に何か載っている間は新しい委譲を"
                "受け付けられない（頼んでも断られる）。その状態で別の用事を頼まれたら、2つ目を委譲せず、"
                "**今なにをやっているかを伝えた上で「終わったらもう一度言ってね」と頼む**。"
                # 実測 2026-08-07(e15_run3 2/2): 「終わったら調べるね」と言うが、タスクは作られず
                # **その約束を果たす仕組みが無い**（覚えておく置き場が無く、次ターンには文脈から
                # 消える）。旧文言はここで「終わったらやるね」と言わせており、こちらが嘘の出所だった。
                "**「終わったらやるね」「あとでやっておく」のような後回しの約束はしない**"
                "（頼まれごとを覚えておく仕組みが無いので、必ず破ることになる）。"
                "覚えていられないことは隠さず正直に言ってよい。"
                "急ぐなら「今のをやめて先にやろうか?」と聞く。"
                "『それと〜も』のような追加依頼も同じ扱い。\n"
                # E15（実測 2026-08-07・slot_run1/2 で 2/2 再現）: 断られるのに「やっとくね」と言い、
                # さらにその依頼の中身を**自分の記憶で答えて**済ませた（run1 は正しく断った直後にやった）。
                # 応答本文は tool より先に喋り終わる（`response/orchestrator.py:453,460`）ので、
                # 「言ってしまった後で直す」では間に合わない＝言う前に止める必要がある。
                "一覧が空でない時は**「やっとくね」「調べるね」のような引き受ける言い方をしない**"
                "（実際には受け付けられないので嘘になる）。**受けられなかった依頼の中身を、"
                "自分の記憶から答えて済ませない**（調べていないのに数字や事実を言わない）。\n"
                # 実測 2026-08-07（slot_run2 t=57.9）: 1件制のゲートに断られた応答LLM は、
                # 「実行中のタスク + 新しい依頼」を**1つの goal に合体させて再委譲**する抜け道を
                # 使った（「2つ目」ではないので上の文だけでは塞げない）。旧 J-2 ① の制約文を
                # 落としたのが原因なので、1件制の文脈に合わせて復活させる。
                "**既に一覧にある実行中/予約済みの内容を、新しい goal に混ぜて出し直さない**"
                "（例: 実行中『素数のプログラム』＋『沖縄の人口も』→ 両方を1つの goal にまとめて"
                "投げ直すのは禁止。それも断られる）。\n"
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
                    # J-3: 否定の例示は NEVER_MEANS 由来（**構成に依らず永久に持たない物だけ**）。
                    # 旧文言はここに「ファイルの作成・移動・削除」「アプリの起動や操作」を手書きして
                    # おり、作業フォルダ能力の追加で嘘になる位置にあった。可能側は上行の
                    # `outward_actions()` 導出が語るので、否定側で列挙し直す必要は無い。
                    f"これ以外の操作（{'、'.join(NEVER_MEANS)}等）は**手段を持っていない**。頼まれたら"
                    "「その手段は持っていない」と正直に短く言い、できるふりや「あとでやっておく」"
                    "「条件を教えてくれたらやる」と引き受け直さない。"
                    # 2026-10-01 ユーザ「不可能作業はタスク依頼しないでほしい。その場でなんでできないか説明させたい。
                    # できないこと引き受けるのはハルシネーション」。「手段が無い」と書いても「タスク担当ならできる」と
                    # 思って回していた。それでも「できない」と言いながら委譲も出すことがある（3/18）ので、
                    # そちらは `response/orchestrator.py` の関門（_says_cannot）が委譲を捨てる。
                    "**手段の無いことは `delegate_task` にも回さない**（回してもできないので、引き受けた時点で嘘になる）。"
                    "その場で、何が無いからできないのかを短く言う。できないと伝えた依頼で `delegate_task` を呼ばない。"
                    "危ないから断るときは危ない理由も言う。\n"
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
        # 「変化後・実況待ち」の暫定か（`vlm/state.py:vision_after_change`）。
        # True の時だけ `# 画面（直前に見えていたもの）` の断り付きブロックにする。
        vision_stale: bool = False,
        speech_decision_reason: str | None = None,
        stale_reference: bool = False,  # ②-6: 下書きの時間表現が経過と食い違う（訂正許可を出す）
        callfunction_result: str | None = None,
        tools_active: bool = False,
        active_tasks: list[str] | None = None,
        last_feedback_iso: str | None = None,  # D1: 内省が対象にした会話スパンの末尾(watermark)
        capabilities: list[str] | None = None,  # D3: 外の世界に手を出せる手段（registry 導出）
        prev_session_gap_sec: float | None = None,  # D1: 前セッション最終発話からの経過秒
        features_off: list[str] | None = None,  # UI: 今オフにしている機能（「無い」とは言わせない）
        self_model: str | None = None,  # イブ自身の好み（`memory/self_model.py` の render()）
        spoken_filler: str | None = None,  # S3: 直前にイブが口にしたつなぎ言葉（USER 発話のみ）
        emotion: str | None = None,  # S5: イブの今の気持ち（例「楽（穏やかで楽しい・リラックスしている）」）
        style_examples: str | None = None,  # 話し方の実例（`eve/persona_examples.py`・ユーザ発話への応答だけ）
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
        # つなぎ言葉はユーザ発話への応答でだけ扱う（自発発話・機能報告にはユーザの話し終わりが無い）
        filler = spoken_filler if (user_text is not None and autonomous_content is None) else None
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._build_system(
                    rag_chunks=rag_chunks, last_feedback=last_feedback, vision=vision,
                    vision_stale=vision_stale,
                    speech_decision_reason=speech_decision_reason, now=now,
                    callfunction_result=callfunction_result, tools_active=tools_active,
                    active_tasks=active_tasks, last_feedback_iso=last_feedback_iso,
                    session_gap_sec=gap, capabilities=capabilities,
                    features_off=features_off, self_model=self_model,
                    spoken_filler=filler,
                    user_reply=user_text is not None and autonomous_content is None,
                    emotion=emotion,
                    style_examples=style_examples,
                ),
            }
        ]
        messages.extend(self._build_conversation(recent_turns))
        if autonomous_content is not None:
            # 自発発話: ユーザ発話でなく“イブが自分から言う一言”の指示（話者取り違え防止）。
            messages.append({
                "role": "user",
                "content": (
                    # 2026-10-05: 沈黙の事実（相手が最後に話してから・その後イブだけが話した回数）を渡す。
                    # 会話の各行には時刻を付けない（leak 防止・_build_conversation）ので、これが無いと
                    # 相手がどれだけ黙っているか分からない。どう反応するかは書かない（イブが決める）
                    f"（指示）今は沈黙が続いています（{silence_facts(recent_turns, now.iso, me='あなた')}）"
                    "直前の会話を踏まえ、ユーザ発話への返事ではなく、"
                    "あなた（イブ）から自分の言葉で一言だけ話しかけてください（相手は今こちらに注意を向けていない）。"
                    # 2026-10-04: 応答を Gemini にしたら、自発発話でだけ問いかけを「。」で終えた（5/5・ユーザ発話への返事では
                    # 「？」を付ける）。読み上げは「？」で語尾が上がるので、問いかけが平らに読まれる
                    "問いかける時は文末に「？」を付ける。\n"
                    # 2026-10-05: 発話判定は「話すか・何について」までを決め、言い回しはイブ（応答モデル）が決める。
                    # 判定の書いた文をそのまま言わせると、判定モデルの言い回し（「そういえば〜？」ばかり）と
                    # 口調（タメ口）がそのまま出ていた
                    f"話しかける中身（発話判定のメモ。言い回しや切り出し方はあなたが決める）: {autonomous_content}"
                    + (
                        # ②-6: コード側のゲートが立った時だけ出す（常時出すと新しい会話でも
                        # 不要に「前に」と言い出す）。メモは判定LLMが「10日前」と正しく
                        # 認識しながら「さっきの件」と書くことがある（実測）。
                        "\n（注意）メモの時間表現が実際の経過と食い違っています。何日も前の話題を"
                        "「さっき」「今の」「直前」とは呼ばず、「前に」「この前」に直して話してください。"
                        if stale_reference else ""
                    )
                ),
            })
        elif user_text is not None:
            content = user_text
            if filler:
                # 再配達の前置き（`response/orchestrator.py` の機能報告）と同じ形: 括弧書きを前に置く
                content = FILLER_PREFIX.format(filler) + "\n" + user_text
            messages.append({"role": "user", "content": content})
        elif callfunction_result is not None and messages[-1]["role"] != "user":
            # 機能の報告ターンは中身を system の「# 機能実行結果」に置くので、会話がイブの発言（または無し）で終わる。
            # Gemini は最後がモデルの発言で終わる依頼を受けない（「Requests ending with a model turn are not supported」・
            # 2026-10-04 実測で報告 2/2 が失敗）。ユーザの発言と取り違えないよう、自発発話と同じ（指示）の形で足す
            messages.append({"role": "user", "content": REPORT_CUE})
        return messages
