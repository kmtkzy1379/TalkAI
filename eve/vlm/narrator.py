"""narrator — 複数フレームを1回の VLM 呼び出しで実況(VisionResult)化する。

`vlm_leaf`(既定 gemini-2.5-flash) に **時系列の連続フレーム**（古い順・最後が今）を base64 JPEG の
multimodal message で渡し、タグ付きテキストを `parse_vision` で頑健に読む。`narrate_fn` を注入できる
（テストは litellm 抜き）。ハルシネ防止(A11): プロンプトで「黒/判読不能なら推測せず visible: no」。
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from .parser import parse_vision
from .types import Frame, VisionResult

logger = logging.getLogger(__name__)

NarrateFn = Callable[[list[Frame]], Awaitable[VisionResult]]

# 2026-09-30 改善（ユーザ指示「画面認識プロンプトを最適化」）。映画（Sintel）の3コマ×10場面の実測で見えた誤りへの対処:
# (1) 旧「直前のフレームと見比べて」で**最後の2枚だけ**比べ、3コマの途中の切り替わりを見落とした（該当場面 12/14）→ 最初から最後まで
# (2) 根拠なく「ゲーム画面」と断定（約25/140）→ 種類は根拠が見える時だけ (3) 動く映像で要点（誰が・何を・どこで）が抜けた
# (4) 暗くて形の分からない物を別の物と断定（黒い袖を「瓦礫」）→ ぼかす例を足した。作業画面（メモ帳の書き換え）は変わらないことを確認済み
#
# 2026-09-30 ユーザ裁定「次の画面を予想させ、乖離しているほど surprise を強める（他の AI には送らない）」
# 「Narration は1〜2文に絞らず、重要な要素を簡潔に事実のみ（感情を入れない）」。実測（`tools/prompt_eval/vlm.py`）:
# 予想を持たない旧 surprise は「予想外の変化（エラー表示・別アプリへの切り替え）」と「普通の続き」を
# **見分けられなかった**（AUC 0.44・エラーが出ても 0 と付ける）。予想との食い違いにすると AUC 0.955
# （予想外 平均82 / 普通の続き 平均20）。実況は1〜2文の縛りを外しても長さはほぼ同じ（中央 76→80字）・主観語 0/20。
VLM_SYSTEM = (
    "あなたはユーザのPC画面を見ている『目』です。渡される複数枚の画像は時系列の連続スクリーン"
    "ショット（古い順・**最後の1枚が今この瞬間**）。**最初の1枚から最後の1枚までを順に見比べて**、"
    "画面で何が起きているか・どこが変わった/動いたかを日本語で述べてください。"
    "**重要な要素を簡潔に、見えている事実だけ**を書く（文の数は決めない。感想・感情・評価は入れない）。\n"
    "途中で画面や場面が**切り替わっていたら**、切り替わる前と後の両方を短く言う"
    "（最後の2枚が同じでも、最初の1枚と違えば変化がある）。\n"
    "**変化が無くても、今画面に表示されている主要なもの（最前面のアプリ/ファイル/内容）は必ず一言入れる**"
    "（『変化なし』だけで終わらせない＝後で『今何が見える?』に答えられるように）。\n"
    "動画やアニメなど**動く映像**が映っている時は、誰が/何が（見た目の特徴）・何をしているか・どんな場所/時間帯かを具体的に言う。\n"
    "複数ウィンドウが開いているのは普通です。最前面/アクティブを主にしつつ、必要なら背景も踏まえる。\n"
    "**確信が持てない所は決めつけない**: アプリ名・ファイル名・読みづらい文字・暗くて形の分からない物は、不確かなら名指しせず"
    "『〜のようなエディタ』『黒っぽい大きな物』等にぼかし、どこがどう分からないかを正直に言う（捏造しない）。\n"
    "**映像の種類（ゲーム/映画/アニメ/配信など）は、画面に根拠（操作画面・再生バー・ロゴ・字幕など）が見える時だけ言う**。"
    "根拠が無ければ『映像』『CGの映像』のように言う。\n"
    "画面が真っ黒・判読不能・取得できない場合は visible: no と返す。\n"
    "出力は次のタグのみ:\n"
    "narration: <実況（1行・改行しない）>\n"
    "notable: <yes/no＝今わざわざ**声をかける**価値がある変化か。**中身が新しくなった時だけ yes**。"
    "同じ内容の移動/リサイズ/スクロール・カーソルや選択の移動・時計/進捗/カウンタの更新のような"
    "『位置や見た目が変わっただけ』は no>\n"
    "prediction: <この後の画面がどうなりそうかの予想を短く1つ（見えている事実を根拠に）>\n"
    "surprise: <0-100＝「前回の予想」と今の画面の食い違い（0=予想どおり 100=まったく予想外。前回の予想が無ければ 20）>\n"
    "visible: <yes/no＝画面の内容を読み取れたか>"
)


# Alchemy（ユーザ作のブラウザ対戦ゲーム・`eve/alchemy/`）が**ユーザの画面に**映っているかを聞く1行（ALCHEMY_ENABLED の時だけ足す）。
# ユーザ指定「画面や会話からAlchemyをやりそうになりそうならプレイできるように準備」の画面の側。語の一覧で実況文を
# 判定せず、画像を見ている画面認識に yes/no で答えさせる（notable / visible と同じ形）。映っていてもゲームは始めない
# （ルールを会話に入れるだけ）。見た目の手がかりはゲームの画面（FlaskRondo `client/`）から書いた
ALCHEMY_TAG = (
    "\nalchemy: <yes/no＝ブラウザゲーム『Alchemy』の画面が映っているか（錬金術がテーマの対戦カードゲーム。"
    "タイトルや部屋の画面に Alchemy の文字や『○○の部屋』、対戦中はレシピのカード・素材箱・フラスコ・ROUND の表示）。"
    "錬金術の本や、別のゲーム・動画なら no>"
)
# イブ自身のゲーム画面を見ている間は、実況の頭にこれを付けて会話へ渡す（ユーザの画面と取り違えない）
GAME_SCREEN_PREFIX = "（イブ自身のゲーム画面）"


def build_messages(frames: list[Frame], own_windows: tuple = (), prediction: Optional[str] = None,
                   screen_note: Optional[str] = None, detect_alchemy: bool = False) -> list[dict]:
    """litellm multimodal messages（OpenAI 形式; gemini 経路で inline_data に変換される）。

    `own_windows` = イブ自身が `show_on_screen` で開いている窓のファイル名。
    `prediction` = 前回の呼び出しで画面認識自身が出した「この後の画面の予想」（無ければ初回扱い）。
    surprise はこれと今の画面の食い違いとして付けさせる（予想を持たないと surprise が意味を持たない・実測 AUC 0.44）。

    E11（実測 2026-08-06/08-07）: これを渡さないと、イブが7秒前に自分で開いた窓を VLM が
    「デスクトップ上に…が新規に開かれています」とユーザの画面変化として実況し、イブが
    自分の出力を相手の作業として語る自己観察ループになる。窓を消すのではなく**誰が開いたかを
    教える**方が副作用が無い（ユーザが見ている窓を勝手に閉じずに済む）。
    """
    head = f"連続スクリーンショット {len(frames)} 枚（古い順・最後が今）。今の画面を実況して。"
    head += f"\n前回のあなたの予想: {(prediction or '').strip() or 'なし（初回）'}"
    if screen_note:
        # イブ自身のゲーム画面（Alchemy の部屋・対戦中）。ユーザの画面ではないことを先に断る
        head += "\n" + screen_note
    elif own_windows:
        head += ("\nなお次のウィンドウは**あなた自身（イブ）が開いたもの**で、ユーザの作業ではない: "
                 + "・".join(own_windows)
                 + "。これらを『ユーザが開いた/新しく現れた』とは言わず、実況の主役にしない。")
    content: list[dict] = [{"type": "text", "text": head}]
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f.jpeg_b64}"},
        })
    system = VLM_SYSTEM + (ALCHEMY_TAG if detect_alchemy and not screen_note else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _content_of(resp: object) -> str:
    """litellm 応答からテキストを頑健に取り出す（壊れ形は空文字）。"""
    try:
        return resp.choices[0].message.content or ""  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def make_narrate_fn(registry, own_windows_fn=None, prediction_fn=None, screen_note_fn=None,
                    detect_alchemy: bool = False) -> NarrateFn:
    """ModelRegistry の `vlm_leaf` を叩く本番 narrate_fn。

    `own_windows_fn` は呼び出しのたびに評価する（窓は開閉するのでスナップショットにしない）。
    未接続（作業フォルダ無効）なら従来どおり何も足さない。
    `prediction_fn` は前回の予想を返す（`VlmWorker.last_prediction`）。無ければ毎回「初回」扱い。
    `screen_note_fn` はイブ自身のゲーム画面を見ている間だけ断り書きを返す（`AlchemyManager.screen_note`）。
    その間の実況には `GAME_SCREEN_PREFIX` を付ける（会話でユーザの画面と取り違えない）。
    `detect_alchemy` は「Alchemy が映っているか」の1行を足す（ALCHEMY_ENABLED の時だけ）。
    """
    async def narrate(frames: list[Frame]) -> VisionResult:
        own: tuple = ()
        if own_windows_fn is not None:
            try:
                own = tuple(own_windows_fn())
            except Exception:  # 自己認識の失敗で画面認識ごと落とさない
                logger.debug("自分の窓一覧の取得に失敗（注入なしで継続）", exc_info=True)
        pred: Optional[str] = None
        if prediction_fn is not None:
            try:
                pred = prediction_fn() or None
            except Exception:  # 予想が取れなくても実況は続ける（初回扱いになるだけ）
                logger.debug("前回の予想の取得に失敗（初回扱いで継続）", exc_info=True)
        note: Optional[str] = None
        if screen_note_fn is not None:
            try:
                note = screen_note_fn() or None
            except Exception:  # 断り書きが取れなくても実況は続ける
                logger.debug("画面の断り書きの取得に失敗（無しで継続）", exc_info=True)
        resp = await registry.complete(
            "vlm_leaf", build_messages(frames, own, pred, screen_note=note, detect_alchemy=detect_alchemy))
        result = parse_vision(_content_of(resp))
        if note and result.narration:
            result.narration = GAME_SCREEN_PREFIX + result.narration
        return result

    return narrate
