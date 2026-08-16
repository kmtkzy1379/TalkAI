"""設定項目の定義表（**何を出すか**だけを持ち、ウィジェットは作らない）。

## なぜ表にするか
Phase 3 で項目が 45 件を超える。1件ずつメソッドを書くと、範囲・既定値・説明文が
コードのあちこちに散り、`.env` や `Config` を変えた時に**片方だけ腐る**。ここに
一覧として置けば、`settings_page.py` は「表を読んでウィジェットを並べる」だけで済む。

## 既定値はここに書かない
既定値は `eve/config.py`（と `ModelRegistry.ROLE_ENV`）から**実行時に引く**。
ここに数字を書き写すと二重管理になり、`config.py` を直した時に静かにズレる。

## 出さないもの（意図的な不在）
- **実測で校正した重み・閾値**: `RAG_W_*` / `RAG_MMR_LAMBDA` / `RAG_RECENCY_TAU` /
  `RAG_DUP_HARDCUT` / `RAG_REL_BASELINE` / `RAG_RELEVANCE_FLOOR` /
  `SPEECH_DUP_EMB_THRESHOLD` / `VLM_DEDUP_RATIO` / `VLM_BLANK_STD_THRESHOLD` /
  `SILENCE_TICK_SEC` ほか。ユーザ側に動かす判断材料が無く、ずらすと静かに劣化するだけ。
- **消費者ゼロの設定**: `VLM_MERGE_MODEL` / `YOUTUBE_MODEL` / `YOUTUBE_API_KEY` /
  `TARGET_CHANNEL_ID` / `VTS_PATH`。使うコードが1行も無いので、出すと
  「設定したのに効かない」という嘘になる。
- **記憶ファイル名**: `HISTORY_FILE` / `RAG_FILE` / `TASK_FILE`。変えると記憶が
  消えたように見える。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    key: str                       # `.env` のキー
    label: str
    kind: str                      # int | float | bool | choice | text
    note: str = ""                 # 説明（画面に小さく出す）
    lo: float = 0.0
    hi: float = 0.0
    step: float = 1.0
    on: str = "1"                  # bool のときの真の表記
    off: str = "0"                 # bool のときの偽の表記
    choices: tuple = ()
    editable: bool = False         # choice を自由入力にもするか
    warn: str = ""                 # 触ると危ないものへの赤い注意


@dataclass(frozen=True)
class Group:
    title: str
    note: str
    items: tuple = field(default_factory=tuple)


def _f(key, label, lo, hi, step, note="", warn=""):
    return Item(key, label, "float", note, lo, hi, step, warn=warn)


def _i(key, label, lo, hi, step=1, note="", warn=""):
    return Item(key, label, "int", note, lo, hi, step, warn=warn)


def _b(key, label, note="", on="1", off="0", warn=""):
    return Item(key, label, "bool", note, on=on, off=off, warn=warn)


def _c(key, label, choices, note="", editable=False, warn=""):
    return Item(key, label, "choice", note, choices=choices, editable=editable, warn=warn)


def _t(key, label, note="", warn=""):
    return Item(key, label, "text", note, warn=warn)


MODEL_CHOICES = (
    "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-5.4-mini",
    "openai/gpt-4o", "openai/gpt-4o-mini",
    "gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash",
    "groq/llama-3.3-70b-versatile",
)


def _model(key, label, note):
    return _c(key, label, MODEL_CHOICES, note, editable=True)


# --- 設定（常時表示）------------------------------------------------------

BASIC: tuple = (
    Group("声", "VOICEVOX の合成パラメータ。話者は VOICEVOX の話者IDです。", (
        _i("VV_SPEAKER", "話者ID", 0, 200),
        _f("VV_SPEED", "話す速さ", 0.50, 2.00, 0.05, "1.00 が標準。大きいほど速い"),
        _f("VV_PITCH", "声の高さ", -0.15, 0.15, 0.01,
           "0.00 が標準。VOICEVOX の上限が ±0.15"),
        _f("VV_INTONATION", "抑揚", 0.00, 2.00, 0.05,
           "1.00 が標準。0 に近いほど平坦、大きいほど起伏が強い"),
    )),
    Group("ウィンドウと画面表示", "", (
        _b("WS_VIEWER_MODE", "イブの表示物をタスクウィンドウ内に集める",
           "切ると、これまでどおり画面の好きな位置に別ウィンドウで開きます",
           on="window", off="external"),
        _b("UI_ALWAYS_ON_TOP", "このウィンドウを常に最前面にする"),
        _b("UI_HIDE_FROM_CAPTURE", "メインウィンドウを画面キャプチャから隠す",
           "既定は ON。切ると、イブの画面認識が自分のログと自分の顔を「ユーザの画面」として"
           "実況し始めます。配信で映したい時だけ切ってください"),
        _b("TASKWIN_HIDE_FROM_CAPTURE", "タスクウィンドウ（表示物）を画面キャプチャから隠す",
           "既定は ON。切ると、メインウィンドウは隠したまま**イブが自分の出した表示物だけ**を"
           "見られるようになります（作らせた電卓やグラフを一緒に見ながら話したい時に）"),
        _b("TASKWIN_HTML_ENABLED", "HTML を動く状態で表示する",
           "既定は ON。イブが作った電卓やカウンタをボタンごと動かせます。"
           "切るとソースがそのまま出ます（PySide6-Addons が必要）"),
        _i("TASKWIN_PROTECT_CLICKS", "操作中とみなすクリック数", 0, 20,
           note="この回数クリックした表示物は、あなたが閉じるまでイブが消しません"
                "（0 で保護なし）"),
    )),
    Group("モデル", "`provider/model` の形式。一覧に無いものは直接入力できます。", (
        _model("RESPONSE_MODEL", "応答", "あなたと話す本体。ここが速さと賢さを決める"),
        _model("DECIDE_MODEL", "発話判定", "自分から話しかけるかを決める。安いモデルでよい"),
        _model("TASK_MODEL", "タスク", "調べ物や作業を進める。賢いモデルほど完遂しやすい"),
        _model("VLM_LEAF_MODEL", "画面認識", "画面を読む。画像を扱えるモデルにすること"),
    )),
)

# --- 高級設定（アコーディオン）--------------------------------------------

ADVANCED: tuple = (
    Group("機能の起動時 ON/OFF",
          "**ここは「次に起動した時の初期値」です。**動かしている間の切り替えは"
          "メニューのボタンで行います（そちらはこの起動のあいだだけ有効）。"
          "そのため、メニューで切ってもこの画面の表示は変わりません。", (
              _b("CALLFUNCTION_ENABLED", "機能呼び出し（これが土台）",
                 "切ると、下のタスク・検索・作業フォルダも全部使えなくなります"),
              _b("TASK_ENABLED", "タスク（依頼を引き受けて後で報告する）",
                 "機能呼び出しが ON であることが前提"),
              _b("SEARCH_ENABLED", "Web 検索", "タスクが ON であることが前提（タスク専用の能力）"),
              _b("WS_ENABLED", "作業フォルダ（ファイルの読み書き）",
                 "タスクが ON であることが前提"),
              _b("WS_EXEC_ENABLED", "プログラム実行",
                 "作業フォルダが ON であることが前提。別途サンドボックスの設定が要ります",
                 warn="イブがこの PC でコマンドを実行できるようになります"),
              _b("WS_VIEWER_ENABLED", "画面への表示（画像やメモを出す）",
                 "作業フォルダが ON であることが前提"),
              _b("VLM_ENABLED", "画面認識",
                 "ON にすると**画面が Gemini に送られます**"),
              _b("VLM_OCR_ENABLED", "文字の変化も見る（OCR）",
                 "pHash では枠の中の文字変化を拾えないため。読み取った文字は LLM に渡さず、"
                 "「変わったかどうか」だけを使います"),
          )),
    Group("モデル（残り）",
          "会話の見た目には出ないところ。安いモデルで十分なことが多いです。", (
              _model("FEEDBACK_MODEL", "内省", "各応答のあと、自分の予測とのズレを振り返る"),
              _model("SUMMARIZE_MODEL", "要約", "長い記憶をまとめる"),
              _model("SEARCH_SUMMARIZE_MODEL", "検索結果の要約",
                     "**ツールを一切渡さない専用役**。信頼できない Web 本文を読むのはここだけ"),
              _model("SELF_MODEL_MODEL", "自己形成",
                     "イブの好み・趣向の台帳を会話から育てる。**合図が出た時だけ**呼ばれる"
                     "（実測でイブ発話の約15%）＝稀なので賢いモデルを選ぶ"),
              _model("DELIVERY_CHECK_MODEL", "報告の配達確認",
                     "報告が実際に伝わったかを見る。**決定論で決められない時だけ**呼ばれる"
                     "＝呼び出しは稀なので、安いモデルより賢いモデルを選ぶ"),
          )),
    Group("音声入力",
          "詰めると聞き逃し、緩めると物音で反応します。変えたら実際に話して確かめてください。", (
              _f("VAD_THRESHOLD", "発話とみなす確からしさ", 0.10, 0.90, 0.05,
                 "小さいほど敏感（物音を拾いやすい）"),
              _f("SILENCE_LIMIT", "発話の終わりとみなす無音[秒]", 0.20, 3.00, 0.10,
                 "短いと文の途中で切れ、長いと応答が遅くなる"),
              _f("PREROLL_SEC", "先頭の継ぎ足し[秒]", 0.00, 2.00, 0.10,
                 "検知は喋り始めより遅れるので、その分だけ手前を足す（頭欠け防止）"),
              _f("VAD_ONSET_SEC", "発話開始の確認窓[秒]", 0.00, 1.00, 0.02,
                 "これだけ続いて初めて「話し始めた」とする。打鍵の単発音で割り込ませないため"),
              _c("STT_BACKEND", "聞き取りの提供元", ("openai", "groq"),
                 "groq は幻聴が多く非推奨（切り戻し用）"),
              _t("STT_MODEL", "聞き取りモデル", "既定は gpt-4o-transcribe"),
          )),
    Group("画面認識のコスト",
          "**ここは直接 API の請求に効きます。**鈍い側に倒すのが既定です。", (
              _f("VLM_TARGET_FPS", "画面を見る頻度[fps]", 0.20, 10.00, 0.20,
                 "撮る頻度。実際に LLM を呼ぶのは「変化した」と判定した時だけ"),
              _f("VLM_MIN_INTERVAL_SEC", "呼び出しの最短間隔[秒]", 0.20, 30.00, 0.50,
                 "変化が続いても、これより短い間隔では呼ばない"),
              _i("VLM_MAX_FRAMES_PER_CALL", "1回に送る枚数", 1, 8, 1,
                 "多いほど動きが分かるが、そのぶん高い"),
              _i("VLM_DOWNSCALE_MAX", "送る画像の長辺[px]", 256, 2048, 128),
              _i("VLM_JPEG_QUALITY", "JPEG 品質", 30, 95, 5),
              _i("VLM_PHASH_THRESHOLD", "変化とみなす差（見た目）", 1, 32, 1,
                 "小さいほど敏感。0-64 の hamming 距離"),
              _f("VLM_OCR_DIFF_PCT", "変化とみなす差（文字）[%]", 5.00, 80.00, 5.00,
                 "実測で 25% に決定。下げると文章作成中に鳴り続けます"),
              _f("VLM_OCR_INTERVAL_SEC", "OCR の最短間隔[秒]", 0.50, 30.00, 0.50),
              _i("VLM_MONITOR", "見るモニタ番号", 0, 8, 1, "0=全画面 / 1=主モニタ"),
          )),
    Group("記憶",
          "増やすほど応答が遅くなり、埋め込みの費用も伸びます。", (
              _i("RECENT_TURN_COUNT", "応答に入れる直近の会話数", 2, 30, 1),
              _i("RAG_TOP_K", "思い出す記憶の件数", 0, 10, 1,
                 "今の会話に関連する記憶をいくつ入れるか"),
              _i("RAG_RANDOM_K", "話題の種の件数", 0, 10, 1,
                 "自分から話しかける時に使う「新しい切り口」の数"),
              _i("RAG_MAX_CHUNKS", "記憶の保持上限", 50, 5000, 50,
                 "v1 は 3000 で過去に逸れた。500 が現在の値"),
              _i("HISTORY_MAX_TURNS", "会話ログの保持数", 10, 1000, 10),
              _c("EMBED_BACKEND", "埋め込みの提供元", ("ruri", "openai"),
                 "ruri はローカル（無料・起動が数秒遅い）/ openai は API（有料・速い）"),
          )),
    Group("タスクと検索", "", (
        _i("TASK_AGENT_MAX_STEPS", "1つのタスクの手数上限", 1, 40, 1,
           "これを超えたら「終わらなかった理由」をまとめて報告します"),
        _f("TASK_AGENT_TIMEOUT_SEC", "1つのタスクの制限時間[秒]", 30.00, 3600.00, 30.00,
           "長くすると、タスクを OFF にする時の待ちも長くなります"),
        _i("TASK_MAX", "タスク履歴の保持数", 10, 5000, 10),
        _i("WS_VIEWER_MAX_WINDOWS", "同時に開ける表示ウィンドウ数", 1, 10, 1),
        _f("WS_QUOTA_GB", "作業フォルダの上限[GB]", 1.00, 1024.00, 1.00),
        _f("WS_RUN_TIMEOUT_SEC", "プログラム実行の制限時間[秒]", 5.00, 600.00, 5.00),
        _i("WS_RUN_MEM_MB", "プログラム実行のメモリ上限[MB]", 128, 16384, 128),
        _t("WS_EXEC_ALLOW", "実行を許すコマンド（カンマ区切り）",
           "**セキュリティ境界ではありません。**python を許した時点で任意のコードが動きます。"
           "本当の境界は専用アカウントの ACL と Job Object です",
           warn="増やす前に、そのコマンドで何ができるかを確認してください"),
        _i("SEARCH_MAX_RESULTS", "検索結果の件数", 1, 20, 1),
        _f("SEARCH_TIMEOUT_SEC", "検索の制限時間[秒]", 5.00, 120.00, 5.00),
        _f("SEARCH_COOLDOWN_SEC", "同じ検索を控える時間[秒]", 0.00, 600.00, 10.00),
    )),
    Group("接続先", "**間違えると起動しません。**", (
        _t("VOICEVOX_URL", "VOICEVOX の場所", "既定は http://127.0.0.1:50021"),
        _t("WS_ROOT", "作業フォルダの場所",
           "空なら自動で決めます。システムフォルダなど不適切な場所は起動時に弾かれます"),
    )),
)


# --- 依存関係の検算 --------------------------------------------------------

# (子, 親, 説明)。親が OFF だと子は**起動時に無効化される**（`voice_loop.py` の構造）。
DEPENDENCIES = (
    ("TASK_ENABLED", "CALLFUNCTION_ENABLED", "タスクは機能呼び出しの上に載っています"),
    ("SEARCH_ENABLED", "TASK_ENABLED", "Web 検索はタスク専用の能力です"),
    ("WS_ENABLED", "TASK_ENABLED", "作業フォルダはタスク専用の能力です"),
    ("WS_EXEC_ENABLED", "WS_ENABLED", "プログラム実行は作業フォルダの中で動きます"),
    ("WS_VIEWER_ENABLED", "WS_ENABLED", "画面への表示は作業フォルダの中のファイルを出します"),
    ("VLM_OCR_ENABLED", "VLM_ENABLED", "文字ゲートは画面認識の一部です"),
)


def check_consistency(values: dict) -> list[str]:
    """「ON にしたのに前提が OFF」を見つけて、人の言葉で返す。

    `voice_loop.py` は起動時に警告ログを出すが、**UI からは誰も見ない**。保存する前に
    画面で言わないと、「設定したのに効かない」で詰まる（起動時 fail-soft なので
    エラーにもならない）。
    """
    out: list[str] = []
    for child, parent, why in DEPENDENCIES:
        if values.get(child) == "1" and values.get(parent) == "0":
            out.append(f"「{_label_of(child)}」は ON ですが、"
                       f"前提の「{_label_of(parent)}」が OFF です（{why}）。"
                       "このままだと起動時に無効化されます。")
    return out


def _label_of(key: str) -> str:
    for group in (*BASIC, *ADVANCED):
        for item in group.items:
            if item.key == key:
                return item.label
    return key


def all_items() -> tuple:
    return tuple(item for group in (*BASIC, *ADVANCED) for item in group.items)
