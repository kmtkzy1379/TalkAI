"""`.env` 読み込みと Config。起動時 `validate()` で不足設定を列挙（UI Start ゲート用）。

python-dotenv が未インストールでも最小パースで `.env` を読む（F0 を依存なしで動かす）。
"""
from __future__ import annotations

import os
from pathlib import Path

# 読み込む `.env`。`EVE_ENV_FILE` で差し替えられる。
# **検証のため**に入れている: 設定が本当に効くかを見るには別の `.env` で起動する必要があり、
# 本物を書き換えて戻す方式だと、途中で落ちた時にユーザの `.env` が壊れたまま残る。
# 副次的に「用途ごとのプロファイル」にも使える。
_ENV_PATH = Path(os.getenv("EVE_ENV_FILE") or (Path(__file__).resolve().parent.parent / ".env"))


def _load_dotenv(path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path)
        return
    except Exception:
        pass  # 未インストール時は下の最小パースにフォールバック
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(_ENV_PATH)


class Config:
    # API キー（Claude=ANTHROPIC は現在停止中 → GPT/Gemini で代用）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
    # OpenRouter（Jev＝typesafe/jev-1.13 の decisions API・S8 フィラーの1段目を場面に寄せる）
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID", "")

    # 外部アプリ
    VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
    VTS_PATH = os.getenv("VTS_PATH", "")
    VOICEVOX_PATH = os.getenv("VOICEVOX_PATH", "")

    # 応答の TTS。既定 = irodori（ユーザ裁定 2026-09-25「応答もフィラーも Irodori」
    # 「感情乗ってないと正確に判別しにくい」）。voicevox は切り替えで残す。
    # **自動フォールバックはしない**（「急に切り替わると不自然だから話者はこっちで選択」）。
    TTS_BACKEND = os.getenv("TTS_BACKEND", "irodori")  # irodori | voicevox
    # Irodori サーバ（`tools/irodori_server.py`）。EVE が全自動で起動する（Irodori の venv の python で）。
    IRODORI_REPO = os.getenv(
        "IRODORI_REPO", str(Path(__file__).resolve().parent.parent.parent / "irodori-bench" / "Irodori-TTS"))
    IRODORI_PYTHON = os.getenv(
        "IRODORI_PYTHON", str(Path(IRODORI_REPO) / ".venv" / "Scripts" / "python.exe"))
    IRODORI_PORT = int(os.getenv("IRODORI_PORT", "50030"))
    # 話者の参照音声（リポジトリ基準の相対パス可）。**フィラーのバンク（fillers.yaml の voices）と同じ声にする**。
    # 話者は未定（当面 cand_04・ユーザ裁定 2026-09-25）。no_ref は seed ごとに話者が変わるので使わない。
    IRODORI_VOICE_REF = os.getenv("IRODORI_VOICE_REF", "voice_candidates/cand_04.wav")
    # 合成の steps と seed。**聴き比べて決める（未決）**。steps 6 は今のフィラーバンクと同じ値
    # （実測 短い語 約211ms / steps 12 で 約338ms）。seed を固定すると同じ文は同じ音になる。
    IRODORI_STEPS = int(os.getenv("IRODORI_STEPS", "6"))
    IRODORI_SEED = int(os.getenv("IRODORI_SEED", "0"))
    # イブの発話の区切り（「、」「。」＝再生の1項目）ごとの間[秒]。出力を開きっぱなしにした（2026-09-26）ので、
    # 以前は項目ごとの開き直しで空いていた間をここで作る。0.29 は試聴で良しとされた値（listen/clause/・
    # 開き直しの実測 約0.2秒＋出力の遅れ 約0.09秒）。直前の音が終わってからこれだけ空けて次を鳴らす
    SPEECH_GAP_SEC = float(os.getenv("SPEECH_GAP_SEC", "0.29"))

    # 応答非依存フィラー（S3・ユーザ指定 2026-09-25）。**機能フラグ**（既定 off・.env で明示する）。
    # ユーザの話し終わり（VAD の確定）に「あー」「えー」等を鳴らし、応答の音声ができたら畳む。
    # 語・抽選の比率・上限4秒・最短発話0.6秒・つなぎの間は fillers.yaml 側（ここには書かない）。
    # つなぎの間（fillers.yaml の timing.join_gap_sec）が未決の間は ON にしても鳴らない。
    FILLER_ENABLED = os.getenv("FILLER_ENABLED", "0") == "1"
    FILLER_TABLE = os.getenv("FILLER_TABLE", "fillers.yaml")  # リポジトリ基準の相対パス可
    # bank/<voice>/（tools/build_filler_bank.py で作る）。eve_calm＝語尾が上がらない候補（ユーザ採用 2026-09-26）
    FILLER_VOICE = os.getenv("FILLER_VOICE", "eve_calm")
    # フィラー前の待ち[秒]（ユーザ決定 2026-10-09「フィラーはA」）。VAD が話し終わりと判断した後（SILENCE_LIMIT）、さらにこれだけ
    # 待ってから1語目を鳴らす＝声が止まってから 0.5＋0.5＝約1.0秒。その間にユーザが話し出したら鳴らさない（話の途中の間で
    # フィラーを被せない）。
    # 応答の速さは変わらない。0 にすると以前どおり話し終わりの判断と同時に鳴らす（`eve/filler/player.py`）
    FILLER_START_DELAY_SEC = float(os.getenv("FILLER_START_DELAY_SEC", "0.5"))
    # 応答依存フィラー（S4）。**機能フラグ**（既定 off）。fillers.yaml の表 B 以下と注記を、ユーザ発話への
    # 応答の system に入れて、応答の書き出しで言わせる（イメージを掴む目的・ユーザ裁定 2026-09-25）。
    # 音を鳴らす S3（FILLER_ENABLED）とは独立に ON/OFF できる。
    FILLER_DEPENDENT_ENABLED = os.getenv("FILLER_DEPENDENT_ENABLED", "0") == "1"
    # 笑い（S6・ユーザ裁定 2026-09-26）。**機能フラグ**（既定 off）。ユーザの話の「、」「。」ごとに
    # gpt-6-luna（ModelRegistry の laugh_judge 役・LAUGH_MODEL）で「イブが思わず笑う場面か」を判定し、
    # 確率が fillers.yaml の laugh.immediate 以上ならその場で笑い、laugh.filler 以上ならフィラーの中で笑う（laugh.filler_style: tone なら笑いの声色）。
    # ストリーミング文字起こし（STT_BACKEND=openai_live）とフィラー（FILLER_ENABLED）が前提。
    LAUGH_ENABLED = os.getenv("LAUGH_ENABLED", "0") == "1"
    # 感情（S5・ユーザ裁定 2026-09-26）。**機能フラグ**（既定 off）。イブの今の感情（喜怒哀楽笑い）を、ユーザの話の
    # 「、」「。」ごと（gpt-6-luna・emotion_judge 役・EMOTION_MODEL）とフィードバックで更新し、応答LLM・
    # フィードバックLLM・Irodori の声（emotion.yaml の voice_emoji）に使う（`eve/emotion.py`）。
    # 「、」「。」ごとの判定はストリーミング文字起こし（STT_BACKEND=openai_live）が前提（無ければフィードバックだけ）。
    EMOTION_ENABLED = os.getenv("EMOTION_ENABLED", "0") == "1"
    EMOTION_TABLE = os.getenv("EMOTION_TABLE", "emotion.yaml")  # リポジトリ基準の相対パス可
    # 感情は5つの数値（各0〜100・合計100以下・0＝平常・ユーザ 2026-10-09）。判定した会話の時点から、
    # 1時間あたりこれだけ引く（起動していない間も同じ時計で減る＝何日も前の会話の感情は平常に戻る）。
    # ユーザ決定 2026-10-09「感情減衰率もいい感じ」＝10（強さ30は3時間・70は7時間で平常）
    EMOTION_DECAY_PER_HOUR = float(os.getenv("EMOTION_DECAY_PER_HOUR", "10"))
    # 声とフィラーに感情を乗せるのは、いちばん強い感情がこの値以上の時だけ（弱い時は普通の声）。
    # ユーザ決定 2026-10-09「最も強い感情値が30以上の時でいい」（実測で判定の約4割）
    EMOTION_TONE_MIN = float(os.getenv("EMOTION_TONE_MIN", "30"))
    # 今の感情の保存先（判定のたびに書き、起動時に読む・リポジトリ基準の相対パス可・.gitignore 済み）
    EMOTION_STATE_FILE = os.getenv("EMOTION_STATE_FILE", "emotion_state.json")
    # フィラーの1段目を場面に寄せる（S8・ユーザ指定 2026-09-27）。**機能フラグ**（既定 off）。ユーザの話の「、」「。」ごとに
    # Jev（OpenRouter・OPENROUTER_API_KEY）へ聞き手の反応の種類（通常・悩む・驚く・悲しむ・不満）を聞き、強い時（fillers.yaml の
    # mood.threshold 以上）だけ1段目をその種類の語にする。前提: FILLER_ENABLED・STT_BACKEND=openai_live
    FILLER_MOOD_ENABLED = os.getenv("FILLER_MOOD_ENABLED", "0") == "1"

    # VOICEVOX 合成パラメータ
    VV_SPEAKER = int(os.getenv("VV_SPEAKER", "1"))
    VV_SPEED = float(os.getenv("VV_SPEED", "1.0"))
    VV_PITCH = float(os.getenv("VV_PITCH", "0.0"))
    # 抑揚（VOICEVOX の intonationScale・0=平坦 / 1=標準 / 大きいほど起伏が強い）。
    # 1.0 が VOICEVOX の既定なので、明示しない限り**今までと同じ音**になる。
    VV_INTONATION = float(os.getenv("VV_INTONATION", "1.0"))

    # 音声入力 / VAD（値は v1 実績設定を踏襲）
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    CHANNELS = int(os.getenv("CHANNELS", "1"))
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
    SILENCE_LIMIT = float(os.getenv("SILENCE_LIMIT", "0.5"))
    # 発話開始検知の直前を継ぎ足す pre-roll（頭欠け防止。検知は喋り始めより遅れるため）
    PREROLL_SEC = float(os.getenv("PREROLL_SEC", "0.4"))
    # 発話開始の確認窓: これだけ連続して発話が続いて初めて「発話開始」とする
    # （クリック/打鍵の単発スパイクで誤って割り込み発火しないように）
    VAD_ONSET_SEC = float(os.getenv("VAD_ONSET_SEC", "0.12"))
    # 音量ゲート（0=無効）。**Silero は音量に依存せず「音声らしさ」だけを見る**ので、
    # イヤホンから漏れた自分の声のような極小音でも高い確率を返す（実測 2026-09-21:
    # ピーク 0.026 の漏れに対し VAD 0.742＝閾値 0.5 を突破し、イブが自分の発話で
    # barge-in して自己ループした）。`VAD_THRESHOLD` を上げても 0.742 は通るため、
    # 振幅という別の軸で切る必要がある。
    # 判定は**発話のまとまりの最大振幅**で行う（チャンク単位だと語尾・語間の減衰部分で
    # 切れてしまい「喋っても反応しない」になる）。実発話は必ずどこかで大きくなる。
    # 実測の分離: ユーザの発話 最大 0.197 / 回り込み 最大 0.026 ＝ 7.5倍。
    # 既定 0.045 は漏れの 1.7倍・実発話の 1/4.4 で、取りこぼし側に安全側。
    # 小声が拾われないと感じたら下げる。自己ループが残るならイヤホン音量を下げる方が先
    # （マイクゲインを上げても声と漏れが同率で増えるだけで比は改善しない）。
    MIC_MIN_PEAK = float(os.getenv("MIC_MIN_PEAK", "0.045"))
    # 最短発話ゲート[秒]（0=長さでは捨てない）。判定は**声の長さ**（発話開始の確認窓の頭〜最後に発話と判定した
    # チャンクの終わり）。以前は pre-roll と無音の尾（SILENCE_LIMIT 分）込みの長さを 0.3秒と比べていたので
    # 一度も効いていなかった（2026-06-18 の初版から。2026-09-29 にユーザ指示「ノイズカットは修正」で直した）。
    # 既定 0 はその「効いていなかった」挙動のまま。実測（2026-09-29・VOICEVOX の合成音6話者×速さ2通り・本番と
    # 同じ VAD と判定）: 「はい」「え？」「ね」「へー」の声の長さは最短 0.096秒（確認窓ちょうど）まで短くなり、
    # 長さだけでは短い返事と物音を分けられない。自分の声と物音を `tools/short_utterance_probe.py` で測ってから決める。
    MIN_SPEECH_SEC = float(os.getenv("MIN_SPEECH_SEC", "0"))

    # STT バックエンド。既定 = openai_live（gpt-live-transcribe のストリーミング・ユーザ裁定
    # 2026-09-25「応答も早くなるし金銭的にもあまり高くない」）。openai = gpt-4o-transcribe の一括
    # （話し終わってから全体を送る）。groq は幻聴多く非推奨=ロールバック用。
    # **openai_live が失敗しても一括方式へ自動で切り替えない**（ユーザ裁定: その発話は捨てて張り直す）。
    STT_BACKEND = os.getenv("STT_BACKEND", "openai_live")  # openai_live | openai | groq
    STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe")  # openai（一括）用
    STT_LIVE_MODEL = os.getenv("STT_LIVE_MODEL", "gpt-live-transcribe")
    # delay: 実測（JVNV 18件・2026-09-25）で low が minimal より誤り率・確定の速さとも良かった
    # （low 6.2%・commit→確定 中央0.55s / minimal 6.5%・0.62s）。
    STT_LIVE_DELAY = os.getenv("STT_LIVE_DELAY", "low")
    # commit 後に確定が来ない時の打ち切り[秒]。実測の commit→確定は 中央0.55s・最大0.94s
    # （JVNV 18件を実時間で送った場合）。その約5倍。超えたらその発話は捨てて接続を張り直す。
    STT_LIVE_COMMIT_TIMEOUT_SEC = float(os.getenv("STT_LIVE_COMMIT_TIMEOUT_SEC", "5.0"))

    # 短期記憶（F3 ConversationCache）
    # 会話ログの永続先（タイムスタンプ付き JSONL・ロケット鉛筆方式）。
    HISTORY_FILE = os.getenv("HISTORY_FILE", "conversation_history.jsonl")
    # メモリに保持する最大ターン数（古いものから押し出す＝ロケット鉛筆）。
    HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "100"))
    # 応答LLM に注入する直近ターン数（≈3往復）。レイテンシと過去逸れのため小さく保つ。
    RECENT_TURN_COUNT = int(os.getenv("RECENT_TURN_COUNT", "6"))

    # 長期記憶（F3.5 RAG / 連想想起）
    RAG_FILE = os.getenv("RAG_FILE", "rag_memory.jsonl")  # 永続 JSONL（埋め込み込み）
    # 自己モデル（イブの好み・趣向の台帳）。会話から自己形成サイドカーが育てる。
    SELF_MODEL_ENABLED = os.getenv("SELF_MODEL_ENABLED", "0") == "1"
    SELF_MODEL_FILE = os.getenv("SELF_MODEL_FILE", "self_model.jsonl")
    RAG_MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "500"))  # ロケット鉛筆の上限
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))  # 注入する記憶件数（top-1必須+MMR）
    # 埋め込み backend（ruri=ローカル日本語特化 / openai=API）。STT と同じ差替え方式。
    EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ruri")  # ruri | openai
    RURI_MODEL = os.getenv("RURI_MODEL", "cl-nagoya/ruri-v3-310m")  # 768次元・日本語最良
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # openai backend 用
    # memory-stream ランキング（Generative Agents 式）。
    # 重み: relevance 最優先。recency は弱く（短期記憶が直近を既にカバー＝高recencyは重複想起の元）。
    # importance も控えめ（高importance記憶が話題無関係でも紛れ込むのを防ぐ）。実測 sweep で調整。
    # sweep 実測(2026-06-20, Ruri)で確定: importance を効かせすぎると高importance記憶が
    # 話題無関係でも紛れ込む → relevance 優勢・importance 控えめが最も自然。
    RAG_W_REL = float(os.getenv("RAG_W_REL", "0.7"))  # relevance 重み（話題の近さ・最優先）
    RAG_W_IMP = float(os.getenv("RAG_W_IMP", "0.18"))  # importance 重み（重要度・控えめ）
    RAG_W_REC = float(os.getenv("RAG_W_REC", "0.12"))  # recency 重み（新しさ・隠し味程度）
    RAG_RECENCY_TAU = float(os.getenv("RAG_RECENCY_TAU", "86400"))  # 減衰時定数[秒]（既定1日）
    RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))  # MMR: 関連0.7/多様0.3
    RAG_DUP_HARDCUT = float(os.getenv("RAG_DUP_HARDCUT", "0.95"))  # これ超の重複は除外
    # 埋め込み異方性の補正基準。cos をこの値基準で再スケール（rel'=(cos-base)/(1-base)）。
    # Ruri/e5 系は無関係でも cos~0.75 に密集するため ~0.77。生コサイン運用なら 0.0。
    RAG_REL_BASELINE = float(os.getenv("RAG_REL_BASELINE", "0.77"))
    # 関連度フロア（再スケール後の rel' に対して）: これ未満は無関係として除外。
    # 高めに取り「無関係が混じる破綻」を防ぐ（クリーン優先＝たまに1-2件に減る方を許容・ユーザ裁定）。
    RAG_RELEVANCE_FLOOR = float(os.getenv("RAG_RELEVANCE_FLOOR", "0.15"))

    # 発話判定 / 自発発話（F5）
    # 沈黙監視 tick（軽い周期チェック）。
    SILENCE_TICK_SEC = float(os.getenv("SILENCE_TICK_SEC", "0.7"))
    # ユーザ 5秒無言で発話判定LLM を回す（企画書準拠）。以後は判定のたびに間隔を倍にする（5→10→20→…秒）。
    # 2026-10-05 ユーザ裁定「両方無言なら間隔を指数関数的に広げる・最大5分」（それまではフラット5秒で
    # 放置中も1時間に数百回判定していた）。ユーザかイブが話すと 5秒に戻る。画面の変化による判定はこの間隔を通らない。
    SILENCE_THRESHOLD_SEC = float(os.getenv("SILENCE_THRESHOLD_SEC", "5.0"))
    SILENCE_MAX_INTERVAL_SEC = float(os.getenv("SILENCE_MAX_INTERVAL_SEC", "300"))
    # 自律発話を**起動時に**有効にするか（既定 on＝これまでの挙動）。UI メニューの ON/OFF は
    # 「今このセッションだけ」で、こちらは「次の起動時の初期値」。両者は別物（UI_DESIGN 裁定 A）。
    AUTOSPEECH_ENABLED = os.getenv("AUTOSPEECH_ENABLED", "1") == "1"
    # surprise は数値で発話を絶対決定しない（指標）。発話判定LLM が surprise+感情+内容を総合判断する
    # （ユーザ裁定）。よって SURPRISE_SPEAK_FORCE/FLOOR の数値ゲートは廃止。
    # 自律発話の「関連記憶」枠から、これより新しいチャンクを除外する[秒]（記憶の自己参照を断つ）。
    # FeedbackLLM は応答ごとに会話の要約チャンクを書くため、直近ユーザ発話をクエリにすると
    # 「数十秒前に自分が書いた今の会話」が relevance 最大で必ず返る（実測 2026-07-26: 関連枠の
    # 68%が5分以内・最短1.8秒）。0 で無効化（ロールバック用）。応答経路の検索には影響しない。
    RAG_AUTO_EXCLUDE_SEC = float(os.getenv("RAG_AUTO_EXCLUDE_SEC", "600"))
    # 沈黙時の「話題の種」取得件数（自律発話用）。沈黙時はコンテキストに余裕があるので 3
    # （関連1 + 完全ランダム1 + 重要度1 の構成＝毎回同じ記憶ばかりにせず新しい切り口を混ぜる）。
    RAG_RANDOM_K = int(os.getenv("RAG_RANDOM_K", "3"))
    # 話題の種のバラけ具合（0=重要度順のみ / 大きいほどランダム性増＝新しい切り口を出す）。
    RAG_TOPIC_JITTER = float(os.getenv("RAG_TOPIC_JITTER", "0.4"))
    # 発話判定ログ（True/False とも記録・ロケット鉛筆・処理には関与しない＝観測専用）。
    SPEECH_LOG_MAX = int(os.getenv("SPEECH_LOG_MAX", "10"))
    # 自発発話の同内容抑制で「直近」とみなす時間窓[秒]（この窓の自発発話とだけ比較する）。
    # 観測用 speech_log（件数上限・黙る判定も混在）を比較元にしていた時は実効約50-60秒しかなく、
    # 90秒空くと前回発話が押し出されて同じ提案が通っていた（実測 2026-07-23: 8回中4回すり抜け）。
    SPEECH_DUP_WINDOW_SEC = float(os.getenv("SPEECH_DUP_WINDOW_SEC", "600.0"))
    SPEECH_DUP_LOG_MAX = int(os.getenv("SPEECH_DUP_LOG_MAX", "50"))  # 窓内件数の安全上限（メモリ bound）
    # 二段目ゲート: 埋め込み cos がこれ以上なら「言い換えただけの同内容」として抑制。
    # 実測校正(2026-07-26・実発話8文/Ruri v3): 同話題 0.909-0.932 / 別話題 0.780-0.834 → 0.87 で分離。
    # 文字bigram だけでは語彙を変えた再提案（実測 0.23 < 0.25）を取りこぼすため二段構えにする。
    SPEECH_DUP_EMB_THRESHOLD = float(os.getenv("SPEECH_DUP_EMB_THRESHOLD", "0.87"))

    # 画面認識（F6・snapshot モード）。**すべてインフラ/コスト knob（Eve の挙動を数値で決めない）**。
    VLM_ENABLED = os.getenv("VLM_ENABLED", "0") == "1"  # 既定 off（実機で明示 on）
    VLM_MONITOR = int(os.getenv("VLM_MONITOR", "1"))  # mss モニタ番号（1=主）
    VLM_TARGET_FPS = float(os.getenv("VLM_TARGET_FPS", "2.0"))  # キャプチャ頻度
    VLM_RING_MAX = int(os.getenv("VLM_RING_MAX", "6"))  # リング上限（=レイテンシ bound・drop-oldest）
    VLM_MAX_FRAMES_PER_CALL = int(os.getenv("VLM_MAX_FRAMES_PER_CALL", "3"))  # 1呼び出しの枚数（窓≈枚/fps≈1.5s）
    VLM_PHASH_THRESHOLD = int(os.getenv("VLM_PHASH_THRESHOLD", "12"))  # 変化ゲート hamming 閾値
    VLM_PERIODIC_FRAMES = int(os.getenv("VLM_PERIODIC_FRAMES", "0"))  # 静止 N フレームで強制再評価（0=無効）
    VLM_MIN_INTERVAL_SEC = float(os.getenv("VLM_MIN_INTERVAL_SEC", "1.0"))  # コスト floor（自己再トリガを律速・A9）
    VLM_DEDUP_RATIO = float(os.getenv("VLM_DEDUP_RATIO", "0.85"))  # 同一実況とみなす類似度（再トリガ抑制）
    # 画質の経緯（**1024/Q70/3枚 が現行の正**・2026-07-29 裁定）:
    # ① 初期は 1280/Q70/4枚。② 軽量化 1024/**Q60**/3枚 を試したが遅延に効かず(遅延は Gemini API 変動)
    #    OCR が悪化したので 1280/70/4 へ戻した。③ その後 .env が 1024/**Q70**/3枚 で運用され、
    #    2026-07-29 の通しE2E（実起動と同一設定）で OCR は正確だった（「買い物リスト」「起動方法と
    #    テストすればいい？」等の日本語・タスクバーの時刻まで読めた）。**悪化の主因は解像度でなく
    #    JPEG 品質(Q60)だった**と整理し、実測で問題の出なかった ③ を既定に採用。
    VLM_DOWNSCALE_MAX = int(os.getenv("VLM_DOWNSCALE_MAX", "1024"))  # 長辺上限（トークン/レイテンシ削減）
    VLM_JPEG_QUALITY = int(os.getenv("VLM_JPEG_QUALITY", "70"))  # 送信 JPEG 品質（60 に落とすと OCR 悪化）
    # 画面情報の鮮度上限。これより古い latest_vision は自発発話系に渡さない（「明らか過去を参照しない」を構造保証）。
    VLM_VISION_TTL_SEC = float(os.getenv("VLM_VISION_TTL_SEC", "7.0"))
    # ユーザ発話への応答に限り、画面が**変化していない間**は古い実況を据え置ける上限[秒]。
    # 「今何が見える?」に静止画面でも答えるため（注入時は「N秒前・以降変化なし」と正直に付す）。
    # 自発発話・発話判定はこの据え置きを使わない（静止中に画面へ引っ張られる固執を断つ層分離）。
    # 実測(2026-07-27): 無実況区間の分布は二峰性（≤52秒 か 数分以上の放置）で、180 はちょうど谷＝
    # どちらにも効いていなかった。328秒前の実況がまだ正しかったのに 180 で捨てて捏造した事故が
    # 出たため 600 へ。上限を残すのは、pHash 閾値未満の局所変化が積もる可能性に対する保険。
    # 実測(2026-08-06 J-3通し3ラン・E6): **同じ事故が 600 でも再発**（3/3ラン）。ユーザに
    # 「このまま画面で作業するね、見てて」と言われた時、直近実況が約1430秒前・内容も
    # 「画面に変化はなく」だったのに 600 で捨てられ、「今は画面が見えていない」と能力ごと否認した。
    # 据え置きの他の関門（実況であること・capture 生存・実況後に変化フレームが無いこと）は
    # すべて通っているので、捨てているのはこの上限だけ。→ 1800 へ。
    # 上限を残す理由は不変（pHash 閾値未満の微小変化の蓄積）。据え置き時は
    # `vision_for_user` が「N秒前の画面。以降ずっと変化なし」を自動付与するので誤解は生じない。
    VLM_STATIC_VISION_MAX_SEC = float(os.getenv("VLM_STATIC_VISION_MAX_SEC", "1800.0"))
    # 「画面が変わったが実況がまだ追いついていない」区間で、直前の実況を**断り付きで**
    # 据え置ける上限[秒]。これが無いと、その区間は画面情報ごと落ちて `# 画面` が不在明示に
    # なり、応答LLM は「今は画面が見えていない」と**能力ごと否認**する（実測 2026-09-21:
    # 空白時 9/9＝100% が否認。ユーザからは「画面が変わった瞬間しか見えていない」に見える）。
    # 実測（同日・VLM 稼働下で画面を強制変化させて計測）: この空白区間の長さは
    # 激しい操作(3秒ごと変化) 中央3.9s/p95 9.8s、普通の操作(12秒ごと) 中央6.5s/p95 8.3s、
    # 全20件で最大 9.8s。**10秒あれば観測した空白の100%を埋められる**。
    # 20 はその約2倍で、API が遅い時も否認へ落ちない余裕。上限を伸ばしても通常時の鮮度は
    # 落ちない（新しい実況が来た瞬間に窓は閉じる）＝効くのは VLM が詰まっている時だけ。
    # 据え置き時は「その後に画面が変わったので、今まさにこの通りとは限らない」と正直に付す
    # （実測: 否認 0/9・断り付き 9/9 で、古い内容を現在と断定しない）。
    VLM_STALE_VISION_MAX_SEC = float(os.getenv("VLM_STALE_VISION_MAX_SEC", "20.0"))
    VLM_BLANK_STD_THRESHOLD = float(os.getenv("VLM_BLANK_STD_THRESHOLD", "6.0"))  # 黒/空白検出（std 未満=取得不可・A11）
    # 文字ゲート（pHash が拾えない「枠内の文字だけの変化」で VLM を起こす二段目）。
    # 実測 2026-08-07: pHash 一致 15/15 ペアで画素完全一致 0/15・OCR 差分は最大24%＝pHash は
    # 文字変化を見ていない。OCR の読み取りは **LLM に一切渡さない**（誤読がハルシネの元）。
    VLM_OCR_ENABLED = os.getenv("VLM_OCR_ENABLED", "0") == "1"  # 既定 off（.env で明示 on）
    # 何%変われば起こすか。**25%**（裁定 2026-08-12）。実測で決めた:
    # 作文中(5.1分)の発火は 10%→3.9回/分 / 20%→1.4 / 25%→0.8 / 30%→0。
    # 通常利用（画面が動く一般的なE2E 82分）の検知は 10%→0.51回/分 / 25%→0.34 とほぼ維持される。
    # 30% は作文中を完全に黙らせるが「12行中6行の入れ替え(20.1%)」も見逃すので採らない。
    # クールダウンは採らない: 大きな変化が立て続けに起きた時に見逃すため（ユーザ裁定）。
    VLM_OCR_DIFF_PCT = float(os.getenv("VLM_OCR_DIFF_PCT", "25.0"))
    VLM_OCR_INTERVAL_SEC = float(os.getenv("VLM_OCR_INTERVAL_SEC", "3.0"))  # OCR の最短間隔
    VLM_OCR_DOWNSCALE = int(os.getenv("VLM_OCR_DOWNSCALE", "1600"))  # OCR 用キャプチャの長辺

    # UI（Live2D ウィンドウ）。**この項目だけ既定 ON**（`docs/UI_DESIGN.md` §4.6）:
    # 機能フラグ（能力）ではなく**ガード**で、OFF は「イブが自分のログと自分の顔を
    # ユーザの画面として実況する」既知の不具合状態（D4 / E11 自己観察ループ）だから。
    # 有効な間は OBS 等の画面キャプチャからも消えるので、配信時は OFF にする
    # （裁定 2026-08-13: 配信中は画面認識を使わない想定なので実運用で衝突しない）。
    UI_HIDE_FROM_CAPTURE = os.getenv("UI_HIDE_FROM_CAPTURE", "1") not in ("0", "false", "no")
    # **タスクウィンドウ（と別プロセスのビューア）は別扱い**（2026-08-15）。
    # 以前は上の1つで両方を隠していたため、「自分の UI は見せたくないが、**自分が出した
    # 表示物は見せたい**」（例: 作らせた電卓を一緒に見ながら話す）ができなかった。
    # 既定は ON＝従来と同じ挙動。OFF にしても隠れるのはメインウィンドウのままで、
    # イブが見えるようになるのは**自分が出したもの**だけ（自己観察ループの本体は残る）。
    TASKWIN_HIDE_FROM_CAPTURE = os.getenv(
        "TASKWIN_HIDE_FROM_CAPTURE", "1") not in ("0", "false", "no")
    # ウィンドウを常に最前面に出す（既定 off）。作業しながら顔を見たい時に使う。
    UI_ALWAYS_ON_TOP = os.getenv("UI_ALWAYS_ON_TOP", "0") == "1"

    # API 使用量と概算費用の記録（既定 off）。役割ごとに1呼び出し1行の JSONL を残す。
    # 価格は litellm の内蔵表から引く（引けないモデルは費用 None＝0 と嘘をつかない）。
    USAGE_LOG_ENABLED = os.getenv("USAGE_LOG_ENABLED", "0") == "1"
    USAGE_LOG_FILE = os.getenv("USAGE_LOG_FILE", "usage.jsonl")
    # litellm の価格表が古い/未収載の時の手動修正。"model=入力:出力"（USD/token）をカンマ区切り。
    USAGE_PRICE_OVERRIDE = os.getenv("USAGE_PRICE_OVERRIDE", "")

    # Call-Function（J-0 即時 Call-Function・increment 1 は read-only 能力のみ）。
    CALLFUNCTION_ENABLED = os.getenv("CALLFUNCTION_ENABLED", "0") == "1"  # 既定 off（実機で明示 on）
    # 応答LLM（RESPONSE_MODEL）に渡す reasoning_effort。空＝渡さない（モデルの既定）。
    # gpt-6-luna（ユーザ採用 2026-09-29）は chat completions で**ツール付きだと none しか通らない**
    # （既定・low は BadRequest「Function tools with reasoning_effort are not supported for gpt-6-luna」・実測 2026-09-28）。
    # ツールはユーザ発話のターンで必ず渡るので、ルナなら none が必須
    RESPONSE_REASONING_EFFORT = os.getenv("RESPONSE_REASONING_EFFORT", "")
    # 話し方の実例（`eve/persona_examples.py`・2026-10-01 ユーザ「N5＋実例で」）: ユーザ発話への応答で毎回見せる数（0＝使わない）。
    # 実測で 8（0/2/4/8/16 を比べ、8 で Neuro-sama の実際の返事に近づき 16 で横ばい）。実例は persona_examples.yaml（人が選ぶ）
    PERSONA_EXAMPLES_K = int(os.getenv("PERSONA_EXAMPLES_K", "8"))

    # タスク管理（J-1 inc1 = 土台ハーネス・単純な予約タスクのみ）。CALLFUNCTION_ENABLED 前提。
    TASK_ENABLED = os.getenv("TASK_ENABLED", "0") == "1"  # 既定 off
    TASK_FILE = os.getenv("TASK_FILE", "tasks.jsonl")  # 永続 JSONL（イベントログ）
    TASK_MAX = int(os.getenv("TASK_MAX", "500"))  # in-memory 上限（古い terminal を捨てる）
    TASK_RECONCILE_TICK_SEC = float(os.getenv("TASK_RECONCILE_TICK_SEC", "1.0"))  # スケジューラ周期
    TASK_ORPHAN_TIMEOUT_SEC = float(os.getenv("TASK_ORPHAN_TIMEOUT_SEC", "3600"))  # 起動時 Running 孤児の age-out
    # TaskAgent（J-1 inc2 = 賢いタスクLLMの境界つきループ）。コードが境界を所有。
    # J-3: 5 → 12。**要件を満たすための必須変更であって調整ではない**。
    # 「エラーが出たら自分で検索して直す」は最短でも 検索→書く→走らせる→失敗→検索→直す→
    # 走らせる＝7手かかり、5手では構造的に完遂できない（_summarize_failure に落ちる）。
    TASK_AGENT_MAX_STEPS = int(os.getenv("TASK_AGENT_MAX_STEPS", "12"))  # ループ最大手数
    # 同上: 実行(最大70s)を複数回含むので 3分では足りない。10分に広げる。
    TASK_AGENT_TIMEOUT_SEC = float(os.getenv("TASK_AGENT_TIMEOUT_SEC", "600.0"))  # 1ゴールの壁(monotonic)時間上限

    # Web検索（J-2 inc1 = ddgs スニペットのみ）。TASK_ENABLED 前提（TaskAgent 専用能力）。
    SEARCH_ENABLED = os.getenv("SEARCH_ENABLED", "0") == "1"  # 既定 off（実機で明示 on）
    SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))  # 上位件数
    SEARCH_TIMEOUT_SEC = float(os.getenv("SEARCH_TIMEOUT_SEC", "20.0"))  # 検索1回の総予算（auto のフォールバック実測 max 14.8s + 余裕）
    SEARCH_SOCKET_TIMEOUT_SEC = float(os.getenv("SEARCH_SOCKET_TIMEOUT_SEC", "5.0"))  # ddgs ソケット側（thread 自然死の保証）
    SEARCH_CACHE_TTL_SEC = float(os.getenv("SEARCH_CACHE_TTL_SEC", "600.0"))  # 同一クエリの再検索抑制（再取得は fresh=true）
    SEARCH_COOLDOWN_SEC = float(os.getenv("SEARCH_COOLDOWN_SEC", "90.0"))  # 連続失敗時の冷却（検知後の回復窓は30sより長い実測 2026-07-17）
    SEARCH_MIN_INTERVAL_SEC = float(os.getenv("SEARCH_MIN_INTERVAL_SEC", "5.0"))  # 実検索の最小間隔（auto=多エンジン分散なら3s連打8/8成功の実測。単一エンジン時の保険として維持）
    SEARCH_RETRY_EMPTY_DELAY_SEC = float(os.getenv("SEARCH_RETRY_EMPTY_DELAY_SEC", "8.0"))  # 0件時の1回だけ再検索の待ち（0=無効）
    SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "auto")  # ddgs バックエンド（auto=マルチエンジン自動フォールバック。DNS 修正後 5 エンジン生存・3s連打8/8の実測 2026-07-18。DNS遮断回線では "duckduckgo" 固定に戻す）
    # ddgs が中で使う HTTP クライアント（primp）の名前解決。既定 system＝OS と同じ名前解決（ブラウザと同じ）。
    # primp の既定は**独自の名前解決**で、この PC の Wi-Fi が配る IPv6 の DNS（問い合わせを拒否する）に聞いて
    # 「dns error: Query Refused」で諦めていた（実測 2026-09-28・Windows 自体はルータの DNS で引ける）。
    # 空にするとライブラリの既定に戻す（DNS サーバの IP を書けばそれを使う・例 192.168.0.1）
    SEARCH_DNS_RESOLVER = os.getenv("SEARCH_DNS_RESOLVER", "system")

    # 作業フォルダ（J-3 inc1 = AI 専用ワークスペース）。TASK_ENABLED 前提（TaskAgent 専用能力）。
    WS_ENABLED = os.getenv("WS_ENABLED", "0") == "1"  # 既定 off（実機で明示 on）
    # **リポジトリ外・ユーザプロファイル外**に置く（既定 C:\eve_ws\workspace）。理由:
    # (1) リポジトリ直下には .env（APIキー）と *.jsonl（会話履歴/RAG/タスク）があり、封じ込めの
    #     穴1個で鍵と全記憶に届く。(2) 128GB を git worktree 内に置くと git status が実用速度を
    #     失い誤 commit の被害が最大化する。(3) inc2 で専用アカウントに切り替える際、プロファイル
    #     配下だと ACL が届かない（C:\Users\tester に BUILTIN\Users の ACE が無い実測）。
    WS_ROOT = os.getenv("WS_ROOT", "")  # 空=既定を計算
    WS_QUOTA_GB = float(os.getenv("WS_QUOTA_GB", "128"))  # 書込ゲートの閾値（超過しても削除はしない）
    WS_WRITE_MAX_CHARS = int(os.getenv("WS_WRITE_MAX_CHARS", "200000"))  # 1回の書込上限（同期実行の妥当性を担保）
    WS_READ_MAX_CHARS = int(os.getenv("WS_READ_MAX_CHARS", "4000"))  # LLM に返す上限（ファイル上限とは別物）
    WS_MAX_FILES = int(os.getenv("WS_MAX_FILES", "20000"))  # 走査の bound（実測: scandir で 77k件/993ms）
    WS_RESCAN_SEC = float(os.getenv("WS_RESCAN_SEC", "600"))  # 台帳ドリフトの定期補正
    # プログラム実行（J-3 inc2）。**WS_ENABLED 前提**かつ setup_sandbox.py apply 済みが前提。
    # ファイル操作とは別フラグにする: 「作業フォルダを与える」と「任意コード実行を許す」は
    # リスクが桁で違い、ユーザが別々に ON/OFF したくなる設定項目だから（CLAUDE.md の規律）。
    WS_EXEC_ENABLED = os.getenv("WS_EXEC_ENABLED", "0") == "1"  # 既定 off
    # 実行を許すファイル名。**セキュリティ境界ではない**（python を許した時点で任意コード実行を
    # 許しており ctypes で Win32 API も呼べる）。本当の境界は専用アカウント + ACL + Job Object。
    # ここが守るのは「うっかり」と予測可能性だけ。
    WS_EXEC_ALLOW = os.getenv("WS_EXEC_ALLOW", "python,pip,git,ffmpeg,curl,tar,7z,node,npm")
    WS_RUN_TIMEOUT_SEC = float(os.getenv("WS_RUN_TIMEOUT_SEC", "60"))  # 内側（runner の subprocess）
    WS_RUN_GRACE_SEC = float(os.getenv("WS_RUN_GRACE_SEC", "10"))  # 外側の猶予（Job 強制終了まで）
    WS_RUN_MEM_MB = int(os.getenv("WS_RUN_MEM_MB", "2048"))  # Job のメモリ上限
    WS_RUN_MAX_PROCS = int(os.getenv("WS_RUN_MAX_PROCS", "32"))  # fork bomb 対策
    WS_MAX_OUTPUT_CHARS = int(os.getenv("WS_MAX_OUTPUT_CHARS", "8000"))  # LLM に返す出力上限
    # 画面表示（J-3 inc3）。**WS_ENABLED 前提**。別プロセスの Tk ビューアで開く（管理者操作は不要）。
    WS_VIEWER_ENABLED = os.getenv("WS_VIEWER_ENABLED", "0") == "1"  # 既定 off
    WS_VIEWER_MAX_WINDOWS = int(os.getenv("WS_VIEWER_MAX_WINDOWS", "5"))  # 同時に開ける窓の上限
    # 表示先。`window`=UI の「タスクウィンドウ」の中に集約（既定）/ `external`=従来どおり
    # 画面の好きな位置に別プロセスで開く。**UI が無いヘッドレス実行では常に external 相当**
    # （UI が `viewer_open_fn` を差さないので自動的に別プロセスへ落ちる）。UI から即時に
    # 切り替えられる（再登録不要）。
    WS_VIEWER_MODE = os.getenv("WS_VIEWER_MODE", "window")
    # タスクウィンドウで HTML を**動かして**表示する（既定 ON・PySide6-Addons が要る）。
    # イブに「押せるもの」を作らせる唯一の経路（サンドボックスで GUI は出せない＝意図的）。
    # 入っていなければ自動で HTML ソースの表示に落ちる（fail-soft）ので、OFF にするのは
    # WebEngine が環境で問題を起こす時の逃げ道。
    TASKWIN_HTML_ENABLED = os.getenv("TASKWIN_HTML_ENABLED", "1") not in ("0", "false", "no")
    # 子ウィンドウを何回クリックしたら「ユーザが操作中」と見なすか（既定 3）。
    # 1回だと誤クリックで居座るため（ユーザ指定 2026-08-15）。0 で保護を切る。
    TASKWIN_PROTECT_CLICKS = int(os.getenv("TASKWIN_PROTECT_CLICKS", "3"))

    # Alchemy（ユーザ作のブラウザ対戦ゲーム・`eve/alchemy/`）。**機能フラグ**（既定 off）。
    # ユーザ指定 2026-10-10「普段は入れずに、画面や会話からAlchemyをやりそうになりそうならプレイできるように準備」
    # 「キーワード対戦でキーワード『AIバトル』として部屋作らせるor部屋に入らせる」「画面認識はAIが操作するはずの画面を
    # 認識させる。ユーザーのPC画面は認識させない」。CALLFUNCTION_ENABLED 前提（部屋の出入りは応答AI の機能）
    ALCHEMY_ENABLED = os.getenv("ALCHEMY_ENABLED", "0") == "1"
    # ゲームの場所（ユーザがブラウザで開くのと同じ所。PC で動かすなら http://127.0.0.1:8000/）
    ALCHEMY_URL = os.getenv("ALCHEMY_URL", "https://alchemy-game.fly.dev/")
    ALCHEMY_KEYWORD = os.getenv("ALCHEMY_KEYWORD", "AIバトル")  # 合言葉の部屋
    ALCHEMY_PLAYER_NAME = os.getenv("ALCHEMY_PLAYER_NAME", "イブ")
    # 遊ばせ方（ユーザ案）: action＝1行動ごとに LLM が候補から選ぶ / strategy＝LLM は作戦（送りたい・減らしたい素材）
    # だけ決め、手はコードがその作戦で打つ。どちらも LLM が締切に間に合わなければコードの手で打つ
    ALCHEMY_PLAY_MODE = os.getenv("ALCHEMY_PLAY_MODE", "action")
    # イブのゲーム画面を窓として出すか（既定 0＝画面なしのブラウザ。1 にするとイブの画面を横で見られる）
    ALCHEMY_SHOW_WINDOW = os.getenv("ALCHEMY_SHOW_WINDOW", "0") == "1"
    # イブの画面を動かすブラウザ（既定 chrome＝PC の Chrome。空なら Playwright 同梱の Chromium）
    ALCHEMY_BROWSER_CHANNEL = os.getenv("ALCHEMY_BROWSER_CHANNEL", "chrome")
    # 準備（ルールを会話に入れた状態）をこの秒数 Alchemy の話が無ければ解く（部屋にいる間は解かない）
    ALCHEMY_PREPARE_TTL_SEC = float(os.getenv("ALCHEMY_PREPARE_TTL_SEC", "900"))
    # 判断役（ModelRegistry の game 役・GAME_MODEL）の reasoning_effort（空＝渡さない）と、1回に待つ上限[秒]
    ALCHEMY_REASONING_EFFORT = os.getenv("ALCHEMY_REASONING_EFFORT", "low")
    ALCHEMY_MAX_THINK_SEC = float(os.getenv("ALCHEMY_MAX_THINK_SEC", "20"))
    # 部屋でイブだけが待ってこの秒数たったら来るかを確かめ、その後この秒数返事が無ければ抜ける（ユーザ 2026-10-11）
    ALCHEMY_NUDGE_SEC = float(os.getenv("ALCHEMY_NUDGE_SEC", "300"))
    ALCHEMY_GIVEUP_SEC = float(os.getenv("ALCHEMY_GIVEUP_SEC", "30"))
    # 確かめて返事はあったのに来ない（待ち直し）がこの回数たまったら、次は確かめずに抜ける
    ALCHEMY_MAX_REWAITS = int(os.getenv("ALCHEMY_MAX_REWAITS", "2"))

    @classmethod
    def validate(cls) -> list[str]:
        """不足している必須設定を列挙。空リストなら起動可（Start ゲートで使う）。"""
        missing: list[str] = []
        # 応答役には最低1つの LLM provider 鍵が要る（Claude 停止中の代用前提）
        if not (cls.OPENAI_API_KEY or cls.GEMINI_API_KEY or cls.GROQ_API_KEY):
            missing.append("LLM provider 鍵が1つも無い (OPENAI/GEMINI/GROQ のいずれか必須)")
        return missing
