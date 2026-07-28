"""`.env` 読み込みと Config。起動時 `validate()` で不足設定を列挙（UI Start ゲート用）。

python-dotenv が未インストールでも最小パースで `.env` を読む（F0 を依存なしで動かす）。
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


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
    TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID", "")

    # 外部アプリ
    VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
    VTS_PATH = os.getenv("VTS_PATH", "")
    VOICEVOX_PATH = os.getenv("VOICEVOX_PATH", "")

    # VOICEVOX 合成パラメータ
    VV_SPEAKER = int(os.getenv("VV_SPEAKER", "1"))
    VV_SPEED = float(os.getenv("VV_SPEED", "1.0"))
    VV_PITCH = float(os.getenv("VV_PITCH", "0.0"))

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

    # STT バックエンド（既定 = gpt-4o-transcribe。groq は幻聴多く非推奨=ロールバック用）
    STT_BACKEND = os.getenv("STT_BACKEND", "openai")  # openai | groq
    STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe")

    # 短期記憶（F3 ConversationCache）
    # 会話ログの永続先（タイムスタンプ付き JSONL・ロケット鉛筆方式）。
    HISTORY_FILE = os.getenv("HISTORY_FILE", "conversation_history.jsonl")
    # メモリに保持する最大ターン数（古いものから押し出す＝ロケット鉛筆）。
    HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "100"))
    # 応答LLM に注入する直近ターン数（≈3往復）。レイテンシと過去逸れのため小さく保つ。
    RECENT_TURN_COUNT = int(os.getenv("RECENT_TURN_COUNT", "6"))

    # 長期記憶（F3.5 RAG / 連想想起）
    RAG_FILE = os.getenv("RAG_FILE", "rag_memory.jsonl")  # 永続 JSONL（埋め込み込み）
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
    # ユーザ 5秒無言で発話判定LLM を回す（企画書準拠）。trigger 兼フラット再評価カデンス。
    # ＝バックオフで間隔を伸ばさず 5秒ごとに連続評価し続ける（実世界を細かく観測する意図）。
    SILENCE_THRESHOLD_SEC = float(os.getenv("SILENCE_THRESHOLD_SEC", "5.0"))
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
    VLM_STATIC_VISION_MAX_SEC = float(os.getenv("VLM_STATIC_VISION_MAX_SEC", "600.0"))
    VLM_BLANK_STD_THRESHOLD = float(os.getenv("VLM_BLANK_STD_THRESHOLD", "6.0"))  # 黒/空白検出（std 未満=取得不可・A11）

    # Call-Function（J-0 即時 Call-Function・increment 1 は read-only 能力のみ）。
    CALLFUNCTION_ENABLED = os.getenv("CALLFUNCTION_ENABLED", "0") == "1"  # 既定 off（実機で明示 on）

    # タスク管理（J-1 inc1 = 土台ハーネス・単純な予約タスクのみ）。CALLFUNCTION_ENABLED 前提。
    TASK_ENABLED = os.getenv("TASK_ENABLED", "0") == "1"  # 既定 off
    TASK_FILE = os.getenv("TASK_FILE", "tasks.jsonl")  # 永続 JSONL（イベントログ）
    TASK_MAX = int(os.getenv("TASK_MAX", "500"))  # in-memory 上限（古い terminal を捨てる）
    TASK_RECONCILE_TICK_SEC = float(os.getenv("TASK_RECONCILE_TICK_SEC", "1.0"))  # スケジューラ周期
    TASK_ORPHAN_TIMEOUT_SEC = float(os.getenv("TASK_ORPHAN_TIMEOUT_SEC", "3600"))  # 起動時 Running 孤児の age-out
    # TaskAgent（J-1 inc2 = 賢いタスクLLMの境界つきループ）。コードが境界を所有。
    TASK_AGENT_MAX_STEPS = int(os.getenv("TASK_AGENT_MAX_STEPS", "5"))  # ループ最大手数
    TASK_AGENT_TIMEOUT_SEC = float(os.getenv("TASK_AGENT_TIMEOUT_SEC", "180.0"))  # 1ゴールの壁(monotonic)時間上限=3分で強制終了

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

    @classmethod
    def validate(cls) -> list[str]:
        """不足している必須設定を列挙。空リストなら起動可（Start ゲートで使う）。"""
        missing: list[str] = []
        # 応答役には最低1つの LLM provider 鍵が要る（Claude 停止中の代用前提）
        if not (cls.OPENAI_API_KEY or cls.GEMINI_API_KEY or cls.GROQ_API_KEY):
            missing.append("LLM provider 鍵が1つも無い (OPENAI/GEMINI/GROQ のいずれか必須)")
        return missing
