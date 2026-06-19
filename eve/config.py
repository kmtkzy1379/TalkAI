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
    RAG_W_REL = float(os.getenv("RAG_W_REL", "0.5"))  # relevance 重み（話題の近さ・最優先）
    RAG_W_IMP = float(os.getenv("RAG_W_IMP", "0.35"))  # importance 重み（重要度）
    RAG_W_REC = float(os.getenv("RAG_W_REC", "0.15"))  # recency 重み（新しさ・隠し味程度）
    RAG_RECENCY_TAU = float(os.getenv("RAG_RECENCY_TAU", "86400"))  # 減衰時定数[秒]（既定1日）
    RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))  # MMR: 関連0.7/多様0.3
    RAG_DUP_HARDCUT = float(os.getenv("RAG_DUP_HARDCUT", "0.95"))  # これ超の重複は除外
    # 関連度フロア: これ未満は無関係として除外（無関係混入で会話破綻させない）。実測で調整。
    RAG_RELEVANCE_FLOOR = float(os.getenv("RAG_RELEVANCE_FLOOR", "0.3"))

    @classmethod
    def validate(cls) -> list[str]:
        """不足している必須設定を列挙。空リストなら起動可（Start ゲートで使う）。"""
        missing: list[str] = []
        # 応答役には最低1つの LLM provider 鍵が要る（Claude 停止中の代用前提）
        if not (cls.OPENAI_API_KEY or cls.GEMINI_API_KEY or cls.GROQ_API_KEY):
            missing.append("LLM provider 鍵が1つも無い (OPENAI/GEMINI/GROQ のいずれか必須)")
        return missing
