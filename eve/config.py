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

    # STT バックエンド（既定 = gpt-4o-transcribe。groq は幻聴多く非推奨=ロールバック用）
    STT_BACKEND = os.getenv("STT_BACKEND", "openai")  # openai | groq
    STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-transcribe")

    @classmethod
    def validate(cls) -> list[str]:
        """不足している必須設定を列挙。空リストなら起動可（Start ゲートで使う）。"""
        missing: list[str] = []
        # 応答役には最低1つの LLM provider 鍵が要る（Claude 停止中の代用前提）
        if not (cls.OPENAI_API_KEY or cls.GEMINI_API_KEY or cls.GROQ_API_KEY):
            missing.append("LLM provider 鍵が1つも無い (OPENAI/GEMINI/GROQ のいずれか必須)")
        return missing
