"""プロンプト検証（2026-09-30・ユーザ依頼「修正案を検証・計測して実際に効くか」）の共通部品。

方針:
- messages は**本番の組み立て関数**（`ResponseOrchestrator._build_messages` 等）で作る。
- 変種は「本番の文字列 → 差し替え」で作る。差し替え元が見つからなければ例外で止める
  （黙って何も変わらない変種＝測ったつもりで測れていない、を防ぐ）。
- 記憶・使用量は一時フォルダ（本番の記憶を汚さない）。

実行は EVE の venv・TalkAI-live のフォルダで（各スクリプトの冒頭を参照）。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "eval_report" / "prompts"


def setup_env(extra: dict | None = None) -> Path:
    """eve を import する前に環境変数で上書きする（.env は既存の値を上書きしない）。"""
    tmp = Path(tempfile.mkdtemp(prefix="eve_prompt_eval_"))
    env = {
        "VLM_ENABLED": "0", "AUTOSPEECH_ENABLED": "0",
        "HISTORY_FILE": str(tmp / "history.jsonl"), "RAG_FILE": str(tmp / "rag.jsonl"),
        "TASK_FILE": str(tmp / "tasks.jsonl"), "SELF_MODEL_FILE": str(tmp / "self_model.jsonl"),
        "USAGE_LOG_ENABLED": "1", "USAGE_LOG_FILE": str(tmp / "usage.jsonl"),
    }
    env.update(extra or {})
    os.environ.update(env)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    OUT.mkdir(parents=True, exist_ok=True)
    return tmp


def stamp(sec_ago: float):
    from eve.clock import Stamp

    iso = (datetime.now(timezone.utc) - timedelta(seconds=sec_ago)).isoformat()
    return Stamp(iso=iso, mono=time.monotonic() - sec_ago)


def iso_ago(sec_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=sec_ago)).isoformat()


def live_persona_body() -> str:
    """今の本番の性格の水平線より後（`eve.persona.persona_path`＝手元の persona.local.md があればそちら）。"""
    from eve.persona import persona_path

    md = Path(persona_path()).read_text(encoding="utf-8").replace("\r\n", "\n")
    return md.split("\n---\n", 1)[1].strip() + "\n"


def sub(text: str, old: str, new: str) -> str:
    """必ず1回以上置き換える（元が無ければ止める＝変種が空振りしない）。"""
    if old not in text:
        raise ValueError(f"差し替え元が見つからない: {old[:60]!r}")
    return text.replace(old, new)


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


async def gather_limited(coros, limit: int = 6):
    sem = asyncio.Semaphore(limit)

    async def one(c):
        async with sem:
            return await c

    return await asyncio.gather(*(one(c) for c in coros))
