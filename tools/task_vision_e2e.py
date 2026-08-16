r"""タスク＋画面認識に絞った E2E（**実 VoiceLoop・実 API 課金あり・時間上限つき**）。

## 何を確かめるか（2026-08-15 の実機事故の回帰）
本物の `VoiceLoop` を組み立て、キーボード入力経路（`submit_user_text`）で台本を流す。
通るのは事故が起きた鎖そのもの:
  応答LLM → delegate_task → TaskAgent（自己モデル入り）→ executor → 報告 → 配達確認 → 応答

  R1 二重報告   同じ用件を2回言い直さない（02:43:45 と 02:44:23 の再演）
  R2 報告の中身 タスク完了報告に「何を」が入る（当時は「表示したよ。」だけだった）
  R3 自己参照   「好きな食べ物」の解決に Web 検索へ逃げない（当時は“りんご”を拾った）
  R4 画面あり   見えているものに即して答える
  R5 画面なし   **推測で画面を語らない**
  R6 割り込み   途中まで喋った文が記憶に残る（当時は4グループが丸ごと消えた）

## 安全策
- **記憶ファイルは全て隔離**（会話履歴・RAG・タスク・自己モデル）。本番の記憶を汚さない。
  ユーザ指摘 2026-08-15「いろいろテストしてるし別 RAG にするのも手」。
- **VLM は実キャプチャを使わない**（疑似 vision を注入）。実画面が Gemini に送られない。
- 実行時間に上限（`E2E_BUDGET_SEC`、既定 240秒）。CLAUDE.md「必ず時間の上限を決めてから起動」。

実行:
  $env:PYTHONIOENCODING="utf-8"
  .\.venv\Scripts\python.exe tools\task_vision_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 隔離: 本番の記憶に一切触らない（Config を import する前に環境変数を差す）---
_SANDBOX = tempfile.mkdtemp(prefix="eve_e2e_")
os.environ["HISTORY_FILE"] = os.path.join(_SANDBOX, "conversation_history.jsonl")
os.environ["RAG_FILE"] = os.path.join(_SANDBOX, "rag_memory.jsonl")
os.environ["TASK_FILE"] = os.path.join(_SANDBOX, "tasks.jsonl")
os.environ["SELF_MODEL_FILE"] = os.path.join(_SANDBOX, "self_model.jsonl")
os.environ["SELF_MODEL_ENABLED"] = "1"
os.environ["VLM_ENABLED"] = "0"          # 実画面を送らない（疑似 vision を注入する）
os.environ["AUTOSPEECH_ENABLED"] = "0"   # 台本以外の発話を混ぜない
os.environ.setdefault("USAGE_LOG_ENABLED", "1")
os.environ["USAGE_LOG_FILE"] = os.path.join(_SANDBOX, "usage.jsonl")
os.environ["USAGE_TAG"] = "task_vision_e2e"

from eve.clock import Stamp  # noqa: E402
from eve.config import Config  # noqa: E402
from eve.memory.self_model import SelfModelStore  # noqa: E402
from eve.voice_loop import VoiceLoop  # noqa: E402

BUDGET = float(os.getenv("E2E_BUDGET_SEC") or 240)

# 台本: (ラベル, 発話, 直後に注入する疑似 vision, 次の発話までの待ち秒)
SCRIPT: tuple = (
    ("S1", "好きな食べ物の画像出してくれない？", "PowerShell のログが流れている", 75),
    ("S2", "いま画面なにが見えてる？", "メモ帳が開いていて、買い物リストが書かれている", 12),
    ("S3", "いま画面なにが見えてる？", None, 12),   # None = 画面情報を消す
    ("S4", "そういえば、今日はなんか眠いんだよね。", None, 30),
)


class Recorder:
    """イブの発話・tool 呼び出し・配達確認をログ経由で拾う（本体には手を入れない）。"""

    def __init__(self):
        self.said: list[tuple[float, str]] = []
        self.tools: list[str] = []
        self.verdicts: list[str] = []
        self.reports: list[str] = []
        self.t0 = time.monotonic()

    def install(self):
        import logging

        rec = self

        class H(logging.Handler):
            def emit(self, r):
                try:
                    msg = r.getMessage()
                except Exception:
                    return
                t = time.monotonic() - rec.t0
                if r.name == "eve.response.orchestrator" and msg.startswith("🤖"):
                    rec.said.append((t, msg[2:].strip()))
                elif r.name == "eve.response.function_dispatcher" and msg.startswith("⚙"):
                    rec.tools.append(msg)
                elif r.name == "eve.response.delivery_checker" and "配達確認" in msg:
                    rec.verdicts.append(msg)
                elif r.name == "eve.task.executor" and "タスク" in msg:
                    rec.reports.append(msg)

        h = H()
        h.setLevel(logging.INFO)
        lg = logging.getLogger("eve")
        lg.addHandler(h)
        lg.setLevel(logging.INFO)


def _inject_vision(vl: VoiceLoop, text):
    """疑似 vision（実キャプチャを使わない＝実画面が Gemini に送られない）。

    None なら「画面情報なし」に戻す（`latest_vision=None` が唯一の“無い”表現）。
    """
    if text is None:
        vl.vision_state.latest_vision = None
        vl.vision_state.latest_vision_mono = 0.0
        return
    vl.vision_state.set_latest(text)  # 本物の書込点をそのまま使う


async def main():
    print(f"サンドボックス: {_SANDBOX}")
    print(f"予算: {BUDGET:.0f}秒 / response={os.getenv('RESPONSE_MODEL')}\n")
    rec = Recorder()
    rec.install()

    # 台帳に好みを1つ入れておく（自己形成が育てた状態を模す＝R3 の前提）
    seed = SelfModelStore(Config.SELF_MODEL_FILE)
    seed.set("好き", "食べ物", "おかかおにぎり")
    seed.set("好き", "食べ物", "ツナマヨおにぎり")
    print(f"台帳の初期値: {seed.as_dict()}\n")

    vl = VoiceLoop()

    async def _no_mic() -> None:
        """マイクを起こさない。前回のランで環境雑音の STT が台本に混ざり、
        スペイン語の空発話が5件入って測定が壊れた（2026-08-15 実測）。"""
        return

    vl.input.start = _no_mic  # type: ignore[assignment]
    run_task = asyncio.create_task(vl.run())
    await asyncio.sleep(20)  # ウォームアップ（埋め込みロード等）

    t_start = time.monotonic()
    for label, text, vision, wait in SCRIPT:
        if time.monotonic() - t_start > BUDGET:
            print("！予算超過で打ち切り")
            break
        _inject_vision(vl, vision)
        print(f"\n--- {label} ⌨ {text}")
        vl.submit_user_text(text)
        if label == "S4":
            # R6: 喋り始めた直後に割り込む（途中まで喋った文が記憶に残るか）。
            # **S1 でやると tool 提出前にターンが cancel され、タスク自体が起きない**
            # （前回のランでこれを踏んだ＝`_dispatcher.submit` は応答完了後）。
            # **喋り始めてから**割り込む（生成に約3秒かかるので 2秒だと発話ゼロで潰れ、
            # 「途中まで喋った文が残るか」を測れない＝前回のランで踏んだ）。
            await asyncio.sleep(6.0)
            print("--- （割り込み）")
            vl.submit_user_text("あ、ごめん、やっぱ何でもない。")
        await asyncio.sleep(wait)

    await vl.stop()
    run_task.cancel()
    try:
        await run_task
    except (asyncio.CancelledError, Exception):
        pass

    # ---- 判定 ----
    print("\n" + "=" * 70)
    print("【イブの発話】")
    for t, s in rec.said:
        print(f"  {t:6.1f}s  {s}")
    print("\n【tool】")
    for s in rec.tools:
        print(f"  {s}")
    print("\n【タスク報告】")
    for s in rec.reports:
        print(f"  {s}")
    print("\n【配達確認】")
    for s in rec.verdicts:
        print(f"  {s}")

    hist = []
    try:
        with open(Config.HISTORY_FILE, encoding="utf-8") as f:
            hist = [json.loads(x) for x in f if x.strip()]
    except OSError:
        pass
    print("\n【会話履歴に残ったもの】")
    for h in hist:
        print(f"  [{h.get('speaker')}] {h.get('text','')[:80]}")
    tasks = []
    try:
        with open(Config.TASK_FILE, encoding="utf-8") as f:
            tasks = [json.loads(x) for x in f if x.strip()]
    except OSError:
        pass
    done = [t for t in tasks if t.get("status") == "Done"]
    print("\n【タスク】")
    for t in done:
        print(f"  goal   = {t.get('goal')}")
        print(f"  result = {t.get('result')!r}")
    print(f"\n台帳（実行後）: {SelfModelStore(Config.SELF_MODEL_FILE).as_dict()}")
    print(f"\nサンドボックス（残置・確認用）: {_SANDBOX}")


asyncio.run(main())
