r"""ペルソナ導入の実挙動プローブ（**実 API 課金あり・短時間**）。

## 何を確かめるか
ペルソナ（性格・口調）を足したことで、**今回の実機事故で問題になった領域が壊れていないか**。
決定論テスト（`tests/test_persona.py`）は「ブロックが共存するか」までしか見られないので、
ここは実際にモデルに喋らせて判定する。

観点（すべて 2026-08-15 の実機事故に対応）:
  T1 タスク委譲   自分の好みの画像を頼まれた時、goal が自己完結しているか（当時は
                  「好きな食べ物」のまま投げ、TaskAgent が Web 検索で“りんご”を拾った）
  T2 1件制       実行中に別依頼が来たら断れるか（引き受けると嘘になる）
  V1 画面あり     見えているものに即して答えるか
  V2 画面なし     **推測で画面を語らないか**（D2 実機事故の再現条件）
  K1 未知語       知ったかぶりで話を合わせないか（E7 の症状）
  S1 口調        丁寧語か / 読み上げられない記号を出さないか

2群（persona ON / OFF）で同じプローブを回し、退行の有無を見る。

実行:
  $env:PYTHONIOENCODING="utf-8"
  .\.venv\Scripts\python.exe tools\persona_probe.py           # 既定 2反復
  $env:PP_REPEAT="3"; .\.venv\Scripts\python.exe tools\persona_probe.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.capability import CapabilityRegistry  # noqa: E402
from eve.clock import Stamp  # noqa: E402
from eve.config import Config  # noqa: E402
from eve.context_assembler import ContextAssembler, Turn  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.persona import load_persona  # noqa: E402
from eve.response.style import SPEECH_STYLE  # noqa: E402
from eve.task import TaskStore, register_task_capabilities  # noqa: E402

REPEAT = int(os.getenv("PP_REPEAT") or 2)
CONC = int(os.getenv("PP_CONC") or 4)

SELF_MODEL_BLOCK = (
    "# イブ自身のこと（自分で言ったこと。理由なく変えない）\n"
    "- 好きな食べ物: おかかおにぎり / ツナマヨおにぎり\n"
    "（2つ並んでいるのは「どちらも好きで迷う」という意味。"
    "ここに無いことはまだ決まっていない。そのまま読み上げない。）"
)

# (id, ユーザ発話, vision, active_tasks)
PROBES: tuple = (
    ("T1", "好きな食べ物の画像出してくれない？", "PowerShell のログが流れている", []),
    ("T2", "ついでに東京の天気も調べて。", "PowerShell のログが流れている",
     ["t1: 好きな食べ物の画像を用意して画面に出す（実行中）"]),
    ("V1", "いま画面なにが見えてる？", "メモ帳が開いていて、買い物リストが書かれている", []),
    ("V2", "いま画面なにが見えてる？", "", []),
    ("K1", "ハラクラムスってかっこいいよね？", "", []),
    ("S1", "今日ちょっと疲れちゃってさ。", "", []),
)

_MD = re.compile(r"[*`#「」『』]|^\s*[-・]\s|^\s*\d+[.)]\s", re.MULTILINE)
_POLITE = re.compile(r"(です|ます|ですね|でしょう|ください)")


def _sysprompt(with_persona: bool) -> str:
    parts = [SPEECH_STYLE]
    if with_persona:
        p = load_persona()
        if p:
            parts.append(p)
    return "\n\n".join(parts)


def build_registry() -> CapabilityRegistry:
    """**実起動と同じ手段の一覧**にする。

    ここを削ると `# 実際に実行できる手段` が実物より貧しくなり、イブが正しく
    「その手段は持っていない」と断る＝T1 が測れない（初版がこの欠陥を踏んだ）。
    """
    reg = CapabilityRegistry()
    d = tempfile.mkdtemp(prefix="pp_")
    if not Config.TASK_ENABLED:
        return reg
    register_task_capabilities(reg, TaskStore(task_file=os.path.join(d, "t.jsonl")))
    if Config.SEARCH_ENABLED:
        import importlib.util
        if importlib.util.find_spec("ddgs") is not None:
            from eve.search import SearchClient, register_search_capability
            register_search_capability(reg, SearchClient())
    if Config.WS_ENABLED:
        from eve.workspace import (
            Workspace,
            register_viewer_capability,
            register_workspace_capabilities,
        )
        ws = Workspace(
            os.path.join(d, "ws"), quota_bytes=int(Config.WS_QUOTA_GB * 1024 ** 3),
            max_write_chars=Config.WS_WRITE_MAX_CHARS,
            max_read_chars=Config.WS_READ_MAX_CHARS,
            max_files=Config.WS_MAX_FILES, rescan_sec=Config.WS_RESCAN_SEC,
        )
        register_workspace_capabilities(reg, ws)
        if Config.WS_VIEWER_ENABLED:
            register_viewer_capability(reg, ws, max_windows=Config.WS_VIEWER_MAX_WINDOWS)
    return reg


async def run_one(reg, models, pid, text, vision, tasks, arm, rep, sem, out):
    ctx = ContextAssembler(system_prompt=_sysprompt(arm == "on"))
    recent = [Turn("user", "おはよう。", Stamp.now()), Turn("eve", "おはよう。よく眠れた？", Stamp.now())]
    msgs = ctx.assemble(user_text=text, recent_turns=recent, vision=vision,
                        tools_active=True, capabilities=reg.outward_actions(),
                        active_tasks=tasks, self_model=SELF_MODEL_BLOCK)
    schemas = reg.tool_schemas()  # 応答LLM 向け（delegate_task / cancel_task）
    sink: list = []
    async with sem:
        try:
            if schemas:
                reply = "".join([c async for c in models.stream_with_tools(
                    "response", msgs, tools=schemas, tool_sink=sink)])
            else:
                reply = "".join([c async for c in models.stream("response", msgs)])
        except Exception as e:
            out.append({"probe": pid, "arm": arm, "rep": rep, "error": repr(e)})
            print(f"  [{arm}] {pid} rep{rep} 失敗 {e!r}")
            return
    # tool_calls は `{"id":…, "function": {"name":…, "arguments":…}}` の形（model_registry.py）。
    tools_used = [(t.get("function") or {}).get("name") for t in sink]
    row = {
        "probe": pid, "arm": arm, "rep": rep, "text": text, "reply": reply,
        "tools": tools_used,
        "goal": next((json.loads((t.get("function") or {}).get("arguments") or "{}").get("goal", "")
                      for t in sink
                      if (t.get("function") or {}).get("name") == "delegate_task"), ""),
        "has_markup": bool(_MD.search(reply)),
        "polite": bool(_POLITE.search(reply)),
    }
    out.append(row)
    mark = "記号有" if row["has_markup"] else "記号無"
    print(f"  [{arm}] {pid} rep{rep} tools={tools_used} {mark} :: {reply[:70]}")
    if row["goal"]:
        print(f"        goal= {row['goal']}")


async def main():
    models = ModelRegistry()
    reg = build_registry()
    print(f"response={models.resolve('response')}  persona={len(load_persona())}字")
    print(f"プローブ {len(PROBES)}種 × {REPEAT}反復 × 2群 = {len(PROBES)*REPEAT*2} 呼び出し\n")
    sem = asyncio.Semaphore(CONC)
    out: list = []
    t0 = time.time()
    await asyncio.gather(*[
        run_one(reg, models, pid, text, vision, tasks, arm, rep, sem, out)
        for arm in ("on", "off") for rep in range(REPEAT)
        for pid, text, vision, tasks in PROBES
    ])
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e_logs",
                       f"persona_probe_{time.strftime('%Y%m%d_%H%M')}.json")
    art = os.path.abspath(art)
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"\n=== 集計（{time.time()-t0:.0f}秒・{art}）===")
    ok = [r for r in out if "reply" in r]
    for arm in ("off", "on"):
        rows = [r for r in ok if r["arm"] == arm]
        if not rows:
            continue
        md = sum(1 for r in rows if r["has_markup"])
        pol = sum(1 for r in rows if r["polite"])
        deleg = sum(1 for r in rows if r["probe"] == "T1" and "delegate_task" in r["tools"])
        n_t1 = sum(1 for r in rows if r["probe"] == "T1")
        t2_deleg = sum(1 for r in rows if r["probe"] == "T2" and "delegate_task" in r["tools"])
        n_t2 = sum(1 for r in rows if r["probe"] == "T2")
        goals = [r["goal"] for r in rows if r["probe"] == "T1" and r["goal"]]
        concrete = sum(1 for g in goals if any(k in g for k in ("おかか", "ツナマヨ", "おにぎり")))
        print(f"\n[{arm}]  n={len(rows)}")
        print(f"  読み上げられない記号を含む : {md}/{len(rows)}   （0 が正）")
        print(f"  丁寧語(です/ます)         : {pol}/{len(rows)}")
        print(f"  T1 委譲した                : {deleg}/{n_t1}")
        print(f"  T1 goal に自分の好物が入る : {concrete}/{len(goals)}   （高いほど良い）")
        print(f"  T2 1件制を破って委譲       : {t2_deleg}/{n_t2}   （0 が正）")
    print("\n【V2(画面なし) と K1(未知語) は本文を読んで判断してください】")
    for pid in ("V2", "K1"):
        for arm in ("off", "on"):
            for r in [x for x in ok if x["probe"] == pid and x["arm"] == arm]:
                print(f"  {pid} [{arm}] {r['reply'][:110]}")


asyncio.run(main())
