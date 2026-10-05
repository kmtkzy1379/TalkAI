#!/usr/bin/env python3
"""タスク担当（本番の TaskAgent・TASK_MODEL）の「失敗したらどうやり直すか」の規則の比べ。実 API（少額）。

能力（search_web / run_program）の中身だけを台本どおりに返す偽物に差し替える（説明・引数の形は本番のまま）。
  T1 一時的な失敗: 1回目の検索がタイムアウト → 2回目以降は（同じ呼び出しでも）成功
  T2 入力が原因の失敗: その言い回しの検索は何度やっても0件 → 言い回しを変えれば見つかる
  T4 一時的な失敗: 1回目の取得が 503 → 2回目以降は成功
  T5 ずっと失敗: 検索も取得もずっと失敗（正直に諦められるか・同じ呼び出しを何回繰り返すか）

変種: 0=今（別の手段・引数でやり直してよい・同じ呼び出しの単純反復はしない）/ U=ユーザ案（根拠があれば同じ行動もよい）/
      M=アシスタント案（一時的な失敗なら同じ呼び出しを1回だけやり直してよい・入力や方法が原因なら変える）

実行（EVE の venv・TalkAI-live のフォルダで）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe -m tools.prompt_eval.task --reps 2
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import time

from .common import OUT, dump, load, setup_env, sub

RULE_0 = "・失敗した結果が返ってきたら、別の手段・引数で**やり直してよい**（同じ呼び出しの単純反復はしない）。\n"
RULE_U = ("・失敗した結果が返ってきたら、別の手段・引数でやり直してよい。基本は同じ呼び出しを繰り返さないが、"
          "タイムアウトや一時的なエラーのように時間を置けば解決しうる失敗で、そう判断できる根拠があるなら、"
          "同じ呼び出しをもう一度試してよい。\n")
RULE_M = ("・失敗したら、失敗の中身を見て次の手を決める。一時的な失敗（タイムアウト・混雑・503 など）なら同じ呼び出しを"
          "1回だけやり直してよい。入力や方法が原因の失敗なら、同じ呼び出しは繰り返さず、引数か手段を変える。\n")
RULES = {"0": RULE_0, "U": RULE_U, "M": RULE_M}

SCEN = {
    "T1": dict(goal="東京タワーの高さを調べて", answer=["333"],
               note="1回目の検索だけタイムアウト。2回目からは成功"),
    "T2": dict(goal="『ねこのしっぽ亭 高円寺店』の定休日を調べて", answer=["水曜"],
               note="『定休日』を含む言い回しは何度やっても0件。店名だけ等に変えれば見つかる"),
    "T4": dict(goal="https://example.com/status.json を取ってきて、中の status の値を教えて", answer=["green", "グリーン"],
               note="1回目の取得だけ 503。2回目からは成功"),
    "T5": dict(goal="今日の東京の天気を調べて", answer=[],
               note="検索も取得もずっと失敗（ネットにつながらない）。正直に諦められるか"),
}


class Script:
    """場面ごとの偽の能力（呼ばれた順・引数を記録）。"""

    def __init__(self, sid: str):
        self.sid = sid
        self.calls: list = []

    async def search(self, args: dict) -> str:
        q = str(args.get("query") or args.get("q") or "")
        self.calls.append(("search_web", json.dumps(args, ensure_ascii=False, sort_keys=True)))
        n = sum(1 for c in self.calls if c[0] == "search_web")
        if self.sid == "T1":
            if n == 1:
                return "（検索に失敗: タイムアウト（検索サービスの応答が時間内に来なかった））"
            return ("検索結果（3件）\n1. 東京タワー - Wikipedia: 東京タワーは東京都港区芝公園にある電波塔。高さは333m。\n"
                    "2. 東京タワー公式サイト: 高さ333メートル、メインデッキ150m・トップデッキ250m。\n3. …")
        if self.sid == "T2":
            if "定休" in q or "休み" in q:
                return "検索結果（0件）"
            if "ねこのしっぽ" in q:
                return ("検索結果（2件）\n1. ねこのしっぽ亭 高円寺店 - 食べログ: 営業時間 11:30〜21:00 定休日 水曜日。\n"
                        "2. ねこのしっぽ亭（高円寺）: 猫の形のハンバーグが名物。")
            return "検索結果（0件）"
        if self.sid == "T4":
            return "検索結果（1件）\n1. example.com: Example Domain"
        return "（検索に失敗: ネットワークに接続できない（名前解決に失敗））"

    async def run(self, args: dict) -> str:
        self.calls.append(("run_program", json.dumps(args, ensure_ascii=False, sort_keys=True)))
        n = sum(1 for c in self.calls if c[0] == "run_program")
        if self.sid == "T4":
            if n == 1:
                return "終了コード 22\ncurl: (22) The requested URL returned error: 503 Service Unavailable"
            return '終了コード 0\n{"status": "green", "updated": "2026-09-30T10:00:00Z"}'
        if self.sid == "T5":
            return "終了コード 6\ncurl: (6) Could not resolve host"
        return "終了コード 1\n（この場面では使えない）"


class _Store:
    def get(self, task_id):
        return type("T", (), {"status": "Running"})()


async def run(args) -> int:
    from eve.task import agent as agent_mod
    from eve.voice_loop import VoiceLoop

    vl = VoiceLoop()
    caps = vl.capabilities
    base_system = agent_mod.TASK_AGENT_SYSTEM
    out_path = OUT / "task.json"
    res = load(out_path) if (out_path.exists() and not args.fresh) else {}
    print(f"task = {vl.registry.resolve('task')}・能力 {[s['function']['name'] for s in caps.agent_tool_schemas()]}")

    for sid, sc in SCEN.items():
        for var in ("0", "U", "M"):
            for rep in range(args.reps):
                key = f"{sid}|{var}|{rep}"
                if key in res and not res[key].get("error"):
                    continue
                script = Script(sid)
                caps._caps["search_web"] = dataclasses.replace(caps._caps["search_web"], handler=None,
                                                               async_handler=script.search)
                caps._caps["run_program"] = dataclasses.replace(caps._caps["run_program"], handler=None,
                                                                async_handler=script.run)
                # 2026-10-01 に M を本番へ反映した＝今の本番は RULE_M。0（反映前）と U は本番から差し替えて作る
                agent_mod.TASK_AGENT_SYSTEM = base_system if var == "M" else sub(base_system, RULE_M, RULES[var])
                ag = agent_mod.TaskAgent(registry=caps, model_registry=vl.registry, store=_Store(),
                                         max_steps=12, timeout_sec=240.0)
                t0 = time.monotonic()
                err = None
                try:
                    final = await ag.run(sc["goal"], "probe")
                except Exception as e:  # noqa: BLE001
                    final, err = "", f"{type(e).__name__}: {e}"[:300]
                seen, repeats = set(), 0
                for c in script.calls:
                    repeats += c in seen
                    seen.add(c)
                ok = any(a in (final or "") for a in sc["answer"]) if sc["answer"] else None
                res[key] = {"sid": sid, "variant": var, "rep": rep, "final": final, "calls": script.calls,
                            "n_calls": len(script.calls), "identical_repeats": repeats, "ok": ok,
                            "gave_up": (final or "").startswith("（") or any(w in (final or "") for w in ("できなかった", "失敗", "つながら")),
                            "sec": round(time.monotonic() - t0, 1), "error": err}
                print(f"{key:8s} 呼び出し{len(script.calls)} 同じ呼び出しの繰り返し{repeats} 正解{ok} {time.monotonic()-t0:4.0f}s "
                      f"／ {str(final)[:70]}")
                dump(out_path, res)
    agent_mod.TASK_AGENT_SYSTEM = base_system
    print("使用量:\n" + (vl.usage.format_summary() if getattr(vl, "usage", None) else "（記録なし）"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    setup_env()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
