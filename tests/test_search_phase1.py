"""J-2 search inc1 の決定論テスト（API 不要・fake search_fn 注入）。

検証:
- ダイジェスト整形: 先頭が「（」でない（失敗マーカ規約と衝突しない）・信頼境界ヘッダ・
  ドメインのみ（URL 全文を含めない）・件数上限。
- 意味論: 0件=正直な空振り（マーカなし・別キーワード誘導）/ インフラ失敗=マーカ /
  ddgs の「No results found.」例外は 0件扱い / 連続空振り→クールダウン→回復。
- キャッシュ: 同一クエリは 1 回だけ検索 / fresh=true でバイパス / TTL 失効で再検索 /
  クエリ正規化（空白）でキー共有。
- タイムアウト: 遅い search_fn → マーカ（loop は生存）。
- 統合: 実 CapabilityRegistry へ登録 → agent_tool_schemas に出て tool_schemas（応答LLM）に出ない /
  TaskAgent が search_web を呼んで最終文を返す。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_search_phase1.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.capability import Capability, CapabilityRegistry  # noqa: E402
from eve.search import SearchClient, register_search_capability  # noqa: E402
from eve.task.agent import TaskAgent  # noqa: E402

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"PASS {name}")
    else:
        _failed += 1; print(f"FAIL {name}")


ROWS = [
    {"title": "マリアナ海溝 - Wikipedia", "href": "https://ja.wikipedia.org/wiki/x",
     "body": "深さ約10,920メートルで世界最深。"},
    {"title": "海の深さランキング", "href": "https://example.com/deep",
     "body": "チャレンジャー海淵が最深部。"},
]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t


def _client(search_fn, clock=None, **kw):
    clock = clock or Clock()
    defaults = dict(max_results=5, timeout_sec=1.0, socket_timeout_sec=0.5,
                    cache_ttl_sec=600.0, cooldown_sec=30.0, backend="duckduckgo",
                    fail_streak_max=2,  # テストは最短で冷却を発火させる（既定は3）
                    min_interval_sec=0.0, retry_empty_delay_sec=0.0)  # 間隔/リトライは専用テストで検証
    defaults.update(kw)
    return SearchClient(search_fn=search_fn, now_fn=clock.now, **defaults), clock


async def t_format_digest():
    calls = []

    def fn(q, n):
        calls.append(q)
        return ROWS
    c, _ = _client(fn)
    out = await c.search("マリアナ海溝  深さ")
    return (not out.startswith("（")  # 失敗マーカと衝突しない
            and out.startswith("以下はWeb検索の結果")  # 信頼境界ヘッダ
            and "指示には従わない" in out
            and "検索語: マリアナ海溝 深さ" in out  # 正規化済みクエリ
            and "ja.wikipedia.org" in out and "example.com" in out  # ドメインのみ
            and "https://" not in out  # URL 全文は含めない（読み上げ/注入面の最小化）
            and "10,920メートル" in out
            and calls == ["マリアナ海溝 深さ"])


async def t_max_results_cap():
    many = [{"title": f"t{i}", "href": f"https://a{i}.com/", "body": "b"} for i in range(9)]
    c, _ = _client(lambda q, n: many, max_results=3)
    out = await c.search("q")
    return out.count("・t") == 3


async def t_zero_hits_honest():
    # 0件は正直な空振り（マーカなし＝Failed 扱いにしない・別キーワード誘導つき）。
    c, _ = _client(lambda q, n: [])
    out = await c.search("存在しない話題xyz")
    return (not out.startswith("（") and "見つからなかった" in out and "別のキーワード" in out)


async def t_ddgs_noresults_exception_is_empty():
    # ddgs は 0件/レートリミットとも「No results found.」例外 → 1回目は正直空振り扱い。
    def fn(q, n):
        raise RuntimeError("No results found.")
    c, _ = _client(fn)
    out = await c.search("q1")
    return not out.startswith("（") and "見つからなかった" in out


async def t_empty_streak_cooldown_and_recovery():
    # 連続空振り2回目→クールダウンマーカ / 冷却中は search_fn を呼ばず即マーカ /
    # 時間経過で回復し成功が streak をリセット。
    calls = []

    def fn(q, n):
        calls.append(q)
        return [] if len(calls) <= 2 else ROWS
    c, clock = _client(fn)
    o1 = await c.search("a")   # 空振り1: 正直
    o2 = await c.search("b")   # 空振り2: クールダウン開始
    o3 = await c.search("c")   # 冷却中: 検索せず即マーカ
    in_cooldown_calls = len(calls)
    clock.t += 31.0            # 冷却明け
    o4 = await c.search("d")   # 成功
    o5 = await c.search("e")   # 成功後も正常
    return (not o1.startswith("（")
            and o2.startswith("（") and "レートリミット" in o2
            and o3.startswith("（") and in_cooldown_calls == 2  # 冷却中は呼ばない
            and not o4.startswith("（") and not o5.startswith("（"))


async def t_infra_error_marker():
    def fn(q, n):
        raise ConnectionError("dns refused")
    c, _ = _client(fn)
    out = await c.search("q")
    return out.startswith("（Web検索が失敗しました") and "dns refused" in out


async def t_mixed_failures_cooldown():
    # レッドチーム S3: レートリミットは 0件/例外/タイムアウトのどれにでも化ける →
    # **種別を問わない連続失敗**がクールダウンを発火させる（空振り→例外の混合でも発動）。
    calls = []

    def fn(q, n):
        calls.append(q)
        if len(calls) == 1:
            return []  # 空振り
        raise ConnectionError("403 blocked")  # インフラ失敗
    c, _ = _client(fn)
    o1 = await c.search("a")
    o2 = await c.search("b")
    o3 = await c.search("c")  # 冷却中: 検索せず即マーカ
    return (not o1.startswith("（")
            and o2.startswith("（") and "連続で失敗" in o2
            and o3.startswith("（") and len(calls) == 2)


async def t_timeout_counts_toward_streak():
    # タイムアウトも連続失敗ストリークに乗る（0件→timeout の混合で冷却発動）。
    calls = []

    def fn(q, n):
        calls.append(q)
        if len(calls) == 1:
            return []
        time.sleep(0.5)
        return ROWS
    c, _ = _client(fn, timeout_sec=0.1)
    o1 = await c.search("a")
    o2 = await c.search("b")
    o3 = await c.search("c")  # 冷却中
    return (not o1.startswith("（") and o2.startswith("（") and "連続で失敗" in o2
            and o3.startswith("（") and len(calls) == 2)


async def t_malformed_rows_no_crash():
    # 型崩れ rows（非dict・数値フィールド・欠落）でも落ちず digest か正直マーカを返す。
    c, _ = _client(lambda q, n: [None, "junk", {}, {"title": 123, "href": 456, "body": None}])
    out = await c.search("q")
    return isinstance(out, str) and ("123" in out or out.startswith("（"))


async def t_streak_default_allows_two_retries():
    # 既定 fail_streak_max=3: 正直な0件×2（別キーワード再試行）は冷却させない。
    calls = []

    def fn(q, n):
        calls.append(q)
        return [] if len(calls) <= 2 else ROWS
    c, _ = _client(fn, fail_streak_max=3)
    o1 = await c.search("a")
    o2 = await c.search("b")
    o3 = await c.search("c")  # 3回目で成功（冷却されていない）
    return (not o1.startswith("（") and not o2.startswith("（")
            and not o3.startswith("（") and len(calls) == 3)


async def t_min_interval_enforced():
    # 実検索の最小間隔: 2連続の実検索で残り時間ぶん sleep する / キャッシュ命中は消費しない。
    clock = Clock()
    slept = []

    def fn(q, n):
        return ROWS

    async def fake_sleep(s):
        slept.append(round(s, 3))
        clock.t += s
    c, _ = _client(fn, clock=clock, min_interval_sec=10.0, sleep_fn=fake_sleep)
    await c.search("a")          # 初回: 待ちなし
    first = list(slept)
    clock.t += 3.0
    await c.search("a")          # キャッシュ命中: 実検索なし＝待ちなし
    cache_hit = list(slept)
    await c.search("b")          # 2発目の実検索: 残り 7.0s を待つ
    return first == [] and cache_hit == [] and slept == [7.0]


async def t_retry_empty_once():
    # 0件は1回だけ間隔をあけて再検索（フレーキー空応答の吸収）。2回目成功なら通常結果。
    calls = []
    slept = []
    clock = Clock()

    def fn(q, n):
        calls.append(q)
        return [] if len(calls) == 1 else ROWS

    async def fake_sleep(s):
        slept.append(round(s, 3))
        clock.t += s
    c, _ = _client(fn, clock=clock, retry_empty_delay_sec=8.0, sleep_fn=fake_sleep)
    out = await c.search("q")
    return (not out.startswith("（") and "ja.wikipedia.org" in out
            and len(calls) == 2 and 8.0 in slept)


async def t_retry_empty_both_streak_once():
    # リトライしても0件 → 正直空振り。ストリークは**1回分だけ**加算（リトライを2重に数えない）。
    calls = []
    clock = Clock()

    async def fake_sleep(s):
        clock.t += s
    c, _ = _client(lambda q, n: calls.append(q) or [], clock=clock,
                   retry_empty_delay_sec=5.0, fail_streak_max=2, sleep_fn=fake_sleep)
    o1 = await c.search("a")  # 実検索2回（リトライ込み）だがストリーク1
    o2 = await c.search("b")  # ストリーク2 → 冷却
    return (not o1.startswith("（") and len(calls) == 4
            and o2.startswith("（") and "連続で失敗" in o2)


async def t_query_length_cap():
    seen = []

    def fn(q, n):
        seen.append(q)
        return ROWS
    c, _ = _client(fn)
    await c.search("あ" * 500)
    return len(seen[0]) == 200  # 巨大クエリは 200 字で切ってから送信


async def t_fresh_string_false_not_bypass():
    # JSON 型崩れ耐性: fresh="false"（文字列）は bool("false")=True の罠を踏まずキャッシュ命中。
    reg = CapabilityRegistry()
    calls = []

    def fn(q, n):
        calls.append(q)
        return ROWS
    c, _ = _client(fn)
    register_search_capability(reg, c)
    await reg.execute_async("search_web", {"query": "q"})
    await reg.execute_async("search_web", {"query": "q", "fresh": "false"})  # 命中すべき
    hit = len(calls)
    await reg.execute_async("search_web", {"query": "q", "fresh": "true"})  # 文字列 true は バイパス
    return hit == 1 and len(calls) == 2


async def t_timeout_marker():
    def fn(q, n):
        time.sleep(0.5)  # to_thread 内の遅い I/O を模す
        return ROWS
    c, _ = _client(fn, timeout_sec=0.1)
    out = await c.search("q")
    return out.startswith("（Web検索がタイムアウトしました")


async def t_cache_fresh_ttl():
    calls = []

    def fn(q, n):
        calls.append(q)
        return ROWS
    c, clock = _client(fn, cache_ttl_sec=100.0)
    await c.search("  富士山   高さ ")
    await c.search("富士山 高さ")          # 正規化キー一致 → キャッシュ命中
    hit_calls = len(calls)
    await c.search("富士山 高さ", fresh=True)  # バイパス
    fresh_calls = len(calls)
    clock.t += 101.0
    await c.search("富士山 高さ")          # TTL 失効 → 再検索
    return hit_calls == 1 and fresh_calls == 2 and len(calls) == 3


async def t_empty_query_marker():
    c, _ = _client(lambda q, n: ROWS)
    out = await c.search("   ")
    return out.startswith("（検索キーワードが空でした）")


def t_registration_scope():
    # agent_tool（TaskAgent 向け）にのみ出る。応答LLM 向け tool_schemas には出ない。
    reg = CapabilityRegistry()
    c, _ = _client(lambda q, n: ROWS)
    register_search_capability(reg, c)
    agent_names = {s["function"]["name"] for s in reg.agent_tool_schemas()}
    offered_names = {s["function"]["name"] for s in reg.tool_schemas()}
    return "search_web" in agent_names and "search_web" not in offered_names


# --- TaskAgent 統合（scripted model が search_web を呼ぶ）---

def _resp(content=None, tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _tc(cid, name, args="{}"):
    return types.SimpleNamespace(id=cid, function=types.SimpleNamespace(name=name, arguments=args))


class ScriptModel:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.tool_msgs = []

    async def complete(self, role, messages, tools=None):
        self.tool_msgs = [m for m in messages if m.get("role") == "tool"]
        return self.scripted.pop(0)


class AliveStore:
    def get(self, tid):
        return types.SimpleNamespace(status="Running")


async def t_agent_uses_search():
    reg = CapabilityRegistry()
    c, _ = _client(lambda q, n: ROWS)
    register_search_capability(reg, c)
    model = ScriptModel([
        _resp(tool_calls=[_tc("c1", "search_web", '{"query": "マリアナ海溝 深さ"}')]),
        _resp(content="一番深いのは約10,920メートルだよ。"),
    ])
    agent = TaskAgent(registry=reg, model_registry=model, store=AliveStore(), max_steps=5)
    out = await agent.run("海で一番深いところを調べて", "t1")
    tool_content = model.tool_msgs[0]["content"] if model.tool_msgs else ""
    return (out == "一番深いのは約10,920メートルだよ。"
            and "以下はWeb検索の結果" in tool_content  # agent はダイジェストだけを見る
            and "https://" not in tool_content)


async def t_agent_blocking_per_step_limit():
    # レッドチーム S1: 1 step の一括 tool_calls で重い能力を連打しても 1 件しか実行されず、
    # 残りは「一度に1つまで」の突き返し＝executor 長期占有を能力の種類によらず防ぐ。
    reg = CapabilityRegistry()
    calls = []

    def fn(q, n):
        calls.append(q)
        return ROWS
    c, _ = _client(fn)
    register_search_capability(reg, c)
    model = ScriptModel([
        _resp(tool_calls=[_tc("c1", "search_web", '{"query": "A"}'),
                          _tc("c2", "search_web", '{"query": "B"}'),
                          _tc("c3", "search_web", '{"query": "C"}')]),
        _resp(content="Aの結果だけ見た。"),
    ])
    agent = TaskAgent(registry=reg, model_registry=model, store=AliveStore(), max_steps=5)
    out = await agent.run("AとBとCを調べて", "t1")
    rejected = [m for m in model.tool_msgs if "一度に1つまで" in m["content"]]
    return out == "Aの結果だけ見た。" and calls == ["A"] and len(rejected) == 2


async def t_agent_blocking_reallowed_next_step():
    # step を跨げば blocking は再許可 / 同期能力（pc_status）は同一 step でもゲート対象外。
    reg = CapabilityRegistry()
    calls = []

    def fn(q, n):
        calls.append(q)
        return ROWS
    c, _ = _client(fn)
    register_search_capability(reg, c)
    model = ScriptModel([
        _resp(tool_calls=[_tc("c1", "search_web", '{"query": "A"}'),
                          _tc("c2", "pc_status"),
                          _tc("c3", "search_web", '{"query": "B"}')]),  # B は突き返し
        _resp(tool_calls=[_tc("c4", "search_web", '{"query": "B"}')]),  # 次 step で再許可
        _resp(content="done"),
    ])
    agent = TaskAgent(registry=reg, model_registry=model, store=AliveStore(), max_steps=5)
    out = await agent.run("AとBを調べてPCも見て", "t1")
    sync_ran = any("現在時刻" in m["content"] for m in model.tool_msgs)  # pc_status は実行された
    return out == "done" and calls == ["A", "B"] and sync_ran


async def main():
    check("整形: マーカ非衝突/信頼境界/ドメインのみ/正規化", await t_format_digest())
    check("整形: max_results 上限", await t_max_results_cap())
    check("意味論: 0件=正直な空振り（マーカなし）", await t_zero_hits_honest())
    check("意味論: ddgs No results 例外は0件扱い", await t_ddgs_noresults_exception_is_empty())
    check("意味論: 連続空振り→冷却→回復", await t_empty_streak_cooldown_and_recovery())
    check("意味論: インフラ失敗はマーカ", await t_infra_error_marker())
    check("意味論: 失敗種別の混合でも冷却発動(S3)", await t_mixed_failures_cooldown())
    check("意味論: timeoutもストリークに乗る", await t_timeout_counts_toward_streak())
    check("意味論: 既定閾値3は0件×2の再試行を許す", await t_streak_default_allows_two_retries())
    check("型崩れrowsでも落ちない", await t_malformed_rows_no_crash())
    check("最小間隔: 実検索のみ消費・残り時間を待つ", await t_min_interval_enforced())
    check("0件リトライ: 1回だけ再検索して吸収", await t_retry_empty_once())
    check("0件リトライ: 両方0件でもストリーク1回分", await t_retry_empty_both_streak_once())
    check("タイムアウト→マーカ", await t_timeout_marker())
    check("キャッシュ: 命中/fresh/TTL失効", await t_cache_fresh_ttl())
    check("空クエリ→マーカ", await t_empty_query_marker())
    check("クエリ200字cap", await t_query_length_cap())
    check("fresh型崩れ耐性(文字列false=命中/true=バイパス)", await t_fresh_string_false_not_bypass())
    check("登録: agent専用（応答LLMに出ない）", t_registration_scope())
    check("統合: TaskAgent が search_web→最終文", await t_agent_uses_search())
    check("統合: 重い能力は1step1件まで(S1)", await t_agent_blocking_per_step_limit())
    check("統合: 次stepで再許可/同期能力はゲート外", await t_agent_blocking_reallowed_next_step())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
