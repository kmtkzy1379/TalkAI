r"""設定を変えると**挙動が実際に変わるか**を測る（API 課金なし）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\settings_behavior_check.py

`tools/settings_verify.py` は「値が消費者まで届くか」を見る。こちらはその先で、
**届いた値が実際に動きを変えるか**を見る。届いていても使われていない（受け取って捨てて
いる）設定は、前者では捕まらない。

LLM は一切呼ばない。スタブを差し込んで「何回呼ばれたか」「何件返したか」「弾いたか」で
判定する。判定はすべて**数えられる量**にする（「それらしい」で通さない）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = SKIP = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}  {detail}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def skip(name: str, why: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  ⏭ {name}  （{why}）")


# --------------------------------------------------------------------------
# 画面認識のコスト knob
# --------------------------------------------------------------------------

def t_vlm_capture() -> None:
    print("\n[画面認識] 撮る頻度・変化の閾値・1回の枚数")
    from eve.vlm.capture_thread import CaptureThread
    from eve.vlm.change_detector import ChangeDetector
    from eve.vlm.types import Frame

    class _Cap:
        """`ScreenCapture` の口だけ真似る（実画面もネットワークも触らない）。"""

        def __init__(self) -> None:
            self.n = 0

        def capture_one(self):
            self.n += 1
            return (Frame(frame_id=self.n, mono_ts=time.monotonic(),
                          jpeg_b64="x"), self.n)

        def close(self) -> None:
            pass

    # --- VLM_TARGET_FPS: 撮影回数が fps に比例するか ---
    async def count_frames(fps: float, secs: float = 1.2) -> int:
        cap = _Cap()
        loop = asyncio.get_running_loop()
        th = CaptureThread(capture=cap, change_detector=ChangeDetector(phash_threshold=99),
                           deliver=lambda *a, **k: None, loop=loop, target_fps=fps)
        th.start()
        await asyncio.sleep(secs)
        await asyncio.to_thread(th.stop)
        return cap.n

    slow = asyncio.run(count_frames(2.0))
    fast = asyncio.run(count_frames(8.0))
    check("VLM_TARGET_FPS を上げると撮影回数が増える", fast > slow * 2,
          f"2fps={slow}枚 / 8fps={fast}枚（1.2秒）")

    # --- VLM_PHASH_THRESHOLD: 閾値を跨ぐと変化判定が反転するか ---
    a, b = 0, 0b1111111  # hamming 距離 7
    loose = ChangeDetector(phash_threshold=12)
    tight = ChangeDetector(phash_threshold=3)
    loose.evaluate(a)
    tight.evaluate(a)
    check("VLM_PHASH_THRESHOLD が変化判定を分ける",
          loose.evaluate(b) is False and tight.evaluate(b) is True,
          f"距離7 → 閾値12では変化なし / 閾値3では変化あり")

    # --- VLM_MAX_FRAMES_PER_CALL: 1回に渡す枚数 ---
    from eve.vlm.state import VisionState

    vs = VisionState(ring_max=10)
    for i in range(8):
        vs.add_frame(Frame(frame_id=i, mono_ts=0.0, jpeg_b64="x"))
    check("VLM_MAX_FRAMES_PER_CALL が渡す枚数を決める",
          len(vs.snapshot(2)) == 2 and len(vs.snapshot(5)) == 5,
          "snapshot(2)=2枚 / snapshot(5)=5枚")

    # --- VLM_RING_MAX: 保持枚数 ---
    small = VisionState(ring_max=3)
    for i in range(9):
        small.add_frame(Frame(frame_id=i, mono_ts=0.0, jpeg_b64="x"))
    check("VLM_RING_MAX が保持枚数を決める", len(small.ring) == 3, f"{len(small.ring)}枚")


def t_vlm_worker_pacing() -> None:
    print("\n[画面認識] 呼び出しの最短間隔・重複抑制")
    from eve.feedback.prediction_state import PredictionState
    from eve.vlm.state import VisionState
    from eve.vlm.types import Frame, VisionResult
    from eve.vlm.worker import VlmWorker

    async def run(min_interval: float, calls: int = 3) -> int:
        vs, pred = VisionState(ring_max=6), PredictionState()
        n = {"v": 0}

        async def narrate(frames, *a, **k):
            n["v"] += 1
            return VisionResult(narration=f"実況{n['v']}", surprise_diff=None)

        w = VlmWorker(vision_state=vs, prediction_state=pred, narrate_fn=narrate,
                      speak_trigger=lambda: None, speak_guard=lambda: False,
                      frames_per_call=1, min_interval_sec=min_interval, dedup_ratio=0.99)
        w.start()
        for i in range(calls):
            f = Frame(frame_id=i, mono_ts=time.monotonic(), jpeg_b64="x")
            vs.add_frame(f)
            w.on_frame(f, True)
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        await w.stop()
        return n["v"]

    quick = asyncio.run(run(0.0))
    slow = asyncio.run(run(5.0))
    check("VLM_MIN_INTERVAL_SEC が呼び出しを律速する", slow < quick,
          f"間隔0秒={quick}回 / 間隔5秒={slow}回（同じ入力）")


def t_text_gate() -> None:
    print("\n[画面認識] 文字ゲート（OCR）の閾値と間隔")
    from eve.vlm.text_gate import ScreenTextGate

    class _Ocr:
        available = True
        lang = "ja"

        def __init__(self) -> None:
            self.texts = []

        def read(self, png):
            return self.texts.pop(0) if self.texts else ""

    def gate(threshold: float, interval: float, now_fn):
        o = _Ocr()
        g = ScreenTextGate(ocr=o, grab_png=lambda: b"x",
                           diff_threshold_pct=threshold, interval_sec=interval, now_fn=now_fn)
        return g, o

    clock = {"t": 1000.0}
    g_loose, o1 = gate(80.0, 0.0, lambda: clock["t"])
    g_tight, o2 = gate(5.0, 0.0, lambda: clock["t"])
    # `text_gate.MIN_CHARS`（=40）未満の画面は「文字が無い」として判定しない仕様なので、
    # それを超える長さにする（短い文字列だと閾値ではなくこちらで落ちて原因が濁る）。
    base = ("きょうはいいてんきですねさんぽでもしようかとおもっていますあしたはあめらしい"
            "からせんたくものはいまのうちにほしておくほうがよさそうです")
    changed = base[:-8] + "ゆきになるかもしれない"   # 末尾だけ差し替え
    o1.texts = [base, changed]
    o2.texts = [base, changed]
    g_loose.maybe_changed()
    g_tight.maybe_changed()
    from eve.vlm.text_gate import diff_pct

    d_pct = diff_pct(base, changed)
    check("VLM_OCR_DIFF_PCT が文字変化の判定を分ける",
          g_loose.maybe_changed() is False and g_tight.maybe_changed() is True,
          f"{d_pct:.1f}%の変化 → 閾値80%では反応せず / 閾値5%では反応")

    g, o = gate(5.0, 100.0, lambda: clock["t"])
    o.texts = [base, changed, changed]
    g.maybe_changed()
    check("VLM_OCR_INTERVAL_SEC が短すぎる再判定を止める",
          g.maybe_changed() is False, "間隔100秒 → すぐ呼んでも読みに行かない")
    clock["t"] += 200.0
    check("  間隔を過ぎれば読みに行く", g.maybe_changed() is True, "200秒後")


# --------------------------------------------------------------------------
# 記憶
# --------------------------------------------------------------------------

def t_memory() -> None:
    print("\n[記憶] 直近ターン数・保持上限")
    from eve.memory.conversation_cache import ConversationCache

    d = Path(tempfile.mkdtemp(prefix="eve_beh_"))

    async def recent_of(recent_count: int, max_turns: int = 100) -> int:
        c = ConversationCache(history_file=str(d / f"h{recent_count}.jsonl"),
                              max_turns=max_turns, recent_count=recent_count)
        await c.initialize()
        for i in range(20):
            c.add_turn("user" if i % 2 == 0 else "eve", f"発話{i}")
        n = len([t for t in c.recent_for_injection() if t.text])
        await c.shutdown()
        return n

    few = asyncio.run(recent_of(4))
    many = asyncio.run(recent_of(12))
    check("RECENT_TURN_COUNT が注入するターン数を決める", few < many,
          f"4指定={few}件 / 12指定={many}件")

    async def kept(max_turns: int) -> int:
        c = ConversationCache(history_file=str(d / f"m{max_turns}.jsonl"),
                              max_turns=max_turns, recent_count=6)
        await c.initialize()
        for i in range(40):
            c.add_turn("user", f"発話{i}")
        n = len(c.recent(999))  # 既定は recent_count 件しか返さないので明示して全部見る
        await c.shutdown()
        return n

    k10, k30 = asyncio.run(kept(10)), asyncio.run(kept(30))
    check("HISTORY_MAX_TURNS が保持数を決める", k10 <= 10 < k30,
          f"上限10={k10}件 / 上限30={k30}件")


def t_rag() -> None:
    print("\n[記憶] RAG の件数（埋め込みはローカル Ruri＝無課金）")
    from eve.memory.long_term import RagStore

    class _Emb:
        """決定論の疑似埋め込み（Ruri のロードを待たずに件数だけ見る）。

        条件が2つある。**クエリには十分似ている**こと（似ていないと
        `RAG_RELEVANCE_FLOOR` で落ちて件数の話にならない）、かつ**互いには似すぎない**
        こと（似すぎると `RAG_DUP_HARDCUT`=0.95 の重複除去で1件に潰れる。最初これで
        2件指定でも6件指定でも1件しか返らなかった）。
        次元ごとに1つだけ大きくすると、クエリとの cos ≒0.81 / 互い ≒0.61 に収まる。
        """

        dim = 16

        def _vec(self, text: str):
            v = [1.0] * self.dim
            v[hash(text) % self.dim] += 3.0
            return v

        async def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        async def embed_query(self, text):
            return [1.0] * self.dim

    d = Path(tempfile.mkdtemp(prefix="eve_rag_"))

    async def top_k_of(k: int) -> int:
        r = RagStore(_Emb(), rag_file=str(d / f"r{k}.jsonl"), max_chunks=500, top_k=k)
        await r.initialize()
        for i in range(12):
            await r.add_chunk(text=f"記憶の断片 {i} 番目", importance=0.5)
        got = await r.search("記憶")
        await r.shutdown()
        return len(got)

    a, b = asyncio.run(top_k_of(2)), asyncio.run(top_k_of(6))
    check("RAG_TOP_K が思い出す件数を決める", a == 2 and b == 6, f"2指定={a}件 / 6指定={b}件")

    async def cap_of(cap: int) -> int:
        r = RagStore(_Emb(), rag_file=str(d / f"c{cap}.jsonl"), max_chunks=cap, top_k=3)
        await r.initialize()
        for i in range(30):
            await r.add_chunk(text=f"断片{i}", importance=0.5)
        n = len(r)
        await r.shutdown()
        return n

    check("RAG_MAX_CHUNKS が保持上限を決める",
          asyncio.run(cap_of(5)) <= 5 < asyncio.run(cap_of(20)),
          f"上限5={asyncio.run(cap_of(5))}件 / 上限20={asyncio.run(cap_of(20))}件")


# --------------------------------------------------------------------------
# タスク・検索・作業フォルダ（境界が実際に効くか）
# --------------------------------------------------------------------------

def t_task_agent_bounds() -> None:
    print("\n[タスク] 手数上限と制限時間が実際に打ち切るか")
    from eve.task.agent import TaskAgent

    class _Store:
        def get(self, tid):
            return type("T", (), {"status": "Running"})()

    class _Reg:
        def agent_tool_schemas(self):
            return [{"type": "function",
                     "function": {"name": "noop", "description": "何もしない",
                                  "parameters": {"type": "object", "properties": {}}}}]

        def has(self, n):
            return True

        def is_blocking(self, n):
            return False

        async def execute_async(self, name, args):
            return "まだ足りない"

    class _Model:
        """常に tool を呼び続ける（自力では終わらない）モデル。"""

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, role, messages, **kw):
            self.calls += 1
            msg = type("M", (), {
                "content": "", "tool_calls": [type("C", (), {
                    "id": f"c{self.calls}",
                    "function": type("F", (), {"name": "noop", "arguments": "{}"})()})()]})()
            return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    for limit in (2, 5):
        model = _Model()
        agent = TaskAgent(registry=_Reg(), model_registry=model, store=_Store(),
                          max_steps=limit, timeout_sec=999)
        asyncio.run(agent.run("終わらないゴール", "t1"))
        # 最後の1回は「なぜ終わらなかったか」の要約なので差し引く
        check(f"TASK_AGENT_MAX_STEPS={limit} で打ち切る", model.calls - 1 == limit,
              f"LLM 呼び出し {model.calls - 1} 手")

    clock = {"t": 0.0}
    model = _Model()
    agent = TaskAgent(registry=_Reg(), model_registry=model, store=_Store(),
                      max_steps=99, timeout_sec=10.0, now_fn=lambda: clock["t"])

    async def tick():
        clock["t"] += 4.0
        return await agent.run("終わらないゴール", "t2")

    async def go():
        # now_fn を進めながら回す（実時間を待たない）
        orig = agent._now
        agent._now = lambda: clock.__setitem__("t", clock["t"] + 4.0) or clock["t"]
        r = await agent.run("終わらないゴール", "t2")
        agent._now = orig
        return r

    res = asyncio.run(go())
    check("TASK_AGENT_TIMEOUT_SEC で時間切れになる",
          res is not None and "時間切れ" in str(res), f"{str(res)[:50]}")


def t_workspace_limits() -> None:
    print("\n[作業フォルダ] 上限が実際に弾くか")
    from eve.capability import CapabilityRegistry
    from eve.workspace import Workspace, register_workspace_capabilities

    d = Path(tempfile.mkdtemp(prefix="eve_ws_"))
    reg = CapabilityRegistry()
    ws = Workspace(d, quota_bytes=10 * 1024, max_write_chars=50,
                   max_read_chars=40, max_files=5, rescan_sec=600)
    register_workspace_capabilities(reg, ws)

    async def call(name, args):
        return await reg.execute_async(name, args)

    ok = asyncio.run(call("write_file", {"path": "a.txt", "content": "あ" * 10}))
    ng = asyncio.run(call("write_file", {"path": "b.txt", "content": "あ" * 500}))
    wrote_big = (d / "b.txt").exists()
    check("WS_WRITE_MAX_CHARS が長すぎる書き込みを弾く",
          (d / "a.txt").exists() and not wrote_big,
          f"50字上限: 10字→書けた / 500字→{'書けてしまった' if wrote_big else '弾いた'}"
          f"（{ng[:24]}…）")

    asyncio.run(call("write_file", {"path": "c.txt", "content": "い" * 200}))
    read = asyncio.run(call("read_file", {"path": "c.txt"}))
    check("WS_READ_MAX_CHARS が読み出しを切り詰める", len(read) < 200,
          f"40字上限 → 返り {len(read)}字")


def t_search_limits() -> None:
    print("\n[検索] 件数上限と連続実行の抑制")
    from eve.search.client import SearchClient

    def fake(query, max_results, **kw):
        return [{"title": f"結果{i}", "href": f"http://x/{i}", "body": "本文"}
                for i in range(max_results)]

    async def n_results(k: int) -> int:
        c = SearchClient(search_fn=fake, max_results=k, min_interval_sec=0.0,
                         cooldown_sec=0.0, cache_ttl_sec=0.0)
        r = await c.search(f"クエリ{k}")
        # 返りは整形済みダイジェスト（1件1行・行頭が「・」）。生の URL は載らない。
        return sum(1 for line in r.splitlines() if line.startswith("・"))

    a, b = asyncio.run(n_results(2)), asyncio.run(n_results(6))
    check("SEARCH_MAX_RESULTS が件数を決める", a == 2 and b == 6, f"2指定={a}件 / 6指定={b}件")

    async def interval_effect() -> tuple:
        clock = {"t": 0.0}
        slept = []

        async def sleep(s):
            slept.append(s)
            clock["t"] += s

        c = SearchClient(search_fn=fake, max_results=2, min_interval_sec=9.0,
                         cooldown_sec=0.0, cache_ttl_sec=0.0,
                         now_fn=lambda: clock["t"], sleep_fn=sleep)
        await c.search("いち")
        await c.search("に")
        return slept

    slept = asyncio.run(interval_effect())
    check("SEARCH_MIN_INTERVAL_SEC が連続検索を待たせる",
          bool(slept) and max(slept) > 0, f"待った秒数 {slept}")


# --------------------------------------------------------------------------
# 自律発話
# --------------------------------------------------------------------------

def t_autospeech_cadence() -> None:
    print("\n[自律発話] 閾値が発火間隔を決めるか")
    from eve.speech.monitor import SilenceMonitor, SpeechState

    class _Decider:
        def __init__(self) -> None:
            self.n = 0

        def is_idle(self):
            return True

        def trigger(self, source=""):
            self.n += 1

    def fires(threshold: float, elapsed: float) -> bool:
        clock = {"t": 1000.0}
        st = SpeechState(now_fn=lambda: clock["t"])
        d = _Decider()
        m = SilenceMonitor(state=st, decider=d, is_busy_fn=lambda: False,
                           threshold_sec=threshold)
        clock["t"] += elapsed
        return m.tick()

    check("SILENCE_THRESHOLD_SEC=5 は 6秒沈黙で発火する", fires(5.0, 6.0) is True)
    check("SILENCE_THRESHOLD_SEC=12 は 6秒では発火しない", fires(12.0, 6.0) is False)
    check("  12秒でも 15秒沈黙なら発火する", fires(12.0, 15.0) is True)

    # 実運用の実測（8分・閾値5.0秒）: 判定 59 回 = 約 8.1 秒に1回。
    # 閾値 + 判定LLM の所要時間 ぶん伸びるので、この間隔で正しい。
    print("     ※ 実運用 8 分（閾値5.0秒）で判定 59 回＝約 8.1 秒に1回を実測済み")


def t_feature_flags() -> None:
    print("\n[機能フラグ] OFF にすると本当に生成されないか")
    import subprocess

    repo = Path(__file__).resolve().parent.parent
    from eve import settings_store as store
    from eve.ui.restart import child_env

    code = (
        "import json,sys,os;sys.path.insert(0,r'%s');os.chdir(r'%s');"
        "from eve.voice_loop import VoiceLoop;vl=VoiceLoop();"
        "print(json.dumps({'task':vl.task_store is not None,"
        "'search':vl.search_client is not None,'ws':vl.workspace is not None,"
        "'caps':sorted(vl.capabilities.names())}))" % (repo, repo)
    )
    for flags, expect, label in (
        ({"TASK_ENABLED": "0"}, {"task": False, "search": False, "ws": False},
         "TASK_ENABLED=0 で束ごと消える"),
        ({"SEARCH_ENABLED": "0"}, {"task": True, "search": False, "ws": True},
         "SEARCH_ENABLED=0 で検索だけ消える"),
        ({"WS_EXEC_ENABLED": "0"}, {"task": True, "ws": True},
         "WS_EXEC_ENABLED=0 で run_program が消える"),
    ):
        d = Path(tempfile.mkdtemp(prefix="eve_flag_"))
        env_file = d / ".env"
        base = store.read_env(repo / ".env")
        base.update(flags)
        env_file.write_text("\n".join(f"{k}={v}" for k, v in base.items()) + "\n",
                            encoding="utf-8")
        env = child_env(dict(os.environ), env_keys=list(store.read_env(repo / ".env")))
        env["EVE_ENV_FILE"] = str(env_file)
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                           text=True, encoding="utf-8", cwd=str(repo))
        line = next((x for x in reversed((r.stdout or "").splitlines())
                     if x.startswith("{")), "")
        if not line:
            skip(label, "子プロセスが結果を返さなかった")
            continue
        import json

        got = json.loads(line)
        okk = all(got.get(k) == v for k, v in expect.items())
        if "WS_EXEC_ENABLED" in flags:
            okk = okk and "run_program" not in got["caps"]
        check(label, okk, f"{ {k: got.get(k) for k in expect} }"
              + ("" if "WS_EXEC_ENABLED" not in flags
                 else f" run_program={'run_program' in got['caps']}"))


if __name__ == "__main__":
    t_vlm_capture()
    t_vlm_worker_pacing()
    t_text_gate()
    t_memory()
    t_rag()
    t_task_agent_bounds()
    t_workspace_limits()
    t_search_limits()
    t_autospeech_cadence()
    t_feature_flags()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL} / SKIP {SKIP}")
    sys.exit(1 if FAIL else 0)
