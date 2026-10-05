"""Tier-1: ウォームアップの並列化と起動進捗（API 不要）。

**なぜテストするか**: 直列に戻っても症状は「起動が遅い」だけで、動作は正しく見える
（＝黙って腐る）。並列であることと、1つ失敗しても他が進むことを構造で固める。

実測の背景（2026-08-14・8ラン）: 直列だと warmup 区間が 15〜36秒。内訳は
STT 0.6s / Ruri ロード 5〜7s / 応答LLM + TTS 4〜8s。4つは互いに独立で、いちばん重い
Ruri は `run_in_executor` の中で動く＝**ループを塞がない**ので並列化が効く。
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


class _Fake:
    """`VoiceLoop.warmup` が触る依存だけを持つ器（本物は API 鍵とモデルが要る）。"""

    def __init__(self, delays: dict, fail: set = frozenset()) -> None:
        self.progress: list = []
        self.on_progress = lambda k, t: self.progress.append((k, t))
        self._delays, self._fail = delays, fail
        self.order: list = []

        async def mk(name):
            self.order.append(("start", name))
            await asyncio.sleep(self._delays.get(name, 0.0))
            if name in self._fail:
                raise RuntimeError(f"{name} が失敗")
            self.order.append(("end", name))

        self.stt = type("S", (), {"warmup": staticmethod(lambda: mk("stt"))})()
        self.rag = type("R", (), {"warmup": staticmethod(lambda: mk("rag"))})()
        self.registry = type("M", (), {
            "complete": staticmethod(lambda *a, **k: mk("llm"))})()
        # TTS は `warmup()` を持つ（VOICEVOX は「。」を1回合成 / Irodori はサーバを用意・S1）
        self.tts = type("T", (), {"warmup": staticmethod(lambda: mk("tts"))})()

    from eve.voice_loop import VoiceLoop  # 実装を借りる（コピーしない）
    warmup = VoiceLoop.warmup
    _progress = VoiceLoop._progress


def _stub_preimport():
    """重い SDK の事前 import をテストでは差し替える（計測にネットワーク/ディスクを混ぜない）。"""
    import eve.voice_loop as vl

    vl._preimport_sdks = lambda: None


def t_parallel() -> None:
    print("\n[1] 4つのウォームアップが並列に走る")
    _stub_preimport()
    f = _Fake({"stt": 0.05, "rag": 0.30, "llm": 0.10, "tts": 0.08})
    t0 = time.monotonic()
    asyncio.run(f.warmup())
    d = time.monotonic() - t0
    total = 0.05 + 0.30 + 0.10 + 0.08
    check("直列の合計より明確に速い（＝重なっている）", d < total * 0.7,
          f"{d:.2f}s（直列なら {total:.2f}s）")
    check("いちばん遅い1つぶんで済んでいる", d < 0.30 * 1.8, f"{d:.2f}s")
    starts = [n for (k, n) in f.order if k == "start"]
    first_end = next(i for i, (k, _n) in enumerate(f.order) if k == "end")
    check("**全部が始まってから最初の完了が来る**（直列なら start→end が交互）",
          first_end >= 4, f"{f.order[:6]}")
    check("4つとも起動された", sorted(starts) == ["llm", "rag", "stt", "tts"], f"{starts}")


def t_failure_isolated() -> None:
    print("\n[2] 1つ失敗しても他は進む（起動を止めない）")
    f = _Fake({"stt": 0.02, "rag": 0.05, "llm": 0.02, "tts": 0.02},
              fail={"llm", "stt"})
    asyncio.run(f.warmup())  # 例外が外へ出ないこと自体が検証
    ended = {n for (k, n) in f.order if k == "end"}
    check("失敗しなかったものは完了している", ended == {"rag", "tts"}, f"{ended}")
    texts = " ".join(t for _k, t in f.progress)
    check("失敗が進捗に出る", "失敗" in texts, texts[:120])
    check("成功も進捗に出る", "OK" in texts, texts[:120])


def t_progress() -> None:
    print("\n[3] 起動進捗の通知")
    f = _Fake({})
    asyncio.run(f.warmup())
    keys = [k for k, _t in f.progress]
    for k in ("stt", "rag", "llm", "tts", "warmup"):
        check(f"{k} の進捗が飛ぶ", k in keys, f"{keys}")
    check("最後に完了がまとまって出る", keys[-1] == "warmup", f"{keys[-1]}")

    # 既定（フック未設定）では何も起きない＝ヘッドレスの挙動は不変
    f2 = _Fake({})
    f2.on_progress = None
    asyncio.run(f2.warmup())
    check("on_progress 未設定でも落ちない（ヘッドレス）", True)

    # 通知が例外を投げても起動は止まらない
    f3 = _Fake({})
    f3.on_progress = lambda k, t: (_ for _ in ()).throw(RuntimeError("UI が死んでいる"))
    asyncio.run(f3.warmup())
    check("進捗フックが例外でもウォームアップは完走する",
          {n for (k, n) in f3.order if k == "end"} ==
          {"stt", "rag", "llm", "tts"})


def t_wiring() -> None:
    print("\n[4] 本体の配線")
    from eve.voice_loop import VoiceLoop

    src = inspect.getsource(VoiceLoop.warmup)
    check("gather を使っている（直列に戻していない）", "asyncio.gather" in src)
    check("4種すべてを含む",
          all(s in src for s in ("self.stt.warmup", "self.rag.warmup",
                                 "self.registry.complete", "self.tts.warmup")))
    # 並列化が持ち込んだ import 競合の再発防止（実起動で踏んだ）:
    # litellm と openai SDK が別スレッドから同時に pydantic を触ると
    # partially-initialized module になり、応答モデルと STT のウォームアップが両方失敗した。
    # docstring/コメントを除いた**実行される行だけ**で順序を見る（説明文の語を拾わない）。
    # `_preimport_sdks` はテストがスタブに差し替えているので、**ファイルの原文**を読む。
    code = "\n".join(ln for ln in src.splitlines()
                     if ln.strip() and not ln.strip().startswith("#") and "。" not in ln)
    check("重い SDK を gather の**前**に直列で import している",
          "_preimport_sdks" in code and code.index("_preimport_sdks") < code.index("gather"),
          "→ 並列 import で pydantic が壊れる")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "eve", "voice_loop.py")
    with open(path, encoding="utf-8") as f:
        body = f.read()
    pre = body[body.index("def _preimport_sdks"):][:900]
    for mod in ("pydantic", "openai", "litellm"):
        check(f"事前 import に {mod} が含まれる", f'"{mod}"' in pre)
    check("事前 import の失敗は握る（provider 未導入でも起動する）", "except Exception" in pre)

    run_src = inspect.getsource(VoiceLoop.run)
    for key in ("memory", "features", "mic", "ready"):
        check(f"run() が {key} の進捗を出す", f'"{key}"' in run_src)

    # 進捗フックは既定 None（ヘッドレスの挙動を変えない）
    init_src = inspect.getsource(VoiceLoop.__init__)
    check("on_progress の既定は None", "self.on_progress = None" in init_src)
    check("Ruri のウォームアップは executor 経由（ループを塞がない＝並列化が効く前提）",
          "run_in_executor" in inspect.getsource(
              __import__("eve.memory.embed.ruri", fromlist=["x"]).RuriEmbedder.embed_query))


if __name__ == "__main__":
    t_parallel()
    t_failure_isolated()
    t_progress()
    t_wiring()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
