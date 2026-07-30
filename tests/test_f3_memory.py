"""F3 短期記憶の決定論テスト（API不要）。

検証: ロケット鉛筆(100件)/直近取得/JSONL永続化往復/壁時計接地での省略マーカ描画/
注入セレクション（自律発話でユーザーから逸れない・省略マーカ）/C5（実際に喋った文だけ記憶）/
kind 別の記録（USER のみ user ターン）。実 LLM/TTS/音声は使わず stream_fn/tts_fn/play_fn を注入。

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_f3_memory.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.context_assembler import OMITTED_SPEAKER, ContextAssembler, Turn, messages_to_text  # noqa: E402
from eve.clock import Stamp  # noqa: E402
from eve.memory import ConversationCache  # noqa: E402
from eve.pipeline import (  # noqa: E402
    AudioPlayQueue,
    Stimulus,
    StimulusKind,
)
from eve.response import ResponseOrchestrator  # noqa: E402

_passed = 0
_failed = 0
_tmpdir = tempfile.mkdtemp(prefix="eve_f3_")
_counter = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name}")


def _tmp() -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"h{_counter}.jsonl")


async def _drain_until(pred, limit: int = 400) -> None:
    for _ in range(limit):
        await asyncio.sleep(0)
        if pred():
            return
    await asyncio.sleep(0.05)


# --- 純ロジック（同期） ---------------------------------------------------

def t_rocket_pencil() -> bool:
    c = ConversationCache(history_file=_tmp(), max_turns=100)
    for i in range(105):
        c.add_turn("user", f"m{i}")
    turns = c.recent(1000)
    return (
        len(turns) == 100
        and turns[0].text == "m5"   # 最古5件が押し出された
        and turns[-1].text == "m104"
    )


def t_recent() -> bool:
    c = ConversationCache(history_file=_tmp(), recent_count=6)
    for i in range(10):
        c.add_turn("eve", f"e{i}")
    r = c.recent(6)
    return len(r) == 6 and r[0].text == "e4" and r[-1].text == "e9"


def t_injection_normal_no_marker() -> bool:
    """通常の往復連続では tail をそのまま返し、省略マーカは出ない。"""
    c = ConversationCache(history_file=_tmp(), recent_count=6)
    for u, e in [("U1", "E1"), ("U2", "E2")]:
        c.add_turn("user", u)
        c.add_turn("eve", e)
    sel = c.recent_for_injection(6)
    return (
        len(sel) == 4
        and not any(t.speaker == OMITTED_SPEAKER for t in sel)
        and sel[0].text == "U1"
        and sel[-1].text == "E2"
    )


def t_injection_autonomous_anchor_marker() -> bool:
    """tail が自律発話(eve のみ)で埋まっても、最後の user 往復を保証し省略マーカを挟む。"""
    c = ConversationCache(history_file=_tmp(), recent_count=6)
    c.add_turn("user", "U1")
    c.add_turn("eve", "E1")
    for i in range(8):
        c.add_turn("eve", f"A{i}")  # ユーザー沈黙中の自律発話8件
    sel = c.recent_for_injection(6)
    # 期待: [U1, E1, 省略マーカ(2), A2, A3, A4, A5, A6, A7]
    markers = [t for t in sel if t.speaker == OMITTED_SPEAKER]
    return (
        sel[0].speaker == "user" and sel[0].text == "U1"
        and sel[1].speaker == "eve" and sel[1].text == "E1"
        and len(markers) == 1 and markers[0].text == "2"
        and sel[-1].text == "A7"  # tail の末尾は最新の自律発話
        and all(t.speaker == "eve" for t in sel[3:])  # アンカー往復の後は eve だけ
    )


def t_omission_marker_render() -> bool:
    """ContextAssembler が省略マーカを自然文で描画する（地続きと誤解させない）。"""
    now = Stamp.now()
    turns = [
        Turn("user", "りんご好き", now),
        Turn("eve", "うん", now),
        Turn(OMITTED_SPEAKER, "3", now),
        Turn("eve", "そういえば", now),
    ]
    msgs = ContextAssembler(system_prompt="S").assemble(
        user_text="今", recent_turns=turns, now=now
    )
    rendered = messages_to_text(msgs)
    return "中略" in rendered and "3件を省略" in rendered and "りんご好き" in rendered


# --- 永続化・配線・C5（非同期） -------------------------------------------

async def t_persistence_roundtrip() -> bool:
    path = _tmp()
    c = ConversationCache(history_file=path)
    await c.initialize()
    c.add_turn("user", "こんにちは")
    c.add_turn("eve", "やあ、元気？")
    await c.shutdown()  # 書き込みキューをドレイン

    # ファイルに ts 付きで書かれていること
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    c2 = ConversationCache(history_file=path)
    await c2.initialize()
    r = c2.recent(10)
    await c2.shutdown()
    return (
        '"ts"' in raw
        and len(r) == 2
        and r[0].speaker == "user" and r[0].text == "こんにちは"
        and r[1].speaker == "eve" and r[1].text == "やあ、元気？"
        and isinstance(r[0].stamp.iso, str) and r[0].stamp.iso
    )


async def t_continuity_wiring() -> bool:
    """実 ResponseOrchestrator+cache で2ターン。ターン2の組立にターン1が含まれる。"""
    async def play_fn(a):
        pass

    audio = AudioPlayQueue(play_fn=play_fn)
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    seen: list[str] = []

    async def stream_fn(messages):
        seen.append("\n".join(m["content"] for m in messages))  # 過去ターンは別 role メッセージ→全結合
        yield "うん。"

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn, conversation_cache=cache)
    worker = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "りんご好き？"))
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "みかんは？"))
    worker.cancel()
    turns = cache.recent(50)
    await cache.shutdown()
    # ターン2の user content に、ターン1の user 発話と eve 応答が直近会話として入る
    turn2_has_history = "りんご好き？" in seen[1] and "うん。" in seen[1]
    # 記録は user1, eve1, user2, eve2 の4件
    return (
        len(turns) == 4
        and [t.speaker for t in turns] == ["user", "eve", "user", "eve"]
        and turn2_has_history
    )


async def t_c5_normal_full() -> bool:
    """割り込み無し: eve ターン = 実発話全文。"""
    async def play_fn(a):
        pass

    audio = AudioPlayQueue(play_fn=play_fn)
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()

    async def stream_fn(messages):
        for s in ["あ。", "い。", "う。"]:
            yield s

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn, conversation_cache=cache)
    worker = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "x"))  # 内部で join まで待つ
    worker.cancel()
    eve = [t.text for t in cache.recent(50) if t.speaker == "eve"]
    await cache.shutdown()
    return len(eve) == 1 and eve[0] == "あ。い。う。"


async def t_c5_bargein_only_spoken() -> bool:
    """barge-in: 生成5文・再生は途中まで → 喋っていない後続文(文3..文5)は記憶に残さない。"""
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()
    played: list = []
    play_started = asyncio.Event()
    release = asyncio.Event()

    async def play_fn(a, should_stop=None):
        played.append(a)
        if len(played) == 2:
            play_started.set()
            await release.wait()  # 2文目で保持 → 決定論的に barge-in できる

    audio = AudioPlayQueue(play_fn=play_fn)

    async def stream_fn(messages):
        for s in ["文1。", "文2。", "文3。", "文4。", "文5。"]:
            yield s

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn, conversation_cache=cache)
    worker = asyncio.create_task(audio.play_worker())
    task = asyncio.create_task(orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "ねえ")))

    await play_started.wait()  # 文1再生済み・文2再生中で保持
    audio.interrupt()          # barge-in: 世代+1 → 未再生の文3..文5 を破棄
    task.cancel()              # 進行中応答をキャンセル（cancel ハンドラで実発話を記録）
    release.set()
    try:
        await task
    except asyncio.CancelledError:
        pass
    worker.cancel()

    eve = [t.text for t in cache.recent(50) if t.speaker == "eve"]
    rec = eve[-1] if eve else ""
    await cache.shutdown()
    # 文1は確実に再生済み・文3/文4/文5 は破棄され記憶に残らない（生成≠発話）
    return "文1" in rec and "文3" not in rec and "文4" not in rec and "文5" not in rec


async def t_kind_user_only() -> bool:
    """USER は user ターンを記録、autonomous は eve ターンのみ（user ターン無し）。"""
    async def play_fn(a):
        pass

    audio = AudioPlayQueue(play_fn=play_fn)
    cache = ConversationCache(history_file=_tmp())
    await cache.initialize()

    async def stream_fn(messages):
        yield "ふむ。"

    async def tts_fn(s):
        return f"[{s}]"

    orch = ResponseOrchestrator(audio, stream_fn, tts_fn, conversation_cache=cache)
    worker = asyncio.create_task(audio.play_worker())
    await orch.handle(Stimulus(StimulusKind.USER_UTTERANCE, "こん"))
    await orch.handle(Stimulus(StimulusKind.AUTONOMOUS_SPEECH, "…"))
    worker.cancel()
    turns = cache.recent(50)
    await cache.shutdown()
    users = [t for t in turns if t.speaker == "user"]
    eves = [t for t in turns if t.speaker == "eve"]
    return len(users) == 1 and users[0].text == "こん" and len(eves) == 2


# --- D1 案2: 復元（前セッション）分は注入しない ------------------------------
# 実機 2026-07-29: 起動直後の挨拶だけで5時間前の未応答依頼を実行した。
# 第1ラウンドのギャップ注記は実測 8/8 で無効。復元会話を注入しなければ 0/8。
# **B5 no-skip（前セッション末尾を RAG に取り込む）は turns_since 経由なので無傷**でなければならない。

def _prev_session_file() -> str:
    """前セッションの履歴を持つ jsonl を作る（末尾は未応答のユーザ依頼）。"""
    d = tempfile.mkdtemp(prefix="eve_d1b_")
    p = os.path.join(d, "h.jsonl")
    old = "2026-07-28T12:26:35.209300+00:00"
    with open(p, "w", encoding="utf-8") as f:
        for sp, tx, ts in (("user", "ネット検索して。", "2026-07-28T12:26:19+00:00"),
                           ("eve", "ごめん、できないんだ。", "2026-07-28T12:26:30+00:00"),
                           ("user", "じゃあPCの状態を教えてくれる?", old)):
            f.write(json.dumps({"ts": ts, "speaker": sp, "text": tx}, ensure_ascii=False) + "\n")
    return p


async def t_d1_restored_not_injected() -> bool:
    c = ConversationCache(history_file=_prev_session_file(), recent_count=6)
    await c.initialize()
    injected = c.recent_for_injection()
    await c.shutdown()
    # 復元は3件されているが、注入は0件（前セッションは「今の会話」ではない）
    return len(c.recent(10)) == 3 and injected == []


async def t_d1_live_injected() -> bool:
    c = ConversationCache(history_file=_prev_session_file(), recent_count=6)
    await c.initialize()
    c.add_turn("user", "おはよう。")
    c.add_turn("eve", "おはよう。")
    injected = c.recent_for_injection()
    await c.shutdown()
    texts = [t.text for t in injected]
    return texts == ["おはよう。", "おはよう。"] and "じゃあPCの状態を教えてくれる?" not in texts


async def t_d1_b5_catchup_intact() -> bool:
    """B5: 復元 tail は turns_since(watermark) から見える＝起動時 catch-up が動く。

    ここが壊れると前セッション末尾が永久に RAG に入らない（記憶喪失）。
    `_load` を止める実装だとこのテストが落ちる。
    """
    c = ConversationCache(history_file=_prev_session_file(), recent_count=6)
    await c.initialize()
    span = c.turns_since("2026-07-28T12:26:00+00:00")  # RAG 最新 ts 相当
    await c.shutdown()
    return len(span) == 3 and span[-1].text == "じゃあPCの状態を教えてくれる?"


async def t_d1_restored_last_iso() -> bool:
    c = ConversationCache(history_file=_prev_session_file(), recent_count=6)
    await c.initialize()
    iso = c.restored_last_iso()
    empty = ConversationCache(history_file=os.path.join(tempfile.mkdtemp(), "none.jsonl"))
    await empty.initialize()
    none_iso = empty.restored_last_iso()
    await c.shutdown()
    await empty.shutdown()
    return iso == "2026-07-28T12:26:35.209300+00:00" and none_iso is None


async def main() -> None:
    # 純ロジック
    check("ロケット鉛筆: 105→100・最古ドロップ", t_rocket_pencil())
    check("recent(6): 直近6件を順序どおり", t_recent())
    check("注入(通常): 往復連続は素通り・マーカ無し", t_injection_normal_no_marker())
    check("注入(自律): user往復を保証 + 省略マーカ", t_injection_autonomous_anchor_marker())
    check("省略マーカを自然文で描画", t_omission_marker_render())
    # 非同期
    check("JSONL 永続化往復（ts付き・復元）", await t_persistence_roundtrip())
    check("配線: ターン2の文脈にターン1が入る", await t_continuity_wiring())
    check("C5 正常: eve=実発話全文", await t_c5_normal_full())
    check("C5 barge-in: 喋ってない文は記憶に残さない", await t_c5_bargein_only_spoken())
    check("kind別: USERのみuserターン", await t_kind_user_only())
    # D1 案2: セッション境界
    check("⭐D1 復元分は注入しない", await t_d1_restored_not_injected())
    check("⭐D1 当セッション分は注入する", await t_d1_live_injected())
    check("⭐D1 B5: 復元分は turns_since で見える（catch-up 無傷）",
          await t_d1_b5_catchup_intact())
    check("D1 前セッション最終発話の iso を出せる", await t_d1_restored_last_iso())


asyncio.run(main())
print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
