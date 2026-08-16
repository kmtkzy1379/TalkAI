r"""設定が**本当に効いているか**を、`.env` から消費者まで通しで確かめる（API 課金なし）。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\settings_verify.py

## 何を確かめるか（2段）
1. **`.env` → `Config`**: 別プロセスを `EVE_ENV_FILE` 付きで起こし、書いた値がそのまま
   `Config` に載るか。**本物の `.env` には一切触らない**。
2. **`Config` → 消費者**: 同じ子プロセスで `VoiceLoop()` を組み立て（**API は叩かない**）、
   値が実際に使う側のオブジェクトへ届いているかを見る。

段2が本命。「設定はあるが誰も読んでいない」は起動しても例外が出ず、実運用で
「設定したのに効かない」として初めて気づく種類の欠陥だから。

## なぜ値を全部いっぺんに変えるか
1件ずつ子プロセスを起こすと 57 回 × 十数秒かかる。設定同士は独立なので、**既定と違う
値を全部同時に入れて1回で見る**。機能フラグは ON のままにする（OFF にすると消費者が
そもそも生成されず、確かめたい配線が消える）。

## 確かめられないもの
クロージャの中に閉じ込められた値（`WS_RUN_*` / `WS_EXEC_ALLOW` など、能力登録時に
引数として渡され、あとから読み出す口が無いもの）は「Config どまり」と正直に報告する。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = Path(__file__).resolve().parent.parent

# `Config` に載らないキー。**`ModelRegistry.resolve()` が `os.getenv` を直に読む**設計なので、
# Config 経由で確かめようとすると「Config に無い」と誤検出する（ここが不在なのは正しい）。
NOT_ON_CONFIG = {
    "RESPONSE_MODEL", "DECIDE_MODEL", "TASK_MODEL", "VLM_LEAF_MODEL", "FEEDBACK_MODEL",
    "SUMMARIZE_MODEL", "SEARCH_SUMMARIZE_MODEL", "DELIVERY_CHECK_MODEL",
    "SELF_MODEL_MODEL",
}

# (env キー, 入れる値, 消費者の読み方 or None)。消費者は子プロセス内で eval される式。
# 値は「既定と違う」かつ「妥当」なものを選ぶ（範囲外だと別の理由で弾かれ、原因が濁る）。
CASES: tuple = (
    # --- 声（VOICEVOX） ---
    ("VV_SPEAKER", "3", "vl.tts.speaker"),
    ("VV_SPEED", "1.23", "vl.tts.speed"),
    ("VV_PITCH", "0.07", "vl.tts.pitch"),
    ("VV_INTONATION", "1.44", "vl.tts.intonation"),
    ("VOICEVOX_URL", "http://127.0.0.1:50099", "vl.tts.base_url"),
    # --- モデル（役割→ID の解決） ---
    ("RESPONSE_MODEL", "openai/gpt-4o", "vl.registry.resolve('response')"),
    ("DECIDE_MODEL", "openai/gpt-4o-mini", "vl.registry.resolve('speech_decide')"),
    ("TASK_MODEL", "openai/gpt-4o", "vl.registry.resolve('task')"),
    ("VLM_LEAF_MODEL", "gemini/gemini-2.5-pro", "vl.registry.resolve('vlm_leaf')"),
    ("FEEDBACK_MODEL", "openai/gpt-4o-mini", "vl.registry.resolve('feedback')"),
    ("SUMMARIZE_MODEL", "openai/gpt-4o", "vl.registry.resolve('summarize')"),
    ("SEARCH_SUMMARIZE_MODEL", "openai/gpt-4o", "vl.registry.resolve('search_summarize')"),
    ("DELIVERY_CHECK_MODEL", "openai/gpt-4o", "vl.registry.resolve('delivery_check')"),
    ("SELF_MODEL_MODEL", "openai/gpt-4o", "vl.registry.resolve('self_model')"),
    ("SELF_MODEL_ENABLED", "1", "vl.self_model is not None"),
    # --- 自律発話 ---
    ("SILENCE_THRESHOLD_SEC", "9.0", "vl.silence_monitor._threshold"),
    ("AUTOSPEECH_ENABLED", "1", None),          # run() の分岐（構築時には現れない）
    # --- 音声入力 / STT ---
    ("STT_MODEL", "gpt-4o-mini-transcribe", "vl.stt.model"),
    ("STT_BACKEND", "openai", "type(vl.stt).__name__"),
    ("VAD_THRESHOLD", "0.62", None),            # AudioInput は input.start() で作られる
    ("SILENCE_LIMIT", "0.8", None),
    ("PREROLL_SEC", "0.55", None),
    ("VAD_ONSET_SEC", "0.2", None),
    # --- 画面認識 ---
    ("VLM_MONITOR", "0", "vl.vlm_capture._monitor"),
    ("VLM_DOWNSCALE_MAX", "768", "vl.vlm_capture._downscale_max"),
    ("VLM_JPEG_QUALITY", "55", "vl.vlm_capture._jpeg_quality"),
    ("VLM_PHASH_THRESHOLD", "9", "vl.vlm_change_detector._threshold"),
    ("VLM_MIN_INTERVAL_SEC", "2.5", "vl.vlm_worker._min_interval"),
    ("VLM_MAX_FRAMES_PER_CALL", "2", "vl.vlm_worker._k"),
    # CaptureThread は run() で作られるので、loop 上で1回だけ起こして間隔を見る（1/fps 秒）
    ("VLM_TARGET_FPS", "1.5", "round(1.0 / vision_interval(vl), 4)"),
    ("VLM_OCR_DIFF_PCT", "33.0", "getattr(vl._build_text_gate(), '_threshold', None)"),
    ("VLM_OCR_INTERVAL_SEC", "4.5", "getattr(vl._build_text_gate(), '_interval', None)"),
    # --- 記憶 ---
    ("RECENT_TURN_COUNT", "9", "vl.cache.recent_count"),
    ("HISTORY_MAX_TURNS", "123", "vl.cache.max_turns"),
    ("RAG_TOP_K", "5", "vl.rag.top_k"),
    ("RAG_MAX_CHUNKS", "321", "vl.rag.max_chunks"),
    ("RAG_RANDOM_K", "4", "vl.speech_decider._k"),
    ("EMBED_BACKEND", "ruri", "type(vl.embedder).__name__"),
    # --- タスク / 検索 / 作業フォルダ ---
    ("TASK_MAX", "222", "vl.task_store.max_tasks"),
    ("TASK_ORPHAN_TIMEOUT_SEC", "1800.0", "vl.task_store.orphan_timeout_sec"),
    ("SEARCH_MAX_RESULTS", "7", "vl.search_client._max_results"),
    ("SEARCH_TIMEOUT_SEC", "33.0", "vl.search_client._timeout"),
    ("SEARCH_COOLDOWN_SEC", "45.0", "vl.search_client._cooldown_sec"),
    ("WS_QUOTA_GB", "64.0", "vl.workspace._quota / (1024**3)"),
    ("WS_WRITE_MAX_CHARS", "12345", "vl.workspace._max_write"),
    ("WS_READ_MAX_CHARS", "2222", "vl.workspace._max_read"),
    ("WS_MAX_FILES", "999", "vl.workspace._max_files"),
    ("TASK_AGENT_MAX_STEPS", "7", "vl.task_executor._agent._max_steps"),
    ("TASK_AGENT_TIMEOUT_SEC", "300.0", "vl.task_executor._agent._timeout"),
    # 能力登録のクロージャに閉じ込められた値は `inspect.getclosurevars` で覗く
    # allow は集合として保持される（順序に意味が無い）ので、書く側も並べ替えた形にする
    ("WS_EXEC_ALLOW", "git,python", "','.join(sorted(closure(vl, 'run_program')['allow_set']))"),
    ("WS_RUN_TIMEOUT_SEC", "45.0", "closure(vl, 'run_program')['timeout_sec']"),
    ("WS_RUN_MEM_MB", "1024", "closure(vl, 'run_program')['mem_mb']"),
    ("WS_MAX_OUTPUT_CHARS", "4321", "closure(vl, 'run_program')['max_output_chars']"),
    ("WS_RUN_MAX_PROCS", "16", "closure(vl, 'run_program')['max_procs']"),
    ("WS_RUN_GRACE_SEC", "7.0", "closure(vl, 'run_program')['grace_sec']"),
    ("WS_VIEWER_MAX_WINDOWS", "3", None),       # クロージャに残らない（登録時に消費される）
    # --- 機能フラグ（ON のまま・消費者が生成されることを見る） ---
    ("CALLFUNCTION_ENABLED", "1", "vl.dispatcher is not None"),
    ("TASK_ENABLED", "1", "vl.task_store is not None"),
    ("SEARCH_ENABLED", "1", "vl.search_client is not None"),
    ("WS_ENABLED", "1", "vl.workspace is not None"),
    ("WS_EXEC_ENABLED", "1", "'run_program' in vl.capabilities.names()"),
    ("WS_VIEWER_ENABLED", "1", "'show_on_screen' in vl.capabilities.names()"),
    ("VLM_ENABLED", "1", None),                 # capture は run() で起こす
    ("VLM_OCR_ENABLED", "1", "vl._build_text_gate() is not None"),
    # --- UI ---
    ("WS_VIEWER_MODE", "external", None),       # UI 側（app.py）が読む
    ("UI_ALWAYS_ON_TOP", "1", None),
    ("UI_HIDE_FROM_CAPTURE", "0", "__import__('eve.capture_guard', fromlist=['x']).enabled_by_env()"),
    ("TASKWIN_HIDE_FROM_CAPTURE", "0",
     "__import__('eve.capture_guard', fromlist=['x']).enabled_by_env('task')"),
    # 消費者は UI（task_window.py）が使う瞬間に Config を直読みする＝3段目で確かめる
    ("TASKWIN_HTML_ENABLED", "0", None),
    ("TASKWIN_PROTECT_CLICKS", "5", None),
    ("USAGE_LOG_ENABLED", "1", "vl.usage is not None"),
)

# 消費者が値をコピーせず、**使う瞬間に `Config` を直接読む**もの。
# この形は「Config が正しければ必ず効く」ので、読み出し口が無くても弱くはない。
# 確認は「その モジュールに `Config.<KEY>` の参照が実在するか」で行う（消費者ゼロを弾く）。
DIRECT_READERS = {
    "VAD_THRESHOLD": "eve/audio_input.py",
    "SILENCE_LIMIT": "eve/audio_input.py",
    "PREROLL_SEC": "eve/audio_input.py",
    "VAD_ONSET_SEC": "eve/audio_input.py",
    "AUTOSPEECH_ENABLED": "eve/voice_loop.py",
    "VLM_ENABLED": "eve/voice_loop.py",
    "WS_VIEWER_MODE": "eve/ui/app.py",
    "UI_ALWAYS_ON_TOP": "eve/ui/app.py",
    "TASKWIN_HTML_ENABLED": "eve/ui/task_window.py",
    "TASKWIN_PROTECT_CLICKS": "eve/ui/task_window.py",
    "WS_VIEWER_MAX_WINDOWS": "eve/voice_loop.py",
    "SAMPLE_RATE": "eve/audio_input.py",
}

CHILD = r'''
import asyncio, inspect, json, os, sys
sys.path.insert(0, r"{repo}")
os.chdir(r"{repo}")
from eve.config import Config

def closure(vl, name):
    """能力ハンドラのクロージャに閉じ込められた設定値を覗く（読み口が無いものの逃げ道）。"""
    cap = vl.capabilities._caps[name]
    h = cap.async_handler or cap.handler
    return {{k: v for k, v in inspect.getclosurevars(h).nonlocals.items()
             if not callable(v) and not hasattr(v, "__dict__")}}

def vision_interval(vl):
    """capture スレッドを loop 上で1回だけ起こして、実際の撮影間隔[秒]を読む。"""
    async def go():
        vl.start_vision()
        iv = vl.capture_thread._interval
        await vl.stop_vision()
        return iv
    return asyncio.run(go())

out = {{"config": {{}}, "consumer": {{}}, "errors": {{}}}}
cases = json.loads(r"""{cases}""")
for key, _val, expr in cases:
    out["config"][key] = repr(getattr(Config, key, "<Config に無い>"))
try:
    from eve.voice_loop import VoiceLoop
    vl = VoiceLoop()
except Exception as e:
    out["errors"]["__construct__"] = f"{{type(e).__name__}}: {{e}}"
    print(json.dumps(out, ensure_ascii=False)); sys.exit(0)
for key, _val, expr in cases:
    if not expr:
        continue
    try:
        out["consumer"][key] = repr(eval(expr))
    except Exception as e:
        out["errors"][key] = f"{{type(e).__name__}}: {{e}}"
print(json.dumps(out, ensure_ascii=False))
'''


def build_env_file(dst: Path) -> None:
    """本物の `.env` を土台に、確かめたい値だけ差し替えたものを作る（鍵はそのまま要る）。"""
    from eve import settings_store as store

    base = store.read_env(REPO / ".env")
    for key, val, _expr in CASES:
        base[key] = val
    lines = [f"{k}={v}" for k, v in base.items()]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def child_env(env_file: Path) -> dict:
    """`.env` のキーを落としてから `EVE_ENV_FILE` を指す（親の値が勝つのを防ぐ）。"""
    from eve import settings_store as store
    from eve.ui.restart import child_env as strip

    env = strip(dict(os.environ), env_keys=list(store.read_env(REPO / ".env")))
    for key, _v, _e in CASES:
        env.pop(key, None)
    env["EVE_ENV_FILE"] = str(env_file)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _num(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def matches(expected: str, got: str) -> bool:
    """`repr` 同士のゆるい比較（`'3'` と `3` と `3.0` を同じとみなす）。

    型名との照合は**大文字小文字を無視**する（`EMBED_BACKEND=ruri` の消費者は
    `RuriEmbedder`、`STT_BACKEND=openai` は `OpenAITranscribeStt`）。
    """
    g = got.strip().strip("'\"")
    e = expected.strip()
    if g == e or _num(e, g):
        return True
    if e in ("0", "1") and g in ("True", "False"):
        return (e == "1") == (g == "True")
    return e.lower() in g.lower()  # 型名やパスの部分一致


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="eve_verify_"))
    env_file = work / ".env"
    build_env_file(env_file)
    code = CHILD.format(repo=str(REPO), cases=json.dumps(CASES, ensure_ascii=False))
    script = work / "child.py"
    script.write_text(code, encoding="utf-8")

    print(f"子プロセスで検証中…（`.env` は {env_file}・本物には触りません）")
    r = subprocess.run([sys.executable, str(script)], env=child_env(env_file),
                       capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
    line = next((ln for ln in reversed((r.stdout or "").splitlines())
                 if ln.startswith("{")), "")
    if not line:
        print("子プロセスが結果を返しませんでした。")
        print((r.stdout or "")[-2000:])
        print((r.stderr or "")[-3000:])
        return 1
    data = json.loads(line)
    if "__construct__" in data["errors"]:
        print("VoiceLoop の構築に失敗:", data["errors"]["__construct__"])
        return 1

    ok_cfg = ok_use = 0
    n_cfg = 0
    fail: list = []
    only_config: list = []
    for key, val, expr in CASES:
        if key not in NOT_ON_CONFIG:
            n_cfg += 1
            got_cfg = data["config"].get(key, "<無>")
            if matches(val, got_cfg):
                ok_cfg += 1
            else:
                fail.append(f"[.env→Config] {key}: 書いた {val!r} / Config {got_cfg}")
        if not expr:
            only_config.append(key)
            continue
        if key in data["errors"]:
            fail.append(f"[Config→消費者] {key}: {data['errors'][key]}")
            continue
        got_use = data["consumer"].get(key, "<無>")
        if matches(val, got_use):
            ok_use += 1
        else:
            fail.append(f"[Config→消費者] {key}: 書いた {val!r} / 消費者 {got_use}")

    n_use = len([1 for _k, _v, e in CASES if e])
    print(f"\n{'=' * 62}")
    print(f"検証した設定: {len(CASES)} 件")
    print(f"1. `.env` → Config      : {ok_cfg}/{n_cfg}"
          f"（モデル {len(NOT_ON_CONFIG)} 件は ModelRegistry が os.getenv を直読みするため対象外）")
    print(f"2. Config → 消費者      : {ok_use}/{n_use}")
    print(f"   （消費者を直接見られない項目: {len(only_config)} 件）")
    if fail:
        print("\n❌ 一致しなかったもの:")
        for f in fail:
            print("  ", f)
    else:
        print("\n✅ 見られる範囲はすべて一致しました。")
    # 3段目: 値をコピーせず使用時に Config を直接読むもの（参照が実在するかで確かめる）
    ok_direct = 0
    unread: list = []
    for key, path in DIRECT_READERS.items():
        src = (REPO / path).read_text(encoding="utf-8", errors="ignore")
        if f"Config.{key}" in src:
            ok_direct += 1
        else:
            unread.append(f"{key}: {path} に `Config.{key}` の参照が無い（消費者ゼロの疑い）")
    print(f"3. Config を直読みする側 : {ok_direct}/{len(DIRECT_READERS)}"
          "（値をコピーしない＝Config が正しければ必ず効く形）")
    if unread:
        fail.extend(unread)
        print("\n❌ 参照が見つからない:")
        for u in unread:
            print("  ", u)

    left = [k for k in only_config if k not in DIRECT_READERS]
    if left:
        print("\n※ どの段でも直接は確かめられなかった項目:")
        print("  " + ", ".join(left))
    else:
        print("\n※ 全項目がいずれかの段で確認できました。")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
