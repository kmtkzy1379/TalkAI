r"""**設定したモデルに本当に届いているか**を実 API で確かめ、かかった費用を出す。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\api_route_check.py

**実 API 課金あり**（ただし極小）。各役割に最短のプロンプトを1回ずつ投げ、
provider が返した `model` が要求と一致するかを見る。`.env` を書き換えるのではなく
`ModelRegistry.set_override` で役割ごとに差し替えるので、**本物の `.env` に触らない**。

`tools/settings_verify.py` は「設定が消費者まで届くか」までしか見られない
（`resolve()` が正しい ID を返すところまで）。実際にその ID で**呼ばれている**ことは、
一度は本物を叩かないと分からない。ここがその1回。

STT は VOICEVOX で作った音声を使う（音声素材の用意も無料）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.model_registry import ROLE_ENV, ModelRegistry  # noqa: E402
from eve.usage import UsageMeter, install, price_of  # noqa: E402

# 消費者ゼロの役は確かめない（呼ぶコードが無いので、通っても意味が無い）。
SKIP_ROLES = {"vlm_merge", "youtube"}
PROMPT = [{"role": "user", "content": "1+1は? 数字だけで答えて。"}]


def bare(model: str) -> str:
    return model.split("/", 1)[-1].lower()


async def check_roles(reg: ModelRegistry) -> list:
    rows = []
    for role in reg.roles():
        if role in SKIP_ROLES:
            rows.append((role, reg.resolve(role), "—", "skip（呼ぶコードが無い役）"))
            continue
        want = reg.resolve(role)
        try:
            t0 = time.monotonic()
            resp = await reg.complete(role, PROMPT, max_tokens=8)
            dt = time.monotonic() - t0
            got = getattr(resp, "model", "") or ""
            # provider は日付入りの正式名を返すことがある（gpt-4o → gpt-4o-2024-08-06）
            ok = bare(want) in got.lower() or got.lower() in bare(want)
            rows.append((role, want, got, f"{'OK' if ok else '不一致'} ({dt:.2f}s)"))
        except Exception as e:
            rows.append((role, want, "—", f"失敗: {type(e).__name__}: {str(e)[:60]}"))
    return rows


async def check_stt() -> str:
    """VOICEVOX で作った音声を実 STT に通す（設定した STT モデルが使われるか）。"""
    from eve.config import Config
    from eve.response.tts import VoicevoxTTS
    from eve.stt import make_stt

    tts = VoicevoxTTS()
    try:
        wav = await tts.generate("今日はいい天気ですね")
    finally:
        await tts.close()
    if not wav:
        return "VOICEVOX が動いていないので STT は未確認"
    import io
    import wave

    with wave.open(io.BytesIO(wav), "rb") as w:
        pcm = w.readframes(w.getnframes())
    stt = make_stt()
    text = await stt.transcribe(pcm)
    return (f"{type(stt).__name__} / model={getattr(stt, 'model', '?')} → 「{text}」"
            if text else f"{type(stt).__name__} は空を返した（設定 {Config.STT_MODEL}）")


async def main() -> int:
    meter = UsageMeter(Path("usage_route_check.jsonl"), tag="route_check")
    install(meter)

    print("実 API で「設定したモデルに届いているか」を確かめます。")
    print("各役割に最短プロンプトを1回ずつ。**本物の .env には触りません**。\n")
    print(f"{'役割':<18}{'要求したモデル':<30}{'返ってきた model':<34}結果")
    reg = ModelRegistry()
    rows = await check_roles(reg)
    for role, want, got, note in rows:
        print(f"{role:<18}{want:<30}{got:<34}{note}")

    print("\n[STT]", await check_stt())

    print("\n" + "=" * 62)
    print(meter.format_summary())
    print("\n価格が引けたか（litellm の内蔵表）:")
    for _role, want, _got, _n in rows:
        p = price_of(want)
        print(f"  {want:<30}"
              + ("引けた" if p else "⚠ 引けない（費用は集計に入りません）"))
    bad = [r for r in rows if r[3].startswith(("不一致", "失敗"))]
    print("\n" + ("✅ すべて設定どおりの宛先に届いています。" if not bad
                  else f"❌ {len(bad)} 件に問題があります（上を参照）。"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
