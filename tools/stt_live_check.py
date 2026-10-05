#!/usr/bin/env python3
"""S0 の実測: 本番と同じ経路で「話し終わり → 確定文字」の時間を測る（実 API・数円）。

マイクの代わりに JVNV の 16kHz 音声を **実時間で** 32ms ずつ流し、本番と同じ
Silero VAD → 区間の状態機械（`eve/audio_input.py` の `_SegmentTracker`）→
`OpenAILiveStt`（start / feed / commit / transcribe）を通す。

測るもの（1発話ごと）:
- 最後の声 → VAD の終了確定（≒ SILENCE_LIMIT）
- VAD の終了確定（= commit） → 確定文字
- 誤り率（JVNV の台本比・記号を除いた文字単位）

実行（EVE の venv・キーは eve-v2 の .env から読む）:
  $env:PYTHONIOENCODING="utf-8"; $env:EVE_ENV_FILE="C:\\Users\\user\\Desktop\\eve-v2\\.env"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\stt_live_check.py --n 8
料金: $0.017/分（送った音声の長さ）。8件 ≒ 75秒 ≒ $0.02。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CORPUS = Path(r"C:/Users/user/Desktop/corpus/jvnv_eval")


def norm(t: str) -> str:
    return re.sub(r"[\s、。，．,.!！?？「」『』…ー〜~]", "", t)


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return float("nan")
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[len(h)] / len(r)


async def run_clip(stt, vad, path: Path, rate: int = 16000, chunk: int = 512) -> dict:
    import numpy as np
    import soundfile as sf
    import torch

    from eve.audio_input import _SegmentTracker

    d, sr = sf.read(str(path), dtype="int16")
    if d.ndim > 1:
        d = d[:, 0]
    assert sr == rate, f"{path} は {sr}Hz（16kHz を想定）"
    # 後ろに 1.0秒の無音を足す（VAD が終了を確定できるように。実際のマイクでも話し終わりの後は無音が続く）
    d = np.concatenate([d, np.zeros(rate, dtype=np.int16)])
    # 1クリップの中に 0.5秒を超える間があると VAD は複数の区間に分ける（本番と同じ）。
    # 区間ごとに「終了確定の時刻」を記録し、本番の消費側（`_consume`）と同じく順に受け取る。
    q: asyncio.Queue = asyncio.Queue()
    ends: list = []
    results: list = []
    last_voice = [None]

    def on_end(accepted: bool, speech_sec: float) -> None:
        (stt.commit if accepted else stt.clear)()
        if accepted:
            ends.append((time.perf_counter(), last_voice[0], speech_sec))

    async def consume() -> None:
        k = 0
        while True:
            pcm = await q.get()
            text = await stt.transcribe(pcm)
            t_end, t_voice, sp = ends[k]
            k += 1
            results.append({"text": text, "end_to_text": time.perf_counter() - t_end,
                            "voice_to_end": (t_end - t_voice) if t_voice else None, "speech_sec": sp})

    tr = _SegmentTracker(
        rate=rate, chunk=chunk, put=q.put_nowait,
        on_segment_start=stt.start, on_segment_audio=stt.feed, on_segment_end=on_end,
    )
    if hasattr(vad, "reset_states"):
        vad.reset_states()  # 前の音声の内部状態を持ち込まない
    cons = asyncio.create_task(consume())
    t0 = time.perf_counter()
    for i in range(0, len(d) - chunk + 1, chunk):
        data = d[i:i + chunk].tobytes()
        f32 = d[i:i + chunk].astype(np.float32) / 32768.0
        prob = vad(torch.from_numpy(f32), rate).item()
        if prob > 0.5:
            last_voice[0] = time.perf_counter()
        tr.step(data, prob, float(np.abs(f32).max()), time.perf_counter())
        # 実時間で流す（マイクと同じ速さ）
        wait = t0 + (i + chunk) / rate - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
    deadline = time.perf_counter() + 10.0
    while len(results) < len(ends) and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
    cons.cancel()
    if not results:
        return {"error": "VAD が区間を確定しなかった"}
    return {
        "text": "".join(r["text"] for r in results),
        "segments": results,
        # 応答が始まるのは最後の区間の確定文字が出た時（それまでの区間は話している最中に確定している）
        "voice_to_end": results[-1]["voice_to_end"],
        "end_to_text": results[-1]["end_to_text"],
    }


async def main_async(args) -> int:
    from eve.audio_input import _load_silero_vad
    from eve.stt.openai_live import OpenAILiveStt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    items = [it for it in json.loads((CORPUS / "items.json").read_text(encoding="utf-8"))
             if it.get("listen") and it.get("group") == "recog"][: args.n]
    tx = json.loads((CORPUS / "text.json").read_text(encoding="utf-8"))
    vad = _load_silero_vad()
    stt = OpenAILiveStt()
    t = time.perf_counter()
    await stt.warmup()
    print(f"接続 {time.perf_counter() - t:.2f}s")
    rows = []
    for it in items:
        r = await run_clip(stt, vad, Path(it["wav16"]))
        if "error" in r:
            print(f"{it['id']:<26} {r['error']}")
            continue
        r["cer"] = cer(it["script"], r["text"])
        r["cer_4o"] = cer(it["script"], tx.get(it["id"], {}).get("stt", ""))
        r["id"] = it["id"]
        rows.append(r)
        segs = " / ".join(f"{s['end_to_text']:.2f}s" for s in r["segments"])
        print(f"{it['id']:<26} 区間 {len(r['segments'])}（各 終了確定→確定文字 {segs}）"
              f"  誤り率 {r['cer']:.3f}（4o一括 {r['cer_4o']:.3f}）  {r['text']}")
        await asyncio.sleep(0.5)
    await stt.close()
    if rows:
        allseg = [s for r in rows for s in r["segments"]]
        e2t = sorted(s["end_to_text"] for s in allseg)
        v2e = sorted(s["voice_to_end"] for s in allseg if s["voice_to_end"] is not None)
        print(f"\n区間数 {len(allseg)}（クリップ {len(rows)}）")
        c = [r["cer"] for r in rows]
        c4 = [r["cer_4o"] for r in rows]
        print("\n── まとめ（n=%d）" % len(rows))
        print(f"終了確定→確定文字: 中央 {statistics.median(e2t):.2f}s / 最大 {e2t[-1]:.2f}s / 最小 {e2t[0]:.2f}s")
        print(f"最後の声→終了確定: 中央 {statistics.median(v2e):.2f}s（SILENCE_LIMIT と VAD の遅れ）")
        print(f"最後の声→確定文字: 中央 {statistics.median(v2e) + statistics.median(e2t):.2f}s")
        print(f"誤り率: 中央 {statistics.median(c):.3f}（同じ音声の gpt-4o-transcribe 一括 {statistics.median(c4):.3f}）")
        out = CORPUS / "stream_stt" / "eve_path_check.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"保存: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="使う JVNV の件数（items.json の recog・listen）")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
