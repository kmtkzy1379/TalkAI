#!/usr/bin/env python3
"""ストリーミング文字起こし（gpt-live-transcribe）の実測（eve-v2 の .venv で実行・aiohttp を使う）。

目的（ユーザ裁定 2026-09-25「笑いの出し方」）: ユーザの発話中に「。」ごとに Jev へ笑い判定を送れるか。
測るもの:
  - 日本語の精度（gpt-4o-transcribe の確定結果・JVNV の台本と比べる）
  - 途中の文字に「。」が付くか、付くならいつ届くか
  - 文字の遅れ（音声を流した位置 → その文字が届いた時刻）
  - 話し終わり（最後の音声を送った時刻）→ 最後の文字が届くまで
音声は実時間で流す（100ms ずつ・24kHz PCM16）。発話の終わりで commit（公式: クライアント側 VAD で commit）。

料金: $0.017/分（公式モデルページ）。出力は corpus/jvnv_eval/stream_stt/（リポジトリ外）。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jvnv_eval import OUT, load_env, rd  # noqa: E402

URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SR = 24000
CHUNK = 0.1


def load24k(path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").copy()


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


async def run_one(key, pcm, turns, delay, prompt=None):
    """pcm を実時間で流す。turns=[(start,end)秒] の end で commit。届いた文字と時刻を返す。"""
    import aiohttp
    events = []  # (音声を流し始めてからの秒, 種類, 文字)
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(URL, headers={"Authorization": f"Bearer {key}"}, heartbeat=20) as ws:
            tr = {"model": "gpt-live-transcribe", "languages": ["ja"], "delay": delay}
            if prompt:
                tr["prompt"] = prompt
            await ws.send_json({"type": "session.update", "session": {
                "type": "transcription",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": SR},
                                    "transcription": tr, "turn_detection": None}}}})
            t0 = None
            done = asyncio.Event()
            n_commit = len(turns)
            n_completed = 0

            async def reader():
                nonlocal n_completed
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    e = json.loads(msg.data)
                    now = time.perf_counter() - t0 if t0 else 0.0
                    ty = e.get("type", "")
                    if ty.endswith("input_audio_transcription.delta"):
                        events.append((now, "delta", e.get("delta", ""), e.get("item_id")))
                    elif ty.endswith("input_audio_transcription.completed"):
                        events.append((now, "done", e.get("transcript", ""), e.get("item_id")))
                        n_completed += 1
                        if n_completed >= n_commit:
                            done.set()
                    elif ty == "error":
                        events.append((now, "error", json.dumps(e.get("error"), ensure_ascii=False), None))
                        print("  error:", e.get("error"))
                        done.set()

            rt = asyncio.create_task(reader())
            await asyncio.sleep(0.5)  # session.update の反映待ち
            t0 = time.perf_counter()
            step = int(SR * CHUNK)
            ends = [int(e * SR) for _, e in turns]
            for i in range(0, len(pcm), step):
                await ws.send_json({"type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(pcm[i:i + step].tobytes()).decode()})
                pos = i + step
                if any(i < x <= pos for x in ends):
                    await ws.send_json({"type": "input_audio_buffer.commit"})
                    events.append((time.perf_counter() - t0, "commit", "", None))
                # 実時間で流す
                wait = t0 + pos / SR - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
            try:
                await asyncio.wait_for(done.wait(), timeout=20)
            except asyncio.TimeoutError:
                print("  timeout: completed が揃わなかった")
            rt.cancel()
    return events


def summarize(events, audio_end):
    deltas = [e for e in events if e[1] == "delta"]
    text = "".join(e[2] for e in deltas)
    final = "".join(e[2] for e in events if e[1] == "done")
    periods = []  # 「。」が届いた時刻
    acc = ""
    for t, _, d, _ in deltas:
        for ch in d:
            acc += ch
            if ch in "。？！?!":
                periods.append(round(t, 2))
    first = deltas[0][0] if deltas else None
    last = deltas[-1][0] if deltas else None
    done_t = max((e[0] for e in events if e[1] == "done"), default=None)
    return {"deltas": len(deltas), "stream_text": text, "final": final, "periods_at": periods,
            "first_delta": first, "last_delta": last, "done": done_t,
            "last_after_end": (last - audio_end) if last is not None else None,
            "done_after_end": (done_t - audio_end) if done_t is not None else None}


async def main_async(args):
    env = load_env()
    key = env["OPENAI_API_KEY"]
    outdir = OUT / "stream_stt"
    outdir.mkdir(parents=True, exist_ok=True)
    tx = rd("text.json")
    items = [it for it in rd("items.json") if it["listen"] and it["group"] == "recog"]
    res = {}
    L = []
    for delay in args.delays:
        for it in items:
            pcm = load24k(it["src"])
            dur = len(pcm) / SR
            ev = await run_one(key, pcm, [(0, dur)], delay)
            s = summarize(ev, dur)
            s.update(cer_script=cer(it["script"], s["final"] or s["stream_text"]),
                     cer_4o=cer(it["script"], tx[it["id"]]["stt"]), dur=dur)
            res[f"{delay}/{it['id']}"] = {**s, "events": ev}
            print(f"[{delay}] {it['id']:<24} {dur:4.1f}s  最初の文字 {s['first_delta'] or 0:.2f}s"
                  f"  最後の文字 話し終わり+{s['last_after_end'] or 0:.2f}s  確定 +{s['done_after_end'] or 0:.2f}s"
                  f"  「。」{s['periods_at']}  誤り率 {s['cer_script']:.2f}（4o {s['cer_4o']:.2f}）")
            (outdir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.clip:
        from youtube_clip import vad_segments
        a16 = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.clip), "-ac", "1", "-ar",
                                            "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout,
                            dtype="float32")
        segs = [(s / 16000, e / 16000) for s, e in vad_segments(a16)]
        pcm = load24k(args.clip)
        ev = await run_one(key, pcm, segs, args.delays[0])
        # 発話区間ごとの「区間の終わり → 最後の delta / completed」
        by = {}
        for t, k, d, iid in ev:
            if iid:
                by.setdefault(iid, []).append((t, k, d))
        commits = [t for t, k, _, _ in ev if k == "commit"]
        rows = []
        for idx, (iid, es) in enumerate(sorted(by.items(), key=lambda x: min(t for t, _, _ in x[1]))):
            end = segs[idx][1] if idx < len(segs) else None
            ds = [t for t, k, _ in es if k == "delta"]
            dn = [t for t, k, _ in es if k == "done"]
            txt = "".join(d for _, k, d in es if k == "done")
            rows.append({"start": segs[idx][0] if idx < len(segs) else None, "end": end,
                         "first_delta": ds[0] if ds else None,
                         "last_after_end": (ds[-1] - end) if ds and end else None,
                         "done_after_end": (dn[0] - end) if dn and end else None, "text": txt,
                         "periods": [round(t, 2) for t, k, d in es if k == "delta" and re.search("[。？！?!]", d)]})
        (outdir / "clip.json").write_text(json.dumps({"segments": rows, "events": ev}, ensure_ascii=False,
                                                     indent=1), encoding="utf-8")
        la = sorted(r["last_after_end"] for r in rows if r["last_after_end"] is not None)
        dn = sorted(r["done_after_end"] for r in rows if r["done_after_end"] is not None)
        print(f"[clip] 区間 {len(segs)} / 文字が届いた区間 {len(rows)}  "
              f"区間の終わり→最後の文字 中央値 {la[len(la)//2] if la else float('nan'):.2f}s  "
              f"→確定 中央値 {dn[len(dn)//2] if dn else float('nan'):.2f}s  "
              f"「。」を含む区間 {sum(1 for r in rows if r['periods'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", nargs="+", default=["low", "minimal"])
    ap.add_argument("--clip", type=Path, help="実際の雑談（発話区間ごとに commit して流す）")
    asyncio.run(main_async(ap.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
