#!/usr/bin/env python3
"""S0〜S4 の通し確認（本物の部品・**マイク/スピーカー/画面は使わない**・実 API 数円・時間上限つき）。

本番の `VoiceLoop` を組み立て、マイクの代わりに JVNV の音声を**実時間で**流す:
  Silero VAD → 区間の状態機械（本番と同じ `_SegmentTracker`）→ `MicSttInputSource` の区間フック
  → gpt-live-transcribe（ストリーミング）→ 応答LLM → Irodori（EVE が自動で起動）→ 再生（**無音で時間だけ模擬**）
フィラー（S3）と応答依存フィラー（S4）も本番どおり動く。1ターンごとに
「話し終わり（VAD の確定）→ フィラーの各語 → 応答の1文目が鳴り始める」までの時刻を出す。

安全のため:
- 画面認識・タスク・検索・作業フォルダ・自発発話・自己形成は切る（画面を送らない・勝手に話さない）
- 会話履歴・RAG・使用量は一時フォルダ（本番の記憶を汚さない）
- つなぎの間は fillers.yaml の値（`--gap` を指定した時だけ、一時ファイルの表でその値に差し替える）
- `--texts "文1|文2"` で、ユーザの発話を TTS で作って流せる（JVNV の代わり・笑いの確認用）。
  1つの発話の中に「⏸」を書くと、そこで `--pause` 秒（既定 0.8＝SILENCE_LIMIT より長い）黙ってから続きを話す
  （間を置いて話し続けた時の割り込み＝S7 の確認用。例 `--texts "昨日さ、駅前に行ったんだけど⏸そこで昔の友達に会ったんだよね。"`）
- 全体の上限 `--limit` 秒（既定 240）。終わったら Irodori サーバも止める

実行（EVE の venv・キーは eve-v2 の .env から読む）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\filler_dryrun.py --n 3
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = Path(r"C:/Users/user/Desktop/corpus/jvnv_eval")
# VAD が1区間で切る（途中に0.5秒超の間が無い）ものを選ぶ（`tools/stt_live_check.py` の実測で区間1）。
# 長い独白だと複数の発話に分かれ、話している途中に前の区間への応答が始まる既存の挙動が混ざる。
PICK = ["F2_happy_regular_37", "F2_anger_regular_35", "F2_disgust_regular_07", "M2_fear_free_09"]


def setup_env(tmp: Path, gap) -> None:
    """eve を import する前に環境変数で上書きする（.env は既存の値を上書きしない）。"""
    import yaml

    tpath = REPO / "fillers.yaml"
    if gap is not None:
        table = yaml.safe_load(tpath.read_text(encoding="utf-8"))
        table["timing"]["join_gap_sec"] = gap  # この確認のためだけ（本物の表は変えない）
        tpath = tmp / "fillers_dryrun.yaml"
        tpath.write_text(yaml.safe_dump(table, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.environ.update({
        "STT_BACKEND": "openai_live", "TTS_BACKEND": "irodori", "LAUGH_ENABLED": "1",
        "FILLER_ENABLED": "1", "FILLER_DEPENDENT_ENABLED": "1", "FILLER_TABLE": str(tpath),
        "VLM_ENABLED": "0", "CALLFUNCTION_ENABLED": "0", "TASK_ENABLED": "0", "SEARCH_ENABLED": "0",
        "WS_ENABLED": "0", "AUTOSPEECH_ENABLED": "0", "SELF_MODEL_ENABLED": "0",
        "HISTORY_FILE": str(tmp / "history.jsonl"), "RAG_FILE": str(tmp / "rag.jsonl"),
        "USAGE_LOG_ENABLED": "1", "USAGE_LOG_FILE": str(tmp / "usage.jsonl"),
    })


class SilentPlayer:
    """再生時間だけ待つ（音は出さない）。割り込まれても鳴っている節は最後まで（本番の再生器と同じ）。"""

    def __init__(self, timeline: list, t_ref: dict) -> None:
        self.timeline = timeline
        self.t_ref = t_ref

    async def play_fn(self, audio):
        if not audio:
            return
        with wave.open(io.BytesIO(audio), "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate())
        # 時計で終わりを決める（sleep(0.02) を積み上げると Windows のタイマー粒度で1.5倍ほど伸びる）
        end = time.monotonic() + dur
        while True:
            remain = end - time.monotonic()
            if remain <= 0:
                return
            await asyncio.sleep(min(0.02, remain))


class FakeMic:
    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()
        self.tracker = None  # 今流している発話の区間の状態機械（本番の AudioInput.in_speech と同じ判定にする）

    async def get_audio(self) -> bytes:
        return await self.q.get()

    def in_speech(self) -> bool:
        return self.tracker is not None and self.tracker.is_speaking

    def stop(self) -> None:
        pass


async def run(args) -> int:
    import numpy as np
    import soundfile as sf
    import torch

    from eve.audio_input import _SegmentTracker, _load_silero_vad
    from eve.cleanup import sweep_orphan_tts_servers
    from eve.filler import audio as fa
    from eve.logsetup import configure
    from eve.voice_loop import VoiceLoop

    configure()
    for noisy in ("httpx", "aiohttp", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("dryrun")
    if args.texts:
        picks = [{"id": f"text{i + 1}", "script": t, "text": t} for i, t in enumerate(args.texts.split("|")) if t]
    else:
        items = {it["id"]: it for it in json.loads((CORPUS / "items.json").read_text(encoding="utf-8"))}
        picks = [items[i] for i in PICK[: args.n] if i in items]

    vl = VoiceLoop()
    timeline: list = []
    t_ref = {"end": None, "start": None}
    player = SilentPlayer(timeline, t_ref)
    vl.audio._play_fn = player.play_fn
    if vl.filler is None:
        log.error("フィラーが用意できなかった（ログを確認）")
        return 1
    vl.filler._play = player.play_fn

    def rel() -> float:
        # そのターンの音声を流し始めた時刻から（話している途中の笑いも同じ時計で並べる）
        return time.monotonic() - t_ref["start"] if t_ref["start"] else float("nan")

    vl.audio.on_sentence_start = lambda text, meta: timeline.append((rel(), text))
    vl.audio.on_sentence_end = lambda text: None

    # VoiceLoop.run() の起動手順（マイクだけ起動しない）
    await vl.cache.initialize()
    await vl.rag.initialize()
    vl.prediction.watermark = vl.rag.latest_timestamp()
    vl.feedback_worker.start()
    await asyncio.to_thread(sweep_orphan_tts_servers)
    await vl.warmup()
    vl._tasks.append(asyncio.create_task(vl.audio.play_worker()))
    vl._tasks.append(asyncio.create_task(vl.runner.run()))
    vl.filler.start()
    mic = FakeMic()
    vl.input._ai = mic
    vl.input._task = asyncio.create_task(vl.input._consume())
    vad = _load_silero_vad()
    rate, chunk = 16000, 512

    def on_end(accepted: bool, speech_sec: float) -> None:
        t_ref["end"] = time.monotonic()
        timeline.append((rel(), f"〈話し終わり（VAD 確定）・声 {speech_sec:.2f}秒〉"))
        vl.input._segment_end(accepted, speech_sec)

    try:
        for it in picks:
            if it.get("text"):
                # ユーザの発話を TTS で作る → 16kHz に落とす。「⏸」の所では --pause 秒黙る
                from eve.stt.resample import StreamResampler

                pieces = [np.zeros(rate // 2, dtype=np.int16)]
                for k, part in enumerate(p for p in it["text"].split("⏸") if p.strip()):
                    wav = await vl.tts.generate(part)
                    x, sr0 = fa.read_wav(wav)
                    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
                    if k:
                        pieces.append(np.zeros(int(rate * args.pause), dtype=np.int16))
                    pieces.append(np.frombuffer(StreamResampler(sr0, rate).process(pcm), dtype="<i2").copy())
                d = np.concatenate(pieces)
            else:
                d, sr = sf.read(it["wav16"], dtype="int16")
                if d.ndim > 1:
                    d = d[:, 0]
            d = np.concatenate([d, np.zeros(rate, dtype=np.int16)])
            tr = _SegmentTracker(rate=rate, chunk=chunk, put=mic.q.put_nowait,
                                 on_speech_start=vl._barge_in, on_segment_start=vl.input._segment_start,
                                 on_segment_audio=vl.input._segment_audio, on_segment_end=on_end)
            mic.tracker = tr
            if hasattr(vad, "reset_states"):
                vad.reset_states()
            timeline.clear()
            log.info("▶ ユーザ（%s）: %s", it["id"], it["script"])
            t_ref["start"] = time.monotonic()
            t0 = time.perf_counter()
            for i in range(0, len(d) - chunk + 1, chunk):
                f32 = d[i:i + chunk].astype(np.float32) / 32768.0
                tr.step(d[i:i + chunk].tobytes(), vad(torch.from_numpy(f32), rate).item(),
                        float(np.abs(f32).max()), time.perf_counter())
                wait = t0 + (i + chunk) / rate - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
            # 応答が鳴り終わるまで待つ（上限 60秒）。次のターンで割り込まないように
            deadline = time.monotonic() + 60
            await asyncio.sleep(1.0)
            while time.monotonic() < deadline and (vl.runner.is_busy() or not vl.audio.is_quiet()
                                                   or vl.filler.active or vl.speech_state.stt_pending
                                                   or vl.queue.qsize()):
                await asyncio.sleep(0.1)
            print(f"\n── タイムライン（ユーザが話し始めてからの秒）: {it['script']}")
            for t, text in timeline:
                print(f"  {t:6.2f}  {text}")
            await asyncio.sleep(1.5)
    finally:
        await vl.stop()
        print("\n使用量:\n" + (vl.usage.format_summary() if vl.usage else "（記録なし）"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="ターン数（JVNV の音声を何件流すか）")
    ap.add_argument("--gap", type=float, default=None, help="この確認でだけ使う、つなぎの間[秒]（既定=fillers.yaml）")
    ap.add_argument("--texts", default="", help="ユーザの発話を TTS で作って流す（| 区切り）。無ければ JVNV")
    ap.add_argument("--pause", type=float, default=0.8,
                    help="--texts の「⏸」の所で黙る秒（SILENCE_LIMIT より長いと VAD が発話を2つに分ける）")
    ap.add_argument("--limit", type=float, default=240.0, help="全体の上限[秒]")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="eve_dryrun_"))
    setup_env(tmp, args.gap)
    sys.path.insert(0, str(REPO))

    async def _go() -> int:
        return await asyncio.wait_for(run(args), timeout=args.limit)

    try:
        return asyncio.run(_go())
    except asyncio.TimeoutError:
        print(f"上限 {args.limit:.0f}秒に達したので打ち切り")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
