#!/usr/bin/env python3
"""Irodori-TTS の常駐サーバ（EVE の応答・フィラー用 TTS・S1）。

**Irodori の venv（GPU 版 torch・Py3.10）で動かす**。EVE（CPU 版 torch・Py3.13）とは別プロセス。
EVE が自動で起動する（`eve/response/tts_irodori.py` の `IrodoriServer`）。手で起動しておけば EVE はそれを使う。

API（127.0.0.1 のみ）:
  GET  /health      → {"ok", "model", "voice", "sample_rate", "steps", "seed", "error", "pid"}
                        モデルのロードとウォームアップが済むまで ok=false
  POST /synthesize  {"text": str, "seconds": float|null, "seed": int|null, "num_steps": int|null,
                     "match_plain": bool}
                    → audio/wav（mono PCM16・コーデックのサンプルレート）
                      ヘッダ X-Synth-Ms = 合成にかかった時間（順番待ちを含まない）
                      match_plain=true: 文に感情の絵文字が付いていても、**絵文字を除いた文と同じ長さ**で合成する
                      （ユーザ 2026-09-26「ゆっくりしゃべられると演技臭いしテンポが悪くなるから速度は揃えて感情載せる」）。
                      長さは Irodori の長さの予測だけを先に走らせて決める（合成を2回しない＝遅れない）

- 参照音声（話者）は起動時に1回だけ latent にしてファイルにキャッシュする（毎回の再エンコードを省く）。
  **no_ref は使わない**（seed ごとに話者が変わる・実測）
- 合成は1本ずつ（Irodori 側でも `_infer_lock` で直列化される）。HTTP は受け付け、処理は順番待ち
- 目印 `--eve-irodori-server`: EVE の起動時の掃除が取り残しを探すため（`eve/cleanup.py`）
- `--owner-pid`: 起動した EVE のプロセス ID。EVE が消えたら自分も終わる
  （Job Object で一緒に落ちるはずだが、それが効かなかった時の保険）

手で起動する例（Irodori のフォルダで）:
  .venv\\Scripts\\python.exe <TalkAI-live>\\tools\\irodori_server.py --voice-ref <wav> --port 50030

（`from __future__ import annotations` は付けない: FastAPI が関数内で定義したリクエスト型を
  文字列の注釈から引けず、本文をクエリ引数と取り違えるため。Py3.10 なので `X | None` はそのまま書ける）
"""
import argparse
import asyncio
import hashlib
import io
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IRODORI = REPO.parent / "irodori-bench" / "Irodori-TTS"
MARKER = "--eve-irodori-server"

logger = logging.getLogger("irodori_server")


class _State:
    def __init__(self) -> None:
        self.ok = False
        self.error: str | None = None
        self.runtime = None
        self.latent_path: str | None = None
        self.sample_rate = 0


def _watch_owner(pid: int) -> None:
    """起動した EVE が終わったら自分も終わる（Windows はプロセスのハンドルを待つ＝pid の再利用に強い）。"""
    if pid <= 0:
        return
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # argtypes / restype を必ず書く（64bit のハンドルが切り捨てられて黙って壊れるため）
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.OpenProcess(synchronize, False, pid)
        if not h:
            logger.warning("起動元（pid=%d）が既に無い。終了する", pid)
            os._exit(0)
        k32.WaitForSingleObject(h, 0xFFFFFFFF)
        logger.info("起動元（pid=%d）が終わったので終了する", pid)
        os._exit(0)
    while True:  # Windows 以外（開発用）
        try:
            os.kill(pid, 0)
        except OSError:
            os._exit(0)
        time.sleep(2.0)


def _wav_bytes(audio, sample_rate: int) -> bytes:
    import numpy as np

    a = audio.detach().to("cpu").float().numpy().reshape(-1)
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def _make_latent(runtime, ref_wav: Path, cache_dir: Path, checkpoint: str) -> str:
    """参照音声を latent にして保存する（`inference_runtime._load_reference_latent` の wav 経路と同じ処理）。"""
    import torch
    from irodori_tts.inference_runtime import SamplingRequest, _load_audio

    data = ref_wav.read_bytes()
    key = hashlib.sha1(data + checkpoint.encode("utf-8")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"ref_{key}.pt"
    if out.exists():
        return str(out)
    req = SamplingRequest(text="-")  # 既定の正規化値（ref_normalize_db / ref_ensure_max）を借りる
    wav, sr = _load_audio(str(ref_wav))
    max_ref = float(runtime.default_max_ref_seconds)
    if max_ref > 0:
        n = max(1, int(max_ref * float(sr)))
        if wav.shape[1] > n:
            wav = wav[:, :n]
    piece = runtime.codec.encode_waveform(
        wav.unsqueeze(0), sample_rate=int(sr),
        normalize_db=req.ref_normalize_db, ensure_max=bool(req.ref_ensure_max)).cpu()
    torch.save(piece, str(out))
    return str(out)


def _emoji_pattern():
    """Irodori の絵文字注釈（`irodori_tts.duration.ALLOWED_ANNOTATION_EMOJIS`）に当たる正規表現。"""
    import re

    from irodori_tts.duration import ALLOWED_ANNOTATION_EMOJIS

    return re.compile("|".join(sorted((re.escape(x) for x in ALLOWED_ANNOTATION_EMOJIS), key=len, reverse=True)))


def predict_seconds(runtime, text: str, latent_path: str) -> float:
    """文の長さの予測だけ（合成しない）。

    `InferenceRuntime.synthesize` の前段（正規化 → tokenize → 話者の条件 → 長さの予測）と同じ手順で、
    Irodori の本体は書き換えない（内部の関数を synthesize と同じ引数で呼ぶだけ）。seconds を渡さずに合成した時と
    同じ長さになる（`tools/irodori_check.py --match` で確かめる）。Irodori の更新で内部が変わったら例外になる
    ＝呼び出し側は長さをそろえずに合成する。
    """
    import math

    import torch
    from irodori_tts.duration import build_duration_features
    from irodori_tts.inference_runtime import SamplingRequest
    from irodori_tts.text_normalization import normalize_text

    rt = runtime
    req = SamplingRequest(text=text, ref_latents=[latent_path])
    norm = normalize_text(text).strip()
    if not norm:
        raise ValueError("text became empty after normalization.")
    text_max_len = int(rt.default_text_max_len)
    dev = rt.model_device
    with rt._infer_lock, torch.inference_mode():
        text_ids, text_mask = rt.tokenizer.batch_encode([norm], max_length=text_max_len)
        text_ids, text_mask = text_ids.to(dev), text_mask.to(dev)
        caption_ids = caption_mask = None
        if rt.model_cfg.use_caption_condition:
            caption_ids, caption_mask = rt.caption_tokenizer.batch_encode(
                [""], max_length=int(rt.default_caption_max_len))
            caption_mask.zero_()
            caption_ids, caption_mask = caption_ids.to(dev), caption_mask.to(dev)
        spk_state, spk_mask = rt._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
        if spk_state is None:
            ref_latent, ref_mask = rt._load_reference_latent(req=req, batch_size=1, messages=[])
        else:
            ref_latent, ref_mask = None, None
        has_spk = torch.zeros((1,), dtype=torch.bool, device=dev)
        if spk_mask is not None:
            has_spk = spk_mask.any(dim=1)
        elif rt.model_cfg.use_speaker_condition_resolved and ref_mask is not None:
            has_spk = ref_mask.any(dim=1)
        feats = build_duration_features([norm], token_counts=text_mask.sum(dim=1),
                                        max_text_len=text_max_len, has_speaker=has_spk).to(dev)
        ts, tm, ss, sm, cs, cm = rt.model.encode_conditions(
            text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent, ref_mask=ref_mask,
            caption_input_ids=caption_ids, caption_mask=caption_mask,
            speaker_state_override=spk_state, speaker_mask_override=spk_mask,
            speaker_uncond_mode=req.speaker_uncond_mode)
        log_frames = rt.model.predict_duration_log_frames(
            text_state=ts, text_mask=tm, speaker_state=ss, speaker_mask=sm, caption_state=cs, caption_mask=cm,
            duration_features=feats, has_speaker=has_spk,
            has_caption=(torch.full((1,), False, dtype=torch.bool, device=dev)
                         if rt.model_cfg.use_caption_condition else None))
        frames = torch.expm1(log_frames).float().mean().item()
    hop = int(rt.codec.model.hop_length)
    sr = int(rt.codec.sample_rate)
    lo = max(1, math.ceil(float(req.min_seconds) * sr / hop))
    hi = max(1, math.floor(float(req.max_seconds) * sr / hop))
    steps = max(lo, min(hi, int(round(frames * float(req.duration_scale)))))
    return steps * hop / float(sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50030)
    ap.add_argument("--voice-ref", type=Path, required=True, help="話者の参照音声（wav）")
    ap.add_argument("--irodori-repo", type=Path, default=DEFAULT_IRODORI)
    ap.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--codec-device", default="cuda")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-schedule", default="sway")
    ap.add_argument("--sway-coeff", type=float, default=-1.0)
    ap.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "eve_irodori")
    # 注意機構の実装。Irodori は cuDNN を最優先にしているが、cuDNN の SDPA は**系列長ごとに実行計画を
    # 組み立てる**ため、初めての長さの音声だけ拡散サンプリングが約1秒遅くなる（会話の応答は毎回新しい
    # 長さ＝ほぼ毎文これを踏む）。実測 2026-09-25（RTX 4090 Laptop・steps 6・新しい文6件）:
    #   cudnn     新しい文 1047〜1256ms / 同じ文の2回目 261〜328ms
    #   efficient 新しい文  258〜 390ms / 同じ文の2回目 247〜265ms
    # 音は同じ（同じ文・seed でスペクトル相関 0.996〜1.000。波形の細部だけ bf16 の誤差で揺れる）。
    ap.add_argument("--attention", choices=("efficient", "cudnn"), default="efficient")
    ap.add_argument("--owner-pid", type=int, default=0)
    ap.add_argument(MARKER, action="store_true", help="EVE が起動した印（掃除用の目印）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.owner_pid:
        threading.Thread(target=_watch_owner, args=(args.owner_pid,), daemon=True,
                         name="owner-watch").start()
    if str(args.irodori_repo) not in sys.path:
        sys.path.insert(0, str(args.irodori_repo))
    if args.attention == "efficient":
        # Irodori の本体は書き換えない。マスク付き注意の優先順位（モジュールの変数）だけ差し替える
        # （`masked_sdpa` は呼ばれるたびにこのリストを読む）。
        from torch.nn.attention import SDPBackend

        import irodori_tts.attention as _att

        _att._SDPA_PRIORITY[:] = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        logger.info("注意機構: efficient（初めての長さでも遅くならない）")

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel

    st = _State()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="irodori")  # 1本ずつ
    voice = str(args.voice_ref.resolve())

    emoji_re = {"re": None}
    match_warned = {"done": False}

    def _plain_seconds(text: str):
        """絵文字を除いた文の長さ（絵文字が無い・除くと空・予測に失敗なら None＝そろえない）。"""
        if emoji_re["re"] is None:
            emoji_re["re"] = _emoji_pattern()
        plain = emoji_re["re"].sub("", text).replace("\ufe0f", "").strip()
        if plain == text.strip() or not plain:
            return None
        try:
            return predict_seconds(st.runtime, plain, st.latent_path)
        except Exception:
            if not match_warned["done"]:
                match_warned["done"] = True
                logger.exception("長さの予測に失敗（絵文字の文の長さはそろえずに合成する）")
            return None

    def _synth(text: str, seconds, seed, steps, match_plain: bool = False) -> tuple[bytes, float]:
        from irodori_tts.inference_runtime import SamplingRequest

        t = time.perf_counter()
        if match_plain and seconds is None:
            seconds = _plain_seconds(text)
        res = st.runtime.synthesize(SamplingRequest(
            text=text, ref_latents=[st.latent_path], seconds=seconds,
            num_steps=int(steps if steps is not None else args.steps),
            t_schedule_mode=args.t_schedule, sway_coeff=args.sway_coeff,
            seed=int(args.seed if seed is None else seed)))
        ms = (time.perf_counter() - t) * 1000.0
        return _wav_bytes(res.audio, res.sample_rate), ms

    def _load() -> None:
        try:
            from irodori_tts.inference_runtime import (
                RuntimeKey, default_runtime_device, download_hf_checkpoint, get_cached_runtime)

            t = time.perf_counter()
            st.runtime, _ = get_cached_runtime(RuntimeKey(
                checkpoint=download_hf_checkpoint(args.checkpoint),
                model_device=default_runtime_device(),
                model_precision=args.precision,
                codec_device=args.codec_device,
            ))
            st.sample_rate = int(st.runtime.codec.sample_rate)
            logger.info("モデルロード %.1fs（%s）", time.perf_counter() - t, args.checkpoint)
            st.latent_path = _make_latent(st.runtime, args.voice_ref.resolve(), args.cache_dir,
                                          args.checkpoint)
            t = time.perf_counter()
            _synth("あ。", None, args.seed, args.steps)  # 初回の遅さ（約1.3秒）をここで払う
            logger.info("ウォームアップ %.0fms", (time.perf_counter() - t) * 1000.0)
            st.ok = True
            logger.info("準備完了 http://%s:%d （話者 %s・steps %d）", args.host, args.port, voice, args.steps)
        except Exception as e:  # ロード失敗は health で EVE に伝える（落ちない＝理由を読める）
            logger.exception("ロードに失敗")
            st.error = f"{type(e).__name__}: {e}"

    app = FastAPI()

    class SynthReq(BaseModel):
        text: str
        seconds: float | None = None
        seed: int | None = None
        num_steps: int | None = None
        match_plain: bool = False

    @app.get("/health")
    def health():
        return {"ok": st.ok, "model": args.checkpoint, "voice": voice,
                "sample_rate": st.sample_rate, "steps": args.steps, "seed": args.seed,
                "attention": args.attention, "error": st.error, "pid": os.getpid(),
                "match_plain": True}

    @app.post("/synthesize")
    async def synthesize(req: SynthReq):
        if not st.ok:
            raise HTTPException(status_code=503, detail=st.error or "loading")
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="empty text")
        loop = asyncio.get_running_loop()
        try:
            wav, ms = await loop.run_in_executor(
                pool, _synth, req.text, req.seconds, req.seed, req.num_steps, req.match_plain)
        except ValueError as e:  # 正規化で空になった等（入力の問題）
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except Exception as e:
            logger.exception("合成に失敗: %.30s", req.text)
            return JSONResponse(status_code=500, content={"detail": f"{type(e).__name__}: {e}"})
        return Response(content=wav, media_type="audio/wav", headers={"X-Synth-Ms": f"{ms:.1f}"})

    threading.Thread(target=_load, daemon=True, name="irodori-load").start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
