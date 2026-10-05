#!/usr/bin/env python3
"""S1 の実測と聴き比べ用の音声づくり（ローカル GPU のみ・API 不要）。

EVE と同じ経路（`IrodoriServer.ensure()` の自動起動 → `IrodoriTTS.generate`）で:
1. サーバの起動〜準備完了の時間
2. 1文の往復時間（HTTP 込み）を steps 6 / 8 / 12 で
3. 聴き比べ用の音声を `listen/s1/` に保存（git 管理外）
   - `steps{6,8,12}.wav`   … 同じ応答文を steps だけ変えて並べたもの（1秒の間）
   - `seed_fixed.wav`      … seed を固定して別々の文を並べたもの（声が揃うか）
   - `seed_vary.wav`       … 同じ文を seed 0〜3 で並べたもの（seed で声が変わるか）
4. 止めた後にサーバのプロセスが残っていないか（ランチャの子まで）
`--match`: 感情の絵文字を付けた文を「絵文字なしと同じ長さ」で合成できるか（サーバの match_plain・S5）だけを確かめる
  （長さが絵文字なしと同じになるか・長さの予測で合成が何ms遅くなるか）

実行（EVE の venv で）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\irodori_check.py
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import statistics
import sys
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SENTENCES = [
    "そうなんだ、それは大変だったね。",
    "今日はいい天気だから、ちょっと散歩に行きたくなるね。",
    "うん、わかった。調べてみるね。",
    "それって、前に話してた映画のこと？",
    "ごめん、今のところもう一回言ってくれる？",
]
# 往復時間の計測用（**steps ごとに別の文**を使う＝会話と同じく毎回初めての文で測る。
# 同じ文を繰り返すと、形ごとの初回コストが隠れて楽観的な値になる・2026-09-25 に実際に踏んだ）
TIMING = {
    6: ["明日は雨が降るらしいよ。", "猫って本当にかわいいよね。", "それは知らなかった、教えてくれてありがとう。",
        "ちょっと待ってね、考えてみる。", "お腹すいたね、何か食べようか。", "その本、面白そうだね。"],
    8: ["私もそう思うよ。", "あはは、それはおかしいね。", "えっと、どうだったかな。",
        "週末は何をする予定？", "音楽を聴きながら作業すると捗るよね。", "それはちょっと難しいかもしれない。"],
    12: ["なるほど、そういうことか。", "また今度話そうね。", "駅まで歩いて十分くらいかな。",
         "今夜は早く寝ようと思ってる。", "新しいゲームが出たらしいよ。", "空がとてもきれいだね。"],
}


def read_pcm(wav: bytes):
    import numpy as np

    with wave.open(io.BytesIO(wav), "rb") as wf:
        sr = wf.getframerate()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return x, sr


def write_concat(path: Path, wavs: list, gap_sec: float = 1.0) -> float:
    import numpy as np

    parts = []
    sr = 48000
    for w in wavs:
        if w is None:
            continue
        x, sr = read_pcm(w)
        parts += [x, np.zeros(int(gap_sec * sr), dtype="<i2")]
    y = np.concatenate(parts) if parts else np.zeros(1, dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    return len(y) / sr


MATCH_CASES = [  # (文, 絵文字) 応答の節と同じ短さ
    ("そっか、", "😭"), ("それはつらかったね。", "😭"), ("え、ほんとに？", "😊"), ("それすごくうれしい！", "😊"),
    ("それはさすがにひどいと思う。", "😠"), ("うん、ゆっくり話そうか。", "😌"), ("ずっと手に持ってたんだね。", "🤭"),
]


async def check_match(srv) -> int:
    """絵文字を付けた文が、絵文字なしと同じ長さで合成されるか（match_plain）と、その時間。"""
    from eve.response.tts_irodori import IrodoriTTS

    tts = IrodoriTTS(srv)
    await tts.generate("あ。")
    s = await tts._get_session()

    async def synth(text, match):
        async with s.post(f"{srv.url}/synthesize", json={"text": text, "seed": 0, "num_steps": tts.steps,
                                                          "match_plain": match}) as r:
            body = await r.read()
            ms = float(r.headers.get("X-Synth-Ms", "nan"))
        return body, ms

    def dur(wav_bytes):
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())

    rows, bad = [], 0
    for text, e in MATCH_CASES:
        plain, ms_p = await synth(text, False)
        nat, _ms_n = await synth(text + e, False)
        mat, ms_m = await synth(text + e, True)
        dp, dn, dm = dur(plain), dur(nat), dur(mat)
        same = abs(dm - dp) < 0.02
        bad += not same
        rows.append((text + e, dp, dn, dm, ms_p, ms_m))
        print(f"  {text + e:<22} 絵文字なし {dp:.2f}秒 / そのまま {dn:.2f}秒 / そろえた {dm:.2f}秒"
              f"{'' if same else '  ← 長さが違う'} | 合成 なし {ms_p:.0f}ms・そろえた {ms_m:.0f}ms")
    over = statistics.median([m - p for *_x, p, m in rows])
    print(f"== 長さがそろった: {len(rows) - bad}/{len(rows)} / 長さの予測で増えた合成時間 中央 {over:+.0f}ms")
    await s.close()
    return 1 if bad else 0


async def main_async(args) -> int:
    import psutil

    from eve.response.tts_irodori import SERVER_MARKER, IrodoriServer, IrodoriTTS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    srv = IrodoriServer()
    t0 = time.perf_counter()
    ok = await srv.ensure()
    t_start = time.perf_counter() - t0
    if not ok:
        print("サーバを用意できなかった（ログ:", srv.log_path, "）")
        return 1
    print(f"\n== サーバ準備 {t_start:.1f}秒（{'起動した' if srv.owned else '既存を使った'}）")
    if args.match:
        try:
            return await check_match(srv)
        finally:
            await srv.stop()
    out = REPO / "listen" / "s1"
    results = {}
    try:
        for steps in (6, 8, 12):
            tts = IrodoriTTS(srv, seed=0, steps=steps)
            await tts.generate("あ。")  # この steps での1回目（HTTP 接続も温める）
            times = []
            for s in TIMING[steps]:
                t = time.perf_counter()
                await tts.generate(s)
                times.append(time.perf_counter() - t)
            results[steps] = times
            wavs = [await tts.generate(s) for s in SENTENCES]  # 聴き比べ用（全 steps で同じ文）
            dur = write_concat(out / f"steps{steps}.wav", wavs)
            print(f"steps {steps:2d}: 初めての文の往復 中央 {statistics.median(times)*1000:.0f}ms / "
                  f"最大 {max(times)*1000:.0f}ms（n={len(times)}）→ {out / f'steps{steps}.wav'}（{dur:.1f}秒）")
            await tts._session.close()
        tts = IrodoriTTS(srv, seed=0, steps=args.listen_steps)
        write_concat(out / "seed_fixed.wav", [await tts.generate(s) for s in SENTENCES])
        vary = []
        for seed in range(4):
            tts.seed = seed
            vary.append(await tts.generate(SENTENCES[0]))
        write_concat(out / "seed_vary.wav", vary)
        print(f"聴き比べ: {out / 'seed_fixed.wav'}（seed 0 で別々の文）/ {out / 'seed_vary.wav'}（同じ文を seed 0〜3）")
        await tts._session.close()
    finally:
        owned = srv.owned
        await srv.stop()
    if owned:
        await asyncio.sleep(1.0)
        left = [p.pid for p in psutil.process_iter(["cmdline"])
                if SERVER_MARKER in (p.info.get("cmdline") or ())]
        print(f"止めた後に残ったサーバのプロセス: {left or 'なし'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-steps", type=int, default=6, help="seed の聴き比べに使う steps")
    ap.add_argument("--match", action="store_true", help="感情の絵文字の文の長さそろえ（match_plain）だけ確かめる")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
