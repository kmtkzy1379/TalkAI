r"""声の設定（速さ・高さ・抑揚・話者）が**実際に音を変えるか**を測る。

    $env:PYTHONIOENCODING="utf-8"
    & .\.venv\Scripts\python.exe tools\voice_params_check.py

**VOICEVOX はローカル HTTP なので API 課金はゼロ。**設定タブに出した以上、
「値は届いているが音は変わらない」を残さないために実際の WAV を測る。
`VV_INTONATION` は今回新設した経路なので、ここを通さないと誰も確かめていないことになる。

測り方（依存を増やさないため標準ライブラリだけ）:
- 速さ: WAV の**長さ**。1.5倍速なら 1/1.5 の長さになるはず。
- 高さ: **ゼロ交差率**（波形が符号を変える回数）。基本周波数が上がれば増える。
- 抑揚: ゼロ交差率の**短時間ごとのばらつき**。抑揚が強いほど高さが動くので増える。
- 話者: 波形そのものが変わること（同一でないこと）。
"""
from __future__ import annotations

import array
import asyncio
import io
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.response.tts import VoicevoxTTS  # noqa: E402

TEXT = "今日はいい天気だね。散歩でもしようか。"


def decode(wav_bytes: bytes) -> tuple:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        channels = w.getnchannels()
    if width != 2:
        raise RuntimeError(f"16bit 以外は未対応: {width * 8}bit")
    samples = array.array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array.array("h", samples[::channels])
    return samples, rate


def zcr(samples, start: int = 0, end: int | None = None) -> float:
    end = len(samples) if end is None else end
    if end - start < 2:
        return 0.0
    crossings = 0
    prev = samples[start]
    for i in range(start + 1, end):
        cur = samples[i]
        if (prev >= 0) != (cur >= 0):
            crossings += 1
        prev = cur
    return crossings / (end - start)


def loud_frames(samples, rate: int, win_ms: int = 40):
    """無音を除いた窓ごとの (開始, 終了)。無音を混ぜるとばらつきが埋もれる。"""
    win = int(rate * win_ms / 1000)
    out = []
    for s in range(0, len(samples) - win, win):
        seg = samples[s:s + win]
        energy = sum(abs(v) for v in seg) / win
        if energy > 500:  # 十分に鳴っている窓だけ
            out.append((s, s + win))
    return out


def stdev(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


async def synth(**kw) -> bytes:
    tts = VoicevoxTTS(**kw)
    try:
        wav = await tts.generate(TEXT)
    finally:
        await tts.close()
    if not wav:
        raise RuntimeError("VOICEVOX から音が返りませんでした（起動していますか？）")
    return wav


async def main() -> int:
    print("VOICEVOX で合成して測ります（ローカル・API 課金なし）\n")
    try:
        base = await synth(speed=1.0, pitch=0.0, intonation=1.0, speaker=1)
    except Exception as e:
        print(f"❌ VOICEVOX に繋がりません: {e}")
        print("   VOICEVOX を起動してから再実行してください。")
        return 2

    bs, rate = decode(base)
    base_dur = len(bs) / rate
    base_zcr = zcr(bs)
    base_var = stdev([zcr(bs, a, b) for a, b in loud_frames(bs, rate)])
    print(f"基準（速さ1.0 / 高さ0.00 / 抑揚1.0 / 話者1）: "
          f"{base_dur:.2f}秒 / ZCR {base_zcr:.4f} / ZCRばらつき {base_var:.4f}")

    ok = True

    def judge(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        ok &= cond
        print(("  ✅ " if cond else "  ❌ ") + name + "  " + detail)

    # --- 速さ ---
    print("\n[速さ] 長さが理論どおり変わるか")
    for speed in (0.75, 1.5):
        s, r = decode(await synth(speed=speed, pitch=0.0, intonation=1.0, speaker=1))
        dur = len(s) / r
        expect = base_dur / speed
        err = abs(dur - expect) / expect
        judge(f"speed={speed}", err < 0.10,
              f"{dur:.2f}秒（理論 {expect:.2f}秒・誤差 {err * 100:.1f}%）")

    # --- 高さ ---
    print("\n[高さ] 上げるとゼロ交差率が上がるか")
    zl = zcr(decode(await synth(speed=1.0, pitch=-0.12, intonation=1.0, speaker=1))[0])
    zh = zcr(decode(await synth(speed=1.0, pitch=0.12, intonation=1.0, speaker=1))[0])
    judge("pitch -0.12 < 0.00 < +0.12", zl < base_zcr < zh,
          f"ZCR {zl:.4f} → {base_zcr:.4f} → {zh:.4f}")

    # --- 抑揚 ---
    print("\n[抑揚] 強めると高さのばらつきが増えるか")
    flat, _r = decode(await synth(speed=1.0, pitch=0.0, intonation=0.0, speaker=1))
    strong, _r2 = decode(await synth(speed=1.0, pitch=0.0, intonation=2.0, speaker=1))
    v_flat = stdev([zcr(flat, a, b) for a, b in loud_frames(flat, rate)])
    v_strong = stdev([zcr(strong, a, b) for a, b in loud_frames(strong, rate)])
    judge("intonation 0.0 < 1.0 < 2.0", v_flat < base_var < v_strong,
          f"ばらつき {v_flat:.4f} → {base_var:.4f} → {v_strong:.4f}")

    # --- 話者 ---
    print("\n[話者] 変えると別の音になるか")
    other = await synth(speed=1.0, pitch=0.0, intonation=1.0, speaker=8)
    judge("speaker 1 ≠ 8", other != base,
          f"WAV バイト数 {len(base)} vs {len(other)}")

    print("\n" + ("✅ 声の設定はすべて実際に音を変えています。"
                  if ok else "❌ 効いていない設定があります（上を参照）。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
