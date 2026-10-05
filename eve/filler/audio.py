"""フィラー音声の下ごしらえ（純関数・numpy のみ・I/O は bytes だけ）。

**フィラーの再生（`player.py`）と試聴ファイル（`tools/filler_audition.py`）の両方がこれを使う**
＝ユーザが聴いて決めた音と、本番で鳴る音が同じ処理を通る。

- `trim`: クリップの前後の無音を刈る。Irodori は `seconds` ぴったりの長さで返すので、短い語
  （あ 0.5秒）は後ろが無音で埋まっている。刈らないと「、」の間が語ごとにばらつく
- `stretch`: 今の語に「ー」を継ぐ（あ→あー）。語の伸びている所で切り、長く伸ばした母音の
  伸びている所から終わりまでを、波形の0点で短いクロスフェードをかけて継ぐ（ユーザ指定の試作）
- `join`: 語と語の間に「、」の間（無音）を入れて並べる
"""
from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np

_HOP_SEC = 0.01


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes → (float32 モノラル, サンプルレート)。PCM16 と float の両方を読む。"""
    try:
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def to_wav(x: np.ndarray, sr: int) -> bytes:
    """float32 モノラル → PCM16 の WAV bytes（`RealAudioPlayer.play_fn` がそのまま読める形）。"""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm)
    return buf.getvalue()


def _envelope_db(x: np.ndarray, sr: int, hop_sec: float = _HOP_SEC) -> np.ndarray:
    hop = max(1, int(sr * hop_sec))
    n = len(x) // hop
    if n == 0:
        return np.full(1, -120.0)
    f = x[: n * hop].reshape(n, hop)
    return 20.0 * np.log10(np.sqrt(np.mean(f * f, axis=1)) + 1e-9)


def _fade(x: np.ndarray, sr: int, fade_in_sec: float, fade_out_sec: float) -> np.ndarray:
    y = x.copy()
    a = min(len(y), int(sr * fade_in_sec))
    b = min(len(y), int(sr * fade_out_sec))
    if a > 0:
        y[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
    if b > 0:
        y[len(y) - b:] *= np.linspace(1.0, 0.0, b, dtype=np.float32)
    return y


def trim(x: np.ndarray, sr: int, *, floor_db: float = 40.0, pad_sec: float = 0.015) -> np.ndarray:
    """前後の無音を刈る（その音声で一番大きい所から `floor_db` 下を無音とみなす）。

    声の立ち上がり・消え際を削りすぎないよう `pad_sec` だけ余白を残し、両端に 5ms のフェードを掛ける
    （切り口のプチッという音を防ぐ）。全体が無音なら長さ0を返す。
    """
    if len(x) == 0:
        return x
    env = _envelope_db(x, sr)
    top = float(env.max())
    if top < -80.0:
        return x[:0]
    on = np.nonzero(env >= top - floor_db)[0]
    hop = max(1, int(sr * _HOP_SEC))
    pad = int(sr * pad_sec)
    s = max(0, int(on[0]) * hop - pad)
    e = min(len(x), (int(on[-1]) + 1) * hop + pad)
    return _fade(x[s:e], sr, 0.005, 0.005)


def _zero_cross_near(x: np.ndarray, i: int, radius: int) -> int:
    """i の近く（±radius）で符号が負→正に変わる位置（無ければ i のまま）。"""
    lo, hi = max(1, i - radius), min(len(x) - 1, i + radius)
    if hi <= lo:
        return i
    seg = x[lo - 1:hi]
    idx = np.nonzero((seg[:-1] < 0) & (seg[1:] >= 0))[0]
    if len(idx) == 0:
        return i
    cand = idx + lo
    return int(cand[np.argmin(np.abs(cand - i))])


def _sustain_end(x: np.ndarray, sr: int, level_db: float = 6.0) -> int:
    """伸びている部分の終わり（最後に「最大から level_db 以内」だった位置）＝減衰が始まる所。"""
    env = _envelope_db(x, sr)
    top = float(env.max())
    on = np.nonzero(env >= top - level_db)[0]
    hop = max(1, int(sr * _HOP_SEC))
    return min(len(x), (int(on[-1]) + 1) * hop) if len(on) else len(x)


def _sustain_start(x: np.ndarray, sr: int, level_db: float = 6.0) -> int:
    env = _envelope_db(x, sr)
    top = float(env.max())
    on = np.nonzero(env >= top - level_db)[0]
    hop = max(1, int(sr * _HOP_SEC))
    return int(on[0]) * hop if len(on) else 0


def pitch(seg: np.ndarray, sr: int, fmin: float = 150.0, fmax: float = 1000.0,
          threshold: float = 0.15) -> Optional[float]:
    """声の高さ[Hz]（YIN・FFT で差分関数を作る）。無声・無音は None。"""
    x = np.asarray(seg, dtype=np.float64)
    x = x - x.mean()
    tmax = int(sr / fmin)
    tmin = max(2, int(sr / fmax))
    w = len(x) - tmax
    if w < tmax or np.sqrt(np.mean(x * x)) < 1e-3:
        return None
    n = 1 << int(np.ceil(np.log2(len(x) + w)))
    fx = np.fft.rfft(x, n)
    fh = np.fft.rfft(x[:w][::-1], n)
    r = np.fft.irfft(fx * fh, n)[w - 1: w - 1 + tmax]      # r[t] = Σ x[i] x[i+t]（i<w）
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    e1 = cs[w]
    e2 = cs[np.arange(tmax) + w] - cs[np.arange(tmax)]
    d = e1 + e2 - 2.0 * r
    cm = np.ones(tmax)
    run = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, tmax) / np.maximum(run, 1e-12)
    for t in range(tmin, tmax - 1):
        if cm[t] < threshold:
            while t + 1 < tmax and cm[t + 1] < cm[t]:
                t += 1
            return sr / t
    return None


def pitch_track(x: np.ndarray, sr: int, *, hop_sec: float = 0.01, win_sec: float = 0.04) -> list:
    """声の高さの並び[Hz]（無声・無音の所は除く）。"""
    win, hop = int(sr * win_sec), max(1, int(sr * hop_sec))
    out = []
    for i in range(0, max(0, len(x) - win), hop):
        p = pitch(x[i:i + win], sr, fmin=120.0, fmax=900.0)
        if p:
            out.append(p)
    return out


def contour(x: np.ndarray, sr: int, ref_hz: float) -> Optional[tuple[float, float]]:
    """抑揚の要約 = (声の高さ[参照比・半音], 語尾[終わり−始め・半音])。測れなければ None。

    ユーザ 2026-09-26「フィラーは次の言葉探してるから基本は音程同じか、若干下がる」を測る物差し:
    語尾が上がる（＋）と「納得いってない」「驚き」に聞こえ、高すぎても感情が乗って聞こえる。
    始め／終わりは有声部分の最初と最後の1/3の中央値（一瞬の揺れに引っ張られないように）。
    """
    f = pitch_track(trim(x, sr), sr)
    if len(f) < 5 or ref_hz <= 0:
        return None
    k = max(2, len(f) // 3)
    level = 12.0 * float(np.log2(float(np.median(f)) / ref_hz))
    slope = 12.0 * float(np.log2(float(np.median(f[-k:])) / float(np.median(f[:k]))))
    return level, slope


def calm_score(c: Optional[tuple[float, float]], *, max_level: float = 3.0,
               slope_lo: float = -2.5, slope_hi: float = 0.5) -> float:
    """落ち着いて聞こえるほど小さい点数（0 = 基準内）。基準: 語尾 −2.5〜+0.5半音・高さ +3半音まで。

    基準内でも「同じか若干下がる」（語尾 −0.5半音・高さ ±0 付近）に近いほど少し良くする。
    """
    if c is None:
        return 50.0
    level, slope = c
    s = 0.0
    s += 3.0 * max(0.0, slope - slope_hi)      # 語尾が上がる（最も目立つ・重く見る）
    s += 1.0 * max(0.0, slope_lo - slope)      # 下がりすぎ（ため息・投げやりに聞こえる）
    s += 1.5 * max(0.0, level - max_level)     # 高すぎる（驚き・興奮に聞こえる）
    s += 0.1 * abs(slope + 0.5) + 0.05 * abs(level)
    return s


def stretch(word: np.ndarray, source, sr: int, *, add_sec: float = 0.4,
            xfade_sec: float = 0.03, max_jump_st: float = 3.0) -> Optional[np.ndarray]:
    """語 `word` の最後の母音を `source`（長く伸ばした同じ母音・1本か複数本）で `add_sec` ほど伸ばす。

    1. 語は伸びている部分の終わり（減衰の手前）で切る
    2. 素材は終わり近く（継ぎ足す長さが add_sec の 0.6〜1.4倍になる範囲）から使う＝自然な消え際を残す。
       ただし素材の伸びている部分より前には戻らない（立ち上がりを混ぜない）。
       **その範囲で、切り口と声の高さが一番近い所（素材が複数なら素材ごと）を選ぶ**
       （短い語と長い母音とで抑揚が違い、そのまま継ぐと高さが跳ぶ・試作で実測）
    3. 切り口を波形の0点に寄せ、音量を語の切り口に合わせて、`xfade_sec` のクロスフェードで継ぐ
    どちらかが短すぎて継げない時は None（呼び出し側は伸ばし棒を使わない）。
    """
    w = trim(word, sr)
    xf = int(sr * xfade_sec)
    if len(w) < 2 * xf:
        return None
    radius = int(sr * 0.003)
    cut = _zero_cross_near(w, _sustain_end(w, sr), radius)
    cut = max(xf, min(len(w), cut))
    look = int(sr * 0.04)
    f_word = pitch(w[max(0, cut - look - xf): max(0, cut - xf)], sr)
    sources = source if isinstance(source, (list, tuple)) else [source]
    # 点数 = 継ぎ目の高さの差[半音] ＋ 継いだ後の語尾が上がる/下がりすぎる分（ユーザ 2026-09-26
    # 「伸ばすと語尾が少し上がるのが変。フィラーは次の言葉探してるから基本は音程同じか、若干下がる」）
    # ＋ 狙いの長さからのずれ（0.1秒で0.5半音相当）
    best = None  # (点数, 素材, 使い始め)
    step = int(sr * 0.01)
    for src in sources:
        s = trim(src, sr)
        if len(s) < 2 * xf:
            continue
        hop = int(sr * 0.01)
        win = int(sr * 0.04)
        track = [pitch(s[i:i + win], sr, fmin=120.0, fmax=900.0) for i in range(0, max(0, len(s) - win), hop)]
        target = len(s) - int(sr * add_sec) - xf
        lo = max(_sustain_start(s, sr), len(s) - int(sr * add_sec * 1.4) - xf)
        hi = max(lo, len(s) - int(sr * add_sec * 0.6) - xf)
        for st in range(lo, hi + 1, step):
            f_src = pitch(s[st: st + look], sr) if f_word else None
            if f_word is None:
                diff = 0.0   # 語の切り口の高さが測れない（息っぽい語尾）＝段差は分からない。形と位置で選ぶ
            elif f_src:
                diff = abs(12.0 * np.log2(f_src / f_word))
            else:
                diff = 24.0  # 素材のその位置が無声＝継ぎ目にしない
            tail = [f for f in track[st // hop:] if f]
            if len(tail) >= 6:
                k = max(2, len(tail) // 3)
                slope = 12.0 * float(np.log2(float(np.median(tail[-k:])) / float(np.median(tail[:k]))))
                shape = 3.0 * max(0.0, slope - 0.5) + 1.0 * max(0.0, -2.5 - slope)
            else:
                shape = 3.0  # 測れない尾は少し不利に
            score = diff + shape + 5.0 * abs(st - target) / sr
            if best is None or score < best[0]:
                best = (score, s, st, diff)
    if best is None:
        return None
    _, s, start, jump = best
    if jump > max_jump_st:
        return None  # 継ぎ目で高さが跳ぶ（段差が聞こえる）＝この組み合わせでは伸ばさない
    start = _zero_cross_near(s, start, radius)
    if len(s) - start < 2 * xf:
        return None
    # 音量合わせ: 切り口直前（語）と継ぎ目直後（素材）の RMS をそろえる
    win = int(sr * 0.05)
    rw = float(np.sqrt(np.mean(w[max(0, cut - win):cut] ** 2)) + 1e-9)
    rs = float(np.sqrt(np.mean(s[start:start + win] ** 2)) + 1e-9)
    tail = s[start:] * min(4.0, rw / rs)
    fade_out = np.cos(np.linspace(0, np.pi / 2, xf, dtype=np.float32))  # 等パワー
    fade_in = np.sin(np.linspace(0, np.pi / 2, xf, dtype=np.float32))
    head = w[:cut].copy()
    mixed = head[-xf:] * fade_out + tail[:xf] * fade_in
    out = np.concatenate([head[:-xf], mixed, tail[xf:]]).astype(np.float32)
    return _fade(out, sr, 0.0, 0.005)


def silence(sec: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(round(sec * sr))), dtype=np.float32)


def join(pieces: list, sr: int, gap_sec: float) -> np.ndarray:
    """語を「、」の間（無音 gap_sec）を挟んで並べる（試聴用。本番は語ごとに鳴らして間を空ける）。"""
    out: list = []
    for i, p in enumerate(pieces):
        if i:
            out.append(silence(gap_sec, sr))
        out.append(p)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
