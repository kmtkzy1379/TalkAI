"""リップシンク（口の開閉）の計画を VOICEVOX の `audio_query` から組み立てる。

**なぜ音量(RMS)ではなくモーラ長なのか**（`spike/live2d_test.py` で実測・2026-08-13）:
日本語 TTS は母音が連続して有声のまま繋がるため、**モーラの切れ目が振幅の谷にならない**。
窓幅・レンジ・release を54通り掃引しても「閉じ 2.4回/秒」が上限で、実モーラ速度 5.0回/秒 に
届かず「口が開きっぱなし」に見えた。`audio_query` には consonant_length / vowel_length が
入っているので、そこから直接組み立てると 5.1〜5.2回/秒 で追随する（掃引実測）。

**このモジュールは純粋関数だけを持つ**（I/O なし・状態なし）＝ Tier-1 決定論テストで固定できる。
UI（別スレッド）へ渡すのは `MouthPlan`＝**float のタプルのリスト**で、loop 所有オブジェクトへの
参照を含まない（`PIPELINE_DESIGN §9.3` の橋渡し契約: 値のコピーだけを渡す）。

**このモデルでは母音別の口の"形"は作れない**（`ParamA/I/U/E/O` が nun model に無い＝実測）。
作れるのは開き"量"だけなので、母音ごとに量を変えて自然に見せる。
"""
from __future__ import annotations

from typing import Any, Optional

# (t0, t1, v0, v1) = 開始秒, 終了秒, 開始時の開き, 終了時の開き（区間内は線形補間）
MouthPlan = list[tuple[float, float, float, float]]

# 母音ごとの開き量。掃引ではなく口の形の常識で置いた初期値（UI 側で調整可能にする）。
VOWEL_OPEN: dict[str, float] = {
    "a": 1.00, "e": 0.72, "o": 0.62, "i": 0.42, "u": 0.34,
    "N": 0.18,   # ん
    "cl": 0.0,   # っ（促音・閉鎖）
    "pau": 0.0,  # 無音
}
# 破裂音は口を完全に閉じる。それ以外の子音はわずかに開いたまま。
PLOSIVE = frozenset({"p", "b", "t", "d", "k", "g", "ch", "ts",
                     "ky", "gy", "by", "py", "ty", "dy"})
# 破裂音以外の子音での開き。掃引実測 2026-08-13（4文・VOICEVOX speaker1・attack20/release40ms）:
# 0.15 → 開口率76% / 閉じ4.6回/秒、**0.08 → 72% / 5.1回/秒**、0.00 → 72% / 5.1回/秒。
# 実モーラ速度 4.5個/秒 に対し 0.08 がほぼ 1:1 で追随しつつ s/n で閉じ切らない分だけ自然。
CONSONANT_OPEN = 0.08
# 長い母音は少しずつ閉じていく（固まって見えないように）。この秒数を超えたら減衰させる。
_DECAY_OVER_SEC = 0.12
_DECAY_TO = 0.78


def plan_from_query(query: dict) -> MouthPlan:
    """`audio_query` の戻り（**speedScale 適用後**のもの）から口の開閉計画を作る。

    速度は `speedScale` で割る（VOICEVOX は長さを割ることで速度を変えるため）。
    壊れた/想定外の query でも例外を投げず、取れたところまでを返す（呼び出し側は fail-soft）。
    """
    try:
        speed = float(query.get("speedScale") or 1.0) or 1.0
    except (TypeError, ValueError):
        speed = 1.0
    t = _f(query.get("prePhonemeLength")) / speed
    plan: MouthPlan = []

    def add(dur: float, v0: float, v1: float) -> None:
        nonlocal t
        if dur > 0:
            plan.append((t, t + dur, v0, v1))
            t += dur

    for phrase in query.get("accent_phrases") or ():
        for mora in phrase.get("moras") or ():
            vowel = mora.get("vowel") or "a"
            # 大文字＝無声化母音（「です」の す 等）。口は開くが浅い
            devoiced = vowel.isupper() and vowel != "N"
            lv = VOWEL_OPEN.get(vowel.lower() if devoiced else vowel, 0.5)
            if devoiced:
                lv *= 0.45
            cl = _f(mora.get("consonant_length"))
            if cl > 0:
                lvc = 0.0 if (mora.get("consonant") or "") in PLOSIVE else CONSONANT_OPEN
                add(cl / speed, lvc, lvc)
            vl = _f(mora.get("vowel_length")) / speed
            add(vl, lv, lv * (_DECAY_TO if vl > _DECAY_OVER_SEC else 1.0))
        pause = phrase.get("pause_mora")
        if pause:
            add(_f(pause.get("vowel_length")) / speed, 0.0, 0.0)
    return plan


def level_at(plan: MouthPlan, t: float) -> float:
    """再生開始からの経過秒 `t` における口の開き（0..1）。区間の外は 0（＝閉じ）。

    UI の描画ループから毎フレーム呼ぶ。plan は短い（1文ぶん・数十区間）ので線形走査で足りる。
    """
    if t < 0.0 or not plan:
        return 0.0
    for t0, t1, v0, v1 in plan:
        if t < t0:
            return 0.0        # 区間の隙間（無音）
        if t < t1:
            span = t1 - t0
            return v0 if span <= 0 else v0 + (v1 - v0) * ((t - t0) / span)
    return 0.0


def plan_duration(plan: MouthPlan) -> float:
    return plan[-1][1] if plan else 0.0


class MouthSmoother:
    """口の開きを時間方向に均す（フレームレート非依存の非対称スムージング）。

    掃引実測 2026-08-13: attack は 15-22ms で差が出ず、release は **40ms が最良**
    （30ms=開口率67%で閉じすぎ / 55ms=74%で開きっぱなし寄り）。
    """

    def __init__(self, attack_sec: float = 0.020, release_sec: float = 0.040) -> None:
        self.attack_sec = attack_sec
        self.release_sec = release_sec
        self.value = 0.0

    def step(self, target: float, dt: float) -> float:
        import math

        tau = self.attack_sec if target > self.value else self.release_sec
        self.value += (target - self.value) * (1.0 - math.exp(-dt / max(1e-3, tau)))
        return self.value

    def reset(self) -> None:
        self.value = 0.0


def _f(v: Any) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def plan_from_query_safe(query: Optional[dict]) -> Optional[MouthPlan]:
    """`plan_from_query` の例外を潰す版（本番経路はこちらを使う＝TTS を口パクで壊さない）。"""
    if not query:
        return None
    try:
        plan = plan_from_query(query)
    except Exception:  # 想定外の query 形でも音声だけは出す
        return None
    return plan or None
