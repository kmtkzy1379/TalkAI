"""Tier-1: 感情 → 動きの係数（`eve/ui/emotion.py`）。Qt も GPU も要らない純ロジック。

**この層は表示専用**という不変条件をテストで固定する:
  - `resolve()` は surprise を**読むだけ**（対話の挙動に影響する経路を持たない）
  - どんな入力でも例外を出さない（LLM の自由記述＋壊れた文字列が来る）
  - ユーザ指定の対応（悲しい/申し訳ない→うつむく、うれしい→速く大きく、
    戸惑い→首をかしげる、surprise 大→揺れも頻度も増す）が数値で成立している
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.ui.emotion import (  # noqa: E402
    NEUTRAL, NEUTRAL_SURPRISE, PROFILES, classify, resolve,
)

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def t_classify_real_outputs() -> None:
    """本番 `rag_memory.jsonl` に実在した表現（2026-08-13 時点の実データから採取）。"""
    print("\n[1] 実際に出力された感情表現の分類")
    cases = {
        "ほっとした": "relieved", "落ち着き": "calm", "穏やか": "calm",
        "様子見": "watching", "気づかい": "warm", "了解": "watching",
        "申し訳なさ": "sorry", "親しみ": "warm", "待機中": "watching",
        "期待": "eager", "少し戸惑い": "puzzled", "前向き": "eager",
        "ほんわか": "warm", "安心": "relieved", "もどかしい": "puzzled",
        "ちょっと拍子抜け": "puzzled", "しみじみ": "calm", "反省": "sorry",
        "心配": "worried", "すっきり": "happy", "うれしい": "happy",
        "興味津々": "eager", "わくわく": "happy", "軽い不安": "worried",
        "ばつが悪い": "sorry", "ちょっと手探り": "puzzled", "冷静": "calm",
        "少し焦り": "worried", "のんびり": "calm", "もやもや": "puzzled",
    }
    bad = [(k, classify(k)[0], v) for k, v in cases.items() if classify(k)[0] != v]
    check(f"実出力 {len(cases)} 種が期待どおりに分類される", not bad, f"外れ: {bad}")


def t_classify_robust() -> None:
    print("\n[2] 壊れた入力でも落ちない")
    for v in ("", None, "物知り気分", "शांतかつ少し名残惜しい", "🙂🙂", "x" * 5000, 123):
        try:
            label, inten = classify(v)  # type: ignore[arg-type]
            ok = isinstance(label, str) and 0.0 < inten <= 2.0
        except Exception as e:
            ok = False
            label = f"例外 {e}"
        check(f"classify({str(v)[:14]!r}) が安全", ok, f"{label}")
    check("未知語は既定ラベルに落ちる", classify("物知り気分")[0] == NEUTRAL)


def t_intensity() -> None:
    print("\n[3] 強度の修飾語")
    check("「ちょっと」で弱まる", classify("ちょっと戸惑い")[1] < classify("戸惑い")[1])
    check("「津々」で強まる", classify("興味津々")[1] > classify("興味")[1])
    weak = resolve("ちょっと申し訳ない", NEUTRAL_SURPRISE)
    strong = resolve("申し訳なさ", NEUTRAL_SURPRISE)
    check("弱い修飾語ならうつむき方も浅い", weak.pitch_bias > strong.pitch_bias,
          f"{weak.pitch_bias:.3f} vs {strong.pitch_bias:.3f}")


def t_user_requested_mapping() -> None:
    """ユーザが明示した対応（2026-08-13）が数値で成立しているか。"""
    print("\n[4] 指定された感情→動きの対応")
    calm = resolve("落ち着き", NEUTRAL_SURPRISE)
    sorry = resolve("申し訳なさ", NEUTRAL_SURPRISE)
    happy = resolve("うれしい", NEUTRAL_SURPRISE)
    puzzled = resolve("戸惑い", NEUTRAL_SURPRISE)
    worried = resolve("心配", NEUTRAL_SURPRISE)

    check("申し訳ない → うつむく（pitch が負）", sorry.pitch_bias < -0.2,
          f"{sorry.pitch_bias:.2f}")
    check("申し訳ない → 動きが減る", sorry.sway < calm.sway, f"{sorry.sway:.2f}")
    check("心配 → 少しうつむく（申し訳ないより浅い）",
          worried.pitch_bias < 0 and worried.pitch_bias > sorry.pitch_bias)
    # ユーザ指摘 2026-08-14: 心配は「遅くする」より「落ち着かない」方が自然
    check("心配 → 速さが増す（そわそわする）", worried.speed > calm.speed * 1.5,
          f"{worried.speed:.2f}")
    check("心配 → よそ見が増える（落ち着きなく視線が動く）", worried.glance < calm.glance * 0.6,
          f"{worried.glance:.2f}")
    check("心配 → 揺れ幅は大きくしない（小刻み）", worried.sway < calm.sway, f"{worried.sway:.2f}")
    # ユーザ指摘 2026-08-14: 差が分からないので 2〜3倍まで開く
    check("うれしい → 揺れが 2倍以上", happy.sway > calm.sway * 2.0, f"{happy.sway:.2f}")
    check("うれしい → 速さが 2倍以上", happy.speed > calm.speed * 2.0, f"{happy.speed:.2f}")
    check("うれしい → 顔が上がる", happy.pitch_bias > 0, f"{happy.pitch_bias:.2f}")
    check("うれしい → 目が笑う", happy.eye_smile > 0.3, f"{happy.eye_smile:.2f}")
    check("戸惑い → 首をかしげる", puzzled.tilt_bias > 0.3, f"{puzzled.tilt_bias:.2f}")
    check("既定(calm)は素の値", abs(calm.sway - 1.0) < 1e-9 and abs(calm.speed - 1.0) < 1e-9)


def t_kidoairaku() -> None:
    """喜怒哀楽が揃っていること（ユーザ指定 2026-08-13）。実データには怒/哀は未出現だが置く。"""
    print("\n[4b] 喜怒哀楽")
    for word, want in (("うれしい", "happy"), ("怒り", "angry"),
                       ("悲しい", "sad"), ("たのしい", "happy")):
        check(f"{word} → {want}", classify(word)[0] == want, f"{classify(word)[0]}")
    for w in ("イライラ", "腹立たしい", "不満", "呆れ"):
        check(f"{w} → angry", classify(w)[0] == "angry", f"{classify(w)[0]}")
    for w in ("寂しさ", "しょんぼり", "切ない", "落ち込み"):
        check(f"{w} → sad", classify(w)[0] == "sad", f"{classify(w)[0]}")

    angry = resolve("怒り", NEUTRAL_SURPRISE)
    sad = resolve("悲しい", NEUTRAL_SURPRISE)
    calm = resolve("落ち着き", NEUTRAL_SURPRISE)
    sorry = resolve("申し訳なさ", NEUTRAL_SURPRISE)
    check("怒り → 動きが速い", angry.speed > calm.speed * 1.3, f"{angry.speed:.2f}")
    check("怒り → 揺れは中程度（喜びほど大きくしない）",
          calm.sway < angry.sway < resolve("うれしい", NEUTRAL_SURPRISE).sway,
          f"{angry.sway:.2f}")
    check("怒り → 相手を見据える（よそ見の間隔が長い）", angry.glance > calm.glance * 1.5,
          f"{angry.glance:.2f}")
    # ユーザ指定 2026-08-14: 怒りはまばたきを**増やす**（減らすと据わった目になり不自然）
    check("怒り → まばたきが増える", angry.blink > calm.blink * 1.5, f"{angry.blink:.2f}")
    check("怒り → 眉が怒り側へ", angry.brow_form > 0.5, f"{angry.brow_form:.2f}")
    check("怒り → 口角が下がる", angry.mouth_form < 0, f"{angry.mouth_form:.2f}")
    check("哀 → いちばん深くうつむく", sad.pitch_bias < sorry.pitch_bias, f"{sad.pitch_bias:.2f}")
    check("哀 → うつむきが露骨（-0.8 以下）", sad.pitch_bias <= -0.8, f"{sad.pitch_bias:.2f}")
    check("哀 → いちばん動きが鈍い", sad.speed < sorry.speed and sad.sway < sorry.sway)
    check("哀 → 困り眉（怒りと逆符号）", sad.brow_form < 0 < angry.brow_form)
    # 感情どうしの差が「見て分かる」水準か（最速/最遅・最大/最小の開き）
    speeds = {k: resolve(v, NEUTRAL_SURPRISE).speed for k, v in
              (("happy", "うれしい"), ("sad", "悲しい"), ("worried", "心配"),
               ("watching", "様子見"), ("angry", "怒り"))}
    check("最速と最遅の速さが 3倍以上開く",
          max(speeds.values()) / min(speeds.values()) >= 3.0,
          f"{ {k: round(v, 2) for k, v in speeds.items()} }")
    sways = {k: resolve(v, NEUTRAL_SURPRISE).sway for k, v in
             (("happy", "うれしい"), ("sad", "悲しい"), ("watching", "様子見"))}
    check("最大と最小の揺れが 5倍以上開く",
          max(sways.values()) / min(sways.values()) >= 5.0,
          f"{ {k: round(v, 2) for k, v in sways.items()} }")


def t_surprise() -> None:
    print("\n[5] surprise で揺れが大きく・頻度が増す")
    lo = resolve("落ち着き", NEUTRAL_SURPRISE)
    hi = resolve("落ち着き", 100)
    check("surprise 大 → 揺れが大きい", hi.sway > lo.sway * 1.5, f"{lo.sway:.2f}→{hi.sway:.2f}")
    check("surprise 大 → 速い", hi.speed > lo.speed * 1.3, f"{lo.speed:.2f}→{hi.speed:.2f}")
    check("surprise 大 → よそ見の間隔が縮む（＝頻度増）", hi.glance < lo.glance * 0.6,
          f"{lo.glance:.2f}→{hi.glance:.2f}")
    check("surprise 大 → まばたきが増える", hi.blink > lo.blink, f"{hi.blink:.2f}")
    check("surprise 大 → 眉が上がる", hi.brow > lo.brow, f"{hi.brow:.2f}")
    check("中立未満の surprise は効かない（0 と 20 が同じ）",
          resolve("落ち着き", 0).sway == lo.sway)
    check("surprise が範囲外でも壊れない",
          resolve("落ち着き", 99999).sway > 0 and resolve("落ち着き", -50).sway > 0)


def t_display_only() -> None:
    """**中核原理の防衛線**: この層は surprise を読むだけで、発話には一切関与しない。"""
    print("\n[6] 表示専用であること")
    import inspect

    import eve.ui.emotion as mod

    import ast

    src = inspect.getsource(mod)
    # コメント/docstring の言及は許す（設計意図の説明に必要）。**実際の import と参照**を見る。
    tree = ast.parse(src)
    imported = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = []
    for n in imported:
        if isinstance(n, ast.ImportFrom):
            names.append(n.module or "")
        else:
            names += [a.name for a in n.names]
    check("本体パイプラインを import していない（表示専用の一方通行）",
          not any(("feedback" in m or "pipeline" in m or "speech" in m or "voice_loop" in m)
                  for m in names), f"{names}")
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("should_speak", "note_feedback_surprise", "note_vlm_surprise", "put"):
        check(f"{forbidden} を呼んでいない", forbidden not in called)
    check("すべてのラベルにプロファイルがある",
          all(lbl in PROFILES for lbl, _ in mod._KEYWORDS))


def t_prediction_state_field() -> None:
    print("\n[7] PredictionState の感情フィールド（単一書込・既定 None）")
    from eve.feedback.prediction_state import PredictionState

    st = PredictionState()
    check("last_emotion の既定は None", st.last_emotion is None)
    check("surprise の既定は中立", st.surprise == NEUTRAL_SURPRISE)
    st.last_emotion = "ほっとした"
    check("感情を入れても surprise は変わらない（独立）", st.surprise == NEUTRAL_SURPRISE)
    m = resolve(st.last_emotion, st.surprise)
    check("PredictionState の値から Motion を作れる", m.sway > 0)


if __name__ == "__main__":
    t_classify_real_outputs()
    t_classify_robust()
    t_intensity()
    t_user_requested_mapping()
    t_kidoairaku()
    t_surprise()
    t_display_only()
    t_prediction_state_field()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
