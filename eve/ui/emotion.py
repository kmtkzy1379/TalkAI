"""感情ラベル付けと、それを Live2D の動きへ翻訳するプロファイル。

**新しい感情分析ライブラリは入れない。** FeedbackLLM が既に毎応答後に「イブの感情」を
短い日本語で出しており（`eve/feedback/parser.py:23` の `FeedbackResult.emotions`）、
それは**会話の文脈込みで判断された値**だから。発話文だけを外部分類器に投げるより情報が多い
（「うん、わかった」は単体では中立だが文脈では安心にも申し訳なさにもなる）。
辞書ベース（pymlask/oseti）は MeCab、BERT 系（WRIME）は数百MB のモデルと 100-300ms の
推論が要る。ここで欲しいのは**動きのバイアス**であって精緻な分類ではない。

**遅延について（ユーザ懸念）**: FeedbackLLM は応答完了後に非同期で走るので、感情は
1ターン遅れる。これは許容する — ここで扱うのは「気分（mood）」であって、文の途中で
切り替わるべきものではないため。即応が要ると分かったら、発話開始フックで届く**文そのもの**を
同じ `classify()` に通す経路を足せる（辞書引きだけなので loop 上でも数μs）。

**ラベルは実データから決めた**（本番 `rag_memory.jsonl` の 269チャンク・98種の実出力）。
怒り・恐怖・強い喜び・悲しみは**一度も出現しない**。実際に出るのは
落ち着き / 気遣い / 申し訳なさ / 戸惑い / 期待 系に偏っている。よって汎用の感情ホイールでは
なく、観測された分布に合わせたラベルを置き、**振れ幅も控えめ**にする（大げさな差を付けると
そもそも発火せず、発火した時に不自然になる）。

このモジュールは **Qt に依存しない純ロジック**（Tier-1 決定論テストで固定できる）。
"""
from __future__ import annotations

from dataclasses import dataclass

NEUTRAL = "calm"

# 実出力98種を眺めて作った対応表。**上から順に部分一致**（先に書いたものが勝つ）。
# 例:「反省しつつ安心」は sorry（姿勢の補正としては反省側が読み取りやすい）。
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 怒・哀は**実データに一度も出現しない**（269チャンク/98種で0件）。それでも置くのは
    # 喜怒哀楽を揃えるため（ユーザ指定 2026-08-13）。出現したら拾える状態にしておく。
    ("angry",    ("怒", "いらだ", "イライラ", "苛立", "むっと", "ムッと", "腹立",
                  "不満", "呆れ", "あきれ", "憤")),
    ("sad",      ("悲し", "寂し", "さみし", "せつな", "切な", "しょんぼり", "落ち込",
                  "泣き", "哀", "つら", "辛い", "虚し", "喪失")),
    ("sorry",    ("申し訳", "反省", "ばつが悪", "うっかり", "すまな", "ごめん")),
    ("worried",  ("心配", "気がかり", "不安", "焦り", "案じ")),
    ("puzzled",  ("戸惑", "困惑", "もどかし", "拍子抜け", "もやもや", "手探り", "迷")),
    # 「興味津々」はここに入れない: ラベルは eager（興味）、高さは `_STRONG` の「津々」が担う。
    # ラベル（何を感じているか）と強度（どれくらいか）を混ぜないほうが表が壊れにくい。
    ("happy",    ("うれし", "嬉し", "たのし", "楽し", "わくわく", "ワクワク",
                  "満足", "すっきり", "さっぱり", "前進", "ほほえまし", "ほっこり")),
    ("eager",    ("期待", "前向き", "興味", "好奇心", "手伝", "協力", "役に立",
                  "雑談したい", "引き受け", "了承", "提案", "気になる", "気になっている")),
    ("warm",     ("親しみ", "気遣", "気づか", "やさし", "優し", "共感",
                  "ほんわか", "ふんわり", "気さく")),
    ("relieved", ("ほっと", "安心")),
    ("watching", ("様子見", "待機", "了解", "受け身", "見守")),
    ("calm",     ("落ち着", "穏やか", "冷静", "静か", "のんびり", "まったり", "しっとり",
                  "しみじみ", "柔軟", "素直", "ふつう", "普通", "集中", "納得", "名残惜し")),
)

# 強度の修飾語（「ちょっと戸惑い」は「戸惑い」より弱い）
_WEAK = ("ちょっと", "少し", "ほんの", "軽い", "軽く", "やや", "ほんのり")
_STRONG = ("津々", "とても", "かなり", "すごく", "めちゃ", "非常に")


def classify(text: str) -> tuple[str, float]:
    """感情の自由記述 → `(ラベル, 強度0.5〜1.3)`。未知語は `(NEUTRAL, 1.0)`。

    LLM の出力なので壊れた文字列（実データに Devanagari 混入の例あり）も来る。
    **決して例外を出さない**。
    """
    if not text:
        return NEUTRAL, 1.0
    s = str(text)
    intensity = 1.0
    if any(w in s for w in _WEAK):
        intensity = 0.6
    elif any(w in s for w in _STRONG):
        intensity = 1.3
    for label, words in _KEYWORDS:
        if any(w in s for w in words):
            return label, intensity
    return NEUTRAL, 1.0


@dataclass(frozen=True)
class Profile:
    """感情ごとの動きの味付け。**すべて控えめ**（実出力が低覚醒に偏っているため）。"""

    sway: float = 1.0          # 揺れの大きさ倍率
    speed: float = 1.0         # 揺れの速さ倍率
    pitch_bias: float = 0.0    # 上下の基準（-=うつむく / +=顔を上げる）-1..1
    tilt_bias: float = 0.0     # 首の傾きの基準 -1..1
    blink: float = 1.0         # まばたき頻度倍率（大きいほど頻繁）
    glance: float = 1.0        # よそ見の間隔倍率（大きいほど落ち着く）
    brow: float = 0.0          # 眉の上下 -1..1（-=下がる）
    brow_form: float = 0.0     # 眉の形 -1..1（-=困り眉 / +=怒り眉。**符号はモデル依存**）
    eye_smile: float = 0.0     # 目を細める 0..1
    cheek: float = 0.0         # 頬の赤み 0..1
    mouth_form: float = 0.35   # 口角の基準（既定 0.35＝わずかな微笑み）


# 振れ幅の方針（**改訂 2026-08-14・ユーザ指摘「差が分からない。2〜3倍でいい」**）:
# 初版は実出力が低覚醒に偏っていることを理由に控えめ（±1.5倍程度）にしたが、実機で見ると
# 感情の区別が付かなかった。**見て分かることの方が優先**なので 2〜3倍まで開く。
# 揺れは `live2d_view` で tanh 飽和させてあるので、大きい値を入れても首が可動域の端に
# 張り付いて固まることはない（線形にクランプすると「振り切ったまま」になり不自然）。
PROFILES: dict[str, Profile] = {
    # 既定。今いちばん多く出るラベル（落ち着き26 / 穏やか21）
    "calm":     Profile(),
    # ほっとした(33)＝最頻。ゆるむ方向
    "relieved": Profile(sway=0.8, speed=0.75, brow=0.15, eye_smile=0.35, mouth_form=0.5,
                        glance=1.3),
    # 親しみ・気遣い。柔らかく、相手を見る時間が長い
    "warm":     Profile(sway=0.9, speed=0.85, eye_smile=0.55, cheek=0.35, mouth_form=0.6,
                        glance=1.5),
    # 喜 → **露骨に大きく・速く**（ユーザ指定 2026-08-14）
    "happy":    Profile(sway=2.8, speed=2.5, pitch_bias=0.30, blink=1.6,
                        eye_smile=0.60, cheek=0.40, mouth_form=0.85, glance=0.45),
    # 期待・興味。happy の一歩手前
    "eager":    Profile(sway=1.8, speed=1.8, pitch_bias=0.18, brow=0.30,
                        eye_smile=0.25, mouth_form=0.55, glance=0.6),
    # 申し訳なさ・反省 → うつむく（哀ほどではない）
    "sorry":    Profile(sway=0.4, speed=0.6, pitch_bias=-0.70, tilt_bias=-0.20,
                        blink=0.8, brow=-0.55, brow_form=-0.5, mouth_form=0.0, glance=1.4),
    # 心配・不安 → **そわそわして落ち着かない**（ユーザ指摘 2026-08-14: 遅くするより速く）。
    # 振れ幅は小さいまま速さとよそ見だけ上げる＝「小刻みに落ち着きなく動く」形にする。
    "worried":  Profile(sway=0.85, speed=1.9, pitch_bias=-0.30, blink=1.7,
                        brow=-0.40, brow_form=-0.45, mouth_form=0.05, glance=0.35),
    # 戸惑い・困惑 → **首をかしげる**。動きはゆっくり
    "puzzled":  Profile(sway=0.8, speed=0.8, pitch_bias=-0.08, tilt_bias=0.85,
                        brow=-0.25, brow_form=-0.35, mouth_form=0.15, glance=0.7),
    # 様子見・待機。**ほぼ静止**して相手を見ている
    "watching": Profile(sway=0.3, speed=0.55, blink=0.85, glance=2.2, mouth_form=0.3),
    # 怒 → 鋭く速い。**揺れは喜びより小さく保つ**（大きく揺らすと怒りでなく興奮に見える）。
    # よそ見をせず相手を見据える（glance を長く）／まばたきを減らす。
    # まばたきは**増やす**（ユーザ指定 2026-08-14: 減らすと据わった目になり不自然）。
    "angry":    Profile(sway=1.6, speed=2.2, pitch_bias=-0.15, tilt_bias=0.0,
                        blink=1.8, glance=2.0, brow=-0.85, brow_form=0.9,
                        cheek=0.20, mouth_form=-0.6),
    # 哀 → **いちばん深くうつむき、いちばん鈍い**
    "sad":      Profile(sway=0.25, speed=0.45, pitch_bias=-0.95, tilt_bias=-0.18,
                        blink=1.1, glance=1.3, brow=-0.75, brow_form=-0.85,
                        mouth_form=-0.4),
}

# surprise（0-100・`PredictionState.surprise`）の中立値。これを下回る分は効かせない。
NEUTRAL_SURPRISE = 20


@dataclass(frozen=True)
class Motion:
    """最終的に Live2D へ渡す動きの係数（感情 × 強度 × surprise の合成結果）。"""

    sway: float
    speed: float
    pitch_bias: float
    tilt_bias: float
    blink: float
    glance: float
    brow: float
    brow_form: float
    eye_smile: float
    cheek: float
    mouth_form: float


def resolve(emotion_text: str, surprise: int) -> Motion:
    """感情の自由記述 + surprise → 動きの係数。

    **surprise は表示専用**（KIKAKUSHO §6「予測誤差が大きいほど動きを激しく」）。
    ここで読むだけで、書き戻しも発話判定への関与も一切しない
    ＝ CLAUDE.md の中核原理「surprise を数値で発話の可否に使わない」を侵さない。
    """
    label, intensity = classify(emotion_text)
    p = PROFILES.get(label, PROFILES[NEUTRAL])
    # 強度: 1.0 からの差分に intensity を掛ける（弱い修飾語なら既定寄りに戻る）
    def lerp(v: float, base: float = 0.0) -> float:
        return base + (v - base) * intensity

    s = max(0.0, min(1.0, (surprise - NEUTRAL_SURPRISE) / (100.0 - NEUTRAL_SURPRISE)))
    return Motion(
        sway=lerp(p.sway, 1.0) * (1.0 + 0.9 * s),      # 大きく
        speed=lerp(p.speed, 1.0) * (1.0 + 0.6 * s),    # 速く
        pitch_bias=lerp(p.pitch_bias),
        tilt_bias=lerp(p.tilt_bias),
        blink=lerp(p.blink, 1.0) * (1.0 + 0.5 * s),    # 頻度増
        glance=lerp(p.glance, 1.0) / (1.0 + 1.2 * s),  # よそ見の間隔が縮む＝頻度増
        brow=lerp(p.brow) + 0.35 * s,                  # 驚くと眉が上がる
        brow_form=lerp(p.brow_form),
        eye_smile=lerp(p.eye_smile) * (1.0 - 0.6 * s),  # 驚くと目が開く
        cheek=lerp(p.cheek),
        mouth_form=lerp(p.mouth_form, 0.35),
    )
