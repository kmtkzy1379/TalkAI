"""増分日本語文分割。

token stream の delta を逐次受け、区切りが出た所で完成した節・文を返す（バッファは内部保持）。
v1 は分割が generate_stream に癒着していた → 単一責務で独立させテスト可能にする。

境界: 「、。！？」全角 + ascii「!?」+ 改行。ascii「.」「,」は境界にしない（"3.14" "1,000" を割らない）。
**「、」でも区切る**（ユーザ裁定 2026-09-22「今まではユーザーしゃべったら速攻黙ったけど不自然。、。までは
喋らせる」/ 2026-09-26「相手がまだ話したら、。まで話して再度割り込ませればいい」）。区切った1つずつが
TTS と再生の1項目になり、割り込まれた時は鳴っている項目を言い切ってから止まる（`AudioPlayQueue`）
＝「、」「。」まで話して譲る。
実測（2026-09-26・Irodori・イブの実際の応答12件）: 「、」の所の間は 文ごとに合成（文の中の間）0.52秒 →
節ごとに合成して続けて鳴らす 0.42秒（中央値）で長くならない。1項目目の合成は 303 → 248ms。
高精度化が要れば ja_sentence_segmenter 等を検討（ただし全文前提で増分には不向き）。

**続く記号は前の節にくっつける**（ユーザ 2026-09-29「！？単体だけ生成してもただの呼吸音として発話されて
しまう。かと言って！？を完全無視すると声に感情が乗らない。何とかして最後の文に記号をくっつけたい」）。
「！」「？」で終わった節は、次の文字が「！」「？」でないと分かるまで出さない（「マジで！？」を1つの節にする。
遅れは次のトークン1つ分）。「、」「。」の節は今までどおりすぐ出す（割り込みの単位は変わらない）。
くっつける先の無い記号だけの節（「そう。？」の「？」）は鳴らさない。
"""
from __future__ import annotations

_BOUNDARIES = "、。！？!?\n"
# この記号で終わった節は保留し、続く同じ種類の記号を後ろに足す（「！？」「？！」「！！」）
_TRAILING = "！？!?"
_SILENT = _BOUNDARIES + " \t\r　"


def _speakable(s: str) -> bool:
    """記号と空白以外の文字が1つでもあるか（記号だけの節は TTS に渡さない）。"""
    return any(c not in _SILENT for c in s)


class JapaneseSentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""
        self._held = ""  # 「！」「？」で終わった節（次の文字を見るまで出さない）

    def feed(self, delta: str) -> list[str]:
        """delta を流し込み、確定した節・文のリストを返す（0件以上）。"""
        out: list[str] = []
        for ch in delta:
            if self._held:
                if ch in _TRAILING:
                    self._held += ch
                    continue
                out.append(self._held)
                self._held = ""
            self._buf += ch
            if ch in _BOUNDARIES:
                s = self._buf.strip()
                self._buf = ""
                if not _speakable(s):
                    continue
                if ch in _TRAILING:
                    self._held = s
                else:
                    out.append(s)
        return out

    def flush(self) -> list[str]:
        """stream 終了時に残った節（保留中の節とバッファ）を返す。"""
        out = [self._held] if self._held else []
        self._held = ""
        s = self._buf.strip()
        self._buf = ""
        if _speakable(s):
            out.append(s)
        return out
