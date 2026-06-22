"""増分日本語文分割。

token stream の delta を逐次受け、文末境界が出た所で完成文を返す（バッファは内部保持）。
v1 は分割が generate_stream に癒着していた → 単一責務で独立させテスト可能にする。

境界: 「。！？」全角 + ascii「!?」+ 改行。ascii「.」は境界にしない（"3.14" を割らない）。
高精度化が要れば ja_sentence_segmenter 等を検討（ただし全文前提で増分には不向き）。
"""
from __future__ import annotations

_BOUNDARIES = "。！？!?\n"


class JapaneseSentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        """delta を流し込み、確定した完成文のリストを返す（0件以上）。"""
        out: list[str] = []
        for ch in delta:
            self._buf += ch
            if ch in _BOUNDARIES:
                s = self._buf.strip()
                if s:
                    out.append(s)
                self._buf = ""
        return out

    def flush(self) -> list[str]:
        """stream 終了時に残ったバッファを完成文として返す。"""
        s = self._buf.strip()
        self._buf = ""
        return [s] if s else []
