"""VisionState — F6 の loop 所有 ephemeral 状態（単一書込・同期読み・ロック不要・§9）。

- ring: 直近 N 枚の**全キャプチャ**（変化フレームだけでなく＝変化前アンカーを含む・A8）。
  書込者は唯一 `on_frame`（capture スレッドから call_soon_threadsafe で loop 上に届く・means-1）。
- latest_vision: 最新ナレーション or 「視認不可」正直マーカ（A11）。書込者は唯一 VlmWorker。
  読み手は ResponseOrchestrator（# 画面 注入）/ SpeechDecider（発話判定の入力）。

**読み手で鮮度の意味が違う（層分離）**:
- `fresh_vision`(厳格・TTL のみ) = 自発発話系（発話判定LLM / 自発発話の応答文脈・RAGクエリ）。
  「今さっき変化した画面」しか渡らない＝静止中は画面ブロックが消える。放置中に同じ画面要素へ
  引っ張られて同話題を繰り返す固執（実機実測: 判定136回中135回に画面が入りっぱなし）を構造で断つ。
- `vision_for_user`(据え置き可) = ユーザ発話への応答。pHash が「変化なし」と言う限り古い実況も
  今なお正しいので有効とし、「N秒前・以降変化なし」と正直に付す（捏造しない・A11 と同じ姿勢）。
  これが無いと静止画面で「今何が見える?」に答えられなくなる（narrator が変化なしでも主要表示を
  一言入れている目的そのもの）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from ..clock import now_mono
from .types import Frame

# capture 生存とみなす最終フレームからの猶予[秒]。capture は既定 2fps なので数フレーム分。
# capture が死んだ/止まった後に「以降変化なし」と言い張らないためのガード。
CAPTURE_ALIVE_SEC = 3.0


class VisionState:
    def __init__(self, ring_max: int = 6) -> None:
        self.ring: deque[Frame] = deque(maxlen=ring_max)  # 自動 drop-oldest（上限=レイテンシ bound）
        self.latest_vision: Optional[str] = None
        self.latest_vision_mono: float = 0.0  # latest_vision をセットした時刻（鮮度判定用）
        self.latest_is_narration: bool = False  # 正直マーカ(A11)でなく実況か（据え置き可否の判定用）
        self.last_frame_mono: float = 0.0  # 最後にフレームが届いた時刻（capture 生存確認）
        self.last_change_mono: float = 0.0  # 最後に「変化あり」フレームが届いた時刻

    def add_frame(self, frame: Frame, changed: bool = False, now: Optional[float] = None) -> None:
        """全キャプチャを積む（変化前アンカー保持・A8）。唯一の ring 書込点。

        changed は「pHash ゲートが変化と判定したか」。実況の据え置き可否（変化していないなら
        古い実況も今なお正しい）を決めるのに使う。
        """
        self.ring.append(frame)
        n = now if now is not None else now_mono()
        self.last_frame_mono = n
        if changed:
            self.last_change_mono = n

    def snapshot(self, k: int) -> list[Frame]:
        """直近 k 枚を value-copy（list 新規・Frame は frozen 不変）。await 前に同期で呼ぶ（A2）。"""
        return list(self.ring)[-k:]

    def set_latest(self, text: str, mono: Optional[float] = None, *, narration: bool = True) -> None:
        """最新ナレーションを時刻付きでセット（唯一の latest_vision 書込点＝VlmWorker）。

        narration=False は A11 の正直マーカ（視認不可）。据え置き対象にしない（「30秒前は取得
        できなかった。以降変化なし」は情報として無意味かつ紛らわしい）。
        """
        self.latest_vision = text
        self.latest_vision_mono = mono if mono is not None else now_mono()
        self.latest_is_narration = narration

    def fresh_vision(self, ttl: float, now: Optional[float] = None) -> Optional[str]:
        """ttl 秒以内に更新された latest_vision のみ返す。古ければ None（明らか過去を参照させない）。"""
        if self.latest_vision is None:
            return None
        n = now if now is not None else now_mono()
        if (n - self.latest_vision_mono) > ttl:
            return None  # 古すぎ → 画面情報なし扱い（応答LLMは正直に「今は見えない」と言える）
        return self.latest_vision

    def vision_for_user(
        self, ttl: float, static_max: float, now: Optional[float] = None
    ) -> Optional[str]:
        """ユーザ発話への応答用（据え置き可）。TTL 内はそのまま、以降は「変化していない間だけ」有効。

        据え置きの条件（全部満たす時のみ）: 実況である（正直マーカでない）/ capture が生きている /
        実況より後に変化フレームが来ていない / 据え置き上限 static_max 以内。
        上限を置くのは、pHash 閾値未満の微小変化が積み上がって実況が実態とズレ得るため。
        """
        if self.latest_vision is None:
            return None
        n = now if now is not None else now_mono()
        age = n - self.latest_vision_mono
        if age <= ttl:
            return self.latest_vision
        if not self.latest_is_narration:
            return None  # 視認不可マーカは据え置かない
        if (n - self.last_frame_mono) > CAPTURE_ALIVE_SEC:
            return None  # capture が止まっている → 「以降変化なし」を保証できない
        if self.last_change_mono > self.latest_vision_mono:
            return None  # 実況の後に画面が変わった（まだ実況が追いついていない）
        if age > static_max:
            return None
        return f"{self.latest_vision}（{int(age)}秒前の画面。以降ずっと変化なし）"
