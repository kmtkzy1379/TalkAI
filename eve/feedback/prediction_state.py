"""PredictionState — フィードバック/予測ループの loop 所有状態（単一書込・同期読み・ロックなし）。

中核原理(surprise)の F4 時点の住処。FeedbackLLM（サイドカー worker）だけが書き、
ResponseOrchestrator（last_feedback の注入）と将来 F5（should_speak が surprise を読む）が
同期で読む。PIPELINE_DESIGN.md §9 の「loop 所有・単一書込者」契約に従う。

- last_prediction: 前回 feedback が立てた「次に何が起きるか」。次 run が現実と比較して
  予測差を出す（FEP 繰越）。feedback 内部用。
- last_feedback: 応答LLM へ注入する直近フィードバックの短い1ブロック（最新1件のみ・公開読み）。
- watermark: 最後に feedback 済みの会話位置(iso・排他)。これより新しいターンが未処理＝
  「フィードバックしてない会話＝記憶喪失」を作らないための指標（B5 の span カバーで使う）。
- surprise: 直近の予測差(0-100)。**メソッド `note_feedback_surprise` でのみ書く**。
  F5 で VLM が第2生産者になっても `note_vlm_surprise` を足し `surprise` プロパティを
  合成読みに変えるだけで済む（呼出側不変＝多生産者化で書き直さない）。
"""
from __future__ import annotations

from typing import Optional

from ..clock import now_mono

# 予測対象が無い/不明な時の中立 surprise（cold start・初期値）。
# 0 は「完璧に予測できた」を意味し沈黙を不当に誘発するため使わない（surprise を装飾化させない）。
NEUTRAL_SURPRISE = 20


def clamp_surprise(value: object) -> int:
    """予測差を 0-100 の int にクランプ（壊れた値は中立に倒す・例外を出さない）。"""
    try:
        return max(0, min(100, int(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return NEUTRAL_SURPRISE


class PredictionState:
    def __init__(self) -> None:
        self.last_prediction: Optional[str] = None
        self.last_feedback: Optional[str] = None
        self.watermark: Optional[str] = None  # iso（排他）。これより後のターンが未 feedback
        # surprise は2生産者（F4 feedback / F6 VLM screen-diff）。各 source を ts 付きで保持し
        # **most-recent-source-wins** で読む（max でない＝過去の画面フラッシュが高値固定する
        # staleness/数値ゲート化を避ける。surprise は「今どれだけ意外か」）。
        self._fb_surprise: Optional[int] = None
        self._fb_ts: float = 0.0
        self._vlm_surprise: Optional[int] = None
        self._vlm_ts: float = 0.0

    @property
    def surprise(self) -> int:
        """直近の予測差(0-100)。feedback / VLM のうち**最後に書いた source の値**を返す。

        両 source 未書込なら NEUTRAL。max でなく most-recent-wins なのは、古い高 surprise が
        固定化して「今」を歪めるのを防ぐため（T2 death-detection は live source 反転で維持）。
        """
        fb, vlm = self._fb_surprise, self._vlm_surprise
        if fb is None and vlm is None:
            return NEUTRAL_SURPRISE
        if vlm is None:
            return fb  # type: ignore[return-value]
        if fb is None:
            return vlm
        return vlm if self._vlm_ts >= self._fb_ts else fb

    def note_feedback_surprise(self, diff: object) -> None:
        """FeedbackLLM(F4) からの予測差を反映（feedback source・most-recent 比較用に ts 記録）。"""
        self._fb_surprise = clamp_surprise(diff)
        self._fb_ts = now_mono()

    def note_vlm_surprise(self, diff: object) -> None:
        """VLM(F6) screen-diff からの予測差を反映（第2生産者・most-recent 比較用に ts 記録）。"""
        self._vlm_surprise = clamp_surprise(diff)
        self._vlm_ts = now_mono()
