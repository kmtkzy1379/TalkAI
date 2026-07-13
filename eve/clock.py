"""二重タイムスタンプ。

- 壁時計(UTC ISO-8601): 表示・永続化用。再起動をまたいで比較できる。
- monotonic 秒: 期間計算用。NTP 補正や DST のジャンプに影響されない（タスクの
  deadline 計算はこちらで行う＝二重発火・巻き戻りを防ぐ。COMPONENT_LOGIC J-1 の時刻パス）。

過去参照防止(T6)の接地として、文脈組立時に経過秒を `humanize()` で相対表現にする。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone


def now_iso() -> str:
    """UTC の ISO-8601 文字列（永続化・表示用）。"""
    return datetime.now(timezone.utc).isoformat()


def now_mono() -> float:
    """単調増加秒（期間計算用。壁時計のジャンプに影響されない）。"""
    return time.monotonic()


@dataclass(frozen=True)
class Stamp:
    """壁時計と monotonic の対。発生時刻の記録に使う。"""

    iso: str
    mono: float

    @classmethod
    def now(cls) -> "Stamp":
        return cls(iso=now_iso(), mono=now_mono())


def elapsed_mono(then: Stamp, now: Stamp | None = None) -> float:
    """同一プロセス内の経過秒（monotonic 差）。直近ターン等の in-session 要素用。"""
    now = now or Stamp.now()
    return max(0.0, now.mono - then.mono)


def elapsed_wall(then_iso: str, now_iso_str: str | None = None) -> float:
    """壁時計 ISO 差の経過秒。永続化済み要素（RAG チャンク・会話ターン）用。

    防御的: tz-naive な ISO（legacy/手編集/外部由来の timestamp）が来ても
    UTC とみなして計算する。naive と aware の減算で TypeError を投げると
    ContextAssembler.assemble がターンごと落ちる（応答が出ない）ため、ここで吸収する。
    """
    try:
        a = datetime.fromisoformat(then_iso)
        if a.tzinfo is None:
            a = a.replace(tzinfo=timezone.utc)
        if now_iso_str:
            b = datetime.fromisoformat(now_iso_str)
            if b.tzinfo is None:
                b = b.replace(tzinfo=timezone.utc)
        else:
            b = datetime.now(timezone.utc)
        return max(0.0, (b - a).total_seconds())
    except (ValueError, TypeError):
        return 0.0  # 壊れた timestamp は「たった今」扱い（落とさない）


def humanize(delta_seconds: float) -> str:
    """経過秒を「3分前」等の相対表現に。文脈の時間接地に使う。"""
    d = max(0.0, delta_seconds)
    if d < 10:
        return "たった今"
    if d < 60:
        return f"{int(d)}秒前"
    if d < 3600:
        return f"{int(d // 60)}分前"
    if d < 86400:
        return f"{int(d // 3600)}時間前"
    return f"{int(d // 86400)}日前"


def humanize_eta(seconds: float) -> str:
    """残り秒を「あと約30秒」等の丸め表現に（`humanize` の未来方向・予約タスクの文脈接地用）。

    秒精度を出さない（10秒単位に丸める）＝応答LLMが復唱しても機械的にならない。
    """
    s = max(0.0, seconds)
    if s <= 10:
        return "まもなく"
    if s < 60:
        return f"あと約{int(s // 10) * 10}秒"
    if s < 3600:
        return f"あと約{int(s // 60)}分"
    return f"あと約{int(s // 3600)}時間"
