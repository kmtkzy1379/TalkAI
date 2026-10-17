"""API 使用量と概算費用の記録（**役割ごと**に、1呼び出し1行の JSONL）。

## なぜ要るか
「どの設定を変えると請求がどう動くか」が分からないまま knob だけ増やしても、ユーザは
判断できない（高級設定の画面認識コスト群がまさにこれ）。役割ごとの実測トークン数が
残れば、`VLM_TARGET_FPS` を上げた時に何が増えたかを**数字で**言える。

## 価格はどこから来るか
**私の記憶からは書かない。**`litellm.model_cost`（実測 3019 エントリ）を引く。
litellm のバージョンに紐づくので古くなり得る。よって:
- 記録には**トークン数を必ず残す**（価格が変わっても後から再計算できる）
- 価格が引けないモデルは `cost_usd=None` で残す（0 と嘘をつかない）
- `PRICE_OVERRIDE`（`.env` の `USAGE_PRICE_OVERRIDE`）で手動修正できる

## 正確さの正直な区別
- `exact`  = provider が返した usage をそのまま使った（非ストリーミング）
- `approx` = ストリーミングでトークン数を数え直した（`litellm.token_counter`）

**ストリーミングでリクエスト引数を足さない**方針にした（`stream_options` を渡すと
provider によっては弾かれ、応答経路そのものを壊す）。本番の応答は全部ストリーミングなので
そこは `approx` になるが、桁を見誤らない程度には合う。

## スレッド契約
`record()` は loop 上から同期で呼ばれる想定（O(1)・追記のみ）。ファイル書き込みは
`to_thread` に逃がさず、行が短いのでそのまま書く（1行 200 バイト程度）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_ROOT / "usage.jsonl"


@dataclass
class Totals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0        # 価格が引けなかった呼び出し（0 と混ぜない）

    def add(self, p: int, c: int, cost: Optional[float]) -> None:
        self.calls += 1
        self.prompt_tokens += p
        self.completion_tokens += c
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += cost


def price_of(model: str) -> Optional[tuple]:
    """(入力単価, 出力単価) を USD/token で返す。引けなければ None（0 にしない）。"""
    override = _overrides().get(model)
    if override is not None:
        return override
    try:
        import litellm

        table = getattr(litellm, "model_cost", {}) or {}
    except Exception:
        return None
    # litellm の表は "gpt-4o" のような裸の名前で持っていることがある（"openai/gpt-4o" は別名）
    entry = table.get(model) or table.get(model.split("/", 1)[-1])
    if not entry:
        return None
    pin = entry.get("input_cost_per_token")
    pout = entry.get("output_cost_per_token")
    if pin is None and pout is None:
        return None
    return (float(pin or 0.0), float(pout or 0.0))


def _overrides() -> dict:
    """`USAGE_PRICE_OVERRIDE="model=in:out,model2=in:out"`（USD/token）。

    litellm の価格表が古い / 未収載のモデルを使う時の逃げ道。**桁を間違えやすい**ので
    壊れた指定は黙って無視する（起動を止めない）。
    """
    raw = os.getenv("USAGE_PRICE_OVERRIDE", "").strip()
    if not raw:
        return {}
    out: dict = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        model, _, pair = part.partition("=")
        try:
            pin, _, pout = pair.partition(":")
            out[model.strip()] = (float(pin), float(pout))
        except ValueError:
            logger.warning("USAGE_PRICE_OVERRIDE の書式が不正（無視）: %s", part.strip()[:40])
    return out


class UsageMeter:
    """役割ごとの使用量を数え、JSONL に1行ずつ残す。**例外を外に出さない**。"""

    def __init__(self, path: Optional[Path] = None, *, enabled: bool = True,
                 tag: str = "") -> None:
        # 相対パスはリポジトリ基準（会話・記憶・タスクと同じ。以前は起動した場所＝cwd 基準だった）
        p = Path(path) if path else DEFAULT_FILE
        self.path = p if p.is_absolute() else REPO_ROOT / p
        self.enabled = enabled
        self.tag = tag                     # 計測ラン名（検証と実運用を混ぜないため）
        self.by_role: dict[str, Totals] = {}
        self.by_model: dict[str, Totals] = {}
        self.total = Totals()
        self._lock = threading.Lock()      # STT/埋め込みは executor から来ることがある

    # --- 記録 -------------------------------------------------------------

    def record(self, *, role: str, model: str, prompt_tokens: int,
               completion_tokens: int, accuracy: str = "exact",
               extra: Optional[dict] = None) -> Optional[float]:
        """1呼び出しぶんを記録し、概算費用（USD・引けなければ None）を返す。"""
        if not self.enabled:
            return None
        try:
            p = max(0, int(prompt_tokens or 0))
            c = max(0, int(completion_tokens or 0))
            price = price_of(model)
            cost = None if price is None else p * price[0] + c * price[1]
            with self._lock:
                self.by_role.setdefault(role, Totals()).add(p, c, cost)
                self.by_model.setdefault(model, Totals()).add(p, c, cost)
                self.total.add(p, c, cost)
                self._append({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "tag": self.tag,
                    "role": role,
                    "model": model,
                    "prompt_tokens": p,
                    "completion_tokens": c,
                    "cost_usd": cost,
                    "accuracy": accuracy,     # exact | approx
                    **(extra or {}),
                })
            return cost
        except Exception:
            logger.exception("使用量の記録に失敗（本体は継続）")
            return None

    def record_cost(self, *, role: str, model: str, cost_usd: float,
                    extra: Optional[dict] = None) -> Optional[float]:
        """トークンでなく**件数で課金される API**（X API の投稿1件 $0.005 等・`eve/xfeed/`）の1回ぶん。

        accuracy は `fixed`（単価×件数で決まる額）。`tools/usage_report.py --recalc` はこの行の額を当て直さない
        （トークンが 0 なので、価格表で計算し直すと 0 か「価格なし」になってしまう）。
        """
        if not self.enabled:
            return None
        try:
            cost = float(cost_usd)
            with self._lock:
                self.by_role.setdefault(role, Totals()).add(0, 0, cost)
                self.by_model.setdefault(model, Totals()).add(0, 0, cost)
                self.total.add(0, 0, cost)
                self._append({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "tag": self.tag,
                    "role": role,
                    "model": model,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": cost,
                    "accuracy": "fixed",
                    **(extra or {}),
                })
            return cost
        except Exception:
            logger.exception("使用量の記録に失敗（本体は継続）")
            return None

    def record_response(self, *, role: str, model: str, response) -> Optional[float]:
        """provider が返した usage をそのまま使う（非ストリーミング＝`exact`）。"""
        usage = getattr(response, "usage", None) or {}
        get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
        return self.record(
            role=role, model=model,
            prompt_tokens=get("prompt_tokens", 0) or 0,
            completion_tokens=get("completion_tokens", 0) or 0,
            accuracy="exact",
        )

    def record_stream(self, *, role: str, model: str, messages: list, output: str,
                      tools: Optional[list] = None) -> Optional[float]:
        """ストリーミングぶんを数え直す（`approx`）。

        リクエストに `stream_options` を足さないのは、provider によっては弾かれて
        **応答経路そのものが落ちる**ため。正確さより本体の生存を優先する。
        """
        return self.record(
            role=role, model=model,
            prompt_tokens=count_tokens(model, messages, tools=tools),
            completion_tokens=count_tokens(model, [{"role": "assistant", "content": output}]),
            accuracy="approx",
        )

    def _append(self, row: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            logger.debug("使用量ログの書き込みに失敗（無視）", exc_info=True)

    # --- 読み出し ---------------------------------------------------------

    def summary(self) -> dict:
        with self._lock:
            return {
                "total": vars(self.total).copy(),
                "by_role": {k: vars(v).copy() for k, v in self.by_role.items()},
                "by_model": {k: vars(v).copy() for k, v in self.by_model.items()},
            }

    def format_summary(self, *, per_hour: Optional[float] = None) -> str:
        """人が読む形。**未価格の呼び出し数を必ず出す**（0 と混ぜない）。"""
        s = self.summary()
        lines = ["役割別の使用量（概算費用は litellm の価格表に基づく）",
                 f"{'役割':<16}{'回数':>5}{'入力tok':>10}{'出力tok':>10}{'USD':>12}"]
        for role, t in sorted(s["by_role"].items(), key=lambda kv: -kv[1]["cost_usd"]):
            lines.append(f"{role:<16}{t['calls']:>5}{t['prompt_tokens']:>10}"
                         f"{t['completion_tokens']:>10}{t['cost_usd']:>12.5f}")
        t = s["total"]
        lines.append(f"{'合計':<16}{t['calls']:>5}{t['prompt_tokens']:>10}"
                     f"{t['completion_tokens']:>10}{t['cost_usd']:>12.5f}")
        if t["unpriced_calls"]:
            lines.append(f"※ 価格を引けなかった呼び出し: {t['unpriced_calls']} 件"
                         "（USD には含まれていません）")
        if per_hour:
            lines.append(f"※ この計測は {per_hour:.1f} 分。1時間あたり概算 "
                         f"{t['cost_usd'] * 60.0 / per_hour:.4f} USD")
        return "\n".join(lines)


def count_tokens(model: str, messages: list, tools: Optional[list] = None) -> int:
    """トークン数を数える。数えられなければ**文字数からの粗い近似**にする。

    近似でも桁は合う（日本語は概ね 1トークン ≒ 1〜1.5 文字）。0 を返して
    「使っていない」ように見せる方が有害。
    """
    try:
        from litellm import token_counter

        n = token_counter(model=model, messages=messages)
        if tools:
            n += token_counter(model=model,
                               messages=[{"role": "system", "content": json.dumps(tools)}])
        return int(n)
    except Exception:
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        return int(chars / 1.2) + 4


# 既定メーター（`ModelRegistry` が参照する。**None なら何も記録しない**＝挙動不変）。
_METER: Optional[UsageMeter] = None


def meter() -> Optional[UsageMeter]:
    return _METER


def install(meter_obj: Optional[UsageMeter]) -> None:
    """メーターを差し込む/外す。**既定は None**（ヘッドレスも本番も、明示するまで無記録）。"""
    global _METER
    _METER = meter_obj


def enabled_by_env() -> bool:
    return os.getenv("USAGE_LOG_ENABLED", "0") == "1"
