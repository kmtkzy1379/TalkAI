"""`.env` の読み書き（設定タブの保存先）。

## なぜ専用モジュールにするか
`.env` は**ユーザが手で書いたファイル**で、コメントも並び順も本人の資産。設定タブが
これを丸ごと書き直すと、コメントが消えたり順序が変わったりして「勝手に荒らされた」
状態になる。ここは**行を保存したまま値だけ差し替える**ことに全責任を持つ。

## 守ること（どれも壊れ方が静か）
- **BOM を付けない**: `eve/config.py` の最小パーサは `utf-8` で読むので、BOM 付きだと
  1行目のキーが `﻿OPENAI_API_KEY` になり**その1個だけ黙って効かなくなる**。
- **改行コードを保つ**: 現状の `.env` は CRLF。LF で書き直すと差分が全行になる。
- **引用符を付けない**: python-dotenv は引用符を外すが、dotenv 未導入時の最小パーサ
  （`config.py:23-28`）は**外さない**。付けると環境によって値が変わる。
- **`#` を含む値は拒否する**: python-dotenv は引用符なしの値の ` #` 以降をコメントとして
  落とすが、最小パーサは落とさない。同じ理由で環境によって値が変わる。
- **アトミックに置き換える**: 書き込み中に落ちても `.env` を壊さない（一時ファイル→`os.replace`）。
- **値をログに出さない**: API キーを扱うので、例外メッセージにも値を載せない。
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
# `eve/config.py` と**同じファイル**を見る（`EVE_ENV_FILE` で差し替え可）。
# ここがズレると「設定タブで保存したのに読み込まれない」が起きる。
ENV_PATH = Path(os.getenv("EVE_ENV_FILE") or (REPO_ROOT / ".env"))
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"


class InvalidValue(ValueError):
    """`.env` に安全に書けない値。**値そのものはメッセージに載せない**。"""


def validate_value(key: str, value: str) -> None:
    """書けない値を弾く。理由は形だけで言い、中身は言わない（キーが漏れる）。"""
    if "\n" in value or "\r" in value:
        raise InvalidValue(f"{key}: 改行を含む値は保存できません")
    if "#" in value:
        # dotenv 有無で解釈が変わる（片方はコメント扱いで切り落とす）＝環境で値が変わる。
        raise InvalidValue(f"{key}: 「#」を含む値は保存できません")
    if value != value.strip():
        raise InvalidValue(f"{key}: 前後の空白は保存できません")
    if "=" in key or not key or key != key.strip():
        raise InvalidValue(f"キー名が不正です: {key!r}")


def _split_line(raw: str) -> tuple[str, str] | None:
    """`KEY=VALUE` 行なら (key, value) を返す。コメント/空行は None。"""
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    return key, val.strip()


def read_env(path: Path | None = None) -> dict[str, str]:
    """`.env` を読む（無ければ空）。**`os.environ` は見ない**＝ファイルの中身そのもの。"""
    p = path or ENV_PATH
    out: dict[str, str] = {}
    for raw in _read_raw(p).splitlines():
        kv = _split_line(raw)
        if kv is not None:
            out[kv[0]] = kv[1]
    return out


def _read_raw(p: Path) -> str:
    """**改行を変換せずに**読む。

    `Path.read_text()` は universal newlines なので CRLF が LF に化け、そのまま書き戻すと
    **ファイル全体が差分になる**（ユーザの `.env` は CRLF）。`newline=""` で素のまま読む。
    """
    if not p.exists():
        return ""
    with open(p, encoding="utf-8-sig", newline="") as f:
        return f.read()


def _newline_of(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def write_env(updates: dict[str, str], path: Path | None = None) -> list[str]:
    """値を差し替えて保存し、**実際に変わったキー**を返す。

    既存行はその場で書き換え（コメント・順序・空行はそのまま）、新しいキーは末尾に足す。
    変更が無ければ**ファイルに触らない**（更新時刻を動かさない）。
    """
    p = path or ENV_PATH
    for k, v in updates.items():
        validate_value(k, v)

    original = _read_raw(p)
    nl = _newline_of(original)
    lines = original.split("\n")
    # split("\n") は CRLF の "\r" を各行末に残すので落としてから扱う（書き戻しは nl で統一）
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]
    trailing_blank = bool(lines) and lines[-1] == ""
    if trailing_blank:
        lines = lines[:-1]

    changed: list[str] = []
    seen: set[str] = set()
    for i, raw in enumerate(lines):
        kv = _split_line(raw)
        if kv is None or kv[0] not in updates:
            continue
        key, old = kv
        seen.add(key)
        new = updates[key]
        if old != new:
            lines[i] = f"{key}={new}"
            changed.append(key)

    added = [k for k in updates if k not in seen]
    if added:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# --- 設定タブから追加 ---")
        for k in added:
            lines.append(f"{k}={updates[k]}")
            changed.append(k)

    if not changed:
        return []

    text = nl.join(lines) + nl  # 末尾は必ず改行で終える
    _atomic_write(p, text)
    logger.info("設定を保存しました（%d 件）: %s", len(changed), ", ".join(sorted(changed)))
    return changed


def _atomic_write(p: Path, text: str) -> None:
    """一時ファイルへ書いてから置換する。**BOM は付けない / 改行は変換しない**。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".env.", suffix=".tmp")
    try:
        # newline="" ＝ Python に改行変換をさせない（text 側が既に正しい改行を持つ）
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)  # 同一ディレクトリなのでアトミック
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sync_example(keys: dict[str, str], path: Path | None = None) -> list[str]:
    """`.env.example` に**欠けているキーだけ**を既定値付きで足す（値の上書きはしない）。

    CLAUDE.md の「`.env.example` に必ず全フラグを明記する」を機械的に守るための口。
    **秘密は載せない**（呼び出し側が API キーを渡さない契約）。
    """
    p = path or ENV_EXAMPLE_PATH
    have = set(read_env(p))
    missing = {k: v for k, v in keys.items() if k not in have}
    if not missing:
        return []
    for k, v in missing.items():
        validate_value(k, v)
    original = _read_raw(p)
    nl = _newline_of(original)
    body = original.rstrip("\r\n")
    parts = [body] if body else []
    parts.append("")
    parts.append("# --- 追記（コードにあるのに未記載だったもの）---")
    parts += [f"{k}={v}" for k, v in missing.items()]
    _atomic_write(p, nl.join(parts) + nl)
    logger.info(".env.example に %d 件追記: %s", len(missing), ", ".join(missing))
    return list(missing)


# --- API キーの扱い（**値を画面に出さない**）--------------------------------

def key_status(env_key: str) -> tuple[str, str]:
    """API キーの状態を返す `(状態, 表示用の指紋)`。**生の値は返さない**。

    状態は "env"（OS の環境変数で設定済み＝`.env` より優先されるので画面から変えられない）
    / "file"（`.env` にある）/ "none"（未設定）。`eve/config.py` の `_load_dotenv` が
    `os.environ.setdefault` を使うため、**OS 側にあると `.env` を書いても効かない**。
    この区別を出さないと「保存したのに変わらない」で詰まる。
    """
    in_file = read_env().get(env_key, "")
    raw = os.environ.get(env_key, "")
    if raw and raw != in_file:
        return "env", fingerprint(raw)
    if in_file:
        return "file", fingerprint(in_file)
    return "none", ""


def fingerprint(value: str) -> str:
    """「どの鍵が入っているか」だけ分かる表示。末尾4文字と長さのみ。

    全体を出すと**イブ自身の画面認識が Gemini に送る**（`VLM_ENABLED=1` のとき）。
    配信中はキャプチャ除外を切る運用なので OBS にも乗る。読み返す用途はほぼ無いので、
    同一性の確認に足りる最小限だけを出す。
    """
    if not value:
        return ""
    return f"…{value[-4:]}・{len(value)}文字"
