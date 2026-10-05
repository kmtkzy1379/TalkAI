"""応答の書き出しのフィラーを応答LLM に言わせる system ブロック（S4）。

文は `fillers.yaml` の `opener_prompt`（ユーザ 2026-09-28 のルナ検証で使い、2026-09-29 に採用した構成）:
フィラー表の初期案（v1・非依存も依存も混ぜて全語に意味）＋後から足した語の意味（表 v2）＋S8 の場面を1つにし、
「一例として使う・特定の語に偏らない・置かない場面も大事」を添えたもの。**ユーザ発話への応答のときだけ** system に入れる。
以前（2026-09-25〜28）は表 v2 の応答依存フィラーと注記から組み立てていた。

語は**足さない・外さない**（ユーザ指定）。例外は「ユーザー名」の呼びかけで、`user_name` が未設定の間は
語そのものが決まらないので `{user_name_line}` の行ごと消す。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

HEADING = "# 応答の書き出し（フィラー）"
USER_NAME_LINE = "{user_name_line}"


def reply_opener_block(table_path: Path) -> Optional[str]:
    """`fillers.yaml` の opener_prompt（無ければ None＝何も足さない）。"""
    import yaml

    cfg = yaml.safe_load(Path(table_path).read_text(encoding="utf-8")) or {}
    text = str(cfg.get("opener_prompt") or "").strip()
    if not text:
        return None
    user_name = str(cfg.get("user_name") or "").strip()
    lines = []
    for ln in text.splitlines():
        if ln.strip() == USER_NAME_LINE:
            if user_name:
                lines.append(f"- {user_name}: 呼びかけ")
            continue  # 名前が未設定なら行ごと消す
        lines.append(ln)
    return "\n".join(lines)
