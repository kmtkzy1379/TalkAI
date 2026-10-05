"""ペルソナ層（性格と口調だけ）。`response/style.py` が予告していた「別層」の実体。

## 層の分け方（混ぜると v1 の失敗を繰り返す）
- **ここ（persona.md）**: 性格・口調。人が書く。遺伝的なもの＝会話で変わらない前提。
- `response/style.py`: 読み上げ規律（記号を出さない等）。**ペルソナではない**。
- `eve/self_model/`: 好み・趣向。AI が会話から育てる。経験によるもの。
- コードゲート: 守らせたい規律そのもの（プロンプトに書いても守られないため）。

CLAUDE.md の「ペルソナは一旦外す」は、v1 で**システムプロンプトが肥大しペルソナと
ロジックが癒着した**ことへの対処だった。ここではファイルを分け、規律を混ぜないことと
長さの上限で、その再発を構造的に防ぐ。

## 読み込みの規約
- ファイルが無ければペルソナ無し（既存挙動のまま・壊れない）。
- 見出し(`#`)・水平線・コメント段落は落とし、**箇条書きの中身だけ**を渡す。
  ファイルの説明文がそのまま system に入ると、イブへの指示として読まれてしまうため。
- `PERSONA_MAX_CHARS` を超えたら**切り詰めて警告**（肥大の再発を上限で止める）。
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 肥大の上限。v1 の失敗は「気づいたら膨れていた」なので、気づける形で止める。
PERSONA_MAX_CHARS = 1200

# system に置く時の見出し。`# 参照` 等と同じ形にして、他ブロックと混ざらないようにする。
PERSONA_HEADING = "# あなた（イブ）の性格と口調"

_SECTION = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$")
_BULLET = re.compile(r"^\s*[-*・]\s+(.+?)\s*$")
_HR = re.compile(r"^\s*-{3,}\s*$")


def _default_path(root: Optional[Path] = None) -> Path:
    """既定は**手元だけの persona.local.md**（あれば）、無ければリポジトリ直下の persona.md（書式の見本）。

    `.env` の `PERSONA_FILE` で差し替え可（`persona_path`・`load_persona` の呼び手が優先する）。
    2026-10-06 ユーザ「コミットはコードとシステムプロンプトフォーマットだけで、実例データや詳細なペルソナは除外」
    → 本物の性格は .gitignore 済みの persona.local.md に置き、リポジトリには書式の見本だけを置く。
    """
    base = root or Path(__file__).resolve().parent.parent
    local = base / "persona.local.md"
    return local if local.exists() else base / "persona.md"


def parse_persona(markdown: str) -> str:
    """persona.md → system に入れる本文（純関数・raise しない）。

    箇条書きだけを拾う。見出しは節の区切りとして使い、**説明段落は落とす**
    （「〜を書かないでください」のような編集者向けの注意書きが、イブへの指示として
    読まれるのを防ぐ）。
    """
    lines: list[str] = []
    section: Optional[str] = None
    seen_hr = False
    for raw in (markdown or "").splitlines():
        if _HR.match(raw):
            seen_hr = True
            continue
        if not seen_hr:
            continue  # 水平線より前は編集者向けの説明＝落とす
        m = _SECTION.match(raw)
        if m:
            section = m.group(1).strip()
            continue
        b = _BULLET.match(raw)
        if b:
            body = b.group(1).strip()
            # 節名を頭に付ける（「性格: …」「口調: …」）。どの側面の指示かが system 上で分かる。
            lines.append((f"{section}: " if section else "") + body)
            continue
        # 箇条書きの**継続行**（インデントされた続き）は直前の項目に足す。
        # これを落とすと「推測で埋めない。…／知ったかぶりで話を合わせない。」の後半が消える。
        if lines and raw.startswith((" ", "\t")) and raw.strip():
            lines[-1] += " " + raw.strip()
    return "\n".join(f"- {ln}" for ln in lines)


def load_persona(path: Optional[str] = None) -> str:
    """ペルソナ本文（見出し付き1ブロック）を返す。無ければ空文字。"""
    p = Path(path) if path else _default_path()
    try:
        if not p.exists():
            return ""
        body = parse_persona(p.read_text(encoding="utf-8"))
    except OSError:
        logger.exception("ペルソナの読み込みに失敗（ペルソナ無しで継続）")
        return ""
    if not body:
        return ""
    if len(body) > PERSONA_MAX_CHARS:
        logger.warning("ペルソナが長すぎるので切り詰めた（%d > %d字）。"
                       "規律や好みが混ざっていないか確認してください。",
                       len(body), PERSONA_MAX_CHARS)
        body = body[:PERSONA_MAX_CHARS]
    return f"{PERSONA_HEADING}\n{body}"


def persona_for(persona: str, how: str) -> str:
    """応答以外の AI（感情判定・フィードバック）に添える性格ブロック。性格が無ければ空文字。

    応答だけが性格を持つと、口の悪いいじりを感情判定が「哀」と読む等、ちぐはぐになる（2026-10-05 に実測）。
    見出しは「あなた（イブ）の」ではなく、その AI が何に使うか（how）を書いた形に差し替える。
    """
    if not persona:
        return ""
    body = persona.split("\n", 1)[1] if persona.startswith(PERSONA_HEADING) else persona
    return f"\n\n# イブの性格（{how}）\n{body}"


def persona_path() -> str:
    """実際に読む場所（起動ログ用）。"""
    return str(os.getenv("PERSONA_FILE") or _default_path())
