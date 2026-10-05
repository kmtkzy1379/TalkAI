#!/usr/bin/env python3
"""ユーザ作の会話例（過去のファインチューニング用のテストデータ）を persona_examples.yaml に移す。API 不要。

  python -m tools.prompt_eval.import_user_examples "C:/Users/user/Downloads/システムプロンプト応答例.md"

2026-10-04 ユーザ「応答例は過去にファインチューニングに挑戦したテストデータから良さそうなのはそのまま引用。
system・状況・相手の様子・イブの気持ちは少し適当。会話内容自体はクオリティ保証できる。system は消すか今のペルソナに統一」
- system（性格・【状況】等）は捨てる（性格は persona.md が持つ）
- <think>…</think>（心の声）は捨て、声に出す部分だけ残す（応答は reasoning なしなので、心の声を見せると声に出しかねない）
- 1行1件の JSON。壊れた行（引用符や括弧の抜け）も、"from" ごとに切って拾う
- 除外: 48（実在の人種を人間扱いしない差別。実例にも検証にも使わない）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXCLUDE = {48: "実在の人種を人間扱いしない差別"}
HEADER = """\
# 話し方の実例（イブの返事のノリの参考）。ユーザ発話への返事のたびに、ここからランダムに PERSONA_EXAMPLES_K 件
# （既定 8・合計 2000字まで）を性格（persona.md）の後に見せる（eve/persona_examples.py）。
#
# - 中身はユーザ作の会話例（2026-10-04・過去のファインチューニングのテストデータから。品質はユーザ保証）。
#   system・【状況】等の注記と <think>（心の声）は外した（tools/prompt_eval/import_user_examples.py）。
# - 今の話しかけと意味が近い実例は、その回は見せない（丸写しの元）。複数ターンの実例は、どれか1つの話しかけが近ければ外す。
# - 要らないものは消してよい。空なら実例なし（性格の文だけで動く）。
#
# 形: - id: 識別子
#       turns: [{user: ユーザの一言}, {eve: イブの返事}, ...]
"""


def _unescape_md(s: str) -> str:
    return re.sub(r"\\([\[\]<>_*`#|~])", r"\1", s)


def _clean_value(v: str) -> str:
    v = v.strip()
    v = re.sub(r"[\s,\]\}]*$", "", v)
    if v.startswith('"'):
        v = v[1:]
    if v.endswith('"'):
        v = v[:-1]
    return v.replace('\\"', '"').replace("\\n", "\n").strip()


# 心の声の前に置いた「考え中のつなぎ」のうち、ただのフィラーは捨てる。本番はフィラーの仕組みが返事の前に「あー」等を
# 既に鳴らすので、実例に残すと二重になり、どの返事も「あー…」で始める口癖の元になる（約半数の実例がこれで始まる）。
# 態度のある前置き（はぁ…・ふんっ…・えぇ…・はいはい… 等）は残す
# 2026-10-04 ユーザ「会話例はそのまま使用して。除外は検証結果で判断するもので最初に外すのは機会損失」→ 前置きは全部残す
GENERIC_LEAD: set[str] = set()


def _spoken(v: str) -> str:
    """心の声（<think>…</think>・閉じ忘れ・<think の > 抜けも）を外し、声に出す部分だけにする。"""
    m = re.search(r"<think", v)
    lead = v[:m.start()].strip() if m else ""
    rest = v[m.start():] if m else v
    rest = re.sub(r"<think>?.*?(</think>|<think>)", "", rest, flags=re.S)
    rest = re.sub(r"</?think>?", "", rest).strip()
    if lead and lead.rstrip("…・.。 ") in GENERIC_LEAD:
        lead = ""
    if lead and rest:
        rest = lead + ("" if re.search(r"[…。！？!?、]$", lead) else "、") + rest
    elif lead:
        rest = lead
    return re.sub(r"\s+", " ", rest).strip()


def parse_line(line: str) -> list[tuple[str, str]]:
    line = _unescape_md(line)
    turns = []
    for chunk in re.split(r'\{\s*"from"\s*:\s*', line)[1:]:
        m = re.match(r'"(\w+)"\s*,\s*"value"\s*:\s*(.*)$', chunk, re.S)
        if not m:
            continue
        role, value = m.group(1), _clean_value(m.group(2))
        if role == "system":
            continue
        text = _spoken(value) if role == "gpt" else value.strip()
        if text:
            turns.append(("user" if role == "human" else "eve", text))
    return turns


def main() -> int:
    import yaml

    src = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Downloads\システムプロンプト応答例.md")
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if '"from"' in ln]
    items, skipped = [], []
    for no, line in enumerate(lines, 1):
        if no in EXCLUDE:
            skipped.append(f"{no}: {EXCLUDE[no]}")
            continue
        turns = parse_line(line)
        if not any(w == "user" for w, _ in turns) or not any(w == "eve" for w, _ in turns):
            skipped.append(f"{no}: 読めなかった")
            continue
        items.append({"id": f"U{no:02d}", "turns": [{w: t} for w, t in turns]})
    body = yaml.safe_dump({"examples": items}, allow_unicode=True, sort_keys=False, width=1000)
    out = REPO / "persona_examples.yaml"
    out.write_text(HEADER + body, encoding="utf-8")
    multi = sum(1 for d in items if sum(1 for t in d["turns"] if "user" in t) > 1)
    print(f"{len(lines)}行 → 実例 {len(items)}件（複数ターン {multi}件）→ {out}")
    for s in skipped:
        print("  外した:", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
