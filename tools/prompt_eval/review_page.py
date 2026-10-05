#!/usr/bin/env python3
"""イブの話し方レビューのページを作る（review_data.py のデータを埋め込む）。API 不要。

  python -m tools.prompt_eval.review_data && python -m tools.prompt_eval.review_page
→ eval_report/prompts/review/review.html（公開は Artifact・評価と選定はページの db に保存される）
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import OUT


def main() -> int:
    tpl = (Path(__file__).parent / "review_template.html").read_text(encoding="utf-8")
    data = (OUT / "review" / "data.json").read_text(encoding="utf-8")
    json.loads(data)  # 壊れていたら止める
    page = tpl.replace("/*__DATA__*/", data.replace("</", "<\\/"))
    out = OUT / "review" / "review.html"
    out.write_text(page, encoding="utf-8")
    print(f"→ {out}（{len(page.encode('utf-8')) // 1024} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
