"""Tier-1: HTML 表示（`eve/ui/web_view.py` と `sniff` の HTML 判定）。

GUI も Chromium も起こさない。**種類判定**と**リクエスト遮断の方針**という、
壊れると静かに危なくなる2点を純関数のまま固める。

守りたい不変条件:
  - 拡張子ではなく中身で HTML と決める（`.html` を騙る PNG は画像のまま）
  - HTML から**ネットに出られない**（fetch / 外部 CDN / 画像 URL）
  - HTML から**他のファイルを読めない**（作業フォルダの中でさえ、表示中の1枚以外は不可）
  - 表示手段が無い環境では**ソースを出す**（押せない UI を描いて嘘をつかない）
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.ui import web_view  # noqa: E402
from eve.workspace.viewer import sniff  # noqa: E402

PASS = FAIL = 0
REPO = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def t_sniff() -> None:
    print("\n[1] 種類判定（中身で決める・拡張子は信じない）")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cases = [
            ("plain.html", b"<!DOCTYPE html>\n<html><body>hi</body></html>", "HTML"),
            ("upper.html", b"<!doctype HTML><html></html>", "HTML"),
            ("nodoctype.html", b"<html>\n<head></head><body>x</body></html>", "HTML"),
            ("bom.html", "\ufeff<!DOCTYPE html><html></html>".encode("utf-8"), "HTML"),
            ("lead_ws.html", b"\n\n   <!DOCTYPE html><html></html>", "HTML"),
            ("comment.html", b"<!-- made by eve -->\n<!DOCTYPE html><html></html>", "HTML"),
            # 拡張子が .txt でも中身が HTML なら HTML（**中身で決める**）
            ("actually.txt", b"<!DOCTYPE html><html><body>calc</body></html>", "HTML"),
            # 逆に .html でも中身がテキスト/画像ならそちら
            ("memo.html", "ふつうのメモです。<html> とは書いていない".encode("utf-8"), "TEXT"),
            ("fake.html", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "PNG"),
            ("frag.html", b"<div>only a fragment</div>", "TEXT"),
        ]
        for name, body, want in cases:
            p = tmp / name
            p.write_bytes(body)
            got = sniff(p)
            check(f"{name} → {want}", got == want, f"got={got}")


def t_request_policy() -> None:
    print("\n[2] リクエスト遮断（HTML は外へ出られない）")
    allowed = "file:///C:/eve_ws/workspace/calc.html"
    ok_cases = [
        (allowed, "表示している当のファイル"),
        (allowed.replace("C:", "c:"), "大小の揺れ（Windows のパスは区別しない）"),
        (allowed + "#result", "同一ページ内の fragment"),
        ("data:image/png;base64,AAAA", "data: URI（ページ内で完結）"),
        ("about:blank", "about:blank"),
        ("blob:file:///abcd", "blob:（ページが自分で作ったもの）"),
    ]
    for url, why in ok_cases:
        check(f"通す: {why}", web_view.allow_request(url, allowed) is True, url)

    ng_cases = [
        ("https://example.com/api", "https（fetch で外部 API）"),
        ("http://127.0.0.1:50021/version", "http（ローカルの VOICEVOX すら通さない）"),
        ("ws://example.com/socket", "WebSocket"),
        ("wss://example.com/socket", "WebSocket(TLS)"),
        ("https://cdn.jsdelivr.net/npm/x.js", "外部 CDN のスクリプト"),
        ("file:///C:/eve_ws/workspace/secret.txt", "**作業フォルダ内の別ファイル**"),
        ("file:///C:/Users/tester/Desktop/eve-v2/.env", "リポジトリの .env"),
        ("file:///C:/eve_ws/workspace/", "作業フォルダそのもの（一覧）"),
        ("ftp://example.com/x", "ftp"),
        ("", "空の URL"),
    ]
    for url, why in ng_cases:
        check(f"止める: {why}", web_view.allow_request(url, allowed) is False, url)


def t_fail_soft() -> None:
    print("\n[3] 使えない環境でも止まらない")
    check("available() が bool を返す", isinstance(web_view.available(), bool))
    check("preinit() が bool を返す（無ければ False で起動を止めない）",
          isinstance(web_view.preinit(), bool))
    src = (REPO / "eve" / "ui" / "task_window.py").read_text(encoding="utf-8")
    check("HTML ビューを作れなければソース表示へ落ちる",
          "_build_html" in src and "return None" in src)
    check("落ちた時に**ソースだと明示**する（押せない UI を描かない）",
          "ソースを出します" in src)
    tk = (REPO / "eve" / "workspace" / "viewer.py").read_text(encoding="utf-8")
    check("Tk ビューア（別プロセス）も HTML は動かせないと正直に言う",
          "HTML を動かせません" in tk)


def t_wiring() -> None:
    print("\n[4] 配線（設定・能力の説明・柵）")
    from eve.config import Config

    check("TASKWIN_HTML_ENABLED が bool", isinstance(Config.TASKWIN_HTML_ENABLED, bool))
    check("TASKWIN_PROTECT_CLICKS が int", isinstance(Config.TASKWIN_PROTECT_CLICKS, int))
    check("誤クリック対策で既定は 3 回", Config.TASKWIN_PROTECT_CLICKS == 3,
          f"{Config.TASKWIN_PROTECT_CLICKS}")
    body = (REPO / ".env.example").read_text(encoding="utf-8")
    for key in ("TASKWIN_HTML_ENABLED", "TASKWIN_PROTECT_CLICKS"):
        check(f"**.env.example に {key} がある**", key in body)

    cap = (REPO / "eve" / "workspace" / "capabilities.py").read_text(encoding="utf-8")
    # `outward_actions()` は description の**1文目**しか応答LLM へ渡さない
    first = cap.split('description="作業フォルダにある')[1].split("。")[0]
    check("show_on_screen の1文目に HTML が入っている", "HTML" in first, first[:80])
    check("tkinter は使えないと TaskAgent へ伝えている", "tkinter" in cap)
    # 実測 2026-08-15: HTML 経路を足したら、写真の依頼にも CSS で絵を描いて返した
    check("写真の依頼に絵で代用させない線引きがある",
          "代用してはいけない" in cap and "取ってきて" in cap)
    agent = (REPO / "eve" / "task" / "agent.py").read_text(encoding="utf-8")
    check("TaskAgent のプロンプトが HTML 経路へ誘導する",
          "HTML" in agent and "show_on_screen" in agent)

    web = (REPO / "eve" / "ui" / "web_view.py").read_text(encoding="utf-8")
    for what, needle in (("ネット遮断", "setUrlRequestInterceptor"),
                         ("遷移の拒否", "acceptNavigationRequest"),
                         ("alert のモーダル封じ", "javaScriptAlert"),
                         ("窓を増やさせない", "createWindow"),
                         ("ローカルファイル横断の禁止", "LocalContentCanAccessFileUrls"),
                         ("痕跡を残さない", "NoPersistentCookies")):
        check(f"柵がある: {what}", needle in web)
    # 生成も import もしていないことを見る（設計メモで名前に触れるのは可）
    check("QWebChannel を使っていない（JS から親の Qt に手が届かない）",
          "QtWebChannel" not in web and "QWebChannel(" not in web)


if __name__ == "__main__":
    t_sniff()
    t_request_policy()
    t_fail_soft()
    t_wiring()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
