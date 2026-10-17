r"""X に本人としてログインし、タイムラインを読むためのトークンを保存する（最初に1回だけ）。

タイムラインは「何を調べるか」の材料にだけ使う（投稿の本文は記憶に入れない・ユーザ決定 2026-10-17）。

準備（X の開発者コンソールのアプリ）:
1. 「User authentication settings」で OAuth 2.0 を有効にする（App permissions は Read）。
   Type of App は Native App（公開クライアント・Client Secret なし）でよい。Web App にしたなら Client Secret も使う
2. Callback URI に http://127.0.0.1:8723/callback を入れる（ポートを変えたら .env の X_REDIRECT_PORT も）
3. .env に X_CLIENT_ID（Web App なら X_CLIENT_SECRET も）を入れる

    $env:PYTHONIOENCODING="utf-8"
    & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\x_login.py

ブラウザが開くので、X にログインして「許可」を押す（30秒以内に引き換えるので、許可したらそのまま待つ）。
保存先は x_token.json（手元だけ・.gitignore 済み）。最後にログインした本人の ID を1回引く（ユーザー情報 1件 $0.01）。
"""
from __future__ import annotations

import asyncio
import os
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.config import Config  # noqa: E402
from eve.xfeed.auth import TOKEN_URL, UserAuth, authorize_url, make_pkce, token_request  # noqa: E402
from eve.xfeed.client import API_BASE, aiohttp_request, error_text  # noqa: E402

WAIT_SEC = 300  # ブラウザで許可を押すまで待つ上限


def wait_for_code(port: int, state: str) -> dict:
    """127.0.0.1:port/callback に来る1回の返事（code か error）を受け取る。"""
    got: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802（http.server の決まり）
            q = parse_qs(urlparse(self.path).query)
            if urlparse(self.path).path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            got.update({k: v[0] for k, v in q.items()})
            ok = got.get("state") == state and "code" in got
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = "ログインできました。このタブは閉じてください。" if ok else "ログインできませんでした。ターミナルを見てください。"
            self.wfile.write(f"<html><body><p>{msg}</p></body></html>".encode("utf-8"))

        def log_message(self, *a):  # 標準のアクセスログは出さない
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1.0
    done = threading.Event()

    def loop():
        while not got and not done.is_set():
            server.handle_request()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    t.join(WAIT_SEC)
    done.set()
    server.server_close()
    return got


async def main() -> int:
    if not Config.X_CLIENT_ID:
        print("X_CLIENT_ID が .env にありません（開発者コンソールのアプリの OAuth 2.0 Client ID）。")
        return 2
    redirect = f"http://127.0.0.1:{Config.X_REDIRECT_PORT}/callback"
    verifier, challenge = make_pkce()
    state = secrets.token_urlsafe(24)
    url = authorize_url(Config.X_CLIENT_ID, redirect, state, challenge)
    print("ブラウザで X のログインと許可の画面を開きます（開かなければ下の URL を開いてください）:\n" + url)
    webbrowser.open(url)
    got = await asyncio.to_thread(wait_for_code, Config.X_REDIRECT_PORT, state)
    if not got:
        print(f"{WAIT_SEC}秒待ちましたが返事がありませんでした（Callback URI が {redirect} か確かめてください）。")
        return 1
    if got.get("state") != state or "code" not in got:
        print(f"ログインできませんでした: {got.get('error') or got}")
        return 1
    headers, form = token_request(Config.X_CLIENT_ID, Config.X_CLIENT_SECRET, {
        "grant_type": "authorization_code", "code": got["code"], "redirect_uri": redirect,
        "code_verifier": verifier})
    status, body, _ = await aiohttp_request("POST", TOKEN_URL, headers=headers, data=form)
    if status >= 400 or not body.get("access_token"):
        print(f"トークンに引き換えられませんでした（{status}）: {error_text(body)}")
        return 1
    if "offline.access" not in str(body.get("scope", "")):
        print("注意: offline.access が許可されていません（2時間でタイムラインが読めなくなります）")
    status, me, _ = await aiohttp_request("GET", f"{API_BASE}/users/me",
                                          headers={"Authorization": f"Bearer {body['access_token']}"})
    user = (me or {}).get("data") or {}
    if status >= 400 or not user.get("id"):
        print(f"本人の ID を引けませんでした（{status}）: {error_text(me)}")
        return 1
    auth = UserAuth(Config.X_TOKEN_FILE, client_id=Config.X_CLIENT_ID, client_secret=Config.X_CLIENT_SECRET)
    auth.save(body, user_id=user["id"], username=user.get("username", ""))
    print(f"保存しました: @{user.get('username', '?')}（{auth.path}）。次の起動からタイムラインを材料に使います"
          f"（X_TIMELINE_POSTS={Config.X_TIMELINE_POSTS}）。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
