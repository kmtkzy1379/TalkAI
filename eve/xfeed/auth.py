"""本人のログイン（OAuth 2.0・PKCE）のトークンの保管と更新（ホームタイムラインを読む時だけ使う）。

最初のログインは `tools/x_login.py`（ブラウザで X にログインして許可する）。ここは保存済みのトークンを使い、
期限（2時間）が近ければ更新する。**更新トークンは使うたびに新しくなる**（X の仕様）ので、更新したらすぐ保存する
（保存に失敗すると次の更新ができず、ログインし直しになる）。保存先は手元だけ（`x_token.json`・.gitignore 済み）。

https://docs.x.com/fundamentals/authentication/oauth-2-0/authorization-code
- 許可の画面 `https://x.com/i/oauth2/authorize`・トークン `https://api.x.com/2/oauth2/token`
- 公開クライアント（Native App）は本文に client_id、機密クライアント（Web App）は Basic 認証の見出しで送る
- 受け取った code は30秒以内に引き換える
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode

from .client import XApiError, error_text

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
# 読むだけ: 投稿（ホームタイムライン）・本人の ID・更新トークン（offline.access が無いと2時間で切れて終わる）
SCOPES = ("tweet.read", "users.read", "offline.access")
REFRESH_MARGIN_SEC = 120.0  # 期限のこの秒数前から更新する

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def make_pkce() -> tuple[str, str]:
    """(code_verifier, code_challenge[S256])。"""
    verifier = secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def authorize_url(client_id: str, redirect_uri: str, state: str, challenge: str) -> str:
    return AUTHORIZE_URL + "?" + urlencode({
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
        "scope": " ".join(SCOPES), "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })


def token_request(client_id: str, client_secret: str, form: dict) -> tuple[dict, dict]:
    """トークンの口に送る (headers, form)。秘密があれば Basic 認証、無ければ本文に client_id（公開クライアント）。"""
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    form = dict(form)
    if client_secret:
        raw = f"{client_id}:{client_secret}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    else:
        form["client_id"] = client_id
    return headers, form


class UserAuth:
    """保存済みのトークン（`x_token.json`）。loop 単一所有。更新は1本ずつ（lock）。"""

    def __init__(self, path: str, *, client_id: str, client_secret: str = "",
                 now_fn: Callable[[], float] = time.time) -> None:
        p = Path(path)
        self.path = p if p.is_absolute() else _REPO_ROOT / p
        self._client_id = client_id
        self._client_secret = client_secret
        self._now = now_fn
        self._lock = asyncio.Lock()
        self._tok: dict = self._load()

    def _load(self) -> dict:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.exception("X のログイン情報を読めなかった（タイムラインは使わない）: %s", self.path)
        return {}

    def ready(self) -> bool:
        """タイムラインを読める状態か（本人の ID があり、更新できるトークンがある）。"""
        return bool(self._tok.get("user_id")) and bool(self._tok.get("refresh_token") or self._tok.get("access_token"))

    @property
    def user_id(self) -> str:
        return str(self._tok.get("user_id") or "")

    @property
    def username(self) -> str:
        return str(self._tok.get("username") or "")

    def save(self, body: dict, *, user_id: Optional[str] = None, username: Optional[str] = None) -> None:
        """トークンの返事（access_token・refresh_token・expires_in）を取り込んで保存する（書きかけで壊さない＝置き換え）。"""
        tok = dict(self._tok)
        if body.get("access_token"):
            tok["access_token"] = body["access_token"]
            tok["expires_at"] = self._now() + float(body.get("expires_in") or 7200)
        if body.get("refresh_token"):
            tok["refresh_token"] = body["refresh_token"]
        if body.get("scope"):
            tok["scope"] = body["scope"]
        if user_id:
            tok["user_id"] = str(user_id)
        if username:
            tok["username"] = str(username)
        self._tok = tok
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(tok, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    async def access_token(self, request_fn) -> str:
        """使えるアクセストークン（期限が近ければ更新してから）。更新できなければ XApiError。"""
        async with self._lock:
            if self._tok.get("access_token") and float(self._tok.get("expires_at") or 0) - REFRESH_MARGIN_SEC > self._now():
                return str(self._tok["access_token"])
            await self._refresh(request_fn)
            return str(self._tok["access_token"])

    async def _refresh(self, request_fn) -> None:
        rt = self._tok.get("refresh_token")
        if not rt:
            raise XApiError(401, "ログインし直しが必要（更新トークンが無い・tools/x_login.py）")
        headers, form = token_request(self._client_id, self._client_secret,
                                      {"grant_type": "refresh_token", "refresh_token": rt})
        status, body, _ = await request_fn("POST", TOKEN_URL, headers=headers, data=form)
        if status >= 400 or not (body or {}).get("access_token"):
            raise XApiError(status if status >= 400 else 500,
                            f"トークンの更新に失敗（ログインし直しが要るかもしれない・tools/x_login.py）: {error_text(body)}")
        try:
            self.save(body)
        except OSError:
            # 新しい更新トークンを保存できないと、次の起動でログインし直しになる（今のプロセスでは使える）
            logger.exception("X の新しいトークンを保存できなかった（次の起動でログインし直しが要る）: %s", self.path)
        logger.info("🐦 X のログインを更新した（%s）", self.username or self.user_id)
