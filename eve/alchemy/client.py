"""イブのゲーム画面（Alchemy の本物のページ）をブラウザで開き、そこから遊ぶ（Playwright・非同期）。

## なぜブラウザか
ユーザ指定「画面認識はAIが操作するはずの画面を認識させる。ユーザーのPC画面は認識させない」。
イブにも自分の画面が要る。ゲームのページ（`client/`）をそのまま開けば、ユーザと同じ画面がイブの席から
描かれる。画面認識はこのページの画面写しを見る（`screen.py`）。

## 情報と操作の経路
- **受け取る**: ページの WebSocket が受けた通信をそのまま横から受け取る（Playwright の `framereceived`）。
  判断は画面ではなくこの通信（サーバーの state・events）から作る＝読み違いが入らない。
- **送る**: ページが持つ通信（`window.__app.net.send`・ゲームの試験用の窓口）からサーバーへ送る。
  合言葉の部屋の出入りは、ページの画面も切り替わるように `window.__app.S` の画面名も合わせて変える。

ページを読み込み直したり通信が切れたりしても、ページ自身がつなぎ直す（そのたびに WebSocket を拾い直す）。
音は鳴らさない（`--mute-audio`・マイクに入るとイブが自分の音に反応する）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict], None]

# ページが読まれる前に、名前とルール説明済みの印を入れる（タイトル画面でルールの説明が開かないように）。
# reducedMotion は演出を軽くする設定（画面写しと CPU の負担を減らす・遊び方は変わらない）
_INIT_JS = """
(() => {{
  try {{
    localStorage.setItem('flaskrondo:name', JSON.stringify({name}));
    localStorage.setItem('flaskrondo:seenRules', 'true');
    localStorage.setItem('flaskrondo:reducedMotion', 'true');
  }} catch (e) {{}}
}})();
"""

_JOIN_JS = """
(kw) => {
  const A = window.__app; const S = A.S;
  S.keywordInput = kw;
  A.net.send({type: 'profile', name: S.profile.name});
  if (S.screen === 'game') { S.game = null; S.screen = 'keyword'; }
  return A.net.send({type: 'keyword.join', keyword: kw});
}
"""

_LEAVE_JS = """
() => {
  const A = window.__app; const S = A.S;
  A.net.send({type: 'room.leave'});
  if (S.keyword && S.keyword.keyword) A.net.send({type: 'keyword.leave'});
  S.keyword = null; S.game = null; S.screen = 'title';
  return true;
}
"""

_BACK_JS = """
() => {
  const A = window.__app; const S = A.S;
  if (S.screen !== 'game') return false;
  S.game = null;
  if (S.keyword && S.keyword.keyword) {
    S.screen = 'keyword';
    A.net.send({type: 'keyword.join', keyword: S.keyword.keyword});
  } else {
    S.screen = 'title';
  }
  return true;
}
"""


class ClientError(RuntimeError):
    """ゲームのページを開けない・つながらない（理由は文で持つ＝そのままイブの報告に使う）。"""


class AlchemyClient:
    def __init__(self, url: str, name: str, *, on_message: Optional[OnMessage] = None,
                 on_lost: Optional[Callable[[str], None]] = None,
                 headless: bool = True, channel: str = "chrome", viewport: tuple = (1024, 576),
                 eval_timeout: float = 5.0) -> None:
        self.url = url if url.endswith("/") else url + "/"
        self.name = name
        self._on_message = on_message
        self._on_lost = on_lost  # ページが落ちた（つなぎ直せない）時に理由の文で呼ぶ
        self._headless = headless
        self._channel = (channel or "").strip()
        self._viewport = viewport
        self._eval_timeout = eval_timeout
        self._pw = None
        self._browser = None
        self._page = None
        self.closed = False
        self.ws_open = False

    # --- 起動と終了 ---------------------------------------------------------
    async def start(self, timeout: float = 25.0) -> None:
        """ブラウザを起こしてゲームを開き、サーバーとつながるまで待つ。失敗は ClientError。"""
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ClientError("ブラウザを動かす部品（playwright）が入っていない") from e
        try:
            self._pw = await async_playwright().start()
            kw: dict = {"headless": self._headless, "args": ["--mute-audio"]}
            if self._channel:
                kw["channel"] = self._channel
            try:
                self._browser = await self._pw.chromium.launch(**kw)
            except Exception:
                if not self._channel:
                    raise
                # 指定のブラウザ（既定は PC の Chrome）が無ければ Playwright 同梱のものを試す
                logger.warning("🎮 ブラウザ %s を起動できなかったので同梱のものを試す", self._channel, exc_info=True)
                kw.pop("channel", None)
                self._browser = await self._pw.chromium.launch(**kw)
            ctx = await self._browser.new_context(
                viewport={"width": self._viewport[0], "height": self._viewport[1]}, locale="ja-JP",
                device_scale_factor=1)
            await ctx.add_init_script(_INIT_JS.format(name=json.dumps(self.name, ensure_ascii=False)))
            page = await ctx.new_page()
            page.on("websocket", self._on_ws)
            page.on("pageerror", lambda e: logger.warning(
                "🎮 ゲームのページでエラー: %s ｜ %s", e, (getattr(e, "stack", "") or "").replace("\n", " / ")[:400]))
            page.on("crash", lambda _p: self._mark_crashed())
            self._page = page
            await page.goto(self.url, wait_until="domcontentloaded", timeout=timeout * 1000)
            await page.wait_for_function(
                "() => window.__app && window.__app.S.conn === 'online'", timeout=timeout * 1000)
        except ClientError:
            await self.close()
            raise
        except Exception as e:
            await self.close()
            raise ClientError(f"ゲームのページ（{self.url}）を開けなかった: {type(e).__name__}") from e
        logger.info("🎮 ゲームのページを開いた（%s・%s）", self.url, "画面なし" if self._headless else "窓あり")

    async def close(self) -> None:
        self.closed = True
        page, browser, pw = self._page, self._browser, self._pw
        self._page = self._browser = self._pw = None
        for what, fn in (("ページ", page.close if page else None), ("ブラウザ", browser.close if browser else None),
                         ("Playwright", pw.stop if pw else None)):
            if fn is None:
                continue
            try:
                await asyncio.wait_for(fn(), timeout=5.0)
            except Exception:
                logger.debug("🎮 %s を閉じる時に例外（無視）", what, exc_info=True)

    @property
    def alive(self) -> bool:
        return not self.closed and self._page is not None

    def _mark_crashed(self) -> None:
        logger.warning("🎮 ゲームのページが落ちた")
        self.closed = True
        if self._on_lost is not None:
            try:
                self._on_lost("ゲームのページが落ちた")
            except Exception:
                logger.exception("🎮 ページが落ちた知らせの処理で例外")

    # --- 受信 ---------------------------------------------------------------
    def _on_ws(self, ws) -> None:
        if not str(getattr(ws, "url", "")).rstrip("/").endswith("/ws"):
            return
        self.ws_open = True
        ws.on("framereceived", self._on_frame)
        ws.on("close", lambda _ws: self._on_ws_close())

    def _on_ws_close(self) -> None:
        self.ws_open = False
        logger.info("🎮 ゲームとの通信が切れた（ページがつなぎ直す）")

    def _on_frame(self, payload: Any) -> None:
        if isinstance(payload, (bytes, bytearray)):
            payload = bytes(payload).decode("utf-8", "replace")
        try:
            msg = json.loads(payload)
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict) or self._on_message is None:
            return
        try:
            self._on_message(msg)
        except Exception:
            logger.exception("🎮 受信の処理で例外（続行）")

    # --- 送信 ---------------------------------------------------------------
    async def _eval(self, js: str, arg: Any = None) -> Any:
        if not self.alive:
            raise ClientError("ゲームのページが開いていない")
        try:
            return await asyncio.wait_for(self._page.evaluate(js, arg), timeout=self._eval_timeout)
        except asyncio.TimeoutError as e:
            raise ClientError("ゲームのページが応答しない") from e
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise ClientError(f"ゲームのページの操作に失敗: {type(e).__name__}") from e

    async def send(self, msg: dict) -> bool:
        """サーバーへ1通送る（ページの通信が切れていれば False）。"""
        return bool(await self._eval("m => window.__app.net.send(m)", msg))

    async def join_keyword(self, keyword: str) -> bool:
        return bool(await self._eval(_JOIN_JS, keyword))

    async def set_ready(self, ready: bool = True) -> bool:
        return await self.send({"type": "keyword.ready", "ready": bool(ready)})

    async def back_to_room(self) -> bool:
        """対戦の結果画面から合言葉の部屋の画面へ戻る（結果画面でなければ何もしない）。"""
        return bool(await self._eval(_BACK_JS))

    async def leave(self) -> None:
        await self._eval(_LEAVE_JS)

    async def screen_name(self) -> str:
        return str(await self._eval("() => window.__app.S.screen"))

    async def screenshot(self, quality: int = 85) -> Optional[bytes]:
        """今の画面（JPEG）。ページが無ければ None。"""
        if not self.alive:
            return None
        try:
            return await asyncio.wait_for(self._page.screenshot(type="jpeg", quality=quality), timeout=5.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("🎮 画面写しに失敗", exc_info=True)
            return None
