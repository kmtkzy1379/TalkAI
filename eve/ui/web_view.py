r"""タスクウィンドウで **HTML を動かして** 表示する（電卓のような「押せるもの」用）。

## なぜ表示側に足すのか（サンドボックスで GUI を出さない理由）
イブが GUI を作る経路は2つ考えられる:

1. `run_program` のサンドボックスで tkinter を起動する
2. HTML を `write_file` して `show_on_screen` で親側に表示する ← **こちら**

1 は採らない。サンドボックスの子は Job Object の `UILIMIT_HANDLES`/`UILIMIT_DESKTOP`
下にあり、これは **shatter 攻撃（他プロセスの窓へメッセージを飛ばして権限を抜く）**と
**シャットダウン封じ**の両方を止めている（`workspace/winexec.py`）。GUI を出すために
対話デスクトップの ACL を開けると、`python` を許している以上「親権限で任意コードを
実行させる」のと同じ穴が開く。加えて `run_program` は**終わるまで待って 60秒で殺す**
契約で、常駐する GUI とは形が合わない。

2 なら隔離は一切触らない。すでに親プロセスで動いている表示経路の**種類を1つ増やす**
だけで済む（画像・テキスト → 画像・テキスト・HTML）。

## 何を許して何を止めるか
「読んで描くだけなら昇格にならない」が `workspace/viewer.py` から続く前提なので、
**HTML が外に手を伸ばせないこと**をここで担保する:

| 止めるもの | 手段 |
|---|---|
| ネットへの通信（`fetch` / `<img src=http…>` / WebSocket） | リクエスト遮断器（allowlist は対象ファイル1つだけ） |
| 作業フォルダの他ファイル・`file://` 横断 | 同上（URL 完全一致でしか通さない） |
| 別ページへの遷移・`window.open` | `acceptNavigationRequest` と `createWindow` |
| `alert()` による**モーダル固着** | `javaScriptAlert/Confirm/Prompt` を握り潰す |
| カメラ・マイク・位置情報・画面共有 | 権限要求を全部拒否 |
| ディスクに残る痕跡（Cookie/キャッシュ） | off-the-record プロファイル |
| 親の Qt オブジェクトへの到達 | `QWebChannel` を**使わない**（JS からは何も見えない） |

**正直に**: これは「悪意ある HTML を安全に動かす」保証ではなく、Chromium のサンドボックス
＋上記の柵まで。中身を書くのはイブ（LLM）で、任意の第三者ではないという前提に立っている。

## 依存（fail-soft）
`QtWebEngine` は `PySide6-Addons`（`PySide6-Essentials` とは**別パッケージ**）。
入っていなければ `available()` が False を返し、呼び出し側は HTML ソースの表示へ落ちる。
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ページ内部で完結する（外へ出られない）スキーム。ここだけは素通しする。
_INTERNAL_SCHEMES = ("data:", "about:", "blob:", "qrc:", "chrome-error:")

_imported = False   # QtWebEngine を import できたか（preinit で確定）


def normalize(url: str) -> str:
    """比較用に URL を均す。Windows のパスは大小を区別しないので畳む。

    `QUrl.toString()` は `file:///C:/eve_ws/...` を返す。同じファイルを指す表記の揺れ
    （大文字小文字・末尾スラッシュ・fragment）で allowlist をすり抜けさせない。
    """
    u = (url or "").strip()
    u = u.split("#", 1)[0]          # fragment は同一ページ内移動なので落として比較する
    return u.rstrip("/").casefold()


def allow_request(url: str, allowed: str) -> bool:
    """このリクエストを通してよいか（**純関数**＝Qt 無しで試験できる）。

    通すのは「表示している当のファイル」と、ページ内部で完結するスキームだけ。
    それ以外（http/https/ws/ftp/他の file）は**全部**止める。
    """
    u = normalize(url)
    if not u:
        return False
    if u == normalize(allowed):
        return True
    return any(u.startswith(s) for s in _INTERNAL_SCHEMES)


def preinit() -> bool:
    """**`QApplication` を作る前に**呼ぶ（QtWebEngine の要求）。

    QtWebEngine は import 時に `AA_ShareOpenGLContexts` を立てる。`QApplication` の後に
    import すると Qt が警告を出し、OpenGL を使う他のウィジェット（Live2D）と文脈を
    共有できない。入っていない環境では False を返すだけで、起動は止めない。
    """
    global _imported
    if _imported:
        return True
    try:
        from PySide6.QtCore import QCoreApplication, Qt

        # import より先に立てる（import 側でも立つが、順序を明示して意図を残す）
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        import PySide6.QtWebEngineWidgets  # noqa: F401
        _imported = True
        return True
    except Exception:
        logger.info("QtWebEngine が無いので HTML はソース表示になる"
                    "（`pip install PySide6-Addons` で有効化）")
        return False


def available() -> bool:
    """HTML を動かして表示できるか（`preinit` 後に呼ぶ）。"""
    if _imported:
        return True
    try:
        import PySide6.QtWebEngineWidgets  # noqa: F401
        return True
    except Exception:
        return False


def build(target: Path):
    """`target` 1ファイルだけを表示するビューを作る。失敗したら例外を投げる
    （呼び出し側が握ってソース表示へ落ちる）。"""
    from PySide6.QtCore import QUrl, Qt
    from PySide6.QtWebEngineCore import (
        QWebEngineProfile, QWebEnginePage, QWebEngineSettings,
        QWebEngineUrlRequestInterceptor,
    )
    from PySide6.QtWebEngineWidgets import QWebEngineView

    url = QUrl.fromLocalFile(str(target.resolve()))
    allowed = url.toString()

    class _Interceptor(QWebEngineUrlRequestInterceptor):
        """ネットワークとファイル横断を止める最後の砦（全リクエストがここを通る）。"""

        def interceptRequest(self, info) -> None:
            try:
                req = info.requestUrl().toString()
                if not allow_request(req, allowed):
                    info.block(True)
                    logger.info("🌐 表示中の HTML から外部アクセスを遮断: %.120s", req)
            except Exception:  # 遮断器で落ちると表示ごと死ぬので握る
                logger.debug("リクエスト遮断で例外（通す）", exc_info=True)

    class _Page(QWebEnginePage):
        """ダイアログ・遷移・権限要求を封じたページ。"""

        def javaScriptAlert(self, origin, msg) -> None:
            # **モーダルを出さない**。出すと UI 全体が入力を受け付けなくなり、
            # イブに話しかけることすらできなくなる（LLM は素で alert() を書く）。
            logger.info("🖥 HTML の alert(): %.200s", msg)

        def javaScriptConfirm(self, origin, msg) -> bool:
            logger.info("🖥 HTML の confirm(): %.200s", msg)
            return False

        def javaScriptPrompt(self, origin, msg, default) -> tuple:
            logger.info("🖥 HTML の prompt(): %.200s", msg)
            return False, ""

        def javaScriptConsoleMessage(self, level, msg, line, source) -> None:
            # ページのエラーは**ログに出す**（動かない時の一次ソースになる）。
            logger.debug("🖥 HTML console[%s] %s:%s %.300s", level, source, line, msg)

        def certificateError(self, error) -> bool:
            return False  # そもそもネットに出られないが、黙って通す口は作らない

        def acceptNavigationRequest(self, url, nav_type, is_main_frame) -> bool:
            ok = allow_request(url.toString(), allowed)
            if not ok:
                logger.info("🌐 表示中の HTML からの遷移を拒否: %.120s", url.toString())
            return ok

    class _View(QWebEngineView):
        def createWindow(self, _type):
            return None  # `window.open` / `target=_blank` で窓を増やさせない

    view = _View()
    # **プロファイルは名前を付けない＝off-the-record**（Cookie もキャッシュもディスクに
    # 残さない）。親（view）に持たせて寿命を縛る＝GC されると Chromium 側が落ちる。
    profile = QWebEngineProfile(view)
    profile.setHttpCacheType(QWebEngineProfile.NoCache)
    profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
    interceptor = _Interceptor(view)
    profile.setUrlRequestInterceptor(interceptor)

    page = _Page(profile, view)
    view.setPage(page)
    view._eve_keep = (profile, interceptor, page)  # 参照を保持（切れると落ちる）

    s = page.settings()
    for attr, on in (
        ("LocalContentCanAccessFileUrls", False),   # file:// から他ファイルを読ませない
        ("LocalContentCanAccessRemoteUrls", False),
        ("JavascriptCanOpenWindows", False),
        ("JavascriptCanAccessClipboard", False),
        ("JavascriptCanPaste", False),
        ("AllowRunningInsecureContent", False),
        ("AllowWindowActivationFromJavaScript", False),
        ("LocalStorageEnabled", False),             # 痕跡を残さない
        ("ScreenCaptureEnabled", False),
        ("FullScreenSupportEnabled", False),        # 枠の外へ広がらせない
        ("PluginsEnabled", False),
        ("PdfViewerEnabled", False),
    ):
        try:
            s.setAttribute(getattr(QWebEngineSettings.WebAttribute, attr), on)
        except AttributeError:
            # Qt のバージョンで名前が無いものは飛ばす（既定値のまま＝落とさない）
            logger.debug("WebEngine 設定 %s はこの Qt に無い", attr)

    # カメラ・マイク・位置情報などの権限要求は**全部拒否**（Qt 6.8 で API が変わったので両対応）
    try:
        page.permissionRequested.connect(lambda p: p.deny())
    except AttributeError:
        try:
            page.featurePermissionRequested.connect(
                lambda origin, feature: page.setFeaturePermission(
                    origin, feature, QWebEnginePage.PermissionDeniedByUser))
        except AttributeError:
            logger.debug("権限要求のシグナルが見つからない（この Qt では既定拒否）")

    view.setContextMenuPolicy(Qt.NoContextMenu)  # 「新しい窓で開く」等を出さない
    view.setUrl(url)
    return view
