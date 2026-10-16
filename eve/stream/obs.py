"""OBS の操作（obs-websocket v5・OBS 28 以降に内蔵）と、OBS 本体の起動。

## なぜ自前の小さなクライアントか
使うのは「認証して要求を送り、返事とイベントを受ける」だけで、プロトコルは JSON 1種類（op 0/1/2/5/6/7）。
既存の依存（aiohttp）で 150 行に収まり、loop の asyncio にそのまま乗る（同期のライブラリをスレッドで回さない）。

## 接続の前提（ユーザが OBS で1回だけやること）
- ツール → WebSocket サーバー設定 → 「WebSocket サーバーを有効にする」に✓（**既定は無効**・実測 2026-10-15）。
  コマンドラインの引数では有効にできない（`--websocket_port` 等は有効化しない・obs-websocket の Config.cpp）。
- ポートとパスワードは `.env`（OBS_WS_PORT / OBS_WS_PASSWORD）が無ければ OBS 自身の設定ファイルから読む。
- 配信先（YouTube のストリームキー）は OBS の 設定 → 配信 に入れておく（ここでは書き換えない）。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# EventSubscription（protocol.md）: 使うのは配信の状態（Outputs: StreamStateChanged）と
# 音声のミュートの切り替え（Inputs: InputMuteStateChanged）。音量メーター等の頻繁な知らせは受けない
SUB_INPUTS = 1 << 3
SUB_OUTPUTS = 1 << 6

STREAM_STARTED = "OBS_WEBSOCKET_OUTPUT_STARTED"
STREAM_STOPPED = "OBS_WEBSOCKET_OUTPUT_STOPPED"


class ObsError(Exception):
    """OBS が要求を断った / つながらない（code は requestStatus.code・接続の失敗は 0）。"""

    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = code


def auth_string(password: str, salt: str, challenge: str) -> str:
    """obs-websocket v5 の認証文字列（protocol.md の手順そのまま）。"""
    secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest()).decode("ascii")
    return base64.b64encode(hashlib.sha256((secret + challenge).encode("utf-8")).digest()).decode("ascii")


class ObsClient:
    """要求→返事の対応づけと、イベントの受け取り。loop 上だけで使う。"""

    def __init__(self, *, on_event: Optional[Callable[[str, dict], None]] = None,
                 on_close: Optional[Callable[[], None]] = None) -> None:
        self.on_event = on_event
        self.on_close = on_close
        self._session = None
        self._ws = None
        self._reader: Optional[asyncio.Task] = None
        self._pending: dict[str, asyncio.Future] = {}
        self.hello: dict = {}
        self.connected = False
        self._closing = False  # 自分から閉じた（切断の通知を出さない）

    async def connect(self, url: str, password: str = "", timeout: float = 5.0,
                      subscriptions: int = SUB_OUTPUTS) -> dict:
        """接続して認証する。Hello の中身（OBS のバージョン等）を返す。失敗は ObsError。"""
        import aiohttp

        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=timeout))
        try:
            self._ws = await self._session.ws_connect(url, protocols=("obswebsocket.json",), heartbeat=20,
                                                      max_msg_size=64 * 1024 * 1024)
            hello = await asyncio.wait_for(self._recv(), timeout)
            if not hello or hello.get("op") != 0:
                raise ObsError("OBS から最初の挨拶（Hello）が来なかった")
            d = hello.get("d") or {}
            ident: dict = {"rpcVersion": 1, "eventSubscriptions": subscriptions}
            auth = d.get("authentication")
            if auth:
                if not password:
                    raise ObsError("OBS の WebSocket にパスワードが要るが、パスワードが分からない")
                ident["authentication"] = auth_string(password, auth.get("salt", ""), auth.get("challenge", ""))
            await self._ws.send_str(json.dumps({"op": 1, "d": ident}))
            ok = await asyncio.wait_for(self._recv(), timeout)
            if not ok or ok.get("op") != 2:
                raise ObsError("OBS の認証に失敗した（パスワードが違う？）")
        except ObsError:
            await self.close()
            raise
        except asyncio.TimeoutError:
            await self.close()
            raise ObsError("OBS の返事が来なかった（時間切れ）") from None
        except Exception as e:
            await self.close()
            raise ObsError(f"OBS につながらない（{type(e).__name__}）") from None
        self.hello = d
        self.connected = True
        self._reader = asyncio.create_task(self._read_loop())
        return d

    async def _recv(self) -> Optional[dict]:
        import aiohttp

        while True:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    return json.loads(msg.data)
                except ValueError:
                    continue
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                return None

    async def _read_loop(self) -> None:
        try:
            while True:
                m = await self._recv()
                if m is None:
                    break
                op, d = m.get("op"), m.get("d") or {}
                if op == 7:
                    fut = self._pending.pop(str(d.get("requestId") or ""), None)
                    if fut is not None and not fut.done():
                        fut.set_result(d)
                elif op == 5 and self.on_event is not None:
                    try:
                        self.on_event(str(d.get("eventType") or ""), d.get("eventData") or {})
                    except Exception:
                        logger.exception("OBS のイベントの処理で例外（続行）")
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("OBS からの受信で例外（接続を閉じる）")
        finally:
            was = self.connected
            self.connected = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ObsError("OBS との接続が切れた"))
            self._pending.clear()
            if was and not self._closing and self.on_close is not None:
                try:
                    self.on_close()
                except Exception:
                    logger.exception("OBS の切断通知で例外")

    async def call(self, request_type: str, data: Optional[dict] = None, timeout: float = 10.0) -> dict:
        """要求を送って返事（responseData）を返す。断られたら ObsError。"""
        if not self.connected or self._ws is None:
            raise ObsError("OBS につながっていない")
        rid = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        msg: dict = {"op": 6, "d": {"requestType": request_type, "requestId": rid}}
        if data is not None:
            msg["d"]["requestData"] = data
        try:
            await self._ws.send_str(json.dumps(msg, ensure_ascii=False))
            res = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise ObsError(f"OBS が {request_type} に答えなかった（時間切れ）") from None
        finally:
            self._pending.pop(rid, None)
        st = res.get("requestStatus") or {}
        if not st.get("result"):
            raise ObsError(f"{request_type}: {st.get('comment') or 'OBS が断った'}", int(st.get("code") or 0))
        return res.get("responseData") or {}

    async def emit(self, event_name: str, data: dict) -> None:
        """ブラウザソース（字幕などの重ね絵）へ JS のイベントを送る（obs-browser の emit_event）。"""
        await self.call("CallVendorRequest", {"vendorName": "obs-browser", "requestType": "emit_event",
                                              "requestData": {"event_name": event_name, "event_data": data}},
                        timeout=5.0)

    async def close(self) -> None:
        self._closing = True
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        self.connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None


# ----------------------------------------------------------------------------- OBS 本体・設定ファイル

def ws_config_path() -> Path:
    return Path(os.environ.get("APPDATA", "")) / "obs-studio" / "plugin_config" / "obs-websocket" / "config.json"


def read_ws_config(path: Optional[Path] = None) -> dict:
    """OBS の WebSocket の設定（enabled / port / password / auth）。読めなければ空の dict。

    パスワードは**接続にだけ使い、ログに出さない**。
    """
    p = path or ws_config_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {"enabled": bool(d.get("server_enabled")), "port": int(d.get("server_port") or 4455),
            "password": str(d.get("server_password") or ""), "auth": bool(d.get("auth_required", True))}


def find_obs_exe(explicit: str = "") -> Optional[Path]:
    """obs64.exe の場所（指定 → インストール情報 → よくある場所）。見つからなければ None。"""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    cands: list[Path] = []
    try:
        import winreg

        for root, sub in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                          (winreg.HKEY_LOCAL_MACHINE,
                           r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                          (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")):
            try:
                key = winreg.OpenKey(root, sub)
            except OSError:
                continue
            with key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        with winreg.OpenKey(key, winreg.EnumKey(key, i)) as k:
                            name = str(winreg.QueryValueEx(k, "DisplayName")[0])
                            if not name.startswith("OBS Studio"):
                                continue
                            loc = str(winreg.QueryValueEx(k, "InstallLocation")[0])
                            if loc:
                                cands.append(Path(loc) / "bin" / "64bit" / "obs64.exe")
                    except OSError:
                        continue
    except ImportError:
        pass
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        cands.append(Path(base) / "obs-studio" / "bin" / "64bit" / "obs64.exe")
        cands.append(Path(base) / "Steam" / "steamapps" / "common" / "OBS Studio" / "bin" / "64bit" / "obs64.exe")
    for c in cands:
        if c.is_file():
            return c
    return None


def obs_running() -> bool:
    try:
        import psutil

        return any((p.info.get("name") or "").lower() == "obs64.exe"
                   for p in psutil.process_iter(["name"]))
    except Exception:
        return False


def launch_obs(exe: Path) -> None:
    """OBS を起動する（イブとは切り離す＝イブを閉じても OBS は残る）。

    作業フォルダは obs64.exe の場所にする（OBS はそこから data を探す）。
    `--disable-shutdown-check`: 前回が異常終了だった時の「セーフモードで起動しますか？」で止まらないように。
    """
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(exe), "--disable-shutdown-check"], cwd=str(exe.parent), creationflags=flags,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     close_fds=True)


def obs_color(hex_rgb: str, alpha: int = 255) -> int:
    """「#RRGGBB」→ OBS の色（0xAABBGGRR）。"""
    h = hex_rgb.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (alpha << 24) | (b << 16) | (g << 8) | r


def window_spec(title: str, cls: str, exe: str) -> str:
    """window_capture / wasapi_process_output_capture の "window" 設定（OBS の window-helpers の書式）。

    「タイトル:クラス:実行ファイル」で、各部の「#」は「#22」、「:」は「#3A」に置き換える。
    """
    def enc(s: str) -> str:
        return (s or "").replace("#", "#22").replace(":", "#3A")

    return f"{enc(title)}:{enc(cls)}:{enc(exe)}"
