r"""配信モードの確認（**配信はしない**）。OBS の準備と YouTube の読み取りを、本番と同じ部品で試す。

  $env:PYTHONIOENCODING="utf-8"
  # OBS: 配信用のシーンを作り、字幕のサンプルを出して、配信の画面の写しを保存する（配信は始めない・API 不要）
  & .\.venv\Scripts\python.exe tools\stream_check.py obs --shot stream_shot.png [--chat <配信中の動画の URL か ID>]
  # YouTube: 配信中の動画を探し、チャットを数回読む（読み取りだけ・約 20〜120 単位）
  & .\.venv\Scripts\python.exe tools\stream_check.py youtube [--video <URL か ID>]

obs では、イブのメインウィンドウの代わりに同じタイトルの窓（Live2D 入り）をこのスクリプトが出す。
本物の UI（tools/ui_chat.py）を起動している時は、同じタイトルの窓が2つになるので先に閉じておく。
OBS の WebSocket（ツール → WebSocket サーバー設定）が有効になっている必要がある。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAIN_TITLE = "イブ — Live2D（リップシンク接続）"


# ----------------------------------------------------------------------------- OBS

def _fake_main_window():
    """本物のメインウィンドウと同じタイトル・似た並びの窓（Live2D は本物の部品・素材が無ければ色の板）。"""
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

    from eve.ui.live2d_view import Live2DView, assets_ready

    w = QWidget()
    w.setWindowTitle(MAIN_TITLE)
    root = QVBoxLayout(w)
    root.addWidget(QLabel("（メニューの代わり）"))
    stage = QHBoxLayout()
    ok, _why = assets_ready()
    if ok:
        view = Live2DView()
    else:
        view = QLabel("Live2D の代わり")
        view.setStyleSheet("background:#c9b6e4; font-size:28px;")
    view.setMinimumSize(320, 214)
    stage.addWidget(view, 5)
    stage.addWidget(QLabel("（ログの代わり）"), 3)
    root.addLayout(stage, 1)
    bottom = QLabel("（発話テキストと入力欄の代わり）")
    bottom.setMinimumHeight(170)
    root.addWidget(bottom)
    w.resize(1280, 860)
    return w, view, ok


async def _obs_check(args, main, geom) -> int:
    from eve.stream.manager import StreamError, StreamManager, StreamSettings
    from eve.stream.scene import SRC_LIVE2D, SceneBuilder, chat_url
    from eve.stream.youtube import video_id_from

    settings = StreamSettings.from_config()
    m = StreamManager(settings, is_free=lambda: False, submit=lambda t, k: None)
    try:
        obs = await m.connect_obs()
    except StreamError as e:
        print(f"❌ OBS につながらない: {e}")
        return 1
    try:
        v = await obs.call("GetVersion")
        print(f"OBS {v.get('obsVersion')} / WebSocket {v.get('obsWebSocketVersion')}")
        sb = SceneBuilder(obs, scene=settings.scene, overlay_path=Path(m._runtime) / "overlay.html")
        vid = video_id_from(args.chat or "")
        await sb.ensure(main=main, task=None, video_id=vid)
        print("シーン:", settings.scene, "／ 作ったソース:", "・".join(sb.created) or "（全部あった）",
              "／ キャンバス:", sb.canvas)
        # OBS の選択肢の値（窓の探し方・取り込み方式）を確かめる（scene.py の定数と合っているか）
        for prop in ("priority", "method"):
            try:
                items = (await obs.call("GetInputPropertiesListPropertyItems",
                                        {"inputName": SRC_LIVE2D, "propertyName": prop})).get("propertyItems") or []
                print(f"  {prop}: " + " / ".join(f"{it.get('itemValue')}={it.get('itemName')}" for it in items))
            except Exception as e:  # noqa: BLE001
                print(f"  {prop}: 取れない（{e}）")
        try:
            items = (await obs.call("GetInputPropertiesListPropertyItems",
                                    {"inputName": SRC_LIVE2D, "propertyName": "window"})).get("propertyItems") or []
            mine = [it for it in items if MAIN_TITLE in str(it.get("itemName"))]
            print("  OBS から見えるイブの窓:", [it.get("itemValue") for it in mine] or "（見つからない）")
        except Exception as e:  # noqa: BLE001
            print(f"  window: 取れない（{e}）")
        # 本番と同じく先にこのシーンを出す（OBS は表示していないシーンのソースを動かさない）
        await obs.call("SetCurrentProgramScene", {"sceneName": settings.scene})
        print("  OBS の番組のシーンを", settings.scene, "にした")
        ok = await sb.update_crop(geom, wait_sec=8.0)
        t = (await obs.call("GetSceneItemTransform", {"sceneName": settings.scene,
                                                       "sceneItemId": sb.items[SRC_LIVE2D]})).get("sceneItemTransform")
        print("Live2D の切り出し:", "OK" if ok else "できていない",
              {k: t.get(k) for k in ("sourceWidth", "sourceHeight", "cropLeft", "cropTop", "cropRight", "cropBottom",
                                     "width", "height")})
        # 字幕のサンプル（読み込みを待ってから送る・状態は何度か送る）
        await asyncio.sleep(3.0)
        for _ in range(2):
            await obs.emit("eve", {"type": "state", "chat": "live" if vid else "waiting",
                                   "note": "チャットを準備しています…"})
            await obs.emit("eve", {"type": "comment", "kind": "superchat", "name": "テストさん",
                                   "text": "配信おめでとう！これからも楽しみにしてます", "badge": "スーパーチャット ￥500"})
            await obs.emit("eve", {"type": "say", "text": "わあ、ありがとうございます！"})
            await obs.emit("eve", {"type": "say", "text": "初配信なので、ちょっと緊張してますけど、ゆっくりしていってくださいね。"})
            await asyncio.sleep(1.0)
        await asyncio.sleep(max(0.0, args.wait))
        if args.shot:
            shot = await obs.call("GetSourceScreenshot", {"sourceName": settings.scene, "imageFormat": "png",
                                                          "imageWidth": 1280}, timeout=30.0)
            data = str(shot.get("imageData") or "")
            raw = base64.b64decode(data.split(",", 1)[1] if "," in data else data)
            Path(args.shot).write_bytes(raw)
            print("配信の画面の写し:", Path(args.shot).resolve(), f"（{len(raw)} バイト）")
        if vid:
            print("チャット欄:", chat_url(vid))
        return 0
    finally:
        await obs.close()


def cmd_obs(args) -> int:
    from eve.ui import web_view

    web_view.preinit()
    import live2d.v3 as live2d
    from PySide6.QtCore import QPoint, QTimer
    from PySide6.QtWidgets import QApplication

    from eve.stream.scene import WindowInfo

    live2d.init()
    app = QApplication.instance() or QApplication([])
    w, view, has_l2d = _fake_main_window()
    w.show()
    app.processEvents()
    result = {"rc": 1}
    done = threading.Event()

    def start() -> None:
        from eve.ui.app import _process_exe, _window_class

        p = view.mapTo(w, QPoint(0, 0))
        geom = {"win": (w.width(), w.height()), "view": (p.x(), p.y(), view.width(), view.height())}
        main = WindowInfo(w.windowTitle(), _window_class(int(w.winId())), _process_exe())
        print("窓:", main, "／ Live2D の位置:", geom, "／ Live2D:", "本物" if has_l2d else "代わりの板")

        def worker() -> None:
            try:
                result["rc"] = asyncio.run(_obs_check(args, main, geom))
            except Exception:
                logging.exception("確認の途中で例外")
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()

    def poll() -> None:
        if done.is_set():
            app.quit()

    QTimer.singleShot(1500, start)  # 窓が出て Live2D が描かれてから
    t = QTimer()
    t.timeout.connect(poll)
    t.start(200)
    try:
        app.exec()
    finally:
        if hasattr(view, "release"):
            view.release()
        live2d.dispose()
    return result["rc"]


# ----------------------------------------------------------------------------- YouTube

async def _yt_check(args) -> int:
    from eve.config import Config
    from eve.stream.youtube import ChatReader, YouTubeClient, YouTubeError, video_id_from

    if not Config.YOUTUBE_API_KEY:
        print("❌ .env に YOUTUBE_API_KEY が無い")
        return 1
    c = YouTubeClient(Config.YOUTUBE_API_KEY)
    try:
        vid = video_id_from(args.video or "")
        if vid:
            live = await c.video(vid)
            print("指定の動画:", live)
        elif Config.TARGET_CHANNEL_ID:
            print("チャンネル:", Config.TARGET_CHANNEL_ID, "／", await c.channel_title(Config.TARGET_CHANNEL_ID))
            live, upcoming = await c.find_live(Config.TARGET_CHANNEL_ID)
            print("配信中:", live, "／ 予定:", upcoming)
        else:
            print("❌ TARGET_CHANNEL_ID も --video も無い")
            return 1
        if live is None or live.state != "live" or not live.chat_id:
            print("（配信中の動画が無いので、チャットは読まない）")
            return 0
        r = ChatReader(c, live.chat_id, min_interval=float(Config.STREAM_CHAT_POLL_SEC))
        for i in range(args.polls):
            msgs = await r.poll()
            print(f"読み{i + 1}: 新しい {len(msgs)} 件（次は {r.interval:.1f} 秒後）"
                  + ("　※最初の1回は過去分を読み捨てる" if i == 0 else ""))
            for msg in msgs[-5:]:
                from eve.stream.manager import format_comment

                print("   ", format_comment(msg) if msg.is_conversation else f"（{msg.kind}）")
            if i + 1 < args.polls:
                await asyncio.sleep(r.interval)
        return 0
    except YouTubeError as e:
        print(f"❌ YouTube API: {e.reason}（HTTP {e.status}）")
        return 1
    finally:
        print(f"使った割り当て: 約 {c.units} 単位")
        await c.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("obs", help="OBS のシーンを作って写しを保存（配信はしない）")
    o.add_argument("--shot", default="stream_shot.png", help="配信の画面の写しの保存先（空なら保存しない）")
    o.add_argument("--chat", default="", help="チャット欄に出す配信中の動画（URL か ID）")
    o.add_argument("--wait", type=float, default=4.0, help="写しを撮る前に待つ秒（チャットの読み込み）")
    y = sub.add_parser("youtube", help="配信中の動画を探してチャットを読む（読み取りだけ）")
    y.add_argument("--video", default="", help="URL か ID（無ければ TARGET_CHANNEL_ID の配信を探す）")
    y.add_argument("--polls", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "obs":
        return cmd_obs(args)
    return asyncio.run(_yt_check(args))


if __name__ == "__main__":
    raise SystemExit(main())
