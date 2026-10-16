"""配信用の OBS シーンを用意する（無ければ作る・あれば配置は触らない）。

## 配置（ユーザの見本 `eve_ui/Sample_layout.png` と同じ並び）
左上に Live2D（イブのウィンドウから Live2D の部分だけを切り出す）、右にチャット、下にイブの発話（字幕）。
イブが画面に出した物（タスクウィンドウ）は Live2D の左上に重ねる（出していない間は何も映らない）。
1920×1080 を基準に書き、OBS のキャンバスの大きさに合わせて伸縮する。

## 既にあるシーンは壊さない
2回目以降の配信では、ユーザが OBS で動かした配置・重ね順をそのまま使う。毎回書き換えるのは
「中身」（チャットの URL・字幕の HTML・どの窓を映すか）と、窓の大きさで変わる Live2D の切り出しだけ。
消されたソースは作り直す（その時だけ既定の位置に置く）。
チャット欄の CSS は、**前回イブが入れたままの時だけ**新しい版に入れ替える（入れた CSS の指紋を手元に控えて比べる＝
ユーザが OBS で書き換えた CSS は触らない）。控えが無ければ触らない。

## 映すもの
- Live2D: イブのメインウィンドウを Windows 10 の方式（WGC）で映す。**最小化すると映らない**（他の窓の下にあるのは可）。
  キャプチャ除外（UI_HIDE_FROM_CAPTURE）が効いていると OBS からも消えるので、配信中は UI 側で外す。
- チャット: YouTube のポップアウトチャット（`live_chat?is_popout=1&v=…`）をブラウザソースで。API を使わない
  ＝割り当てを食わず、遅れも無い。見た目は CSS（`assets/chat.css`）で整える。
- 字幕・読んでいるコメント: 手元の HTML（`assets/overlay.html`）。中身は obs-browser の emit_event で送る。
- イブの声: 「イブ 声」（イブのアプリの音だけを取り込む）。普段はデスクトップ音声に入っているのでミュートしておき、
  デスクトップ音声が切れている間だけ入れる（二重にしない・manager の `_route_voice`）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .obs import ObsClient, ObsError, obs_color

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"

# 1920×1080 での位置 (x, y, 幅, 高さ)
DESIGN_W, DESIGN_H = 1920, 1080
SLOT_LIVE2D = (28, 28, 1308, 782)
SLOT_CHAT = (1364, 28, 528, 782)        # チャットの枠全体（字幕の HTML が枠と見出しを描く）
CHAT_HEAD = 52                          # 見出しの帯の高さ（overlay.html の HEAD と同じ）
SLOT_CHAT_BODY = (SLOT_CHAT[0], SLOT_CHAT[1] + CHAT_HEAD, SLOT_CHAT[2], SLOT_CHAT[3] - CHAT_HEAD)
SLOT_SUBTITLE = (28, 838, 1864, 214)
SLOT_TASK = (52, 52, 460, 330)
# 背景の色（字幕の HTML の角の塗りと同じ色にする＝Live2D の角が丸く見える）
BG_HEX = "#EAE5EF"
# 縦に削る時、上から削る割合（残りは下から削る）。既定の窓（1280×860）の実写で、Live2D の表示は上に天井が広く
# （耳の先が上から約3割）、下は机。0.6 で耳の先が上から約2割・机が少し残る（0.25 だと天井が広く机が切れた）
LIVE2D_TOP_BIAS = 0.6

# ソースの名前（OBS では名前が全シーン共通なので「イブ」を頭に付ける）
SRC_BG = "イブ 背景"
SRC_LIVE2D = "イブ Live2D"
SRC_TASK = "イブ 作業画面"
SRC_CHAT = "イブ チャット"
SRC_OVERLAY = "イブ 字幕"
SRC_VOICE = "イブ 声"

# 下から上への重ね順（新しく作った時だけ並べる）
ORDER = (SRC_BG, SRC_LIVE2D, SRC_TASK, SRC_CHAT, SRC_OVERLAY, SRC_VOICE)

# OBS の bounds の種類（obs-websocket の ObsBoundsType）
BOUNDS_NONE = "OBS_BOUNDS_NONE"
BOUNDS_STRETCH = "OBS_BOUNDS_STRETCH"
BOUNDS_INNER = "OBS_BOUNDS_SCALE_INNER"
ALIGN_TOP_LEFT = 5  # OBS_ALIGN_TOP | OBS_ALIGN_LEFT
WGC = 2  # window_capture の method: Windows 10 (1903 以降)
# 窓の探し方（OBS の window-helpers.h: CLASS=0 / TITLE=1 / EXE=2）。TITLE＝「タイトルが一致する窓だけ」
# （CLASS だと、タイトルが違っても同じ種類の窓＝イブのタスクウィンドウ等を映してしまう）
MATCH_TITLE = 1


@dataclass(frozen=True)
class WindowInfo:
    """OBS に映す窓（UI スレッドが調べた値のコピー）。"""

    title: str
    cls: str
    exe: str


def chat_url(video_id: str) -> str:
    return f"https://www.youtube.com/live_chat?is_popout=1&v={video_id}" if video_id else "about:blank"


def scaled(slot: tuple, canvas: tuple[int, int]) -> tuple[float, float, float, float]:
    sx, sy = canvas[0] / DESIGN_W, canvas[1] / DESIGN_H
    x, y, w, h = slot
    return x * sx, y * sy, w * sx, h * sy


def live2d_crop(geom: dict, src_w: int, src_h: int, slot_w: float, slot_h: float,
                top_bias: float = LIVE2D_TOP_BIAS) -> tuple[int, int, int, int]:
    """イブのウィンドウの映像から Live2D の部分を、枠の縦横比で切り出す（左, 上, 右, 下の削り量・ピクセル）。

    geom は UI が測った論理座標（`win`=(幅, 高さ)・`view`=(x, y, 幅, 高さ)）。映像のピクセルとの比で換算する
    （DPI の拡大率はここで吸収される）。枠より横長なら左右を、縦長なら上下を削る。
    """
    try:
        win_w, win_h = (float(v) for v in geom["win"])
        vx, vy, vw, vh = (float(v) for v in geom["view"])
    except (KeyError, TypeError, ValueError):
        return 0, 0, 0, 0
    if min(win_w, win_h, vw, vh, src_w, src_h, slot_w, slot_h) <= 0:
        return 0, 0, 0, 0
    sx, sy = src_w / win_w, src_h / win_h
    aspect = slot_w / slot_h
    if vw / vh > aspect:
        cw, ch = vh * aspect, vh
        cx, cy = vx + (vw - cw) / 2.0, vy
    else:
        cw, ch = vw, vw / aspect
        cx, cy = vx, vy + (vh - ch) * top_bias
    left, top = round(cx * sx), round(cy * sy)
    right = max(0, src_w - round((cx + cw) * sx))
    bottom = max(0, src_h - round((cy + ch) * sy))
    return max(0, left), max(0, top), right, bottom


def overlay_html(canvas: tuple[int, int]) -> str:
    """字幕の HTML（配置の数字を埋め込む＝Python と HTML で数字を二重に持たない）。"""
    layout = {"W": DESIGN_W, "H": DESIGN_H, "live2d": SLOT_LIVE2D, "chat": SLOT_CHAT, "head": CHAT_HEAD,
              "subtitle": SLOT_SUBTITLE, "bg": BG_HEX, "canvas": list(canvas)}
    tpl = (ASSETS / "overlay.html").read_text(encoding="utf-8")
    return tpl.replace("/*LAYOUT*/null", json.dumps(layout))


def chat_css() -> str:
    return (ASSETS / "chat.css").read_text(encoding="utf-8")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _transform(slot, canvas, *, bounds: str = BOUNDS_INNER, crop=(0, 0, 0, 0)) -> dict:
    x, y, w, h = scaled(slot, canvas)
    t = {"positionX": x, "positionY": y, "alignment": ALIGN_TOP_LEFT, "rotation": 0.0,
         "cropLeft": crop[0], "cropTop": crop[1], "cropRight": crop[2], "cropBottom": crop[3]}
    if bounds == BOUNDS_NONE:
        t.update({"boundsType": BOUNDS_NONE, "scaleX": 1.0, "scaleY": 1.0})
    else:
        t.update({"boundsType": bounds, "boundsWidth": max(1.0, w), "boundsHeight": max(1.0, h),
                  "boundsAlignment": 0})
    return t


class SceneBuilder:
    """シーンとソースを用意し、配信中に変わるもの（チャットの URL・Live2D の切り出し）を更新する。"""

    def __init__(self, obs: ObsClient, *, scene: str, overlay_path: Path) -> None:
        self.obs = obs
        self.scene = scene
        self.overlay_path = overlay_path
        self.canvas: tuple[int, int] = (DESIGN_W, DESIGN_H)
        self.items: dict[str, int] = {}
        self.created: list[str] = []
        self.voice_ok = False  # イブの声だけを取り込むソースを作れたか

    async def ensure(self, *, main: WindowInfo, task: Optional[WindowInfo], video_id: str = "") -> None:
        vs = await self.obs.call("GetVideoSettings")
        self.canvas = (int(vs.get("baseWidth") or DESIGN_W), int(vs.get("baseHeight") or DESIGN_H))
        w, h = self.canvas
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        self.overlay_path.write_text(overlay_html(self.canvas), encoding="utf-8")

        kinds = set((await self.obs.call("GetInputKindList", {"unversioned": False})).get("inputKinds") or [])
        color_kind = max((k for k in kinds if k.startswith("color_source")), default="color_source_v3")
        scenes = await self.obs.call("GetSceneList")
        if self.scene not in [s.get("sceneName") for s in scenes.get("scenes") or []]:
            await self.obs.call("CreateScene", {"sceneName": self.scene})
            logger.info("🎬 OBS にシーン「%s」を作った", self.scene)
        inputs = {i.get("inputName"): i.get("inputKind")
                  for i in (await self.obs.call("GetInputList")).get("inputs") or []}
        have = {it.get("sourceName"): int(it.get("sceneItemId"))
                for it in (await self.obs.call("GetSceneItemList", {"sceneName": self.scene})).get("sceneItems") or []}

        from .obs import window_spec

        main_spec = window_spec(main.title, main.cls, main.exe)
        cx, cy, cw, ch = scaled(SLOT_CHAT_BODY, self.canvas)
        # (名前, 種類, 作る時の設定, 毎回入れ直す設定のキー, 作った時の位置)。既にあるソースは「毎回」の分だけ
        # 入れ直す（ユーザが OBS で変えた色・CSS・大きさを上書きしない）
        want: list[tuple[str, str, dict, tuple, Optional[dict]]] = [
            (SRC_BG, color_kind, {"color": obs_color(BG_HEX), "width": w, "height": h}, ("width", "height"),
             _transform((0, 0, DESIGN_W, DESIGN_H), self.canvas, bounds=BOUNDS_STRETCH)),
            # 切り出しは窓の大きさが分かってから（update_crop）
            (SRC_LIVE2D, "window_capture", {"window": main_spec, "method": WGC,
                                            "priority": MATCH_TITLE, "cursor": False, "client_area": True},
             ("window",),
             _transform(SLOT_LIVE2D, self.canvas)),
            (SRC_CHAT, "browser_source", {"url": chat_url(video_id), "is_local_file": False,
                                          "width": round(cw), "height": round(ch), "css": chat_css(),
                                          "reroute_audio": True, "shutdown": False}, ("url",),
             _transform(SLOT_CHAT_BODY, self.canvas, bounds=BOUNDS_NONE)),
            (SRC_OVERLAY, "browser_source", {"is_local_file": True, "local_file": str(self.overlay_path),
                                             "width": w, "height": h, "css": "", "reroute_audio": True,
                                             "shutdown": False, "restart_when_active": False},
             ("is_local_file", "local_file", "width", "height"),
             _transform((0, 0, DESIGN_W, DESIGN_H), self.canvas, bounds=BOUNDS_NONE)),
        ]
        if task is not None:
            want.insert(2, (SRC_TASK, "window_capture",
                            {"window": window_spec(task.title, task.cls, task.exe), "method": WGC,
                             "priority": MATCH_TITLE, "cursor": False, "client_area": True}, ("window",),
                            _transform(SLOT_TASK, self.canvas)))
        # イブの声だけを取り込むソース（デスクトップ音声が切れている間だけ入れる・入れ切りは manager の _route_voice）
        if "wasapi_process_output_capture" in kinds:
            want.append((SRC_VOICE, "wasapi_process_output_capture", {"window": main_spec, "priority": MATCH_TITLE},
                         ("window",), None))
            self.voice_ok = True
        else:
            logger.warning("OBS にアプリの音声を取り込むソースが無い（Windows 10 2004 以降・OBS 28 以降）"
                           "→ イブの声は OBS のデスクトップ音声に任せる")

        self.created = []
        for name, kind, settings, dynamic, transform in want:
            if name in have:
                item = have[name]
                await self.obs.call("SetInputSettings", {"inputName": name, "inputSettings": {
                    k: settings[k] for k in dynamic}})
            elif name in inputs:
                # 別のシーンに同じ名前のソースがある（シーンから外しただけ）→ 中身はそのまま入れ直す
                await self.obs.call("SetInputSettings", {"inputName": name, "inputSettings": {
                    k: settings[k] for k in dynamic}})
                item = int((await self.obs.call("CreateSceneItem", {"sceneName": self.scene, "sourceName": name}))
                           .get("sceneItemId"))
                self.created.append(name)
            else:
                res = await self.obs.call("CreateInput", {"sceneName": self.scene, "inputName": name,
                                                          "inputKind": kind, "inputSettings": settings})
                item = int(res.get("sceneItemId"))
                self.created.append(name)
            self.items[name] = item
            if name in self.created and transform is not None:
                await self.obs.call("SetSceneItemTransform", {"sceneName": self.scene, "sceneItemId": item,
                                                              "sceneItemTransform": transform})
        if SRC_CHAT in self.created:
            self._save_css_mark(chat_css())
        elif SRC_CHAT in self.items:
            await self._upgrade_chat_css()
        if SRC_OVERLAY in self.items and SRC_OVERLAY not in self.created:
            # 字幕の HTML は今書き直した。OBS は同じファイルを開いたままだと読み直さないので、読み直させる
            try:
                await self.obs.call("PressInputPropertiesButton", {"inputName": SRC_OVERLAY,
                                                                   "propertyName": "refreshnocache"})
            except ObsError as e:
                logger.warning("字幕の HTML を読み直させられなかった（OBS でソースを更新してください）: %s", e)
        if self.created:
            # 新しく作った物があれば重ね順を揃える（ユーザが並べ替えた後に1つだけ消して作り直した時も、全体の順を保つ）
            order = [n for n in ORDER if n in self.items]
            for idx, name in enumerate(order):
                try:
                    await self.obs.call("SetSceneItemIndex", {"sceneName": self.scene,
                                                              "sceneItemId": self.items[name],
                                                              "sceneItemIndex": idx})
                except ObsError:
                    logger.warning("OBS の重ね順を揃えられなかった: %s", name)
            logger.info("🎬 OBS のソースを用意した: %s", "・".join(self.created))

    # --- チャット欄の CSS の入れ替え（前回イブが入れたままの時だけ）---

    def _css_mark(self) -> Path:
        return self.overlay_path.parent / "chat_css.sha256"

    def _save_css_mark(self, css: str) -> None:
        try:
            self._css_mark().write_text(_sha(css), encoding="ascii")
        except OSError:
            logger.warning("チャット欄の CSS の控えを書けなかった（次から CSS を新しくできない）")

    async def _upgrade_chat_css(self) -> None:
        want = chat_css()
        try:
            cur = str(((await self.obs.call("GetInputSettings", {"inputName": SRC_CHAT})).get("inputSettings") or {})
                      .get("css") or "")
        except ObsError:
            return
        if cur == want:
            self._save_css_mark(want)
            return
        try:
            mark = self._css_mark().read_text(encoding="ascii").strip()
        except OSError:
            return  # 控えが無い＝誰が入れた CSS か分からない → 触らない
        if _sha(cur) != mark:
            return  # ユーザが OBS で書き換えた → そのまま使う
        await self.obs.call("SetInputSettings", {"inputName": SRC_CHAT, "inputSettings": {"css": want}})
        self._save_css_mark(want)
        logger.info("🎬 チャット欄の CSS を新しい版にした（前回イブが入れたままだったので）")

    async def update_crop(self, geom: Optional[dict], *, wait_sec: float = 3.0) -> bool:
        """Live2D の切り出しを今の窓の大きさに合わせる。映像の大きさがまだ分からなければ少し待つ。"""
        item = self.items.get(SRC_LIVE2D)
        if item is None or not geom:
            return False
        deadline = asyncio.get_running_loop().time() + wait_sec
        while True:
            t = (await self.obs.call("GetSceneItemTransform", {"sceneName": self.scene, "sceneItemId": item})
                 ).get("sceneItemTransform") or {}
            sw, sh = int(t.get("sourceWidth") or 0), int(t.get("sourceHeight") or 0)
            if sw > 0 and sh > 0:
                break
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("OBS がイブのウィンドウを映せていない（最小化している・タイトルが違う？）")
                return False
            await asyncio.sleep(0.3)
        x, y, w, h = scaled(SLOT_LIVE2D, self.canvas)
        left, top, right, bottom = live2d_crop(geom, sw, sh, w, h)
        await self.obs.call("SetSceneItemTransform", {
            "sceneName": self.scene, "sceneItemId": item,
            "sceneItemTransform": {"cropLeft": left, "cropTop": top, "cropRight": right, "cropBottom": bottom}})
        return True

    async def set_chat(self, video_id: str) -> None:
        if SRC_CHAT in self.items:
            await self.obs.call("SetInputSettings", {"inputName": SRC_CHAT,
                                                     "inputSettings": {"url": chat_url(video_id)}})
