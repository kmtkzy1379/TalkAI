"""見た目（配色・枠画像・ボタンの状態表現）を1箇所に集める。

**なぜ1ファイルに集めるか**: 色や角丸をウィジェット側に散らすと、意匠を直すたびに
あちこちを触ることになる。ここだけ見れば全体の見た目が分かる状態を保つ。

**9スライス**: `log_bvackground.png` / `text_background.png` は角に装飾があるので、
単純に引き伸ばすと角が歪む。QSS の `border-image` は**四隅を固定して辺だけ伸ばす**
（CSS の 9-slice と同じ）ので、枠つき画像はこれで貼る。
"""
from __future__ import annotations

from pathlib import Path

UI_DIR = Path(__file__).resolve().parent.parent.parent / "eve_ui"
LOG_FRAME = UI_DIR / "log_bvackground.png"      # ファイル名の綴りは素材のまま（改名は別途）
TEXT_FRAME = UI_DIR / "text_background.png"

# 素材が白基調なので、文字は黒寄り・アクセントは彩度低めで合わせる
INK = "#2b2b33"
SUB = "#6b6b76"
LINE = "#d8d5dd"
BG = "#f7f6f9"

# 発話の種別ごとの色（ユーザ裁定 2026-08-14: 会話ログは会話だけ。自発発話は色で分ける）
SPEAKER_COLORS = {
    "user": "#1f6feb",       # ユーザ
    "eve": "#2b2b33",        # イブ（応答）
    "auto": "#8250df",       # 自発発話（イブが自分から話した）
    "system": "#6b6b76",     # 起動・状態など
}
SPEAKER_LABELS = {"user": "あなた", "eve": "イブ", "auto": "イブ（自分から）", "system": "システム"}


def _url(p: Path) -> str:
    return str(p).replace("\\", "/")


def frame_qss(image: Path, border: int = 120, radius: int = 0) -> str:
    """9スライスで枠画像を貼る QSS。`border` は四隅として固定する幅（素材の装飾サイズ）。"""
    if not image.exists():
        return f"background: white; border: 1px solid {LINE}; border-radius: 10px;"
    return (f"border-image: url({_url(image)}) {border} {border} {border} {border} stretch stretch;"
            f"border-width: {border // 8}px; border-radius: {radius}px;")


def menu_button_qss(state: str) -> str:
    """メニューボタンの見た目。**4状態**を色で区別する（ユーザ要件: OFF は半透明グレー）。

    on       : 有効（不透明・アクセント）
    off      : 無効（**半透明グレー**）
    pending  : 切替中（点滅させずに色で示す。押した直後〜安全地点に着くまで）
    disabled : そもそも使えない（構成で無効・例: WS_ENABLED=0）
    """
    base = ("QPushButton{border-radius:9px; padding:7px 14px; font-weight:bold;"
            "border:1px solid %s;}" % LINE)
    if state == "on":
        return base + ("QPushButton{background:#eaf3ff; color:#1f6feb; border-color:#a9ccf5;}"
                       "QPushButton:hover{background:#dceaff;}")
    if state == "pending":
        return base + ("QPushButton{background:#fff6de; color:#9a6700; border-color:#e8d08a;}")
    if state == "disabled":
        return base + ("QPushButton{background:#f1f1f3; color:#b6b6bd; border-color:#e6e6ea;}")
    # off: 半透明グレー
    return base + ("QPushButton{background:rgba(140,140,150,0.18); color:rgba(70,70,80,0.55);"
                   "border-color:rgba(140,140,150,0.28);}"
                   "QPushButton:hover{background:rgba(140,140,150,0.28);}")


def app_qss() -> str:
    """アプリ全体の下地。"""
    return f"""
    QWidget {{ color: {INK}; font-size: 13px; }}
    QLabel#speech {{ font-size: 17px; color: {INK}; padding: 10px 16px; }}
    QLabel#hint {{ color: {SUB}; }}
    QTabBar::tab {{ padding: 5px 12px; }}
    QTabBar::tab:selected {{ color: {SPEAKER_COLORS['user']}; font-weight: bold; }}
    QTextEdit, QPlainTextEdit {{ background: transparent; border: none; }}
    QLineEdit {{ border: 1px solid {LINE}; border-radius: 8px; padding: 7px 10px;
                 background: white; }}
    """
