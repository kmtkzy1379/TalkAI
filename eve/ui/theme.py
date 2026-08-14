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


def frame_qss(image: Path, *, scale: float = 0.35,
              inner_only: bool = False) -> tuple[str, tuple[int, int, int, int]]:
    """枠画像を 9スライスで貼る QSS と、**中身を置くべき余白**を返す。

    素材は実描画範囲で切り出してから使う（`assets.prepare`）。切り出さないと絵が
    キャンバスの一部にしか来ず、内側の紙の縁が2本目の枠に見える（実測で確認）。

    `scale` は「素材の余白 → 画面上の余白」の縮尺。素材は 1000px 級なので、そのまま
    使うと枠だけで画面が埋まる。四隅の装飾が潰れない範囲で小さくする。
    """
    from .assets import prepare

    f = prepare(image, inner_only=inner_only)
    if not f.ok:
        return (f"background: {BG}; border: 1px solid {LINE}; border-radius: 12px;",
                (14, 12, 14, 12))
    # border-width は「画面上で四隅に割く幅」。素材側の slice と比を合わせると歪まない。
    bw = tuple(max(6, int(v * scale)) for v in (f.top, f.right, f.bottom, f.left))
    qss = (f"border-image: url({_url(f.path)}) {f.top} {f.right} {f.bottom} {f.left} "
           f"stretch stretch;"
           f"border-style: solid;"
           f"border-width: {bw[0]}px {bw[1]}px {bw[2]}px {bw[3]}px;")
    pad = (bw[3] + 10, bw[0] + 8, bw[1] + 10, bw[2] + 8)  # 左,上,右,下
    return qss, pad


def panel_qss(paper: str = "#fdfcfa") -> str:
    """枠素材が合わない場所（横に長い帯など）で使う、無地の落ち着いたパネル。

    hallmark の「抑制」に従い、装飾を無理に引き伸ばすより無地で整える方を選ぶ。
    """
    return (f"background: {paper}; border: 1px solid {LINE}; border-radius: 14px;")


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
    QLabel#speech {{ font-size: 19px; line-height: 155%; color: {INK}; background: transparent; }}
    QLabel#hint {{ color: {SUB}; background: transparent; }}
    /* **枠素材の上に置くものは面を持たない**（持たせると「箱の中の箱」になる） */
    QTabWidget::pane {{ border: none; background: transparent; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{ padding: 4px 11px; background: transparent; color: {SUB};
                    border: none; margin-right: 2px; }}
    QTabBar::tab:selected {{ color: {SPEAKER_COLORS['user']}; font-weight: bold;
                             border-bottom: 2px solid {SPEAKER_COLORS['user']}; }}
    QTextEdit, QPlainTextEdit {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 7px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: rgba(120,115,130,0.30); border-radius: 3px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QLineEdit {{ border: 1px solid {LINE}; border-radius: 9px; padding: 8px 12px;
                 background: rgba(255,255,255,0.86); }}
    QProgressBar {{ background: rgba(120,115,130,0.16); border: none; border-radius: 4px; }}
    QProgressBar::chunk {{ background: {SPEAKER_COLORS['user']}; border-radius: 4px; }}
    """
