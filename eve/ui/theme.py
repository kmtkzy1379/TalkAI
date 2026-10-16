"""見た目（配色・ボタンの状態表現）を1箇所に集める。

**なぜ1ファイルに集めるか**: 色や角丸をウィジェット側に散らすと、意匠を直すたびに
あちこちを触ることになる。ここだけ見れば全体の見た目が分かる状態を保つ。

**枠素材（`eve_ui/*_background.png`）は使わない**（ユーザ裁定 2026-08-14）。
9スライスで貼る実装を数回試みたがいずれも面が入れ子になり、意匠として成立しなかった。
Live2D 以外は素の配置に留め、意匠は後日あらためて扱う。素材ファイル自体は残す。
"""
from __future__ import annotations

# 文字は黒寄り・アクセントは彩度低めで合わせる
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
    "viewer": "#c2410c",     # 配信の視聴者のコメント
}
SPEAKER_LABELS = {"user": "あなた", "eve": "イブ", "auto": "イブ（自分から）", "system": "システム",
                  "viewer": "リスナー"}


def menu_button_qss(state: str) -> str:
    """メニューボタンの見た目。**4状態**を色で区別する（ユーザ要件: OFF は半透明グレー）。

    on       : 有効（不透明・アクセント）
    off      : 無効（**半透明グレー**）
    pending  : 切替中（点滅させずに色で示す。押した直後〜安全地点に着くまで）
    disabled : そもそも使えない（構成で無効・例: WS_ENABLED=0）
    """
    base = ("QPushButton{border-radius:9px; padding:7px 14px; font-weight:bold;"
            "border:1px solid %s;}" % LINE)
    if state == "live":  # 配信中（赤・目立たせる）
        return base + ("QPushButton{background:#e5484d; color:#fff; border-color:#c93a3f;}"
                       "QPushButton:hover{background:#d93d42;}")
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
    /* **面を持つのは外側の1枚だけ**（内側に持たせると「箱の中の箱」になる） */
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
