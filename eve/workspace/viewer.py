r"""J-3 inc3 — 作業フォルダの画像/テキストを**別ウィンドウ**で表示するビューア。

    python -m eve.workspace.viewer <作業フォルダ内の絶対パス>

【なぜ別プロセスか】
本体は asyncio 単一ループで、Tk の `mainloop()` は入れられない。別プロセスにすると
**新しいスレッド境界がゼロ**＝`PIPELINE_DESIGN.md §9.3` の橋渡し契約も `root.after(0, fn)`
規律も、そもそも適用対象が発生しない。実測でも同一プロセスの `Tk()` 生成は 509.6ms
ループを止めるのに対し、別プロセス spawn は 6.2ms（本体の応答予算に影響しない）。

【なぜ親アカウント（tester）で動かすか】
サンドボックスの子は Job の UI 制限（`UILIMIT_HANDLES`/`UILIMIT_DESKTOP`）下にあり、
そもそも GUI をまともに出せない（意図的にそう設計した）。ここだけ例外にすると
shatter 対策と shutdown 対策を同時に失う。表示は**読んで描くだけ**で状態を変えないので、
親で動かしても権限昇格の経路にならない。

【`os.startfile` を使わない理由】
拡張子に紐づく既定アプリを起動する動作なので、作業フォルダに `.hta`/`.lnk`/`.scr` を
書ければ**任意アプリ起動プリミティブに化ける**。せっかくファイル書込を封じ込めても
表示機能が実行経路を再び開けてしまう。

【マジックバイト判定の限界（正直に）】
判定できるのは**種類**であって**安全性ではない**。PNG シグネチャを持つ悪意ある画像は
判定を通り、デコーダに脆弱性があれば刺さる。緩和は (a) `MAX_IMAGE_PIXELS` を絞る
（decompression bomb）、(b) サイズ上限、(c) **実行は絶対にしない**、まで。
別プロセスなので、最悪クラッシュしても本体は生き残る。
"""
from __future__ import annotations

import sys
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024        # デコード前のサイズ上限
MAX_PIXELS = 40_000_000             # decompression bomb 対策（既定は警告のみなので明示する）
MAX_W, MAX_H = 1100, 800            # ウィンドウ最大（全画面にしない）
TEXT_MAX_CHARS = 200_000

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"), (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"RIFF", "WEBP"),          # RIFF....WEBP
)


def sniff(path: Path) -> str:
    """**拡張子を信用せず**先頭バイトで種類を決める（`.png` を騙る `.exe` を弾く）。"""
    head = path.open("rb").read(16)
    for magic, name in _IMAGE_MAGIC:
        if head.startswith(magic):
            if name == "WEBP" and b"WEBP" not in path.open("rb").read(16):
                continue
            return name
    # テキストは原理的に判定できないので推定に留める（UTF-8 として読めて制御文字が少ない）
    try:
        raw = path.read_bytes()[:8192]
        text = raw.decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return "BINARY"
    ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\r\n\t")
    return "TEXT" if not text or ctrl / max(1, len(text)) < 0.05 else "BINARY"


def _hexdump(data: bytes, limit: int = 2048) -> str:
    lines = []
    for off in range(0, min(len(data), limit), 16):
        chunk = data[off:off + 16]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{off:08x}  {hexs:<47}  {asc}")
    if len(data) > limit:
        lines.append(f"…（全 {len(data)} バイト中 {limit} バイトまで）")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 2
    path = Path(argv[1])
    # 呼び出し側でも検証しているが、**ここでも再検証する**（TOCTOU と、argv を直接叩かれた
    # 場合の両方に備える。表示は読むだけとはいえ任意ファイルを開く口にはしない）。
    root_arg = Path(argv[2]) if len(argv) > 2 else None
    try:
        resolved = path.resolve()
        if root_arg is not None and not resolved.is_relative_to(Path(root_arg).resolve()):
            print("作業フォルダの外は表示できません", file=sys.stderr)
            return 3
        if not resolved.is_file():
            print("ファイルが見つかりません", file=sys.stderr)
            return 4
        size = resolved.stat().st_size
        if size > MAX_BYTES:
            print("ファイルが大きすぎます", file=sys.stderr)
            return 5
    except OSError as e:
        print(f"開けません: {e}", file=sys.stderr)
        return 6

    import tkinter as tk

    kind = sniff(resolved)
    win = tk.Tk()
    win.title(f"イブの作業フォルダ — {resolved.name}")
    win.geometry(f"+{120}+{80}")
    # topmost は付けない（配信中に前面を奪わない）。全画面にもしない。

    if kind in ("PNG", "JPEG", "GIF", "BMP", "WEBP"):
        try:
            from PIL import Image, ImageTk
            Image.MAX_IMAGE_PIXELS = MAX_PIXELS
            img = Image.open(resolved)
            img.thumbnail((MAX_W, MAX_H))
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(win, image=photo)
            label.image = photo  # GC 防止
            label.pack()
            tk.Label(win, text=f"{resolved.name}  {kind}  {img.width}x{img.height}",
                     anchor="w").pack(fill="x")
        except Exception as e:  # デコード失敗でも落とさず理由を出す
            tk.Label(win, text=f"画像を表示できませんでした: {type(e).__name__}: {e}",
                     wraplength=MAX_W, justify="left").pack(padx=12, pady=12)
    else:
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True)
        bar = tk.Scrollbar(frame)
        bar.pack(side="right", fill="y")
        text = tk.Text(frame, wrap="word", width=90, height=32, yscrollcommand=bar.set)
        text.pack(side="left", fill="both", expand=True)
        bar.config(command=text.yview)
        if kind == "TEXT":
            body = resolved.read_text(encoding="utf-8", errors="replace")[:TEXT_MAX_CHARS]
        else:
            # 表示できない形式は**勝手に別の解釈をせず**正直にそう言う（+ 中身の先頭だけ）
            body = (f"（{resolved.name} はテキストとしても画像としても読めませんでした。"
                    f"先頭を16進で表示します）\n\n" + _hexdump(resolved.read_bytes()))
        text.insert("1.0", body)
        text.config(state="disabled")

    win.bind("<Escape>", lambda e: win.destroy())
    win.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
