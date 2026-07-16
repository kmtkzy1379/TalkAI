r"""J-2 search 実機スモーク（実 ddgs 経路・決定論テストが偽装する境界の検証）。

検証項目（レッドチーム S4/S5）:
 1. 実 ddgs で region="jp-jp" + backend + timeout の引数互換（例外を出さず title/body/href が返る）
 2. SearchClient 本番経路（to_thread + wait_for + キャッシュ）の通し
 3. ダイジェスト長の実測（トークン圧迫の見積り）
 4. BURST=1 指定時のみ: 連続クエリでレートリミットを誘発し**実例外メッセージ文字列**を採取
    （クールダウン発火条件の判定語の一次データ。ネットワーク負荷があるので既定 off）

実行: $env:PYTHONIOENCODING="utf-8"; ..\portfolio8-VLM-AI\venv\Scripts\python.exe tools\search_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.logsetup import configure  # noqa: E402
from eve.search import SearchClient  # noqa: E402


async def main() -> None:
    configure()
    ok = True

    # 1) 実 ddgs 引数互換（SearchClient 本番経路）
    c = SearchClient()
    t0 = time.monotonic()
    out = await c.search("マリアナ海溝 深さ 何メートル")
    dt = time.monotonic() - t0
    print(f"\n[1] 実検索 {dt:.2f}s / {len(out)}字")
    print(out[:400])
    if out.startswith("（"):
        print("  → NG: 実引数互換 or ネットワークで失敗")
        ok = False
    else:
        print("  → OK（ヘッダ/ドメイン/件数を目視確認）")

    # 2) キャッシュ命中（2回目は即時）
    t0 = time.monotonic()
    out2 = await c.search("マリアナ海溝 深さ 何メートル")
    dt2 = time.monotonic() - t0
    print(f"\n[2] キャッシュ命中 {dt2 * 1000:.0f}ms → {'OK' if dt2 < 0.05 and out2 == out else 'NG'}")
    ok = ok and dt2 < 0.05

    # 3) 日本語の別クエリ + ダイジェスト長実測
    out3 = await c.search("VOICEVOX とは")
    print(f"\n[3] 別クエリ: {'OK' if not out3.startswith('（') else 'NG: ' + out3[:80]} / {len(out3)}字")

    # 4) レートリミット誘発（任意・実例外語の採取）
    if os.getenv("BURST") == "1":
        print("\n[4] バースト誘発（実例外語の採取）...")
        from ddgs import DDGS
        for i in range(10):
            try:
                rows = DDGS(timeout=5).text(f"テスト クエリ {i}", region="jp-jp",
                                            max_results=3, backend="duckduckgo")
                print(f"  {i}: OK {len(rows)}件")
            except Exception as e:
                print(f"  {i}: {type(e).__name__}: {str(e)[:120]!r}")
            await asyncio.sleep(0.2)

    print(f"\n=== スモーク {'合格' if ok else '不合格'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
