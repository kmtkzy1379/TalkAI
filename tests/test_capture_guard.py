"""Tier-1: キャプチャ除外（`eve/capture_guard.py`）の決定論部分。

GUI も画面キャプチャも要らない範囲だけをここで固める。**実際に画面から消えるか**は
`tools/capture_guard_check.py`（実測ハーネス）で確かめる（Tier-1 は窓を開かない）。

守りたい不変条件:
  - どんな入力でも**例外を出さない**（失敗しても表示機能や対話を止めない）
  - 既定は **ON**（OFF は自己観察ループが復活する既知の不具合状態）
  - 環境変数の解釈が親子プロセスで一致する（ビューアは環境変数を継承して読む）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve import capture_guard  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def t_env() -> None:
    print("\n[1] 環境変数の解釈（親→ビューアの継承で使う）")
    saved = os.environ.pop(capture_guard._ENV, None)
    try:
        check("未設定なら既定 ON（ガードなので off にしない）",
              capture_guard.enabled_by_env() is True)
        for raw, want in (("1", True), ("0", False), ("true", True), ("false", False),
                          ("no", False), ("", False), ("  1  ", True), ("yes", True)):
            os.environ[capture_guard._ENV] = raw
            got = capture_guard.enabled_by_env()
            check(f"{raw!r} → {want}", got is want, f"got={got}")
        os.environ.pop(capture_guard._ENV, None)
        check("default=False を渡せば OFF に倒せる",
              capture_guard.enabled_by_env(default=False) is False)
    finally:
        if saved is None:
            os.environ.pop(capture_guard._ENV, None)
        else:
            os.environ[capture_guard._ENV] = saved


def t_shared_state() -> None:
    """共有状態: **既に開いているビューア**へ実行中の切替を届ける経路。

    環境変数は起動時に固定されるので、実行中の ON/OFF には使えない。
    親から他プロセスの窓へ直接適用することもできない（ERROR_ACCESS_DENIED・実測）。
    そこで**状態だけ**をファイルで共有し、各プロセスが自分の窓に掛け直す。
    """
    print("\n[1b] 共有状態（実行中の切替を別プロセスへ渡す）")
    saved_env = os.environ.pop(capture_guard._ENV, None)
    try:
        for want in (True, False, True):
            capture_guard.publish(want)
            check(f"publish({want}) → desired() が {want}",
                  capture_guard.desired() is want)
        capture_guard.state_path().unlink(missing_ok=True)
        os.environ[capture_guard._ENV] = "0"
        check("共有ファイルが無ければ環境変数に落ちる", capture_guard.desired() is False)
        capture_guard.state_path().write_text("broken", encoding="ascii")
        check("壊れた内容なら環境変数に落ちる（例外を出さない）",
              capture_guard.desired() is False)
        os.environ.pop(capture_guard._ENV, None)
        check("共有も環境変数も無ければ既定 ON", capture_guard.desired() is True)
    finally:
        capture_guard.state_path().unlink(missing_ok=True)
        if saved_env is None:
            os.environ.pop(capture_guard._ENV, None)
        else:
            os.environ[capture_guard._ENV] = saved_env


def t_scopes() -> None:
    """メイン／タスクウィンドウを**独立に**切り替えられること（2026-08-15 ユーザ指定）。

    守りたいのは「片方を切ったらもう片方も切れた」という混線が起きないこと。
    これが崩れると「自分の UI は隠したまま、作らせた電卓だけイブに見せる」が成立しない。
    """
    print("\n[1c] scope の分離（メイン / タスクウィンドウ）")
    ui, task = capture_guard.UI, capture_guard.TASK
    saved = {s: os.environ.pop(capture_guard.env_name(s), None) for s in (ui, task)}
    try:
        check("環境変数が別（UI_HIDE_FROM_CAPTURE / TASKWIN_HIDE_FROM_CAPTURE）",
              capture_guard.env_name(ui) != capture_guard.env_name(task),
              f"{capture_guard.env_name(ui)} / {capture_guard.env_name(task)}")
        check("共有ファイルも別（片方の書込が他方を巻き添えにしない）",
              capture_guard.state_path(ui) != capture_guard.state_path(task))
        check("タスク側の既定も ON（従来の挙動を変えない）",
              capture_guard.enabled_by_env(task) is True)

        os.environ[capture_guard.env_name(ui)] = "0"
        os.environ[capture_guard.env_name(task)] = "1"
        check("メインだけ OFF にできる", capture_guard.enabled_by_env(ui) is False)
        check("その時タスク側は ON のまま", capture_guard.enabled_by_env(task) is True)
        os.environ[capture_guard.env_name(ui)] = "1"
        os.environ[capture_guard.env_name(task)] = "0"
        check("**タスクウィンドウだけ OFF にできる**（電卓を一緒に見る用途）",
              capture_guard.enabled_by_env(task) is False)
        check("その時メインは隠れたまま", capture_guard.enabled_by_env(ui) is True)

        capture_guard.state_path(ui).unlink(missing_ok=True)
        capture_guard.state_path(task).unlink(missing_ok=True)
        capture_guard.publish(True, task)
        check("publish(task) は task の desired だけを動かす",
              capture_guard.desired(task) is True and capture_guard.desired(ui) is True)
        capture_guard.publish(False, task)
        check("task を OFF にしても ui は環境変数（ON）のまま",
              capture_guard.desired(task) is False and capture_guard.desired(ui) is True)
        check("describe が scope で文言を変える（取り違え防止）",
              capture_guard.describe(True, ui) != capture_guard.describe(True, task))
        check("知らない scope は UI 扱い（落とさない）",
              capture_guard.env_name("nonsense") == capture_guard.env_name(ui))
    finally:
        for s in (ui, task):
            capture_guard.state_path(s).unlink(missing_ok=True)
            if saved[s] is None:
                os.environ.pop(capture_guard.env_name(s), None)
            else:
                os.environ[capture_guard.env_name(s)] = saved[s]


def t_safe() -> None:
    print("\n[2] 壊れた入力でも落ちない（表示や対話を止めない）")
    for hwnd in (0, None, -1, 123456789, "abc", 2 ** 63):
        try:
            r = capture_guard.set_window(hwnd, True)  # type: ignore[arg-type]
            ok = r is False or r is True
        except Exception as e:
            ok = False
            print(f"      例外: {type(e).__name__}: {e}")
        check(f"set_window({hwnd!r}) が例外を出さない", ok)
    try:
        n = capture_guard.set_own_windows(True)
        capture_guard.set_own_windows(False)
        ok = isinstance(n, int) and n >= 0
    except Exception:
        ok = False
    check("set_own_windows が例外を出さず int を返す", ok)
    check("available() が bool", isinstance(capture_guard.available(), bool))
    check("describe() が状態で文言を変える",
          capture_guard.describe(True) != capture_guard.describe(False))


def t_config() -> None:
    print("\n[3] Config と .env.example")
    from eve.config import Config

    check("UI_HIDE_FROM_CAPTURE が bool", isinstance(Config.UI_HIDE_FROM_CAPTURE, bool))
    check("TASKWIN_HIDE_FROM_CAPTURE が bool",
          isinstance(Config.TASKWIN_HIDE_FROM_CAPTURE, bool))
    env_example = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               ".env.example")
    with open(env_example, encoding="utf-8") as f:
        body = f.read()
    # D6 の再発防止（.env.example に全フラグを明記する規律）
    check("**.env.example に明記されている**", "UI_HIDE_FROM_CAPTURE" in body)
    check("**.env.example にタスク側も明記されている**",
          "TASKWIN_HIDE_FROM_CAPTURE" in body)


def t_viewer_applies_itself() -> None:
    """ビューアは**別プロセス**なので自分で適用する必要がある（親からは適用できない）。"""
    print("\n[4] ビューアが自分で適用する配線")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "eve", "workspace", "viewer.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    check("viewer が capture_guard を呼んでいる", "capture_guard" in src)
    check("viewer が共有状態を見ている（実行中の切替に追従する）", "desired" in src)
    # ビューアは**イブの表示物**＝タスク側の設定に従う（メインとは独立に切れる）
    check("viewer が TASK scope を見ている", "desired(TASK)" in src)
    check("viewer が自プロセスの窓に適用している", "set_own_windows" in src)
    check("viewer が周期的に見直している（開いたままでも追従する）", "after(1000" in src)
    check("表示機能を止めないよう例外を握っている",
          "except Exception" in src.split("capture_guard")[1][:800])


if __name__ == "__main__":
    t_env()
    t_shared_state()
    t_scopes()
    t_safe()
    t_config()
    t_viewer_applies_itself()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
