"""Tier-1: `.env` の読み書き（API 不要）。

**なぜ厳しく縛るか**: ここの失敗はどれも静かで、しかも**ユーザの資産（手書きの .env）を
壊す**。BOM を1つ付けただけで1行目のキーが黙って効かなくなり、引用符を付けただけで
dotenv の有無によって値が変わる。壊れた時に気づけないので、書き手側で全部塞ぐ。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve import settings_store as ss  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def tmp_env(text: str, newline: str = "\n") -> Path:
    d = Path(tempfile.mkdtemp(prefix="eve_env_"))
    p = d / ".env"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", newline))
    return p


def raw(p: Path) -> bytes:
    return p.read_bytes()


def t_read() -> None:
    print("\n[1] 読み取り")
    p = tmp_env("# コメント\nA=1\n\nB = two \nexport C=3\n不正な行\n")
    d = ss.read_env(p)
    check("キーと値を読む", d.get("A") == "1", f"{d}")
    check("前後の空白を落とす", d.get("B") == "two", f"{d}")
    check("export 付きも読む", d.get("C") == "3", f"{d}")
    check("コメント/空行/不正行は無視", set(d) == {"A", "B", "C"}, f"{set(d)}")
    check("存在しないファイルは空", ss.read_env(p.parent / "nope") == {})

    # BOM 付きファイルも読めること（既に BOM 付きで存在する .env を壊さない）
    b = p.parent / ".env.bom"
    b.write_bytes("﻿A=1\n".encode("utf-8"))
    check("BOM 付きでも読める", ss.read_env(b).get("A") == "1")


def t_preserve() -> None:
    print("\n[2] 書き込み: ユーザの資産を保つ")
    src = ("# 見出し\nOPENAI_API_KEY=old\n\n# 音声\nVV_SPEED=1.0\nVV_PITCH=0.0\n")
    p = tmp_env(src, newline="\r\n")
    changed = ss.write_env({"VV_SPEED": "1.2"}, p)
    text = p.read_text(encoding="utf-8")
    check("変わったキーだけ返す", changed == ["VV_SPEED"], f"{changed}")
    check("値が置き換わる", "VV_SPEED=1.2" in text)
    check("**コメントが残る**", "# 見出し" in text and "# 音声" in text)
    check("**空行が残る**", b"\r\n\r\n" in raw(p), str(raw(p)))
    check("触っていないキーはそのまま", "VV_PITCH=0.0" in text and "OPENAI_API_KEY=old" in text)
    check("**BOM を付けない**", not raw(p).startswith(b"\xef\xbb\xbf"), str(raw(p)[:6]))
    check("**CRLF を保つ**", b"\r\n" in raw(p) and b"\n" not in raw(p).replace(b"\r\n", b""))
    check("**引用符を付けない**", '"' not in text and "'" not in text)
    check("末尾は改行で終わる", raw(p).endswith(b"\r\n"))
    # 順序が変わらないこと（差分が全行にならない）
    keys = [ln.split("=")[0] for ln in text.splitlines() if "=" in ln and not ln.startswith("#")]
    check("順序が変わらない", keys == ["OPENAI_API_KEY", "VV_SPEED", "VV_PITCH"], f"{keys}")


def t_lf_kept() -> None:
    print("\n[3] LF のファイルは LF のまま")
    p = tmp_env("A=1\nB=2\n", newline="\n")
    ss.write_env({"A": "9"}, p)
    check("LF が CRLF に変わらない", b"\r\n" not in raw(p), str(raw(p)))


def t_append() -> None:
    print("\n[4] 新しいキーは末尾に足す")
    p = tmp_env("A=1\n")
    changed = ss.write_env({"A": "1", "NEW_KEY": "x"}, p)
    text = p.read_text(encoding="utf-8")
    check("値が同じキーは変更に数えない", changed == ["NEW_KEY"], f"{changed}")
    check("新キーが入る", "NEW_KEY=x" in text)
    check("既存キーは重複しない", text.count("A=1") == 1, text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    check("末尾に足す（先頭を荒らさない）", lines[0] == "A=1", f"{lines}")


def t_no_touch() -> None:
    print("\n[5] 変更が無ければファイルに触らない")
    p = tmp_env("A=1\n")
    before = p.stat().st_mtime_ns
    changed = ss.write_env({"A": "1"}, p)
    check("変更キーは空", changed == [], f"{changed}")
    check("更新時刻が動かない", p.stat().st_mtime_ns == before)


def t_reject() -> None:
    print("\n[6] 書けない値は拒否する（環境で意味が変わるもの）")
    p = tmp_env("A=1\n")
    for label, key, val in (
        ("改行", "A", "x\ny"),
        ("復帰", "A", "x\ry"),
        ("# を含む", "A", "abc#def"),
        ("前後の空白", "A", " abc"),
    ):
        try:
            ss.write_env({key: val}, p)
            check(f"{label} を拒否する", False, "拒否しなかった")
        except ss.InvalidValue as e:
            check(f"{label} を拒否する", True)
            check(f"  {label}: **値をメッセージに載せない**", val.strip() not in str(e), str(e))
    check("拒否したらファイルを触らない", p.read_text(encoding="utf-8") == "A=1\n")


def t_atomic() -> None:
    print("\n[7] 壊れた書き込みを残さない")
    p = tmp_env("A=1\n")
    ss.write_env({"A": "2"}, p)
    leftovers = [x.name for x in p.parent.iterdir() if x.name.startswith(".env.")
                 and x.name.endswith(".tmp")]
    check("一時ファイルを残さない", leftovers == [], f"{leftovers}")
    import inspect

    src = inspect.getsource(ss._atomic_write)
    check("os.replace で置き換える（途中で落ちても .env が壊れない）", "os.replace" in src)
    check("newline='' で改行変換を止めている", 'newline=""' in src)
    check("同じディレクトリに一時ファイルを作る（replace をアトミックに保つ）",
          "dir=str(p.parent)" in src)


def t_example() -> None:
    print("\n[8] .env.example への追記")
    d = Path(tempfile.mkdtemp(prefix="eve_ex_"))
    p = d / ".env.example"
    p.write_text("A=1\nB=2\n", encoding="utf-8")
    added = ss.sync_example({"A": "9", "C": "3", "D": "4"}, p)
    text = p.read_text(encoding="utf-8")
    check("欠けているキーだけ足す", sorted(added) == ["C", "D"], f"{added}")
    check("**既存の値は上書きしない**", "A=1" in text and "A=9" not in text, text)
    check("追記された", "C=3" in text and "D=4" in text)
    check("BOM なし", not p.read_bytes().startswith(b"\xef\xbb\xbf"))
    check("2回目は何もしない", ss.sync_example({"A": "9", "C": "3"}, p) == [])

    # **コメントで説明済みのキーを二重に足さない**（本物の .env.example を汚した欠陥）
    p2 = d / "commented.example"
    p2.write_text("# 以下はコード既定で運用中（.env に書かなくても動く）\n"
                  "# TASK_MODEL=openai/gpt-5.5\n"
                  "#   VLM_OCR_ENABLED=1\n"
                  "REAL=1\n", encoding="utf-8")
    added2 = ss.sync_example({"TASK_MODEL": "openai/gpt-4o", "VLM_OCR_ENABLED": "0",
                              "NEW_ONE": "x"}, p2)
    check("**コメント行のキーは「記載済み」とみなす**", added2 == ["NEW_ONE"], f"{added2}")
    t2 = p2.read_text(encoding="utf-8")
    check("  実行時の値で上書きしない", "TASK_MODEL=openai/gpt-4o" not in t2, t2)
    check("  本当に新しいものは足す", "NEW_ONE=x" in t2)
    doc = ss.documented_keys(p2)
    check("  目録はコメント行も拾う",
          {"TASK_MODEL", "VLM_OCR_ENABLED", "REAL"} <= doc, f"{sorted(doc)}")
    check("  説明文の日本語をキーと誤認しない",
          not any("以下" in k for k in doc), f"{sorted(doc)}")


def t_example_sandboxed() -> None:
    print("\n[9] EVE_ENV_FILE 中は目録もサンドボックスを見る")
    import importlib

    d = Path(tempfile.mkdtemp(prefix="eve_sbx_"))
    (d / ".env").write_text("A=1\n", encoding="utf-8")
    old = os.environ.get("EVE_ENV_FILE")
    os.environ["EVE_ENV_FILE"] = str(d / ".env")
    try:
        mod = importlib.reload(ss)
        check("**`.env` を差し替えたら `.env.example` も同じ場所**",
              mod.ENV_EXAMPLE_PATH.parent == d, f"{mod.ENV_EXAMPLE_PATH}")
        check("  リポジトリの本物を指していない",
              mod.ENV_EXAMPLE_PATH != mod.REPO_ROOT / ".env.example")
    finally:
        if old is None:
            os.environ.pop("EVE_ENV_FILE", None)
        else:
            os.environ["EVE_ENV_FILE"] = old
        importlib.reload(ss)
    check("環境変数が無ければリポジトリの本物を見る",
          ss.ENV_EXAMPLE_PATH == ss.REPO_ROOT / ".env.example", f"{ss.ENV_EXAMPLE_PATH}")


def t_key_status() -> None:
    print("\n[9] API キーは値を出さない")
    check("指紋は末尾4文字と長さだけ",
          ss.fingerprint("sk-proj-abcdefgh1234") == "…1234・20文字",
          ss.fingerprint("sk-proj-abcdefgh1234"))
    check("空なら空", ss.fingerprint("") == "")
    fp = ss.fingerprint("sk-secret-value-XYZW")
    check("**先頭や中身が出ない**", "secret" not in fp and "sk-" not in fp, fp)

    import inspect

    src = inspect.getsource(ss)
    check("key_status は生の値を返さない（戻り値は状態と指紋のみ）",
          "return \"env\", fingerprint(raw)" in src and "return \"file\", fingerprint(in_file)" in src)
    # OS 環境変数が .env より優先される（config.py の setdefault）ことを見分けているか
    check("OS 環境変数で設定済みの場合を区別する（保存しても効かないため）",
          '"env"' in src and "os.environ.get" in src)

    # 実際の分岐（一時的に環境変数を立てて確認）
    name = "EVE_TEST_FAKE_KEY_XYZ"
    os.environ.pop(name, None)
    check("未設定は none", ss.key_status(name)[0] == "none")
    os.environ[name] = "abcd1234"
    try:
        st, fp2 = ss.key_status(name)
        check("OS 環境変数にあれば env", st == "env", st)
        check("その指紋も末尾だけ", fp2 == "…1234・8文字", fp2)
    finally:
        os.environ.pop(name, None)


def t_no_secret_logging() -> None:
    print("\n[10] 値をログに出さない")
    import inspect

    src = inspect.getsource(ss.write_env)
    check("ログにキー名だけ出す（値を書式に入れない）",
          "updates[" not in src.split("logger.info")[1].split(")")[0]
          if "logger.info" in src else False)
    check("ログ行に値の変数が入っていない",
          'join(sorted(changed))' in src)


if __name__ == "__main__":
    t_read()
    t_preserve()
    t_lf_kept()
    t_append()
    t_no_touch()
    t_reject()
    t_atomic()
    t_example()
    t_example_sandboxed()
    t_key_status()
    t_no_secret_logging()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
