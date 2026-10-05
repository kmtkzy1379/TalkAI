"""ペルソナ層の決定論テスト（API 不要）。

検証の軸は「**ペルソナが既存の規律ブロックを壊さないか**」。今回の実機事故（2026-08-15）は
タスクと画面認識で出たので、そこを重点的に見る:
- `# 予約タスク` / `# 機能(Call-Function)の使い方` / `# 実際に実行できる手段` と共存するか
- `# 画面（今この瞬間）` の「推測で作らない」規則と**矛盾しない**か
  （ペルソナの「好奇心旺盛」が、見えていない画面を推測で語る方向に効いたら事故になる）
- system 全体が肥大していないか（応答レイテンシのハード制約 ≤5s に効く）

実行: $env:PYTHONIOENCODING="utf-8"; python tests\test_persona.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from eve.context_assembler import ContextAssembler  # noqa: E402
from eve.persona import (  # noqa: E402
    PERSONA_HEADING,
    PERSONA_MAX_CHARS,
    _default_path,
    load_persona,
    parse_persona,
)
from eve.response.style import SPEECH_STYLE, sanitize_for_speech  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


def _write(text: str) -> str:
    p = os.path.join(tempfile.mkdtemp(prefix="eve_persona_"), "persona.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


# ==== パース ====
SAMPLE = """# イブのペルソナ

ここは編集者向けの説明です。**好みは書かないでください。**
この段落は system に入ってはいけません。

---

## 性格

- 親しみやすい悪友。
- 推測で埋めない。
  知ったかぶりで話を合わせない。

## 口調

- 話し言葉。
"""
body = parse_persona(SAMPLE)
check("箇条書きだけを拾う", "親しみやすい悪友" in body and "話し言葉" in body)
check("**編集者向けの説明は落とす**（イブへの指示として読ませない）",
      "編集者向け" not in body and "書かないでください" not in body)
check("節名が前に付く（どの側面の指示か分かる）", "性格: " in body and "口調: " in body)
check("継続行も拾う（設計意図が半分消えない）", "知ったかぶり" in body)

check("ファイルが無ければ空（壊れない・既存挙動のまま）", load_persona("/no/such/persona.md") == "")
check("空ファイルは空", load_persona(_write("")) == "")
check("水平線が無ければ空（説明だけのファイルを流し込まない）",
      load_persona(_write("# 見出し\n\n- なにか\n")) == "")

long_body = "# x\n\n---\n\n## 性格\n\n" + "".join(f"- {'あ' * 80}\n" for _ in range(40))
check("長すぎるペルソナは切り詰める（肥大の再発を上限で止める）",
      len(load_persona(_write(long_body))) <= PERSONA_MAX_CHARS + len(PERSONA_HEADING) + 1)

# ==== 実ファイル（リポジトリの persona.md＝書式の見本）====
# 本物の性格は手元だけの persona.local.md（あればそちらを読む・.gitignore 済み）。2026-10-06 ユーザ「コミットはコードと
# システムプロンプトフォーマットだけで、実例データや詳細なペルソナは除外」→ リポジトリのテストは書式の見本で、中身は見ない
REPO_PERSONA = str(Path(__file__).resolve().parent.parent / "persona.md")
real = load_persona(REPO_PERSONA)
check("実ファイルが読める", real.startswith(PERSONA_HEADING) and len(real) > 50, f"{len(real)}字")
check("性格と口調が両方入る", "- 性格: " in real and "- 口調: " in real)
check("**好み・趣向が混ざっていない**（自己モデルの領分）",
      not any(w in real for w in ("好きな食べ物", "好物", "好きな色")))
check("**規律が混ざっていない**（style.py とコードゲートの領分）",
      not any(w in real for w in ("delegate_task", "予約タスク", "関数", "tool")))
check("ペルソナは短い（system 全体の予算を圧迫しない）", len(real) < 900, f"{len(real)}字")
# 実際に読む性格（手元の persona.local.md があればそれ）も同じ予算に収まる
check("実際に読む性格も短い（手元の本物があればそれ）", len(load_persona()) < 900, f"{len(load_persona())}字")

# 手元の persona.local.md があればそちらを読む（無ければリポジトリの persona.md）
_root = Path(tempfile.mkdtemp(prefix="eve_persona_root_"))
(_root / "persona.md").write_text("x", encoding="utf-8")
check("手元の persona.local.md が無ければ persona.md（書式の見本）", _default_path(_root) == _root / "persona.md")
(_root / "persona.local.md").write_text("x", encoding="utf-8")
check("手元の persona.local.md があればそちらを読む（本物の性格はコミットしない）",
      _default_path(_root) == _root / "persona.local.md")

# ==== 既存ブロックとの共存（今回の事故領域）====
ctx = ContextAssembler(system_prompt=SPEECH_STYLE + "\n\n" + real)
sysmsg = ctx.assemble(
    user_text="好きな食べ物の画像出してくれない？",
    vision="PowerShell のログが流れている",
    tools_active=True,
    capabilities=["Web検索", "画面表示"],
    active_tasks=["t1: 画像を用意（実行中）"],
    self_model="# イブ自身のこと（自分で言ったこと。理由なく変えない）\n- 好きな食べ物: おかかおにぎり",
)[0]["content"]

check("ペルソナが system に入る", PERSONA_HEADING in sysmsg)
check("読み上げ規律が残る", "音声で読み上げられます" in sysmsg)
check("タスクの規律が残る", "# 予約タスク" in sysmsg and "delegate_task" in sysmsg)
check("手段の制限が残る", "# 実際に実行できる手段" in sysmsg)
check("画面ブロックが残る", "# 画面（今この瞬間）" in sysmsg)
check("自己モデルが別ブロックで残る", "# イブ自身のこと" in sysmsg)
check("役割アンカーが残る（話者取り違え防止）", "assistant のメッセージ" in sysmsg)

# 画面が**無い**時に、ペルソナの好奇心が推測を許してしまわないこと
sys_novision = ctx.assemble(user_text="画面なにが見えてる？", vision="")[0]["content"]
check("画面不在の禁止規則がペルソナに潰されない",
      "推測や過去の記憶から作らない" in sys_novision
      and "今は読み取れていないと伝える" in sys_novision)
# 性格の側の文面（知ったかぶりの扱い等）は中身なので見ない。画面を読めていない時に中身を作らないことは
# Gemini と本物の性格で確かめた（読めていない場面 4/4 で作らなかった・prod_check 2026-10-04）

# 肥大していないこと（≤5s のハード制約に効く）
check("system 全体が肥大していない", len(sysmsg) < 4500, f"{len(sysmsg)}字")

# ==== 口調の規律をコードでも守る（「」は読み上げられない）====
check("かぎかっこを落とす", sanitize_for_speech("「수도권」は韓国語だよ") == "수도권は韓国語だよ")
check("二重かぎかっこも落とす", sanitize_for_speech("『トリコ』の話") == "トリコの話")
check("通常文は不変", sanitize_for_speech("今日はいい天気。") == "今日はいい天気。")
check("プロンプト側でも禁止している", "「」" in SPEECH_STYLE)

print(f"\n合計: PASS {_passed} / FAIL {_failed}")
sys.exit(1 if _failed else 0)
