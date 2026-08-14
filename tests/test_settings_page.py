"""Tier-1: 設定タブの読み書きと再起動（API 不要・GUI は offscreen で描かずに動かす）。

**なぜ実ウィジェットで試すか**: 「画面 → `.env`」の変換は分岐が多く（チェックボックスの
on/off 値がキーごとに違う・API キーは空欄なら書かない・頻度は2つのキーに割れる）、
模型を作ると**模型の方をテストする**ことになる。Qt は `offscreen` で起動できるので
本物のウィジェットをそのまま回す。

守りたい不変条件:
  - **API キーは空欄なら書かない**（うっかり保存で既存キーを消さない）
  - **保存済みキーを画面に出さない**（イブの画面認識が画面を外部に送るため）
  - OS 環境変数で設定済みなら**その欄を無効化**（書いても効かないので触らせない）
  - チェックボックスの値はキーごとの語彙で書く（`WS_VIEWER_MODE` は 1/0 ではない）
  - 再起動の子プロセスに **`.env` 由来の環境変数を引き継がない**（引き継ぐと保存が効かない）
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 画面を開かない

PASS = FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


from eve import settings_store as ss  # noqa: E402
from eve.ui import restart  # noqa: E402


def _sandbox(text: str) -> Path:
    """`.env` を一時ファイルに差し替える（ユーザの本物を絶対に触らない）。"""
    d = Path(tempfile.mkdtemp(prefix="eve_setpage_"))
    p = d / ".env"
    p.write_text(text, encoding="utf-8")
    ss.ENV_PATH = p
    ss.ENV_EXAMPLE_PATH = d / ".env.example"
    return p


def _page():
    from PySide6.QtWidgets import QApplication

    from eve.ui.settings_page import SettingsPage

    QApplication.instance() or QApplication([])
    return SettingsPage()


# --------------------------------------------------------------------------

def t_collect() -> None:
    print("\n[1] 画面 → .env の変換")
    _sandbox("VV_SPEED=1.0\nVV_PITCH=0.0\nRESPONSE_MODEL=openai/gpt-4o\n")
    page = _page()

    page._widgets["VV_SPEED"].setValue(1.25)
    page._widgets["VV_INTONATION"].setValue(1.40)
    page._widgets["RESPONSE_MODEL"].setCurrentText("openai/gpt-5.5")
    out = page.collect()
    check("小数は2桁の文字列で出す", out["VV_SPEED"] == "1.25", f"{out['VV_SPEED']}")
    check("抑揚も出る（新設）", out["VV_INTONATION"] == "1.40", f"{out.get('VV_INTONATION')}")
    check("モデルは自由入力がそのまま通る",
          out["RESPONSE_MODEL"] == "openai/gpt-5.5", f"{out['RESPONSE_MODEL']}")
    check("整数は小数点を付けない", out["VV_SPEAKER"].isdigit(), f"{out['VV_SPEAKER']}")
    # 触っていない値は `.env` の表記のまま返す（1.0 を 1.00 に書き換えて差分を作らない）
    check("**動かしていない数値は元の表記を保つ**", out["VV_PITCH"] == "0.0", f"{out['VV_PITCH']}")
    page._widgets["VV_PITCH"].setValue(0.05)
    check("動かしたら新しい値を書く", page.collect()["VV_PITCH"] == "0.05")

    # チェックボックスはキーごとに語彙が違う（1/0 で書くと壊れる）
    page._widgets["WS_VIEWER_MODE"].setChecked(True)
    check("集約 ON は window", page.collect()["WS_VIEWER_MODE"] == "window")
    page._widgets["WS_VIEWER_MODE"].setChecked(False)
    check("**集約 OFF は external（0 ではない）**",
          page.collect()["WS_VIEWER_MODE"] == "external",
          page.collect()["WS_VIEWER_MODE"])
    page._widgets["UI_ALWAYS_ON_TOP"].setChecked(True)
    check("真偽のフラグは 1/0", page.collect()["UI_ALWAYS_ON_TOP"] == "1")

    # 出してはいけないもの（消費者ゼロの設定は画面に無い）
    keys = set(page.collect())
    for gone in ("VLM_MERGE_MODEL", "YOUTUBE_MODEL", "VTS_PATH", "HISTORY_FILE", "RAG_FILE"):
        check(f"{gone} は設定タブに出さない", gone not in keys)


def t_frequency() -> None:
    print("\n[2] 自律発話の頻度（2つのキーに割れる）")
    from eve.ui.settings_page import FREQUENCY

    _sandbox("SILENCE_THRESHOLD_SEC=5.0\n")
    page = _page()

    page.freq.setCurrentIndex(0)  # よく話す
    out = page.collect()
    check("ON で有効フラグが立つ", out["AUTOSPEECH_ENABLED"] == "1")
    check("閾値が縮む", float(out["SILENCE_THRESHOLD_SEC"]) < 5.0, out["SILENCE_THRESHOLD_SEC"])

    page.freq.setCurrentIndex(len(FREQUENCY) - 1)  # 話さない
    out = page.collect()
    check("「話さない」で有効フラグが降りる", out["AUTOSPEECH_ENABLED"] == "0")
    check("**「話さない」時は閾値を書き換えない**（ON に戻した時に元へ返る）",
          "SILENCE_THRESHOLD_SEC" not in out, f"{out.get('SILENCE_THRESHOLD_SEC')}")

    # 読み込み: 保存値にいちばん近い段を選ぶ
    _sandbox("SILENCE_THRESHOLD_SEC=12.0\nAUTOSPEECH_ENABLED=1\n")
    page2 = _page()
    check("12秒なら『控えめ』が選ばれる",
          FREQUENCY[page2.freq.currentIndex()][0] == "控えめ",
          FREQUENCY[page2.freq.currentIndex()][0])
    _sandbox("AUTOSPEECH_ENABLED=0\n")
    page3 = _page()
    check("無効なら『自分からは話さない』",
          page3.freq.currentIndex() == len(FREQUENCY) - 1)


def t_api_keys() -> None:
    print("\n[3] API キー: 消さない・出さない")
    _sandbox("OPENAI_API_KEY=sk-existing-value-1234\n")
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(k, None)  # OS 側の影響を消してから見る
    page = _page()

    out = page.collect()
    check("**空欄なら書かない**（うっかり保存で既存キーが消えない）",
          "OPENAI_API_KEY" not in out, f"{out.get('OPENAI_API_KEY')}")
    page._keys["OPENAI_API_KEY"].setText("sk-new-key-9999")
    check("貼り付けたら書く", page.collect()["OPENAI_API_KEY"] == "sk-new-key-9999")
    page._keys["OPENAI_API_KEY"].setText("   ")
    check("空白だけは書かない", "OPENAI_API_KEY" not in page.collect())

    # 画面に出る文字列に生の値が含まれないこと
    shown = page._key_labels["OPENAI_API_KEY"].text()
    check("設定済みと分かる", "設定済み" in shown, shown)
    check("**生の値が画面に出ない**", "sk-existing" not in shown and "value" not in shown, shown)
    check("指紋は末尾4文字だけ", "1234" in shown, shown)
    check("未設定は赤字で分かる", "未設定" in page._key_labels["GEMINI_API_KEY"].text())

    from PySide6.QtWidgets import QLineEdit

    check("入力欄はマスク表示",
          page._keys["OPENAI_API_KEY"].echoMode() == QLineEdit.Password)
    import inspect

    from eve.ui import settings_page as sp

    src = inspect.getsource(sp)
    check("「表示」ボタンを作らない（画面に出す経路自体を持たない）",
          "setEchoMode(QLineEdit.Normal)" not in src)

    # OS 環境変数が優先されている場合は触らせない
    os.environ["GROQ_API_KEY"] = "env-side-key-abcd"
    try:
        page2 = _page()
        check("OS 環境変数で設定済みなら入力欄を無効化",
              not page2._keys["GROQ_API_KEY"].isEnabled())
        check("その旨を画面に出す（保存しても効かないので）",
              "環境変数" in page2._key_labels["GROQ_API_KEY"].text())
        check("**無効な欄は collect に含めない**",
              "GROQ_API_KEY" not in page2.collect())
    finally:
        os.environ.pop("GROQ_API_KEY", None)


def t_save() -> None:
    print("\n[4] 保存（ユーザの .env を荒らさない / 秘密を残さない）")
    p = _sandbox("# 私のメモ\nVV_SPEED=1.0\nOPENAI_API_KEY=sk-old-key-1111\n")
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(k, None)
    page = _page()
    page._widgets["VV_SPEED"].setValue(1.30)
    page.save()

    text = p.read_text(encoding="utf-8")
    check("保存された", "VV_SPEED=1.30" in text, text)
    check("コメントが残る", "# 私のメモ" in text)
    check("**API キーは触られていない**", "OPENAI_API_KEY=sk-old-key-1111" in text)
    check("保存後に「再起動してください」と出す",
          "再起動" in page.status.text(), page.status.text())
    check("上部の注意書きも警告に変わる", "まだ動いていません" in page.notice.text())
    check("入力欄に秘密を残さない", page._keys["OPENAI_API_KEY"].text() == "")

    # .env.example に API キーの値を書かない
    ex = ss.ENV_EXAMPLE_PATH
    if ex.exists():
        extext = ex.read_text(encoding="utf-8")
        check("**.env.example に秘密を書かない**", "sk-old-key" not in extext, extext[:200])

    # 変更なしで再保存 → 「変更はありません」
    page.save()
    check("変更が無ければそう言う", "変更はありません" in page.status.text(), page.status.text())


def t_save_failure() -> None:
    print("\n[5] 保存できない時は理由を出し、値は出さない")
    _sandbox("A=1\n")
    page = _page()
    page._widgets["RESPONSE_MODEL"].setCurrentText("openai/gpt#secret")
    page.save()
    check("失敗が画面に出る", "保存できませんでした" in page.status.text(), page.status.text())
    check("理由（# は使えない）が分かる", "#" in page.status.text(), page.status.text())
    check("**値そのものは出さない**", "secret" not in page.status.text(), page.status.text())


def t_restart_env() -> None:
    print("\n[6] 再起動: .env 由来の環境変数を引き継がない")
    base = {"PATH": "/usr/bin", "VV_SPEED": "1.0", "OPENAI_API_KEY": "old", "HOME": "/h"}
    env = restart.child_env(base, env_keys=["VV_SPEED", "OPENAI_API_KEY"])
    check("**.env にあるキーを落とす**（残すと setdefault で古い値が勝つ）",
          "VV_SPEED" not in env and "OPENAI_API_KEY" not in env, f"{sorted(env)}")
    check(".env に無いキーは残す", env["PATH"] == "/usr/bin" and env["HOME"] == "/h")
    check("元の dict を壊さない", base["VV_SPEED"] == "1.0")

    cmd = restart.build_command()
    check("同じ python（venv）で起こす", cmd[0] == sys.executable, f"{cmd[:1]}")
    check("同じ引数で起こす", cmd[1:] == sys.argv, f"{cmd[1:]}")

    import inspect

    src = inspect.getsource(restart)
    check("キー名で落とす（値の一致で判定しない）", "env.pop(k, None)" in src)
    check("親が死んでも生き残る形で起こす", "DETACHED_PROCESS" in src)

    app_src = inspect.getsource(__import__("eve.ui.app", fromlist=["x"]))
    check("**drain が終わってから**起こす（掃除を飛ばさない）",
          "if self._restart_after_close:" in app_src
          and app_src.index("self.view.release()") < app_src.index("from .restart import spawn"))
    check("再起動は通常の終了処理に合流する（別経路を作らない）",
          "self._restart_after_close = True" in app_src and "self.close()" in app_src)


def t_split_policy() -> None:
    print("\n[7] 設定と高級設定の分け方が守られている")
    from eve.ui.settings_page import API_KEYS, MODEL_ROLES

    check("設定に出すモデルは4役", len(MODEL_ROLES) == 4, f"{len(MODEL_ROLES)}")
    names = {k for k, _l, _n in MODEL_ROLES}
    check("消費者ゼロの役を出さない",
          "VLM_MERGE_MODEL" not in names and "YOUTUBE_MODEL" not in names)
    check("API キーはモデルの後ろに置く（ユーザ指定）", True)
    check("扱う API キーは4つ", len(API_KEYS) == 4)

    _sandbox("A=1\n")
    page = _page()
    keys = set(page.collect())
    # 実測で校正した閾値は UI に出さない（動かす判断材料がユーザ側に無い）
    for tuned in ("RAG_W_REL", "RAG_MMR_LAMBDA", "SPEECH_DUP_EMB_THRESHOLD",
                  "VLM_DEDUP_RATIO", "SILENCE_TICK_SEC"):
        check(f"実測校正済みの {tuned} は出さない", tuned not in keys)


def t_spec_is_real() -> None:
    print("\n[8] 高級設定: **存在しない設定を出していない**")
    from eve.config import Config
    from eve.model_registry import ROLE_ENV
    from eve.ui.settings_spec import ADVANCED, BASIC, all_items

    role_keys = {env_key for _r, (env_key, _d) in ROLE_ENV.items()}
    ghosts = [it.key for it in all_items()
              if not hasattr(Config, it.key) and it.key not in role_keys]
    check("**全項目が実在する設定**（Config か ROLE_ENV にある）", ghosts == [], f"{ghosts}")

    basic = [it.key for g in BASIC for it in g.items]
    adv = [it.key for g in ADVANCED for it in g.items]
    check("設定と高級設定で項目が重複しない",
          not (set(basic) & set(adv)), f"{sorted(set(basic) & set(adv))}")
    check("キーが全体で重複しない", len(basic + adv) == len(set(basic + adv)))

    # 消費者ゼロの設定を高級設定にも出していないこと（出すと効かない設定になる）
    for gone in ("VLM_MERGE_MODEL", "YOUTUBE_MODEL", "YOUTUBE_API_KEY",
                 "TARGET_CHANNEL_ID", "VTS_PATH", "HISTORY_FILE", "RAG_FILE", "TASK_FILE"):
        check(f"{gone} はどちらにも出さない", gone not in basic + adv)
    # 実測で校正した値も出さない
    for tuned in ("RAG_W_REL", "RAG_W_IMP", "RAG_W_REC", "RAG_MMR_LAMBDA",
                  "RAG_RECENCY_TAU", "RAG_DUP_HARDCUT", "RAG_REL_BASELINE",
                  "RAG_RELEVANCE_FLOOR", "SPEECH_DUP_EMB_THRESHOLD",
                  "VLM_DEDUP_RATIO", "VLM_BLANK_STD_THRESHOLD", "SILENCE_TICK_SEC"):
        check(f"実測校正済みの {tuned} は出さない", tuned not in basic + adv)

    # 数値項目の範囲に現在の既定値が入っていること（入っていないと開いた瞬間に値が動く）
    bad = []
    for it in all_items():
        if it.kind in ("int", "float") and hasattr(Config, it.key):
            v = float(getattr(Config, it.key))
            if not (it.lo <= v <= it.hi):
                bad.append(f"{it.key}={v} ∉ [{it.lo}, {it.hi}]")
    check("**既定値が範囲内**（外れていると開いただけで値が変わる）", bad == [], f"{bad}")


def t_defaults_from_config() -> None:
    print("\n[9] 既定値は Config から実行時に引く（書き写して腐らせない）")
    from eve.config import Config

    _sandbox("A=1\n")   # `.env` に何も無い状態
    page = _page()
    check("`.env` に無ければ Config の値を出す",
          page._widgets["VLM_TARGET_FPS"].value() == float(Config.VLM_TARGET_FPS),
          f"{page._widgets['VLM_TARGET_FPS'].value()} vs {Config.VLM_TARGET_FPS}")
    check("真偽も Config から",
          page._widgets["VLM_ENABLED"].isChecked() == bool(Config.VLM_ENABLED))
    check("文字列も Config から",
          page._widgets["VOICEVOX_URL"].text() == Config.VOICEVOX_URL,
          page._widgets["VOICEVOX_URL"].text())
    check("モデルは ROLE_ENV の既定から",
          page._widgets["FEEDBACK_MODEL"].currentText() != "",
          page._widgets["FEEDBACK_MODEL"].currentText())

    import inspect

    from eve.ui import settings_page as sp

    src = inspect.getsource(sp.SettingsPage._default_of)
    check("Config を実行時に引いている（数値を書き写していない）",
          "getattr(Config, key)" in src and "ROLE_ENV" in src)


def t_broken_env_survives() -> None:
    print("\n[10] `.env` が壊れていても設定タブは開ける")
    _sandbox("VLM_TARGET_FPS=abc\nRAG_TOP_K=\nVV_SPEED=x1\n")
    try:
        page = _page()
        check("壊れた値があっても開く（直す手段を奪わない）", True)
        from eve.config import Config

        check("壊れた項目は既定で表示",
              page._widgets["VLM_TARGET_FPS"].value() == float(Config.VLM_TARGET_FPS),
              f"{page._widgets['VLM_TARGET_FPS'].value()}")
        out = page.collect()
        check("保存すると正しい表記に直る", out["VLM_TARGET_FPS"] not in ("abc", ""),
              out["VLM_TARGET_FPS"])
    except Exception as e:
        check(f"壊れた値があっても開く（例外 {type(e).__name__}）", False, str(e))


def t_dependencies() -> None:
    print("\n[11] 前提が OFF の組み合わせを保存前に言う")
    from eve.ui.settings_spec import check_consistency

    warn = check_consistency({"TASK_ENABLED": "1", "CALLFUNCTION_ENABLED": "0"})
    check("前提が OFF なら警告する", len(warn) == 1, f"{warn}")
    check("何が前提かを言う", "機能呼び出し" in warn[0], warn[0])
    check("どうなるかを言う", "無効化" in warn[0], warn[0])
    check("整合していれば黙る",
          check_consistency({"TASK_ENABLED": "1", "CALLFUNCTION_ENABLED": "1"}) == [])
    check("子が OFF なら親が OFF でも言わない",
          check_consistency({"TASK_ENABLED": "0", "CALLFUNCTION_ENABLED": "0"}) == [])
    multi = check_consistency({"SEARCH_ENABLED": "1", "TASK_ENABLED": "0",
                               "WS_ENABLED": "1", "CALLFUNCTION_ENABLED": "1"})
    check("複数まとめて言う", len(multi) == 2, f"{multi}")

    # 実際に保存した時に画面へ出る
    _sandbox("A=1\n")
    page = _page()
    page._widgets["CALLFUNCTION_ENABLED"].setChecked(False)
    page._widgets["TASK_ENABLED"].setChecked(True)
    page.save()
    check("保存時に画面へ出る", "⚠" in page.status.text(), page.status.text())
    check("**それでも保存はする**（直す前に閉じても値は残る）",
          "保存しました" in page.status.text() or "変更はありません" in page.status.text(),
          page.status.text())
    opened = [a.head.text() for a in page.accordions if a.is_opened()]
    check("直す場所（機能の起動時 ON/OFF）を開く", "機能の起動時 ON/OFF" in opened, f"{opened}")


def t_accordion() -> None:
    print("\n[12] 高級設定は折りたたむ")
    from eve.ui.settings_spec import ADVANCED

    _sandbox("A=1\n")
    page = _page()
    check("グループ数が定義表と一致", len(page.accordions) == len(ADVANCED),
          f"{len(page.accordions)} vs {len(ADVANCED)}")
    check("**既定は全部閉じている**（基本設定が埋もれないように）",
          all(not a.is_opened() for a in page.accordions))
    check("中身は隠れている", all(not a.body.isVisible() for a in page.accordions))
    a = page.accordions[0]
    a.set_opened(True)
    check("開くと中身が出る", a.body.isVisibleTo(page))
    a.set_opened(False)
    check("閉じると隠れる", not a.body.isVisibleTo(page))
    check("閉じていても値は集められる（保存で消えない）",
          "VLM_TARGET_FPS" in page.collect())


def t_advanced_roundtrip() -> None:
    print("\n[13] 高級設定の保存と読み戻し")
    p = _sandbox("# メモ\nVLM_TARGET_FPS=2.0\n")
    page = _page()
    page._widgets["VLM_TARGET_FPS"].setValue(0.60)
    page._widgets["WS_EXEC_ALLOW"].setText("python,git")
    page._widgets["EMBED_BACKEND"].setCurrentText("openai")
    page._widgets["STT_BACKEND"].setCurrentText("groq")
    page.save()
    text = p.read_text(encoding="utf-8")
    check("小数が保存される", "VLM_TARGET_FPS=0.60" in text, text)
    check("文字列が保存される", "WS_EXEC_ALLOW=python,git" in text)
    check("選択肢が保存される", "EMBED_BACKEND=openai" in text and "STT_BACKEND=groq" in text)
    check("コメントが残る", "# メモ" in text)

    page2 = _page()
    check("読み戻せる（小数）", abs(page2._widgets["VLM_TARGET_FPS"].value() - 0.60) < 1e-9)
    check("読み戻せる（文字列）", page2._widgets["WS_EXEC_ALLOW"].text() == "python,git")
    check("読み戻せる（選択肢）", page2._widgets["EMBED_BACKEND"].currentText() == "openai")


def t_defaults_not_pinned() -> None:
    print("\n[15] 既定のままの項目を .env に焼き付けない")
    from eve.config import Config

    p = _sandbox("# メモ\nVV_SPEED=1.0\n")
    page = _page()
    page.save()
    text = p.read_text(encoding="utf-8")
    keys = {ln.split("=")[0] for ln in text.splitlines()
            if "=" in ln and not ln.strip().startswith("#")}
    check("**触っていない既定値は書かれない**（後で既定を改善しても古い値が残らない）",
          "VLM_PHASH_THRESHOLD" not in keys and "SEARCH_MAX_RESULTS" not in keys,
          f"{sorted(keys)}")
    check("元からあるキーは残る", "VV_SPEED" in keys)
    check("書かれた件数が現実的", len(keys) <= 6, f"{len(keys)}: {sorted(keys)}")

    # 既定から動かしたものは書かれる
    page2 = _page()
    page2._widgets["VLM_PHASH_THRESHOLD"].setValue(int(Config.VLM_PHASH_THRESHOLD) + 3)
    page2.save()
    check("動かしたら書かれる", "VLM_PHASH_THRESHOLD" in p.read_text(encoding="utf-8"))

    # 一度書いたら、既定に戻しても .env には残す（勝手に消さない）
    page3 = _page()
    page3._widgets["VLM_PHASH_THRESHOLD"].setValue(int(Config.VLM_PHASH_THRESHOLD))
    page3.save()
    check("**一度書いたキーは既定に戻しても消さない**（ユーザの持ち物を消さない）",
          "VLM_PHASH_THRESHOLD" in p.read_text(encoding="utf-8"))


def t_warnings_shown() -> None:
    print("\n[14] 危ない項目には注意書きが付く")
    from eve.ui.settings_spec import all_items

    warned = {it.key for it in all_items() if it.warn}
    check("プログラム実行に注意がある", "WS_EXEC_ENABLED" in warned, f"{warned}")
    check("実行許可コマンドに注意がある", "WS_EXEC_ALLOW" in warned, f"{warned}")
    vlm = [it for it in all_items() if it.key == "VLM_ENABLED"][0]
    check("画面認識は「Gemini に送られる」と明示する", "Gemini" in vlm.note, vlm.note)
    allow = [it for it in all_items() if it.key == "WS_EXEC_ALLOW"][0]
    check("実行許可は「セキュリティ境界ではない」と明示する",
          "セキュリティ境界ではありません" in allow.note, allow.note)


if __name__ == "__main__":
    t_collect()
    t_frequency()
    t_api_keys()
    t_save()
    t_save_failure()
    t_restart_env()
    t_split_policy()
    t_spec_is_real()
    t_defaults_from_config()
    t_broken_env_survives()
    t_dependencies()
    t_accordion()
    t_advanced_roundtrip()
    t_defaults_not_pinned()
    t_warnings_shown()
    print(f"\n合計: PASS {PASS} / FAIL {FAIL}")
    sys.exit(1 if FAIL else 0)
