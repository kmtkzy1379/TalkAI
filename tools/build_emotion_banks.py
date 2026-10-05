#!/usr/bin/env python3
"""感情ごとのフィラーのバンクを作る（S5・ユーザ 2026-09-26「フィラーも感情載せるでいいと思う。変だったら変えればいいだけ」）。

`emotion.yaml` の voice_emoji で絵文字が決まっている感情だけ、`bank/<FILLER_VOICE><BANK_SUFFIX>/` を作る
（例 bank/eve_calm_emo_sad/）。作り方は `bank/eve_calm` と同じ（fillers.yaml の冒頭のコマンド）で、
合成する文字の後ろにその感情の絵文字を付ける。伸ばし棒の素材は作らない（感情の声の語は伸ばさない）。
既にあって絵文字も同じならそのまま・絵文字が変わっていたら作り直す。Irodori の venv で1感情ずつ動かす（ローカル GPU・API 不要）。

実行（EVE の venv から・中で Irodori の venv を呼ぶ）:
  $env:PYTHONIOENCODING="utf-8"
  & C:\\Users\\user\\Desktop\\eve-v2\\.venv\\Scripts\\python.exe tools\\build_emotion_banks.py
  # 長い版（ユーザ 2026-09-28「1フィラー、1〜1.5秒」→ 2026-09-29 試聴で 1秒）: 基のバンク名を環境変数で指定し、長さを渡す
  $env:FILLER_VOICE="eve_calm_long"
  & ...python.exe tools\\build_emotion_banks.py --lengths 1.0 --min-voiced-sec 0.9 --variants "W…|Wー…" --roles independent
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.config import Config  # noqa: E402
from eve.emotion import BANK_SUFFIX, load_voice_emoji  # noqa: E402
from eve.response.tts_irodori import resolve_repo_path  # noqa: E402

# bank/eve_calm と同じ作り方（fillers.yaml の冒頭）。伸ばし棒の素材（--stretch-sources）だけ除く
BUILD_ARGS = ["--voice", "eve", "--roles", "independent,terminal", "--shortest",
              "--candidates", "12", "--keep", "3", "--select", "calm", "--variants", "W|W…"]


def built_with(d: Path, emoji: str) -> bool:
    """既にあるバンクが、この絵文字で作ってあるか（合成した文字の後ろ）。"""
    try:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    texts = [str(c.get("synth_text") or "") for c in man.get("clips") or []]
    return bool(texts) and all(t.endswith(emoji) for t in texts)


def _copy_combos(src: Path, dst: Path) -> int:
    """src のバンクの「語＋っと・っとね」をまとめた音を dst に写す（同じ絵文字の感情・例 喜と楽）。写した数。"""
    sman = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dman = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    have = {c["file"] for c in dman.get("clips") or []}
    n = 0
    for c in sman.get("clips") or []:
        if "combo_terminal" in (c.get("role") or []) and c["file"] not in have:
            shutil.copyfile(src / c["file"], dst / c["file"])
            dman.setdefault("clips", []).append(c)
            n += 1
    (dst / "manifest.json").write_text(json.dumps(dman, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def add_combos(args, table: dict, py: str, irodori: str, env: dict) -> int:
    """今ある感情のバンクに「語＋っと・っとね」をまとめた音を足す（ユーザ 2026-09-29「普通に繋げてすべて作成」）。

    作り方は基のバンクの組と同じ（build_filler_bank.py --combos）＋その感情の絵文字。同じ絵文字の感情は作った方から写す。
    """
    base = ["--voice", "eve", "--combos", "--candidates", "12", "--keep", "3", "--select", "calm",
            "--variants", args.variants]
    if args.lengths:
        base += ["--lengths", args.lengths]
    if args.min_voiced_sec:
        base += ["--min-voiced-sec", args.min_voiced_sec]
    built: dict = {}
    for label, emoji in table.items():
        name = Config.FILLER_VOICE + BANK_SUFFIX[label]
        d = REPO / "bank" / name
        if not (d / "manifest.json").is_file():
            print(f"{label}: {name} が無い（先に感情のバンクを作る）— 飛ばす")
            continue
        if emoji in built:
            n = _copy_combos(built[emoji], d)
            print(f"{label}: {name} は {built[emoji].name} と同じ絵文字（{emoji}）なので組を写した（{n}本）")
            continue
        print(f"{label}: {name} に組を {emoji} で足す …")
        r = subprocess.run([py, str(REPO / "tools" / "build_filler_bank.py"), *base,
                            "--name", name, "--emoji", emoji], cwd=irodori, env=env)
        if r.returncode != 0:
            print(f"{label}: 失敗（rc={r.returncode}）")
            return r.returncode
        built[emoji] = d
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="", help="応答非依存フィラーの長さ[秒]（build_filler_bank.py に渡す・長い版）")
    ap.add_argument("--min-voiced-sec", default="", help="刈った後に短すぎる候補を後回し（build_filler_bank.py に渡す）")
    ap.add_argument("--variants", default="W|W…", help="合成の書き方（build_filler_bank.py に渡す）")
    ap.add_argument("--roles", default="independent,terminal",
                    help="作る役割（長い版は independent だけ）")
    ap.add_argument("--combos", action="store_true",
                    help="今ある感情のバンクに「語＋っと・っとね」をまとめた音を足す（長さは --lengths・既定 1.2）")
    args = ap.parse_args()
    build_args = list(BUILD_ARGS)
    build_args[build_args.index("--variants") + 1] = args.variants
    build_args[build_args.index("--roles") + 1] = args.roles
    if args.lengths:
        build_args += ["--lengths", args.lengths]
    if args.min_voiced_sec:
        build_args += ["--min-voiced-sec", args.min_voiced_sec]
    table = load_voice_emoji(Path(resolve_repo_path(Config.EMOTION_TABLE)))
    if not table:
        print("emotion.yaml の voice_emoji が全部未決（null）— 作るものが無い")
        return 0
    py = resolve_repo_path(Config.IRODORI_PYTHON)
    irodori = resolve_repo_path(Config.IRODORI_REPO)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if args.combos:
        return add_combos(args, table, py, irodori, env)
    built: dict = {}  # 絵文字 → 作ってあるバンク（同じ絵文字の感情はコピーで済ませる・例 喜と楽が同じ 😊）
    for label, emoji in table.items():
        name = Config.FILLER_VOICE + BANK_SUFFIX[label]
        d = REPO / "bank" / name
        if d.exists() and built_with(d, emoji):
            print(f"{label}: {name} は {emoji} で作ってある（そのまま）")
            built.setdefault(emoji, d)
            continue
        if d.exists():
            print(f"{label}: {name} は別の絵文字で作ってあるので作り直す（→ {emoji}）")
            shutil.rmtree(d)
        if emoji in built:
            shutil.copytree(built[emoji], d)
            print(f"{label}: {name} は {built[emoji].name} と同じ絵文字（{emoji}）なのでコピー")
            continue
        print(f"{label}: {name} を {emoji} で作る …")
        r = subprocess.run([py, str(REPO / "tools" / "build_filler_bank.py"), *build_args,
                            "--name", name, "--emoji", emoji], cwd=irodori, env=env)
        if r.returncode != 0:
            print(f"{label}: 失敗（rc={r.returncode}）")
            return r.returncode
        built[emoji] = d
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
