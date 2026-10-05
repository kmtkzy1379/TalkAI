#!/usr/bin/env python3
"""フィラー音声バンクの生成（`fillers.yaml` → `bank/<voice>/`）。

なぜ事前合成なのか（詳細は docs/FILLER_DESIGN.md）:
  フィラーはミリ単位で鳴らす必要があるが、Irodori は最速でも 139〜215ms かかる。
  「その瞬間に音が既に存在している」ことが即時性の条件なので、先に作って置く。
  長さは伸縮ではなく **要素の連結** で調整するため、各要素は短く複数の長さで作る。

声の差し替え:
  `fillers.yaml` の voices に profile を足し、`--voice <名前>` で指定する。
  出力は bank/<voice>/ に分かれるので既存バンクを壊さない。

実行（Irodori の venv で）:
  cd <irodori-bench>/Irodori-TTS
  ./.venv/Scripts/python.exe <repo>/tools/build_filler_bank.py --voice default
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IRODORI = REPO.parent / "irodori-bench" / "Irodori-TTS"


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- 声の指紋（長時間平均スペクトル）------------------------------------
# 話者性はスペクトル包絡に出る。**厳密な話者照合ではない**が、同一テキスト・
# 同一長さの候補から「参照に近いもの」を選ぶ用途には足りる。
# 生成側の条件付け（ref_wavs）が効いているかに依存しない点が要点。
def _ltas(d, sr):
    import numpy as np

    n, hop = 2048, 512
    w = np.hanning(n)
    frames = [np.abs(np.fft.rfft(d[i:i + n] * w))
              for i in range(0, max(len(d) - n, 1), hop)]
    S = np.mean(frames, axis=0) if frames else np.abs(np.fft.rfft(d[:n] * w))
    L = 20 * np.log10(S + 1e-9)
    return (L - L.mean()) / (L.std() + 1e-9)


def _cos(a, b) -> float:
    import numpy as np

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _fingerprint_file(path: Path):
    import soundfile as sf

    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    return _ltas(d, sr)


def _filler_audio():
    """`eve/filler/audio.py`（抑揚の物差し）を読む。Irodori の venv で動くので eve パッケージは import しない。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("filler_audio", REPO / "eve" / "filler" / "audio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=REPO / "fillers.yaml")
    ap.add_argument("--voice", default="default", help="fillers.yaml の voices のキー")
    ap.add_argument("--out", type=Path, default=REPO / "bank")
    ap.add_argument("--irodori-repo", type=Path, default=DEFAULT_IRODORI)
    ap.add_argument("--force", action="store_true", help="既存の wav も作り直す")
    ap.add_argument("--dry-run", action="store_true", help="生成せず件数だけ出す")
    # 声の揺れ対策: 多めに作って参照音声に近いものだけ残す。
    # 生成側の条件付けが効いているかに依存しない（実測でクローンの効果を確認できなかったため）。
    ap.add_argument("--candidates", type=int, default=0,
                    help="1スロットあたりの生成数（0=seeds をそのまま使う・選抜しない）")
    ap.add_argument("--keep", type=int, default=3, help="--candidates 使用時に残す数")
    # S2（IMPL_PLAN §5）: 鳴らす分だけ作る
    ap.add_argument("--roles", default="",
                    help="この role のエントリだけ作る（カンマ区切り。例 independent,terminal）。空=全部")
    ap.add_argument("--shortest", action="store_true",
                    help="各エントリの lengths のうち最短だけ作る（「1語は短く」ユーザ指定 2026-09-25）")
    ap.add_argument("--stretch-sources", action="store_true",
                    help="伸ばし棒の素材（fillers.yaml の stretch.sources）も作る")
    # 注意機構は応答の TTS サーバ（tools/irodori_server.py）と同じ経路にそろえる（声を揃えるため）
    ap.add_argument("--attention", choices=("efficient", "cudnn"), default="efficient")
    # ユーザ 2026-09-26「フィラーは次の言葉探してるから基本は音程同じか、若干下がる」
    # 「感情乗りすぎ」「お系は驚いてる」→ 語尾が上がらず高すぎない候補を残す（calm）
    ap.add_argument("--select", choices=("ref", "calm"), default="ref",
                    help="候補の残し方: ref=参照音声に近い順 / calm=語尾が上がらず高すぎないものを優先"
                         "（その中で参照音声に近い順）")
    ap.add_argument("--variants", default="W",
                    help="合成に使う文字の書き方（| 区切り・W が語）。例 'W|W…'。候補は書き方ごとに作る")
    ap.add_argument("--emoji", default="", help="合成の文字の後ろに付ける絵文字（例 🤭＝笑いの声色）")
    ap.add_argument("--laugh", action="store_true",
                    help="笑いの声色で作る（fillers.yaml の laugh.emoji を付ける・コマンドラインで絵文字を渡さずに済む）")
    ap.add_argument("--labels", default="", help="この label を持つエントリだけ作る（例 笑い）")
    ap.add_argument("--name", default="", help="出力フォルダ名（既定=--voice）。既存のバンクを残して別に作る時")
    # ユーザ 2026-09-28「1フィラー、1〜1.5秒くらいが最適かも」→ 応答非依存フィラーだけ長さを上書きする
    # （特別フィラー っと・っとね は語の後ろに付く短い音なので --shortest のまま）
    ap.add_argument("--lengths", default="",
                    help="応答非依存フィラー（role independent）の長さ[秒]を上書き（カンマ区切り・例 1.0,1.5）")
    ap.add_argument("--min-voiced-sec", type=float, default=0.0,
                    help="前後の無音を刈った長さ（鳴らす時の長さ）がこれ未満の候補は後回し（長い版のフィラー用）")
    # ユーザ 2026-09-29「普通に繋げてすべて作成してもいいかも…現在のほぼランダムフィラーならすべて作成したほうが
    # クオリティ高い」（語と っと を間を空けて繋いだ音は「少し不自然」）→ 付けてよい語 × っと・っとね を1つの音で作る
    ap.add_argument("--combos", action="store_true",
                    help="応答非依存フィラー＋特別フィラー（っと・っとね）をまとめて1つの音で作る（あの・その には付けない）。"
                         "長さは --lengths（既定 1.2）。この時は --roles / --labels を使わない")
    args = ap.parse_args()

    if str(args.irodori_repo) not in sys.path:
        sys.path.insert(0, str(args.irodori_repo))

    cfg = load_yaml(args.table)
    if args.laugh and not args.emoji:
        args.emoji = str((cfg.get("laugh") or {}).get("emoji") or "")
        if not args.emoji:
            raise SystemExit("--laugh には fillers.yaml の laugh.emoji が要る")
    voices, syn, entries = cfg["voices"], cfg["synthesis"], cfg["entries"]
    if args.voice not in voices:
        raise SystemExit(f"voice '{args.voice}' が fillers.yaml に無い（候補: {list(voices)}）")
    voice = dict(voices[args.voice] or {})
    # ref_wavs / lora_adapter は **このリポジトリ基準**で解決する
    # （実行 cwd は Irodori 側になるため、相対パスのままだと開けない）
    def _abs(p: str) -> str:
        q = Path(p)
        return str(q if q.is_absolute() else (REPO / q).resolve())

    if voice.get("ref_wavs"):
        voice["ref_wavs"] = [_abs(p) for p in voice["ref_wavs"]]
        for p in voice["ref_wavs"]:
            if not Path(p).is_file():
                raise SystemExit(f"参照音声が見つからない: {p}")
    if voice.get("lora_adapter"):
        voice["lora_adapter"] = _abs(voice["lora_adapter"])

    # 生成する組み合わせ。--candidates 指定時は多めに作って選抜する
    n_cand = args.candidates or len(syn["seeds"])
    n_keep = min(args.keep, n_cand) if args.candidates else n_cand
    slots = []  # (entry, seconds) の一覧
    default_lengths = syn.get("default_lengths") or [0.5, 0.8, 1.2]
    user_name = (cfg.get("user_name") or "").strip()
    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    if args.combos:
        # 表の語は足していない: fillers.yaml の special（抽選する特殊フィラーの系）の語 × 特別フィラーの組だけ
        # （no_terminal の語は special に書けない＝ eve/filler/bank.py が止める）
        sp_words = {w for f in ((cfg.get("special") or {}).get("families") or {}).values()
                    for w in ((f or {}).get("words") or [])}
        words = [e for e in entries if "independent" in (e.get("role") or []) and e["text"] in sp_words
                 and not e.get("no_terminal")]
        if not words:
            print("fillers.yaml の special に語が無い（組を作らない）")
        terms = [e for e in entries if "terminal" in (e.get("role") or [])]
        combo_lengths = [float(x) for x in (args.lengths or "1.2").split(",") if x.strip()]
        for w in words:
            for tm in terms:
                # combo_text = 組にする時の書き方（例 う → うん＝うんっと・ユーザ 2026-09-29）
                ent = {"id": f"{w['id']}+{tm['id']}", "text": (w.get("combo_text") or w["text"]) + tm["text"],
                       "labels": [], "role": ["combo_terminal"], "word": w["id"], "terminal": tm["id"]}
                for sec in combo_lengths:
                    slots.append((ent, sec))
        entries = []  # 組だけを作る（下の通常の語は作らない）
    for e in entries:
        if roles and not (set(e.get("role") or []) & roles):
            continue
        if labels and not (set(e.get("labels") or []) & labels):
            continue
        if "{user_name}" in e["text"]:
            if not user_name:
                print(f"  スキップ: {e['id']}（fillers.yaml の user_name が未設定）")
                continue
            e = dict(e, text=e["text"].replace("{user_name}", user_name))
        # lengths を書いていないエントリは共通の既定値を使う
        lengths = [float(x) for x in (e.get("lengths") or default_lengths)]
        if args.lengths and "independent" in (e.get("role") or []):
            lengths = [float(x) for x in args.lengths.split(",") if x.strip()]
        elif args.shortest:
            lengths = [min(lengths)]
        for sec in lengths:
            slots.append((e, sec))
    if args.stretch_sources:
        st = cfg.get("stretch") or {}
        for vowel, text in (st.get("sources") or {}).items():
            # 表の語ではなく「ー」の素材。id は stretch_<母音>
            slots.append(({"id": f"stretch_{vowel}", "text": text, "labels": [],
                           "role": ["stretch_src"], "vowel": vowel},
                          float(st.get("source_sec") or 1.6)))
    print(f"表: {args.table}  声: {args.voice}")
    n_var = len([v for v in args.variants.split("|") if v] or ["W"])
    print(f"スロット {len(slots)} × 候補 {n_cand}（書き方 {n_var} 通りごと）→ 採用 {n_keep}"
          f"  = 生成 {len(slots)*n_cand*n_var} / 最終 {len(slots)*n_keep} クリップ"
          f"（選び方: {args.select}）")
    if args.candidates:
        print("  ※ 参照音声に最も近いものを残す（声の揺れを生成側に頼らない）")
    if args.dry_run:
        return 0

    if args.attention == "efficient":
        # tools/irodori_server.py と同じ差し替え（理由と実測はそちらのコメント）
        from torch.nn.attention import SDPBackend

        import irodori_tts.attention as _att

        _att._SDPA_PRIORITY[:] = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    from irodori_tts.inference_runtime import (
        RuntimeKey, SamplingRequest, default_runtime_device,
        download_hf_checkpoint, get_cached_runtime, save_wav,
    )

    t0 = time.perf_counter()
    runtime, _ = get_cached_runtime(RuntimeKey(
        checkpoint=download_hf_checkpoint(syn["checkpoint"]),
        model_device=default_runtime_device(),
        model_precision=syn["precision"],
        codec_device=syn["codec_device"],
    ))
    print(f"モデルロード {time.perf_counter() - t0:.1f}s")

    out_dir = args.out / (args.name or args.voice)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    # 既存 manifest は引き継ぐ（tier を分けて回しても積み上がる）
    clips: dict[str, dict] = {}
    if manifest_path.exists() and not args.force:
        try:
            clips = {c["file"]: c for c in json.loads(
                manifest_path.read_text(encoding="utf-8"))["clips"]}
        except Exception:
            clips = {}

    # 参照音声の指紋（長時間平均スペクトル）。選抜の基準に使う
    ref_fp = None
    if args.candidates and voice.get("ref_wavs"):
        ref_fp = _fingerprint_file(REPO / voice["ref_wavs"][0])
    fa = _filler_audio()
    ref_hz = 0.0
    if args.select == "calm":
        if not voice.get("ref_wavs"):
            raise SystemExit("--select calm は参照音声（声の高さの基準）が要る")
        import soundfile as sf

        rx, rsr = sf.read(voice["ref_wavs"][0], dtype="float32")
        if rx.ndim > 1:
            rx = rx.mean(axis=1)
        import numpy as np

        ref_hz = float(np.median(fa.pitch_track(rx, rsr)))
        print(f"声の高さの基準（参照音声の中央値）{ref_hz:.0f}Hz")

    made = skipped = failed = 0
    walls: list[float] = []
    for e, sec in slots:
        keep_names = [f"{e['id']}_{sec:.1f}_{k}.wav" for k in range(n_keep)]
        if not args.force and all(
                (out_dir / n).exists() and n in clips for n in keep_names):
            skipped += n_keep
            continue
        # 絵文字はテキストにそのまま埋める（Irodori はインライン注釈として読む）
        variants = [v for v in args.variants.split("|") if v] or ["W"]
        cands = []
        for ci in range(n_cand * len(variants)):
            text = variants[ci % len(variants)].replace("W", e["text"]) + args.emoji
            seed = int(syn["seeds"][ci]) if ci < len(syn["seeds"]) else 1000 + ci
            req = SamplingRequest(
                text=text,
                no_ref=not voice.get("ref_wavs"),
                ref_wavs=voice.get("ref_wavs") or None,
                caption=voice.get("caption") or None,
                lora_adapter=voice.get("lora_adapter") or None,
                seconds=sec,
                num_steps=syn["num_steps"],
                t_schedule_mode=syn["t_schedule_mode"],
                sway_coeff=syn["sway_coeff"],
                seed=seed,
            )
            t = time.perf_counter()
            try:
                res = runtime.synthesize(req)
            except Exception as ex:  # 1件の失敗で全体を止めない
                print(f"  ✗ {e['id']} {sec:.1f}s seed={seed}: "
                      f"{type(ex).__name__}: {str(ex)[:60]}")
                failed += 1
                continue
            walls.append(time.perf_counter() - t)
            d = res.audio.detach().cpu().numpy().astype("float32").ravel()
            sim = _cos(_ltas(d, res.sample_rate), ref_fp) if ref_fp is not None else 1.0
            trimmed = fa.trim(d, res.sample_rate)  # 鳴らす時はこの長さ（`bank.py` が読み込みで同じく刈る）
            voiced = len(trimmed) / res.sample_rate
            con = None
            if args.select == "calm":
                # 伸ばし棒の素材は、継ぎ足しに使う後ろ側（0.6秒）の抑揚で見る
                x = trimmed
                if "stretch_src" in e["role"]:
                    x = x[-int(res.sample_rate * 0.6):]
                con = fa.contour(x, res.sample_rate, ref_hz)
            cands.append((sim, seed, res, text, con, voiced))

        if not cands:
            continue

        def long_enough(c) -> bool:
            return c[5] >= args.min_voiced_sec

        if args.select == "calm":
            # 落ち着いた候補（calm_score ≤ 1）の中で参照音声に近い順。足りなければ calm_score の小さい順で補う
            # （--min-voiced-sec 指定時は、刈った後に短すぎる候補を後回し）
            good = sorted([c for c in cands if fa.calm_score(c[4]) <= 1.0 and long_enough(c)],
                          key=lambda c: -c[0])
            rest = sorted([c for c in cands if not (fa.calm_score(c[4]) <= 1.0 and long_enough(c))],
                          key=lambda c: (not long_enough(c), fa.calm_score(c[4])))
            cands = good + rest
        else:
            # 参照に近い順に n_keep 件だけ残す（--candidates 未指定なら全件そのまま）
            cands.sort(key=lambda c: (not long_enough(c), -c[0]))
        for k, (sim, seed, res, text, con, voiced) in enumerate(cands[:n_keep]):
            name = f"{e['id']}_{sec:.1f}_{k}.wav"
            save_wav(out_dir / name, res.audio, res.sample_rate)
            clips[name] = {
                "file": name, "id": e["id"], "text": e["text"], "synth_text": text,
                "labels": e.get("labels") or [], "role": e["role"],
                "requested_sec": sec,
                "actual_sec": round(res.audio.shape[-1] / res.sample_rate, 3),
                "voiced_sec": round(voiced, 3),  # 前後の無音を刈った長さ（鳴らす時の長さ）
                "seed": seed, "ref_similarity": round(sim, 4),
            }
            if con is not None:
                clips[name]["pitch_level_st"] = round(con[0], 2)   # 参照音声の中央値との差（半音）
                clips[name]["pitch_slope_st"] = round(con[1], 2)   # 語尾（終わり−始め・半音）
            if e.get("vowel"):
                clips[name]["vowel"] = e["vowel"]
            if e.get("word"):  # 語＋特別フィラーをまとめた音（--combos）
                clips[name]["word"] = e["word"]
                clips[name]["terminal"] = e["terminal"]
            made += 1

    manifest_path.write_text(json.dumps({
        "voice": args.voice,
        "voice_config": voice,
        "synthesis": syn,
        "labels": cfg.get("labels", []),
        "clips": sorted(clips.values(), key=lambda c: (c["id"], c["requested_sec"], c["seed"])),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    med = sorted(walls)[len(walls) // 2] * 1000 if walls else 0.0
    total = sum(c["actual_sec"] for c in clips.values())
    print(f"\n生成 {made} / スキップ {skipped} / 失敗 {failed}")
    print(f"1件あたり中央値 {med:.0f}ms   バンク総再生時間 {total:.1f}s")
    print(f"出力: {out_dir}")
    print(f"目録: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
