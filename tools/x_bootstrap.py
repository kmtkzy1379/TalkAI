r"""X で話題を最初にまとめて取り込む（ユーザ 2026-10-17「最初は100件ぐらいトレンドでもなんでも取り込んでみる」）。

今の記憶はゲームの話が多く、ゲームの話がゲームの記憶を呼んで、またゲームの話が増える。最初に幅広い話題を入れて薄める。
流れは本番の仕入れ（`eve/xfeed/collector.py`）と同じ: 情報収集AI が選ぶ → X で読む → 中立に要約 → イブが感想。
種類に「いろいろ」（どれにも当てはまらない分野）も使う。トレンドとタイムラインは最初に1回だけ取って使い回す。

    $env:PYTHONIOENCODING="utf-8"
    & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\x_bootstrap.py --count 100
    # 試し: X で読んで感想まで作るが、本番の記憶には入れない（x_preview\ に書いて、記憶の中身を全部出す）
    & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\x_bootstrap.py --count 5 --preview
    # 選ぶところまで（X の検索・感想はしない＝X はトレンド $0.01＋タイムライン分だけ）
    & C:\Users\user\Desktop\eve-v2\.venv\Scripts\python.exe tools\x_bootstrap.py --count 30 --dry-run

- **イブを止めてから使う**（記憶のファイルに書くのを1人にする。動いていたら止まる。--preview と --dry-run は本番のファイルに
  書かないので、動いていても使える）
- 費用の目安: 100件で X が約 $5（1話題 10件 × $0.005）＋ AI 約 $0.5。--max-usd（既定 8・X の分）を超えそうなら止まる。
  1日の上限（X_DAILY_BUDGET_USD）には数えない
- 取り消すには x_memory.jsonl と x_state.json を消す（会話の記憶 rag_memory.jsonl は触らない）
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eve.config import Config  # noqa: E402
from eve.logsetup import configure  # noqa: E402

EVE_SCRIPTS = ("voice_chat.py", "ui_chat.py", "stream_rehearsal.py")
PREVIEW_DIR = REPO / "x_preview"  # 試しの書き先（.gitignore 済み・毎回作り直す）
X_ROLES = ("x_api", "x_curate", "x_digest", "x_impression", "x_interest", "self_model")


def eve_running() -> list[str]:
    """動いているイブ（本番の起動経路）のプロセス。psutil が無ければ空（確かめられない）。"""
    try:
        import psutil
    except ImportError:
        return []
    me = os.getpid()
    hits: list[str] = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if any(s in cmd for s in EVE_SCRIPTS):
                hits.append(f"{p.info['pid']}: {cmd[:100]}")
        except (psutil.Error, OSError):
            continue
    return hits


def print_memories(records: list[dict]) -> None:
    """記憶の中身を全部（`# 参照` で見せる本文そのまま）。"""
    for i, r in enumerate(records, 1):
        print(f"\n===== {i}. [{r.get('category', '')}] {r.get('topic', '')}（検索語「{r.get('query', '')}」・"
              f"投稿 {r.get('posts', '?')}件・興味 {r.get('interest', '?')}）")
        print(r.get("text", ""))


def print_costs(meter) -> None:
    if meter is None:
        return
    by_role = meter.summary()["by_role"]
    rows = [(k, by_role[k]) for k in X_ROLES if k in by_role]
    if rows:
        print("費用: " + " / ".join(f"{k} {v['calls']}回 ${v['cost_usd']:.4f}" for k, v in rows)
              + f"（合計 ${sum(v['cost_usd'] for _k, v in rows):.4f}）")


async def main() -> int:
    ap = argparse.ArgumentParser(description="X で話題を最初にまとめて取り込む")
    ap.add_argument("--count", type=int, default=100, help="記憶に入れる件数（既定 100）")
    ap.add_argument("--batch", type=int, default=10, help="情報収集AI に1回で選ばせる件数（既定 10）")
    ap.add_argument("--timeline", type=int, default=50, help="材料にするタイムラインの件数（ログイン時だけ・1件 $0.005）")
    ap.add_argument("--max-usd", type=float, default=8.0, help="X の費用の上限（既定 $8）")
    ap.add_argument("--preview", action="store_true",
                    help="試し: X で読んで感想まで作るが、本番の記憶には入れない（x_preview\\ に書いて中身を全部出す）")
    ap.add_argument("--dry-run", action="store_true", help="選ぶところまで（X の検索・要約・感想・記憶への書き込みはしない）")
    ap.add_argument("--force", action="store_true", help="イブが動いていても続ける（記憶のファイルが壊れうる）")
    args = ap.parse_args()
    configure()
    if not Config.X_BEARER_TOKEN:
        print("X_BEARER_TOKEN が .env にありません（X の開発者コンソールのアプリの Bearer Token）。")
        return 2
    running = eve_running()
    if running and not (args.force or args.preview or args.dry_run):
        print("イブが動いています。止めてから使ってください（記憶のファイルに書くのを1人にするため）:\n  "
              + "\n  ".join(running))
        return 2

    from eve.memory import RagStore
    from eve.memory.embed import make_embedder
    from eve.memory.self_model import SelfModelStore, SelfRecall
    from eve.model_registry import ModelRegistry
    from eve.persona import load_persona
    from eve.self_model.worker import SelfModelWorker
    from eve.usage import UsageMeter, install
    from eve.xfeed import XApiError, build_collector

    sm_file = Config.SELF_MODEL_FILE
    if args.preview:
        shutil.rmtree(PREVIEW_DIR, ignore_errors=True)  # 前の試しは捨てる（毎回まっさらから）
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        mem_file, state_file, log_file = (str(PREVIEW_DIR / n) for n in ("x_memory.jsonl", "x_state.json", "x_log.jsonl"))
        # 好みの台帳も写しで試す（感想から好みが増えるか・本物の self_model.jsonl は触らない）
        src = REPO / Config.SELF_MODEL_FILE if not Path(Config.SELF_MODEL_FILE).is_absolute() else Path(Config.SELF_MODEL_FILE)
        sm_file = str(PREVIEW_DIR / "self_model.jsonl")
        if src.exists():
            shutil.copyfile(src, sm_file)
    else:
        mem_file, state_file, log_file = Config.X_MEMORY_FILE, None, None  # 本番（None＝.env の X_STATE_FILE・X_LOG_FILE）
    # 費用を最後に出すため、使用量は必ず数える（USAGE_LOG_ENABLED=1 なら本番の usage.jsonl にも残る）
    usage_path = Config.USAGE_LOG_FILE if Config.USAGE_LOG_ENABLED else os.path.join(tempfile.mkdtemp(), "usage.jsonl")
    meter = UsageMeter(usage_path, tag="x_preview" if args.preview else "x_bootstrap")
    install(meter)
    registry = ModelRegistry()
    embedder = make_embedder()
    memory = RagStore(embedder)  # 会話の記憶（材料に読むだけ）
    await memory.initialize()
    store = RagStore(embedder, rag_file=mem_file, max_chunks=Config.X_MEMORY_MAX)
    await store.initialize()
    self_model = SelfModelStore(sm_file) if Config.SELF_MODEL_ENABLED else None
    recall = SelfRecall(self_model, embedder) if self_model is not None else None
    likes_before = self_model.as_dict() if self_model is not None else {}
    # 感想から好みを育てる係（本番と同じ。ここでは回し続けず、1回ぶんごとに flush する）
    sm_worker = (SelfModelWorker(model_registry=registry, store=self_model, cache=None)
                 if self_model is not None and not args.dry_run else None)
    collector = build_collector(
        registry=registry, store=store, memory=memory, persona=load_persona(os.getenv("PERSONA_FILE")),
        self_model_fn=(lambda: self_model.render()) if self_model is not None else None,
        self_model_pick=recall.block_for if recall is not None else None,
        on_impression=sm_worker.note_text if sm_worker is not None else None, embedder=embedder,
        state_path=state_file, log_path=log_file)
    client = collector.client
    print(f"会話の記憶 {len(memory)}件 / Xで見た記憶 {len(store)}件（{store.rag_path}）"
          + ("　※試し: 本番の記憶には入れない" if args.preview else ""))
    print(f"情報収集AI: {registry.resolve('x_curate')} / 要約: {registry.resolve('search_summarize')} / "
          f"感想: {registry.resolve('response')}")
    try:
        try:
            mats = await collector.gather(timeline_posts=args.timeline)
        except XApiError as e:
            print(f"X API が使えません（{e}）。X_BEARER_TOKEN と開発者コンソールの残高を確かめてください。")
            return 1
        print(f"トレンド {len(mats.trends)}件・タイムライン {len(mats.timeline)}件"
              + ("" if collector.has_timeline else "（ログインしていないので使わない・tools/x_login.py）")
              + f"（X ${client.spent_usd:.3f}）")
        if args.dry_run:
            picks = await collector.curate(mats, count=args.count, candidates=args.count, bootstrap=True)
            for i, p in enumerate(picks, 1):
                print(f"{i:3d}. [{p.category}] {p.topic}｜{p.query}｜{p.why}")
            print(f"\n候補 {len(picks)}件（X ${client.spent_usd:.3f}・記憶には入れていない）")
            print_costs(meter)
            return 0
        # まとめて増やす時は、もう持っている X の記憶とも、この取り込みで入れた話題とも似た話題（言い換え）を取らない
        # （ユーザ 2026-10-17「一気に記憶増えるときは類似度近い奴は除外」・`collector.SIMILAR_TOPIC`）
        pool = [v for v in [await collector.topic_vec(r.get("topic", "")) for r in store.records()] if v is not None]
        total, empty, round_no = 0, 0, 0
        while total < args.count:
            left = args.max_usd - client.spent_usd
            if left < 0.06:
                print(f"X の費用が上限 ${args.max_usd:.2f} に近いので止めた")
                break
            n = min(args.batch, args.count - total)
            round_no += 1
            rec = await collector.run_once(reason=f"{'試し' if args.preview else '最初の取り込み'} {round_no}回目",
                                           count=n, candidates=n + 4, bootstrap=True, materials=mats,
                                           max_usd=left, daily=False, similar_pool=pool)
            total += rec["stored"]
            if sm_worker is not None:
                await sm_worker.flush()  # この回の感想から好みを育てる（次の回の情報収集AI・感想に効く）
            print(f"--- {round_no}回目: {rec['stored']}件（合計 {total}/{args.count}・X ${client.spent_usd:.3f}）")
            for it in rec.get("items") or []:
                if it.get("skip"):
                    print(f"    飛ばした: [{it.get('category', '')}] {it.get('topic', '')} … {it['skip']}")
            empty = empty + 1 if rec["stored"] == 0 else 0
            if empty >= 2:
                print("2回続けて1件も入らなかったので止めた（x_log.jsonl の skip を見てください）")
                break
        if args.preview:
            print_memories(store.records())
            print(f"\n試しの記憶 {total}件（本番の x_memory.jsonl には入れていない・中身と読んだ投稿は {PREVIEW_DIR}）")
        else:
            print(f"\n記憶に入れた: {total}件 / Xで見た記憶は全部で {len(store)}件")
        if self_model is not None:
            after = self_model.as_dict()
            changed = {k: v for k, v in after.items() if likes_before.get(k) != v}
            gone = [k for k in likes_before if k not in after]
            print("好みの台帳" + ("（写し）" if args.preview else "") + ": "
                  + ("変わらず" if not changed and not gone else
                     " / ".join([f"{k}: {' > '.join(v)}" for k, v in changed.items()] + [f"{k}: 消えた" for k in gone])))
        print(f"X: {client.requests}回・投稿 {client.posts_read}件")
        print_costs(meter)
        return 0
    finally:
        await store.shutdown()
        await memory.shutdown()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
