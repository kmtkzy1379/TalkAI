r"""RAG 自律発話の発火テスト（headless・実 decide LLM・要 .env・コスト小）。

ユーザ懸念「RAGからの自律的発話が確認できない」を、**画面なし(vision=None)で純粋に
沈黙×記憶だけ**の経路に絞って測る。fake clock で沈黙秒数を制御し、本番と同じ
SilenceMonitor ゲート（5秒沈黙＋再評価カデンス）と実 SpeechDecider/RagStore/decide LLM を駆動。

各沈黙窓で「何秒無言で・どの記憶を種に・話したか/黙ったか・記憶に触れたか」を出す。
比(自律/ユーザ)も集計。decide モデルは実機 voice_chat.py 既定に合わせ gpt-4o-mini（引数で gpt-4o 可）。

実行: $env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\rag_autonomous_session.py [gpt-4o|gpt-4o-mini]
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eve.clock import Stamp  # noqa: E402
from eve.config import Config  # noqa: E402
from eve.memory import ConversationCache, RagStore  # noqa: E402
from eve.memory.embed import make_embedder  # noqa: E402
from eve.feedback import PredictionState  # noqa: E402
from eve.model_registry import ModelRegistry  # noqa: E402
from eve.pipeline import Stimulus, StimulusKind, StimulusQueue  # noqa: E402
from eve.speech import SpeechDecider, SpeechState, make_decide_fn  # noqa: E402

# エピソード記憶（フィードバック修正後の品質を代表＝事実中心）。(表示文, 検索キー, 重要度pd)。
SEED = [
    ("ユーザはチーズケーキが好きで、カフェで美味しいチーズケーキを探していた", "チーズケーキ 甘いもの スイーツ カフェ デザート", 70),
    ("ユーザは雨の日に家でゲームやボウリングをして遊ぶのが好き", "雨 室内 ゲーム ボウリング 遊び 家", 65),
    ("ユーザは最近運動不足を気にしてジョギングを始めたがっていた", "運動 ジョギング 健康 散歩 運動不足", 60),
    ("ユーザは週末に映画を見に行く予定があると言っていた", "週末 映画 予定 お出かけ", 50),
    ("ユーザの好きなおにぎりの具は梅と鮭", "おにぎり 梅 鮭 食べ物", 40),
    ("ユーザはコーヒーより紅茶派", "紅茶 コーヒー 飲み物", 25),
    ("ユーザはプログラミングの個人プロジェクトを進めている", "プログラミング 開発 コード", 35),
    ("ユーザは仕事が忙しくて疲れ気味だと話していた", "仕事 疲れ 忙しい 休憩", 45),
]

# (user発話, eve返答(台本), 続く沈黙秒). 沈黙窓ごとに自律判定が走る。
SCRIPT = [
    ("やっほー、イブ", "やっほー！今日はどうしたの？", 12),
    ("あー、甘いものでも食べたい気分だな", "いいね、何か気になるのある？", 22),
    ("雨だと家で何しようかな", "そうだねえ、どうしよっか。", 20),
    ("最近ちょっと運動不足でさ", "わかる、なまっちゃうよね。", 18),
    ("週末ちょっと暇なんだよね", "へえ、何かしたいことある？", 16),
    ("ふー、なんか疲れたな", "おつかれさま、ゆっくりしな。", 14),
]


def sh(s, n=60):
    return (str(s) if s is not None else "—")[:n].replace("\n", " ")


def mem_hit(content):
    """自律発話が種記憶に触れたか（雑なキーワード一致）。"""
    keys = ["チーズケーキ", "カフェ", "甘い", "ボウリング", "ゲーム", "ジョギング", "運動",
            "映画", "週末", "おにぎり", "紅茶", "プログラミング", "仕事", "疲れ"]
    hits = [k for k in keys if k in (content or "")]
    return hits


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    reg = ModelRegistry(overrides={"speech_decide": f"openai/{model}"})
    print(f"decide model -> {reg.resolve('speech_decide')}  (vision=None ＝純 RAG 自律のみ)\n")

    clock = [1000.0]
    state = SpeechState(now_fn=lambda: clock[0])
    cache = ConversationCache(history_file=os.path.join(os.environ.get("TEMP", "/tmp"), "ras_h.jsonl"))
    await cache.initialize()
    rag = RagStore(make_embedder(), rag_file=os.path.join(os.environ.get("TEMP", "/tmp"), "ras_r.jsonl"))
    await rag.warmup()
    # 既存ファイルを汚さないよう毎回新規（initialize はせず空から seed）
    for disp, key, pd in SEED:
        await rag.add_chunk(text=disp, search_text=key, prediction_diff=pd)
    pred = PredictionState()
    queue = StimulusQueue()

    log = []  # (sil, speak, seeds, content)
    _decide = make_decide_fn(reg)

    async def decide_wrap(*, surprise, silence_seconds, recent_turns, topic_seeds, last_feedback=None, vision=None):
        res = await _decide(surprise=surprise, silence_seconds=silence_seconds, recent_turns=recent_turns,
                            topic_seeds=topic_seeds, last_feedback=last_feedback, vision=vision)
        seeds = " / ".join(sh(getattr(c, "text", ""), 28) for c in (topic_seeds or [])) or "—"
        log.append((silence_seconds, res.speak, seeds, res.content, res.reason))
        return res

    decider = SpeechDecider(state=state, cache=cache, rag=rag, prediction_state=pred,
                            queue=queue, decide_fn=decide_wrap, vision_state=None)

    users = 0
    autos = 0
    threshold = Config.SILENCE_THRESHOLD_SEC
    print("========== セッション ==========")
    for u_text, e_text, sil in SCRIPT:
        # ユーザ発話 → eve 返答（台本）。沈黙時計は eve 発話完了からカウント。
        clock[0] += 1.0
        cache.add_turn("user", u_text)
        state.mark_user_utterance()
        users += 1
        cache.add_turn("eve", e_text)
        state.mark_eve_activity()
        print(f"  🧑 {u_text}")
        print(f"  🤖 {e_text}")
        # 沈黙窓: 本番 SilenceMonitor と同じゲートで判定（5秒沈黙＋カデンス＋idle）。
        clock[0] += sil
        if state.eval_due(threshold) and decider.is_idle():
            before = len(log)
            await decider._decide_once()
            sil_s, speak, seeds, content, reason = log[before]
            if speak:
                # 自律刺激が積まれたはず → drain
                got = None
                while queue.qsize() > 0:
                    st = await queue.get()
                    if st.kind == StimulusKind.AUTONOMOUS_SPEECH:
                        got = st.payload
                if got:
                    autos += 1
                    hits = mem_hit(got.content)
                    tag = f"記憶◎{hits}" if hits else "記憶○なし"
                    print(f"  💭[{sil:.0f}秒沈黙] 自律発話 {tag}: {sh(got.content,70)}")
                    print(f"      └理由: {sh(reason,60)} / 種: {seeds}")
                state.mark_eve_activity()  # 喋ったら沈黙時計リセット（実機同様）
            else:
                print(f"  …[{sil:.0f}秒沈黙] 黙る: {sh(reason,56)}")
                print(f"      └種: {seeds}")
        print()

    decided = len(log)
    spoke = sum(1 for e in log if e[1])
    mem_connected = sum(1 for e in log if e[1] and mem_hit(e[3]))
    print("========== 集計 ==========")
    print(f"  decideモデル: {reg.resolve('speech_decide')}")
    print(f"  ユーザ発話={users} / 自律発話={autos} / 比(自律/ユーザ)={autos/max(1,users):.2f}")
    print(f"  沈黙窓での判定={decided} → 話した={spoke}（うち記憶に触れた={mem_connected}）/ 黙った={decided-spoke}")
    sils = [f"{e[0]:.0f}s" for e in log if e[1]]
    print(f"  自律発話が出た沈黙秒数: {', '.join(sils) or '—'}")
    await rag.shutdown()
    await cache.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
