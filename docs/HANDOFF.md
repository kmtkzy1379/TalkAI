# Eve v2 — セッション引き継ぎ & 監査ステータス（2026-06-20）

> 新しいセッションはまず本書 → `CLAUDE.md` → `docs/PIPELINE_DESIGN.md` / `docs/COMPONENT_LOGIC.md` の順で読む。
> **食い違いはコードが正。** 本書は 4 エージェント並列監査＋一次ソース（実コード行）確認で作成。

## 現在地
- ブランチ: `feat/f3.5-long-term-rag`（main 未マージ）。
- 実装済（実装順）: **F0 基盤 / F1 2キュー骨格 / F2 応答背骨 / F2.5 声ループ / F3 短期記憶 / F3.5 長期RAG（連想想起）**。
- 未実装: FeedbackLLM / SurpriseBus・中核原理(surprise) / 発話判定(沈黙nudge) / VLM / Call-Function(task/search/screen-op) / YouTube / UI / 配線層PORT(vts/run/launcher/app)。
- テスト: **Tier-1 9ファイル 94件 2回連続 PASS**（API不要・決定論）。flaky なし。
- venv は v1 のものを流用: `C:\Users\tester\Desktop\portfolio8-VLM-AI\venv`（`sentence-transformers` 導入済＝Ruri 用）。

## 実行コマンド
- Tier-1: `$env:PYTHONIOENCODING="utf-8"; & C:\Users\tester\Desktop\portfolio8-VLM-AI\venv\Scripts\python.exe tests\test_<name>.py`（各ファイル個別。runner/discovery 無し）。
- 声ループ実機: `python tools\voice_chat.py`（要 .env キー + VOICEVOX 起動）。
- RAG 精度判定: `python tools\rag_experiment.py --backend ruri [--sweep] [--pool]`（仮データ seed → 抽出ログ）。

## 新セッション用の引き継ぎプロンプト（例）
> 「eve-v2 (C:\Users\tester\Desktop\eve-v2) の続き。まず docs/HANDOFF.md と CLAUDE.md を読んで現状把握して。
> 次は <FeedbackLLM> を実装したい。ワークフロー（feature ごとに branch / テスト2回連続緑で合格 /
> 非自明な調査・設計は複数エージェント＋一次ソース確認 / 規律はプロンプトでなくコードで強制 / 不明点は質問）に従って。」

## アーキ要点（コードが正）
- **単一 asyncio ループ前提**。記憶層（`ConversationCache`/`RagStore`）は同期 API、背景書き込みのみ非同期。
  mic/VAD も現状ループ上で動く（別スレッド化は未実施。下記 P2 参照）。
- **2キュー**: `StimulusQueue`(応答起動の単一窓口・priority/merge/dedup/aging/USER coalesce) /
  `AudioPlayQueue`(できた音声を seq+generation で順序保証・barge-in・C5 の on_played 報告)。
- **ResponseOrchestrator.handle**: 直近会話(`cache.recent_for_injection`) + 長期RAG(`rag.search`) を
  ContextAssembler に注入 → 応答LLM stream → 文分割 → TTS → AudioPlayQueue。barge-in は世代比較で停止。
  C5: 実際に再生し終えた文だけを eve ターンとして記録（`spoken` via on_played）。
- **埋め込み**: `make_embedder(ruri|openai)`（**ModelRegistry とは別系統**、`make_stt` と同方式）。既定 Ruri v3-310m。
- **RAG ランキング**: memory-stream（relevance + importance + recency）+ **異方性 baseline 補正**
  `rel'=(cos-baseline)/(1-baseline)` + 関連度フロア + top-1(最類似)保証 + MMR 多様化。
  確定値: `REL0.7 / IMP0.18 / REC0.12 / baseline0.77 / floor0.15`（Ruri sweep 実測 2026-06-20）。env 可変。
- **圧縮埋め込み/展開注入**: RAG chunk は検索キー=要約+タグのみ埋め込み、注入=完全版 text。
  feedback 実装時は `add_chunk(text=<完全版>, summary=<要約>, topic_tags=..., emotions=...)` を呼ぶだけ（土台実装済）。

## 監査で見つかった未対応問題（新セッションで優先度順に）
- **[P2 RESOLVED 2026-06-21] スレッドモデル裁定 = (a) 採用**: 単一ループを正とし `AudioPlayQueue.set_loop`/`_loop`/cross-thread `interrupt` 分岐（呼び出し0の死にコード）を**削除済**。`interrupt()` は `bump_generation()` の薄いラッパに簡素化（ループ上前提を docstring 明記）。将来の OS スレッド（VLM 連続 capture／audio callback）は **`PIPELINE_DESIGN.md §9` の橋渡し契約 + サイドカー契約**経由（OS スレッドは loop 所有 state を直接 mutate 禁止）。VAD 推論はループ上同期を**据え置き**（軽量・壊れていない／別スレッド化は audio callback API 移行時にまとめて＝§9.3 の最初の利用者）。回帰ガード: `tests/test_f1_pipeline.py`。
- **[P2] `StreamFn` 型が2箇所で不整合**（`model_registry.py` vs `response/orchestrator.py`）。VoiceLoop の adapter で
  実害は出ていないが統一推奨（監査 AUDIT C3）。
- **[P2] `handle()` の `await audio.join()`** は play_worker 稼働が前提。停止順序/将来 refactor で hang しうる。
  `wait_for(timeout)` か per-generation の drained-event 化を検討。
- **[P3] C5 barge-in の spoken 記録**が再生中の1文分ズレうる（単一スレッドで競合ではない・許容範囲）。
- **[P3] `drain_user_texts` はロックなし**（単一 consumer 前提。将来 `get`→`drain` 間に await を挟むと壊れる landmine）。

## テスト未カバー（重要・将来機能に影響）
- **実 embedder(ruri/openai) の挙動**は決定論テスト外（fake は `rel_baseline=0`）。RAG 実精度は `tools/rag_experiment.py` で手動判定。
- **audio_input VAD/mic・MicSttInputSource**（barge-in トリガ）にテスト無し。
- **C5/barge-in の実再生タイミング**、**autonomous-drift の実 orchestrator 多ターン**は未テスト（ロジックは unit でカバー）。
- VoiceLoop **構築**は `tests/test_voiceloop_wiring.py` で新規カバー（引数ドリフト検出）。`run()` 実行は未カバー。

## このセッションで修正済
- `clock.elapsed_wall`: tz-naive timestamp を UTC 扱い＋例外吸収（naive/aware 減算でターンが落ちるのを防止）。
- `audio_play_queue.bump_generation`: `_reserve` の世代ごと無制限増加を prune（監査 D1）。
- `tests/test_voiceloop_wiring.py` 追加（配線スモーク）/ `test_f0` に elapsed_wall 防御テスト追加。

## 既存 docs の訂正（古い記述・コードが正）
- **AUDIT_F0-F2.md** は 2026-06-17 スナップショット。**A1/A2/A3・B2/B3/B4・C1・C4・C5・D2 は解消済**（F2.5/F3/F3.5）。
  未解消: **C2(barge-in 二箇所)・C3(StreamFn 不整合)**。D1(_reserve) は本セッションで修正。
- **PIPELINE_DESIGN / COMPONENT_LOGIC** の以下は実装で覆った（裁定済）:
  - STT **partial 投機は不採用**（VAD 区間 → gpt-4o-transcribe final）。
  - **ソフト AEC は不採用**（イヤホン前提＋常時リッスンで物理解決）。
  - RAG 件数 **300 → 500**、注入 **5 → 6 ターン**（≈3往復）、`RAG_MAX_CHUNKS`/`RECENT_TURN_COUNT`。
  - 埋め込みは **ModelRegistry role でなく `make_embedder`**。`summarize` role は追加済（実装先行）。
- **中核原理(surprise/SurpriseBus/should_speak) は未実装**。FeedbackLLM 未実装 → RAG は**仮データ運用**中。

## 次の実装候補（自然な順）
1. **FeedbackLLM**: 各応答後に非同期で {要約 / 感情 / 次予測 / 予測差 / 理由 / タグ} を生成。
   → RAG へ `add_chunk` で書込（土台あり）／中核原理の **surprise(prediction_diff)** を供給。
2. SurpriseBus + 発話判定LLM（沈黙 nudge・random RAG=話題の種に `rag.random(2)` を供給）。
3. VLM（capture→Gemini×3→統合）／ Call-Function（task/search/screen-op）／ YouTube ／ UI ／ 配線層 PORT。
