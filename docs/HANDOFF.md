# Eve v2 — セッション引き継ぎ & 監査ステータス（2026-06-20）

> 新しいセッションはまず本書 → `CLAUDE.md` → `docs/PIPELINE_DESIGN.md` / `docs/COMPONENT_LOGIC.md` の順で読む。
> **食い違いはコードが正。** 本書は 4 エージェント並列監査＋一次ソース（実コード行）確認で作成。

## 現在地
- ブランチ: `feat/f6-vlm-snapshot`（main から分岐・**main 未マージ**・F5 までは main 統合済）。
- 実装済（実装順）: **F0 / F1 / F2 / F2.5 / F3 / F3.5 / P2 / F4 FeedbackLLM / 応答プロンプト leak 修正 / F5 発話判定 / Fix1-4 話者取り違え修正 / F6 画面認識VLM(snapshot モード)**。
- 未実装: (b)文脈不整合の自己懐疑(タスク管理隣接) / 多生産者 SurpriseBus(VLM は現状 PredictionState 第2生産者で直結) / Call-Function(task/search/screen-op) / YouTube / UI / 配線層PORT(vts/run/launcher/app) / VLM の Gemini Live ストリーム mode(将来)。
  - **F6 で実装（snapshot モード・`eve/vlm/`・既定 off＝`VLM_ENABLED=1`）**: 採択は **単発・複数フレーム VLM**（旧 ×3 self-consistency は探索で廃止＝動き不可/高コスト）。専用スレッド capture→pHash ゲート→`VisionState.ring`(全キャプチャ=変化前アンカー込み A8)→single-flight `VlmWorker` が直近K枚を1回の `vlm_leaf` 呼び出しで実況→`latest_vision`/`note_vlm_surprise`→ガード付き should_speak。**staleness 解= latest-window + single-flight + A1 自己再トリガ + A9 min-interval**。**A11 ハルシネ防止**: 黒/空白/grab失敗は「視認不可」正直マーカ（VLM に語らせない）。発話は F5 経由（v1 nudge スパム回避）。詳細 COMPONENT_LOGIC §H。
  - **未検証（実機のみ）**: 実 mss キャプチャ + 実 Gemini への multi-frame 画像送信（litellm の `image_url`→Gemini `inline_data` 変換）。`tests/test_f6_vlm.py` は grab/narrate を注入し決定論で配線検証済（実 mss/GPU/API は不使用）。
  - **F5 で実装**: 5秒沈黙→`should_speak`→ True で `AUTONOMOUS_SPEECH` 刺激。**surprise は指標（Fix2 裁定）**＝数値の強制ゲート(HI/LO)は撤廃し、surprise+感情(直近FB)+内容を発話判定LLMが総合判断。唯一の hard ゲートは pending_obligation。T2 は「surprise が必須引数として判定に効く配線」（Optional 化禁止・surprise を読む fake で反転）。**Fix1: ユーザ発話が自発発話を中止・削除**（判定中は user_activity_seq で破棄 / barge-in で queue.discard_kind）。発話判定ログ(True/False+理由・deque10・観測専用)。**Q3: バックオフ/再挨拶抑制/沈黙カテゴリ不採用**（5秒連続評価は意図的）。`eve/speech/{decider,monitor}.py`。
  - **Fix3/Fix4（話者取り違えの構造的修正・実機で解消確認）**: 会話を1個の user ブロブに詰めるとモデルが「自分=イブ」を見失い**自分の発話に自答**していた。ContextAssembler を **native チャットロール messages**（system[スタイル+ロールアンカー+RAG/FB等] / user・assistant ターン列 / 最終 user=発話 or 自発指示）へ作り替え。`assemble(...)` は messages リストを返す（旧 AssembledContext/render は廃止）。自発は「返事でなく自分から一言」指示。Fix4b: 直近ターンの相対時刻前置きは応答LLMが復唱する leak のため撤去（接地は RAG 側 timestamp）。
  - **既知の今後課題（未対処）**: (b)自己懐疑＝急な話題転換/同質問の繰り返し等に自分から疑問を持たない（surprise ゲート＋タスク管理フェーズで）。日本語のぎこちなさ＝モデル依存（差し替え/廉価モデル待ち・コード不変）。モデルは `.env` の `RESPONSE_MODEL`/`DECIDE_MODEL`/`FEEDBACK_MODEL` で差替（評価: `tools/f5_model_eval.py`）。
  - **F5 へ繰越（VLM/タスク時）**: 多生産者 `SurpriseBus`（FB diff + VLM screen-diff・現状は `PredictionState.surprise` 単一生産者を直接読む）/ (b)自己懐疑（応答経路・タスク管理隣接）/ 締切近接抑制（`pending_obligation` no-op フック）。
- テスト: **Tier-1 12ファイル 199件 5回連続 PASS**（API不要・決定論）。flaky なし。F6 は `tests/test_f6_vlm.py`（36件・backpressure/single-flight/A1/A11 ⭐含む）。F5 は `tests/test_f5_speech.py`（29件）。実機自然さ評価は `tools/f5_model_eval.py`。
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
- **F4 実装済**（2026-06-21）: FeedbackLLM が各応答後に非同期で {要約/感情/ユーザ感情/次予測/予測差(0-100)/理由/タグ} を生成し、RAG へ `add_chunk`（圧縮埋め込み/展開注入）+ `PredictionState` へ surprise + 直近フィードバック注入。**SurpriseBus 多生産者集約 / should_speak / 完全 T2 は F5**。RAG は実 feedback 運用へ（仮データ運用は終了）。
  - F4 設計の要点: `eve/feedback/`（PredictionState=loop 所有・単一書込・surprise はメソッド API で F5 の VLM 第2生産者化に耐える / parser=タグ付きテキスト・raise しない / FeedbackWorker=single-flight + **watermark/span**）。**watermark 方式で「フィードバックしてない会話＝記憶喪失」を作らない**（前回 feedback 地点〜最新を必ずカバー・起動時 catch-up・shutdown 未完は watermark 未前進で回収）。サイドカー契約は `PIPELINE_DESIGN.md §9.4`。

## 次の実装候補（自然な順）
1. **SurpriseBus + 発話判定LLM(should_speak)**: F4 の `PredictionState.surprise` を per-source 化（FB + VLM screen-diff の集約）し、沈黙 nudge を surprise 必須引数でゲート。**完全な T2 death-detection（surprise 反転で should_speak 反転）をここで追加**。random RAG=話題の種に `rag.random(2)` を供給。
2. VLM（capture→Gemini×3→統合・screen-diff を SurpriseBus へ）／ Call-Function（task/search/screen-op）／ YouTube ／ UI ／ 配線層 PORT。
   - いずれも `PIPELINE_DESIGN.md §9.2-9.4` のサイドカー契約（single-flight + 背圧 latest-wins/watermark）と橋渡し契約に従う。VLM 連続 capture が真スレッド昇格の最初の候補（§9.5）。
