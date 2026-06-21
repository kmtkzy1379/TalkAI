# CLAUDE.md — Eve v2

## これは何か

Eve v2 = AI VTuber + 画面認識(VLM) 統合のデスクトップアプリ。v1（`../portfolio8-VLM-AI`）を**パッチ継続せず、新企画書に従って作り直す**プロジェクト。v1 は1年半の地層化で頭脳部ほぼ全域に破綻が広がったため、配線層のみ移植し中身は白紙から再構築する（判断根拠と範囲は `docs/SALVAGE_MANIFEST.md`、統合パイプライン契約は `docs/PIPELINE_DESIGN.md`）。

- 言語: Python（コメント日本語・識別子英語）/ Python 3.13
- Framework: Tkinter(UI) / asyncio(パイプライン本体) / threading(VLM・音声)
- 永続化: JSONL / TXT フラットファイル（DB なし）
- 外部API: OpenAI / Gemini / Groq / (Anthropic=現在停止中) / VOICEVOX(HTTP 127.0.0.1:50021) / VTube Studio(WebSocket) / YouTube Data API

## 実装状況（2026-06-20・コードが正）

実装順に **F0 / F1 / F2 / F2.5 / F3 / F3.5 / P2 スレッド掃除(裁定a) / F4 FeedbackLLM / F5 発話判定(沈黙→自発発話)** まで完了。
Tier-1 決定論テスト **152件が2回連続 PASS**。未実装: SurpriseBus(多生産者集約・VLM時)・(b)自己懐疑(タスク隣接)・VLM・Call-Function・YouTube・UI・配線層PORT(vts/run/launcher/app)。**中核原理 surprise は生産者(F4 `PredictionState`)+消費ゲート(F5 `should_speak`・決定論T2)が両方稼働**。多生産者 SurpriseBus は VLM(第2生産者)時に。
- **引き継ぎ・未対応問題(P1-P3)・docs訂正は `docs/HANDOFF.md` に集約**（新セッションは最初に読む）。
- 現状は**単一 asyncio ループ前提**（mic read=executor／VAD 推論=ループ上同期）。cross-thread 機構は **P2 裁定(a)で削除済**＝loop が全共有 state の唯一所有者。将来 OS スレッドは `PIPELINE_DESIGN.md §9.3` の橋渡し契約経由（VAD 別スレッド化＝最初の利用候補）。
- 埋め込みは `eve/memory/embed/make_embedder(ruri|openai)`（**ModelRegistry とは別系統**・`make_stt` と同方式）。Ruri v3-310m 既定。

## 中核原理（最優先・絶対に薄めない）

**予測誤差(surprise)をリアルタイム一級信号にする。** surprise は (a)自発発話の要否/内容 と (b)文脈不整合の自己懐疑 の両方を**必須引数として**ゲートする。`should_speak(...)` は surprise を `Optional` にしない。surprise を反転したら発話/沈黙判定が反転する death-detection テストが通らなければビルド失敗（v1 で FEP が装飾化した＝症状の根、を再発させない）。

## アーキテクチャ（要点・詳細は docs/PIPELINE_DESIGN.md）

- **2キュー分離**: `StimulusQueue`(応答LLM起動刺激を1本化, priority+merge) と `AudioPlayQueue`(できた音声を順次再生, seq+generation で順序/barge-in)。
- **エッジだけストリーミング**: 入力=増分STT(50–300ms partial で投機開始)、出力=token stream→文分割→TTS→順次再生。間の応答LLMはターン制（VAD/沈黙で境界）。フルデュプレックスにはしない（確定済の設計判断）。
- **ModelRegistry(role→model 間接層)**: provider 非依存。`.env` 既定 + UI から swap/temp。**Claude API は現在停止中 → Sonnet 役は GPT/Gemini で代用**。後で Claude に戻せるよう間接層は必須。
- **surprise の単一更新者**: `SurpriseBus`/`PredictionState` は asyncio loop 所有・同期読み・ロックなし。FeedbackLLM の prediction-diff(0-100) と VLM の screen-diff を集約。
- **スレッドモデルの正**: loop 単一所有・単一書込・ロックなし。`run_in_executor` は state 非接触で値を返すだけ。真の OS スレッドは連続 capture が強制する時のみ＋`PIPELINE_DESIGN.md §9.3` 橋渡し契約必須（OS スレッドは loop 所有 state を直接 mutate 禁止）。サイドカー(FeedbackLLM/VLM/task/search)は §9.4 契約に従う（single-flight + 背圧 latest-wins/watermark）。

## v2 で必ず潰す v1 の問題（必要なものだけ記憶）

1. STT 聞き間違いが多い → 一括 Whisper を捨て**増分ストリーミング**に。
2. RAG が過去に逸れ、今の会話と絡まない → ランキングは優秀だったが**チャンク定義(FB1+応答1)・件数(300, v1は3000)・無言時 pure-random** が真因。ContextAssembler が現在文脈との関連を必須化、全要素にタイムスタンプ。
3. VLM が YOLO 等の低レイヤでハルシネ多く遅い → 重量級CV(YOLO/tracking/analysis/aggregation/saliency)を**全廃**し、capture→Gemini Flash×3→統合 のVLM直読みに。
4. 応答が全体に遅い → ≤3s 予算(下記)を配線で証明(T1)。
5. システムプロンプト肥大でペルソナがロジックに癒着 → **ペルソナは一旦外す**。挙動を先に見る。規律はプロンプトでなく code gate で強制。
6. FEP が応答を駆動しない装飾 → 中核原理で解消。
7. 自発発話がほぼ無言 or 直近会話由来で不自然 → 発話判定LLM + surprise + ランダムRAG(話題の種)で改善。
8. 文脈破綻・タスク未完を疑わず解答だけ返す → surprise による自己懐疑。
9. UI が画面外でStart押せない等の不便 → 完全作り直し。設定ロック + 全設定valid時のみStart + topmost。

## 制約（ハード）

- **応答レイテンシ ≤3s**（精度を保てない場合のみ ≤5s 許容）。本フェーズはパイプライン検証なので 3s 基準。
- **再利用は「客観的・合理的に最適」な場合のみ**。既存コードの引力で非合理な引用・重複コードを書かない。PORT 対象は `docs/SALVAGE_MANIFEST.md` の①のみ（理解した上で移植・blind copy 禁止）。
- **規律はプロンプトでなくコードで強制**（v1 実測: gpt-5.4系は長文プロンプトの禁止規則を守れない）。default-off の feature flag を禁止（死にコード再蓄積の温床）。
- 例外は握りつぶさず logger + 安全フォールバック。突然落とさない。UI更新は `root.after(0, fn)` 経由のみ。

## 進め方（ディシプリン）

1. まず機能を**単純実装**して流れ・レイテンシを確認 → その後深掘り（②→③→④）。
2. 機能ごとに just-in-time プランニング、**branch を切る**。
3. テストは **2回連続成功で合格**、最終統合マージは **5回連続成功**。
4. 非自明な調査・設計・レビュー・実装・テストは **3+エージェントを別観点で並列**起動。意見が割れたら**多数決→多い方を再調査**、**全会一致で実装**。
5. **エージェントの結論は鵜呑みにしない。重要な主張は一次ソース(実ログ行・実コード行)で自分で確認**（v1 で実在バグをエージェントが否定→一次データで覆った教訓）。
6. 不明点・矛盾・仕様未確定は実装前に必ず報告/質問。推測でTech Stack/構成を決めない。

## テスト（docs/PIPELINE_DESIGN.md §7 が正）

決定論(API不要)を基本に T1 レイテンシ配線 / T2 surprise death-detection / T3 順序・世代 / T4 barge-in(自己エコー含む) / T5 provider代用 / **T6 過去参照なし** / **T7 パイプライン破綻なし(E2E 4シナリオ)**。v1 `tools/` Tier-1/2/3 の実機事故シナリオを**回帰仕様として最初に移植**。

## 既知の落とし穴（v1 由来・要注意）

- v1 CLAUDE.md の「`vlm_orig/` 編集禁止 upstream」は**誤り**（実在しない）。
- v1 `config.py` に **GEMINI_API_KEY 欠落**・`AI2_MODEL` 既定が旧ID。v2 では .env に GEMINI を含め、モデルIDは ModelRegistry 既定で管理。
- `feedback_llm.py` を PORT する際は gpt-4o ハードコード/anthropic 既定を撤去し ModelRegistry に一般化。

## .env（v1 からの鍵 + 追加）

`GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`(停止中) / **`GEMINI_API_KEY`(必須)** / `YOUTUBE_API_KEY` / `TARGET_CHANNEL_ID` / `VOICEVOX_URL` / `VTS_PATH` / `VOICEVOX_PATH` / ROLE別モデル既定(`RESPONSE_MODEL` 等)。

## Update Policy
作業後、恒久情報(目的/Stack/構成/設計判断/制約/同じミス防止の指示/既存記述の誤り)が増えた時のみ本書を更新。一時的作業ログ・今回限りの判断・未確定情報は入れない。不明点は推測せず確認。
