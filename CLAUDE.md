# CLAUDE.md — Eve v2

## これは何か

Eve v2 = AI VTuber + 画面認識(VLM) 統合の**常駐音声対話システム（CLI・GUI は未実装）**。v1（`../portfolio8-VLM-AI`）を**パッチ継続せず、新企画書に従って作り直す**プロジェクト。v1 は1年半の地層化で頭脳部ほぼ全域に破綻が広がったため、配線層のみ移植し中身は白紙から再構築する（判断根拠と範囲は `docs/SALVAGE_MANIFEST.md`＝完了・歴史資料、統合パイプライン契約は `docs/PIPELINE_DESIGN.md`）。

- 言語: Python（コメント日本語・識別子英語）/ Python 3.13
- 実行形態: asyncio 単一ループ本体 + 専用 OS スレッド（VLM capture のみ）。**UI は未実装**（Tkinter は企画上の予定）
- 永続化: JSONL フラットファイル（DB なし・`.txt` を読み書きするコードは無い）
- 外部API: OpenAI / Gemini / Groq / (Anthropic=現在停止中) / VOICEVOX(HTTP 127.0.0.1:50021) / VTube Studio(未実装) / YouTube Data API(未実装)

## 実装状況・既知の問題

**`docs/HANDOFF.md` が唯一の情報源**（本書には実装状況のスナップショットを置かない＝腐る構造をやめる）。
新セッションは HANDOFF を最初に読む。起動方法・テスト方法・E2E ハーネスの使い方も HANDOFF にある。

## 起動とテスト（詳細は HANDOFF）

```powershell
# 実起動（唯一の本番経路。UI/ランチャは無い）
$env:PYTHONIOENCODING="utf-8"
& .\.venv\Scripts\python.exe tools\voice_chat.py

# Tier-1 決定論テスト（runner 無し・1ファイルずつ実行）
& .\.venv\Scripts\python.exe tests\test_f5_speech.py
```

- **`$env:PYTHONIOENCODING="utf-8"` は必須**（日本語＋絵文字を出すので付けないと cp932 で落ちる）
- venv は v1 のものを流用（v2 に venv は無い）。pytest は未導入＝`pytest tests/` は動かない
- **機能フラグは `.env` で明示する**（下記）

## 中核原理（最優先・絶対に薄めない）

**予測誤差(surprise)をリアルタイム一級信号にする。** surprise は (a)自発発話の要否/内容 と (b)文脈不整合の自己懐疑 の両方を**必須引数として**ゲートする設計。
**現状 (a) のみ稼働・(b) は未実装**（消費者未着手）。`should_speak(...)` は surprise を `Optional` にしない。surprise を反転したら発話/沈黙判定が反転する death-detection テストが通らなければビルド失敗（v1 で FEP が装飾化した＝症状の根、を再発させない）。

**surprise は数値で発話を絶対決定しない（Fix2 裁定・生存中）**。発話判定LLM が surprise + 感情 + 内容を総合判断する「指標」として渡す。閾値比較のコードを入れてはならない。

## アーキテクチャ（要点・詳細は docs/PIPELINE_DESIGN.md と docs/COMPONENT_LOGIC.md）

- **2キュー分離**: `StimulusQueue`(応答LLM起動刺激を1本化, priority+merge) と `AudioPlayQueue`(できた音声を順次再生, seq+generation で順序/barge-in)。
- **エッジだけストリーミング**: 入力=VAD 区間ごとの STT（**partial 投機は不採用**）、出力=token stream→文分割→TTS→順次再生。応答LLMはターン制。フルデュプレックスにはしない（確定済）。
- **ModelRegistry(role→model 間接層)**: provider 非依存。`.env` 既定 + 将来 UI から swap。**Claude API は現在停止中 → Sonnet 役は GPT/Gemini で代用**。
- **surprise の集約は `PredictionState`**（`SurpriseBus` というクラスは作らない）。FeedbackLLM の prediction-diff と VLM の screen-diff を **most-recent-source-wins** で合成（max ではない＝古い高 surprise の固着を避ける）。
- **スレッドモデルの正**: loop 単一所有・単一書込・ロックなし。`run_in_executor` は state 非接触。真の OS スレッドは現在 **1本（`vlm-capture`）**で、`PIPELINE_DESIGN.md §9.3` の橋渡し契約に従う（loop 所有 state を直接 mutate 禁止）。サイドカー(FeedbackLLM/VLM/task/search/配達確認)は §9.4 契約に従う（single-flight + 背圧 latest-wins/watermark）。
- **規律はコードゲートで強制**: プロンプトで守られない規律は、判定後のコードで止める。実装済みゲートの一覧は `docs/COMPONENT_LOGIC.md` の自律発話節にある（同内容抑制・話題の丸投げ抑制・時制ゲート・既出用件・STT待ち窓・1ホップ抑制 等）。

## v2 で必ず潰す v1 の問題（必要なものだけ記憶）

1. STT 聞き間違いが多い → 一括 Whisper を捨て VAD 区間ごとの STT に（partial 投機は探索の結果不採用）。
2. RAG が過去に逸れ、今の会話と絡まない → 真因は**チャンク定義・件数(500, v1は3000)・無言時 pure-random**。ContextAssembler が現在文脈との関連を必須化、全要素にタイムスタンプ。
3. VLM が YOLO 等の低レイヤでハルシネ多く遅い → 重量級CVを全廃し、capture→Gemini Flash 直読みに。
4. 応答が全体に遅い → 予算を配線で証明（下記「制約」参照）。
5. システムプロンプト肥大でペルソナがロジックに癒着 → **ペルソナは一旦外す**。規律はプロンプトでなく code gate で強制。
6. FEP が応答を駆動しない装飾 → 中核原理で解消。
7. 自発発話がほぼ無言 or 直近会話由来で不自然 → 発話判定LLM + surprise + 記憶からの話題の種。**J-2 で大幅改修済**（HANDOFF 参照）。
8. 文脈破綻・タスク未完を疑わず解答だけ返す → surprise による自己懐疑（**未実装**）。
9. UI が画面外でStart押せない等の不便 → 完全作り直し（未実装）。

## 制約（ハード）

- **応答レイテンシ: 理想は ≤3s / 現状は ≤5s を許容**（裁定 2026-07-29）。
  実測（通しE2E×3回 pooled・n=92・2026-07-30）: 中央値 3.14s / 平均 3.29s / 最大 9.70s / 3秒超 54% / 5秒超 11%。
  （旧: 単一ラン n=32・2026-07-29 では 中央値 2.57s / 平均 3.01s / 3秒超 44%。**悪化しており ≤5s も 10/92 で超過**。
  system プロンプト増加が疑わしいが未検証＝要調査。）
  応答モデルを品質優先で `gpt-5.5` にした代償（`gpt-5.4-mini` は 1.5-1.8s だが数値歪曲・先回り・tool_calls 漏れの品質問題があり不採用）。
  **≤3s の理想は下ろさない**。より賢くて速いモデルが出た時点で `RESPONSE_MODEL` を差し替えて再計測する（ModelRegistry の間接層があるので差し替えは既定値の変更だけで済む）。
- **再利用は「客観的・合理的に最適」な場合のみ**。既存コードの引力で非合理な引用・重複コードを書かない。
- **機能フラグの扱い**: 機能フラグは**ユーザが ON/OFF する設定項目**（将来 UI から操作する前提）であって、死にコードを隠すための default-off flag ではない。したがって:
  - コード既定は off でよいが、**`.env.example` に必ず全フラグを明記する**（新環境で無効のまま気づかない事故を防ぐ）
  - **実起動経路（`tools/voice_chat.py` と同じ `.env`）でも必ず検証する**。E2E ハーネスだけが強制 ON という状態を作らない（2026-07-28 に約1か月見逃した実害あり）
  - 死にコード（呼び出し元ゼロのモジュール・未使用 role）は別問題として禁止のまま
- 例外は握りつぶさず logger + 安全フォールバック。突然落とさない。UI更新は `root.after(0, fn)` 経由のみ（UI 実装時）。

## 進め方（ディシプリン）

1. まず機能を**単純実装**して流れ・レイテンシを確認 → その後深掘り。
2. 機能ごとに just-in-time プランニング、**branch を切る**。
3. テストは **2回連続成功で合格**、最終統合マージは **5回連続成功**。
4. 非自明な調査・設計・レビュー・実装・テストは **3+エージェントを別観点で並列**起動。意見が割れたら**多数決→多い方を再調査**、**全会一致で実装**。
5. **エージェントの結論は鵜呑みにしない。重要な主張は一次ソース(実ログ行・実コード行)で自分で確認**（実際にエージェントの反証が一次データで覆った事例が複数ある）。
6. 不明点・矛盾・仕様未確定は実装前に必ず報告/質問。推測でTech Stack/構成を決めない。
7. **効果の判定は判定単位の指標で行う**。自発発話の「件数」は1セッション3-4件しか出ないためノイズに埋もれる（条件付きPoisson検定で p=1.00 の実例あり）。判定回数（数十〜百）を母数にする指標を使う。

## テスト（docs/PIPELINE_DESIGN.md §7 が正）

決定論(API不要)を基本に T1 レイテンシ配線 / T2 surprise death-detection / T3 順序・世代 / T4 barge-in / T5 provider代用 / T6 過去参照なし / T7 パイプライン破綻なし。
実機は `tools/search_e2e_test.py`（VOICEVOX 合成音声→実STT→本番 VoiceLoop の通し・**実 API 課金あり**）。`REAL_STATE=1` で**実起動と同じ設定＋本番記憶のコピー**で回せる。使い方は HANDOFF。

## 既知の落とし穴

- **`.env` に機能フラグを書き忘れると実起動で機能が全滅する**（最重要・上記「制約」参照）。
- `Config.validate()` は**呼び出し元がゼロ**＝起動前チェックは事実上存在しない。
- Git Bash の grep は絵文字入りログで壊れる → **Python で解析する**。
- E2E は実 API 課金 + 実画面キャプチャ + メモ帳の自動開閉。VLM 有効時は**実際の画面が Gemini に送られる**。
- `*.jsonl`（会話履歴・RAG・タスク）は gitignore。実起動で回すと本番の記憶が育つ。

## Update Policy

作業後、恒久情報(目的/Stack/構成/設計判断/制約/同じミス防止の指示/既存記述の誤り)が増えた時のみ本書を更新。
**実装状況・テスト件数・未実装リスト・既知問題は本書に書かない**（`docs/HANDOFF.md` に集約する。二重管理で腐った実績があるため）。一時的作業ログ・今回限りの判断・未確定情報は入れない。不明点は推測せず確認。
