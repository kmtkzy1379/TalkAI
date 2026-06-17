# Salvage Manifest — Eve v2 リビルド

> 目的: v1 を新企画書に従い v2 として作り直すにあたり、既存コードを **1ファイル単位**で
> 「作り直す / 移植する / 参照する / 廃棄する」に分類する。
> **原則: デフォルト = REBUILD。再利用(PORT)は「客観的・合理的にそれが最適」な場合のみ。**
> 作成: 4並列エージェント調査 + 一次コード再確認（PORT判定と反証は手動検証済み）。

## verdict 定義
- **PORT** … コードが良く spec 整合・battle-tested・低リスク。**理解した上で移植**（blind copy 禁止）。
- **REFERENCE** … コピーしない。挙動・教訓・テストシナリオ・実装パターンとして読む。
- **REBUILD** … ロジックが古い/間違い/癒着。新企画書を契約に白紙から。
- **DROP** … v2 に不要。廃棄。

## v2 中核原理（リビルドの組織原理）
予測誤差(surprise)を**非同期の装飾数値ではなくリアルタイム一級信号**にし、
(a) 自発発話の要否/内容 と (b) 文脈不整合の自己懐疑 の両方をゲートする。
（v1 では FEP が feedback ループ内で完結し応答 behavior を駆動していなかった＝症状の根。）

---

## ① PORT（移植・理解の上で・客観根拠あり）

| file | LOC | 役割 | PORT 根拠（行番号） |
|---|---|---|---|
| `vts.py` | 293 | 別プロセス VTS WebSocket 制御 | 応答AI側へ依存ゼロ（自己完結を確認）。認証永続化(80-92)/再接続backoff/ping/瞬きFSM(163-205)。spec が「別プロセス vts.py」を明示 |
| `modules/player.py` | 131 | 音声キュー再生 + 割り込み | asyncio.Queue 再生 + 多重 interrupt レース対策(79-122)。spec「できた音声から順に流す/応答中は溜める」の中核 |
| `modules/tts.py` | 44 | VOICEVOX 2段API(audio_query→synthesis) | 薄く正しい。キュー設計は呼び出し側で新造する前提の「合成部品」 |
| `launcher.py` | 96 | VTS/VOICEVOX 起動 + HTTP 待機 | `wait_for_voicevox` 指数backoff。spec ランチャー役割と一致 |
| `run.py` | 139 | 統合ランチャー（subprocess + vts.py 別プロセス） | atexit/signal/CREATE_NEW_PROCESS_GROUP が堅実。固定 `time.sleep(2)`(111) のみ要改善 |
| `modules/feedback_llm.py` | 234 | provider 抽象 LLM クライアント | 3段 fallback(107-133)/cache_control(401)/provider 自動判定(27)。spec 中立な汎用インフラ |
| `vlm/capture/screen.py` | 94 | mss スクショ + downscale + pHash | v2 もスクショ取得は必須。薄く依存少 |
| `vlm/capture/change_detector.py` | 114 | pHash→SSIM の画面変化ゲート | **唯一強く残す低レイヤCV**。静止画面で VLM を叩かない=コスト/レイテンシ合理的(48-95)。4段→「変化/無変化+大変化」2値に簡素化推奨 |
| `vlm/common/config.py` | 46 | 汎用 YAML ローダー(deep merge/dotpath) | CV 非依存。v2 設定機構にそのまま |
| `modules/vision_analyzer.py`* | 108 | 単発画像→日本語解説(Groq) | *borderline: パターンは可搬だが spec は Gemini 指定。パラメータ外出し+provider 差替前提なら PORT、でなければ REFERENCE |

## ② REFERENCE（コピーせず読む — 教訓/パターン/テスト）

| file | LOC | 読む価値 |
|---|---|---|
| **`tools/` Tier-1/2/3 ハーネス** | — | **最重要。35個の fix が残した実機事故シナリオ = v2 回帰仕様。コードでなく仕様として移植** |
| `modules/conversation_cache.py` | 268 | write-queue/atomic add_turn/silence 算出。100件 deque は spec 一致だが「…×X」圧縮ログ未実装 |
| `modules/task_schema.py` | 215 | jsonable/from_jsonable 防御パターン（状態語彙は v1 設計結合で作り直し） |
| `modules/hierarchical_surprise.py` | 127 | 純関数の surprise 集計。ただし係数恣意的(100-101)・駆動先ゼロ |
| `modules/vlm_bridge.py` | 494 | cooldown(466-486)/dedup(342-371)/watch の統合層ロジック。内部 pipeline 結合は作り直し |
| `modules/vision_buffer.py` | 182 | スレッドセーフ リングバッファ。VisionMeta は VLM 刷新で再設計 |
| `vlm/narration/llm_client.py` | 154 | litellm マルチ provider + fallback + rate-limit 骨格 |
| `vlm/narration/prompt_builder.py` | 203 | 画像 base64→OpenAI message 組立(171-190)。SYSTEM_PROMPT 内容は廃 |
| `vlm/narration/context_manager.py` | 33 | ナレーション sliding window の発想（実装は trivial） |
| `vlm/common/datatypes.py` | 229 | `CapturedFrame`/`NarrationResult` のみ流用、CV 型は廃 |
| `vlm/common/validators.py` | 25 | 画像 shape/dtype 検証 |
| `app.py` | 88 | 起動配線パターン（WindowsSelectorEventLoopPolicy/別スレ asyncio） |

## ③ REBUILD（白紙から・新企画書を契約に）

| file | LOC | REBUILD 理由（行番号） |
|---|---|---|
| `modules/llm.py` | 1846 | ペルソナが `__init__` ハードコード(267-434)でロジック癒着。`_build_system_prompt` 15ブロック巨大関数(745-1155)。沈黙判定が prompt 内 A/B/C/D(724-742)。Bug-F shadow 事故痕(810-812) |
| `modules/rag.py` | 439 | **ランキングは優秀**だがチャンク定義が episode要約中心で spec『フィードバック1+応答1』不一致、`maxlen=3000`(47) が spec『300』の10倍、無言時 pure-random。ランキング式は参考に |
| `modules/task_manager.py` | 1583 | spec『claw-code 模倣』とは別物（短期約束 期限FSM + fact store）。Plan/Task DAG はほぼデッド。Fix-3/6/7/RC1/RC2 の地層化、完了所有権が二転三転(1488-1560) |
| `modules/feedback.py` | 2275 | FEP 出力が応答を駆動せず loop 内完結(611→1331/2063/628)。責務癒着(audit/TaskManager/RAG/bias 同居)。Phase 地層 |
| `modules/feedback_prompts.py` | 602 | 肥大プロンプト(11ブロック)+regex 抽出前提(85,154)。「FEP 用語禁止」等の守れない自己注意 |
| `modules/precision.py` | 285 | 自己申告 affect→learning rate 結合(119,135-146)=ハルシネ温床。`pe_factor`(243)/動的重み(250-267) は死にコード(default OFF) |
| `modes/base_mode.py` | 864 | `_run_response_pipeline`(394-712) が 318行に RAG/TTS/dedup/fact/audit/nudge を癒着。`_postfulfill_mode` ゲートは config 既定で死蔵 |
| `modes/talk_mode.py` | 616 | ターン制 STT(355)。dispatch が文字列 prefix 駆動(402-429)。`last_user_event_ts` は write-only dead field(25,57,608) |
| `modes/game_mode.py` | 390 | キーフック+モデルハードコード(151,259)、token streaming/差替UI なし |
| `modes/youtube_mode.py` | 237 | 取得が「最新1件」で spec『上から1つずつ』と逆。モデル差替/Opus 化 UI なし |
| `prompts/talk_prompt.py` | 207 | ペルソナ(4-102)+ロジック規則(116-202) 完全癒着。spec『ペルソナ外す』。code gate と二重管理 |
| `prompts/game_prompt.py` | 51 | 同上 |
| `prompts/youtube_prompt.py` | 51 | 同上 |
| `ui/main_window.py` | 746 | ユーザー明示「完全作り直し」。設定ロック/「全設定完了でのみ Start」安全ゲート未実装(583,709)。`ui/settings_dialog.py`(256, 担当外) も同時に設計 |
| `config.py` | 234 | .env 思想は流用だが **GEMINI_API_KEY 欠落**(13-16)、`AI2_MODEL` 既定が旧ID(32)、FB_/PE_ 死にパラメータ過多 |
| `modules/stt.py` | 43 | 一括 Whisper。spec『増分 50-300ms ストリーミング』と真逆=聞き間違いの根本 |
| `modules/audio_input.py` | 148 | VAD ターン切り出し(117-141)=増分粒度に非整合。VAD/デバイス層(14-71)は REFERENCE 的に拾う |
| `vlm/main.py` | 868 | 9段 orchestrator。v2『capture→VLM×3→Sonnet統合』は別トポロジ。3スレッド/queue/poison-pill 骨格(100-108)は参考 |

## ④ DROP（v2 不要・廃棄 — 重量級 CV 低レイヤ）

| module | LOC | DROP 理由 |
|---|---|---|
| `vlm/detection/yolo_detector.py` + `base.py` | 193 | **ハルシネーション主因**。VLM が物体名を直接読む |
| `vlm/tracking/` (id_authority/working_memory/track_store) | 541 | track ID は VLM 直叩きに存在しない概念 |
| `vlm/analysis/` (optical_flow/pose/expression/motion/per_id_analyzer) | 674 | px動き/姿勢/表情は VLM×3→Sonnet 統合が上位代替。crop/bbox 前提 |
| `vlm/aggregation/` (delta_encoder/scene_graph/feature_store/token_budget) | 670 | 構造化テキスト中間層。画像入力では不要。token_budget は既に死にコード |
| `vlm/capture/predictive_coder.py` + `saliency.py` | 447 | 変化領域座標/顕著性は meta 止まりで nudge 未使用。VLM が代替 |
| `vlm/common/device.py` | 83 | ローカル GPU 推論が消える=デバイス判定不要 |
| `modes/__init__.py` / `prompts/__init__.py` | 7 | 自明な re-export |

---

## 集計（おおよその LOC）
- **REBUILD ≈ 11,500** … 頭脳部のほぼ全て（応答LLM/RAG/task/feedback/FEP/modes/prompts/UI/STT/VLM orchestrator）
- **DROP ≈ 2,600** … 重量級 CV 低レイヤ（YOLO/tracking/analysis/aggregation/saliency）
- **REFERENCE ≈ 2,000** … パターン・教訓・**テスト corpus（最重要）**
- **PORT ≈ 1,300** … クリーンな配線層のみ（vts/player/tts/launcher/run/feedback_llm/screen/change_detector/config-loader）

> **含意**: 価値創出コードの約85%が REBUILD/DROP。クリーン移植できるのは ~1,300行の配線層だけ。
> = 「全書き直し」が second-system trap ではなく合理的、を定量的に裏付け。重い CV(2.6k) は丸ごと消える。

## 必ず引き継ぐもの（コードでなく知識）
1. **テスト corpus**（`tools/` Tier-1/2/3 の実機事故シナリオ）を v2 回帰仕様として最初に移植 — 同じ穴に落ちない唯一の保険。
2. **アーキ非依存の教訓**: 沈黙/タイミング規律はプロンプトで守れない → code gate で決定論的に強制。
3. **CLAUDE.md の修正事項**: `vlm_orig/` は存在しない / GEMINI_API_KEY 追加 / AI2_MODEL 既定更新。
