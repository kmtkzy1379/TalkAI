# HANDOFF — Eve v2 引き継ぎ（実装状況・手順・既知問題の単一情報源）

最終更新: **2026-07-29** / ブランチ `feat/j2-search`（HEAD `7f90c71`・main は 27コミット遅れ）

> **新セッションはまずこれを読む。食い違いはコードが正。**
> 恒久的な設計原則 = `CLAUDE.md` / 設計契約 = `docs/PIPELINE_DESIGN.md`・`docs/COMPONENT_LOGIC.md` / 原契約 = `docs/KIKAKUSHO.md`。
> 本書は 3エージェント並列監査 + 一次ソース（実コード行・実行結果）確認で作成。

---

## 1. 起動方法

UI もランチャも無い。**実起動パスは `tools/voice_chat.py` の1本のみ**。

```powershell
cd C:\Users\tester\Desktop\eve-v2
$env:PYTHONIOENCODING="utf-8"
& C:\Users\tester\Desktop\portfolio8-VLM-AI\venv\Scripts\python.exe tools\voice_chat.py
# Ctrl+C で終了（全サイドカーを drain して停止）
```

### 前提
- **venv は v1 のものを流用**: `C:\Users\tester\Desktop\portfolio8-VLM-AI\venv`（v2 に venv は無い）
- **VOICEVOX 起動必須**（`http://127.0.0.1:50021`）。未起動でも落ちないが声が出ない。確認 `curl -s http://127.0.0.1:50021/version`
- **マイク + イヤホン**（AEC は不採用＝スピーカーだと自分の声を拾う）
- **`.env` が必要**（`.env.example` をコピーして API キーを入れる）
- 初回のみネットワーク: Silero VAD（torch.hub キャッシュ）/ Ruri 埋め込み（HuggingFace DL）

### 起動ログで機能の有効/無効を必ず確認する
```
Call-Function 稼働（read-only 能力）        ← CALLFUNCTION_ENABLED=1
タスク管理 稼働（予約タスク）                ← TASK_ENABLED=1
Web検索 稼働（search_web・TaskAgent 専用）   ← SEARCH_ENABLED=1 かつ ddgs 導入済
画面認識(VLM) 稼働                          ← VLM_ENABLED=1
VoiceLoop 稼働。話しかけてください。
```
**この行が出ていない機能は動いていない。** `Config` は `VoiceLoop()` 構築時に1回だけ読むので `.env` 変更は再起動が必要。
**search は TASK_ENABLED の内側**でしか生成されない（`SEARCH_ENABLED=1` 単独では無効）。

---

## 2. テスト方法

### Tier-1 決定論テスト（API不要・ネットワーク不要・全18ファイルで約14秒）

runner は無い（**pytest 未導入**＝`pytest tests/` は動かない）。各ファイルが独立スクリプト。

```powershell
$env:PYTHONIOENCODING="utf-8"
& C:\Users\tester\Desktop\portfolio8-VLM-AI\venv\Scripts\python.exe tests\test_f5_speech.py
```

全件一括（合計を出す）:
```powershell
$env:PYTHONIOENCODING="utf-8"; $tot=0; $fail=0
Get-ChildItem tests\test_*.py | ForEach-Object {
  $o = & C:\Users\tester\Desktop\portfolio8-VLM-AI\venv\Scripts\python.exe $_.FullName 2>&1 |
       Select-String -Pattern "合計" | Select-Object -Last 1
  if ($o -match "PASS (\d+) / FAIL (\d+)") { $tot+=[int]$Matches[1]; $fail+=[int]$Matches[2] }
}
"PASS $tot / FAIL $fail"
```

**実測 2026-07-29: 18ファイル 378件 PASS / FAIL 0**（2回連続一致・flaky なし）

| ファイル | 件数 | ファイル | 件数 |
|---|---|---|---|
| test_callfunction_phase1.py | 45 | test_f4_feedback.py | 25 |
| test_cancel_resolver_phase1.py | 11 | test_f5_speech.py | 61 |
| test_delivery_checker.py | 10 | test_f6_vlm.py | 44 |
| test_f0_foundation.py | 21 | test_search_phase1.py | 23 |
| test_f1_pipeline.py | 16 | test_stt_filter.py | 10 |
| test_f2_5_integration.py | 4 | test_task_agent_phase1.py | 13 |
| test_f2_5_robustness.py | 5 | test_task_phase1.py | 14 |
| test_f2_response.py | 18 | test_voiceloop_wiring.py | 27 |
| test_f3_5_rag.py | 21 | | |
| test_f3_memory.py | 10 | **合計** | **378** |

規律: 機能ごと **2回連続 PASS で合格** / 最終統合マージは **5回連続**。
`test_f6_vlm.py` は例外注入テストで traceback をログ出力するが FAIL ではない。

### E2E 実機テスト `tools/search_e2e_test.py`

本番 `VoiceLoop` をそのまま起動し、**マイクだけスタブ化**して VOICEVOX 合成音声→実 STT→刺激投入で実機と同じイベント順（発話開始=barge-in → 発話終了 → STT → 投入）を再現する。
**実 API 課金あり / 実画面をキャプチャ / メモ帳を自動で開閉する。**

```powershell
# 実起動と同じ設定・本番記憶のコピーで「通し総合16項目」だけ回す（約18分）
$env:PYTHONIOENCODING="utf-8"; $env:REAL_STATE="1"
$env:E2E_ART="C:\Users\tester\Desktop\eve-v2\e2e_logs\full_$(Get-Date -Format yyyyMMdd)"
$env:SKIP="SSRC,SVLM,SIDLE,SAUTO,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11"
& C:\Users\tester\Desktop\portfolio8-VLM-AI\venv\Scripts\python.exe tools\search_e2e_test.py
```

**`REAL_STATE=1` が重要**: `.env` のフラグ/モデルをそのまま使い、本番の `conversation_history.jsonl` / `rag_memory.jsonl` / `tasks.jsonl` を artifacts に**コピーして**使う（本物は汚れない）。付けないとフラグを強制 ON し記憶が空になるので**実起動の挙動は測れない**（この差が D6 の事故を1か月見逃した原因）。

| キー | シナリオ |
|---|---|
| `SFULL` | 通し総合16項目（通常会話 / 記憶・画面・会話起点の自発発話 / タスク / 検索 / 複数 / 割り込み / キャンセル / タスク中の雑談 / 不可能依頼 / ハルシネーション / 自律検索 / 自律タスク） |
| `SSRC` | 自発発話の由来切り分け（画面 / 直近会話 / 記憶） |
| `SVLM` | 画面認識の全場面（静止の据え置き回答・変化中・捏造なし・発話中割り込み） |
| `SIDLE` | 3分放置×3の自律発話頻度 |
| `SAUTO` | 4ラウンド放置の自律発話計測 |
| `S1`〜`S11` | 検索/タスクの混線・別タスク・別検索・キャンセル・画面移動・深掘り・RAG想起・割り込み・連投・長時間放置・反復ストレス |

環境変数: `REAL_STATE` / `E2E_ART`（出力先）/ `SKIP`（キーをカンマ区切り）/ `REAL_AUDIO=1`（実スピーカー再生）/ 放置秒の調整 `FULL_IDLE_SEC` `IDLE_NEGLECT_SEC` `SRC_PHASE_SEC` `AUTO_ROUNDS` `AUTO_NEGLECT_SEC` `S10_NEGLECT_SEC`。

出力（`E2E_ART` 直下）: `timeline.jsonl`（全イベント）/ `terminal.log` / `summary.json`（handles・latencies・decisions・suppressions・seed_calls・cap_calls・search_calls・tasks_final）/ 記憶の隔離コピー3種。
**`e2e_logs/` は gitignore**（記憶のコピーを含む）＝残したい結論は docs に転記する。

### その他のツール（`tools/` 33本）
- 軽いスモーク: `search_smoke.py` / `task_smoke.py` / `callfunction_smoke.py` / `autonomous_probe.py`
- VLM: `f6_realtest.py`（通し）/ `f6_precheck.py`（前提チェック）/ `f6_latency_ab.py`
- RAG: `rag_diag.py`（**API不要**・実 rag_memory + 実 Ruri）/ `rag_experiment.py`
- マイク切り分け: `mic_check.py`（mic→VAD→STT を文字表示）
- `tools/_*.py` はローカル診断用（gitignore）

---

## 3. 実装状況

### 実装済み
| 機能 | 主なモジュール |
|---|---|
| F0 基盤（ModelRegistry / Config / clock / ContextAssembler） | `eve/model_registry.py` `eve/config.py` `eve/clock.py` `eve/context_assembler.py` |
| F1 2キュー骨格 | `eve/pipeline/` |
| F2 応答の背骨（stream→文分割→TTS→順次再生） | `eve/response/` |
| F2.5 VoiceLoop（mic→VAD→STT→LLM→TTS） | `eve/voice_loop.py` `eve/audio_input.py` `eve/stt/` |
| F3 短期記憶 / F3.5 長期記憶RAG | `eve/memory/` |
| F4 FeedbackLLM（内省→RAG書込→surprise） | `eve/feedback/` |
| F5 発話判定（沈黙→自発発話） | `eve/speech/` |
| F6 画面認識VLM（snapshot・**main マージ済**） | `eve/vlm/` |
| **J-0 Call-Function（read-only 能力層）** | `eve/capability/` `eve/response/function_dispatcher.py` |
| **J-1 タスク管理（TaskAgent / CancelResolver / ReconcileTimer）** | `eve/task/` |
| **J-2 Web検索（ddgs スニペット + 深掘り調査）** | `eve/search/`（`client.py` `capability.py` `deep.py`） |
| **報告の配達確認 / barge-in 再配達** | `eve/response/delivery_checker.py` / `voice_loop.py` |
| **自律発話の改修群（J-2 ①〜③）** | `eve/speech/` `eve/memory/long_term.py` `eve/context_assembler.py` |

自律発話の改修群（2026-07-26〜28・すべてコードゲート）:
①既出の用件ブロック / ①-b 発話見本の削除 / ②-1,②-2 同内容抑制（bigram 0.25 + 埋め込み 0.87・600秒窓）/ ②-3 記憶の自己参照除外 / ②-4 話題の丸投げ抑制 / ②-5 種の固着解消（クエリ不変時スキップ + メタ要約除外）/ ②-6 時制の混同検出 + 画面不在マーカ / ③ 沈黙バイアスの撤廃。**各ゲートの詳細は `docs/COMPONENT_LOGIC.md` の自律発話節**。

### 未実装
UI（Tkinter・**tkinter 参照は0件**）/ 配線層PORT（vts / run / launcher / app）/ Live2D 連動 / YouTube 配信モード / J-3 画面操作（window_op・launch_app・download 等）/ **(b) 文脈不整合の自己懐疑** / VLM の Gemini Live ストリーム mode。

### 作らないと確定したもの
- **`SurpriseBus` クラス**: `PredictionState` が2生産者を most-recent-wins で集約済み。別クラスを作る合理性が無い
- **STT partial 投機 / AEC（自己エコー除去）**: 探索の結果いずれも不採用（イヤホン前提）
- **VLM ×3 self-consistency**: 廃止（単発・複数フレーム VLM に確定）

---

## 4. 未対応の問題

### 実機で確認済み（2026-07-29 通しテスト・生ログは `e2e_logs/full_20260729/`）
| ID | 内容 |
|---|---|
| **D1** | **起動直後に前セッションの未応答依頼を勝手に実行する。** 復元履歴の最終ターンが5時間前の「PCの状態を教えて」だったため、挨拶しただけで `pc_status` を実行し「さっきの」と呼んだ。②-6 の時制ゲートは**自発発話の下書きにしか掛かっておらず**、ユーザ発話への応答経路は素通り |
| **D2** | **タスクキャンセルの競合。** 検索完了が取消要求に先行するとタスクが terminal になり取消対象が消える → 結果が配達された上に「止められる予約はなかった」と矛盾した2連発 |
| **D3** | **能力の無い依頼を承諾する。** メール送信を「いいよ」と受けた。危険系（ファイル削除）は拒否できるので「危険だから断る」は効き「**能力が無いから断る**」が無い |
| **D4** | 自己観察ループ（画面に自分のログが映るとそれを画面内容として実況する・軽微） |
| **D5** | レイテンシ 中央値 2.57s / **平均 3.01s / 最大 5.62s / 3秒超 44%**。**2026-07-29 裁定: gpt-5.5 の品質を取り ≤5s 運用を許容**（理想の ≤3s は下ろさない）。より賢く速いモデルが出たら `RESPONSE_MODEL` を差し替えて再計測する |
| **D6** | （**解決済 2026-07-29**）`.env` に機能フラグが無く実起動でタスク/検索/Call-Function が全滅していた。E2E ハーネスだけが強制 ON だったため約1か月気づかなかった。→ `.env` と `.env.example` にフラグを明記 |

### 設計上の既知（継続・実害小）
- `StreamFn` 型が2箇所で不整合（`model_registry.py` と `response/orchestrator.py`）。VoiceLoop の adapter が吸収
- `ResponseOrchestrator.handle()` の `await audio.join()` に timeout が無い（後発サイドカー3本は `wait_for` 済＝非対称）
- C5: barge-in 時の spoken 記録ズレ（設計上許容）
- `drain_user_texts` はロック無し（「await を挟まない」前提の意図的な landmine）
- `Config.validate()` は**呼び出し元がゼロ**＝起動前チェックは存在しない

### 裁定済み（2026-07-29）
- **機能フラグ**: コード既定は off のまま、**`.env` で明示する運用**。将来 UI から ON/OFF する設定項目という位置づけ。`.env.example` に全フラグを明記して事故の再発を防ぐ
- **VLM 画質**: **1024px / JPEG品質70 / 3枚** が正（通しE2Eで OCR 良好だった実測に基づく）。コード既定も `.env` に合わせた。過去に却下されたのは **Q60** の組み合わせで、悪化の主因は解像度でなく JPEG 品質だったと整理
- **レイテンシ**: gpt-5.5 の品質を取り **≤5s 運用を許容**（理想 ≤3s は維持）

### 未決事項（ユーザ裁定待ち）
1. **`pending_obligation`**: `should_speak` の唯一の hard ゲートだが本番から渡していない（テスト専用の死んだ引数）。実配線するか
2. **未使用 role**: `vlm_merge`（廃止された×3案の残骸）と `youtube`（未実装）を残すか
3. **main マージ計画**: main が27コミット遅れ。どの単位で5回連続テストを回してマージするか

---

## 5. 設定の要点

- **機能フラグは `.env` で明示する**（コード既定は全て off）。将来 UI から ON/OFF する設定項目という位置づけ。`.env.example` に全フラグを明記済み
- 実運用モデル: response=`gpt-5.5` / decide=`gpt-5.4-mini` / feedback=`gpt-5.4` / vlm_leaf=`gemini-2.5-flash`。`task` `search_summarize` `delivery_check` はコード既定のまま
- Web検索は `SEARCH_BACKEND=auto`（DNS 遮断回線では `duckduckgo` 固定へ。復旧手順は `docs/DNS_BACKUP_2026-07-17.md`）
- 主要な閾値の実測校正値は `eve/config.py` のコメントに日付つきで残してある（同内容抑制 0.25/0.87、記憶の自己参照除外 600秒、画面据え置き 600秒 等）

## 6. 落とし穴

1. `$env:PYTHONIOENCODING="utf-8"` を忘れると日本語＋絵文字で落ちる
2. Git Bash の grep は絵文字入りログで壊れる → **Python で解析する**
3. `.env` 変更は再起動が必要（`Config` は構築時に1回だけ読む）。E2E ハーネスは `eve.config` を import する**前**に環境変数を書き換えている＝この順序を崩すと効かない
4. E2E は課金・実画面・メモ帳の自動操作を伴う。VLM 有効時は**実際の画面が Gemini に送られる**
5. 実起動で回すと本番の記憶（`*.jsonl`）が育つ＝D1 の温床
6. `.env` と `*.jsonl` は gitignore＝設定の修正はコミットに残らない
7. **自発発話の効果測定は「件数」で判断しない**。1セッション3-4件しか出ずノイズに埋もれる（条件付き Poisson 検定で p=1.00 の実例あり）。判定回数（数十〜百）を母数にした指標を使う
