# HANDOFF — Eve v2 引き継ぎ（実装状況・手順・既知問題の単一情報源）

最終更新: **2026-07-30** / ブランチ **`main`**（D1/D2/D3 修正まで統合済み・`feat/j2-search` は main に含まれる）

> **新セッションはまずこれを読む。食い違いはコードが正。**
> 恒久的な設計原則 = `CLAUDE.md` / 設計契約 = `docs/PIPELINE_DESIGN.md`・`docs/COMPONENT_LOGIC.md` / 原契約 = `docs/KIKAKUSHO.md`。
> 本書は 3エージェント並列監査 + 一次ソース（実コード行・実行結果）確認で作成。

---

## 1. 起動方法

UI もランチャも無い。**実起動パスは `tools/voice_chat.py` の1本のみ**。

```powershell
cd <リポジトリのルート>
$env:PYTHONIOENCODING="utf-8"
& .\.venv\Scripts\python.exe tools\voice_chat.py
# Ctrl+C で終了（全サイドカーを drain して停止）
```

### 前提
- **セットアップ**: `python -m venv .venv` → `.\.venv\Scripts\pip install -r requirements.txt`
  （開発時は前作の venv を流用していたが、`requirements.txt` は実 import を反映済みで単体構築できる）
- **VOICEVOX 起動必須**（`http://127.0.0.1:50021`）。未起動でも落ちないが声が出ない。確認 `curl -s http://127.0.0.1:50021/version`
- **マイク + イヤホン**（AEC は不採用＝スピーカーだと自分の声を拾う）
- **`.env` が必要**（`.env.example` をコピーして API キーを入れる）
- 初回のみネットワーク: Silero VAD（torch.hub キャッシュ）/ Ruri 埋め込み（HuggingFace DL）

### 起動ログで機能の有効/無効を必ず確認する
```
Call-Function 稼働（read-only 能力）        ← CALLFUNCTION_ENABLED=1
タスク管理 稼働（予約タスク）                ← TASK_ENABLED=1
Web検索 稼働（search_web・TaskAgent 専用）   ← SEARCH_ENABLED=1 かつ ddgs 導入済
作業フォルダ 稼働（C:\eve_ws\workspace・…）  ← WS_ENABLED=1
プログラム実行 稼働（run_program・…）        ← WS_EXEC_ENABLED=1 かつ setup_sandbox.py apply 済
画面表示 稼働（show_on_screen・別ウィンドウ） ← WS_VIEWER_ENABLED=1
画面認識(VLM) 稼働                          ← VLM_ENABLED=1
VoiceLoop 稼働。話しかけてください。
```
**この行が出ていない機能は動いていない。** `Config` は `VoiceLoop()` 構築時に1回だけ読むので `.env` 変更は再起動が必要。
**search / 作業フォルダは TASK_ENABLED の内側**でしか生成されない（単独で 1 にしても無効）。

### J-3 プログラム実行の事前セットアップ（PC ごとに1回・管理者）
```powershell
# 管理者 PowerShell（**自分のアカウントのまま昇格する**。別の管理者だと DPAPI で復号できない）
& <python> setup_sandbox.py check    # 非管理者OK・診断のみ
& <python> setup_sandbox.py apply    # 専用アカウント eve_sandbox / ACL / seclogon / FW / venv
& <python> setup_sandbox.py probe    # 通常 PowerShell で。全チェーンを実際に走らせて検証
```
`apply` は冪等。`probe` が 5/5 にならなければ `run_program` は動かない（起動は続く＝fail-soft）。

---

## 2. テスト方法

### Tier-1 決定論テスト（API不要・ネットワーク不要・全18ファイルで約11秒）

runner は無い（**pytest 未導入**＝`pytest tests/` は動かない）。各ファイルが独立スクリプト。

```powershell
$env:PYTHONIOENCODING="utf-8"
& .\.venv\Scripts\python.exe tests\test_f5_speech.py
```

全件一括（合計を出す）:
```powershell
$env:PYTHONIOENCODING="utf-8"; $tot=0; $fail=0
Get-ChildItem tests\test_*.py | ForEach-Object {
  $o = & .\.venv\Scripts\python.exe $_.FullName 2>&1 |
       Select-String -Pattern "合計" | Select-Object -Last 1
  if ($o -match "PASS (\d+) / FAIL (\d+)") { $tot+=[int]$Matches[1]; $fail+=[int]$Matches[2] }
}
"PASS $tot / FAIL $fail"
```

**実測 2026-07-29: 18ファイル 434件 PASS / FAIL 0**（2回連続一致・flaky なし）

| ファイル | 件数 | ファイル | 件数 |
|---|---|---|---|
| test_callfunction_phase1.py | 45 | test_f4_feedback.py | 25 |
| test_cancel_resolver_phase1.py | 19 | test_f5_speech.py | 61 |
| test_delivery_checker.py | 10 | test_f6_vlm.py | 44 |
| test_f0_foundation.py | 53 | test_search_phase1.py | 23 |
| test_f1_pipeline.py | 26 | test_stt_filter.py | 10 |
| test_f2_5_integration.py | 4 | test_task_agent_phase1.py | 13 |
| test_f2_5_robustness.py | 5 | test_task_phase1.py | 14 |
| test_f2_response.py | 18 | test_voiceloop_wiring.py | 29 |
| test_f3_5_rag.py | 21 | | |
| test_f3_memory.py | 14 | **合計** | **434** |

規律: 機能ごと **2回連続 PASS で合格** / 最終統合マージは **5回連続**。
`test_f6_vlm.py` は例外注入テストで traceback をログ出力するが FAIL ではない。

### E2E 実機テスト `tools/search_e2e_test.py`

本番 `VoiceLoop` をそのまま起動し、**マイクだけスタブ化**して VOICEVOX 合成音声→実 STT→刺激投入で実機と同じイベント順（発話開始=barge-in → 発話終了 → STT → 投入）を再現する。
**実 API 課金あり / 実画面をキャプチャ / メモ帳を自動で開閉する。**

```powershell
# 実起動と同じ設定・本番記憶のコピーで「通し総合16項目」だけ回す（約18分）
$env:PYTHONIOENCODING="utf-8"; $env:REAL_STATE="1"
$env:E2E_ART=".\e2e_logs\full_$(Get-Date -Format yyyyMMdd)"
$env:SKIP="SRB,SD1,SD2,SD3,SSRC,SVLM,SIDLE,SAUTO,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11"
& .\.venv\Scripts\python.exe tools\search_e2e_test.py
```

**`REAL_STATE=1` が重要**: `.env` のフラグ/モデルをそのまま使い、本番の `conversation_history.jsonl` / `rag_memory.jsonl` / `tasks.jsonl` を artifacts に**コピーして**使う（本物は汚れない）。付けないとフラグを強制 ON し記憶が空になるので**実起動の挙動は測れない**（この差が D6 の事故を1か月見逃した原因）。

全20シナリオ。`SKIP` は `名前の _ より前` の完全一致で判定する（例 `SD1`）。

| キー | シナリオ |
|---|---|
| `SRB` | **再起動時の挙動（会話 / 自律発話 / タスク）**。`D1_SEED=1` で「前セッションが未応答依頼を残して終了」を作る。`REBOOT_TASK=1` で Pending タスクも残す |
| `SD1` | **前セッション未応答依頼（D1 回帰）**。`D1_SEED=1` で履歴/RAG を N時間前で打ち切り末尾に未応答依頼を仕込む。`D1_NOTE_OFF=1` で修正を無効化した対照実験ができる（約1分） |
| `SD3` | **能力境界（D3 回帰）**。不能依頼を承諾しないこと＋可能依頼/雑談を過剰に断らないことを実起動経路で確認（配線確認が主目的。効果量は `tools/d3_probe.py`）（約2分） |
| `SD2` | **完了後キャンセル（D2 回帰）**。検索完了が取消要求を追い越す窓を作り、未配達の報告が抑止されること・矛盾2連発が出ないことを確認（約2分・`D2_DELAY_SEC` で発話タイミング調整）。S4 とは別経路 |
| `SFULL` | 通し総合16項目（通常会話 / 記憶・画面・会話起点の自発発話 / タスク / 検索 / 複数 / 割り込み / キャンセル / タスク中の雑談 / 不可能依頼 / ハルシネーション / 自律検索 / 自律タスク） |
| `SSRC` | 自発発話の由来切り分け（画面 / 直近会話 / 記憶） |
| `SVLM` | 画面認識の全場面（静止の据え置き回答・変化中・捏造なし・発話中割り込み） |
| `SIDLE` | 3分放置×3の自律発話頻度 |
| `SAUTO` | 4ラウンド放置の自律発話計測 |
| `S1`〜`S11` | 検索/タスクの混線・別タスク・別検索・キャンセル・画面移動・深掘り・RAG想起・割り込み・連投・長時間放置・反復ストレス |

環境変数: `REAL_STATE` / `E2E_ART`（出力先）/ `SKIP`（キーをカンマ区切り）/ `REAL_AUDIO=1`（実スピーカー再生）/ 放置秒の調整 `FULL_IDLE_SEC` `IDLE_NEGLECT_SEC` `SRC_PHASE_SEC` `AUTO_ROUNDS` `AUTO_NEGLECT_SEC` `S10_NEGLECT_SEC`。

出力（`E2E_ART` 直下）: `timeline.jsonl`（全イベント）/ `terminal.log` / `summary.json`（handles・latencies・decisions・suppressions・seed_calls・cap_calls・search_calls・tasks_final）/ 記憶の隔離コピー3種。
**`e2e_logs/` は gitignore**（記憶のコピーを含む）＝残したい結論は docs に転記する。

### その他のツール（`tools/` 追跡30本・ディスク35本。`tools/_*.py` `tools/_*.ps1` は gitignore）
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
| **J-3 作業フォルダ（ファイル4能力・128GB 書込ゲート）** | `eve/workspace/`（`paths.py` `store.py` `capabilities.py`） |
| **J-3 サンドボックス実行（専用アカウント + Job Object）** | `eve/workspace/winexec.py` `runner.py` / `setup_sandbox.py` |
| **J-3 画面表示（別プロセス Tk ビューア）** | `eve/workspace/viewer.py` |
| **報告の配達確認 / barge-in 再配達** | `eve/response/delivery_checker.py` / `voice_loop.py` |
| **自律発話の改修群（J-2 ①〜③）** | `eve/speech/` `eve/memory/long_term.py` `eve/context_assembler.py` |

自律発話の改修群（2026-07-26〜28・すべてコードゲート）:
①既出の用件ブロック / ①-b 発話見本の削除 / ②-1,②-2 同内容抑制（bigram 0.25 + 埋め込み 0.87・600秒窓）/ ②-3 記憶の自己参照除外 / ②-4 話題の丸投げ抑制 / ②-5 種の固着解消（クエリ不変時スキップ + メタ要約除外）/ ②-6 時制の混同検出 + 画面不在マーカ / ③ 沈黙バイアスの撤廃。**各ゲートの詳細は `docs/COMPONENT_LOGIC.md` の自律発話節**。

### 未実装
UI（Tkinter・本体プロセス側）/ 配線層PORT（vts / run / launcher / app）/ Live2D 連動 / YouTube 配信モード / J-3 の残り（window_op・launch_app・download）/ **(b) 文脈不整合の自己懐疑** / VLM の Gemini Live ストリーム mode。

### J-3 の実測（2026-08-06）
- **Tier-1: 470 PASS**（J-3 前は 434）。
- **実践通しテスト `tools/ws_e2e_test.py`（実 API）: 14/14**。判定は発話ではなく**実ファイルの状態**。
  S4/S5（エラー自己修復）が **3/3 + 3/3**＝「動かす→エラー→原因を調べる→直す→動く」を
  TaskAgent が1タスク内で完遂している。`TASK_AGENT_MAX_STEPS` を 5→12 にしなければ手数上限で落ちる経路。
  S7（`.env` を読ませようとする）は**委譲すらせず拒否**し鍵は漏れない。
- `setup_sandbox.py probe` 5/5: eve_sandbox として起動 / API キー非継承 / `.env` 読取が PermissionError /
  作業フォルダ書込可 / リポジトリ書込が PermissionError。
- 実行の隔離（14/14）: 孫プロセスまで消える（`KILL_ON_JOB_CLOSE`）/ メモリ上限が効く /
  allowlist 外は spawn 前に拒否 / **実行中も loop が回る**（3.1秒で 50 tick＝マイクと barge-in が生きている）。
- 応答LLM の system 増分は **+89字**（`tools=` は `offered` フィルタのため **0字**）。

### 作らないと確定したもの
- **`SurpriseBus` クラス**: `PredictionState` が2生産者を most-recent-wins で集約済み。別クラスを作る合理性が無い
- **STT partial 投機 / AEC（自己エコー除去）**: 探索の結果いずれも不採用（イヤホン前提）
- **VLM ×3 self-consistency**: 廃止（単発・複数フレーム VLM に確定）

---

## 4. 未対応の問題

### 実機で確認済み（2026-07-29 通しテスト・生ログは `e2e_logs/full_20260729/`）
| ID | 内容 |
|---|---|
| **D1** | （**解決済 2026-07-30** ブランチ `fix/d1-session-boundary`）起動直後に前セッションの未応答依頼を勝手に実行した。**第1ラウンドのギャップ注記（文脈整形）は実測で無効**（同一文脈でユーザが別依頼をすると蒸し返し 8/8）。→ **`recent_for_injection` が復元（前セッション）分を返さない**（案2）。同条件で 0/8。切る場所は cache 層で `_load` は残す＝**B5 no-skip（前セッション末尾を RAG に取り込む）は `turns_since` 経由なので無傷**（回帰テストで固定）。併せて独立のバグを修正: 内省チャンクの timestamp が書込時刻で、catch-up 由来チャンクが `[過去の記憶/たった今]` と描画されていた（スパン末尾の iso を渡す）。②-6 時制ゲートが恒久 no-op になる問題も反転で解消。E2E `SD1` 対照実験で配線確認（対照で復元会話の注入を再現 / 修正 arm 2回連続クリーン・起動後に黙ったままの経路も新規に観測し無根拠発話なし） |
| **D2** | （**解決済 2026-07-29** ブランチ `fix/d2-cancel-tombstone`）タスクキャンセルの競合。検索完了が取消要求に先行するとタスクが terminal になり取消対象が消え、結果が配達された上に「止められる予約はなかった」と矛盾2連発。→ `StimulusQueue` の墓標(suppress)で put を関門化＋`orchestrator` に第2関門。**取消対象は「時計で直近の terminal」ではなく「キューに未配達の報告が残っている terminal」で選ぶ**（実機 D2 時点は90秒窓に terminal 3件あり、時計基準だと特定不能に落ちて直らなかった）。E2E `SD2` で3回連続再現・抑止を確認 |
| **D3** | （**解決済 2026-07-30** ブランチ `fix/d3-capability-boundary`）能力の無い依頼（メール送信）を承諾していた。→ `# 実際に実行できる手段` ブロックを **registry 導出**（`CapabilityRegistry.outward_actions()`）で USER ターンの system に注入し、`delegate_task` の description にも境界を追記。**一覧を手書きにしない**（search_web は SEARCH_ENABLED ∧ ddgs ∧ TASK_ENABLED の内側でしか登録されず構成で変わる／J-3 で腐る）。**閉世界宣言にしない**（手段に対応しない goal が3件 Done している実績＝考えて答えるだけの依頼は許可）。効果測定 `tools/d3_probe.py` n=204: 過剰承諾 20/42→0/42（Fisher p=9.6e-08）／過剰拒否は可能依頼 1/24→0/24・会話 0/36→0/36 で悪化なし。残課題: 危険∧不能ケースで危険理由への言及が 6/6→4/6 に減少 |
| **D4** | 自己観察ループ（画面に自分のログが映るとそれを画面内容として実況する・軽微） |
| **D5** | レイテンシ **悪化を観測（未対応）**。2026-07-30 の通し3回 pooled（n=92）: 中央値 3.14s / 平均 3.29s / **最大 9.70s / 3秒超 54% / 5秒超 11%**。旧測定（2026-07-29・単一ラン n=32）は 中央値 2.57s / 平均 3.01s / 3秒超 44% で、**≤5s 運用の許容ラインも 10/92 で超過**。心当たりは D1/D3 で増えた system ブロックだが**未検証**。**2026-07-29 裁定: gpt-5.5 の品質を取り ≤5s 運用を許容**（理想の ≤3s は下ろさない）。より賢く速いモデルが出たら `RESPONSE_MODEL` を差し替えて再計測する |
| **D6** | （**解決済 2026-07-29**）`.env` に機能フラグが無く実起動でタスク/検索/Call-Function が全滅していた。E2E ハーネスだけが強制 ON だったため約1か月気づかなかった。→ `.env` と `.env.example` にフラグを明記 |

### 実機で確認済み（2026-08-06 J-3 通し3ラン・生ログは `e2e_logs/j3_rep{1,2,3}/`）
27シナリオ×3ラン（各93〜100分・実LLM/実API・`REAL_STATE=1`）。**修正方針は決めたが未着手**。
判定は発話ではなく実ファイル/実ログで取っている。分析は4観点の並列エージェント + 一次ログでの自己検証。

| ID | 内容 | 再現 | 修正方針 |
|---|---|---|---|
| **E1** | **報告が古い値を喋る**。2回目の時刻報告で1回目の値（約1時間前）をそのまま発話。`# 機能実行結果` には正しい値が入っており、**会話履歴の古い値に引っ張られている**。`DeliveryChecker` は `forgotten` と正しく判定し再配達を予約したが、後続発話に3回中断され `REDELIVER_MAX_ATTEMPTS` で断念。**事後訂正の成功率 0/2** | 2/3ラン | **報告ターンのみ発話前に検証**（`CALLFUNCTION_RESULT` 限定・`USER_UTTERANCE` には入れない）。判定LLMは既に走っているので API 費用は中立、初音が +2〜4s。報告は非同期なので許容（ユーザ裁定）。補助: 中断がユーザ発話由来なら attempts を消費しない |
| **E2** | **ビューア窓がランを跨いで残り、自分の出力を「ユーザの画面」として実況**。rep2 は `海.txt` を t=1292 で語ったが実際の生成は t=5572（72分後）＝前ランの窓を VLM が拾っている。D4 が J-3 の表示能力で軽微でなくなった | 2/3ラン | 窓の自動クローズ（TTL or 新規表示時に古い窓を畳む）+ 本体終了時に道連れ |
| **E3** | **取消の TOCTOU**。取消の LLM 照合（2〜5s）中にタスクが完了し報告が pop されると `suppress()` が「除去0件」で空振りし、結果が配達される。rep3 T9 は「了解、ここで止めるね。結果だけ言うと、一般的にはナイル川とされてるよ」と自己矛盾 | 3/12取消 | `cancel_task` **のみ**応答完了前に dispatch（**−2.9秒**）。`delegate_task` に広げると自分で D2 レースを作るので不可。不感時間の実測は 8.1〜14.6s（内訳: 応答LLMターン 5.7s / resolver 3.0s） |
| **E4** | **無関係タスクの報告を誤破棄**。「**川**の調べ物やめて」で「**琵琶湖**の面積」の報告を破棄。`cancel_resolver.py:175-186` が候補1件のとき reference を照合せず決め打ちする | 1/12取消 | 候補1件でも reference と照合する。2件以上には既に「誤破棄より誤配達に倒す」裁定があるのに1件だけ効いていない |
| **E5** | **能力の過小申告**。`search_web` の description 1文目が「タイトルと概要」なので、rep2 は「ページの中身まで読む手段は持ってない」と言いながら**同じランで deep 検索を7回実行**した | 1/3ラン | description の**1文目**を修正（`outward_actions()` が自動導出＝二重管理ゼロ）。`show_on_screen` の「保存した」も同様に狭い |
| **E6** | 「画面が見えていない」と断った後に画面を実況（+21.6s / +87.6s / +258.3s）。**捏造ではない**: 3ランとも直近VLMが 約1430秒前で、内容も「画面に変化はなく」＝据え置き上限 `VLM_STATIC_VISION_MAX_SEC=600` を超えて `vision==""` に落ち、「画面情報なし」ブロックに忠実に答えていた | 3/3ラン | **`.env` の `VLM_STATIC_VISION_MAX_SEC` を上げる**（コード変更不要）。`vision_for_user` は据え置き時に「(N秒前の画面。以降ずっと変化なし)」を自動付与するので誤解は生じない |
| **E7** | **未知語の知ったかぶり**。「ガララワニってかっこいいよね？」に3ランとも検索せず見た目を断定。**賛同要求形で 0/6**（同型の「ふしでってかわいいよね」含む）／**質問形なら 3/3 で正直**。`style.py:17,20` に同趣旨の禁止文が既にあり負けている（rep2 は「たぶん」を言いながら断定を続けた＝ヘッジ語1個で条件を満たす逃げ道） | 0/6 | **プローブで3案を実測して選ぶ**（(1)出典テスト型 (2)推測禁止・検索型 (3)解説禁止型）。測定群は U群=架空語 / **K群=ピカチュウ・フシデ級＝過剰検索ガード** / Q群=質問形の回帰ガード。**素朴な「未知語は検索」は不可**（`d3_probe.py:69` が「ポケモンのフシデって知ってる?」を CONVERSATIONAL＝最重要ガードに分類しており、それを自分で壊す） |
| **E8** | 自律発話の単調さ（同一ラン内の既出話題 **59%** / 「そういえば」始まり **50%** / 問いかけ終わり **10%**） | 3/3ラン | **口癖は許容**（ユーザ裁定）。先頭N文字比較は不可: 「そういえば前に」が35%を占め、**内容が違うのに差し戻す誤爆**になる（直前比較なら 8/110 しか捕捉できず弱すぎる）。対処するなら同内容抑制が言い換えを取りこぼしている側 |
| **E9** | 検索が手数上限で失敗（2/82タスク）。**1手目で答えが揃っていた**のに⑪手目まで再検索し、228秒待たせて謝罪。同じ依頼が別ランでは5〜7手で成功 | 2/82 | **上限は上げない**（12→20 でコスト約2.7倍・謝るまで約6分。外側 `SEARCH_TIMEOUT_SEC` 到達は 136回中1回=0.7% で律速でもない）。限界離脱時に **notes ではなく収集済みの事実**を返す（`agent.py:164-182`）＝Failed が実質 Done 化 |
| **E10** | J-3 実装の積み残し3件: `Workspace.maybe_rescan()` の**呼び出し元がゼロ**（定期クォータ補正が動かない）／`agent.py:109` が「時間切れ(3分)」表示（実際は600秒）／`.eve_run/` が `list_files` に見える | — | 配線・文言修正・`.` 始まり除外 |

#### 第1弾の修正結果（2026-08-07・ブランチ `fix/e2e-round1`・検証ログ `e2e_logs/verify_round1/`）
検証は「効果検証 / レッドチーム / 前後比較」の3エージェント + 一次ログでの自己検証。**主張は自分で再検算した**。

| ID | 状態 | 残っている問題 |
|---|---|---|
| **E6** | **部分的**（上限 600→1800） | **構造は未修正**。`vision==""` が「鮮度切れ」と「画面能力なし」を同じ状態に潰す設計は残る。上限は観測最大(1453s)の1.24倍しか余裕がなく **180→600→1800 と3度目の引き上げ**＝次の長い放置で再発する。恒久解は `VLM_PERIODIC_FRAMES` を有効化して静止中も5分に1回実況を更新すること（現在 既定0＝無効）。**pHash 閾値12 は窓の中身を全文差し替えても hamming 0** を返す（実測）ので「以降ずっと変化なし」の注記は**保証していない次元に付いている** |
| **E10** | **部分的** | `maybe_rescan` の呼び出しは `run_program`（`force=True`）**1箇所のみ**＝`WS_RESCAN_SEC=600` の**定期経路は今も呼び出し元ゼロ**。`WS_EXEC_ENABLED=0` の構成では再びゼロになる。さらに **`list()` は `.` を隠すが `iter_files()`（会計）は数える**ため、一覧「1ファイル」／会計「2ファイル」の不整合を新設した（実証済）。構造解は `.eve_run` を作業フォルダの外（`C:\eve_ws\runtime\jobs\`）へ出すこと |
| **E5** | **部分的** | `outward_actions()` は**1文目しか取らない**ため、`run_program` に足した「ネットからファイルを取ってくる」（2文目）は**応答LLM に届いていない**。3件中 `search_web` だけが実証（同一シナリオ S6 の前後対比で過小申告が消えた）。加えて **system の `NEVER_MEANS`「外部サービスへの書き込みは手段が無い」が curl/python の存在で偽になる**矛盾を作った |
| **E4** | **差し戻し（`6cb6f25`）** | 直した不具合（1/12取消）より大きい回帰を作ったため revert。実測: 「さっきの検索はやめて」が実タスク6件中4件をブロック（goal が「調べて」だと残渣「検索」が掛からない）／「大丈夫」「忘れて」「無しで」「もう結構」「中断して」等の**日常的な取消表現 9/9 をブロック**／`_VAGUE_TOKENS` の単文字助詞が語中を食う（おにぎり→おぎり・ぷよぷよ→ぷぷ）。**正しい設計は「単独照合」ではなく「相対照合」**（terminal 集合全体と突き合わせ、別タスクの方が明確にスコアが高い時だけ抑止を見送る）。実機事故（川 vs 琵琶湖）は競合タスクが実在したので相対比較で検出でき、「忘れて」はどれにも掛からず相対差が付かないので従来どおり抑止される |
| **E2** | **部分的** | 実証済（開4=閉4・残留0）。ただし塞いだのは**ラン跨ぎだけ**で、**ラン内の自己観察は修正後ランでも再現**（`verify_round1 03:54:34`: 自分が7秒前に開いた窓を「デスクトップ上に…が新規に開かれています」とユーザの画面として実況）。異常終了時は依然残る（起動時の孤児掃除が無い）。**窓を閉じる capability が無い**ため、5窓上限に当たるとイブは自力で回復できない |

**第1弾で新たに判明した、着手すべき課題**:
- **E11 自己観察ループ（ラン内）**: E5「見せた方が早い」→表示増→窓が居座る→VLM が自分の出力を実況→E6 が最大1800秒それを据え置く、という**3修正の合成**で D4 が悪化する方向に揃っている。構造解は「`ws.viewers` が非空の間、VLM narrator に『この窓はあなた自身が開いたもの』を注入する」＝窓を消さずに済むので E2 の判断反転自体が不要になりうる
- **E12 窓を閉じる能力が無い**: `close_window` / `close_all_windows` が未実装
- **E13 私のテストが欠陥クラスを守れていない**: E10(1) のテストは `maybe_rescan` を**自分で直接呼ぶ**ので `capabilities.py:181` を消しても PASS する（元の欠陥＝呼び出し元ゼロと同じクラスを検出できない）。E10(3) のテストは書式をテスト側で再実装した**トートロジー**で `agent.py` の分岐を通らない
- **E14 D3 プローブが旧文言のまま**: `run_program` の1文目を広げたのに `tools/d3_probe.py` の最終実行は 2026-07-30。過剰承諾 20/42→0/42 は**旧文言での測定値**

**測定で否定され、問題リストから外したもの**（再調査しないこと）:
- **画面捏造**: 画面言及 27件を全数 VLM ナレーションと突合し**捏造ゼロ**。`⚠画面捏造の疑い` 2件は「画面」の語だけで発火する検出器の誤検知
- **D1 時制の誤り（rep2）**: 124秒前の同一セッション発言を正しく参照。ハーネスがセッション全体にキーワード照合するための誤検出
- **レイテンシ悪化**: 会話（USER_UTTERANCE）は中央 **2.87s / 5秒超 10%** で D5 基準（3.14s / 11%）より良好。全体値 3.22s が悪く見えるのは自律発話（中央 3.77s）の構成比が 18.5%→27.8% に増えたため。同一シナリオ限定では +0.43s（p=0.028）だが**独立単位はラン3対3で原理的に確定不可**＋測定時刻の交絡あり
- **「検索してみるね」の強制**: 委譲82件中 **55件(67%)が既に言っている**。残る26%は LLM が無言だったケースで `_FALLBACK_ACK` が既に埋めている

**その他の実測値**: タスク成功率 82件中 Done 75 (91.5%)／複合タスクは15件44手（理論最小42手）で迷いゼロ／話速 4.91字/秒（E2E の擬似プレイヤーは再生時間を1.57倍に伸ばすためログからは測れず VOICEVOX 再合成で校正）／自律発話 speak率 0.092（3ラン間に有意差なし・D5 比も変化なし）／RAG は 200→332件で暴走なし（ただし追加分の 47.6% がイブ自身の発話由来）

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
3. （**解決済 2026-07-30**）main マージ: Tier-1 434件を5回連続 PASS で確認し `main` に fast-forward 統合済み

---

## 5. 設定の要点

- **機能フラグは `.env` で明示する**（コード既定は全て off）。将来 UI から ON/OFF する設定項目という位置づけ。`.env.example` に全フラグを明記済み
- 実運用モデル（role 名は `eve/model_registry.py` の `ROLE_ENV` キー）: `response`=gpt-5.5 / **`speech_decide`**=gpt-5.4-mini（env は `DECIDE_MODEL`・"decide" は role 名ではない） / `feedback`=gpt-5.4 / `summarize`=gpt-4o-mini / `vlm_leaf`=gemini-2.5-flash。`task` `search_summarize` `delivery_check` はコード既定のまま。ROLE_ENV は全10 role で、`vlm_merge` と `youtube` は消費者ゼロ
- Web検索は `SEARCH_BACKEND=auto`（DNS が遮断される回線では `SEARCH_BACKEND=duckduckgo` に固定する）
- 主要な閾値の実測校正値は `eve/config.py` のコメントに日付つきで残してある（同内容抑制 0.25/0.87、記憶の自己参照除外 600秒、画面据え置き 600秒 等）

## 6. 落とし穴

1. `$env:PYTHONIOENCODING="utf-8"` を忘れると日本語＋絵文字で落ちる
2. Git Bash の grep は絵文字入りログで壊れる → **Python で解析する**
3. `.env` 変更は再起動が必要（`Config` は構築時に1回だけ読む）。E2E ハーネスは `eve.config` を import する**前**に環境変数を書き換えている＝この順序を崩すと効かない
4. E2E は課金・実画面・メモ帳の自動操作を伴う。VLM 有効時は**実際の画面が Gemini に送られる**
5. 実起動で回すと本番の記憶（`*.jsonl`）が育つ＝D1 の温床
6. `.env` と `*.jsonl` は gitignore＝設定の修正はコミットに残らない
7. **自発発話の効果測定は「件数」で判断しない**。1セッション3-4件しか出ずノイズに埋もれる（条件付き Poisson 検定で p=1.00 の実例あり）。判定回数（数十〜百）を母数にした指標を使う
8. **作業フォルダをユーザプロファイル配下に置くと `run_program` が必ず失敗する**。`C:\Users\<user>` に `BUILTIN\Users` の ACE が無く eve_sandbox から到達できないため（起動時に `exec_compatible` が検出して実行だけ無効化する）。E2E ハーネスも一時ディレクトリを使って実際に踏んだ
9. `wmic` は Windows 11 24H2 以降で**削除済み**。セットアップ系スクリプトで使わない（`Set-LocalUser` 等の PowerShell に寄せる）
10. **`Path.exists()` はアクセス拒否時に False ではなく `PermissionError` を送出する**。「読めないこと」を確かめる検査は try で囲まないと、成功の証拠が例外になって以降の検証を巻き込む
11. J-3 の能力は `.env` に **`WS_ENABLED` / `WS_EXEC_ENABLED` / `WS_VIEWER_ENABLED`** を書かないと丸ごと無効（D6 と同じ構造）。`WS_EXEC` はさらに `setup_sandbox.py apply` 済みが前提
