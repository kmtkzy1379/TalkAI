# Eve v2 — コンポーネント挙動仕様（優先順位決定の前提）

> 目的: 各ロジックが「何をどう処理し、どう振る舞うか」と「どの機能と関わるか」を、**具体コードの前に**確定する。これがないと優先順位を決めても後で破綻する（ユーザ指示）。
> 出典: `PIPELINE_DESIGN.md`（骨格）+ 4並列調査エージェント（claw-code / Scrapling / screen-op安全 / OSSサーベイ、いずれも一次ソース確認済）。
> 凡例: 【確定】=企画書/調査で決まり。【推奨】=本書の設計提案。【要決定】=ユーザ裁定待ち。

---

## A. 入力層

### A-1. mic → VAD → 増分STT 【確定】
- **挙動**: マイク音声を VAD で常時監視。発話開始で取り込み、**partial(50–300ms)が出た時点で応答パイプラインに投機投入**（最終確定を待たない）。VAD がターン終端を確定したら final で確定/差し替え。
- **関わり**: SurpriseBus(barge-in 検出) / StimulusQueue(user_utterance 投入) / AudioPlayQueue(自己エコー除去=AEC で自分のTTSを拾わない)。
- **OSS**: silero-vad(VAD/barge-in, MIT) + RealtimeSTT(partial土台, MIT) + 日本語backend(kotoba-whisper/faster-whisper)。【推奨】

### A-2. YouTube 取得 【確定】
- **挙動**: YT Data API でコメントを**上から1件ずつ**取得（v1 は最新1件で逆だった）。リアルタイム音声なし。YTプロンプト + 強モデル(Opus 4.8 相当)。Call-Function は **task のみ**許可。
- **関わり**: StimulusQueue(youtube刺激) / ModelRegistry(youtube役) / Call-Function(task のみ)。

---

## B. StimulusQueue（応答起動の単一窓口）【確定】
- **挙動**: 「次に応答LLMを起動する刺激」を1本化。priority(user > callfunction_result > autonomous_speech > vision) + merge(vision/feedback 複数は畳む) + 逐次(CallFunction 複数) で drain。busy 中は保持、barge-in は別経路。
- **P4 starvation 対策**: surprise/VLM 連発時に低優先(タスク結果報告)が永久に出ない優先度逆転を防ぐため、各itemに enqueue 時刻を持たせ**控えめな bounded aging**(待機が閾値~30s超で優先度を1段昇格)。aging は強すぎると優先度の意味を失うので控えめに。
- **関わり**: 全入力源の合流点 / ResponseOrchestrator(下流) / SurpriseBus(drain 優先度の補正)。

---

## C. ContextAssembler（過去参照防止の要）【確定/推奨】
- **挙動**: 応答LLM への文脈を組み立てる。入力 = 発話 + systemプロンプト + 直近5ターン + RAG2 + 直近feedback1 + 画面認識(統合済) + 発話判定理由 + CallFunction定義。
- **過去参照防止(Msg6)**: 全要素に ISO-8601+monotonic タイムスタンプ → 組立時に相対時刻(「3分前」等)を明示注入。RAG/feedback/vision は**応答前に非同期準備済み**にしキャッシュ参照（≤3s 予算のため）。無言時 random RAG は「話題の種」と明示ラベルし「思い出話」と峻別。
- **関わり**: RAG / FeedbackLLM / VLM / SurpriseBus(自己懐疑ヒント) / 応答LLM。**ここが「今の会話に接地」を強制する責任点。**

---

## D. 応答オーケストレータ（神経系・即時）【確定】
- **挙動**: StimulusQueue から1刺激 drain → ContextAssembler → 応答LLM を token stream → **文単位で split** → 文ごとに TTS → AudioPlayQueue。CallFunction を stream 中に抽出し**応答完了後に**逐次実行。surprise を**必須引数**で受け、文脈不整合時は自己懐疑（v1 の「疑わず解答だけ返す」欠陥の修復点）。
- **関わり**: StimulusQueue / ContextAssembler / ModelRegistry(response役) / SurpriseBus(必須) / TTS / AudioPlayQueue / FunctionDispatcher。
- **OSS**: ja_sentence_segmenter(文分割, 【推奨そのまま使う】)、pipecat(barge-in フレームモデルの設計参考)。

---

## E. SurpriseBus / PredictionState（中核原理）【確定】
- **挙動**: 予測誤差(surprise)を asyncio loop 所有・単一更新者・同期読みで保持。FeedbackLLM の prediction-diff(0-100) と VLM の screen-diff を集約。**(a)自発発話の要否/内容 と (b)文脈不整合の自己懐疑 を必須引数でゲート**。surprise 反転で発話/沈黙判定が反転しなければビルド失敗(T2)。
- **関わり**: FeedbackLLM(書込) / VLM(書込) / 発話判定LLM(読) / 応答オーケストレータ(読) / StimulusQueue(drain優先度)。
- **OSS**: pymdp(自由エネルギーの数式リファレンス)、PUMA論文(LLM対話で belief更新+EFEで応答選択=Eveの発想の先行例)。いずれも**理論支柱**であり実行時組込みは過剰 → 軽量自作。

---

## F. 発話判定LLM（沈黙経路・クリティカルパス外）【確定】
- **挙動**: 5s 沈黙で起動。入力 = "…" + 直近会話 + ランダムRAG2 + 画面認識 + **surprise**。True なら(理由+応答LLMへの入力)を返し autonomous_speech 刺激に。False なら(理由→ログのみ)で無音。VAD のターン終端とは役割分離（VAD=話し終え検出 / 沈黙タイマー=誰も話さない時間）。
- **関わり**: SurpriseBus(読) / RAG(random) / VLM / StimulusQueue / ModelRegistry(軽量高速役)。

---

## G. FeedbackLLM（内分泌系・遅延許容・非同期）【確定】
- **挙動**: 各応答後に非同期起動。出力 = emotion / summary / user-emotion推定 / next-prediction / **prediction-diff(0-100)**。沈黙時は話題提案。応答クリティカルパスに乗らない。
- **関わり**: SurpriseBus(diff を書込) / RAG(summary を1チャンク=FB1+応答1 で保存) / ContextAssembler(直近feedback1)。
- **PORT**: v1 `feedback_llm.py` の3段fallback/cache_control を ModelRegistry に一般化（gpt-4o ハードコード/anthropic既定は撤去）。

---

## H. 画面認識VLM（capture→Gemini×3→統合）【確定/裁定済】
- **挙動**: 画面変化時(pHash→SSIM ゲートで静止画はスキップ)にスクショ → **同一フレームを Gemini Flash 2.5 ×3 の self-consistency**（誤読を平均化=YOLO起因ハルシネの代替）→ より賢いモデルで統合ナレーション。重量級CV(YOLO/tracking/analysis/aggregation/saliency)は全廃。
- **関わり**: SurpriseBus(screen-diff 書込) / StimulusQueue(vision刺激, 複数はmerge) / ContextAssembler(画面認識) / screen-op(責務分離: VLM=読む / screen-op=作用する)。
- **PORT**: `vlm/capture/screen.py`(mss+pHash) + `change_detector.py`(変化ゲート, 2値に簡素化)。**OSS**: Open-LLM-VTuber の視覚認識アーキが設計参考。

---

## I. RAG 【確定/推奨】
- **挙動**: 1チャンク = **フィードバック1 + 応答1**(v1 の episode要約中心を是正)、件数 **300**(v1 3000 の是正)、意味+連想検索。ランキング式(importance+recency+relevance→hard-cut→MMR)は v1 が優秀なので**式は参考**に再実装。無言時のみ random 2件(話題の種)。全チャンクにタイムスタンプ。
- **関わり**: ContextAssembler(通常2件) / 発話判定LLM(random2件) / FeedbackLLM(書込) / async write queue(JSONL永続化)。
- **REFERENCE**: v1 `rag.py:297-415` のランキング、`conversation_cache.py` の write-queue/「…×X」圧縮。

---

## J. Call-Function（task / search / screen-op）— ここが本調査の核心

共通契約【確定】: 応答LLM が stream 中に呼ぶ → **応答完了後に逐次実行**（複数は並列にしない）→ **結果が出たら刺激として StimulusQueue に再投入** → 後で自発発話の材料に。実行は非同期サイドカーで本体応答をブロックしない。

> **ユーザ裁定(2026-06-17)で判明した統合洞察**: task / search / screen-op は別物ではなく、**1つの共有「Capability(能力)層」を3つの起動経路から叩く**構造にする。ユーザの言う「screen-op はタスクや検索にも応用・関わる」がこれ。下記 J-0 が共通基盤、J-1〜J-3 はその上の経路/能力。

### J-0. 共有 Capability(能力)層 【確定・統合】
- **挙動**: Eve が外界/自分に作用する手段を**構造化された能力(enum + 型付き引数)**として1箇所に列挙。各能力は固有の安全ゲートを持つ。ユーザ指定の初期パレット:
  - `window_op`（ウィンドウ操作: 列挙/前面化/最小化 等）
  - `download`（URL→ファイルDL。保存先は Eve 専用作業ディレクトリ）
  - `launch_app`（許可済みアプリ起動）
  - `pc_status`（CPU/メモリ/時刻/プロセス等のPC状態確認＝読取）
  - `self_status`（Eve 自身の状態確認: queue長/タスク数/直近エラー等＝読取）
  - `search`（ネット検索。中身は J-2）
  - （将来）`terminal_recipe`（事前定義済みコマンドのみ。生コマンド禁止）
- **3つの起動経路**:
  1. **即時 Call-Function**: 応答LLM が「今これをやる」と能力を直接呼ぶ（応答完了後に実行）。
  2. **task 経由（遅延/予約/複数手順）**: J-1 のタスク executor が予定時刻に能力を呼ぶ。
  3. （screen-op/search は能力名であって独立機能ではない＝経路1か2のどちらかで実行される）
- **安全ゲート（全能力共通・コードで強制）**: 読取系(pc_status/self_status/window列挙)は自動。状態変更系(download/launch_app/window前面化)は allowlist + 出力サイズ/タイムアウト制限。**破壊的操作(削除)は Eve が生成/DL したファイル・フォルダのみに限定**（ユーザ指定の安全制約）。無人配信を想定し危険操作の無承認=拒否。
- **関わり**: 応答LLM(経路1) / task executor(経路2) / FunctionDispatcher(検証) / StimulusQueue(結果再投入) / UI(承認) / surprise(想定外結果)。

### J-1. task（タスク管理）— claw-code 調査 + ユーザの「人間的タスク」像で統合 【確定】
claw-code 一次調査の結論: tasks は **inert なメタデータ記録**で status 遷移は外部が握り、subprocess 実行は**未実装**・永続化なし。→ **形(shape)だけ採用し、欠けている実行系・永続化・予約時刻・再計画を Eve が足す。**

ユーザ裁定: deadline は**別概念にせず通常タスクに統合**する。v1 が壊れたのは「時刻をタスクに乗せたこと」ではなく「完了所有権の二転三転 + fact-store/audit との癒着」。その規律さえ守れば人間的な予約タスクは安全に作れる。

- **タスク記録(memo)のフィールド**: `task_id` / `what`(やること=自然言語意図 or 能力呼出) / `when`(予約時刻 `scheduled_at`。null=応答後すぐ、「5分後」=now+300s) / `order`(複数時の順序) / `status` / `result` / `attempts` / `created_at`/`updated_at` / `parent_id`(再計画の親リンク)。
- **status**: `Pending → Running → (Done | Failed | Cancelled)`。terminal は**再遷移禁止をコードゲート**で（v1「完了所有権が二転三転」の直接修復）。**遷移は executor のみが所有**、LLM は create/cancel を*要求*するだけで `Done` を主張できない。
- **人間的ループ(ユーザの記述どおり)**:
  1. **列挙**: やることを順にタスク記録化（memo を書く）。各に任意で `when`。
  2. **時計を見る**: 周期スケジューラ(reconcile timer ~1s、沈黙中も動く＝v1 RC1 の良かった点)が「`scheduled_at<=now` の Pending」を探す。
  3. **時間が来たら実行**: due タスクを executor が非同期サイドカーで実行（= J-0 能力を呼ぶ）。`Pending→Running`。
  4. **確認**: 成功→`Done`+result を StimulusQueue へ（Eve が「終わったよ」と言える）。
  5. **失敗時の再計画**: 失敗→`Failed`(理由つき、terminal のまま固定)→ LLM が「なぜ失敗/次どうするか」を考え、**新しいフォローアップタスクを作る**(`parent_id` でリンク)。terminal-stays-terminal を守りつつユーザの「タスクを振り直す」を実現（復活させない＝v1 修復）。
- **永続化**: JSONL(async write queue)。起動時 reconcile で orphan な Running を age out。**dedup/dispatch ガード**で重複タスク→重複刺激を防ぐ。
- **捨てる**: in-memory only / subprocess 枠組み / task_packet・team・lane-board / DAG プランナ(CLAUDE.md 禁止) / messages インボックス。
- **実行係(executor)の定義**: **LLM ではなくコード**(async関数)。due タスクを拾い→能力層を呼び→**実際の実行結果で** status を確定→結果を StimulusQueue へ。LLM は道具として呼ぶだけ(タスク化判断/検索クエリ/要約/失敗時の再計画推論/P2のjudged verdict)。= 「制御フロー=コード、中身の判断=LLM」(OpenAI Agents SDK / LangGraph の主流)。
- **検証で判明した必須パッチ（GitHub 一次調査 2026-06-17・実例裏取り済）**:
  1. **P1 再計画の暴走停止(最優先)**: `parent` フォローアップは無限増殖しうる(opencode #17169 で $100+ 実損害, AutoGPT 成功率24%)。→ **チェーン深さ上限(~3)** + **non-retryable 分類**(認証なし/URL不正/未対応は再試行せず即 Failed, Temporal式) + **重複検出**(親と同 what/when の子は作らない) + per-goal の **step/token 予算**。上限到達は terminal Failed + 「諦め」を刺激化。
  2. **P2 完了判定の二層化**: タスクに `verdict_kind: deterministic | judged`。`deterministic`(DL/ファイル/screen-op)はコードが status 単独所有(現方針)。`judged`(検索品質・要約十分性など曖昧な成功)は**コードが Running→「結果確定」まで所有し、最終 status だけ LLM の構造化 verdict `{outcome: ok|weak|fail, reason}`** で決める。レイテンシは judged のみに限定し即応経路を汚さない(arxiv 2508.16671)。
  3. **P3 依存リンク追加**(DAGプランナではない): `depends_on:[task_id]` + `input_from:task_id`(親結果のバインド)を1フィールド追加し「検索→結果でDL→報告」の動的1段chainを明示化(BabyAGI 自身がフラット→グラフ移行した教訓)。事前グラフ構築はしない。
  4. **時刻パスの締め**: 期間は **monotonic clock**、`Pending→Running` は **atomic**(次tickの二重発火防止)、沈黙/ダウン中に過ぎた予約は **misfire grace** で1回だけ発火 or 意図的に破棄(古い nudge は捨てる)。`when` は UTC で JSONL 永続。
  5. **非ブロッキング不変条件**: 即時Call-Function経路から呼べる能力は必ず非ブロッキング(長時間opは即タスク化して即return)。即時/タスクの振り分けは**コードで固定、LLMに判断させない**。
- **参考実装**: `steveyegge/beads`(JSONL+キャッシュ+依存グラフ+ready検出+git-merge)が Eve のタスクストアに酷似。スキーマ確定前に一読(依存にはしない=Go CLI)。`tenacity`(機械的リトライ), `APScheduler`(任意, 自前1秒ループでも可)。
- **関わり**: 応答LLM(要求) / executor(コード・遷移所有・能力呼出) / スケジューラ(時計) / J-0能力層 / StimulusQueue(結果再投入) / JSONL / UI(読取専用リスト) / surprise(stale Running・失敗→自己懐疑, P6は操作的に=surprise反転で下流挙動が変わることをTier-3計測)。

### J-2. search（検索）— Scrapling 調査 + ユーザ裁定 = ddgs + Scrapling 【確定】
Scrapling 一次調査の結論: Scrapling は「**取得+パース+Markdown抽出**」層で、**検索エンジン機能を持たない**。async/ステルス/robots対応あり(BSD-3)。→ 検索エンジン層が別途必要。**ユーザ裁定: ddgs(無料・キー不要メタ検索) + Scrapling(取得・抽出)** を採用（コスト0・依存最小・ローカル常駐 Eve と相性）。
- **採用する挙動（段別）** — J-0 の `search` 能力の中身:
  1. **判定+クエリ生成**: 応答LLM or task が「ネット検索が要る」と判断 → 検索キーワード+意図ラベルを生成（ユーザ確認質問への回答=Yes、この流れ）。
  2. **検索(URL発見)**: **ddgs** で上位3–5件の URL+スニペット。
  3. **取得(fetch)**: Scrapling `Fetcher`(HTTP高速)既定 → ブロック時のみ `StealthyFetcher`。**async セッション**で別 executor。`robots_txt_obey=True`。
  4. **抽出→要約**: Scrapling の Markdown/text 抽出 → **要約LLM**で1–3文に圧縮。**要約モデルは ModelRegistry で差替可**（ユーザ関心: ローカルは答えがブレるので Qwen 14B 級 / 日本語なら Ollama 系。サイズ/ローカルvsクラウドは実装時に検証）。
  5. **再投入**: 要約+出典URL を StimulusQueue へ。
- **タイムアウト/同時実行上限/キャッシュ/dedup を必須**。失敗は握って no-stimulus(落とさない)。検索結果は信頼できないデータ扱い、current context を上書きさせない(RAG の affect 弱ヒント原則と同様)。
- **関わり**: 応答LLM/task(クエリ) / StimulusQueue(再投入) / RAG(TTL/重要度判定後に保存可) / ModelRegistry(要約モデル) / surprise(予想外の事実→優先度↑)。
- **補完候補(後日)**: SearXNG(自前ホスト・JSON) / Tavily(LLM特化1API・有料枠) / markitdown(各種→Markdown 正規化)。

### J-3. screen-op（画面操作）— 安全調査 + ユーザ裁定 = 仮決定 C 拡張 【仮決定・task/search 確定後に再議論】
安全調査の一次結論(複数ソース一致): **「LLM が出した生コマンドをそのまま実行」は配信VTuberで採用不可。** denylist/substring は必ず破られる、信頼境界は OS サンドボックスで引く、攻撃面はコマンド名でなく**引数**。
- **ユーザ裁定(仮)**: screen-op は独立機能でなく **J-0 能力層そのもの**。初期パレット = window_op / download / launch_app / pc_status / self_status（=案C「構造化限定アクション」。生コマンドは出させない）。実装/精度の都合で一部能力を **allowlist 済みターミナルレシピ(案A 風)** で内部実装してもよいが、その場合**削除は Eve が生成/DL したファイル・フォルダのみ**に限定。**task/search のロジックが固まったら再議論**（ユーザ明言）。
- **採用すべき安全ゲート（多層・全てコードで強制）**:
  - **段0(採用)**: 生 bash をやめ、**構造化限定アクション(enum+型付き引数)**だけ LLM に出させる。引数インジェクションの大半が構造的に消える。
  - 段1: 生コマンド/レシピも `shell=True` 禁止、argv 配列実行、パイプ/リダイレクト/コマンド置換/複数コマンドは拒否。
  - 段2: **allowlist**(バイナリ/レシピ列挙) + **フラグ検証**(`-exec`/`--pre`/`-c` 等のコード実行・任意書込フラグ拒否) + GTFOBins 監査 + `--` セパレータ。
  - 段3: **OSサンドボックス**(作業ディレクトリのみ書込, ネット deny-by-default) + タイムアウト + 出力サイズ上限 + 非対話(TTYなし)。Windows は srt 非対応 → Win32 App Isolation / Job Object / WSL2-microVM を要検討(未確認)。
  - 段4: 出力は**信頼できない外部入力**として再投入(二次プロンプトインジェクション防止: 出力からの新規 Call-Function を1ホップ抑制)。破壊的操作は**UI 承認**、無人配信を想定し**無承認=拒否**を既定。
- **関わり**: 応答LLM/task(能力要求) / FunctionDispatcher(検証→実行) / StimulusQueue(結果再投入) / VLM(読む=VLM / 作用する=screen-op の責務分離) / UI(承認ゲート) / surprise(想定外 exit→反応)。

---

## K. ModelRegistry（provider 非依存・Claude 代用）【確定】
- **挙動**: role→model 間接層。`.env` 既定 + UI から swap/temp。Claude 停止中は response 役を GPT/Gemini 代用、後で戻せる。litellm で provider 差吸収。
- **関わり**: 全LLM役(response/speech_decide/feedback/vlm_leaf/vlm_merge/youtube) / UI(差替) / feedback_llm(PORT 一般化先)。

---

## L. UI / Start-Stop 安全ゲート【確定】
- **挙動**: `STOPPED→VALIDATING→RUNNING→STOPPING` 状態機械。**全設定valid時のみ Start 有効**(同期バリデータ)。RUNNING 中は設定ロック。topmost。各LLMの swap/temp/prompt編集/ログ/モード/VLM on-off。widget更新は `root.after(0,fn)`。
- **関わり**: ModelRegistry / 全パイプライン(Start/Stop) / Call-Function承認(screen-op)。

---

## M. 配線層（PORT・理解の上で移植）【確定】
- vts.py(別プロセスVTS) / player.py(AudioPlayQueue, seq+generation 追加) / tts.py(VOICEVOX 2段) / launcher.py / run.py / vlm capture(screen+change_detector) / common config-loader。詳細は SALVAGE_MANIFEST.md ①。

---

## 機能間インタラクション・マップ（破綻防止のため境界を明示）

| 起点 | 終点 | 何が流れるか |
|---|---|---|
| 入力(mic/YT/silence/screen) | StimulusQueue | 刺激(優先度つき) |
| StimulusQueue | 応答オーケストレータ | drain 1件(or merge) |
| ContextAssembler | 応答LLM | タイムスタンプ付き接地文脈 |
| 応答LLM | TTS→AudioPlayQueue | 文stream(seq+generation) |
| 応答LLM | FunctionDispatcher | CallFunction(完了後逐次) |
| FunctionDispatcher(task/search/screen-op) | StimulusQueue | 実行結果=再投入刺激 |
| FeedbackLLM | SurpriseBus / RAG / ContextAssembler | diff(0-100) / FB1+応答1 / 直近FB |
| VLM | SurpriseBus / StimulusQueue / ContextAssembler | screen-diff / vision刺激 / 統合ナレ |
| SurpriseBus | 発話判定 / 応答オーケストレータ / StimulusQueue | 必須ゲート信号 |
| ModelRegistry | 全LLM役 / UI | role→model 解決 / 差替 |
| UI | 全体 / screen-op | Start-Stop / 承認 |

**特に密結合で要注意の境界**:
1. **CallFunction結果 → StimulusQueue 再投入**: 再投入刺激が再挨拶/連続nudge/二次インジェクションを誘発しないよう、dedup・nudge抑制窓・データラベルを全 Call-Function に共通適用。
2. **surprise の単一更新者**: FeedbackLLM と VLM が両方書く → loop 所有の単一更新者で競合回避(ロック不要)。
3. **VLM と screen-op の責務分離**: 読む=VLM / 作用する=screen-op。スクショ取得をどちらが持つか要整理(VLM 側に寄せるのが綺麗)。
4. **task と deadline は統合**(`when` フィールド + 失敗→新タスク再計画)。v1 破綻源は「時刻搭載」でなく「完了所有権の二転三転 + fact-store/audit 癒着」だったので、executor 単独所有 + terminal 固定 + 復活禁止 の規律で統合は安全。
5. **3経路が同じ Capability 層を共有**(即時Call-Function / task executor)。screen-op/search は能力名であって独立機能でない。

---

## ユーザ裁定の記録（2026-06-17）
1. **screen-op scope = 仮決定 C 拡張**(window_op/download/launch_app/pc_status/self_status の能力パレット。生コマンド不可。一部は allowlist レシピで内部実装可だが削除は Eve 自作ファイルのみ)。**task/search ロジック確定後に再議論**。
2. **search = ddgs + Scrapling 確定**。要約モデルのサイズ/ローカルvsクラウドは実装時に検証(Qwen14B/Ollama 関心)。
3. **deadline = 通常 task に統合確定**(`when` + 再計画ループ)。

## 検証の到達点（2026-06-17）
- **task モデルは GitHub 一次調査(エージェント基盤横断 + 実装パターン + レッドチーム)で検証済**: 骨格4本柱は業界定石と一致、全面作り直し不要。必須パッチ P1〜P3 + 時刻締め + 非ブロッキング不変条件 + P4 aging を反映済。
- **実行係=コード / 完了→StimulusQueue 経由** をユーザ質問に回答し確定。
- **deadline=通常タスク統合 / search=ddgs+Scrapling** 確定。**screen-op=仮決定C拡張**(実装時に task/search と合わせ再議論=ユーザ明言)。
- 「その他機能はおおむね同意・実装時に詰める」(ユーザ) → **ステップ③(実装優先順位)へ進める状態**。
