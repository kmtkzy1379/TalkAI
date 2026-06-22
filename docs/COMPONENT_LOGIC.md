# Eve v2 — コンポーネント挙動仕様（優先順位決定の前提）

> **⚠ 実装で更新された点 (2026-06-20・コードが正)**: A-1 STT は **partial 投機なし**（VAD区間→final）・AEC 不採用。
> §I RAG は **件数 500**（300は旧値）、注入は直近6ターン、ランキングに **異方性 baseline 補正**を追加、
> 埋め込みは **make_embedder**（ModelRegistry role でない）。実装状況・最新仕様は `docs/HANDOFF.md`。

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

## C. ContextAssembler（過去参照防止の要 + 話者ロール接地）【実装済・native ロール（Fix4）】
- **挙動**: 応答LLM への文脈を **native チャットロール messages** で組み立てる（`assemble(...) -> list[dict]`）。
  - `system` = systemプロンプト(スタイル) + **ロールアンカー「assistant=イブ自身／user=相手・自分の発話に返事しない」** + 文脈(過去の記憶RAG/話題の種/画面/直近feedback/発話判定理由)。
  - 会話 = `user`/`assistant` の **native ターン列**（連続同roleはマージ・中略マーカ保持）。
  - 最終 = ユーザ発話(user) or **自発指示**「返事でなくイブ自身から一言」(user)。
- **なぜ native ロール（Fix3/Fix4）**: 1個の `role:"user"` ブロブに会話を詰めるとモデルが話者を取り違え、**自分(イブ)の発話に自分で返事/自分の質問に自答**する実機事故が出た。assistant=イブをモデル本来のロール構造で示し構造的に防ぐ（実機で解消確認）。自発の content は「ユーザ発話」枠でなく自発指示として渡す。
- **過去参照防止(Msg6)**: RAG/feedback/vision は **system に相対時刻付き**で接地（無言時 random RAG は「話題の種」と明示ラベルし「過去の記憶」と峻別）。直近会話ターンは本文のみ（相対時刻前置きは応答LLMが復唱する leak のため撤去＝Fix4b）。
- **関わり**: RAG / FeedbackLLM / VLM / 応答LLM。**ここが「今の会話に接地」+「話者ロール接地」を強制する責任点。**

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

## F. 発話判定LLM（沈黙経路・クリティカルパス外）【実装済 2026-06-21・F5】
- **実装**: `eve/speech/decider.py`(should_speak/パーサ/decide_fn) + `eve/speech/monitor.py`(SpeechState/SilenceMonitor/SpeechDecider)。
- **挙動**: **5秒沈黙**で起動（フラット5秒で連続再評価＝実世界を細かく観測）。入力 = "…" + 直近会話 + ランダムRAG2(`rag.random`=話題の種) + **イブの今の感情/要約(直近フィードバック)** + 画面認識(VLM後続) + **surprise**。True なら(理由+応答LLMへの入力 content)を返し `AUTONOMOUS_SPEECH` 刺激に。False なら(理由→**発話判定ログのみ**・応答LLMには入れない＝「楽な False」偏り防止)。speak で content 空なら全 speak 経路で fallback。VAD のターン終端とは役割分離（VAD=話し終え検出 / 沈黙監視=誰も話さない時間）。
- **surprise は「指標」（数値で絶対決定しない・ユーザ裁定 Fix2）**: HI/LO の数値強制ゲートは**撤廃**。人間も予想が外れたから必ず話す/当たったから必ず黙る訳ではない（感情/思考が高ぶる/安定するだけ）。surprise+感情+内容を**発話判定LLMが総合判断**。唯一の hard ゲートは `pending_obligation`（予約締切等の事実・将来 Call-Function）。**T2 death-detection は「surprise が必須引数として判定に効く配線」へ作り替え**（surprise を読む fake で振ると判定が反転・Optional 化禁止）。
- **発話判定ログ**: True/False とも `{ts,speak,reason,content}` を deque(10) で記録・**処理には関与しない**（観測専用）。
- **Q3 裁定（企画書どおり単純化）**: バックオフ/再挨拶抑制/沈黙カテゴリは**不採用**。モノローグ/再挨拶が出たら抑制で隠さず should_speak/文脈/feedback を直す。
- **ガード（loop・OS スレッド0・ロック0）**: 応答中(`runner.is_busy()`)/ユーザ発話中(`user_speaking`)/5秒未満 では発火しない。SpeechDecider は single-flight。
- **関わり**: `PredictionState.surprise`(読・単一生産者) / RAG(random) / StimulusQueue / ModelRegistry(role `speech_decide`)。

---

## G. FeedbackLLM（内分泌系・遅延許容・非同期）【実装済 2026-06-21・F4】
- **実装**: `eve/feedback/`（`prediction_state.py` / `parser.py` / `prompts.py` / `feedback_llm.py` / `worker.py`）。各応答後に `FeedbackWorker`（single-flight サイドカー）が非同期起動。出力 = summary / emotion / user-emotion / next-prediction / **prediction-diff(0-100)** / reason / tags。応答クリティカルパスに乗らない（トリガは O(1)・worker は別タスク）。
- **watermark/span 方式（ユーザ裁定）**: 入力は「前回フィードバック地点〜最新」のスパンを必ずカバー＝**未フィードバックの会話＝記憶喪失を作らない**。watermark は RAG 書込成功時のみ前進・起動時 catch-up・shutdown 未完は未前進で次回回収。
- **関わり**: `PredictionState`(diff=surprise をメソッド API で書込・単一書込) / RAG(`add_chunk` で 1チャンク=FB1+応答1・圧縮埋め込み/展開注入) / ContextAssembler(`last_feedback` を build 時に同期読み)。
- **F5 へ繰越**: 沈黙時の話題提案は発話判定LLM側へ。多生産者 `SurpriseBus`(FB diff + VLM screen-diff) は F5（F4 は単一生産者 `PredictionState`・surprise はメソッド API なので2生産者化で呼出側不変）。
- **PORT 方針の実際**: v1 `feedback_llm.py` の3段 fallback は**採用せず**（lean・サイドカーは失敗を no-op で許容し次 run で回復）。cache_control も現状不要。モデルは ModelRegistry role="feedback"（gpt-4o ハードコード/anthropic 既定は持ち込まない）。出力は JSON でなく**タグ付きテキスト**（小モデルの部分出力に頑健・parser は raise しない）。

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

- **タスク記録(memo)のフィールド**: `task_id` / `what`(やること) / **`success_criterion`(合格条件＝何を満たせば成功か)** / **`verdict_kind: deterministic|judged`(②判定をコードでやるかLLM審判か)** / `when`(予約時刻。null=応答後すぐ、「5分後」=now+300s) / `order` / **`depends_on:[id]` / `input_from:id`(P3 依存・親結果バインド)** / `status` / `result` / `attempts` / **`reflection`(失敗分析＝次の別アプローチの根拠, 親から持ち越す)** / `created_at`/`updated_at` / `parent_id`(再計画の親リンク)。
- **status**: `Pending → Running → (Done | Failed | Cancelled)`。terminal は**再遷移禁止をコードゲート**で（v1「完了所有権が二転三転」の直接修復）。**遷移は executor のみが所有**、LLM は create/cancel を*要求*するだけで `Done` を主張できない。
- **人間的ループ(ユーザの記述どおり)**:
  1. **列挙**: やることを順にタスク記録化（memo を書く）。各に任意で `when`。
  2. **時計を見る**: 周期スケジューラ(reconcile timer ~1s、沈黙中も動く＝v1 RC1 の良かった点)が「`scheduled_at<=now` の Pending」を探す。
  3. **時間が来たら実行**: due タスクを executor が非同期サイドカーで実行（= J-0 能力を呼ぶ）。`Pending→Running`。
  4. **確認**: 成功→`Done`+result を StimulusQueue へ（Eve が「終わったよ」と言える）。
  5. **失敗時の再計画**: 失敗→`Failed`(理由つき、terminal のまま固定)→ LLM が「なぜ失敗/次どうするか」を考え、**新しいフォローアップタスクを作る**(`parent_id` でリンク)。terminal-stays-terminal を守りつつユーザの「タスクを振り直す」を実現（復活させない＝v1 修復）。
- **永続化**: JSONL(async write queue)。起動時 reconcile で orphan な Running を age out。**dedup/dispatch ガード**で重複タスク→重複刺激を防ぐ。
- **捨てる**: in-memory only / subprocess 枠組み / task_packet・team・lane-board / DAG プランナ(CLAUDE.md 禁止) / messages インボックス。
- **役割を3つに厳密分離（誤解防止・最重要）**: 「完了判定=コード」は不正確。下記3つを混同しない（v1 はこれを混同して破綻）。
  - **① 状態の記録・所有権** = **常にコード**。status を書くのは executor だけ＝v1「完了が二転三転」の修復。これは"判定"ではなく"権限の一本化"。
  - **② 完了/失敗の判定（審判）** = **タスク種別で分岐**。機械的(DL成功/ファイル存在/exit0)=コードが照合。曖昧(検索が役立つ結果か/要約十分か)=**LLM審判**が結果を読み `{outcome: ok|weak|fail, reason}` を返す。どちらでも最後に①(コード)が status を書く。
  - **③ 失敗時に次どうするか（再計画）** = **常にLLM**。問題を分析し別アプローチを立案。コード単独では不可能（ユーザ指摘の通り）。
  - 汎用性の担保: タスク作成時に**合格条件(success criterion)も一緒に宣言**(CrewAI `expected_output` 相当)。条件がコードで測れれば②=コード、意味的なら②=LLM審判。「コードが万能判定」はしない。
- **「同じことを繰り返す」(v1失敗)の修復 = Reflexion パターン**: ループの原因は「失敗分析を次試行に渡さなかった」こと。修復: (1)失敗を**理由つきで具体記録**(timeout/結果なし/権限/本文取れず)、(2)LLMに**失敗理由つきで「別アプローチ」を反省立案**させ、(3)その反省を**新タスクに同梱して持ち越す**(コピーではない)、(4)新アプローチが前回と実質同一(同 what)なら**コードが却下**、(5)深さ上限/non-retryable で止める。→ 無限ループせず毎回違う手を打つ。
- **executor の正体**: コード(async関数)。due タスクを拾い→能力層を呼び→②の審判結果で①status確定→結果を StimulusQueue へ。コードは「記録係＋笛(停止条件・重複却下・非ブロッキング強制)＋機械的判定」、LLMは「曖昧判定＋失敗分析＋別アプローチ立案」。= 「制御フロー=コード、中身の判断=LLM」(OpenAI Agents SDK / LangGraph の主流)。
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
