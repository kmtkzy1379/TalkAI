# Eve v2 — 統合パイプライン設計（契約ドキュメント）

> ## ⚠ 本書の読み方（2026-07-29 更新・**コードが正**）
>
> 本書は「**守るべき不変条件の契約**」。実装状況・手順・既知問題は書かない（→ `docs/HANDOFF.md` が単一情報源）。
> 本文には設計当初（2026-06）の記述が残っている箇所があるため、**下の訂正表が本文より優先する**。
>
> | 本文の記述 | 実際（コード） |
> |---|---|
> | STT partial 投機 | **不採用**。VAD区間 → final STT |
> | ソフト AEC / 自己エコー除去 | **不採用**（イヤホン前提） |
> | RAG 300件 / 直近5ターン | **500件 / 直近6ターン**（`eve/config.py`） |
> | 埋め込みを ModelRegistry role で解決 | `make_embedder(ruri\|openai)`（別系統・`eve/memory/embed/`） |
> | `SurpriseBus` が集約する | **そのクラスは無い**。`PredictionState` が2生産者を **most-recent-source-wins** で合成（`eve/feedback/prediction_state.py`） |
> | `SurpriseBus` が drain 優先度を補正 | **しない**。`StimulusQueue` は kind + aging のみ（`eve/pipeline/stimulus_queue.py`） |
> | `vision_update` 刺激が流れる | **producer ゼロ**（予約 kind）。VLM は `SpeechDecider.trigger("vlm")` を叩く（`eve/vlm/worker.py`） |
> | VLM ×3 self-consistency（§8） | **廃止**。単発・複数フレーム VLM（`eve/vlm/narrator.py`） |
> | OS スレッドは現状ゼロ（§9） | **1本稼働**（`vlm-capture`・`eve/vlm/capture_thread.py`） |
> | 橋渡しの既定は手段3(queue)（§9.3） | 実装は **手段1 `call_soon_threadsafe`**（変化ゲート後の低頻度フレームのみ渡すため許容） |
> | ModelRegistry は6 role | **10 role**（`task` `summarize` `search_summarize` `delivery_check` を追加。`vlm_merge` と `youtube` は定義のみで未使用） |
> | 発話判定の入力は「ランダムRAG2」 | `autonomous_memories(k=3)` ＝ **関連1 + 完全ランダム1 + 重要度1**（`eve/memory/long_term.py`） |
> | RAGチャンク = フィードバック1 + 応答1 | 実際は **FeedbackLLM 出力のみ**（応答本文は含まない・`eve/feedback/feedback_llm.py`） |
> | T7 = E2E 4シナリオ | E2E ハーネスは **20シナリオ**（`tools/search_e2e_test.py` の `SCENARIOS`）。T1-T7 の ID は一部だけテスト側に実在する（T1/T3/T7=`tests/test_f1_pipeline.py` / T2=`tests/test_f5_speech.py` / T6=`tests/test_f0_foundation.py`）。**T4・T5 の ID は存在しない**（barge-in は T3 と `tests/test_callfunction_phase1.py` で検証） |
>
> **中核原理の現状**: surprise の消費者は `should_speak` の1本のみ。**(b) 文脈不整合の自己懐疑は未実装**（§0 の「2点をゲートする」は設計意図であって現状ではない）。

> 本書は新企画書を契約として、**統合パイプラインの骨格**を定義する（実装ステップ②）。
> 個別機能の深掘りは後続（③優先順位 → ④機能別プランニング）。本書は「まず流れとレイテンシを通す」ための最小骨格と、その不変条件を確定する。
> 出典: 3並列設計エージェント（A=トポロジ / B=状態・並行 / C=レッドチーム）の提案を統合。意見が割れた点は §8 で裁定（多数決/全会一致ルール）。

---

## 0. 中核原理（組織原理）

**予測誤差(surprise)をリアルタイム一級信号にする。** v1 では FEP が feedback ループ内で完結し応答 behavior を駆動しなかった（症状の根）。v2 では surprise は次の2点を**必須引数として**ゲートする:

1. **自発発話の要否/内容** — 沈黙時に「何を/話すべきか」を surprise が決める。
2. **文脈不整合の自己懐疑** — タスク未完・無視・文脈破綻を「疑問に思わず解答だけ返す」v1 の欠陥を、surprise を見て自己点検することで塞ぐ。

設計判断: **surprise は `Optional` にしない。** `should_speak(...)` と応答オーケストレータの自己懐疑フックは surprise を必須引数で受ける。surprise を反転させたら発話/沈黙の判定が反転する「death detection テスト」(§7 T2) が通らなければビルド失敗とする。＝装飾化の再発を型とテストで防ぐ。

---

## 1. 全体トポロジ

```
                         ┌─────────────────────────────────────────────┐
 入力源                  │              StimulusQueue (単一)              │
 ─────                   │   priority + merge ルールで drain (§4)         │
  mic ──VAD/STT(増分)──▶ │  ・user_utterance (最高)                       │
  YouTube ─────────────▶ │  ・callfunction_result                         │
  silence(5s)──────────▶ │  ・autonomous_speech ("…" 判定経由)            │
  screen(変化時)────────▶│  ・vision_update                               │
                         └───────────────────┬──────────────────────────┘
                                             │ drain(1件 or merge)
                                             ▼
                           ┌──────────────────────────────────┐
        SurpriseBus  ────▶ │      ResponseOrchestrator         │ ◀── ModelRegistry(role→model)
       (loop所有/同期読み) │  文脈組立→応答LLM stream→文分割    │ ◀── ContextAssembler
                           └───────┬───────────────┬──────────┘
                                   │ token stream  │ CallFunction 抽出
                                   ▼               ▼
                       文単位 split          FunctionDispatcher
                            │                (応答完了後に実行 →
                            ▼                 結果を StimulusQueue へ再投入)
                    ┌──────────────┐
                    │ TTS(VOICEVOX)│  文ごとに合成
                    └──────┬───────┘
                           ▼
                  ┌──────────────────┐
                  │  AudioPlayQueue   │  seq+generation トークンで順序保証/barge-in
                  │ (PORT player.py)  │
                  └──────┬───────────┘
                         ▼
                      再生(順番通り)

 非同期サイドカー（応答クリティカルパス外）:
  FeedbackLLM(内分泌系/遅延許容) ── emotion/summary/user-emotion/next-prediction/diff(0-100)
        └─ prediction-diff(0-100) ─▶ SurpriseBus.current
  VLM(capture→Gemini Flash ×3→統合) ── 画面ナレーション ─▶ vision_update stimulus + screen-diff ─▶ SurpriseBus
  RAG(FB1+応答1 を1チャンク, 300件) ── ContextAssembler から検索
```

**設計の心臓部はこの2キュー分離**:
- **StimulusQueue** = 「次に応答LLMを起動する刺激」を1本化。優先度とマージで「沈黙/空入力時は最速の刺激を、busy 中は溜める、複数 CallFunction は逐次、複数 vision/feedback はマージ」という spec のパイプライン規則をここに集約。
- **AudioPlayQueue** = 「できた音声から順に流す/応答中は溜める/割り込みで捨てる」。`seq`(単調増加) と `generation`(発話世代) の2トークンで順序と barge-in を1箇所が権威的に管理。

---

## 2. 役割別 LLM の入力契約（spec 準拠）

| 役割 | 起動契機 | 入力 | 出力 | モデル(既定/代用) |
|---|---|---|---|---|
| **応答LLM** (神経系/即時) | user発話 / CallFunction結果 / 自発発話True / vision | 発話 + systemプロンプト + 直近5ターン + RAG2 + 直近feedback1 + 画面認識(統合済) + 発話判定理由 + CallFunction定義 | token stream（文に分割）+ CallFunction呼び出し | Claude Sonnet → **代用: GPT/Gemini**(§6) |
| **発話判定LLM** | 5s 沈黙 | "…" + 直近会話 + ランダムRAG2 + 画面認識 | True(理由+応答LLMへの入力) / False(理由→ログのみ) | 軽量・高速モデル(クリティカルパス外) |
| **FeedbackLLM** (内分泌系/遅延許容) | 各応答後(非同期) | 直近会話 + 応答 | emotion/summary/user-emotion推定/next-prediction/diff(0-100)、沈黙時は話題提案 | 中位モデル |
| **VLM 統合** | 画面変化時 | スクショ→Gemini Flash 2.5 ×3 の結果 | 統合ナレーション(自然言語) | 統合: より賢いモデル(§8で裁定) |
| **YouTubeモード** | YTコメント(上から1件ずつ) | YTプロンプト + コメント。リアルタイムなし、CallFunctionはtaskのみ | 応答 | **Opus 4.8**(強モデル) |

**過去参照の防止（Msg6 / spec「タイムスタンプで過去を話してないか」）**: 全ターン・全RAGチャンクに ISO-8601 + monotonic の二重タイムスタンプを付け、ContextAssembler は組み立て時に各要素へ相対時刻(「3分前」等)を明示注入する。RAG は意味+連想検索だが、**現在会話との関連を ContextAssembler が必須化**（無言時の random RAG2 は "話題の種" と明示ラベルし「思い出話」と峻別）。これにより v1 の「過去の記憶から話が逸れる」を構造的に抑止。テストは §7 T6。

---

## 3. レイテンシ予算（応答 ≤3s、本フェーズはパイプライン検証なので 3s 基準）

| 段 | 予算 | 備考 |
|---|---|---|
| VAD エンドポイント検出 | 200–400ms | 発話終端の確定 |
| STT 最終確定 | 0–150ms | **増分(partial)で投機開始**し最終待ちでブロックしない(C の最重要指摘) |
| 文脈組立(ContextAssembler) | 50–150ms | RAG/feedback/vision は事前計算・キャッシュ参照 |
| 応答LLM TTFT | 600–1200ms | **ボトルネック**。ここを縮めるのが最優先 |
| 1文目切り出し | 100–400ms | 句点/読点での split |
| TTS 1文目合成 | 300–600ms | VOICEVOX audio_query→synthesis |
| **合計(最初の音声まで)** | **≈1.5–2.9s** | ≤3s に収まる。精度優先時のみ ≤5s 許容 |

クリティカルパス短縮の3原則: ①STT は partial で投機開始（最終確定で破棄可能に）。②RAG/feedback/vision は**応答前に非同期で準備済み**にし組立時はキャッシュ参照。③TTS は文単位パイプライン（全文を待たない）。

---

## 4. StimulusQueue drain ルール（spec パイプライン規則の集約）

優先度（高→低）: `user_utterance` > `callfunction_result` > `autonomous_speech` > `vision_update`。

| 状況(spec) | ルール |
|---|---|
| 沈黙・空入力 | 最速の刺激(発話判定 or vision)を投入。発話判定 False なら何も出さずログのみ |
| 応答中(busy) | 新規刺激は drain せず保持。割り込み発話(barge-in)は別扱い(§5) |
| CallFunction 2件以上 | **逐次**実行。並列にしない(spec明示) |
| vision/feedback 2件以上 | **マージ**して1刺激に畳む(spec明示) |
| barge-in(ユーザーが応答中に発話) | 現 generation を cancel → AudioPlayQueue flush → 新 user_utterance を最優先 drain |

---

## 5. 順序保証と barge-in（C の単一権威指摘）

AudioPlayQueue だけが `seq`/`generation` を発行・管理する単一権威:
- `seq`: 1発話内の文順。合成は並列でも再生は seq 順。
- `generation`: 発話の世代。barge-in/新ターンで +1。**旧 generation の合成結果は再生前に破棄**（player.py の interrupt_async を seq/generation 対応に拡張）。
- **自己エコー対策(AEC)**: mic が自分の TTS 出力を拾って STT→barge-in する自己割り込みを、再生中フラグ + エコーキャンセルで弾く（C 指摘の見落としやすい SPOF）。

---

## 6. ModelRegistry（provider 非依存・Claude 代用）

役割→モデルの間接参照を1箇所に集約（B 提案）。`.env` で既定を与え、UI から swap/temp 可変。**Claude API は現在停止中**のため、Sonnet 役は GPT/Gemini に代用しつつ、後で Claude に戻せるよう role→model の間接層を必須とする。

```
ROLE_DEFAULTS = {
  "response":      env("RESPONSE_MODEL",  "<claude-sonnet→代用: gpt/gemini>"),
  "speech_decide": env("DECIDE_MODEL",    "<軽量高速>"),
  "feedback":      env("FEEDBACK_MODEL",  "<中位>"),
  "vlm_leaf":      env("VLM_LEAF_MODEL",  "gemini-flash-2.5"),  # ×3
  "vlm_merge":     env("VLM_MERGE_MODEL", "<§8裁定>"),
  "youtube":       env("YOUTUBE_MODEL",   "<opus-4.8 相当>"),
}
```
litellm 等で provider 差を吸収。`feedback_llm.py` の3段 fallback と cache_control はこの層に一般化して PORT（旧コードの gpt-4o ハードコード/anthropic 既定は撤去）。

---

## 7. テスト戦略（決定論で配線を証明 + Msg6 の2追加）

ディシプリン: 各機能テストは**2回連続成功で合格**、最終マージは**5回連続成功**。3+エージェントで議論、割れたら多数決→再調査、全会一致で実装。

| ID | 目的 | 方法（API不要・決定論を基本） |
|---|---|---|
| **T1** レイテンシ配線 | 並列で動く(直列でない)ことを証明 | 各段を固定遅延スタブにし、合計が「直列和」でなく「並列の最大経路」になることを assert。≤3s を数値で検証 |
> ※実装（`tests/test_f1_pipeline.py`）が assert するのは「サイドカーが応答経路を塞がない」「並列合成が直列和より短い」のみで、**≤3s の数値検証は無い**（レイテンシは E2E 実測が正）。
| **T2** surprise death detection | surprise が本当に駆動しているか | surprise を反転 → `should_speak`/自己懐疑の出力が反転することを assert。反転しなければ**ビルド失敗**(装飾化の再発防止) |
| **T3** 順序/世代 | 音声が順番通り・barge-inで旧世代破棄 | seq シャッフル投入→再生は seq 順 / generation+1 後に旧 seq が再生されないこと |
| **T4** barge-in | 割り込みで現発話が即停止 | 再生中に user_utterance 投入→AudioPlayQueue flush + 新世代開始を assert。※AEC は**不採用**（イヤホン前提・冒頭の訂正表参照） |
| **T5** provider 代用 | Claude 停止でも動く | ModelRegistry の response 役を GPT/Gemini に差替えても全テストが緑 |
| **T6** ⭐過去参照なし(Msg6) | 「過去のことを話してない」 | タイムスタンプ付きの混在文脈(古い記憶+新会話)を与え、応答が**現在文脈に接地**し古い話に逸れないことを検証。無言時 random RAG は "話題の種" 扱いで「思い出話」にならないこと |
| **T7** ⭐パイプライン破綻なし(Msg6) | E2E でエラー/明らかな遅延がない | 4シナリオ(§下)を端から端まで流し、例外0・キュー詰まり0・予算内・順序保持を一括検証 |

**移植必須**: v1 `tools/` Tier-1/2/3 の実機事故シナリオを**回帰仕様として最初に移植**（コードでなく仕様として）。同じ穴に落ちない唯一の保険。

### E2E シナリオ（T7 が流す4本）
1. **通常応答**: user発話→STT(partial投機)→応答stream→文分割→TTS→順次再生（≤3s）。
2. **沈黙→自発発話**: 5s沈黙→発話判定LLM(True)→surprise参照で話題選択→応答。Falseなら無音+ログのみ。
3. **CallFunction**: 応答中に task/search/screen-op 呼出→応答完了後に逐次実行→結果を刺激として再投入→次応答。
4. **barge-in**: 応答再生中に user 割り込み→現 generation cancel→AudioPlayQueue flush→新ターン最優先。

---

## 8. エージェント分岐の裁定（多数決/全会一致ルール適用）

| 論点 | 裁定 | 根拠 |
|---|---|---|
| VLM ×3 の意味 | **同一フレーム×3の self-consistency**（多数派+spec「ハルシネ削減」目的に整合） | 3枚の異なるスクショ並列はナレーション分散を招く。同一フレームを3回読み統合で誤読を平均化 → YOLO起因ハルシネの代替目的に合致 |
| VLM 統合モデル provider | **暫定 Gemini Pro/Flash 系で統合**（Claude停止のため）、ModelRegistry で後日 Claude/GPT に差替可 | 葉が Gemini Flash 2.5 指定。同系統で統合しレイテンシ/整合を優先。§6 の間接層で可搬 |
| vts.py lip-sync IPC | **当面 autonomous-only を維持**（IPC追加しない） | vts.py は自己完結確認済(別プロセス・入力なし)。lip-sync連動は ④ の独立機能として後で評価。今は配線を増やさない |
| VAD 終端 vs 5s沈黙タイマー | **VAD=ターン終端確定**、**5s沈黙=発話判定トリガ**として役割分離 | 二重発火を避ける。VADは「ユーザーが話し終えた」検出、沈黙タイマーは「誰も話さない時間」検出で別軸 |
| CallFunction 実行タイミング | **応答stream完了後に逐次実行**し結果を刺激として再投入 | spec「応答の後に実行/できたら刺激としてフィードバック」を直訳。応答のレイテンシを汚さない |

> いずれも実装着手前に再確認可能な「暫定裁定」。④で各機能を開く際に一次データで再検証する。

---

## 9. スレッド/プロセスモデル（loop 単一所有を正とする・将来契約／P2 裁定 a・2026-06-21）

> **核不変条件**: 共有状態は **loop 所有・単一書込者・同期読み・ロックなし**。
> StimulusQueue / AudioPlayQueue(seq+generation 権威) / ConversationCache / RagStore /
> （F4〜）PredictionState・（F5〜）SurpriseBus はすべて asyncio loop が唯一の書込者。
> 永続化など I/O のみ background worker に逃がす（state は触らず値を返すだけ）。
> 旧 `AudioPlayQueue.set_loop`+cross-thread `interrupt` 分岐は**削除済**（呼び出し0の死にコード）。

### 9.1 実行単位の分類（4種）

| 分類 | 定義 | 例 |
|---|---|---|
| **(i) 単一 asyncio loop** | パイプライン本体。共有状態を所有・書込・読込 | StimulusQueue / Orchestrator / AudioPlayQueue / SurpriseBus / 全サイドカー(F4 FeedbackLLM・F5 発話判定・VLM 推論/統合・task executor・search・screen-op の制御部) / VAD 推論(現状) |
| **(ii) executor offload** | ブロッキング呼び出しを `run_in_executor`/`asyncio.to_thread` で逃がす。**loop 所有状態に一切触れず、値を返すだけ**。返り値は await したコルーチンが loop 上で適用する | mic read / file I/O(cache・long_term) / embeddings(ruri/openai) / STT(openai/groq) / TTS 再生 / VLM capture(mss) / search fetch / screen-op argv 実行 |
| **(iii) 真の OS スレッド** | 連続リアルタイム capture が強制する時のみ生成。**loop 所有状態を直接 mutate 禁止**。§9.3 の橋渡し契約で loop に渡す | （現状ゼロ）将来候補=VLM 連続 capture(N fps)・audio callback(PortAudio thread) |
| **(iv) 別プロセス** | 完全分離・IPC なし | `vts.py`(VTube Studio 制御) |

UI(Tkinter) は別途メインスレッド。widget 更新は `root.after(0, fn)`、UI→loop は `run_coroutine_threadsafe`。

### 9.2 将来プロデューサの分類（既定=async-task+executor、真スレッドは最後の手段）

| プロデューサ | 分類 | 根拠 |
|---|---|---|
| FeedbackLLM(F4) / 発話判定(F5) / VLM 推論×3+統合 / task executor / search / screen-op | (i) loop サイドカー（§9.4） | いずれも awaitable。新 OS スレッド不要。ブロッキング部分のみ (ii) へ |
| VLM capture | (ii) 既定 / (iii) 連続撮影時のみ | 単発トリガは `to_thread`。N fps 連続が実測で必要なら capture スレッド(iii)へ昇格 |
| audio callback(将来) | (iii) | PortAudio が own thread で callback→唯一の本物の cross-thread。§9.3 の最初の実利用者 |
| VAD 推論 | (i) ループ上同期（**現状維持**） | 軽量(Silero・数ms)。executor 化は audio callback API 移行時にまとめて（先回りしない） |

**真の OS スレッドを足してよいのは「連続リアルタイム capture が他の手段で成立しない」と実測で示せた時だけ。**
推測で先回りスレッドを作らない（v1 の VLM 3スレッド肥大の再発防止）。

### 9.3 橋渡し契約（OS スレッド → loop）

不変条件:
> **OS スレッドは loop 所有状態（StimulusQueue / AudioPlayQueue / PredictionState / SurpriseBus / caches）を直接 mutate しない。** 必ず下記3手段のいずれかで loop に渡す。

| 手段 | API | 用途 |
|---|---|---|
| **1. fire-and-forget の state poke** | `loop.call_soon_threadsafe(fn)` | 戻り値不要の即時通知（barge-in の世代+1 等） |
| **2. awaitable work の実行** | `asyncio.run_coroutine_threadsafe(coro, loop)` | 結果が要る稀ケースのみ。基本使わない |
| **3. データ受け渡し（既定・推奨）** | thread-safe queue に push → loop task が `await q.get()` で drain | 連続データ(capture frame / audio chunk)はこれ |

**barge-in の原則的置換**（削除した `set_loop`+cross-thread `interrupt` の正しい後継）:
将来 audio が callback ベース(PortAudio thread)に移っても、callback スレッドは **生 PCM を queue に push するだけ**（手段3）。VAD 終端判定・STT enqueue・barge-in 判断（再生中 AND ユーザ発話 → `bump_generation`）は**すべて loop 上の drain task** が行う。即時停止が要るなら手段1で `call_soon_threadsafe(bump_generation)`。
**スレッドが `bump_generation` / queue.put を直接呼ぶ設計は禁止**（= 削除した旧 cross-thread 分岐。スレッド安全性をスレッド側で再獲得しようとせず、loop に集約して獲得する）。

### 9.4 サイドカー契約（FeedbackLLM=F4 を一般化・VLM/search/task が従う）

応答クリティカルパス外で loop 所有状態を更新する非同期 worker の共通形。

| 要素 | 規則 |
|---|---|
| **単一書込** | loop 所有状態(PredictionState/SurpriseBus/RAG)はサイドカー worker からのみ書く → ロック不要 |
| **single-flight** | worker は同時1件（per-turn `create_task` 禁止）。前job 進行中の新着はトリガで待つ |
| **off-critical-path トリガ** | 起動側は dirty フラグ + `asyncio.Event.set()` のみ（O(1)・非ブロッキング）。bounded queue の `await put` 禁止（満杯で応答経路がブロック） |
| **背圧2方式** | ①**latest-wins**（lossy・最新のみ重要な VLM screen-diff 等）／②**watermark/span-coverage**（no-skip・蓄積が必要な FeedbackLLM 等＝未処理を作らない・起動時 catch-up・記憶喪失を防ぐ） |
| **lifecycle** | `VoiceLoop.run` で `create_task` 起動 / `stop` で進行中を drain → cancel。state 前進は永続化成功時のみ（落ちても次回 catch-up が回収） |
| **off-critical-path** | 応答 stream / TTS / 再生を絶対ブロックしない。失敗は logger + no-op フォールバック（落とさない） |

実装 precedent: `ConversationCache._write_worker` / `RagStore._write_worker`（loop 所有 background task を queue で駆動する既存形）。サイドカーはこれに **トリガ + 背圧方式 + state 単一書込** を足したもの。

### 9.5 真スレッド昇格条件（チェックリスト）

新 OS スレッドを足す前に全部 Yes を確認:
1. その仕事は **連続・リアルタイム capture**（撮影/録音の定常ストリーム）か？
2. `to_thread`/`run_in_executor`(単発 offload) では**フレーム落ち/レイテンシが実測で不足**するか？
3. スレッドは **生データを queue に push するだけ**で、loop 所有状態に触れないか？（§9.3 手段3）
4. lifecycle（start/stop・drain・例外時の安全停止）を VoiceLoop が所有できるか？
いずれか No → サイドカー(i)+executor(ii) で実装する。

---

## 10. Start/Stop 安全ゲート（UI 完全作り直し・spec）

状態機械 `STOPPED → VALIDATING → RUNNING → STOPPING`。Start は**全設定が valid な時のみ**有効化（同期バリデータ）。RUNNING 中は設定ロック。UI は topmost、画面外で押せなくなる v1 不便を解消。各LLMの swap/temp/prompt編集/ログ/モード/VLM on-off を備える。

---

## 次のステップ
- ③ 各機能の実装優先順位（本骨格上で「まず流れとレイテンシを通す最小実装」順）を確定。
- ④ 機能別 just-in-time プランニング → branch 切って実装 → T1–T7 を 2回連続合格 → 統合は5回連続。
