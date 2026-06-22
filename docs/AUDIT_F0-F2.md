# F0–F2 監査記録（4エージェント並列調査 + 一次ソース確認）

> **⚠ ステータス更新 (2026-06-20)**: 本書は 2026-06-17 のスナップショット。その後 F2.5/F3/F3.5 を実装し
> **A1/A2/A3・B0–B4・C1・C4・C5・D2 は解消済**。未解消: **C2(barge-in 二箇所)・C3(StreamFn 不整合)**。
> D1(_reserve 無制限増加) は 2026-06-20 修正。テストは現在 94件。最新の正は `docs/HANDOFF.md` とコード。

> 2026-06-17。4観点（並行処理 / アーキテクチャ / 仕様・プラン整合 / QA・統合）で並列監査し、各所見を
> **実コード行で本人確認**して集約。決定論テストは 46/46 緑（F0:15 / F1:13 / F2:18）。
> **方針（ユーザ指示）**: 修正は今やらず **F2.5 でまとめて実装**。例外処理の過剰実装でコードを汚さない。
> システムロジック上の重要度で優先。きれいなコードを保つ。プランは必要なら修正。
> 凡例 確認: 🔴複数エージェント+本人確認 / 🟠1エージェント+本人確認 / ⚪本人確認のみ。

## P0 — 正しさの急所（F2.5 で必ず直す。例外乱用ではなくロジック修正）

| # | 問題 | 該当 | 確認 | 直し方（最小・きれい） |
|---|---|---|---|---|
| **A1** | **TTS が None→以降の文が永久停止（head-of-line デッドロック）**。`_emit` が TTS 前に seq 予約→None だと enqueue されず、`_drain_buffer` は連続 seq でしか進まない。1文目 None で**ターン丸ごと無音**。VOICEVOX は非200/例外で None を返す | `response/orchestrator.py:_emit/_tts_and_enqueue`, `pipeline/audio_play_queue.py:_drain_buffer` | 🟠QA再現+本人 | **seq連続性のロジック修正**: None 時も seq を前進（番兵 enqueue or worker が放棄 seq をスキップ）。try/except ではない |
| **A2** | **PipelineRunner.run が一般例外でループ即死**。`CancelledError` しか拾わず、応答中の例外で単一 consumer が永久停止→全刺激が drain されない | `pipeline/orchestrator.py:62-67` | 🔴2+本人 | 1箇所だけ `except Exception: log; continue`（CancelledError の break は残す）。単一consumerの生存条件であって防御過多ではない |
| **A3** | **stream 中の LLM 例外が handle() を貫通**。`async for delta` に保護なし→A2 と組んでループ死 | `response/orchestrator.py:75-86` | 🟠QA再現+本人 | **この1境界だけ** try/except（log→flush 残り→last_response 確定→return）。他所には足さない |

## P1 — F2.5 を成立させる前提（声で会話する以上 必須）

| # | 問題 | 確認 | 対応（F2.5） |
|---|---|---|---|
| **B0** | **STT 幻聴＝v1 の本丸**。Whisper 系が無音/クリック音を「ご視聴ありがとうございました」等の**危険な定型句**に誤認（字幕学習由来）。**実測(2026-06-18)**: 無音→ Groq whisper-large-v3 と OpenAI whisper-1 は確実に「ご視聴ありがとうございました」、**gpt-4o-transcribe は出さない**。**実録音(ユーザのクリック/タイピング, 2026-06-18)でさらに明確**: gpt-4o-transcribe は**無音・クリックで空文字 `''`**（幻聴ゼロ）、タイピングで `[drumming]`/`Thank you.` 程度（危険な定型句なし・捨てやすい）。Groq/whisper-1 は全入力で危険な日本語定型句。**→ STT = gpt-4o-transcribe 採用確定** | 🔴実録音+本人 | **2点に簡素化**（Whisper固有の松葉杖=no_speech_prob閾値/幻聴ブラックリストは廃止）: ①**エンジンを gpt-4o-transcribe に交換**（OPENAIキー保有・定型句を出さない・`prompt`で固有語注入）②**VADは既存(v1 audio_input)をPORT＆整理するだけ**（ユーザ満足: 息継ぎ/終端認識OK、割り込み時のターン統合機構も流用）。再設計しない。VADは endpointing＋"非音声をSTTに渡さない源ゲート"の確実な仕事に限定（"ユーザー声判別"ではない）＋最小長0.3s。partial投機はやらない（精度優先）。**差し替えは `transcribe(bytes)->str` の中身1メソッドのみ**。実機A/Bで最終確認 |
| ~~**B1**~~ | ~~自己エコー(AEC)~~ → **ユーザ裁定(2026-06-18): ソフトAECは作らない**。**イヤホン前提（AIの声がmicに入らない）＋常時リッスン**（mic を塞がない＝発話中もユーザーを聞く）。ソフトで声を聞き分けるAECは複雑/危険で不採用、イヤホンが古典的だがきれい・安全 | ユーザ裁定 | barge-in は「ユーザーが本当に話し出したら Eve が今の発話を止め譲る」だけ実装。**割り込み位置のメタ情報はLLMに注入しない**（毎回割り込みの話をする副作用回避）。将来スピーカ運用に戻すならソフトAECを再検討 |
| **B2** | **StimulusQueue が別スレッドから put 不可**（asyncio.Condition束縛）。mic/VAD は別スレッド。`AudioPlayQueue` は `set_loop`/cross-thread 済だが Queue は非対称 | 🔴2+本人 | ~~`set_loop()` + `run_coroutine_threadsafe(queue.put(s), loop)` 経路を追加~~ → **【RESOLVED 2026-06-21・P2裁定a】単一ループ採用で本経路は不要。mic/STT は MicSttInputSource でループ上動作＝別スレッド put は発生しない。`set_loop` は削除。将来 OS スレッドは `PIPELINE_DESIGN.md §9.3` の queue→loop drain 経由** |
| **B3** | **mid-sentence barge-in 不可**（今鳴ってる文は鳴り切る）。`player.py` が世代を見ない（docstring で F2.5 明記済の意図的スコープ） | 🔴2+本人 | `play_fn` のチャンクループに `if gen != current: break` を1行（停止粒度~数十ms）。`_play` に generation を渡す配線 |
| **B4** | **cold-start ウォームアップが設計にも本番コードにも無い**（初回 TTFT~4s、計測ツールが暗黙に捨てているだけ） | 🟠1+本人 | Start シーケンス（VALIDATING→RUNNING 間）で各役へ throwaway 1回 + aiohttp/VOICEVOX 接続事前確立 |

## P2 — 実装が増える前に整える構造（きれいさ・将来交差点）

| # | 問題 | 確認 | 対応 |
|---|---|---|---|
| **C1** | **ResponseOrchestrator が stimulus.kind を無視し user_text しか配線しない**。全 kind が「ユーザ発話(今)」として描画され、RAG/feedback/vision/直近ターンが応答LLMに届かない＝**T6 過去参照防止の器が E2E で死蔵**。F3 が全部ここに集中 | 🔴2+本人 | `stimulus→assemble引数` の薄い変換層（kind 分岐）+ コンテキスト源は注入(provider)で orchestrator が源を直接知らない形に。**F2.5/F3 着手前に開くのが最も安い** |
| **C2** | **barge-in 発火が runner と AudioPlayQueue に二分**。runner は USER 発話で**無条件**世代+1（再生中でなくても） | 🔴2+本人 | 発火判断を AudioPlayQueue 1箇所に寄せ「再生中 AND ユーザ発話」に。runner は通知のみ |
| **C3** | **`StreamFn` 型が2箇所で別定義・不整合**（model_registry: model=,messages= / orchestrator: messages のみ）。realcheck が毎回 adapter 手書き | 🟠1+本人 | 名前を分ける or Orchestrator が `ModelRegistry + role` を受け内部で `registry.stream(role, msgs)`（role 切替=response/autonomous/youtube も一元化） |
| **C4** | **多ターン記憶（ConversationCache）が空白**。`Turn` 型はあるが貯める器・add_turn・5件window・100件保持が未実装 | 🟠1+本人 | COMPONENT_LOGIC に ConversationCache 節新設（後続フィーチャ）。下記プラン矛盾も解消 |
| **C5** | **記憶は「実際に喋った文」だけ記録すべき**（生成≠発話）。実機ログ(2026-06-19)で確認: 🤖 ログ/`last_response` は**生成時点**で全文出るが TTS/再生は遅く、barge-in で実発話は途中で切れる（例「club」7文生成・実発話1-2文）。`last_response`(生成済み全文) を記憶に入れると喋ってない文まで残る | 🔴実機+本人 | C1/ConversationCache 実装時: **再生レイヤが「実際に再生し終えた文」を報告 → 記憶はそれを使う**。AudioPlayQueue の項目に文テキストを持たせ、play 完了時に spoken として記録。ついでに 🤖 表示も再生時点に移すと表示が正直になる。今は記録のみ（ユーザ「まだ対処不要」） |

## P3 — 軽微 / 自明 / テスト

| # | 問題 | 確認 | 対応 |
|---|---|---|---|
| D1 | `AudioPlayQueue._reserve` dict が世代ごとに増え続ける（実害極小） | 🔴3+本人 | `bump_generation` で古い世代を prune（1行）or 現世代単一カウンタに簡素化 |
| D2 | `VoicevoxTTS.close`/`RealAudioPlayer.close` を本番経路で呼ぶ所有者が無い（ツールのみ） | 🟠1+本人 | Start/Stop host が STOPPING で close を所有 |
| D3 | **H7=真**: フルループ(StimulusQueue→PipelineRunner→実 ResponseOrchestrator→audio)が**自動テスト皆無**（realcheck は handle 直叩き） | 🔴2+本人 | F2.5 で決定論 E2E を追加（単発/連続2ターン/barge-in 変種、provider差替=T5 も注入で安価に） |

## 非問題（過剰対応しないと明記）
- TTS/player/play_worker の broad-except = CLAUDE.md「落とさない」通り。**追加不要**（例外処理を増やさない）。
- StimulusQueue 自前実装 / clock 二重時刻 / splitter / sanitize / DI = 妥当・きれい。**触らない**（YAGNI）。
- 記号だけの文の seq 欠落 = 非バグ（`_emit` が reserve_seq 前に return）。seq 欠落の危険は A1（TTS None）に固有。
- 空 stream / 空応答 = 正しく処理（クラッシュ/`join` ハングなし）。
- surprise/should_speak が F0–F2 に無いこと自体 = 正しい（F3 スコープ）。ただし C1 と H2(下記)の継ぎ目は今裁定。

## プラン（docs）の修正が要る点
- **PIPELINE_DESIGN §0 と §9 の surprise 運搬形式**: 「§0 必須引数(Optional禁止)」と「§9 SurpriseBus loop所有・同期読み」は**矛盾ではなく両立**で確定 → 「**bus が保持し、判定関数(should_speak/自己懐疑/handle)は bus から read した値を必須引数で受ける**」。death-detection(T2) は引数強制で担保。F3 着手前に §0/§9/COMPONENT_LOGIC D,E に明記。**`handle` シグネチャは F3 で surprise 必須引数を取る前提**を今 docs に固定（後付け破壊を防ぐ）。
- **直近ターン数 5 vs 100 の食い違いは矛盾でない**: **ConversationCache は ~100件保持、応答LLM へは直近5ターンを供給**。COMPONENT_LOGIC に両方を明記。
- **T2/T4/T6 と SurpriseBus は未実装**であることを docs に明示（「F2 緑＝中核原理 enforced」ではない）。
- PIPELINE_DESIGN §3 に cold-start 行、§10 に warmup ステップ（B4）。§6 ROLE_DEFAULTS に `summarize` 追記（実装先行分）。
- **F2.5 プランに織り込む**: A1/A2/A3（堅牢化・最小・済）, **B0 STT交換(gpt-4o-transcribe)＋VAD源ゲート(最優先)**, B2 thread-safe put, B3 mid-sentence（=割り込み時にEveが譲る）, B4 warmup, C1 文脈配線, D3 E2E テスト。**B1 ソフトAECは不採用（イヤホン前提＋常時リッスン）**。STT は「**silero-VAD区間→gpt-4o-transcribe→final→刺激（投機なし）**」で精度優先、partial 投機はやらない。STTは `Stt` 抽象で差し替え可能に（実機A/Bで最終モデル確定）。

## F2.5 着手前に直すと最も安い Top3（構造）
1. **C1**: ResponseOrchestrator の kind 分岐 + 文脈源の注入（F2.5/F3 がここに集中する前に開く）。
2. **A1 + A2 + A3**: 応答経路の「黙る/固まる」3欠陥を最小修正（ロジック修正 + 2つの load-bearing try/except のみ。防御過多にしない）。
3. **B0 STT交換(gpt-4o-transcribe)＋VAD源ゲート**: v1 の最大の不満点。実測でエンジン交換だけでは非音声ゴミが残るが gpt-4o-transcribe は危険な定型句を出さない→VADで源を断てば実用十分。ソフトAECは不採用（イヤホンで物理解決）。
