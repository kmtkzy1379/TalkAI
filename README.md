# TalkAI

**画面を見ながら声で会話する、常駐型の AI 対話システム。**
マイク入力 → VAD → STT → 応答LLM → 文分割 → 音声合成 → 順次再生を asyncio の単一ループで回し、
内省（FeedbackLLM）・長期記憶（RAG）・画面認識（VLM）・予約タスク・Web検索を横に足したもの。

CLI 常駐プロセスです（**GUI は未実装**）。キャラクター名は「イブ」。

> ⚠ **プライバシー**: `VLM_ENABLED=1` にすると**実際のデスクトップ画面が Gemini API に送信されます**。
> 機密情報が映る環境では有効化しないでください。

---

## 何が面白いか（設計の核）

**1. 予測誤差(surprise)を一級の信号として扱い、装飾化しない**

前作では FEP（自由エネルギー原理）風の内省を実装したものの、出力が応答を駆動しない「飾り」になりました。
本作では発話判定 `should_speak(surprise: int, ...)` の **surprise を `Optional` にしない**（型で強制）。
さらに **surprise の閾値比較コードを一切書かない**（`eve/` 全体で `surprise >` 等のヒットは0件）。
数値で機械的に発話を決めるのではなく、判定LLM に「指標」として渡して総合判断させます。
surprise を反転させると発話/沈黙が反転する death-detection テストが回帰として入っています。

**2. プロンプトで守られない規律は、判定後のコードで止める**

「〜しないでください」と書いても LLM は破ります。そこで実測で壊れた挙動は**コードゲート**にしました。
同内容抑制（bigram 0.25 + 埋め込み cos 0.87・600秒窓）／記憶の自己参照除外／話題の丸投げ抑制／
時制の混同検出／既出用件のブロック／取消済み報告の墓標（tombstone）など。
各ゲートの実装位置は `docs/COMPONENT_LOGIC.md` にあります。

**3. 閾値は実測で校正し、根拠を日付つきでコードに残す**

例（`eve/config.py` / `eve/speech/decider.py` のコメントより）:

- 同内容判定の埋め込み閾値 **0.87** — 実発話8文・Ruri v3 で「同話題 0.909-0.932 / 別話題 0.780-0.834」に分離
- 時制ゲート **900秒** — 直近指示語を含む発話の経過が「正当=106秒以下 / 誤り=10日以上」に完全分離し、間ならどこでも精度1.00

**4. 実機で壊れた事象を、対照実験つきで潰す**

実機 E2E で見つかった欠陥は、修正前後を比較できる形（対照アーム）で検証しています。
例: 「起動直後に前セッションの未応答依頼を勝手に実行する」問題は、
文脈への注記では**8/8 で再発**したため、復元会話を注入しない設計に変更して **0/8** になりました。

---

## 構成

```
マイク → VAD(Silero) → STT ─┐
画面 → キャプチャ → VLM ────┼→ StimulusQueue ─→ 応答LLM ─→ 文分割 ─→ TTS ─→ AudioPlayQueue → 再生
沈黙タイマー → 発話判定LLM ─┘        （priority+merge）              （seq/generation で順序と barge-in）
                                          ↑
              FeedbackLLM（内省）→ RAG 書込 + surprise ─┘
```

- **2キュー分離**: 応答の起動刺激を1本化する `StimulusQueue` と、できた音声を順に鳴らす `AudioPlayQueue`
- **エッジだけストリーミング**: 入力は VAD 区間ごとの確定 STT、出力は token stream → 文分割 → 逐次 TTS。応答LLM はターン制
- **ModelRegistry**: role → model の間接層（provider 非依存）。全10 role
- **スレッドモデル**: asyncio 単一ループ所有・単一書込。真の OS スレッドは `vlm-capture` の**1本だけ**

詳細は `docs/PIPELINE_DESIGN.md`（配線契約）と `docs/COMPONENT_LOGIC.md`（各コンポーネントの判断ロジック）。

---

## 実装済み

| 機能 | 実体 |
|---|---|
| 音声対話ループ（mic→VAD→STT→応答→TTS→再生） | `eve/voice_loop.py`, `eve/response/` |
| barge-in（ユーザ発話開始で即中断・世代管理） | `eve/pipeline/audio_play_queue.py` |
| 短期記憶（JSONL・注入セレクション） | `eve/memory/conversation_cache.py` |
| 長期記憶 RAG（Ruri v3 ローカル埋め込み + MMR） | `eve/memory/long_term.py` |
| 内省 FeedbackLLM（→RAG 書込 + surprise 生成） | `eve/feedback/` |
| 自発発話（沈黙時に判定LLM + 話題の種 + コードゲート群） | `eve/speech/` |
| 画面認識 VLM（capture → pHash 変化ゲート → Gemini Flash） | `eve/vlm/` |
| Call-Function 能力層（read-only） | `eve/capability/` |
| 予約タスク / TaskAgent / 取消解決 | `eve/task/` |
| Web検索（ddgs スニペット + 本文取得の深掘り） | `eve/search/` |
| 報告の配達確認 / barge-in 後の再配達 | `eve/response/delivery_checker.py` |

**実行できる能力は6つ**: `self_status` / `pc_status`（常時）、
`delegate_task` / `list_tasks` / `cancel_task`（`TASK_ENABLED=1`）、
`search_web`（`SEARCH_ENABLED=1` かつ `TASK_ENABLED=1` かつ ddgs 導入時）。

## 未実装

UI（Tkinter 参照は0件）/ VTube Studio・Live2D 連動（設定値の定義のみ）/ YouTube 配信モード（同）/
画面操作系（window_op・launch_app 等）/ **文脈不整合の自己懐疑**（surprise の消費者は発話判定のみ）/
`Config.validate()` の呼び出し（起動前チェックは存在しない）。

**作らないと確定したもの**: STT の partial 投機 / AEC（イヤホン前提）/ VLM の複数回 self-consistency。

---

## セットアップ

前提: **Python 3.13** / Windows / マイク + **イヤホン**（AEC 不採用のためスピーカーだと自分の声を拾う）

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# VOICEVOX を起動しておく（http://127.0.0.1:50021）。未起動でも落ちないが声が出ない
copy .env.example .env    # API キーを入れる
```

> **`.env` の機能フラグは必ず書く。** コード既定は全て off なので、
> `CALLFUNCTION_ENABLED` / `TASK_ENABLED` / `SEARCH_ENABLED` / `VLM_ENABLED` を
> `.env` に書かないとタスク・検索・画面認識が丸ごと無効のまま起動します
> （これを1か月見逃した事故が実際にありました）。

```powershell
$env:PYTHONIOENCODING = "utf-8"   # 日本語+絵文字を出すため必須
.\.venv\Scripts\python.exe tools\voice_chat.py
```

起動ログに出る `Call-Function 稼働` / `タスク管理 稼働` / `Web検索 稼働` / `画面認識(VLM) 稼働` の行で
実際に有効な機能を確認できます。**この行が出ていない機能は動いていません。**

初回のみネットワークからダウンロードします: Silero VAD（torch.hub, MIT）/ Ruri v3 埋め込み（HuggingFace, Apache-2.0）。

---

## テスト

**Tier-1 決定論テスト（API・ネットワーク不要）**: 18ファイル・**434 assertion**・約11秒。
pytest は使っておらず、各ファイルが独立スクリプトです。

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe tests\test_f5_speech.py     # 1ファイルずつ実行
```

**実機 E2E**（`tools/search_e2e_test.py`・全20シナリオ）は本番の `VoiceLoop` をそのまま起動し、
マイクだけスタブ化して合成音声→実 STT→刺激投入で実機と同じイベント順を再現します。

> ⚠ **実 API 課金が発生し、実際の画面をキャプチャし、メモ帳を自動で開閉します。**

---

## 実測（2026-07-30 時点）

通し E2E ×3回 pooled（n=92）の発話までの時間: **中央値 3.14s / 平均 3.29s / 最大 9.70s**（3秒超 54%）。
理想は ≤3s、運用許容は ≤5s としていますが、**現状は 10/92 が 5秒を超えており未達**です。
応答モデルの品質を優先した結果で、原因の切り分けは未着手（既知の課題として `docs/HANDOFF.md` に記載）。

---

## このリポジトリについて

前作（筆者自身が約1年半かけて作った AI VTuber）を、パッチ継続せず**設計から作り直した2作目**です。
`docs/SALVAGE_MANIFEST.md` は**その前作を筆者自身がファイル単位で解体レビューした記録**で、
「価値創出コードの約85%が作り直し対象」という判定に至った経緯が残っています。
他人のコードを批評した文書ではありません。

ドキュメントは AI エージェントとの協働を前提に書かれており、`CLAUDE.md` は
エージェント向けの作業規範です。コミットには `Co-Authored-By: Claude` が付いています。

**ドキュメントと実装が食い違う場合はコードが正**です。
`docs/PIPELINE_DESIGN.md` と `docs/COMPONENT_LOGIC.md` の冒頭には、
本文の記述と実コードの差分を列挙した訂正表を置いています。

---

## ライセンス / クレジット

MIT License（`LICENSE`）。

- 音声合成に **VOICEVOX** を使用しています。生成音声を公開する場合は各キャラクターの利用規約に従ってください
- 埋め込みモデル: [cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)（Apache-2.0）
- VAD: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)（MIT）
