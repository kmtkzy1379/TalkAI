# spike/ — P0 技術検証（**捨てるコード**）

`docs/UI_DESIGN.md` §7 P0 の判定に使う使い捨てスパイク。**本番には使わない**。
ここで得た結論だけを `docs/UI_DESIGN.md` に転記し、実装は P1 以降で本体側に書く。

## live2d_test.py

```powershell
$env:PYTHONIOENCODING="utf-8"

# ① 対話モード（ウィンドウが出る）
& .\.venv\Scripts\python.exe spike\live2d_test.py

# ② 自己検証（ウィンドウを一瞬出して PNG を書き、機械判定して終了。VOICEVOX 不要）
& .\.venv\Scripts\python.exe spike\live2d_test.py --selftest

# ③ リップシンクの品質確認（音を出さずに口の動きを数値と図で見る。VOICEVOX 必要）
& .\.venv\Scripts\python.exe spike\live2d_test.py --lipsync-check "こんにちは。今日はいい天気だね。"
```

対話モードでできること: テキストを VOICEVOX で喋らせて口を動かす / 疑似発話（音なしで口だけ） /
マウスで視線追従 / 口の開き・attack・release・音ズレ・拡大・上下をスライダで調整 /
SDK の自動まばたきの ON-OFF 比較。

**依存**（`.venv` に導入済み・2026-08-13）: `PySide6-Essentials` 6.11.1 / `live2d-py` 0.7.0.4。
本番へ持っていく際は `requirements-ui.txt` に分ける（`docs/UI_DESIGN.md` U8）。

## 実測結果（2026-08-13・自己検証 2回連続 5/5 PASS）

| 項目 | 結果 |
|---|---|
| ① モデル読み込み・描画 | **PASS**（Cubism Core 05.01.0000・`nun model.moc3`） |
| ② 3層合成（奥背景→Live2D→手前背景） | **PASS**。`QPainter` + `beginNativePainting()` で成立。`live2d.clearBuffer()` は呼ばない |
| ③ `ParamMouthOpenY` の外部駆動 | **PASS**（変化121px・重心(0.50,0.51)・面積比0.0001＝口の位置に局在） |
| ④ 描画レート | **61fps** / 1350×900px（論理900×600・**devicePixelRatio 1.50 で HiDPI が効いている**） |
| 参考: SDK 自動まばたき | **効かない**（6秒間 `ParamEyeLOpen` が 1.0 のまま）＝`model3.json` の EyeBlink グループが空という読みが実機で裏付けられた |

出力: `spike/out/mouth_closed.png` / `mouth_open.png` / `mouth_compare.png`（gitignore）。

### リップシンク: RMS 方式は不採用、**モーラ長方式**を採用

WAV の音量包絡（RMS）でやると **口が開きっぱなしになる**。日本語 TTS は母音が連続して有声のまま
繋がるため、モーラの切れ目が振幅の谷として現れない。パラメータを掃引しても改善しなかった:

| 方式 | 開口率 | 閉じ/秒 | 実モーラ速度 |
|---|---|---|---|
| RMS（窓・レンジ・release を 54通り掃引した最良） | 87% | **2.4** | 5.0 /秒 |
| **モーラ長（`audio_query` の consonant_length / vowel_length）** | 78% | **5.2** | 5.0 /秒 |

モーラ方式の調整値（4文で掃引・`CONSONANT_OPEN` / attack / release）:

| 子音の開き | attack | release | 開口率 | 閉じ/秒 |
|---|---|---|---|---|
| 0.15 | 20ms | 40ms | 76% | 4.6 |
| **0.08（採用）** | **20ms** | **40ms** | **72%** | **5.1** |
| 0.00 | 20ms | 40ms | 72% | 5.1 |
| 0.08 | 20ms | 30ms | 69% | 4.9（閉じすぎ） |
| 0.08 | 20ms | 55ms | 75% | 4.7（開きっぱなし寄り） |

母音別の**口の形**は作れない（`ParamA/I/U/E/O` がモデルに無い）が、母音別の**開き量**
（あ1.00 / え0.72 / お0.62 / い0.42 / う0.34 / ん0.18）で自然に見せている。

**本体側に要る改修は1つだけ**: `eve/response/tts.py` の `generate()` が WAV しか返さず
`audio_query` の結果を捨てている（`tts.py:47`）。`(wav, query)` を返すようにする。
