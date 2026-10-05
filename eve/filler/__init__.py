"""応答非依存フィラー（S2・S3）。

- `audio`:  音声の下ごしらえ（刈り込み・伸ばし棒・並べる）。試聴ファイルと本番で共通
- `bank`:   バンク（`bank/<voice>/manifest.json`）と `fillers.yaml` を読み、抽選する
- `player`: ユーザの話し終わりに鳴らす常駐の再生係（`AudioPlayQueue` の外で鳴らしてプレイヤーを共有）
"""
