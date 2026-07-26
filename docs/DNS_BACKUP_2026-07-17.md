# DNS 設定バックアップ（2026-07-17・J-2 search 調査前の状態）

アクティブアダプタ: Wi-Fi (InterfaceIndex=5)
AddressFamily=2 Servers=192.168.0.1
AddressFamily=23 Servers=2404:1a8:7f01:b::3,2404:1a8:7f01:a::3

## 復元コマンド（管理者 PowerShell）
`Set-DnsClientServerAddress -InterfaceIndex 5 -ResetServerAddresses`
（= DHCP 自動取得に戻す。上記 Servers が空/未表示なら元々 DHCP 自動）

## 変更コマンド（管理者 PowerShell・テスト時）
`Set-DnsClientServerAddress -InterfaceIndex 5 -ServerAddresses 1.1.1.1,8.8.8.8`
