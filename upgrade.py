"""
upgrade.py — 外部公開版ビルダー
====================================
GitHub Actions から呼び出すエントリーポイント。
update.py の全ロジックをそのまま使用しつつ、
PUBLIC_BUILD=True を設定することで
  - OPTIMIZATION タブの Configuration カード を非表示
  - 埋め込み JSON から "config" キーを除去
した docs/index.html を生成する。

【使い分け】
  GitHub Actions (外部公開) : python upgrade.py
  ローカル     (自分用確認) : python update.py  → ブラウザで docs/index.html を開く
"""

import update

# 公開ビルドモードを有効化
update.PUBLIC_BUILD = True

# update.py のメイン処理をそのまま実行
update.main()
