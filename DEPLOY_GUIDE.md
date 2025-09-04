# Streamlit Community Cloud デプロイガイド

## 前提条件
- GitHubアカウント
- OpenAI APIキー

## デプロイ手順

### 1. GitHubリポジトリの準備
1. GitHubに新しいリポジトリを作成
2. このプロジェクトのファイルをpush

### 2. Streamlit Community Cloudでのデプロイ
1. [Streamlit Community Cloud](https://share.streamlit.io/) にアクセス
2. GitHubアカウントでサインイン
3. "New app" をクリック
4. GitHubリポジトリを選択
5. デプロイ設定:
   - Main file path: `main.py`
   - Requirements file: `requirements.txt` (自動検出)

### 3. 環境変数（Secrets）の設定
1. アプリのダッシュボードで "Settings" をクリック
2. "Secrets" セクションで以下を追加:
```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```
3. "Save" をクリック

### 4. デプロイ完了
- アプリが自動的にビルド・デプロイされます
- 数分でアクセス可能になります

## 注意事項

### ファイルサイズ制限
- Streamlit Community Cloudは1GBのリソース制限があります
- 大きなPDFファイルがある場合、一部のファイルを除外する必要があります

### 教科書データの調整
現在25個のPDFファイルを読み込む設定になっています。メモリ制限に達する場合は、constants.pyのPDF_FILESを調整してください:

```python
PDF_FILES = [
    "./data/教科書データ/313生シ_1_1.pdf",
    "./data/教科書データ/313生シ_1_2.pdf",
    # 必要に応じてファイル数を削減
]
```

### メモリ最適化
- FAISS_MAX_CHUNKSを500以下に設定することを推奨
- チャンクサイズを大きくしてチャンク数を削減

## トラブルシューティング

### メモリエラーの場合
1. constants.pyでFAISS_MAX_CHUNKSを削減
2. PDF_FILESから一部のファイルを除外
3. FAISS_CHUNK_SIZEを1000に増加

### API制限の場合
- OpenAI APIの利用制限を確認
- アプリの同時利用者数を制限

## 本番運用の推奨設定

```python
# constants.py での推奨設定
FAISS_CHUNK_SIZE = 1000
FAISS_MAX_CHUNKS = 300
PDF_FILES = [
    # 主要なファイルのみを選択（10-15個程度）
]
```

これにより安定したクラウドデプロイが可能になります。
