# 生産技術授業支援アプリ

## 概要
このアプリは、生産技術の教科書・教材を活用したRAG（Retrieval-Augmented Generation）ベースのAI検索・問い合わせシステムです。FAISS（Facebook AI Similarity Search）を使用して高速なセマンティック検索を実現し、OpenAI GPTと連携して学生向けに分かりやすい回答を生成します。

## 主な技術スタック

- **Streamlit**: WebUIフレームワーク + LaTeX数式表示
- **LangChain**: RAGパイプライン構築
- **FAISS**: ベクトルデータベース（高速セマンティック検索）
- **OpenAI API**: GPT-4o（物理計算・数式処理特化）、text-embedding-3-small（埋め込み）
- **PyMuPDF**: PDF文書処理

## 機能

1. **教科書検索**: 入力キーワードと関連性が高い教科書・教材の内容を高速検索
2. **AI問い合わせ**: 生産技術に関する質問に対して、GPT-4oが工業高校生向けに分かりやすく回答
3. **物理計算対応**: オームの法則、キルヒホッフの法則などの計算問題を詳細解説
4. **数式表示**: LaTeX記法による美しい数式レンダリング

## 対応教材
- 生産技術教科書（PDF形式）
- 授業テキスト（DOCX形式）
- 副教材（PDF形式）

## セットアップ

### 1. 必要なライブラリをインストール:
```bash
pip install -r requirements.txt
```

### 2. Streamlit Community Cloudでのデプロイ

#### 必要なファイル
- `main.py`: メインアプリケーション
- `requirements_deploy.txt`: 軽量版依存関係リスト
- `.streamlit/config.toml`: Streamlit設定
- `data/`: 教科書・教材データ

#### デプロイ手順
1. GitHubリポジトリにコードをpush
2. Streamlit Community Cloudにアクセス
3. GitHubアカウントでログイン
4. 新しいアプリを作成し、リポジトリを選択
5. シークレット設定で `OPENAI_API_KEY` を追加
6. `requirements_deploy.txt` を使用してデプロイ

#### 環境変数設定
Streamlit Community Cloudのダッシュボードで以下を設定:
```
OPENAI_API_KEY = "your_openai_api_key_here"
```
```

### 2. 環境変数の設定:
```bash
# .envファイルを作成してOpenAI APIキーを設定
echo OPENAI_API_KEY=your_actual_api_key_here > .env
```

### 3. アプリの起動:
```bash
# メインアプリの起動
streamlit run integrated_app.py
```

## アーキテクチャ

### ファイル構成
- **`integrated_app.py`**: メインアプリケーション（統合版RAG）
- **`constants.py`**: 設定値・定数の一元管理
- **`components.py`**: UI表示コンポーネント
- **`utils.py`**: ユーティリティ関数
- **`data/`**: 教科書・教材データフォルダ
- **`logs/`**: アプリケーションログ

### RAGワークフロー
1. PDF文書を800文字のチャンクに分割
2. OpenAI text-embedding-3-smallで埋め込みベクトルを生成
3. FAISSでベクトルデータベースを構築
4. ユーザークエリをベクトル化して類似チャンクを検索
5. GPT-3.5-turboで検索結果をもとに回答を生成

## 環境設定の詳細

### 環境変数

- `OPENAI_API_KEY`: OpenAI APIキー（必須）

### 注意事項

このアプリは初回起動時にPDF文書の解析とベクトルデータベースの構築を行うため、少し時間がかかる場合があります。

## Web公開について

このアプリはStreamlitで構築されており、Streamlit Cloud、Heroku、AWSなどのプラットフォームでWeb公開が可能です。

### Streamlit Cloudでの公開手順

1. GitHubにコードをプッシュ
2. Streamlit Cloudでリポジトリを連携
3. 環境変数（OpenAI APIキー）を設定
4. デプロイ

学生や教員が簡単にアクセスできるWebアプリとして活用できます。