# Web Deployment Configuration for 生産技術授業支援アプリ

# Streamlit Cloud用
[build-system]
requires = ["streamlit>=1.28.0"]

# Heroku用のProcfile内容
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0

# 環境変数の説明
# OPENAI_API_KEY: OpenAI APIキー（必須）
# USER_AGENT: ユーザーエージェント（自動設定）
