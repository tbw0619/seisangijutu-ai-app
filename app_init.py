"""
アプリケーション初期化処理
"""

import streamlit as st
from dotenv import load_dotenv
import constants as ct


def initialize():
    """
    アプリケーションの初期化処理
    """
    # 環境変数の読み込み
    load_dotenv()
    
    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []
    
    if "mode" not in st.session_state:
        st.session_state.mode = ct.ANSWER_MODE_2  # 問い合わせモードに固定
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
