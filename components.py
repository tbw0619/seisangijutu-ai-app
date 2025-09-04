"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import os
import utils
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    回答モードを問い合わせモードに固定
    """
    # 問い合わせモードに固定
    st.session_state.mode = ct.ANSWER_MODE_2


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    # 初期メッセージの内容を定義
    initial_message = {
        "role": "assistant",
        "content": {
            "mode": "initial",
            "message": "こんにちは。私は生産技術の教科書・教材の情報をもとに回答する生成AIチャットボットです。画面下部のチャット欄からメッセージを送信してください。"
        }
    }
    
    # セッション状態に初期メッセージを追加（表示は display_conversation_log() で行う）
    st.session_state.messages.append(initial_message)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # LLMからの回答の場合
            else:
                # 初期メッセージの場合
                if message["content"]["mode"] == "initial":
                    st.markdown(message["content"]["message"])
                # 「問い合わせ」の場合の表示処理
                else:
                    # LLMからの回答を表示（LaTeX対応）
                    st.markdown(message["content"]["answer"], unsafe_allow_html=True)


def display_contact_llm_response(llm_response):
    """
    「問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # LaTeX数式の整形処理
    answer = utils.format_latex_equations(llm_response["answer"])
    
    # LLMからの回答を表示（unsafe_allow_htmlでLaTeX処理を有効化）
    st.markdown(answer, unsafe_allow_html=True)

    # 表示用の会話ログに格納するためのデータを用意
    content = {}
    content["mode"] = ct.ANSWER_MODE_2
    content["answer"] = answer

    return content


def display_faiss_initialization_sidebar():
    """
    FAISS-RAG初期化ボタンを表示
    
    Returns:
        bool: 初期化ボタンが押されたかどうか
    """
    # 要件チェック
    api_key_ok = bool(os.environ.get("OPENAI_API_KEY"))
    
    if not api_key_ok:
        st.error("OpenAI APIキーが設定されていません。")
        return False
    else:
        if not st.session_state.get('rag_initialized', False):
            return st.button("🚀 RAG機能を初期化", help="教科書・教材データベースを初期化")
        else:
            st.success("✅ RAG機能が有効です")
            if st.button("🔄 RAG機能を再初期化"):
                st.session_state.rag_initialized = False
                st.rerun()
            return False


def display_faiss_rag_status():
    """
    FAISS-RAG機能のステータス表示
    """
    if st.session_state.get('rag_initialized', False):
        chunks_count = len(st.session_state.get('pdf_chunks', []))
        st.success(f"✅ RAG機能が有効です（{chunks_count}チャンク）")
    else:
        st.info("⚡ RAG機能を初期化してから質問を開始してください。")


def display_faiss_search_results(search_results):
    """
    FAISS検索結果の詳細表示
    
    Args:
        search_results: FAISS検索結果のリスト
    """
    if not search_results:
        return
    
    with st.expander("🔍 検索結果詳細", expanded=False):
        for i, result in enumerate(search_results, 1):
            st.markdown(f"**検索結果 {i}**")
            st.markdown(f"- **類似度スコア**: {result['similarity_score']:.3f}")
            st.markdown(f"- **出典ファイル**: {result['metadata'].get('source_file', 'unknown')}")
            st.markdown(f"- **内容プレビュー**: {result['content'][:100]}...")
            st.markdown("---")