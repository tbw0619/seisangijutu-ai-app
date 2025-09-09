"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
# OS機能を使用するためのモジュール
import os
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from app_init import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME
)

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    initialize()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 4. 初期表示
############################################################
# タイトル表示
st.title("生産技術授業支援アプリ")

# サイドバー
with st.sidebar:
    st.header("利用方法")
    
    # 「問い合わせ」の説明と例
    st.info("生産技術に関する質問に対して、教科書・教材の情報をもとに回答を得られます。")
    st.markdown("**[入力例]**")
    st.code("キルヒホッフの法則の計算方法を詳しく教えて", language=None)
    st.code("オームの法則について詳しく説明して", language=None)
    
    # RAG初期化ボタン
    st.markdown("---")
    st.markdown("**🧠 RAG機能の初期化**")
    
    # 要件チェック
    api_key_ok = bool(os.environ.get("OPENAI_API_KEY"))
    
    if not api_key_ok:
        st.error("OpenAI APIキーが設定されていません。")
    else:
        if not st.session_state.get('rag_initialized', False):
            if st.button("🚀 RAG機能を初期化", help="教科書・教材データベースを初期化"):
                with st.spinner("📚 教科書・教材データを読み込み中..."):
                    try:
                        # RAG初期化処理
                        utils.initialize_rag()
                        st.session_state.rag_initialized = True
                        st.success("✅ RAG機能が初期化されました！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"初期化エラー: {e}")
        else:
            st.success("✅ RAG機能が有効です")
            if st.button("🔄 RAG機能を再初期化"):
                st.session_state.rag_initialized = False
                st.rerun()

# モードは常に「問い合わせ」に固定
st.session_state.mode = ct.ANSWER_MODE_2

# 初期メッセージ表示（messagesが空の場合のみ）
if len(st.session_state.messages) == 0:
    cn.display_initial_ai_message()

# 初期表示用の警告メッセージ（初期メッセージのみの場合に表示）
if len(st.session_state.messages) == 1 and st.session_state.messages[0]["content"]["mode"] == "initial":
    st.warning("具体的に入力したほうが明確な回答を得やすいです。", icon=":material/warning:")


############################################################
# 5. 会話ログの表示
############################################################
try:
    # 会話ログの表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()


############################################################
# 6. チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得（ストリーミング）
    # ==========================================
    with st.chat_message("assistant"):
        try:
            # ストリーミング表示用のコンテナを作成
            response_container = st.empty()
            full_response = ""
            formatted_response = ""
            
            # RAG機能を使ってストリーミング回答を取得
            llm_response_stream = utils.get_rag_chain_answer_qa_streaming(chat_message)
            
            # ストリーミングレスポンスを逐次表示
            for chunk in llm_response_stream:
                if chunk and "content" in chunk:
                    full_response += chunk["content"]
                    # LaTeX数式の整形処理
                    formatted_response = utils.format_latex_equations(full_response)
                    response_container.markdown(formatted_response)
            
            # ストリーミングが空の場合のフォールバック
            if not formatted_response:
                formatted_response = utils.format_latex_equations(full_response)
            
            # 最終的なレスポンスの整形
            content = {
                "mode": ct.ANSWER_MODE_2,
                "answer": formatted_response
            }
            
            # AIメッセージのログ出力
            logger.info({"message": content, "application_mode": st.session_state.mode})
            
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})