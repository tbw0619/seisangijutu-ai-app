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