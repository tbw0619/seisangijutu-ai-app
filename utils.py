"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
import constants as ct

# PDF処理とベクターストアのためのインポート
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores import FAISS
    VECTOR_SUPPORT = True
except ImportError as e:
    VECTOR_SUPPORT = False
    IMPORT_ERROR = str(e)


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response


def initialize_rag():
    """
    RAG機能の初期化
    """
    if not VECTOR_SUPPORT:
        raise ImportError(f"必要なライブラリがインストールされていません: {IMPORT_ERROR}")
    
    # 存在するファイルのみを選択
    existing_files = [pdf_path for pdf_path in ct.PDF_FILES if os.path.exists(pdf_path)]
    
    print(f"検索対象PDFファイル数: {len(ct.PDF_FILES)}")
    print(f"存在するPDFファイル数: {len(existing_files)}")
    
    if not existing_files:
        raise ValueError("利用可能なPDFファイルがありません。")
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI APIキーが設定されていません。")
    
    # 全PDFファイルの読み込み
    all_documents = []
    for pdf_path in existing_files:
        try:
            print(f"読み込み中: {pdf_path}")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            
            # メタデータにファイル名を追加
            for doc in documents:
                doc.metadata['source_file'] = os.path.basename(pdf_path)
            
            all_documents.extend(documents)
            print(f"  → {len(documents)}ページ読み込み完了")
            
        except Exception as e:
            print(f"ファイル読み込みエラー {pdf_path}: {str(e)}")
            continue
    
    print(f"合計ドキュメント数: {len(all_documents)}")
    
    if not all_documents:
        raise ValueError("PDFファイルの読み込みに失敗しました。")
    
    # テキスト分割
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.FAISS_CHUNK_SIZE,
        chunk_overlap=ct.FAISS_CHUNK_OVERLAP,
        separator="\n"
    )
    
    # ドキュメント分割
    split_docs = text_splitter.split_documents(all_documents)
    
    # チャンク数制限
    max_chunks = min(ct.FAISS_MAX_CHUNKS, len(split_docs))
    test_chunks = split_docs[:max_chunks]
    
    # 埋め込みベクター作成
    embeddings = OpenAIEmbeddings(
        model=ct.OPENAI_EMBEDDING_MODEL
    )
    
    # FAISSベクターストア作成
    vectorstore = FAISS.from_documents(test_chunks, embeddings)
    
    # セッション状態に保存
    st.session_state.vectorstore = vectorstore
    st.session_state.pdf_chunks = test_chunks
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": ct.SEARCH_K})
    
    # 会話履歴の初期化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    return True


def format_latex_equations(text):
    """
    LaTeX数式の整形処理
    """
    import re
    
    # [ ] で囲まれた数式を $$ $$ に変換
    text = re.sub(r'\[\s*([^\[\]]+?)\s*\]', r'$$\1$$', text)
    
    # 単一の$で囲まれた数式を$$に変換（既に$$で囲まれていない場合のみ）
    text = re.sub(r'(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)', r'$$\1$$', text)
    
    # 数式内のアンダースコアを適切に処理（I_1 → I_{1}）
    text = re.sub(r'\$\$([^$]*?)([A-Za-z])_([0-9]+)([^$]*?)\$\$', r'$$\1\2_{\3}\4$$', text)
    
    # 複数の連続する$$を整理
    text = re.sub(r'\$\$\s*\$\$', r'$$', text)
    
    # 数式の前後に適切な改行とスペースを追加
    text = re.sub(r'\$\$([^$]+?)\$\$', r'\n\n$$\1$$\n\n', text)
    
    return text


def get_rag_chain_answer_qa(user_input):
    """
    RAGチェーンを使った問い合わせ回答の取得
    """
    if not st.session_state.get('rag_initialized', False) or st.session_state.get('retriever') is None:
        return {
            "answer": "RAG機能が初期化されていません。サイドバーの「🚀 RAG機能を初期化」ボタンをクリックしてください。",
            "source_documents": []
        }
    
    try:
        # LLMのオブジェクトを用意
        llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)

        # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
        question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
        question_generator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question_generator_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # 問い合わせ用のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question_answer_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
        history_aware_retriever = create_history_aware_retriever(
            llm, st.session_state.retriever, question_generator_prompt
        )

        # LLMから回答を取得する用のChainを作成
        question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
        # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # LLMへのリクエストとレスポンス取得
        chat_history = st.session_state.get('chat_history', [])
        llm_response = chain.invoke({"input": user_input, "chat_history": chat_history})
        
        # LLMレスポンスを会話履歴に追加
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.extend([HumanMessage(content=user_input), llm_response["answer"]])

        return {
            "answer": llm_response["answer"],
            "source_documents": llm_response.get("context", [])
        }
    
    except Exception as e:
        return {
            "answer": f"回答生成中にエラーが発生しました: {str(e)}",
            "source_documents": []
        }