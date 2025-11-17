"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import pickle
import hashlib
import re
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# テキスト分割（最新版向け）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF読み込みとベクターストア関連
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores import FAISS
    VECTOR_SUPPORT = True
except ImportError as e:
    VECTOR_SUPPORT = False
    IMPORT_ERROR = str(e)

import constants as ct

# .env読み込み（最初に1回だけでOK）
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
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    return icon


def build_error_message(message):
    """
    画面上のエラーメッセージ＋管理者問い合わせテンプレを結合
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得（モードに応じてプロンプト切替）
    """
    llm = ChatOpenAI(
        model_name=ct.MODEL,
        temperature=ct.TEMPERATURE,
        max_tokens=ct.OPENAI_MAX_TOKENS
    )

    # 会話履歴から独立した質問テキストを作るプロンプト
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 回答用プロンプト（モードで切り替え）
    if st.session_state.mode == ct.ANSWER_MODE_1:
        system_prompt_for_answer = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        system_prompt_for_answer = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_for_answer),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 履歴を考慮したRetriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        st.session_state.retriever,
        question_generator_prompt,
    )

    # ドキュメントを詰めて回答させるチェーン
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)

    # Retrieval Chain（= RAG + 会話履歴）
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 実行
    llm_response = chain.invoke(
        {
            "input": chat_message,
            "chat_history": st.session_state.chat_history
        }
    )

    # 履歴を更新（人間→AIの順で保存）
    st.session_state.chat_history.extend([
        HumanMessage(content=chat_message),
        llm_response["answer"]
    ])

    return llm_response


def get_files_hash(file_paths):
    """
    ファイルリストのハッシュを生成（キャッシュファイル名用）
    """
    hash_obj = hashlib.md5()
    for file_path in sorted(file_paths):
        if os.path.exists(file_path):
            hash_obj.update(file_path.encode())
            mtime = str(os.path.getmtime(file_path))
            hash_obj.update(mtime.encode())
    return hash_obj.hexdigest()


def load_cached_rag_data(cache_dir, files_hash):
    """
    キャッシュされたRAGデータを読み込み
    """
    try:
        vectorstore_path = os.path.join(
            cache_dir,
            f"{files_hash}_{ct.VECTORSTORE_CACHE_FILE}"
        )
        chunks_path = os.path.join(
            cache_dir,
            f"{files_hash}_{ct.CHUNKS_CACHE_FILE}"
        )

        if os.path.exists(vectorstore_path) and os.path.exists(chunks_path):
            print("キャッシュされたRAGデータを読み込み中...")

            with open(vectorstore_path, "rb") as f:
                vectorstore = pickle.load(f)

            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)

            print("キャッシュからRAGデータを読み込みました。")
            return vectorstore, chunks
    except Exception as e:
        print(f"キャッシュ読み込みエラー: {str(e)}")

    return None, None


def save_rag_data_to_cache(vectorstore, chunks, cache_dir, files_hash):
    """
    RAGデータをキャッシュに保存
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)

        vectorstore_path = os.path.join(
            cache_dir,
            f"{files_hash}_{ct.VECTORSTORE_CACHE_FILE}"
        )
        chunks_path = os.path.join(
            cache_dir,
            f"{files_hash}_{ct.CHUNKS_CACHE_FILE}"
        )

        with open(vectorstore_path, "wb") as f:
            pickle.dump(vectorstore, f)

        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        print(f"RAGデータをキャッシュに保存しました: {cache_dir}")

    except Exception as e:
        print(f"キャッシュ保存エラー: {str(e)}")


def initialize_rag():
    """
    RAG機能の初期化（PDF読み込み→分割→FAISS→Retriever）
    キャッシュ対応あり
    """
    if not VECTOR_SUPPORT:
        raise ImportError(f"必要なライブラリがインストールされていません: {IMPORT_ERROR}")

    # 存在するファイルのみ抽出
    existing_files = [
        pdf_path for pdf_path in ct.PDF_FILES
        if os.path.exists(pdf_path)
    ]

    print(f"検索対象PDFファイル数: {len(ct.PDF_FILES)}")
    print(f"存在するPDFファイル数: {len(existing_files)}")

    if not existing_files:
        raise ValueError("利用可能なPDFファイルがありません。")

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI APIキーが設定されていません。")

    # ファイルハッシュでキャッシュキーを作る
    files_hash = get_files_hash(existing_files)

    # キャッシュ試行
    vectorstore, chunks = load_cached_rag_data(ct.RAG_CACHE_DIR, files_hash)

    if vectorstore is not None and chunks is not None:
        # キャッシュヒット
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_chunks = chunks
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": ct.SEARCH_K}
        )
        print("キャッシュからRAGデータを読み込みました。初期化をスキップします。")
    else:
        # キャッシュなし→新規構築
        print("キャッシュが見つからないため、RAGデータを新規作成します...")

        all_documents = []
        for pdf_path in existing_files:
            try:
                print(f"読み込み中: {pdf_path}")
                loader = PyMuPDFLoader(pdf_path)
                documents = loader.load()

                # メタデータにファイル名を付与
                for doc in documents:
                    doc.metadata["source_file"] = os.path.basename(pdf_path)

                all_documents.extend(documents)
                print(f"  → {len(documents)}ページ読み込み完了")

            except Exception as e:
                print(f"ファイル読み込みエラー {pdf_path}: {str(e)}")
                continue

        print(f"合計ドキュメント数: {len(all_documents)}")

        if not all_documents:
            raise ValueError("PDFファイルの読み込みに失敗しました。")

        # テキスト分割（RecursiveCharacterTextSplitterで統一）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ct.FAISS_CHUNK_SIZE,
            chunk_overlap=ct.FAISS_CHUNK_OVERLAP,
            separators=["\n"]
        )

        split_docs = text_splitter.split_documents(all_documents)

        # チャンク数を制限
        max_chunks = min(ct.FAISS_MAX_CHUNKS, len(split_docs))
        chunks = split_docs[:max_chunks]

        print(f"使用チャンク数: {len(chunks)} / {len(split_docs)}")

        # 埋め込みベクトル作成
        embeddings = OpenAIEmbeddings(
            model=ct.OPENAI_EMBEDDING_MODEL
        )

        print("埋め込みベクター作成中...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # キャッシュ保存
        save_rag_data_to_cache(vectorstore, chunks, ct.RAG_CACHE_DIR, files_hash)

        # セッションに格納
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_chunks = chunks
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": ct.SEARCH_K}
        )

    # 会話履歴がなければ初期化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # rag_initialized フラグはここで立てておくと安全
    st.session_state.rag_initialized = True

    return True


def convert_to_subscript(text):
    """
    数字をUnicode下付き文字に変換
    """
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    result = ''
    for char in text:
        result += subscript_map.get(char, char)
    return result


def format_latex_equations(text):
    """
    LaTeX数式を厳格なルールに従って$記号で囲む形式に変換
    """
    # LaTeX記号を厳格ルールに従って変換
    latex_conversions = {
        r'\\times': '×',
        r'\\cdot': '×',
        r'\\div': '÷',
        r'\\pm': '±',
        r'\\mp': '∓',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\propto': '∝',
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'\1/\2',
        r'\^\{2\}': '²',
        r'\^2': '²',
        r'\^\{3\}': '³',
        r'\^3': '³',
        r'\^\{4\}': '⁴',
        r'\^4': '⁴',
        r'\^\{([^}]+)\}': r'^(\1)',
        r'\^([0-9]+)': r'^\1',
        r'\\Omega': 'Ω',
        r'\\omega': 'ω',
        r'\\pi': 'π',
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\delta': 'δ',
        r'\\theta': 'θ',
        r'\\lambda': 'λ',
        r'\\mu': 'μ',
        r'\\sum': 'Σ',
        r'\\int': '∫',
        r'\\sqrt\{([^}]+)\}': r'√(\1)',
        r'\\dots': '…',
        r'\\ldots': '…',
        r'\\text\{([^}]+)\}': r'\1',
        r'\\mathrm\{([^}]+)\}': r'\1',
    }

    def convert_math(match):
        math_content = match.group(1) if match.group(1) else match.group(2)

        # LaTeX記号を順次変換
        for latex_pattern, replacement in latex_conversions.items():
            math_content = re.sub(latex_pattern, replacement, math_content)

        # 下付き
        math_content = re.sub(
            r'([A-Za-z])_\{([^}]+)\}',
            lambda m: m.group(1) + convert_to_subscript(m.group(2)),
            math_content
        )
        math_content = re.sub(
            r'([A-Za-z])_([0-9]+)',
            lambda m: m.group(1) + convert_to_subscript(m.group(2)),
            math_content
        )

        return f"${math_content}$"

    # 【...】→数式
    text = re.sub(r'【([^】]+)】', convert_math, text)

    # $$...$$
    text = re.sub(r'\$\$\s*([^$]+?)\s*\$\$', convert_math, text)
    # $...$
    text = re.sub(r'(?<!\$)\$\s*([^$]+?)\s*\$(?!\$)', convert_math, text)
    # [ ... ]
    text = re.sub(r'\[\s*([^\[\]]+?)\s*\]', convert_math, text)

    # $$ が重なってしまった場合の掃除
    text = re.sub(r'\$\$+', '$', text)

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
        llm = ChatOpenAI(
            model_name=ct.MODEL,
            temperature=ct.TEMPERATURE,
            max_tokens=ct.OPENAI_MAX_TOKENS
        )

        question_generator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_INQUIRY),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            st.session_state.retriever,
            question_generator_prompt,
        )

        question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = st.session_state.get('chat_history', [])
        llm_response = chain.invoke(
            {"input": user_input, "chat_history": chat_history}
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.extend([
            HumanMessage(content=user_input),
            llm_response["answer"]
        ])

        return {
            "answer": llm_response["answer"],
            "source_documents": llm_response.get("context", [])
        }

    except Exception as e:
        return {
            "answer": f"回答生成中にエラーが発生しました: {str(e)}",
            "source_documents": []
        }


def get_rag_chain_answer_qa_streaming(user_input):
    """
    RAGチェーンを使った問い合わせ回答の取得（ストリーミング）
    呼び出し側は `for chunk in get_rag_chain_answer_qa_streaming(...):` で順次書き出す想定
    """
    if not st.session_state.get('rag_initialized', False) or st.session_state.get('retriever') is None:
        def error_generator():
            yield {"content": "RAG機能が初期化されていません。サイドバーの「🚀 RAG機能を初期化」ボタンをクリックしてください。"}
        return error_generator()

    try:
        llm = ChatOpenAI(
            model_name=ct.MODEL,
            temperature=ct.TEMPERATURE,
            max_tokens=ct.OPENAI_MAX_TOKENS,
            streaming=True
        )

        question_generator_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_INQUIRY),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            st.session_state.retriever,
            question_generator_prompt,
        )

        question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = st.session_state.get('chat_history', [])

        full_answer = ""
        for chunk in chain.stream({"input": user_input, "chat_history": chat_history}):
            if "answer" in chunk:
                content = chunk["answer"]
                full_answer += content
                yield {"content": content}

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.extend([
            HumanMessage(content=user_input),
            full_answer
        ])

    except Exception as e:
        def error_generator():
            yield {"content": f"回答生成中にエラーが発生しました: {str(e)}"}
        return error_generator()
