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
    import pickle
    import hashlib
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
    # LLMのオブジェクトを用意（コスト最適化）
    llm = ChatOpenAI(
        model_name=ct.MODEL, 
        temperature=ct.TEMPERATURE,
        max_tokens=ct.OPENAI_MAX_TOKENS  # トークン数制限でコスト削減
    )

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


def get_files_hash(file_paths):
    """
    ファイルリストのハッシュを生成（キャッシュファイル名用）
    """
    hash_obj = hashlib.md5()
    for file_path in sorted(file_paths):
        if os.path.exists(file_path):
            hash_obj.update(file_path.encode())
            # ファイルの更新日時もハッシュに含める
            mtime = str(os.path.getmtime(file_path))
            hash_obj.update(mtime.encode())
    return hash_obj.hexdigest()


def load_cached_rag_data(cache_dir, files_hash):
    """
    キャッシュされたRAGデータを読み込み
    """
    try:
        vectorstore_path = os.path.join(cache_dir, f"{files_hash}_{ct.VECTORSTORE_CACHE_FILE}")
        chunks_path = os.path.join(cache_dir, f"{files_hash}_{ct.CHUNKS_CACHE_FILE}")
        
        if os.path.exists(vectorstore_path) and os.path.exists(chunks_path):
            print("キャッシュされたRAGデータを読み込み中...")
            
            # ベクターストアを読み込み
            with open(vectorstore_path, 'rb') as f:
                vectorstore = pickle.load(f)
            
            # チャンクデータを読み込み
            with open(chunks_path, 'rb') as f:
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
        # キャッシュディレクトリを作成
        os.makedirs(cache_dir, exist_ok=True)
        
        vectorstore_path = os.path.join(cache_dir, f"{files_hash}_{ct.VECTORSTORE_CACHE_FILE}")
        chunks_path = os.path.join(cache_dir, f"{files_hash}_{ct.CHUNKS_CACHE_FILE}")
        
        # ベクターストアを保存
        with open(vectorstore_path, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        # チャンクデータを保存
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        print(f"RAGデータをキャッシュに保存しました: {cache_dir}")
        
    except Exception as e:
        print(f"キャッシュ保存エラー: {str(e)}")


def initialize_rag():
    """
    RAG機能の初期化（キャッシュ機能付き）
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
    
    # ファイルハッシュを生成
    files_hash = get_files_hash(existing_files)
    
    # キャッシュからデータを読み込み試行
    vectorstore, chunks = load_cached_rag_data(ct.RAG_CACHE_DIR, files_hash)
    
    if vectorstore is not None and chunks is not None:
        # キャッシュから読み込み成功
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_chunks = chunks
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": ct.SEARCH_K})
        print("キャッシュからRAGデータを読み込みました。初期化をスキップします。")
    else:
        # キャッシュがないので新規作成
        print("キャッシュが見つからないため、RAGデータを新規作成します...")
        
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
        chunks = split_docs[:max_chunks]
        
        print(f"使用チャンク数: {len(chunks)} / {len(split_docs)}")
        
        # 埋め込みベクター作成
        embeddings = OpenAIEmbeddings(
            model=ct.OPENAI_EMBEDDING_MODEL
        )
        
        print("埋め込みベクター作成中...")
        # FAISSベクターストア作成
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # キャッシュに保存
        save_rag_data_to_cache(vectorstore, chunks, ct.RAG_CACHE_DIR, files_hash)
        
        # セッション状態に保存
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_chunks = chunks
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": ct.SEARCH_K})
    
    # 会話履歴の初期化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    return True


def format_latex_equations(text):
    """
    LaTeX数式を厳格なルールに従って$記号で囲む形式に変換
    """
    import re
    
    # LaTeX記号を厳格ルールに従って変換
    latex_conversions = {
        # 基本演算子
        r'\\times': '×',
        r'\\cdot': '×',
        r'\\div': '÷',
        r'\\pm': '±',
        r'\\mp': '∓',
        
        # 比較演算子
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\propto': '∝',
        
        # 分数を/形式に変換
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'\1/\2',
        
        # 累乗を²,³,⁴形式に変換
        r'\^\{2\}': '²',
        r'\^2': '²',
        r'\^\{3\}': '³',
        r'\^3': '³',
        r'\^\{4\}': '⁴',
        r'\^4': '⁴',
        r'\^\{([^}]+)\}': r'^(\1)',
        r'\^([0-9]+)': r'^\1',
        
        # ギリシャ文字（単位として使用）
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
        
        # 数学記号
        r'\\sum': 'Σ',
        r'\\int': '∫',
        r'\\sqrt\{([^}]+)\}': r'√(\1)',
        r'\\dots': '…',
        r'\\ldots': '…',
        
        # 下付き文字の処理
        r'([A-Za-z])_\{([^}]+)\}': r'\1\2',
        r'([A-Za-z])_([0-9]+)': r'\1\2',
        
        # textコマンドを削除
        r'\\text\{([^}]+)\}': r'\1',
        r'\\mathrm\{([^}]+)\}': r'\1',
    }
    
    def convert_math(match):
        math_content = match.group(1) if match.group(1) else match.group(2)
        
        # LaTeX記号を順次変換
        for latex_pattern, replacement in latex_conversions.items():
            math_content = re.sub(latex_pattern, replacement, math_content)
        
        # 数式を$記号で囲む（厳格ルールに従う）
        return f"${math_content}$"
    
    # 【】で囲まれた数式も$記号に変換（既存の出力対応）
    text = re.sub(r'【([^】]+)】', convert_math, text)
    
    # $$ で囲まれた数式を変換
    text = re.sub(r'\$\$\s*([^$]+?)\s*\$\$', convert_math, text)
    # 単一$で囲まれた数式はそのまま処理
    text = re.sub(r'(?<!\$)\$\s*([^$]+?)\s*\$(?!\$)', convert_math, text)
    
    # [ ] で囲まれた数式も変換
    text = re.sub(r'\[\s*([^\[\]]+?)\s*\]', convert_math, text)
    
    # 重複した$記号を整理
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
        # LLMのオブジェクトを用意（コスト最適化）
        llm = ChatOpenAI(
            model_name=ct.MODEL, 
            temperature=ct.TEMPERATURE,
            max_tokens=ct.OPENAI_MAX_TOKENS  # トークン数制限でコスト削減
        )

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


def get_rag_chain_answer_qa_streaming(user_input):
    """
    RAGチェーンを使った問い合わせ回答の取得（ストリーミング）
    """
    if not st.session_state.get('rag_initialized', False) or st.session_state.get('retriever') is None:
        def error_generator():
            yield {"content": "RAG機能が初期化されていません。サイドバーの「🚀 RAG機能を初期化」ボタンをクリックしてください。"}
        return error_generator()
    
    try:
        # LLMのオブジェクトを用意（ストリーミング対応）
        llm = ChatOpenAI(
            model_name=ct.MODEL, 
            temperature=ct.TEMPERATURE,
            max_tokens=ct.OPENAI_MAX_TOKENS,
            streaming=True  # ストリーミングを有効化
        )

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

        # LLMへのリクエストとストリーミングレスポンス取得
        chat_history = st.session_state.get('chat_history', [])
        
        # ストリーミング処理
        full_answer = ""
        for chunk in chain.stream({"input": user_input, "chat_history": chat_history}):
            if "answer" in chunk:
                content = chunk["answer"]
                full_answer += content
                yield {"content": content}
        
        # LLMレスポンスを会話履歴に追加
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.extend([HumanMessage(content=user_input), full_answer])

    except Exception as e:
        def error_generator():
            yield {"content": f"回答生成中にエラーが発生しました: {str(e)}"}
        return error_generator()
