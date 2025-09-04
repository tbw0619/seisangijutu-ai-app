"""
統合版生産技術授業支援アプリ
FAISS-RAGと既存のコンポーネント構造を統合
"""

import streamlit as st
import os
import re
from dotenv import load_dotenv

# 内部モジュールのインポート
import components
import constants as ct
import utils

# PDF処理とベクターストアのためのインポート
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    import numpy as np
    VECTOR_SUPPORT = True
except ImportError as e:
    VECTOR_SUPPORT = False
    IMPORT_ERROR = str(e)

# 環境変数読み込み
load_dotenv()

# ページ設定
st.set_page_config(page_title=f"{ct.APP_NAME}（統合版）", page_icon="🔧")

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
    st.session_state.mode = ct.ANSWER_MODE_1


############################################################
# FAISS-RAG機能
############################################################

def load_pdf_with_faiss():
    """コスト最適化されたFAISS RAG初期化"""
    from cost_optimizer import cost_optimizer, vector_manager
    
    try:
        # OpenAI APIキーの確認
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI APIキーが設定されていません。")
            return False
        
        # 埋め込みオブジェクトを作成
        embeddings = OpenAIEmbeddings(
            model=ct.OPENAI_EMBEDDING_MODEL,
            show_progress_bar=True
        )
        
        # 永続化データの確認
        if vector_manager.is_cache_valid():
            st.info("🔄 永続化されたベクターストアを読み込み中...")
            vectorstore, cached_chunks = vector_manager.load_vector_store(embeddings)
            
            if vectorstore is not None and cached_chunks is not None:
                # キャッシュから復元
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_chunks = cached_chunks
                
                # ファイル別分布を再計算
                file_distribution = {}
                for chunk in cached_chunks:
                    source = chunk.metadata.get('source_file', 'unknown')
                    file_distribution[source] = file_distribution.get(source, 0) + 1
                st.session_state.file_distribution = file_distribution
                
                st.success(f"✅ キャッシュから復元: {len(cached_chunks)}チャンク（API使用なし）")
                return True
        
        # 新規作成の場合
        st.warning("🆕 新しいベクターストアを作成します（API使用）")
        
        # 存在するファイルのみを選択
        existing_files = [pdf_path for pdf_path in ct.PDF_FILES if os.path.exists(pdf_path)]
        
        if not existing_files:
            st.error("利用可能なPDFファイルがありません。")
            return False
        
        st.info(f"📄 {len(existing_files)}個のPDFファイルを読み込み中...")
        
        # 全PDFファイルの読み込み
        all_documents = []
        for i, pdf_path in enumerate(existing_files, 1):
            st.info(f"📚 PDFファイル {i}/{len(existing_files)} を読み込み中: {os.path.basename(pdf_path)}")
            
            try:
                loader = PyMuPDFLoader(pdf_path)
                documents = loader.load()
                
                # メタデータにファイル名を追加
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                
                all_documents.extend(documents)
                st.success(f"✅ {os.path.basename(pdf_path)}: {len(documents)}ページ")
                
            except Exception as e:
                st.error(f"❌ {os.path.basename(pdf_path)}の読み込みエラー: {str(e)}")
                continue
        
        if not all_documents:
            st.error("PDFファイルの読み込みに失敗しました。")
            return False
        
        st.info(f"📚 合計 {len(all_documents)} ページを読み込み完了")
        
        # テキスト分割
        st.info("📝 テキストを分割中...")
        text_splitter = CharacterTextSplitter(
            chunk_size=ct.CHUNK_SIZE,
            chunk_overlap=ct.CHUNK_OVERLAP,
            separator="\n"
        )
        
        # ドキュメント分割
        split_docs = text_splitter.split_documents(all_documents)
        
        # チャンク数制限
        max_chunks = min(ct.MAX_CHUNKS, len(split_docs))
        test_chunks = split_docs[:max_chunks]
        st.info(f"🧪 {len(test_chunks)} チャンクを使用 (全{len(split_docs)}チャンクから)")
        
        # ファイル別の分布を表示
        file_distribution = {}
        for chunk in test_chunks:
            source = chunk.metadata.get('source_file', 'unknown')
            file_distribution[source] = file_distribution.get(source, 0) + 1
        
        # 埋め込みベクター作成（API使用）
        st.info("🤖 埋め込みベクターを作成中... (OpenAI API使用)")
        vectorstore = FAISS.from_documents(test_chunks, embeddings)
        
        # ベクターストアを永続化（次回からAPI不要）
        st.info("💾 ベクターストアを永続化中...")
        vector_manager.save_vector_store(vectorstore, test_chunks)
        
        # セッション状態に保存
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_chunks = test_chunks
        st.session_state.file_distribution = file_distribution
        
        st.success(f"✅ FAISS-RAG初期化完了: {len(test_chunks)}チャンク ({len(existing_files)}ファイル)")
        
        return True
        
    except Exception as e:
        st.error(f"FAISS-RAG初期化エラー: {str(e)}")
        return False


def faiss_search(query, k=ct.FAISS_SEARCH_K):
    """FAISS検索"""
    try:
        if st.session_state.vectorstore is None:
            return []
        
        # 類似度検索
        results = st.session_state.vectorstore.similarity_search_with_score(query, k=k)
        
        # 結果を整形
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score,
                'search_type': 'FAISS similarity'
            })
        
        return formatted_results
        
    except Exception as e:
        st.error(f"FAISS検索エラー: {str(e)}")
        return []


def clean_and_format_text(text):
    """教科書テキストを読みやすく整形"""
    # 改行を適切に処理
    text = re.sub(r'\n+', '\n', text)  # 複数改行を1つに
    text = re.sub(r'\s+', ' ', text)   # 複数スペースを1つに
    
    # 不要な文字を削除
    text = re.sub(r'[^\w\s\.\,\!\?\(\)\[\]\{\}\-\+\=\×\÷\°\%\：\；\、\。\（\）\「\」\『\』]', '', text)
    
    # 句読点の後にスペースを追加
    text = re.sub(r'([。\.])\s*', r'\1 ', text)
    text = re.sub(r'([、\,])\s*', r'\1 ', text)
    
    # 数式を見やすく
    text = re.sub(r'([A-Z])\s*=\s*', r'\n**\1 = ', text)
    text = re.sub(r'(\d+)\s*\.', r'\n\1. ', text)
    
    return text.strip()


def extract_key_points(content):
    """教科書内容から重要ポイントを抽出"""
    key_points = []
    
    # 数式を抽出
    formulas = re.findall(r'[A-Z]\s*=\s*[^。\n]+', content)
    for formula in formulas:
        key_points.append(f"📐 **公式**: {formula.strip()}")
    
    # 重要な概念を抽出
    concepts = re.findall(r'([ア-ン]{2,}の法則|[ア-ン]{2,}の定理)', content)
    for concept in concepts:
        key_points.append(f"🔑 **重要概念**: {concept}")
    
    # 定義を抽出
    definitions = re.findall(r'([^。\n]*とは[^。\n]*)', content)
    for definition in definitions[:2]:  # 最大2つ
        key_points.append(f"💡 **定義**: {definition.strip()}")
    
    return key_points


def display_math_enhanced_response(response):
    """数式表示を強化したレスポンス表示"""
    import re
    
    # 各パターンを順次処理
    processed_response = response
    
    # まず [ ] パターンを処理
    bracket_pattern = r'\[\s*([^]]+)\s*\]'
    def replace_bracket_latex(match):
        formula = match.group(1)
        # LaTeX記法を整理
        clean_formula = (formula
                        .replace('\\times', '×')
                        .replace('\\text{A}', 'A')
                        .replace('\\text{V}', 'V')
                        .replace('\\text{Ω}', 'Ω')
                        .replace('\\text{W}', 'W')
                        .replace('\\text{', '')
                        .replace('}', '')
                        .replace('\\Omega', 'Ω')
                        .replace('\\,', ' ')
                        .replace('\\dots', '…')
                        .replace(',', '')
                        .replace('R2S', 'R2')  # R2Sの修正
                        .replace('\\', '')
                        .strip())
        return f"\n\n$${clean_formula}$$\n\n"
    
    processed_response = re.sub(bracket_pattern, replace_bracket_latex, processed_response)
    
    # $ $ パターンを処理
    dollar_pattern = r'\$([^$]+)\$'
    def replace_dollar_latex(match):
        formula = match.group(1)
        clean_formula = (formula
                        .replace('\\times', '×')
                        .replace('\\text{A}', 'A')
                        .replace('\\text{V}', 'V')
                        .replace('\\text{Ω}', 'Ω')
                        .replace('\\text{W}', 'W')
                        .replace('\\text{', '')
                        .replace('}', '')
                        .replace('\\Omega', 'Ω')
                        .replace('\\,', ' ')
                        .replace('\\dots', '…')
                        .replace(',', '')
                        .replace('R2S', 'R2')  # R2Sの修正
                        .replace('\\', '')
                        .strip())
        return f"\n\n$${clean_formula}$$\n\n"
    
    processed_response = re.sub(dollar_pattern, replace_dollar_latex, processed_response)
    
    # $$ で囲まれた数式を検出して表示
    parts = re.split(r'\$\$([^$]+)\$\$', processed_response)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # 通常のテキスト
            if part.strip():
                st.markdown(part)
        else:  # LaTeX数式
            try:
                # さらなる清理
                latex_formula = (part.strip()
                               .replace('R2S', 'R2')
                               .replace('\\dots', '\\ldots')  # LaTeX用の省略記号
                               .replace('…', '\\ldots'))
                
                # Streamlitのst.latex()で表示
                st.latex(latex_formula)
                
                # デバッグ情報（開発時のみ）
                with st.expander("🔧 数式デバッグ情報", expanded=False):
                    st.code(f"Original: {part}")
                    st.code(f"Cleaned: {latex_formula}")
                    
            except Exception as e:
                # LaTeX表示に失敗した場合
                try:
                    # 代替表示方法
                    clean_formula = (part.replace('×', ' × ')
                                   .replace('=', ' = ')
                                   .replace('R2S', 'R2')
                                   .replace('\\dots', '…')
                                   .replace('…', ' … '))
                    st.markdown(f"### 📐 数式: `{clean_formula}`")
                    st.warning(f"LaTeX表示エラー: {e}")
                except:
                    # 最終的なフォールバック
                    st.markdown(f"**数式**: {part}")
                    st.error("数式の表示に問題が発生しました")
                # Streamlitのst.latex()で表示
                st.latex(part.strip())
            except Exception as e:
                # LaTeX表示に失敗した場合
                try:
                    # 代替表示方法
                    clean_formula = part.replace('×', ' × ').replace('=', ' = ')
                    st.markdown(f"### 📐 数式: `{clean_formula}`")
                except:
                    # 最終的なフォールバック
                    st.markdown(f"**数式**: {part}")


def enhance_math_display(text):
    """数式表示を強化するための後処理"""
    import re
    
    # \text{A}、\text{V}、\text{Ω}などのLaTeX記法を通常の文字に変換
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = text.replace('\\,', ' ')  # LaTeX空白を通常空白に
    text = text.replace('\\dots', '…')  # \dotsを省略記号に変換
    
    # 一般的な物理公式パターンを検出して$記号で囲む
    math_patterns = [
        # 基本的な公式
        (r'\bV\s*=\s*I\s*[×*]\s*R\b', r'$V = I × R$'),
        (r'\bP\s*=\s*V\s*[×*]\s*I\b', r'$P = V × I$'),
        (r'\bP\s*=\s*I\s*[²2]\s*[×*]\s*R\b', r'$P = I² × R$'),
        (r'\bP\s*=\s*V\s*[²2]\s*/\s*R\b', r'$P = V²/R$'),
        
        # 抵抗の接続（省略記号付き）
        (r'\bR\s*=\s*R1\s*\+\s*R2\s*\+\s*R3\s*\+\s*[…\\dots]+\s*\+\s*Rn\b', r'$R = R1 + R2 + R3 + … + Rn$'),
        (r'\bR\s*=\s*R1\s*\+\s*R2S?\s*\+\s*R3\s*\+\s*[…\\dots]+\s*\+\s*Rn\b', r'$R = R1 + R2 + R3 + … + Rn$'),
        (r'\bR\s*=\s*R1\s*\+\s*R2\b', r'$R = R1 + R2$'),
        (r'\b1/R\s*=\s*1/R1\s*\+\s*1/R2\s*\+\s*[…\\dots]+\s*\+\s*1/Rn\b', r'$1/R = 1/R1 + 1/R2 + … + 1/Rn$'),
        (r'\b1/R\s*=\s*1/R1\s*\+\s*1/R2\b', r'$1/R = 1/R1 + 1/R2$'),
        
        # キルヒホッフの法則
        (r'\bΣV\s*=\s*0\b', r'$ΣV = 0$'),
        (r'\bΣI\s*=\s*0\b', r'$ΣI = 0$'),
        
        # 数値を含む計算式（より詳細に）
        (r'V\s*=\s*(\d+(?:\.\d+)?)\s*[,，]?\s*[A]\s*[×*]\s*(\d+(?:\.\d+)?)\s*[,，]?\s*[ΩΩ]\s*=\s*(\d+(?:\.\d+)?)\s*[,，]?\s*[V]', r'$V = \1A × \2Ω = \3V$'),
        (r'P\s*=\s*(\d+(?:\.\d+)?)\s*[,，]?\s*[V]\s*[×*]\s*(\d+(?:\.\d+)?)\s*[,，]?\s*[A]\s*=\s*(\d+(?:\.\d+)?)\s*[,，]?\s*[W]', r'$P = \1V × \2A = \3W$'),
    ]
    
    # 各パターンを適用
    for pattern, replacement in math_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 既存の$記号が含まれている部分の清理
    text = re.sub(r'\$([^$]*),\s*\\text\{([^}]+)\}([^$]*)\$', r'$\1\2\3$', text)
    
    # LaTeX記法の問題を修正
    text = re.sub(r'\$([^$]*R2S[^$]*)\$', lambda m: m.group(0).replace('R2S', 'R2'), text)
    text = re.sub(r'\$([^$]*\\dots[^$]*)\$', lambda m: m.group(0).replace('\\dots', '…'), text)
    
    # 残っているLaTeX記法を清理
    text = text.replace('\\times', '×')
    text = text.replace('\\Omega', 'Ω')
    
    return text
    text = text.replace('\\times', '×')
    text = text.replace('\\Omega', 'Ω')
    
    return text


def process_latex_in_text(text):
    """テキスト内のLaTeX記法を処理"""
    # 様々なLaTeX記法を統一
    text = re.sub(r'\[\s*([^]]+)\s*\]', r'$$\1$$', text)  # [ ] を $$ $$ に変換
    text = re.sub(r'\$([^$]+)\$', r'$$\1$$', text)        # $ $ を $$ $$ に変換
    
    # LaTeX記法を整理
    text = text.replace('\\times', '×')
    text = text.replace('\\text{', '')
    text = text.replace('}', '')
    text = text.replace('\\Omega', 'Ω')
    text = text.replace('\\mathrm{', '')
    text = text.replace('\\,', ' ')
    
    return text


def safe_latex_format(text):
    """LaTeX数式を安全な形式に変換"""
    # 複雑なLaTeX記法を簡素化
    text = text.replace('\\times', '×')
    text = text.replace('\\frac{', '(')
    text = text.replace('}{', ')/(')
    text = text.replace('}', ')')
    text = text.replace('\\sum', 'Σ')
    text = text.replace('^2', '²')
    text = text.replace('^3', '³')
    text = text.replace('\\', '')
    return text


def generate_openai_student_answer(query, context_text):
    """コスト最適化されたOpenAI API回答生成"""
    from cost_optimizer import cost_optimizer
    
    try:
        # キャッシュされた回答をチェック
        cached_response = cost_optimizer.get_cached_response(query)
        if cached_response:
            st.info("💰 キャッシュから回答を取得（API使用なし）")
            return cached_response
        
        # 日次制限をチェック
        if not cost_optimizer.check_daily_limit():
            return "本日のAPI使用制限に達しました。キャッシュされた回答のみ利用可能です。"
        
        # 入力パラメータの検証
        if not query:
            query = "質問内容なし"
        if not context_text:
            context_text = "関連する教科書の内容が見つかりませんでした。"
        
        # OpenAI ChatGPTモデルを初期化（コスト最適化済み）
        llm = ChatOpenAI(
            model=ct.OPENAI_CHAT_MODEL,  # gpt-4o-miniに変更済み
            temperature=ct.OPENAI_TEMPERATURE,
            max_tokens=ct.OPENAI_MAX_TOKENS  # 1500に削減済み
        )
        
        # プロンプト作成（LaTeX記法を避けてシンプルに）
        try:
            prompt = ct.SYSTEM_PROMPT_STUDENT_FRIENDLY.format(
                query=query,
                context=context_text
            )
        except Exception as format_error:
            st.error(f"プロンプトフォーマットエラー: {format_error}")
            # フォールバック用の簡単なプロンプト
            prompt = f"""工業高校生向けに分かりやすく回答してください。
            
質問: {query}

教科書の内容: {context_text}

数式は$記号で囲んで表示してください（例：$V = I × R$）。"""
        
        # API使用量をインクリメント
        cost_optimizer.increment_usage()
        
        # OpenAI APIで回答生成
        st.info("🤖 GPT-4o-miniで回答生成中...")
        response = llm.invoke(prompt)
        
        # レスポンスをキャッシュ
        cost_optimizer.cache_response(query, response.content)
        
        return response.content
        
    except Exception as e:
        error_message = str(e)
        st.error(f"OpenAI API エラー: {error_message}")
        
        # エラーの種類に応じて対処法を提示
        if "rate_limit" in error_message.lower():
            st.warning("API利用制限に達しました。しばらく待ってから再試行してください。")
        elif "invalid_request" in error_message.lower():
            st.warning("リクエストに問題があります。プロンプトを簡略化して再試行します。")
        elif "authentication" in error_message.lower():
            st.error("OpenAI APIキーの認証に失敗しました。設定を確認してください。")
        
        # フォールバック応答
        return f"""申し訳ございませんが、回答生成中にエラーが発生しました。

**エラー詳細**: {error_message}

**教科書の関連内容**:
{context_text[:500] if context_text else "内容が見つかりませんでした"}...

教科書の内容をもとに、手動で回答を確認してください。"""


def generate_faiss_response(query, search_results, mode):
    """FAISS検索結果から応答生成"""
    if not search_results:
        return "関連する情報が見つかりませんでした。質問を変えてみてください。"
    
    if mode == ct.ANSWER_MODE_1:  # 教科書検索
        response = f"## 📚「{query}」に関連する教科書の内容\n\n"
        
        for i, result in enumerate(search_results, 1):
            # テキストを読みやすく整形
            cleaned_content = clean_and_format_text(result['content'])
            
            # 重要ポイントを抽出
            key_points = extract_key_points(result['content'])
            
            score = result['similarity_score']
            source_file = result['metadata'].get('source_file', 'unknown')
            
            response += f"### 📖 検索結果 {i} (類似度: {score:.3f})\n"
            response += f"**出典**: {source_file}\n\n"
            
            # 重要ポイントがあれば最初に表示
            if key_points:
                response += "**重要ポイント**:\n"
                for point in key_points[:3]:  # 最大3つ
                    response += f"- {point}\n"
                response += "\n"
            
            # 整形されたテキスト
            if len(cleaned_content) > 400:
                response += f"**内容**: {cleaned_content[:400]}...\n\n"
            else:
                response += f"**内容**: {cleaned_content}\n\n"
            
            response += "---\n\n"
        
        return response
    
    else:  # 問い合わせモード
        # 検索結果から関連コンテンツを抽出
        context_content = []
        for result in search_results[:3]:  # 最大3つの結果を使用
            cleaned_content = clean_and_format_text(result['content'])
            source_file = result['metadata'].get('source_file', 'unknown')
            context_content.append(f"【出典: {source_file}】\n{cleaned_content}")
        
        # 教科書コンテンツを結合
        context_text = "\n\n".join(context_content)
        
        # OpenAI APIを使って工業高校生向けの回答を生成
        answer = generate_openai_student_answer(query, context_text)
        
        # 数式表示を強化するための後処理
        answer = enhance_math_display(answer)
        
        # 追加の数式パターンマッチング
        if '=' in answer and ('V' in answer or 'I' in answer or 'R' in answer or 'P' in answer):
            # 数式が含まれている可能性が高い場合は、さらに処理
            answer = re.sub(r'([VIRPvipr])\s*=\s*([^$\n]+?)(?=\n|$)', 
                          lambda m: f"${m.group(1)} = {m.group(2).strip()}$" if '$' not in m.group(0) else m.group(0), 
                          answer)
        
        # 参考情報を追加
        answer += "\n\n---\n\n**📚 参考にした教科書の内容**:\n"
        for i, result in enumerate(search_results[:2], 1):
            source_file = result['metadata'].get('source_file', 'unknown')
            score = result['similarity_score']
            preview = clean_and_format_text(result['content'])[:150] + "..."
            answer += f"\n{i}. **{source_file}** (関連度: {score:.1f})\n{preview}\n"
        
        return answer


############################################################
# メインアプリケーション
############################################################

def main():
    """メインアプリケーション"""
    
    # タイトル表示
    components.display_app_title()
    st.markdown("**統合版 - FAISS高精度検索対応**")
    
    # サイドバー
    with st.sidebar:
        st.header("利用目的")
        
        # モード選択
        components.display_select_mode()
        
        # モード別説明
        if st.session_state.mode == ct.ANSWER_MODE_1:
            st.info("FAISS検索で高精度な教科書内容検索ができます。重要ポイントを整理して表示します。")
            st.code("キルヒホッフの法則について", language=None)
        else:
            st.info("工業高校生向けの分かりやすい回答をGPT-4o-miniで提供します。計算問題も詳細に解説します。")
            st.code("キルヒホッフの法則を分かりやすく教えて", language=None)
            st.code("オームの法則で電流2A、抵抗5Ωの時の電圧は？", language=None)
            st.code("直列回路の合成抵抗の計算方法は？", language=None)
        
        # コスト管理パネル
        st.markdown("---")
        st.markdown("**💰 コスト管理**")
        
        from cost_optimizer import cost_optimizer, vector_manager
        
        # 使用統計の表示
        usage_stats = cost_optimizer.get_usage_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("本日の使用", f"{usage_stats['today_calls']}")
        with col2:
            st.metric("残り回数", f"{usage_stats['remaining_calls']}")
        
        # プログレスバー
        progress = usage_stats['today_calls'] / ct.MAX_DAILY_API_CALLS
        st.progress(progress, text=f"日次制限: {usage_stats['today_calls']}/{ct.MAX_DAILY_API_CALLS}")
        
        # キャッシュ管理
        st.markdown("**🗄️ キャッシュ管理**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 応答キャッシュクリア"):
                cost_optimizer.clean_old_cache()
                st.success("古いキャッシュを削除しました")
        with col2:
            if st.button("🔄 ベクターキャッシュクリア"):
                vector_manager.clear_cache()
                st.session_state.rag_initialized = False
                st.rerun()
        
        # FAISS-RAG初期化
        st.markdown("---")
        st.markdown("**🧠 RAG機能の初期化**")
        
        if components.display_faiss_initialization_sidebar():
            with st.spinner(ct.SPINNER_TEXT):
                success = load_pdf_with_faiss()
                if success:
                    st.session_state.rag_initialized = True
                    st.success("FAISS-RAG機能の初期化が完了しました！")
                    st.rerun()
                else:
                    st.session_state.rag_initialized = False
    
    # FAISS-RAG機能のステータス表示
    components.display_faiss_rag_status()
    
    # 初期メッセージの追加（初回のみ）
    if not st.session_state.messages:
        components.display_initial_ai_message()
    
    # 会話履歴表示
    components.display_conversation_log()
    
    # チャット入力
    if st.session_state.rag_initialized:
        if prompt := st.chat_input(ct.CHAT_INPUT_HELPER_TEXT):
            # ユーザーメッセージを表示
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # AI応答を生成
            with st.chat_message("assistant"):
                with st.spinner(ct.SPINNER_TEXT):
                    # FAISS検索実行
                    search_results = faiss_search(prompt, k=ct.FAISS_SEARCH_K)
                    
                    # 応答生成
                    response = generate_faiss_response(prompt, search_results, st.session_state.mode)
                    
                    # 応答形式に応じて表示
                    if st.session_state.mode == ct.ANSWER_MODE_1:
                        # 教科書検索モード：そのまま表示
                        st.write(response)
                        content = {
                            "mode": ct.ANSWER_MODE_1,
                            "answer": response
                        }
                    else:
                        # 問い合わせモード：数式強化表示
                        with st.expander("🔧 数式処理情報（デバッグ用）", expanded=False):
                            st.text("元の回答:")
                            st.text(response[:200] + "..." if len(response) > 200 else response)
                            
                            # 数式パターンの検出状況
                            math_found = []
                            if '$' in response:
                                math_found.append("$記号あり")
                            if 'V = I' in response:
                                math_found.append("オームの法則")
                            if 'P = V' in response:
                                math_found.append("電力公式")
                            
                            st.text(f"検出された数式パターン: {', '.join(math_found) if math_found else 'なし'}")
                        
                        display_math_enhanced_response(response)
                        content = {
                            "mode": ct.ANSWER_MODE_2,
                            "answer": response
                        }
                    
                    # 検索結果詳細表示
                    components.display_faiss_search_results(search_results)
            
            # 会話ログに追加
            st.session_state.messages.append({
                "role": "assistant", 
                "content": content
            })
    
    else:
        st.info("⚡ FAISS-RAG機能を初期化してから質問を開始してください。")
    
    # ステータス表示
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("アプリ状態", "✅ 正常" if VECTOR_SUPPORT else "❌ エラー")
    
    with col2:
        rag_status = "✅ 有効" if st.session_state.rag_initialized else "⚠️ 無効"
        st.metric("FAISS-RAG", rag_status)
    
    with col3:
        chunk_count = len(st.session_state.pdf_chunks) if st.session_state.pdf_chunks else 0
        st.metric("ベクター数", f"{chunk_count}件")
    
    # 統合版の説明
    with st.expander("ℹ️ 統合版について"):
        st.markdown(f"""
        **コスト最適化版の特徴:**
        - **🏦 永続化ストレージ**: ベクターストアをローカル保存（再embeddingなし）
        - **💾 レスポンスキャッシュ**: 同じ質問の回答をキャッシュ（24時間）
        - **📊 使用量制限**: 日次API呼び出し制限（{ct.MAX_DAILY_API_CALLS}回/日）
        - **🤖 軽量モデル**: GPT-4o-miniでコスト削減（従来の1/10の料金）
        - **⚡ 高速検索**: FAISS意味的類似度検索
        - **📐 LaTeX数式**: 美しい数式レンダリング
        
        **コスト削減効果:**
        - 初回のみembedding API使用（次回からローカル読み込み）
        - GPT-4o → GPT-4o-mini（約90%コスト削減）
        - レスポンスキャッシュで重複質問のAPI使用ゼロ
        - 日次制限で予算管理
        
        **計算機能:**
        - オームの法則計算
        - キルヒホッフの法則
        - 電力計算
        - 抵抗の直列・並列接続
        - 数式のLaTeX表示
        
        **PDFファイル:**
        - 対象ファイル数: {len(ct.PDF_FILES)}
        - チャンクサイズ: {ct.FAISS_CHUNK_SIZE}
        - 最大チャンク数: {ct.FAISS_MAX_CHUNKS}
        """)


if __name__ == "__main__":
    main()
