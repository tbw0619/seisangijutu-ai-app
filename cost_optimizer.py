"""
コスト最適化モジュール
RAGシステムのAPI使用量とコストを削減するための機能を提供
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import constants as ct


class CostOptimizer:
    """コスト最適化クラス"""
    
    def __init__(self):
        self.usage_file = "./data/api_usage.json"
        self.cache_dir = Path("./data/cache/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, text: str) -> str:
        """テキストからキャッシュキーを生成"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def load_usage_data(self) -> dict:
        """API使用量データを読み込み"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {"daily_calls": {}, "total_calls": 0}
    
    def save_usage_data(self, data: dict):
        """API使用量データを保存"""
        try:
            os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
            with open(self.usage_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            st.warning(f"使用量データの保存に失敗: {e}")
    
    def check_daily_limit(self) -> bool:
        """1日あたりのAPI呼び出し制限をチェック"""
        data = self.load_usage_data()
        today = datetime.now().strftime("%Y-%m-%d")
        daily_calls = data["daily_calls"].get(today, 0)
        
        if daily_calls >= ct.MAX_DAILY_API_CALLS:
            st.error(f"本日のAPI使用制限（{ct.MAX_DAILY_API_CALLS}回）に達しました。明日お試しください。")
            return False
        
        return True
    
    def increment_usage(self):
        """API使用量をインクリメント"""
        data = self.load_usage_data()
        today = datetime.now().strftime("%Y-%m-%d")
        
        data["daily_calls"][today] = data["daily_calls"].get(today, 0) + 1
        data["total_calls"] = data.get("total_calls", 0) + 1
        
        # 古いデータを削除（7日以上前）
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        data["daily_calls"] = {
            date: count for date, count in data["daily_calls"].items()
            if date >= cutoff_date
        }
        
        self.save_usage_data(data)
    
    def get_usage_stats(self) -> dict:
        """使用統計を取得"""
        data = self.load_usage_data()
        today = datetime.now().strftime("%Y-%m-%d")
        today_calls = data["daily_calls"].get(today, 0)
        
        return {
            "today_calls": today_calls,
            "remaining_calls": max(0, ct.MAX_DAILY_API_CALLS - today_calls),
            "total_calls": data.get("total_calls", 0)
        }
    
    def cache_response(self, query: str, response: str):
        """レスポンスをキャッシュ"""
        if not ct.ENABLE_RESPONSE_CACHE:
            return
            
        cache_key = self.get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "expires": (datetime.now() + timedelta(hours=ct.CACHE_EXPIRY_HOURS)).isoformat()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
        except Exception as e:
            st.warning(f"レスポンスキャッシュの保存に失敗: {e}")
    
    def get_cached_response(self, query: str) -> str:
        """キャッシュされたレスポンスを取得"""
        if not ct.ENABLE_RESPONSE_CACHE:
            return None
            
        cache_key = self.get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # 有効期限をチェック
                expires = datetime.fromisoformat(cache_data["expires"])
                if datetime.now() < expires:
                    return cache_data["response"]
                else:
                    # 期限切れのキャッシュを削除
                    cache_file.unlink()
        except Exception:
            pass
        
        return None
    
    def clean_old_cache(self):
        """古いキャッシュファイルを削除"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    expires = datetime.fromisoformat(cache_data["expires"])
                    if datetime.now() >= expires:
                        cache_file.unlink()
                except Exception:
                    # 破損したキャッシュファイルを削除
                    cache_file.unlink()
        except Exception as e:
            st.warning(f"キャッシュクリーンアップに失敗: {e}")


class VectorStoreManager:
    """ベクターストア永続化マネージャー"""
    
    def __init__(self):
        self.vector_store_dir = Path(ct.VECTOR_STORE_PATH)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.vector_store_dir / ct.VECTOR_INDEX_FILE
        self.chunks_path = self.vector_store_dir / ct.CHUNKS_CACHE_FILE
        
    def save_vector_store(self, vectorstore, chunks):
        """ベクターストアとチャンクを永続化"""
        try:
            # FAISSインデックスを保存
            vectorstore.save_local(str(self.vector_store_dir), ct.VECTOR_INDEX_FILE)
            
            # チャンクデータを保存
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(chunks, f)
                
            st.success("✅ ベクターストアを永続化しました")
            return True
            
        except Exception as e:
            st.error(f"ベクターストア保存エラー: {e}")
            return False
    
    def load_vector_store(self, embeddings):
        """永続化されたベクターストアとチャンクを読み込み"""
        try:
            from langchain_community.vectorstores import FAISS
            
            # FAISSインデックスが存在するかチェック
            if not (self.vector_store_dir / f"{ct.VECTOR_INDEX_FILE}.faiss").exists():
                return None, None
            
            # ベクターストアを読み込み
            vectorstore = FAISS.load_local(
                str(self.vector_store_dir), 
                embeddings,
                ct.VECTOR_INDEX_FILE,
                allow_dangerous_deserialization=True
            )
            
            # チャンクデータを読み込み
            chunks = None
            if self.chunks_path.exists():
                with open(self.chunks_path, 'rb') as f:
                    chunks = pickle.load(f)
            
            return vectorstore, chunks
            
        except Exception as e:
            st.warning(f"永続化データの読み込みに失敗: {e}")
            return None, None
    
    def is_cache_valid(self) -> bool:
        """キャッシュが有効かどうかチェック"""
        try:
            # FAISSファイルの存在確認
            faiss_file = self.vector_store_dir / f"{ct.VECTOR_INDEX_FILE}.faiss"
            if not faiss_file.exists():
                return False
            
            # ファイルの更新日時をチェック（24時間以内）
            file_time = datetime.fromtimestamp(faiss_file.stat().st_mtime)
            current_time = datetime.now()
            
            return (current_time - file_time).total_seconds() < ct.CACHE_EXPIRY_HOURS * 3600
            
        except Exception:
            return False
    
    def clear_cache(self):
        """キャッシュをクリア"""
        try:
            # FAISSファイルを削除
            for file_pattern in [f"{ct.VECTOR_INDEX_FILE}.*", ct.CHUNKS_CACHE_FILE]:
                for file_path in self.vector_store_dir.glob(file_pattern):
                    file_path.unlink()
            
            st.success("キャッシュをクリアしました")
            
        except Exception as e:
            st.error(f"キャッシュクリアに失敗: {e}")


# グローバルインスタンス
cost_optimizer = CostOptimizer()
vector_manager = VectorStoreManager()