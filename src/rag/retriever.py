# src/rag/retriever.py
import os
from typing import List, Dict, Any
from src.vector.milvus_db import MilvusManager
from src.models.embedding import EmbeddingClient

class VectorRetriever:
    """
    基于向量相似度的检索器
    """
    def __init__(self, milvus_client: MilvusManager, embedding_client: EmbeddingClient, collection_name: str = "documents"):
        self.milvus_client = milvus_client
        self.embedding_client = embedding_client
        self.collection_name = collection_name
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的文档片段
        """
        try:
            # 将查询转换为向量
            query_vector = self.embedding_client.embed_query([query])[0]
            
            # 在 Milvus 中进行向量相似度搜索
            search_results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["id", "text", "source", "chunk_index"]
            )
            
            return search_results
        except Exception as e:
            print(f"检索过程中出错: {e}")
            return []
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索策略（可扩展为结合关键词和向量检索）
        """
        # 当前使用向量检索，可扩展为结合 BM25 等稀疏检索方法[2](@ref)
        return self.retrieve(query, top_k)