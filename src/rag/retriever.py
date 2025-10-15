# src/rag/retriever.py
import os
from typing import List, Dict, Any
from src.vector.milvus_db import MilvusManager
from src.models.embedding import EmbeddingClient
from src.log.log_config import setup_logger

# 配置日志
logger = setup_logger(__name__)

class VectorRetriever:
    """
    基于向量相似度的检索器
    
    Args:
        milvus_client: Milvus客户端实例
        embedding_client: 嵌入模型客户端实例
        collection_name: 向量集合名称，默认为"documents"
        
    Attributes:
        milvus_client: Milvus客户端实例
        embedding_client: 嵌入模型客户端
        collection_name: 向量集合名称
    """
    def __init__(self, milvus_client: MilvusManager, embedding_client: EmbeddingClient, collection_name: str = "documents"):
        self.milvus_client = milvus_client
        self.embedding_client = embedding_client
        self.collection_name = collection_name
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的文档片段
        
        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量，默认为5
            
        Returns:
            List[Dict[str, Any]]: 检索到的文档列表，每个文档包含id、text和metadata
        """
        logger.info(f"开始检索查询: {query[:50]}...")
        try:
            # 将查询转换为向量
            logger.debug("生成查询向量")
            query_vector = self.embedding_client.embed_query(query)
        
            # 在 Milvus 中进行向量相似度搜索
            logger.debug(f"在集合 {self.collection_name} 中执行向量搜索")
            search_results = self.milvus_client.search(
                collection_name=self.collection_name,
                query_vectors=[query_vector],
                top_k=top_k,
                fields=["id", "text", "metadata"]
            )
            
            logger.info(f"检索完成，找到 {len(search_results)} 个相关文档")
            return search_results
        except Exception as e:
            logger.error(f"检索过程中出错: {e}")
            return []
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索策略（可扩展为结合关键词和向量检索）
        """
        # 当前使用向量检索，可扩展为结合 BM25 等稀疏检索方法
        return self.retrieve(query, top_k)