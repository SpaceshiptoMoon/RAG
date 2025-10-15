# src/rag/rag_system.py
import os
import hashlib
from typing import List, Dict, Any
from src.docs_read.data_read import ReadFiles
from src.vector.milvus_db import MilvusManager
from src.models.embedding import  EmbeddingModelFactory
from src.vector.vectorstore import DocumentVectorizer
from src.rag.retriever import VectorRetriever
from src.rag.generator import AnswerGenerator

class RAGSystem:
    """
    完整的 RAG 系统：索引构建 + 检索增强生成
    """
    def __init__(self, data_path: str, collection_name: str = "documents"):
        self.data_path = data_path
        self.collection_name = collection_name
        
        # 初始化组件
        self._initialize_components()
        
        # 确保集合存在
        self._ensure_collection()
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 嵌入客户端
        self.embedding_client = EmbeddingModelFactory.create_model({'enable_cache': True})
        
        # Milvus 客户端
        host = os.getenv("MILVUS_HOST", "127.0.0.1")
        self.milvus_client = MilvusManager(host=host, port=19530)
        
        # 检索器
        self.retriever = VectorRetriever(
            self.milvus_client, 
            self.embedding_client, 
            self.collection_name
        )
        
        # 生成器
        self.generator = AnswerGenerator()
        
        # 文档读取器
        self.doc_reader = ReadFiles(self.data_path)
    
    def _ensure_collection(self):
        """确保 Milvus 集合存在"""
        try:
            # 检查集合是否存在
            collections = self.milvus_client.list_collections()
            if self.collection_name not in collections:
                self.milvus_client()
        except Exception as e:
            print(f"检查集合时出错: {e}")
            self._create_collection()
    
    def _create_collection(self):
        """创建 Milvus 集合"""
        schema_config = {
            "collection_name": self.collection_name,
            "dimension": 1536,  # 需要与嵌入模型维度匹配
            "fields": [
                {"name": "id", "type": "VARCHAR", "is_primary": True, "max_length": 100},
                {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 1536},
                {"name": "text", "type": "VARCHAR", "max_length": 65535},
                {"name": "source", "type": "VARCHAR", "max_length": 500},
                {"name": "chunk_index", "type": "INT64"}
            ]
        }
        self.milvus_client.create_collection(schema_config)
    
    def build_index(self) -> bool:
        """
        构建文档索引：读取文档、分块、生成向量、存储到 Milvus
        """
        try:
            print("开始构建文档索引...")
            
            # 读取并分块文档
            documents = self.doc_reader.get_content(max_token_len=600, cover_content=150)
            print(f"读取到 {len(documents)} 个文档块")
            
            # 准备批量插入的数据
            insert_data = []
            for i, doc_chunk in enumerate(documents):
                # 生成文档ID
                doc_id = hashlib.md5(f"{i}_{doc_chunk[:50]}".encode()).hexdigest()[:16]
                
                # 生成向量（批量处理更高效）
                insert_data.append({
                    "id": doc_id,
                    "text": doc_chunk,
                    "source": f"document_{i}",
                    "chunk_index": i
                })
            
            # 批量生成向量并插入
            batch_size = 50
            for i in range(0, len(insert_data), batch_size):
                batch = insert_data[i:i + batch_size]
                batch_texts = [item["text"] for item in batch]
                
                # 生成批量向量
                batch_vectors = self.embedding_client.embed_query(batch_texts)
                
                # 准备最终插入数据
                final_batch = []
                for j, item in enumerate(batch):
                    if j < len(batch_vectors):
                        final_batch.append({
                            **item,
                            "embedding": batch_vectors[j]
                        })
                
                # 插入到 Milvus
                if final_batch:
                    self.milvus_client.insert(
                        collection_name=self.collection_name,
                        data=final_batch
                    )
                    print(f"已插入 {len(final_batch)} 个文档块")
            
            print("文档索引构建完成！")
            return True
            
        except Exception as e:
            print(f"构建索引时出错: {e}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        RAG 查询：检索 + 生成
        """
        try:
            # 1. 检索相关文档
            print("正在检索相关文档...")
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            print(f"检索到 {len(retrieved_docs)} 个相关文档")
            
            # 2. 生成答案
            print("正在生成答案...")
            result = self.generator.generate_answer(question, retrieved_docs)
            
            return {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
                "retrieved_docs": retrieved_docs,
                "retrieved_count": len(retrieved_docs)
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"处理查询时出现错误：{str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            collections = self.milvus_client.list_collections()
            collection_info = self.milvus_client.describe_collection(self.collection_name)
            
            return {
                "collection_exists": self.collection_name in collections,
                "collection_name": self.collection_name,
                "data_path": self.data_path,
                "components_initialized": True
            }
        except Exception as e:
            return {
                "collection_exists": False,
                "error": str(e)
            }