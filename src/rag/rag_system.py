# src/rag/rag_system.py
import os
from typing import Dict, Any
from src.docs_read.data_read import ReadFiles
from src.vector.milvus_db import MilvusManager
from src.models.embedding import EmbeddingModelFactory
from src.vector.vectorstore import DocumentVectorizer
from src.rag.retriever import VectorRetriever
from src.rag.generator import AnswerGenerator
from src.models.llm import get_llm
from src.log.log_config import setup_logger

# 配置logger
logger = setup_logger(__name__)

class RAGSystem:
    """
    完整的 RAG 系统实现类
    
    Args:
        data_path: 文档数据路径
        collection_name: Milvus集合名称，默认为"documents"
        
    Attributes:
        data_path: 文档路径
        collection_name: 向量数据库集合名称
        embedding_client: 嵌入模型客户端
        milvus_client: Milvus数据库客户端
        doc_reader: 文档读取器
        doc_vectorizer: 文档向量化器
        retriever: 向量检索器
        generator: 答案生成器
    """
    def __init__(self, data_path: str, collection_name: str = "documents"):
        logger.info(f"初始化RAG系统 - 数据路径: {data_path}, 集合名称: {collection_name}")
        self.data_path = data_path
        self.collection_name = collection_name
        
        # 初始化组件
        self._initialize_components()
        logger.info("RAG系统初始化完成")
    
    def _initialize_components(self):
        """
        初始化RAG系统所需的所有组件
        
        Args:
            None
            
        Returns:
            None
            
        Note:
            创建嵌入模型、Milvus客户端、文档读取器、向量化器、检索器和生成器实例
            采用单例模式创建底层客户端，避免重复创建
        """
        logger.info("开始初始化RAG系统组件")
        # 嵌入客户端（单例式创建）
        self.embedding_client = EmbeddingModelFactory.create_model({'enable_cache': True})

        # Milvus 客户端（单一连接实例，DocumentVectorizer 会在内部调用 connect）
        host = os.getenv("MILVUS_HOST", "127.0.0.1")
        port = os.getenv("MILVUS_PORT", "19530")
        self.milvus_client = MilvusManager(host=host, port=port)

        # 文档读取器
        self.doc_reader = ReadFiles(self.data_path)

        # 文档向量化与存储器（会确保集合存在并建立连接）
        self.doc_vectorizer = DocumentVectorizer(
            embedding_client=self.embedding_client,
            milvus_client=self.milvus_client,
            doc_reader=self.doc_reader,
            collection_name=self.collection_name,
        )

        # 向量检索器
        self.retriever = VectorRetriever(
            self.milvus_client,
            self.embedding_client,
            self.collection_name,
        )

        # LLM 与答案生成器（通过工厂创建，便于由 env 控制 provider）
        llm = get_llm()
        self.generator = AnswerGenerator(llm)

    
    def _create_collection(self):
        """
        创建 Milvus 集合（基于当前嵌入模型维度）
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            Exception: 集合创建失败时抛出异常
        """
        try:
            dim = self.embedding_client.get_dimension()
            logger.info(f"尝试直接创建维度为{dim}的集合")
            self.milvus_client.create_collection(self.collection_name, dim)
            logger.info(f"直接创建维度为{dim}的集合成功")
        except Exception as e:
            logger.error(f"强制创建集合失败: {e}")
    
    def build_file_index(self) -> bool:
        """
        构建文档索引，包含文档读取、分块、向量化和存储流程
        
        Args:
            None
            
        Returns:
            bool: 索引构建是否成功
            
        Raises:
            Exception: 构建过程中的任何错误
        """
        logger.info("开始构建文档索引")
        try:
            self.doc_vectorizer.process_document()
            logger.info("文档索引构建完成")
            return True
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            return False    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        执行RAG查询，包括检索和生成答案的完整流程
        
        Args:
            question: 用户问题
            top_k: 检索的相关文档数量
            
        Returns:
            Dict[str, Any]: 包含以下字段的结果字典：
                - question: 原始问题
                - answer: 生成的答案
                - sources: 使用的参考源
                - confidence: 答案置信度
                - retrieved_docs: 检索到的文档
                - retrieved_count: 检索到的文档数量
        """
        logger.info(f"开始处理查询: {question[:50]}...")
        try:
            # 1. 检索相关文档
            logger.info("执行文档检索...")
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

            logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
            
            # 2. 生成答案
            logger.info("开始生成答案...")
            result = self.generator.generate_answer(question, retrieved_docs)
            logger.info("答案生成完成")
            
            response = {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
                "retrieved_docs": retrieved_docs,
                "retrieved_count": len(retrieved_docs)
            }
            logger.info(f"查询处理完成，置信度: {result['confidence']}")
            return response
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return {
                "question": question,
                "answer": f"处理查询时出现错误：{str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取RAG系统的当前状态信息
        
        Returns:
            Dict[str, Any]: 包含以下字段的系统信息字典：
                - collection_exists: 集合是否存在
                - collection_name: 集合名称
                - data_path: 数据路径
                - components_initialized: 组件是否已初始化
        """
        try:
            collections = self.milvus_client.list_collections()
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