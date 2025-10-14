"""
vector.py - 文档向量化与向量存储模块
功能：将文档转换为向量并存入向量数据库
支持格式：txt, pdf, docx等
"""


import time
import logging
from typing import List, Optional, Union
from langchain_core.documents.base import Document
from dotenv import load_dotenv

from src.models.embedding import EmbeddingClient, EmbeddingModelFactory
from src.vector.milvus_db import MilvusManager
from src.docs_read.data_read import ReadFiles

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentVectorizer:
    """文档向量化处理器"""
    
    def __init__(self,
                 collection_name: str = "documents_collection",
                 embedding_config: Optional[dict] = None,
                 milvus_host: str = "localhost",
                 milvus_port: Union[str, int] = 19530,
                 ):
        """
        初始化向量化处理器
        
            Args:
            collection_name: Milvus集合名称
            embedding_config: 嵌入模型配置，见 src/models/embedding.py
            milvus_host: Milvus主机
            milvus_port: Milvus端口
        """
        self.collection_name = collection_name
        # 默认嵌入模型配置（可被用户传入配置覆盖）
        self.embedding_config = embedding_config or {
            "model_type": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "enable_cache": True,
        }

        # 初始化嵌入模型与 Milvus 管理器
        self.embedding_client: EmbeddingClient = self._initialize_embeddings()

        # 强制将端口转换为 int（MilvusManager 期望 int）
        try:
            port_val = int(milvus_port)
        except Exception:
            port_val = 19530

        self.milvus: MilvusManager = MilvusManager(host=milvus_host, port=port_val)
        # 立即连接并确保集合存在
        try:
            self.milvus.connect()
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

        self._ensure_collection()
        
             
    def __init_doc_operater(self, path):
        return ReadFiles(path)
        
    def _initialize_embeddings(self) -> EmbeddingClient:
        """初始化嵌入模型"""
        try:
            load_dotenv()
            embeddings = EmbeddingModelFactory.create_model(self.embedding_config)
            logger.info(f"嵌入模型初始化成功")
            return embeddings
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise

    def _ensure_collection(self) -> None:
        """确保Milvus集合已存在，若不存在则创建"""
        # 依据嵌入维度创建集合（若已存在则加载）
        dimension = self.embedding_client.get_dimension()
        if not self.milvus.has_collection(self.collection_name):
            self.milvus.create_collection(self.collection_name, dimension)
            logger.info(f"创建 Milvus 集合 {self.collection_name} dim={dimension}")
        else:
            coll = self.milvus.get_collection(self.collection_name)
            coll.load()
            logger.info(f"已存在 Milvus 集合 {self.collection_name}，已加载")
        
    def txt_to_Document(self, chunk: Union[List, str])->List[Document]:
        docs = []
        if isinstance(chunk, list):
            for txt in chunk:
                if isinstance(txt, str) and txt.strip():
                    docs.append(Document(page_content=txt))
        elif isinstance(chunk, str) and chunk.strip():
            docs.append(Document(page_content=chunk))
        return docs
        
    def create_vector_store(self, chunks: List, persist: bool = True) -> None:
        """对传入chunks做嵌入并插入Milvus（兼容旧接口）"""
        try:
            texts = [d.page_content if isinstance(d, Document) else str(d) for d in chunks]
            if not texts:
                return
            vectors = self.embedding_client.embed_documents(texts)
            # MilvusManager.insert 接收 vectors, texts, metadatas
            self.milvus.insert(self.collection_name, vectors, texts, None)
            logger.info("向量插入完成")
        except Exception as e:
            logger.error(f"向量数据库创建/插入失败: {e}")
            raise
        
    def delete_db(self) -> None:
        """清空当前Milvus集合：删除后重建"""
        try:
            # 使用 MilvusManager 的 drop_collection
            if self.milvus.has_collection(self.collection_name):
                self.milvus.drop_collection(self.collection_name)
                logger.info(f"已删除集合: {self.collection_name}")
            # 重新创建并加载集合
            self._ensure_collection()
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
                
    def add_to_existing_store(self, chunks: List) -> None:
        """
        向现有向量数据库添加文档
        
        Args:
            chunks: 新的文本块列表
        """
        try:
            texts = [d.page_content if isinstance(d, Document) else str(d) for d in chunks]
            if not texts:
                return
            vectors = self.embedding_client.embed_documents(texts)
            self.milvus.insert(self.collection_name, vectors, texts, None)
            logger.info(f"成功添加 {len(chunks)} 个文本块到向量数据库")
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def process_document(self, file_path: str, is_text: str = True, add_to_existing: bool = True, batch_size: int = 30) -> bool:
        """
        处理单个文档的完整流程
        
        Args:
            file_path: 文档路径
            is_text: 是否为文本类型
            add_to_existing: 是否添加到现有数据库
            batch_size: 嵌入模型每次处理的chunk大小
            
        Returns:
            处理是否成功
        """
        try:
            # 确保批次大小不超过API限制
            if batch_size > 64:
                logger.warning(f"批次大小 {batch_size} 超过API限制(64)，已自动调整为64")
                batch_size = 64
                
            # 1. 加载文档操作器
            documents_operater = self.__init_doc_operater(file_path)
            
            # 2. 分割文档
            symbol_chunks = documents_operater.get_symbol_content()
            token_chunks = documents_operater.get_content()
            
            if is_text:
                symbol_chunks = self.txt_to_Document(symbol_chunks)
                token_chunks = self.txt_to_Document(token_chunks)
                
            chunks = token_chunks + symbol_chunks 
            
            # 3. 创建或更新向量数据库         
            total_chunks  = len(chunks)
    
            for i in range(0, total_chunks , batch_size):
                batch = chunks[i:i + batch_size]

                logger.info(f"正在处理批次 {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({len(batch)}个文本块)")
                
                
                if add_to_existing:
                    self.add_to_existing_store(batch)
                else:
                    self.create_vector_store(batch)
                
                time.sleep(2)
            
            logger.info(f"文档处理完成: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"文档处理失败 {file_path}: {e}")
            return False
    
    def query_similarity(self, query: str, top_k: int = 5) -> List:
        """
        查询相似文档
        
        Args:
            query: 查询文本
            top_k: 返回最相似文档数量
            
        Returns:
            相似文档列表
        """
        try:
            query_vector = self.embedding_client.embed_query(query)
            # milvus.search expects a list of vectors
            results = self.milvus.search(self.collection_name, [query_vector], top_k=top_k)
            logger.info(f"相似度查询完成，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"相似度查询失败: {e}")
            return []
        
    def _initialize_vector_store(self, collection_name: str = "documents_collection") -> None:
        """
        初始化或加载向量存储
        
        Args:
            collection_name: 集合名称，用于区分不同的文档集合
            
        Returns:
            None
        """
        try:
            self.collection_name = collection_name
            self._ensure_collection()
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")

