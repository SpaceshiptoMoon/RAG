"""
vector.py - 文档向量化与向量存储模块
功能：将文档转换为向量并存入向量数据库
支持格式：txt, pdf, docx等
"""

import os
import time
import logging
from typing import List, Optional, Union
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_core.documents.base import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from src.docs_read.data_read import ReadFiles

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentVectorizer:
    """文档向量化处理器"""
    
    def __init__(self, 
                 vector_store_path: str 
                 ):
        """
        初始化向量化处理器
        
        Args:
            vector_store_path: 向量数据库存储路径
        """
        self.vector_store_path = vector_store_path
        
        # 初始化嵌入模型
        self.embeddings = self._initialize_embeddings()
        self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings
                )
             
    def __init_doc_operater(self, path):
        return ReadFiles(path)
        
    def _initialize_embeddings(self) -> OpenAIEmbeddings:
        """初始化嵌入模型"""
        try:
            load_dotenv()
            model = os.getenv("SILICONFLOW_MODEL_NAME")
            base_url = os.getenv("SILICONFLOW_BASE_URL")
            api_key = os.getenv("SILICONFLOW_API_KEY")
            embeddings = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)
            logger.info(f"嵌入模型初始化成功")
            return embeddings
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
        
    def txt_to_Document(self, chunk: Union[List, str])->List[Document]:
        docs = []
        if isinstance(chunk, List):
            for txt in chunk:
                docs.append(Document(page_content=txt))
        else:
            docs.append(page_content=chunk)
        return docs
        
    def create_vector_store(self, chunks: List, persist: bool = True) -> Chroma:
        """
        创建向量存储
        
        Args:
            chunks: 文本块列表
            persist: 是否持久化存储
            
        Returns:
            向量存储实例
        """
        try:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_store_path if persist else None
            )
            
          
            if persist:
                self.vector_store.persist()
                logger.info(f"向量数据库已持久化到: {self.vector_store_path}")
            
            logger.info("向量数据库创建成功")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"向量数据库创建失败: {e}")
            raise
        
    def delete_db(self) -> None:
        if self.vector_store:
            client = self.vector_store._client
            
            # 获取所有集合
            collections = client.list_collections()
            
            logger.info(f"清空下面集合{collections}")
            
            # 逐个清空集合内容
            for collection in collections:
                # 获取集合中的所有文档ID
                items = collection.get()
                if items['ids']:
                    # 删除集合中的所有文档
                    collection.delete(ids=items['ids'])
                    print(f"已清空集合: {collection.name}")
                
    def add_to_existing_store(self, chunks: List) -> None:
        """
        向现有向量数据库添加文档
        
        Args:
            chunks: 新的文本块列表
        """
        if self.vector_store is None:
            # 尝试加载现有向量数据库
            try:
                self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings
                )
                logger.info("加载现有向量数据库成功")
            except:
                logger.info("未找到现有向量数据库，创建新的数据库")
                self.create_vector_store(chunks)
                return
        
        # 添加新文档
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        logger.info(f"成功添加 {len(chunks)} 个文本块到向量数据库")
    
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
                
            if is_text:
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
                    self.create_vector_store(chunks)
                
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
        if self.vector_store is None:
            raise ValueError("向量数据库未初始化，请先处理文档")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)
        
        logger.info(f"相似度查询完成，返回 {len(results)} 个结果")
        return results
        
    def _initialize_vector_store(self, collection_name: str = "documents_collection") -> Chroma:
        """
        初始化或加载向量存储
        
        Args:
            collection_name: 集合名称，用于区分不同的文档集合
            
        Returns:
            初始化后的向量存储实例
        """
        try:
            # 检查是否已经存在持久化的向量数据库
            if os.path.exists(self.vector_store_path):
                logger.info(f"加载现有向量数据库: {self.vector_store_path}")
                
                # 从持久化目录加载现有向量存储
                self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                # 验证集合是否存在且包含文档
                collections = self.vector_store._client.list_collections()
                collection_exists = any(col.name == collection_name for col in collections)
                
                if collection_exists:
                    # 检查集合中是否有文档
                    collection = self.vector_store._client.get_collection(collection_name)
                    count = collection.count()
                    logger.info(f"向量数据库加载成功，集合 '{collection_name}' 中包含 {count} 个文档")
                else:
                    logger.info("集合不存在，将创建新的空向量存储")
                    self.vector_store = None
                    
            # 如果不存在持久化数据或加载失败，创建新的向量存储
            if self.vector_store is None:
                logger.info("创建新的向量数据库")
                
                # 确保存储目录存在
                os.makedirs(self.vector_store_path, exist_ok=True)
                
                # 创建空的向量存储 [7](@ref)
                self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                logger.info(f"新的向量数据库创建成功: {self.vector_store_path}")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            
            # 备用方案：创建内存中的临时向量存储
            logger.warning("使用内存中的临时向量存储作为备用方案")
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            return self.vector_store

