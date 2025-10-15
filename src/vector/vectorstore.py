"""
vector.py - 文档向量化与向量存储模块
功能：将文档转换为向量并存入向量数据库
支持格式：txt, pdf, docx等
"""

import os
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
                 milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1"),
                 milvus_port: Union[str, int] = os.getenv("MILVUS_PORT", "19530​"),
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
            "model_type": "siliconflow",
            "model_name": "BAAI/bge-m3",
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
            results = self.milvus.search(self.collection_name, [query_vector], top_k=top_k, fields=['text', 'metadata'])
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

    def process_document_enhanced(self, file_path: str, is_text: bool = True, 
                                add_to_existing: bool = True, batch_size: int = 30) -> bool:
        """
        处理单个文档的完整流程（适配新版insert方法）
        
        Args:
            file_path: 文档路径
            is_text: 是否为文本类型
            add_to_existing: 是否添加到现有数据库
            batch_size: 每批处理的chunk数量
            
        Returns:
            处理是否成功
        """
        try:
            # 确保批次大小合理（可根据您的嵌入模型API限制调整）
            if batch_size > 64:
                logger.warning(f"批次大小 {batch_size} 超过推荐值(64)，已自动调整为64")
                batch_size = 64
                
            # 1. 加载文档并分割
            documents_operater = self.__init_doc_operater(file_path)
            symbol_chunks = documents_operater.get_symbol_content()
            token_chunks = documents_operater.get_content()
            
            if is_text:
                symbol_chunks = self.txt_to_Document(symbol_chunks)
                token_chunks = self.txt_to_Document(token_chunks)
                
            chunks = token_chunks + symbol_chunks
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                logger.warning(f"文档 {file_path} 未提取到有效内容")
                return False
            
            logger.info(f"开始处理文档: {file_path}, 共 {total_chunks} 个文本块")
            
            # 2. 分批处理
            successful_batches = 0
            for batch_index, i in enumerate(range(0, total_chunks, batch_size)):
                batch_chunks = chunks[i:i + batch_size]
                current_batch_size = len(batch_chunks)
                
                logger.info(f"处理批次 {batch_index + 1}/{(total_chunks-1)//batch_size + 1} ({current_batch_size}个文本块)")
                
                try:
                    # 关键适配步骤：准备插入数据
                    # a. 为批次生成唯一ID（幂等性保障）
                    batch_ids = self._generate_batch_ids(file_path, i, current_batch_size)
                    
                    # b. 将文本块转换为向量
                    vectors = self.embedding_client.batch_embed_with_progress(batch_chunks)
                    
                    # c. 准备元数据
                    metadatas = self._prepare_metadatas(file_path, batch_chunks, i)
                    
                    # 确定目标集合名称
                    collection_name = self._get_target_collection_name(add_to_existing)
                    
                    # 3. 调用新的insert方法
                    if add_to_existing:
                        # 添加到现有集合
                        result_ids = self.insert(
                            collection_name=collection_name,
                            vectors=vectors,
                            texts=batch_chunks,  # 直接使用文本内容
                            metadatas=metadatas,
                            ids=batch_ids,  # 传入幂等性ID
                            max_retries=3
                        )
                    else:
                        # 创建新集合（需要先创建集合，这里假设有相应方法）
                        self.create_vector_store(collection_name, len(vectors[0]))
                        result_ids = self.insert(
                            collection_name=collection_name,
                            vectors=vectors,
                            texts=batch_chunks,
                            metadatas=metadatas,
                            ids=batch_ids,
                            max_retries=3
                        )
                    
                    if result_ids and len(result_ids) == current_batch_size:
                        successful_batches += 1
                        logger.debug(f"批次 {batch_index + 1} 插入成功，获得 {len(result_ids)} 个ID")
                    else:
                        logger.warning(f"批次 {batch_index + 1} 插入结果ID数量不匹配")
                        
                except Exception as batch_error:
                    logger.error(f"批次 {batch_index + 1} 处理失败: {batch_error}")
                    # 可以选择继续处理后续批次而不是立即失败
                    continue
                    
                # 控制处理速率
                time.sleep(1)  # 可根据需要调整
            
            logger.info(f"文档处理完成: {file_path}. 成功处理 {successful_batches} 个批次")
            return successful_batches > 0  # 至少有一个批次成功即视为整体成功
            
        except Exception as e:
            logger.error(f"文档处理失败 {file_path}: {e}")
            return False

    def _generate_batch_ids(self, file_path: str, start_index: int, batch_size: int) -> List[str]:
        """
        为批次生成唯一ID（幂等性关键）
        
        格式: {file_hash}_{chunk_start_index}_{chunk_index}
        示例: "abc123_0_0", "abc123_0_1", ...
        """
        import hashlib
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        
        ids = []
        for j in range(batch_size):
            # 全局唯一的ID：文件指纹 + 起始偏移 + 批次内索引
            chunk_id = f"{file_hash}_{start_index}_{j}"
            ids.append(chunk_id)
        
        return ids

    def _prepare_metadatas(self, file_path: str, chunks: List[str], start_index: int) -> List[Dict[str, Any]]:
        """准备元数据列表"""
        metadatas = []
        for j, chunk in enumerate(chunks):
            metadata = {
                "source_file": file_path,
                "chunk_index": start_index + j,
                "chunk_length": len(chunk),
                "timestamp": time.time()
            }
            metadatas.append(metadata)
        return metadatas

    def _get_target_collection_name(self, add_to_existing: bool) -> str:
        """获取目标集合名称"""
        if add_to_existing:
            return "your_existing_collection_name"  # 替换为您的实际集合名
        else:
            # 可以基于时间戳或文档名生成新集合名
            return f"doc_collection_{int(time.time())}"