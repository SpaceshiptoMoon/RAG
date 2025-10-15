"""
vector.py - 文档向量化与向量存储模块
功能：将文档转换为向量并存入向量数据库
支持格式：txt, pdf, docx等
"""
import hashlib
import time
import logging
from typing import List, Optional, Union, Dict, Any
from langchain_core.documents.base import Document
from src.docs_read.data_read import ReadFiles

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentVectorizer:
    """
    文档向量化处理器。

    Args:
        embedding_client: 嵌入模型实例（负责文本向量化）。
        milvus_client: Milvus 客户端实例（负责向量存储与检索）。
        doc_reader: 文档读取器实例（负责读取和分块文档）。
        collection_name (str): 向量集合名称。
    """
    def __init__(self, embedding_client, milvus_client, doc_reader, collection_name: str = "documents_collection"):
        self.embedding_client = embedding_client
        self.milvus = milvus_client
        self.doc_reader = doc_reader
        self.collection_name = collection_name
        try:
            self.milvus.connect()
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise
        self._ensure_collection()
        
    # 已废弃自动初始化嵌入模型，所有嵌入操作均使用 self.embedding_client

    def _ensure_collection(self) -> None:
        """确保Milvus集合已存在，若不存在则创建"""
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
            self.milvus.insert(self.collection_name, vectors, texts, None)
            logger.info("向量插入完成")
        except Exception as e:
            logger.error(f"向量数据库创建/插入失败: {e}")
            raise
        
    def delete_db(self) -> None:
        """清空当前Milvus集合：删除后重建"""
        try:
            if self.milvus.has_collection(self.collection_name):
                self.milvus.drop_collection(self.collection_name)
                logger.info(f"已删除集合: {self.collection_name}")
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
    def process_document(self, file_path: str, is_text: bool = True, 
                        add_to_existing: bool = True, batch_size: int = 50) -> bool:
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
            if batch_size > 100:
                logger.warning(f"批次大小 {batch_size} 超过推荐值(64)，已自动调整为64")
                batch_size = 100
            # 1. 加载文档并分割
            documents_operater = self.doc_reader
            symbol_chunks = documents_operater.get_symbol_content()
            token_chunks = documents_operater.get_content()
            if is_text:
                symbol_chunks = self.txt_to_Document(symbol_chunks)
                token_chunks = self.txt_to_Document(token_chunks)
            chunks = symbol_chunks + token_chunks 
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
                    batch_ids = self._generate_batch_ids(file_path, i, current_batch_size)
                    batch_chunks = [d.page_content if isinstance(d, Document) else str(d) for d in batch_chunks]
                    vectors = self.embedding_client.batch_embed_with_progress(batch_chunks, batch_size=batch_size)
                    metadatas = self._prepare_metadatas(file_path, batch_chunks, i)
                    if add_to_existing:
                        result_ids = self.milvus.insert(
                            collection_name=self.collection_name,
                            vectors=vectors,
                            texts=batch_chunks,
                            metadatas=metadatas,
                            ids=batch_ids,
                            max_retries=3
                        )
                    else:
                        self.create_vector_store(self.collection_name, len(vectors[0]))
                        result_ids = self.milvus.insert(
                            collection_name=self.collection_name,
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
                    continue
                time.sleep(1)
            logger.info(f"文档处理完成: {file_path}. 成功处理 {successful_batches} 个批次")
            return successful_batches > 0
        except Exception as e:
            logger.error(f"文档处理失败 {file_path}: {e}")
            return False

    def _generate_batch_ids(self, file_path: str, start_index: int, batch_size: int) -> List[str]:
        """
        为批次生成唯一ID（幂等性关键）
        
        格式: {file_hash}_{chunk_start_index}_{chunk_index}
        示例: "abc123_0_0", "abc123_0_1", ...
        """

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
