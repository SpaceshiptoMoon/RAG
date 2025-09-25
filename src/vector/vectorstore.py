"""
vector.py - 文档向量化与向量存储模块
功能：将文档转换为向量并存入向量数据库
支持格式：txt, pdf, docx等
"""

import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentVectorizer:
    """文档向量化处理器"""
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 vector_store_path: str = "./vector_db"
                 ):
        """
        初始化向量化处理器
        
        Args:
            embedding_model: 嵌入模型名称
            vector_store_path: 向量数据库存储路径
        """
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        
        # 初始化嵌入模型
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        
    def _initialize_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """初始化嵌入模型"""
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"嵌入模型 {self.embedding_model} 初始化成功")
            return embeddings
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
    
    
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
    
    def process_document(self, file_path: str, add_to_existing: bool = False) -> bool:
        """
        处理单个文档的完整流程
        
        Args:
            file_path: 文档路径
            add_to_existing: 是否添加到现有数据库
            
        Returns:
            处理是否成功
        """
        try:
            # 1. 加载文档
            documents = self.load_document(file_path)
            
            # 2. 分割文档
            chunks = self.split_documents(documents)
            
            # 3. 创建或更新向量数据库
            if add_to_existing:
                self.add_to_existing_store(chunks)
            else:
                self.create_vector_store(chunks)
            
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

# 使用示例和测试函数
def main():
    """主函数 - 使用示例"""
    # 初始化向量化处理器
    vectorizer = DocumentVectorizer(
        embedding_model="BAAI/bge-small-zh-v1.5",
        vector_store_path="./my_vector_db",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # 示例文档路径（请替换为实际路径）
    sample_doc_path = "sample_document.txt"
    
    # 检查示例文件是否存在，如果不存在则创建
    if not os.path.exists(sample_doc_path):
        with open(sample_doc_path, 'w', encoding='utf-8') as f:
            f.write("""这是示例文档内容。
            
自然语言处理是人工智能的重要分支。
向量化技术可以将文本转换为数值表示。
相似度搜索可以帮助我们找到相关的文档内容。
            
机器学习算法需要数值数据作为输入。
文本向量化使得计算机能够理解文字信息。""")
        print(f"已创建示例文件: {sample_doc_path}")
    
    try:
        # 处理文档
        success = vectorizer.process_document(sample_doc_path)
        
        if success:
            print("文档向量化处理成功！")
            
            # 测试查询
            test_query = "什么是文本向量化？"
            results = vectorizer.query_similarity(test_query)
            
            print(f"\n查询: '{test_query}'")
            print("相似文档结果:")
            for i, doc in enumerate(results):
                print(f"{i+1}. {doc.page_content[:100]}...")
                
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()