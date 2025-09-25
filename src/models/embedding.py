import os
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
from abc import ABC
import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
import re
from bs4 import BeautifulSoup
from IPython.display import display, Code, Markdown
import random
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv

class BaseEmbeddings(ABC):
    """向量化基类，用于将文本转化为向量表示。子类需实现具体向量化逻辑。

    特性：
        - 强制子类实现 `get_embedding` 方法（抽象方法）
        - 提供通用的余弦相似度计算工具方法
        - 支持本地模型与API模式切换
    """
    def __init__(self, path: Optional[str] = None, is_api: bool = False) -> None:
        """初始化向量化工具
        
        Args:
            path: 模型路径（本地模式必需，API模式可选）
            is_api: 是否使用API模式（默认False）
        """
        self.path = path
        self.is_api = is_api

    @abstractmethod
    def get_embedding(self, text: str, model: str) -> List[float]:
        """获取文本的向量表示（子类必须实现）
        
        Args:
            text: 待向量化的文本
            model: 使用的模型标识符
            
        Returns:
            文本对应的向量（浮点数列表）
            
        Raises:
            NotImplementedError: 子类未实现时抛出
        """
        pass  

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """计算两个向量的余弦相似度（工具方法）
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            相似度值（范围[-1, 1]）
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return dot_product / magnitude if magnitude else 0.0
    
class API_Embedding(BaseEmbeddings):
    """
    使用 OpenAI 的 Embedding API 来获取文本向量的类， 继承自 BaseEmbeddings。
    """
    def __init__(self, is_api:bool = False, is_ollama:bool = True) -> None:
        """初始化类， 设置OpenAI API 客户端， 如果使用的是API调用。
        
        参数：
        path (str) - 本地模型的路径，使用API时可以为空
        is_api (bool) - 是否通过 API 获取 Embedding， 默认为 True
        """
        super().__init__(is_api=is_api)
        if self.is_api & is_ollama:
            raise ValueError("is_api和is_ollama不能同时为True！")
        
        load_dotenv()
        
        if self.is_api:
            # 初始化 OpenAI API 客户端
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_AI_EMBEDDING_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
            self.model_name = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
            
        if is_ollama:
            # 初始化嵌入模型
            from langchain_community.embeddings import OllamaEmbeddings
            self.is_ollama = is_ollama
            self.embeddings = OllamaEmbeddings()
            self.embeddings.base_url = os.getenv("OLLAMA_BASE_URL")
            self.embeddings.model = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME")
             
    def get_embedding(self, text: str) -> List[float]:
        """使用 OpenAI 的 Embedding API 获取文本的向量表示。
        
        参数：
        text (str) - 需要转化为向量的文本
        
        返回：
        list[float] - 文本的向量表示
        """
        if self.is_api:
            # 去掉文本的换行符
            text.replace('\n', '')
            # 调用 OpenAI API 获取文本的向量表示
            return self.client.embeddings.create(inpute=[text], model=self.model_name).data[0].embedding
        
        elif self.is_ollama:
            # 去掉文本的换行符
            text.replace('\n', '')
            return self.embeddings.embed_query(text)
    
class Local_Embedding(BaseEmbeddings):
    """
    使用 OpenAI 的 Embedding API 来获取文本向量的类， 继承自 BaseEmbeddings。
    """
    def __init__(self, path: str = '') -> None:
        """初始化类， 设置OpenAI API 客户端， 如果使用的是API调用。
        
        参数：
        path (str) - 本地模型的路径，使用API时可以为空
        """
        super().__init__(path=path)
        self.path = path 
        self.model = AutoModel.from_pretrained(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
    
    def get_embedding(self, text: str) -> List[float]:
        """使用 OpenAI 的 Embedding API 获取文本的向量表示。
        
        参数：
        text (str) - 需要转化为向量的文本
        
        返回：
        list[float] - 文本的向量表示
        """
        input = self.tokenizer(text)
        with torch.no_grad():
            output = self.model(**input)
        return output

if __name__ == '__main__':
    embedding = API_Embedding(is_ollama = True)
    vector1 = embedding.get_embedding('我爱吃水果')
    vector2 = embedding.get_embedding('苹果我每天都吃，还会吃橘子')
    similarity_rate = embedding.cosine_similarity(vector1, vector2)
    print(similarity_rate)
    

        
        
        

    
    
        
        
        
        
        
        
        
        
    