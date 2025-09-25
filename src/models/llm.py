import os
from dotenv import load_dotenv
from typing import List
from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
from src.prompt.prompt_template import PROMPT_TEMPLATE

class BaseModel(ABC):
    """
    基础模型类， 作为所有模型的基类。
    包含一些通用的接口，如加载模型、生成回答等。
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    @abstractmethod
    def chat(self, prompt: str, content: str = '', history: List = []) -> str:
        """
        使用模型生成回答的抽象方法。
        :param prompt: 用户的提问内容
        :param history: 用户的提问内容
        :param content: 用户的提问内容
        """
        pass
    
    # @abstractmethod
    def load_model(self):
        """
        加载模型的方法, 通常用于本地模型
        """
        pass 
    
class Ollama_Model(BaseModel):
    """
    基于 ollama 框架的对话类，继承自 BaseModel。
    主要用于通过 ollama url 来生成对话回答。
    """
    def __init__(self) -> None:
        """
        初始化 本地ollama 的模型。
        :param api_key: ollama API 的密钥
        :param base_url: 用于访问 OpenAI API 的基础 URL，默认为代理 URL
        """
        load_dotenv()
        self.llm = Ollama()  # 初始化 OpenAI 客户端
        self.llm.base_url = os.getenv("OLLAMA_BASE_URL")
        self.llm.model = os.getenv("OLLAMA_LLM_MODEL_NAME")
        
    
    def chat(self, question: str, context: str = '', history: List = []) -> str:
        """
        使用 ollama本地模型 生成回答。
        :param question: 用户的提问
        :param context: 可参考的上下文信息（可选）
        :param history: 之前的对话历史（可选）
        :return: 生成的回答
        """
        full_prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        
        respon = self.llm.invoke(full_prompt)
        return respon
        
        
          
if __name__ == "__main__":
    llm = Ollama_Model()
    prompt = '你是谁'
    content = '大幅度离开洒家发'
    print(llm.chat(prompt, content))
    
    