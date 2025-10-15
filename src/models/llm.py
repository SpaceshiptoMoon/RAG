
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional
from abc import ABC, abstractmethod

# 日志配置：时间 文件 信息
logging.basicConfig(
    format='%(asctime)s %(filename)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# 可选依赖，运行时按需导入，避免导入错误导致模块加载失败
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from src.prompt.prompt_template import PROMPT_TEMPLATE



class OllamaModel:
    """
    Ollama LLM 实现。

    Args:
        model_name (str, optional): Ollama 模型名称。
        base_url (str, optional): Ollama 服务地址。
        kwargs: 其他参数。
    """
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> None:
        """
        初始化 Ollama 模型。
        """
        load_dotenv()
        if ChatOllama is None:
            logging.error(f"{__file__} Ollama SDK not installed or failed to import")
            raise RuntimeError("Ollama SDK not installed or failed to import")
        self.llm = ChatOllama()
        self.llm.base_url = base_url or os.getenv("OLLAMA_BASE_URL")
        self.llm.model = model_name or os.getenv("OLLAMA_LLM_MODEL_NAME")

    def chat(self, prompt: str, context: str = '', history: Optional[List] = None) -> str:
        """
        使用 Ollama 生成回复。

        Args:
            prompt (str): 用户问题。
            context (str): 上下文信息。
            history (List, optional): 对话历史。

        Returns:
            str: 生成的回复文本。
        """
        full_prompt = PROMPT_TEMPLATE.format(question=prompt, context=context)
        logging.info(f"{__file__} OllamaModel.chat 调用，prompt={prompt}")
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(full_prompt)
        return self.llm(full_prompt)



class OpenAIModel:
    """
    OpenAI LLM 实现。

    Args:
        api_key (str, optional): OpenAI API 密钥。
        model_name (str, optional): OpenAI 模型名称。
        base_url (str, optional): OpenAI 服务地址。
        kwargs: 其他参数。
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> None:
        """
        初始化 OpenAI 模型。
        """
        load_dotenv()
        if ChatOpenAI is None:
            logging.error(f"{__file__} ChatOpenAI not installed or failed to import")
            raise RuntimeError("ChatOpenAI not installed or failed to import")
        api_key = api_key or os.getenv("LLM_API_KEY") 
        model_name = model_name or os.getenv("LLM_MODEL")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        self.llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)

    def chat(self, prompt: str, context: str = '', history: Optional[List] = None) -> str:
        """
        使用 OpenAI 生成回复。

        Args:
            prompt (str): 用户问题。
            context (str): 上下文信息。
            history (List, optional): 对话历史。

        Returns:
            str: 生成的回复文本。
        """
        full_prompt = PROMPT_TEMPLATE.format(question=prompt, context=context)
        logging.info(f"{__file__} OpenAIModel.chat 调用，prompt={prompt}")
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(full_prompt)
        return self.llm(full_prompt)



class EchoModel:
    """
    调试用的回显模型，会返回格式化后的输入，便于本地测试。

    Args:
        prefix (str): 回显前缀。
    """
    def __init__(self, prefix: str = "ECHO:") -> None:
        """
        初始化 Echo 模型。
        """
        self.prefix = prefix

    def chat(self, prompt: str, context: str = '', history: Optional[List] = None) -> str:
        """
        回显输入内容。

        Args:
            prompt (str): 用户问题。
            context (str): 上下文信息。
            history (List, optional): 对话历史。

        Returns:
            str: 格式化后的回显文本。
        """
        logging.info(f"{__file__} EchoModel.chat 调用，prompt={prompt}")
        return f"{self.prefix} prompt={prompt}\ncontext={context}\nhistory={history}"



def get_llm(provider: Optional[str] = None, **kwargs) -> None:
    """
    工厂函数：根据 provider 返回对应的模型实例。

    Args:
        provider (str, optional): 'ollama' | 'openai' | 'echo'。如果为空则从环境变量 LLM_PROVIDER 读取，默认 'ollama'
        kwargs: 其他参数，传递给模型构造函数。

    Returns:
        BaseModel: 对应的 LLM 实例。
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    logging.info(f"{__file__} get_llm 调用，provider={provider}")
    try:
        if provider == "ollama":
            return OllamaModel(**kwargs)
        elif provider == "echo":
            return EchoModel(**kwargs)
        else: 
            return OpenAIModel(**kwargs)
    except Exception as e:
        logging.error(f"{__file__} get_llm 初始化失败: {e}")




# 向后兼容：保留原有类名别名
Ollama_Model = OllamaModel

class DualLLM:
    """
    双 LLM 管理器，分别用于决策和生成。

    Attributes:
        decision_llm: 用于决策的 LLM 实例。
        generate_llm: 用于生成的 LLM 实例。
    """
    def __init__(self,
        decision_provider: Optional[str] = None,
        generate_provider: Optional[str] = None,
        decision_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None):
        """
        初始化双 LLM。

        Args:
            decision_provider (str, optional): 决策 LLM 类型。
            generate_provider (str, optional): 生成 LLM 类型。
            decision_kwargs (dict, optional): 决策 LLM 参数。
            generate_kwargs (dict, optional): 生成 LLM 参数。
        """
        self.decision_llm = get_llm(decision_provider, **(decision_kwargs or {}))
        self.generate_llm = get_llm(generate_provider, **(generate_kwargs or {}))
        logging.info(f"{__file__} DualLLM 初始化，decision={decision_provider}, generate={generate_provider}")

    def decision(self, prompt: str, context: str = '', history: Optional[List] = None) -> str:
        """
        使用决策 LLM 生成回复。

        Args:
            prompt (str): 决策问题。
            context (str): 上下文。
            history (List, optional): 对话历史。

        Returns:
            str: 决策回复。
        """
        logging.info(f"{__file__} DualLLM.decision 调用，prompt={prompt}")
        return self.decision_llm.chat(prompt, context, history)

    def generate(self, prompt: str, context: str = '', history: Optional[List] = None) -> str:
        """
        使用生成 LLM 生成回复。

        Args:
            prompt (str): 生成问题。
            context (str): 上下文。
            history (List, optional): 对话历史。

        Returns:
            str: 生成回复。
        """
        logging.info(f"{__file__} DualLLM.generate 调用，prompt={prompt}")
        return self.generate_llm.chat(prompt, context, history)


if __name__ == "__main__":
    # 本地自测：分别用 echo provider 初始化决策和生成 LLM
    dual_llm = DualLLM(decision_provider="echo", generate_provider="echo")
    prompt = '你是谁'
    content = '示例上下文'
    print("决策 LLM:", dual_llm.decision(prompt, content))
    print("生成 LLM:", dual_llm.generate(prompt, content))

