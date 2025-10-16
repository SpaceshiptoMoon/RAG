"""
models 包：封装模型客户端（嵌入、LLM 等）。
"""
from src.log.log_config import setup_logger
from .embedding import EmbeddingClient, EmbeddingModelFactory
from .llm import OllamaModel, OpenAIModel, EchoModel, get_llm

logger = setup_logger(__name__)

__all__ = ["EmbeddingClient", "EmbeddingModelFactory", "OllamaModel", "OpenAIModel", "EchoModel", "get_llm"]
