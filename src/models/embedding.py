import os
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()

class EmbeddingClient:
    """
    嵌入模型客户端
    支持多种嵌入模型，提供缓存、监控、错误处理等企业级功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化嵌入模型客户端
        
        Args:
            config: 配置字典，包含模型类型、参数等
        """
        # 统一配置：用户传入优先，其余从环境变量补齐；键保持统一标准
        self.config = self._merge_with_env(config)
        self.logger = self._setup_logging()
        self.embedding_model = None
        self.cache_enabled = self.config.get('enable_cache', True)
        self._initialize_model()
        

    def _merge_with_env(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """将用户传入配置与环境变量配置合并，形成统一键的配置。
        统一键约定（仅使用以下环境变量）：
          - enable_cache: False
        """
        env_cfg = self._load_config_from_env()
        # 仅允许少量可选覆盖（保持同一键名）
        merged = {**env_cfg, **({} if user_config is None else user_config)}
        return merged

    def _load_config_from_env(self) -> Dict[str, Any]:
        """从环境变量加载统一的嵌入模型配置。
        仅使用以下环境变量：
          - EMBED_PROVIDER: openai/qwen/siliconflow/ollama
          - EMBED_MODEL: 模型名称
          - EMBED_API_KEY: API 密钥（ollama 不需要）
          - EMBED_BASE_URL: 可选，覆盖默认 base_url
        """
        provider = (os.getenv('EMBED_PROVIDER') or 'openai').lower()
        model_name = os.getenv('EMBED_MODEL') or 'text-embedding-3-small'
        api_key = os.getenv('EMBED_API_KEY')
        env_base_url = os.getenv('EMBED_BASE_URL')


        return {
            'model_type': provider,
            'model_name': model_name,
            'api_key': api_key,
            'base_url': env_base_url
        }
    
    def _setup_logging(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(f"EmbeddingClient")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(model_type)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_model(self):
        """根据配置初始化嵌入模型（仅支持 openai/qwen/siliconflow/ollama）。"""
        model_type = self.config.get('model_type', 'openai').lower()
        try:
            if model_type == 'openai':
                self.embedding_model = self._init_openai()
            elif model_type == 'qwen':
                self.embedding_model = self._init_qwen()
            elif model_type == 'siliconflow':
                self.embedding_model = self._init_siliconflow()
            elif model_type == 'ollama':
                self.embedding_model = self._init_ollama()
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            self.logger.info(f"成功初始化 {model_type} 嵌入模型", extra={'model_type': model_type})
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}", extra={'model_type': model_type})
            raise

    def _init_openai(self) -> OpenAIEmbeddings:
        """初始化 OpenAI 官方或兼容接口（默认）。"""
        return OpenAIEmbeddings(
            model=self.config.get('model_name', 'text-embedding-3-small'),
            base_url=self.config.get('base_url'),
            openai_api_key=self.config.get('api_key')
        )
    
    def _init_qwen(self) -> OpenAIEmbeddings:
        """使用阿里 Qwen 的 OpenAI 兼容接口。"""
        return OpenAIEmbeddings(
            model=self.config.get('model_name', 'text-embedding-3-small'),
            base_url=self.config.get('base_url'),
            openai_api_key=self.config.get('api_key')
        )
    def _init_siliconflow(self) -> OpenAIEmbeddings:
        """使用 SiliconFlow 的 OpenAI 兼容接口。"""
        return OpenAIEmbeddings(
            model=self.config.get('model_name', 'text-embedding-3-small'),
            base_url=self.config.get('base_url'),
            openai_api_key=self.config.get('api_key')
        )

    def _init_ollama(self) -> OpenAIEmbeddings:
        """使用 OpenAIEmbeddings 连接到 Ollama 的 OpenAI 兼容接口。"""
        return OpenAIEmbeddings(
            model=self.config.get('model_name', 'nomic-embed-text'),
            base_url=self.config.get('base_url'),
            openai_api_key=self.config.get('api_key')  # 允许为空
        )


    @lru_cache(maxsize=1000)
    def _cached_embed_query(self, text: str) -> List[float]:
        """带缓存的查询嵌入"""
        return self.embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档列表（批量处理）
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        try:
            self.logger.info(f"开始嵌入 {len(texts)} 个文档", 
                           extra={'model_type': self.config.get('model_type')})
            
            if not texts:
                return []
            
            # 批量嵌入处理
            embeddings = self.embedding_model.embed_documents(texts)
            
            self.logger.info(f"文档嵌入完成，生成 {len(embeddings)} 个向量", 
                           extra={'model_type': self.config.get('model_type')})
            return embeddings
            
        except Exception as e:
            self.logger.error(f"文档嵌入失败: {str(e)}", 
                           extra={'model_type': self.config.get('model_type')})
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本（支持缓存）
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 向量
        """
        try:
            if self.cache_enabled:
                return self._cached_embed_query(text)
            else:
                return self.embedding_model.embed_query(text)
                
        except Exception as e:
            self.logger.error(f"查询嵌入失败: {str(e)}", 
                           extra={'model_type': self.config.get('model_type')})
            raise

    def get_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            int: 向量维度
        """
        # 测试嵌入一个短文本来获取维度
        try:
            test_text = "test"
            embedding = self.embed_query(test_text)
            return len(embedding)
        except:
            # 返回常见默认维度
            return 768

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.config.get('model_type'),
            'dimension': self.get_dimension(),
            'config': {k: v for k, v in self.config.items() if k != 'api_key'}
        }

    def batch_embed_with_progress(self, texts: List[str], batch_size: int = 90) -> List[List[float]]:
        """
        带进度批处理的嵌入方法
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            List[List[float]]: 向量列表
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # 记录进度
            current_batch = (i // batch_size) + 1
            self.logger.info(f"处理进度: {current_batch}/{total_batches}", 
                           extra={'model_type': self.config.get('model_type')})
        
        return all_embeddings


class EmbeddingModelFactory:
    """嵌入模型工厂类"""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> EmbeddingClient:
        """
        创建嵌入模型实例
        
        Args:
            config: 模型配置
            
        Returns:
            EmbeddingClient: 嵌入模型客户端实例
        """
        return EmbeddingClient(config)
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """获取受支持的模型类型列表（精简版）。"""
        return {
            "supported": ["openai", "qwen", "siliconflow", "ollama"]
        }


if __name__ == "__main__":
    # 简要示例：从环境变量加载并创建
    config = {'enable_cache': True}
    client = EmbeddingModelFactory.create_model(config)
    print(client.get_model_info())
    print(client.embed_query("这是一个测试"))