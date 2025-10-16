"""
knowledge_agent.py - 知识问答系统核心类
"""
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI

from src.agent.agent import DeepAgent
from src.vector.vectorstore import DocumentVectorizer
from src.models.llm import Ollama_Model
from src.agent.router_agent import QueryRouter, RouteDecision

logger = logging.getLogger(__name__)

class KnowledgeAgentSystem:
    """知识问答系统核心类"""
    
    def __init__(self, vector_store_path: str):
        load_dotenv()
        
        # 初始化核心组件
        self.llm = Ollama_Model()
        self.agent = DeepAgent()
        self.vectorizer = DocumentVectorizer(vector_store_path)
        
        load_dotenv()
        api_key = os.getenv("OPENAI_AI_EMBEDDING_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_LLM_MODEL_NAME")
        llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
        
        self.router = QueryRouter(llm)
        
    
    def process_query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        处理用户查询的核心方法
        
        Args:
            question: 用户问题
            top_k: RAG检索返回文档数量
            
        Returns:
            包含回答和元数据的字典
        """
        # 1. 路由决策
        route_decision = self.router.route_query(question)
        print(route_decision.strategy)
        print(route_decision.confidence)
        print(route_decision)
        # 2. 根据策略处理查询
        if route_decision.strategy == "direct":
            return self._handle_direct_query(question, route_decision)
        elif route_decision.strategy == "rag":
            return self._handle_rag_query(question, top_k, route_decision)
        elif route_decision.strategy == "agent":  
            return self._handle_agent_query(question, route_decision)
    
    def _handle_direct_query(self, question: str, decision: RouteDecision) -> Dict[str, Any]:
        """处理直接回答查询"""
        try:
            answer = self.llm.chat(question=question, context="")
            
            return {
                "answer": answer,
                "strategy": "direct",
                "confidence": decision.confidence,
                "sources": [],
                "metadata": {
                    "processing_time": 0,
                    "retrieval_count": 0
                }
            }
        except Exception as e:
            logger.error(f"直接查询处理失败: {e}")
            return self._fallback_response(question, e)
    
    def _handle_rag_query(self, question: str, top_k: int, decision: RouteDecision) -> Dict[str, Any]:
        """处理RAG检索查询"""
        try:
            # 检索相关文档

            relevant_docs = self.vectorizer.query_similarity(question, top_k)

            # 构建上下文
            context = self._build_context_from_docs(relevant_docs)
            
            # 生成回答
            answer = self.llm.chat(question=question, context=context)
            
            return {
                "answer": answer,
                "strategy": "rag",
                "confidence": decision.confidence,
                "sources": [doc.page_content[:100] + "..." for doc in relevant_docs],
                "metadata": {
                    "processing_time": 0,
                    "retrieval_count": len(relevant_docs),
                    "retrieved_docs": [doc.metadata for doc in relevant_docs]
                }
            }
        except Exception as e:
            logger.error(f"RAG查询处理失败: {e}")
            return self._fallback_response(question, e)
    
    def _handle_agent_query(self, question: str, decision: RouteDecision) -> Dict[str, Any]:
        """处理Agent联网搜索查询"""
        try:
            answer = self.agent.query(question)
            return {
                "answer": answer,
                "strategy": "agent",
                "confidence": decision.confidence,
                "sources": ["web_search"],
                "metadata": {
                    "processing_time": 0,
                    "search_used": True
                }
            }
        except Exception as e:
            logger.error(f"Agent查询处理失败: {e}")
            return self._fallback_response(question, e)
    
    def _build_context_from_docs(self, documents: List) -> str:
        """从文档构建上下文"""
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"[文档{i+1}]: {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _fallback_response(self, question: str, error: Exception) -> Dict[str, Any]:
        """降级响应处理"""
        fallback_answer = f"抱歉，处理您的问题时遇到错误：{str(error)}。正在使用基础模式回答。"
        
        try:
            direct_answer = self.llm.chat(question=question, context="")
            fallback_answer = direct_answer
        except Exception as e:
            fallback_answer += f" 基础模式也失败了：{str(e)}"
        
        return {
            "answer": fallback_answer,
            "strategy": "fallback",
            "confidence": 0.1,
            "sources": [],
            "metadata": {"error": str(error)}
        }
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """添加文档到知识库"""
        try:
            success = self.vectorizer.process_document(file_path, add_to_existing=True)
            return {
                "success": success,
                "message": "文档添加成功" if success else "文档添加失败",
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"文档添加失败: {e}")
            return {
                "success": False,
                "message": f"文档添加失败: {str(e)}",
                "file_path": file_path
            }
    def check_vector_store_status(self):
        """检查向量库状态"""
        try:
            # 检查集合中文档数量
            if hasattr(self.vector_store, '_collection'):
                count = self.vector_store._collection.count()
                print(f"向量库中文档数量: {count}")
                return count > 0
            return False
        except Exception as e:
            print(f"检查向量库状态失败: {e}")
            return False