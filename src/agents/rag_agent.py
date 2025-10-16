"""agents.rag_agent

演示性 RAGAgent：使用检索工具获取上下文，再调用 llm 来生成回答。
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RAGAgent:
    def __init__(self, retriever, llm):
        """初始化 RAGAgent。

        Args:
            retriever: 提供 query_similarity(query, top_k) 接口的检索器/工具
            llm: 提供 chat(prompt, context) 接口的语言模型
        """
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            docs = self.retriever.query_similarity(question, top_k=top_k)
            context = "\n\n".join([d.get("entity", {}).get("text", "")[:1000] for d in docs])
            answer = self.llm.chat(question, context=context)
            return {"answer": answer, "sources": docs}
        except Exception as e:
            logger.error(f"RAGAgent.answer failed: {e}")
            return {"error": str(e)}
