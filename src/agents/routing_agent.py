"""agents.routing_agent

简单路由器，根据输入选择处理策略或工具。
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RoutingAgent:
    def __init__(self):
        pass

    def route(self, text: str) -> Dict[str, Any]:
        """
        简单的路由逻辑：根据关键词决定使用何种策略。
        返回字典包含 strategy 和 metadata。
        """
        text_low = text.lower()
        if any(w in text_low for w in ["search", "find", "where", "who"]):
            return {"strategy": "search", "tool": "search"}
        if any(w in text_low for w in ["calc", "compute", "sum", "average", "+", "-"]):
            return {"strategy": "calculator", "tool": "calculator"}
        # 默认使用 RAG
        return {"strategy": "rag", "tool": "retriever"}
