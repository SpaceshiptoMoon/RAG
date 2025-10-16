"""tools.retriever_tool

简单的检索工具示例，封装 DocumentVectorizer 查询逻辑。
"""
from typing import List, Any, Dict
from .base_tool import BaseTool
from src.rag.retriever import VectorRetriever


class RetrieverTool(BaseTool):
    def __init__(self, name: str, vector_store:VectorRetriever):
        super().__init__(name)
        self.vector_store = vector_store
        # retriever is read-only -> idempotent
        self.idempotent = True

    def validate_args(self, query: str, top_k: int = 5) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    def call(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.vector_store.retrieve(query, top_k=top_k)
