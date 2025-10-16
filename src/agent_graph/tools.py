
import ast
import httpx
import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from src.rag.rag_system import RAGSystem
from src.models.llm import get_llm

loger = get_llm(__name__)
def safe_eval(expr: str) -> Any:
    """
    使用 ast.literal_eval 安全求值
    """
    try:
        return ast.literal_eval(expr)
    except Exception as e:
        return f"计算错误: {e}"


@tool(
    name="calculator",
    description="安全表达式计算器，支持加减乘除和括号。参数: expression(str)",
    args_schema={"expression": str}
)
def calculator(expression: str) -> Dict[str, Any]:
    """
    安全计算表达式，结构化输出
    """
    if not isinstance(expression, str) or not expression:
        return {"error": "参数 expression 必须为非空字符串"}
    result = safe_eval(expression)
    return {"result": result}


@tool(
    name="weather",
    description="天气查询工具，参数: city(str)",
    args_schema={"city": str}
)
def weather(city: str) -> Dict[str, Any]:
    """
    查询指定城市天气，结构化输出
    """
    if not isinstance(city, str) or not city:
        return {"error": "参数 city 必须为非空字符串"}
    try:
        resp = httpx.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return {"result": resp.text}
    except Exception as e:
        return {"error": f"天气查询失败: {e}"}


@tool(
    name="web_search",
    description="联网搜索（DuckDuckGo Instant Answer），参数: query(str)",
    args_schema={"query": str}
)
def web_search(query: str) -> Dict[str, Any]:
    """
    使用 DuckDuckGo Instant Answer API 进行简单联网搜索，返回简要摘要
    Args:
        query: 搜索关键词

    Returns:
        Dict[str, Any]: 包含 result 或 error
    """
    if not isinstance(query, str) or not query:
        return {"error": "参数 query 必须为非空字符串"}
    try:
        # DuckDuckGo Instant Answer API（无API key）
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = httpx.get("https://api.duckduckgo.com/", params=params, timeout=8)
        data = resp.json()
        # 首选 AbstractText，否则尝试 RelatedTopics 的文本
        summary = data.get("AbstractText")
        if not summary:
            related = data.get("RelatedTopics", [])
            if related and isinstance(related, list):
                # 寻找第一个含有文本的条目
                for item in related:
                    if isinstance(item, dict) and item.get("Text"):
                        summary = item.get("Text")
                        break
        if not summary:
            summary = data.get("Heading") or "未找到摘要"
        return {"result": summary}
    except Exception as e:
        return {"error": f"web_search 失败: {e}"}


@tool(
    name="rag_query",
    description="在本地知识库上执行RAG查询，参数: question(str), data_path(opt str), collection_name(opt str)",
    args_schema={"question": str, "data_path": Optional[str], "collection_name": Optional[str]}
)
def rag_query(question: str, data_path: Optional[str] = None, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    在本地知识库上执行检索增强生成（RAG）查询。

    Args:
        question: 查询问题
        data_path: 可选的数据路径，若为空使用默认 './data'
        collection_name: 可选集合名称，默认为 'documents'

    Returns:
        Dict[str, Any]: RAG 查询结果（以 RAGSystem.query 返回结构为准）
    """
    if not isinstance(question, str) or not question:
        return {"error": "参数 question 必须为非空字符串"}
    try:
        data_path = data_path or os.getenv("DATA_PATH") or "data"
        collection_name = collection_name or (os.getenv("RAG_COLLECTION") or "documents")
        rag = RAGSystem(data_path=data_path, collection_name=collection_name)
        res = rag.query(question)
        return {"result": res}
    except Exception as e:
        return {"error": f"rag_query 失败: {e}"}


@tool(
    name="summarize_document",
    description="对文本进行摘要，参数: text(str), max_length(opt int)",
    args_schema={"text": str, "max_length": Optional[int]}
)
def summarize_document(text: str, max_length: Optional[int] = 200) -> Dict[str, Any]:
    """
    使用当前配置的 LLM 对文本进行摘要。

    Args:
        text: 待摘要的文本
        max_length: 可选的最大摘要长度提示

    Returns:
        Dict[str, Any]: 包含 result 或 error
    """
    if not isinstance(text, str) or not text:
        return {"error": "参数 text 必须为非空字符串"}
    try:
        llm = get_llm()
        prompt = f"请用中文简明扼要地总结以下文本（控制在 {max_length} 字以内）：\n\n{text}"
        # get_llm 返回的对象上可能有 chat 方法
        if hasattr(llm, "chat"):
            summary = llm.chat(prompt)
        else:
            # 兼容一些 wrapper，尝试直接调用
            summary = str(llm)
        return {"result": summary}
    except Exception as e:
        return {"error": f"summarize_document 失败: {e}"}


TOOLS = [calculator, weather, web_search, rag_query, summarize_document]
