import ast
import httpx
<<<<<<< HEAD
from decimal import Decimal
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from typing import Union, Dict, Any, Literal
from src.log.log_config import setup_logger

logger = setup_logger(__name__)
=======
import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from src.rag.rag_system import RAGSystem
from src.models.llm import get_llm
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3

loger = get_llm(__name__)
def safe_eval(expr: str) -> Any:
    try:
        return ast.literal_eval(expr)
    except Exception as e:
        return f"计算错误: {e}"

<<<<<<< HEAD
# 通用数值类型（支持原生 float、int、Decimal 等）
Number = Union[float, int, Decimal]

class CalculatorArgs(BaseModel):
    a: Number = Field(..., description="第一个操作数", examples=[2.0, 3.5])
    b: Number = Field(..., description="第二个操作数", examples=[1.0, -2.5])
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        ...,
        description="运算类型，支持加/减/乘/除",
        examples=["add", "divide"]
    )

    @field_validator("a", "b", mode="before")
    @classmethod
    def to_float(cls, v):
        if isinstance(v, float):
            return v
        elif isinstance(v, int):
            return float(v)
        elif isinstance(v, Decimal):
            # 避免浮点精度问题可按需保留为 Decimal，或在此转 float
            return float(v)
        else:
            raise ValueError(f"字段需要数值类型，得到 {type(v)}")
=======

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
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3


@tool(
    name_or_callable="calculator",
    description="安全表达式计算器，支持加减乘除。参数: a(Number), b(Number), operation(str)",
    args_schema=CalculatorArgs,
    # return_direct=True  # 先注释，便于排查
)
def calculator(a: float, b: float, operation: str) -> float:
    """Calculate two numbers. operation: add, subtract, multiply, divide"""
    try:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ZeroDivisionError("除数不能为零")
            return a / b
        else:
            raise ValueError(f"不支持的运算类型: {operation}")
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        raise

class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名称，非空字符串", examples=["北京", "上海"])

@tool(
    name_or_callable="weather",
    description="天气查询工具，参数: city(str)",
    args_schema=WeatherArgs,
    # return_direct=True  # 先注释，便于排查
)
def weather(city: str) -> Dict[str, Any]:
    """
    查询指定城市天气，结构化输出
    """
    if not isinstance(city, str) or not city.strip():
        return {"error": "参数 city 必须为非空字符串"}
    try:
        resp = httpx.get(f"https://wttr.in/{city.strip()}?format=3", timeout=5.0)
        resp.raise_for_status()
        return {"result": resp.text.strip()}
    except Exception as e:
        return {"error": f"天气查询失败: {e}"}

<<<<<<< HEAD
TOOLS = [calculator, weather]
=======

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
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3
