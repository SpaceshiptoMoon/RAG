
import ast
import httpx
from langchain_core.tools import tool
from typing import Dict, Any

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

TOOLS = [calculator, weather]
