import ast
import httpx
from decimal import Decimal
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from typing import Union, Dict, Any, Literal
from src.log.log_config import setup_logger

logger = setup_logger(__name__)

loger = get_llm(__name__)
def safe_eval(expr: str) -> Any:
    try:
        return ast.literal_eval(expr)
    except Exception as e:
        return f"计算错误: {e}"

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

TOOLS = [calculator, weather]
