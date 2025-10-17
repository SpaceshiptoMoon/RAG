
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Union,  Literal
from decimal import Decimal
from src.log.log_config import setup_logger

logger = setup_logger(__name__)

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