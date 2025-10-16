"""tools.calculator_tool

简单的计算工具，支持 eval 风格的表达式（仅限受信任环境）。
"""
from typing import Any
from .base_tool import BaseTool


class CalculatorTool(BaseTool):
    def __init__(self, name: str = "calculator"):
        super().__init__(name)

    def call(self, expression: str) -> Any:
        # 注意：使用 eval 有安全风险，这里仅作示例
        return eval(expression, {})
