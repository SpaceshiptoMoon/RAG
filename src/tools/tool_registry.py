"""tools.tool_registry

简单的工具注册与获取实现。
"""
from typing import Dict
from .base_tool import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools.get(name)

    def list_tools(self):
        return list(self._tools.keys())
