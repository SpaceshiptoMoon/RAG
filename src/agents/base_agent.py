"""agents.base_agent

定义智能体基础接口，所有智能体应继承并实现 run/step 等方法。
"""
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """基础智能体接口。

    子类应实现 handle 方法来处理输入并返回结果。
    """
    def __init__(self, name: str = "base_agent"):
        self.name = name

    def handle(self, input: str, **kwargs) -> Dict[str, Any]:
        """处理输入并返回结构化结果。"""
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        """兼容调用接口，默认调用 handle。"""
        return self.handle(*args, **kwargs)
