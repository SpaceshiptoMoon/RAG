"""tools.base_tool

定义工具基础接口。工具可被 ToolRegistry 注册并由 ToolAgent 调用。
"""
from typing import Any, Dict


class BaseTool:
    """基础工具接口，子类需实现 call 方法。

    该基类还定义了：
    - validate_args: 在执行前检查参数的合法性，默认不抛出。子类可重写并在参数非法时抛出 ValueError。
    - idempotent: 标记工具是否幂等（可安全缓存调用结果）。
    """
    def __init__(self, name: str):
        self.name = name
        # 是否幂等，默认 True。非幂等工具（例如对外写入的 HTTP POST）应设置为 False。
        self.idempotent: bool = True

    def validate_args(self, *args, **kwargs) -> None:
        """验证输入参数；子类在参数非法时应抛出 ValueError。"""
        return None

    def call(self, *args, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑，必须被子类实现。"""
        raise NotImplementedError()
