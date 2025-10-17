
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

class ToolCall(TypedDict):
    """
    工具调用轨迹结构，记录工具调用的完整信息
    """
    name: str  # 工具名称
    args: Dict[str, Any]  # 工具参数
    result: Any  # 执行结果
    error: Optional[str]  # 错误信息（如果有）
    thought: Optional[str]  # 调用工具的思考过程

class AgentState(TypedDict):
    """
    智能体状态结构
    包含:
    1. 消息历史 - 包括用户输入、AI回应、工具调用等
    2. 工具调用记录 - 当前待执行的工具调用
    3. 当前计划 - 子任务列表与进度
    """
    messages: List[BaseMessage]  # 使用 LangChain 消息类型
    tool_calls: List[ToolCall]  # 当前待执行的工具调用
    plan: Optional[Dict[str, Any]]  # 当前执行计划，包含子任务列表与进度
