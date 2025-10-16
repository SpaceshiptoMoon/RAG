
from typing import TypedDict, Annotated, List, Dict, Any, Optional

class ToolCall(TypedDict):
    """
    工具调用轨迹结构
    """
    name: str
    args: Dict[str, Any]
    result: Any
    error: Optional[str]

class Message(TypedDict):
    """
    对话消息结构
    """
    role: str  # 'user' | 'assistant' | 'tool'
    content: str
    tool_call_id: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_result: Optional[Any]

class AgentState(TypedDict):
    """
    智能体状态结构，包含消息列表和工具调用轨迹
    """
    messages: Annotated[List[Message], "add_messages"]
    tool_calls: List[ToolCall]
