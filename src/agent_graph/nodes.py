

from langchain_core.language_models import ChatOpenAI
from langchain_core.tools import ToolMessage
from src.agent_graph.tools import TOOLS
from src.agent_graph.utils import logger
from src.agent_graph.state import AgentState, Message, ToolCall
from typing import Any, Dict

def agent_node(state: AgentState) -> AgentState:
    """
    智能体节点：推理、决定是否调用工具
    """
    llm = ChatOpenAI().bind_tools(TOOLS)
    messages = state["messages"]
    # 推理：调用 LLM，判断是否需要工具调用
    logger.info(f"AgentNode: 当前消息数 {len(messages)}")
    # 这里只做伪推理，实际应调用 llm.invoke(messages)
    last_msg = messages[-1]
    if "计算" in last_msg["content"]:
        # 触发计算器工具
        tool_calls = [{
            "name": "calculator",
            "args": {"expression": "1+2"},
            "result": None,
            "error": None
        }]
        logger.info("AgentNode: 触发 calculator 工具调用")
        state["tool_calls"] = tool_calls
    elif "天气" in last_msg["content"]:
        tool_calls = [{
            "name": "weather",
            "args": {"city": "北京"},
            "result": None,
            "error": None
        }]
        logger.info("AgentNode: 触发 weather 工具调用")
        state["tool_calls"] = tool_calls
    else:
        # 直接生成最终答案
        answer = f"最终答案: {last_msg['content']}"
        messages.append({"role": "assistant", "content": answer})
        state["messages"] = messages
        state["tool_calls"] = []
    return state

def tool_node(state: AgentState) -> AgentState:
    """
    工具节点：执行工具调用并回填 ToolMessage
    """
    tool_calls = state.get("tool_calls", [])
    messages = state["messages"]
    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]
        # 查找工具
        tool_func = next((t for t in TOOLS if t.name == tool_name), None)
        if tool_func:
            try:
                result = tool_func.run(**args)
                call["result"] = result.get("result")
                call["error"] = result.get("error")
                logger.info(f"ToolNode: 工具 {tool_name} 执行成功，结果: {result}")
            except Exception as e:
                call["result"] = None
                call["error"] = str(e)
                logger.error(f"ToolNode: 工具 {tool_name} 执行异常: {e}")
        else:
            call["result"] = None
            call["error"] = f"工具 {tool_name} 未找到"
            logger.error(f"ToolNode: 工具 {tool_name} 未找到")
        # 回填 ToolMessage 到消息
        tool_msg = {
            "role": "tool",
            "content": f"{tool_name} 执行结果: {call['result'] or call['error']}",
            "tool_call_id": None,
            "tool_name": tool_name,
            "tool_args": args,
            "tool_result": call["result"]
        }
        messages.append(tool_msg)
    state["messages"] = messages
    state["tool_calls"] = []  # 工具调用已处理
    return state

def router(state: AgentState) -> str:
    """
    路由节点：判断是否需要工具调用或结束
    """
    messages = state["messages"]
    # 如果有 tool_calls，进入工具节点
    if state.get("tool_calls"):
        logger.info("Router: 进入 tool_node")
        return "tool_node"
    # 如果最后一条消息是 assistant 且包含最终答案，则结束
    if messages and messages[-1]["role"] == "assistant" and "最终答案" in messages[-1]["content"]:
        logger.info("Router: 达到结束条件，流程终止")
        return "END"
    # 否则继续 agent_node
    logger.info("Router: 继续 agent_node 推理")
    return "agent_node"
