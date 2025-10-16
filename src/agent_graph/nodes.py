

from src.agent_graph.tools import TOOLS
from src.log.log_config import logger
from src.agent_graph.state import AgentState
from typing import Any, Dict, List


def agent_node(state: AgentState) -> AgentState:
    """
    智能体节点：对最新消息进行推理并决定需调用的工具（可同时触发多个工具）

    Args:
        state: 当前智能体状态，包含 messages 列表

    Returns:
        AgentState: 更新后的状态，包含可能的 tool_calls
    """
    messages: List[Dict[str, Any]] = state.get("messages", [])
    logger.info(f"agent_node: 处理最新消息，当前消息数={len(messages)}")
    if not messages:
        logger.warning("agent_node: 无消息可处理")
        return state

    last_msg = messages[-1]
    content = last_msg.get("content", "")

    # 简单关键词触发策略：可以被替换为 LLM 输出解析
    tool_calls: List[Dict[str, Any]] = []

    if "计算" in content or "计算器" in content:
        tool_calls.append({"name": "calculator", "args": {"expression": "1+2"}, "result": None, "error": None})
    if "天气" in content or "气候" in content:
        tool_calls.append({"name": "weather", "args": {"city": "北京"}, "result": None, "error": None})
    if "搜索" in content or "查找" in content or "网络" in content:
        # 将用户原始内容作为搜索query
        tool_calls.append({"name": "web_search", "args": {"query": content}, "result": None, "error": None})
    if "RAG" in content or "检索" in content or "检索增强" in content:
        tool_calls.append({"name": "rag_query", "args": {"question": content}, "result": None, "error": None})
    if "摘要" in content or "总结" in content:
        tool_calls.append({"name": "summarize_document", "args": {"text": content, "max_length": 200}, "result": None, "error": None})

    if tool_calls:
        logger.info(f"agent_node: 触发工具调用 count={len(tool_calls)} names={[c['name'] for c in tool_calls]}")
        state["tool_calls"] = tool_calls
    else:
        # 未触发工具，直接生成回答（占位）
        answer = f"最终答案: {content}"
        messages.append({"role": "assistant", "content": answer})
        state["messages"] = messages
        state["tool_calls"] = []

    return state

def tool_node(state: AgentState) -> AgentState:
    """
    工具节点：执行所有待处理的工具调用，并将结果回填到消息队列

    Args:
        state: 当前智能体状态，包含 tool_calls 和 messages

    Returns:
        AgentState: 更新后的状态，tool_calls 清空，messages 增加工具响应
    """
    tool_calls = state.get("tool_calls", [])
    messages = state.get("messages", [])

    if not tool_calls:
        logger.info("tool_node: 无工具调用")
        return state

    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args", {})
        tool_func = next((t for t in TOOLS if getattr(t, "name", None) == tool_name), None)
        if not tool_func:
            call["result"] = None
            call["error"] = f"工具 {tool_name} 未找到"
            logger.error(f"tool_node: {call['error']}")
        else:
            try:
                logger.info(f"tool_node: 调用工具 {tool_name}")
                result = tool_func.run(**args)
                call["result"] = result.get("result")
                call["error"] = result.get("error")
                logger.info(f"tool_node: 工具 {tool_name} 返回")
            except Exception as e:
                call["result"] = None
                call["error"] = str(e)
                logger.error(f"tool_node: 工具 {tool_name} 执行异常: {e}")

        # 回填 ToolMessage 到消息队列（便于后续LLM消费或记录）
        tool_msg = {
            "role": "tool",
            "content": f"{tool_name} 执行结果: {call.get('result') or call.get('error')}",
            "tool_call_id": None,
            "tool_name": tool_name,
            "tool_args": args,
            "tool_result": call.get("result")
        }
        messages.append(tool_msg)

    state["messages"] = messages
    state["tool_calls"] = []
    return state

def router(state: AgentState) -> str:
    """
    路由节点：根据当前状态决定下一个节点（agent_node / tool_node / END）

    Args:
        state: 当前智能体状态

    Returns:
        str: 下一个节点的名称
    """
    messages = state.get("messages", [])
    if state.get("tool_calls"):
        logger.info("router: 有工具调用，切换到 tool_node")
        return "tool_node"

    if messages and messages[-1].get("role") == "assistant" and "最终答案" in messages[-1].get("content", ""):
        logger.info("router: 达到结束条件，流程终止")
        return "END"

    logger.info("router: 继续执行 agent_node")
    return "agent_node"
