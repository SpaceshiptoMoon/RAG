<<<<<<< HEAD
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent_graph.tools import TOOLS
from src.agent_graph.utils import logger
from src.agent_graph.state import AgentState, ToolCall
from typing import Any, Dict, List, Optional
import os
import json
from dotenv import load_dotenv
import re

def create_agent_prompt() -> ChatPromptTemplate:
    """创建Agent提示模板"""
    return ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的AI助手，遵循以下思考流程：

            1. 意图识别：分析用户需求，提取关键信息
            2. 计划制定：分解为2-5个子任务，明确是否需要工具
            3. 执行决策：选择工具或直接回答
            4. 结果验证：检查执行结果，必要时调整计划

            输出格式：
            思考：[分析意图和需求]
            计划：[列出子任务]
            行动：[使用工具或生成回答]
            """),
        MessagesPlaceholder(variable_name="messages"),
    ])

def agent_node(state: AgentState) -> AgentState:
    """
    简化且稳定的 agent 节点实现：

    - 不再依赖外部 LLM 执行工具调用判定（避免不稳定）
    - 使用简单的规则解析用户最新的 HumanMessage，识别是否需要调用工具（计算/天气）
    - 若不需要工具，则直接生成一个 AIMessage 作为最终答案
    """
    messages = state.get("messages", []) or []

    # 只保留标准消息类型
    messages = [m for m in messages if isinstance(m, (HumanMessage, AIMessage, ToolMessage))]

    logger.info(f"AgentNode: 开始处理，当前消息数 {len(messages)}")

    # 判断是否有工具结果（ToolMessage），如果有则生成最终答案
    last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    if last_tool is not None:
        # 工具结果交给 agent_node，由 agent_node 生成最终答案
        if last_tool.name == "calculator":
            a = last_tool.args.get("a")
            b = last_tool.args.get("b")
            op = last_tool.args.get("operation")
            result_str = last_tool.result if last_tool.result is not None else last_tool.content
            answer = f"最终答案：{a} {op} {b} = {result_str}"
        elif last_tool.name == "weather":
            city = last_tool.args.get("city")
            result_str = last_tool.result if last_tool.result is not None else last_tool.content
            answer = f"最终答案：{city}天气：{result_str}"
        else:
            result_str = last_tool.result if last_tool.result is not None else last_tool.content
            answer = f"最终答案：{last_tool.name}结果：{result_str}"
        messages.append(AIMessage(content=answer))
        state["messages"] = messages
        state["tool_calls"] = []
        logger.info("AgentNode: 工具结果已处理，生成最终答案")
        return state

    # 找到最近的用户输入
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        # 无用户输入，直接返回
        logger.warning("AgentNode: 未发现 HumanMessage，直接返回 state")
        state["messages"] = messages
        state["tool_calls"] = []
        return state

    text = (last_human.content or "").strip()

    tool_calls: List[ToolCall] = []

    # 1) 简单的加法识别，例如："1加2"、"1 + 2"、"1加 2 等于多少"
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:加|\+)\s*([+-]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            a = float(m.group(1))
            b = float(m.group(2))
            tool_calls.append({
                "name": "calculator",
                "args": {"a": a, "b": b, "operation": "add"},
                "result": None,
                "error": None,
                "thought": f"检测到加法请求: {a} + {b}"
            })
            state["tool_calls"] = tool_calls
            state["messages"] = messages
            logger.info("AgentNode: 识别到加法请求，准备调用 calculator 工具")
            return state
        except Exception as e:
            logger.error(f"AgentNode: 解析数字失败: {e}")

    # 2) 天气查询识别：包含关键词 '天气'，尝试提取城市名（尽量简单）
    if "天气" in text:
        # 尝试从文本中抽取城市名（尽量使用连续中文字符作为候选）
        chinese_words = re.findall(r"[\u4e00-\u9fff]+", text)
        city = None
        for w in chinese_words:
            if w in ("天气", "今天", "怎样", "怎么样", "现在"):
                continue
            # 简单排除短词
            if 1 < len(w) <= 6:
                city = w
                break

        if city is None:
            # 如果没能解析出城市，默认使用通用城市 '北京'
            city = "北京"

        tool_calls.append({
            "name": "weather",
            "args": {"city": city},
            "result": None,
            "error": None,
            "thought": f"检测到天气查询，城市: {city}"
        })
        state["tool_calls"] = tool_calls
        state["messages"] = messages
        logger.info(f"AgentNode: 识别到天气请求，准备调用 weather 工具 (city={city})")
        return state

    # 其他情况：直接产生一个简单的回答（作为最终答案）
    reply = f"最终答案：我收到你的问题：{text}。如果你想要我计算或查询天气，请直接写类似 '1加2' 或 '北京天气'."
    messages.append(AIMessage(content=reply))
    state["messages"] = messages
    state["tool_calls"] = []
    logger.info("AgentNode: 未检测到工具请求，生成直接回答")
=======


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

>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3
    return state

def tool_node(state: AgentState) -> AgentState:
    """
<<<<<<< HEAD
    工具节点：执行工具调用并记录结果

    1. 执行每个待处理的工具调用
    2. 记录执行结果
    3. 添加工具消息到对话历史
    4. 更新执行计划进度
    """
    tool_calls = state.get("tool_calls", []) or []
    messages = state.get("messages", []) or []
    plan = state.get("plan", {}) or {}
=======
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
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3

    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args", {})
<<<<<<< HEAD
        logger.info(f"ToolNode: 执行工具 {tool_name}，参数: {args}")

        # 查找工具：支持 tool 对象（有 .name 或 .__name__）或普通可调用
        tool_func = None
        for t in TOOLS:
            tname = getattr(t, "name", None) or getattr(t, "__name__", None)
            if tname == tool_name:
                tool_func = t
                break

        if tool_func is None:
            call["result"] = None
            call["error"] = f"工具 {tool_name} 未找到"
            logger.error(call["error"])
        else:
            try:
                # langchain tool 对象必须用 run(dict)，普通函数用 **kwargs
                if hasattr(tool_func, "run") and callable(getattr(tool_func, "run")):
                    raw = tool_func.run(args)
                else:
                    raw = tool_func(**args)

                # 兼容工具返回值：如果是 dict 且包含 result/error 字段，则按字段取值
                if isinstance(raw, dict):
                    call["result"] = raw.get("result") if "result" in raw else raw
                    call["error"] = raw.get("error")
                else:
                    call["result"] = raw
                    call["error"] = None

                logger.info(f"ToolNode: 工具 {tool_name} 执行成功，结果: {call['result']}")
            except Exception as e:
                call["result"] = None
                call["error"] = str(e)
                logger.error(f"ToolNode: 工具 {tool_name} 执行异常: {e}")

        # 只添加工具消息到对话历史，不自动生成 AIMessage
        messages.append(ToolMessage(
            content=f"{tool_name} 执行结果: {call['result'] if call['result'] is not None else call['error']}",
            tool_call_id=str(id(call)),
            name=tool_name,
            args=args,
            result=call.get("result")
        ))

        # 更新计划进度（如果有计划）
        if plan and "current_step" in plan:
            try:
                plan["current_step"] = int(plan.get("current_step", 0)) + 1
            except Exception:
                plan["current_step"] = 0

    # 保存状态并清空待执行的工具调用
    state["messages"] = messages
    state["tool_calls"] = []
    if plan:
        state["plan"] = plan

=======
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
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3
    return state

def router(state: AgentState) -> str:
    """
<<<<<<< HEAD
    路由节点：确定下一步行动
    
    决策逻辑:
    1. 有工具调用 -> 工具节点
    2. 计划完成 -> 结束
    3. 其他情况 -> 继续思考
    """
    messages = state["messages"]
    plan = state.get("plan", {})
    
    # 检查是否有待执行的工具调用
    if state.get("tool_calls"):
        logger.info("Router: 发现工具调用，转到tool_node")
        return "tools"  # 与图中节点名称保持一致
    
    # 检查是否达到结束条件
    last_msg = messages[-1] if messages else None
    is_final_answer = (
        isinstance(last_msg, AIMessage) and
        ("最终答案" in last_msg.content or "结论" in last_msg.content)
    )
    if is_final_answer:
        logger.info("Router: 检测到最终答案，流程结束")
        return "END"
    
    # 检查计划完成情况
    if plan:
        total_steps = len(plan.get("tasks", []))
        current_step = plan.get("current_step", 0)
        if current_step >= total_steps:
            logger.info("Router: 计划已完成，流程结束")
            return "END"
    
    # 继续思考
    logger.info("Router: 继续agent推理")
    return "agent"  # 与图中节点名称保持一致
=======
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
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3
