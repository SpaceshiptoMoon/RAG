from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent_graph.tools import TOOLS
from src.agent_graph.utils import logger
from src.agent_graph.state import AgentState
from src.models.llm import get_llm
import json

llm = get_llm().llm
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
    意图识别节点：使用 LLM 进行意图识别

    行为:
    - 读取最新的 `HumanMessage`
    - 使用 LLM 分析用户意图，识别是否需要调用工具
    - 将识别出的 `tool_calls` 放入 state 中，但不执行工具
    - 如果未识别出工具调用，生成直接回复，并清空 `tool_calls`

    Args:
        state (AgentState): 当前代理状态，包含 `messages`, `tool_calls`, `plan` 等字段

    Returns:
        AgentState: 更新后的状态，包含 `tool_calls`（如果有）和 `messages`
    """
    messages = state.get("messages", []) or []
    messages = [m for m in messages if isinstance(m, (HumanMessage, AIMessage, ToolMessage))]
    logger.info(f"AgentNode: LLM意图识别，当前消息数 {len(messages)}")

    # 找到最近的用户输入
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human is None:
        logger.warning("AgentNode: 未发现 HumanMessage，保持现有 state")
        state["messages"] = messages
        state["tool_calls"] = []
        return state

    # 准备工具描述
    tools_desc = []
    for tool in TOOLS:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        desc = getattr(tool, "description", None) or "无描述"
        tools_desc.append(f"- {name}: {desc}")

    # 创建 LLM
    try:
        llm = get_llm().llm
    except Exception as e:
        logger.error(f"AgentNode: LLM 初始化失败: {e}")
        state["messages"] = messages
        state["tool_calls"] = []
        return state

    # 意图识别提示
    prompt = f"""分析用户输入，判断是否需要使用工具。可用工具列表：
{chr(10).join(tools_desc)}

用户输入: {last_human.content}

请用 JSON 格式输出你的分析结果：
{{
    "需要工具": true/false,
    "选择工具": "工具名称（如果需要）",
    "工具参数": {{"参数名": "参数值"}},
    "思考过程": "分析和推理说明"
}}
"""
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content)
        state["messages"].append(response)
        
        if result.get("需要工具", False):
            tool_name = result.get("选择工具")
            tool_args = result.get("工具参数", {})
            thought = result.get("思考过程", "")
            
            # 验证工具存在性
            if any(getattr(t, "name", None) == tool_name or getattr(t, "__name__", None) == tool_name for t in TOOLS):
                state["tool_calls"] = [{
                    "name": tool_name,
                    "args": tool_args,
                    "result": None,
                    "error": None,
                    "thought": thought
                }]
                logger.info(f"AgentNode: LLM识别到工具调用需求: {tool_name}")
            else:
                logger.warning(f"AgentNode: LLM建议的工具 {tool_name} 不存在")
                messages.append(AIMessage(content=f"我理解你可能需要 {tool_name} 工具，但该工具当前不可用。请尝试其他方式。"))
        else:
            messages.append(AIMessage(content=result.get("思考过程", "我理解你的问题不需要使用工具，请问还有什么可以帮你的吗？")))
            logger.info("AgentNode: LLM判断不需要工具调用")
            
    except Exception as e:
        logger.error(f"AgentNode: LLM分析失败: {e}")
        messages.append(AIMessage(content="抱歉，我在分析你的需求时遇到了问题。请用更清晰的方式描述，或稍后重试。"))
        state["tool_calls"] = []
    
    state["messages"] = messages
    return state

def tool_node(state: AgentState) -> AgentState:
    """
    工具节点：执行工具调用并记录结果

    1. 执行每个待处理的工具调用
    2. 记录执行结果
    3. 添加工具消息到对话历史
    4. 更新执行计划进度
    """
    tool_calls = state.get("tool_calls", []) or []
    messages = state.get("messages", []) or []
    plan = state.get("plan", {}) or {}

    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args", {})
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

                logger.info(f"ToolNode: 工具 {tool_name} 执行成功!")
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

    return state


def audit_node(state: AgentState) -> AgentState:
    """
    审核节点：使用 LLM 审核生成的回复是否满足用户意图

    行为:
    - 分析最近的用户意图和生成的回复
    - 使用 LLM 评估回复是否满足用户需求
    - 如果不满足，将状态返回到 agent 节点继续处理
    - 如果满足，保持当前状态继续执行

    Args:
        state (AgentState): 当前状态，包含消息历史和工具调用结果

    Returns:
        AgentState: 更新后的状态，包含审核结果标记
    """
    messages = state.get("messages", []) or []
    
    # 获取最近的用户输入和系统回复
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    tool_msge = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    
    if not last_human or not last_ai:
        logger.warning("AuditNode: 缺少用户输入或系统回复，跳过审核")
        state["audit_passed"] = True
        return state
    
    # 准备 LLM
    try:
        llm = get_llm().llm
    except Exception as e:
        logger.error(f"AuditNode: LLM 初始化失败: {e}")
        state["audit_passed"] = True  # 如果 LLM 失败，默认通过
        return state
    
    # 构建审核提示
    prompt = f"""作为回复质量审核员，请评估系统的回复是否充分满足用户的需求。

用户原始输入: {last_human.content}

工具的调用: {tool_msge.content}

系统的回复: {last_ai.content}

请分析回复是否满足以下标准：
1. 是否准确理解用户意图
2. 是否完整回答用户问题
3. 是否需要更多工具调用或信息收集
4. 是否达到用户预期的最终目标

请用 JSON 格式输出你的评估结果：
{{
    "满意度": 0-100,  # 评分，反映满足用户需求的程度
    "需要继续": true/false,  # 是否需要更多处理
    "原因": "详细解释为什么满足或不满足",
    "建议": "如果需要继续，建议下一步操作"
}}
"""
    
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content)
        
        satisfaction = result.get("满意度", 0)
        needs_continue = result.get("需要继续", True)
        reason = result.get("原因", "未提供原因")
        suggestion = result.get("建议", "")
        
        audit_result = {
            "satisfaction": satisfaction,
            "proceed": needs_continue,
            "reason": reason,
            "suggestion": suggestion
            }

        state["audit"].append(audit_result)

        logger.info(f"AuditNode: 满意度 {satisfaction}/100, 需要继续: {needs_continue}")
        logger.info(f"AuditNode: 原因: {reason}")
        
        if needs_continue or satisfaction < 60:  # 设置满意度阈值
            # 添加审核反馈到消息历史
            feedback = f"我觉得当前回答还不够完善：{reason}\n建议：{suggestion}"
            messages.append(AIMessage(content=feedback))
            logger.info("AuditNode: 审核未通过，需要继续完善")
        else:
            logger.info("AuditNode: 审核通过，回复满足用户需求")
            
    except Exception as e:
        logger.error(f"AuditNode: 审核过程出错: {e}")
    return state


def generate_node(state: AgentState) -> AgentState:
    """
    生成节点：使用 LLM 基于工具执行结果生成自然语言回复。

    行为:
    1. 收集对话历史和工具结果
    2. 使用 LLM 理解工具结果并生成自然语言回复
    3. 确保回复针对用户原始问题，并利用所有相关工具结果

    Args:
        state (AgentState): 当前状态，包含消息历史和工具调用结果

    Returns:
        AgentState: 更新后的状态，包含生成的回复
    """
    messages = state.get("messages", []) or []
    
    # 获取最近的用户问题和工具结果
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    
    if not tool_msgs:
        logger.info("GenerateNode: 未发现工具结果，使用默认回复")
        tool_msgs = []
        
    if not last_human:
        logger.warning("GenerateNode: 未找到用户问题，使用默认回复")
        state["messages"] = messages
        return state
    
    # 准备 LLM
    try:
        llm = get_llm().llm
    except Exception as e:
        logger.error(f"GenerateNode: LLM 初始化失败: {e}")
        # 降级为简单拼接
        summary = "\n".join(f"[{getattr(tm, 'name', 'tool')}] {tm.content}" for tm in tool_msgs)
        messages.append(AIMessage(content=f"工具执行结果：\n{summary}"))
        state["messages"] = messages
        return state
    
    # 构建上下文提示
    tool_results = []
    if tool_msgs:
        for tm in tool_msgs:
            name = getattr(tm, "name", "tool")
            result = getattr(tm, "result", None)
            args = getattr(tm, "args", {})
            tool_results.append({
                "工具": name,
                "参数": args,
                "结果": result if result is not None else tm.content
            })
    else:
        tool_results = [{"工具": "无", "参数": {}, "结果": "无"}]
    
    prompt = f"""作为AI助手，请基于以下信息生成一个清晰的回复：

用户原始问题：{last_human.content}

工具调用结果：
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

请根据以上信息：
1. 理解工具返回的原始数据
2. 将技术结果转化为用户友好的自然语言
3. 确保回复直接针对用户的原始问题
4. 如果发现结果不完整或需要补充，可以在回复中说明

回复格式：
{{
    "回复内容": "完整的自然语言回复",
    "是否完整": true/false,  # 当前结果是否足够回答用户问题
    "下一步建议": "如果不完整，建议接下来需要获取什么信息"
}}"""

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content)
        
        # 添加生成的回复
        reply = result.get("回复内容", "抱歉，我无法生成有效的回复。")
        is_complete = result.get("是否完整", False)
        next_step = result.get("下一步建议", "")
        
        if not is_complete and next_step:
            reply = f"{reply}\n\n为了更好地回答您的问题，我建议：{next_step}"
        
        messages.append(AIMessage(content=reply))
        logger.info(f"GenerateNode: 已生成回复，完整性: {is_complete}")
        
    except Exception as e:
        logger.error(f"GenerateNode: 生成回复失败: {e}")
        messages.append(AIMessage(content="抱歉，我在处理结果时遇到了问题。让我们尝试用其他方式回答您的问题。"))
    
    state["messages"] = messages
    return state

def router(state: AgentState) -> str:
    """
    路由节点：确定下一步行动
    
    决策逻辑:
    1. 如果审核未通过 -> 返回 agent 节点重新处理
    2. 有工具调用 -> 工具节点
    3. 达到终止条件 -> 结束
    4. 其他情况 -> 继续思考
    """
    # 检查是否有待执行的工具调用
    if state.get("tool_calls"):
        logger.info("Router: 发现工具调用，转到tool_node")
        return "tools"
    
    logger.info("Router: 不需要使用工具，转到generate节点")
    return "generate"


def audit_router(state: AgentState) -> str:
    """
    路由节点：确定下一步行动
    
    决策逻辑:
    1. 如果审核未通过 -> 返回 agent 节点重新处理
    2. 达到终止条件 -> 结束
    """
    audit_passed = state.get("audit")[-1].get("proceed")
    
    # 优先检查审核结果
    if audit_passed:
        logger.info("audit_router: 审核通过，继续agent推理")
        return "end"
    else:
        logger.info("audit_router: 审核未通过，返回agent节点重新处理")
        return "agent"
    
