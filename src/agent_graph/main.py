
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, START

from src.agent_graph.state import AgentState
from src.agent_graph.nodes import agent_node, tool_node, router,audit_router, audit_node, generate_node
from src.agent_graph.utils import logger

def create_workflow():
    workflow = StateGraph(AgentState)
    
    # 添加所有节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("generate", generate_node)
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 关键修复：添加入口边 - 从 START 到 agent
    workflow.add_edge(START, "agent")  # 添加这一行！
    
    # Agent 节点的条件路由
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "generate": "generate",
        }
    )
    
    # Tools -> Generate 的固定边
    workflow.add_edge("tools", "generate")
    
    # Generate -> Audit 的固定边
    workflow.add_edge("generate", "audit")
    
    # Audit 节点的条件路由
    workflow.add_conditional_edges(
        "audit",
        audit_router,
        {
            "agent": "agent",  # 审核不通过，返回 agent 重新处理
            "end": END,        # 审核通过，结束流程
        }
    )
    
    return workflow.compile()

def create_initial_state(query: str) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=query)],
        tool_calls=[],
        plan=[],
        audit=[]
    )

def get_final_response(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    plan = state.get("plan")
    answer = next(
        (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
        "未能生成答案"
    )
    tool_calls = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_calls.append({
                "name": msg.name,
                "args": msg.args,
                "result": msg.result
            })
    return {
        "answer": answer,
        "plan": plan,
        "tool_calls": tool_calls,
        "messages": messages
    }

def run_agent(query: str) -> Dict[str, Any]:
    graph = create_workflow()
    state = create_initial_state(query)
    try:
        final_state = graph.invoke(state)
        
        result = get_final_response(final_state)
        logger.info("Agent执行完成")
        return result
    except Exception as e:
        logger.error(f"Agent执行失败: {e}")
        return {
            "answer": f"执行过程中出现错误: {str(e)}",
            "plan": state.get("plan"),
            "tool_calls": [],
            "messages": state.get("messages", [])
        }
