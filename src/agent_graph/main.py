
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

from src.agent_graph.state import AgentState
from src.agent_graph.nodes import agent_node, tool_node, router, audit_node, generate_node
from src.agent_graph.utils import logger

def create_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    # 新增审核与生成节点
    workflow.add_node("audit", audit_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            # 当 router 决定需要工具时，先进入 audit 节点进行复审
            "tools": "audit",
            "agent": "agent",
            "END": END
        }
    )
    # 审核通过后进入工具执行
    workflow.add_edge("audit", "tools")
    # 工具执行完成后进入生成节点，生成完返回 agent 继续或结束
    workflow.add_edge("tools", "generate")
    workflow.add_edge("generate", "agent")
    return workflow.compile()

def create_initial_state(query: str) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=query)],
        tool_calls=[],
        plan=None
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
    graph = create_agent_graph()
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

if __name__ == "__main__":
    queries = [
        "1加2等于多少?",
        "北京今天天气怎么样?",
        "你是谁?"
    ]
    for query in queries:
        print(f"\n处理查询: {query}")
        result = run_agent(query)
        if result["plan"]:
            print("\n执行计划:")
            for i, task in enumerate(result["plan"].get("tasks", []), 1):
                print(f"{i}. {task}")
        if result["tool_calls"]:
            print("\n工具调用:")
            for call in result["tool_calls"]:
                print(f"- {call['name']}: {call['result']}")
        print(f"\n最终答案: {result['answer']}")