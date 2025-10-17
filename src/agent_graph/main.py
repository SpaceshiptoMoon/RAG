
<<<<<<< HEAD
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

=======
# 将原来的导入语句替换为：
from langgraph.graph import StateGraph
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3
from src.agent_graph.state import AgentState
from src.agent_graph.nodes import agent_node, tool_node, router
from src.agent_graph.tools import TOOLS
from src.agent_graph.utils import logger

<<<<<<< HEAD
def create_agent_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    config = config or {"max_iterations": 6, "timeout": 30}
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "agent": "agent",
            "END": END
        }
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
=======
def run_agent(user_input: str, max_steps: int = 8):
    """
    主流程编排，支持最大步数与结构化输出
    """
    state = AgentState(messages=[{"role": "user", "content": user_input}], tool_calls=[])
    graph = StateGraph()
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("router", router)
    graph.add_edge("agent_node", "router")
    graph.add_edge("router", "tool_node", condition=lambda s: s.get("tool_calls"))
    graph.add_edge("router", "END", condition=lambda s: not s.get("tool_calls"))
    graph.add_edge("tool_node", "agent_node")
    graph.compile()
>>>>>>> 8541714ab2b821a29b196d756e67b837af0af4d3

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

def run_agent(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    graph = create_agent_graph(config)
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