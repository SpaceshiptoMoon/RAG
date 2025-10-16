
from langgraph import Graph
from src.agent_graph.state import AgentState
from src.agent_graph.nodes import agent_node, tool_node, router
from src.agent_graph.utils import logger

def run_agent(user_input: str, max_steps: int = 8):
    """
    主流程编排，支持最大步数与结构化输出
    """
    state = AgentState(messages=[{"role": "user", "content": user_input}], tool_calls=[])
    graph = Graph()
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("router", router)
    graph.add_edge("agent_node", "router")
    graph.add_edge("router", "tool_node", condition=lambda s: s.get("tool_calls"))
    graph.add_edge("router", "END", condition=lambda s: not s.get("tool_calls"))
    graph.add_edge("tool_node", "agent_node")
    graph.compile()

    step = 0
    while step < max_steps:
        logger.info(f"==== 第{step+1}步 ====")
        next_node = router(state)
        if next_node == "END":
            logger.info("流程终止，输出最终答案")
            break
        elif next_node == "tool_node":
            state = tool_node(state)
        else:
            state = agent_node(state)
        step += 1
    # 输出结构化结果
    messages = state["messages"]
    answer = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    return {
        "messages": messages,
        "final_answer": answer
    }

if __name__ == "__main__":
    # 示例：用户输入包含“计算”或“天气”触发工具，否则直接生成答案
    user_input = input("请输入问题：")
    result = run_agent(user_input)
    print("\n==== 对话轨迹 ====")
    for msg in result["messages"]:
        print(f"[{msg['role']}] {msg['content']}")
    print("\n最终答案:", result["final_answer"])
