from src.agent_graph.main import run_agent
queries = [
    "1加2等于多少?",
    "北京今天天气怎么样?",
    "你是谁?"
]

for query in queries:
    print(f"\n处理查询: {query}")
    result = run_agent(query)
    
    # 输出执行计划
    if result["plan"]:
        print("\n执行计划:")
        for i, task in enumerate(result["plan"].get("tasks", []), 1):
            print(f"{i}. {task}")
            
    # 输出工具调用
    if result["tool_calls"]:
        print("\n工具调用:")
        for call in result["tool_calls"]:
            print(f"- {call['name']}: {call['result']}")
            
    # 输出最终答案
    print(f"\n最终答案: {result['answer']}")