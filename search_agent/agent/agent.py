import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.agents import tool, initialize_agent, AgentType

from src.agent.tools import google_search



class DeepAgent:
    def __init__(self):
        load_dotenv()
        llm = Ollama(
            model=os.getenv("OLLAMA_LLM_MODEL_NAME", "llama2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        tools = [google_search]
        
        # 更简单的提示模板
        prompt = PromptTemplate(
            template="""请回答以下问题。你可以使用以下工具：

            {tools}

            请按照以下格式思考：

            问题: {input}
            思考: 我应该做什么
            行动: {tool_names} 中的一个
            行动输入: 搜索查询
            观察: 搜索结果
            ... (这个思考/行动/观察过程可以重复)
            思考: 我现在知道最终答案了
            最终答案: 问题的答案

            开始！

            问题: {input}
            思考:{agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 使用结构化聊天代理
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def query(self, query: str) -> str:
        try:
            result = self.agent_executor.run(query)
            return result
        except Exception as e:
            return f"执行错误: {str(e)}"
        
        
if __name__ == "__main__":
    agent = DeepAgent()
    result = agent.query("请问近期携程有什么大的新闻")
    print("查询结果:", result)