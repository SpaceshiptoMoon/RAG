"""
- 智能查询路由模块
功能：根据问题类型自动选择最佳回答策略
"""

from typing import Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """路由决策模型"""
    strategy: Literal["direct", "rag", "agent"] = Field(
        ...,
        description="路由策略：direct-直接回答, rag-知识库检索, agent-联网搜索"
    )
    confidence: float = Field(
        default=0.5,  # 添加默认值
        description="决策置信度(0-1)"
    )

class QueryRouter:
    """智能查询路由器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.structured_llm_router = llm.with_structured_output(RouteDecision)
        
        # 路由提示模板
        self.route_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的查询分类专家。请根据用户问题选择最合适的回答策略。请确保输出为有效的JSON格式，并且必须包含 strategy 和 confidence 两个字段。
                
                策略选择规则，只能在这三种策略中选一个，不能出现其他策略：
                1. direct策略：简单事实性问题，模型本身知识足以回答
                2. rag策略：涉及特定文档、私有知识的问题
                3. agent策略：需要最新信息、实时数据的问题

                返回格式必须严格遵循：
                {{
                    "strategy": "direct|rag|agent",
                    "confidence": 0.0-1.0之间的数值
                }}
                
                请分析问题并选择最佳策略，返回完整的JSON格式结果。"""),
                ("human", "用户问题: {question}")
            ])
        
        self.router_chain = self.route_prompt | self.structured_llm_router
    
    def route_query(self, question: str) -> RouteDecision:
        """路由用户查询"""
        try:
            decision = self.router_chain.invoke({"question": question})
            print(decision)
            logger.info(f"查询路由决策: {decision.strategy}, 置信度: {decision.confidence}")
            return decision
        except Exception as e:
            logger.error(f"路由查询失败: {e}")
            # 默认回退策略
            return RouteDecision(strategy="rag", confidence=0.5)
        
        
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os 
    load_dotenv()
    api_key = os.getenv("OPENAI_AI_EMBEDDING_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_LLM_MODEL_NAME")
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
    # response = llm.invoke([{  
    #                 "role": "user",  
    #                 "content": "编写Python异步爬虫教程，包含代码示例和注意事项"  
    #             }],  
    #             temperature=0.7,  
    #             max_tokens=10)  
    # print(response)
    
    router = QueryRouter(llm)
    question = '你好'
    res = router.route_query(question)
    print(res)