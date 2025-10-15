# src/rag/generator.py
from typing import List, Dict, Any
from langchain.schema import BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class AnswerGenerator:
    """
    基于检索结果的答案生成器
    """
    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model_name=model_name
        )
        self.setup_prompts()
    
    def setup_prompts(self):
        """设置提示词模板"""
        self.qa_prompt = PromptTemplate(
            template="""请根据以下上下文信息回答问题。如果上下文信息不足以回答问题，请如实告知。
            
上下文信息：
{context}

问题：{question}

请提供准确、详细的回答：""",
            input_variables=["context", "question"]
        )
    
    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于检索到的上下文生成答案
        """
        if not context_docs:
            return {
                "answer": "抱歉，我没有找到相关的信息来回答这个问题。",
                "sources": [],
                "confidence": 0.0
            }
        
        # 构建上下文
        context = self._format_context(context_docs)
        
        # 生成答案
        try:
            response = self.llm.predict(
                self.qa_prompt.format(context=context, question=question)
            )
            
            return {
                "answer": response,
                "sources": [doc.get("source", "未知来源") for doc in context_docs],
                "confidence": self._calculate_confidence(context_docs),
                "context_docs": context_docs
            }
        except Exception as e:
            return {
                "answer": f"生成答案时出现错误：{str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """格式化上下文文档"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "未知文档")
            text = doc.get("text", "")[:500]  # 限制长度
            context_parts.append(f"[文档 {i} - 来源: {source}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """计算回答置信度（简化版）"""
        if not docs:
            return 0.0
        # 基于检索结果的数量和质量计算置信度
        base_confidence = min(len(docs) * 0.2, 1.0)
        return round(base_confidence, 2)