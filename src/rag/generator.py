# src/rag/generator.py

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.prompt.prompt_template import QA_PROMPT
from src.models.llm import OpenAIModel
from src.log.log_config import setup_logger

# 配置日志
logger = setup_logger(__name__)

class AnswerGenerator:
    """
    基于检索结果的答案生成器
    
    Args:
        llm: OpenAI模型实例
        
    Attributes:
        llm: 语言模型实例
        qa_prompt: 问答提示模板
    """
    def __init__(self, llm: OpenAIModel):
        self.llm = llm.llm
        self.qa_prompt = ChatPromptTemplate.from_template(QA_PROMPT)
         
    
    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于检索到的上下文生成答案
        
        Args:
            question: 用户问题
            context_docs: 检索到的相关文档列表
            
        Returns:
            Dict[str, Any]: 包含答案、来源和置信度的字典
        """
        logger.info(f"开始生成答案 - 问题: {question[:50]}...")
        if not context_docs:
            return {
                "answer": "抱歉，我没有找到相关的信息来回答这个问题。",
                "sources": [],
                "confidence": 0.0,
                "retrieved_docs": "",
                "retrieved_count": ""
            }
        
        # 构建上下文
        context = self._format_context(context_docs)
        
        # 生成答案
        try:
            qa_prompt = self.qa_prompt.invoke({"context": context, "question":question})
            logger.debug("调用语言模型生成答案")
            response = self.llm.invoke(qa_prompt)
            
            result = {
                "answer": response,
                "sources": [doc.get("metadata").get('source_file', '未知来源') for doc in context_docs],
                "confidence": self._calculate_confidence(context_docs),
                "retrieved_docs": context_docs,
                "retrieved_count": len(context_docs)
            }
            logger.info("答案生成完成")
            return result
            
        except Exception as e:
            logger.error(f"生成答案时出现错误: {e}")
            return {
                "answer": f"生成答案时出现错误：{str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        将检索到的文档列表格式化为 LLM 可消费的上下文字符串。

        Args:
            docs (List[Dict[str, Any]]): 检索到的文档，每个文档包含 text 和 metadata。

        Returns:
            str: 按顺序拼接的上下文字符串。
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = (doc.get("metadata") or {}).get('source_file', '未知来源')
            text = doc.get("text", "")[:500]  # 限制长度
            context_parts.append(f"[文档 {i} - 来源: {source}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """
        计算回答置信度（简化版）。

        Args:
            docs (List[Dict[str, Any]]): 用于计算置信度的检索文档列表。

        Returns:
            float: 置信度值，范围在 0.0 - 1.0 之间。
        """
        if not docs:
            return 0.0
        # 基于检索结果的数量和质量计算置信度
        base_confidence = min(len(docs) * 0.2, 1.0)
        return round(base_confidence, 2)