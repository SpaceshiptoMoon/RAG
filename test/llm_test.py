from src.rag.generator import AnswerGenerator
from src.models.llm import OpenAIModel, get_llm

llm = get_llm('qwen')
generator = AnswerGenerator(llm=llm)
result = generator.generate_answer("什么是人工智能？", [{"source": "文档1", "text": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"}])
print(result)