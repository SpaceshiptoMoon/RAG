from src.vector.vectorstore import DocumentVectorizer
from src.knowlage_agent import KnowledgeAgentSystem
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
vector_store_path = "vector_db" 
knowledge_system = KnowledgeAgentSystem(vector_store_path)
test_results = knowledge_system.vectorizer.vector_store.similarity_search("上海", k=3)
print(f"测试查询结果数量: {len(test_results)}")

# test_results = knowledge_system.vectorizer.vector_store.similarity_search("上海", k=3)
# print(f"测试查询结果数量: {len(test_results)}")
# result = knowledge_system.process_query(
#     question="帮我从数据文档中检索出上海实业公司信息",
#     top_k=5
# )
# print(result)