from src.vector.vectorstore import DocumentVectorizer
from src.knowlage_agent import KnowledgeAgentSystem
vector_store_path = "vector_db" 
knowledge_system = KnowledgeAgentSystem(vector_store_path)
result = knowledge_system.process_query(
    question="帮我从数据文档中检索出上海实业公司信息",
    top_k=5
)
print(result)