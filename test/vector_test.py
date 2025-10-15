from src.vector.vectorstore import DocumentVectorizer
from src.vector.milvus_db import MilvusManager
from src.models.embedding import EmbeddingClient
from src.rag.retriever import VectorRetriever
from dotenv import load_dotenv
import os
load_dotenv()

host = os.getenv("MILVUS_HOST", "127.0.0.1")
port  =  os.getenv("MILVUS_PORT", "19530") 
provider = (os.getenv('EMBED_PROVIDER') or 'openai').lower()
model_name = os.getenv('EMBED_MODEL') or 'text-embedding-3-small'
api_key = os.getenv('EMBED_API_KEY')
env_base_url = os.getenv('EMBED_BASE_URL')
user_config = {'model_type': provider,
            'model_name': model_name,
            'api_key': api_key,
            'base_url': env_base_url,
            'enable_cache': True}
collection_name="insert_test_collection"
vectorizer = MilvusManager(host=host, port=port)
embedding_client = EmbeddingClient(config=user_config)
vectorizer = VectorRetriever(vectorizer, embedding_client, collection_name=collection_name)
search_results = vectorizer.retrieve("知识点名称：方差分析教程（ANOVA 教程）", top_k=5)

print(search_results)
