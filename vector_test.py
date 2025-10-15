from src.vector.vectorstore import DocumentVectorizer
from src.vector.milvus_db import MilvusManager
from dotenv import load_dotenv
import os
load_dotenv()

host = os.getenv("MILVUS_HOST", "127.0.0.1")
port  =  os.getenv("MILVUS_PORT", "19530") 
# client = MilvusManager(host, port)
# client.create_collection("test_collection", 1024)
provider = (os.getenv('EMBED_PROVIDER') or 'openai').lower()
model_name = os.getenv('EMBED_MODEL') or 'text-embedding-3-small'
api_key = os.getenv('EMBED_API_KEY')
env_base_url = os.getenv('EMBED_BASE_URL')
user_config = {'model_type': provider,
            'model_name': model_name,
            'api_key': api_key,
            'base_url': env_base_url,
            'enable_cache': True}

vectorizer = DocumentVectorizer(collection_name="test_collection", milvus_host=host, milvus_port=port, embedding_config=user_config)
# vectorizer.process_document("./data")
results =  vectorizer.query_similarity("什么是人工智能？", top_k=3)
print(results)