from src.vector.vectorstore import DocumentVectorizer
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
doc = DocumentVectorizer("insert_test_collection", milvus_host=host, milvus_port=port, embedding_config=user_config)
print(doc.milvus.host, doc.milvus.port)
print(type(doc.milvus.port),type(doc.milvus.host))
# doc.delete_db()
# doc.process_document("./data")