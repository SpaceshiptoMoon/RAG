from src.rag.rag_system import RAGSystem
from src.docs_read.data_read import ReadFiles
from src.vector.milvus_db import MilvusManager
from src.models.embedding import EmbeddingModelFactory
from src.vector.vectorstore import DocumentVectorizer
from src.rag.retriever import VectorRetriever
from src.rag.generator import AnswerGenerator
from src.models.llm import get_llm
from dotenv import load_dotenv
import os

load_dotenv()
host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")

file_name = r'./data/confidence_intervals.md'
collection_name = "insert_test_collection"

# reader = ReadFiles(file_name)
# milvus_client = MilvusManager(host, port)
# embedding_client = EmbeddingModelFactory.create_model({'enable_cache': True})
# llm = get_llm("qwen")
# vectorizer = DocumentVectorizer(embedding_client, milvus_client, reader, collection_name)
# retriever = VectorRetriever(milvus_client, embedding_client, collection_name)
# generator = AnswerGenerator(llm)

rag_system = RAGSystem(file_name, collection_name)
rag_system.build_index()
# query_result = rag_system.query("什么是置信区间？")
# print(query_result)
# print(rag_system.get_system_info())

