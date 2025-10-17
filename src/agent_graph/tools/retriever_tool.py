from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.log.log_config import setup_logger
from src.rag.retriever import VectorRetriever

logger = setup_logger(__name__)

