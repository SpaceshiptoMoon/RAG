"""
Top-level src package for the project.
Provides package metadata and root logger.
"""
from src.log.log_config import setup_logger

__version__ = "0.1.0"
logger = setup_logger(__name__)

__all__ = [
    "agent_graph",
    "docs_read",
    "log",
    "models",
    "prompt",
    "rag",
    "utils",
    "vector",
]
