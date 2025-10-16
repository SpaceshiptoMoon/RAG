"""
utils 包：常用工具集合
"""
from src.log.log_config import setup_logger
from .search import search

logger = setup_logger(__name__)

__all__ = ["search"]
