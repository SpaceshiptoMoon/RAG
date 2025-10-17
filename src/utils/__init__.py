"""
utils 包：常用工具集合
"""
from src.log.log_config import setup_logger
from src.utils.search import GoogleSearch

logger = setup_logger(__name__)

__all__ = ["GoogleSearch"]
