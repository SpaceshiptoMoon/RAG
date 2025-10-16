"""
docs_read 包：文档读取器集合。
导出主要的 ReadFiles 类并提供包级 logger。
"""
from src.log.log_config import setup_logger
from .data_read import ReadFiles

logger = setup_logger(__name__)

__all__ = ["ReadFiles"]
