"""
prompt 包：包含提示模板和常用 prompt 定义。
"""
from src.log.log_config import setup_logger
from .prompt_template import QA_PROMPT, PROMPT_TEMPLATE

logger = setup_logger(__name__)

__all__ = ["QA_PROMPT", "PROMPT_TEMPLATE"]
