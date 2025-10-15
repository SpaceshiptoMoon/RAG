"""
日志配置模块
"""
import os
import logging
import time
from logging.handlers import RotatingFileHandler

def setup_logger(module_name: str) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        module_name: 模块名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(module_name)
    
    if not logger.handlers:  # 避免重复添加处理器
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        log_file = os.path.join(log_dir, f'{module_name}_{time.strftime("%Y%m%d")}.log')
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        
        # 日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger