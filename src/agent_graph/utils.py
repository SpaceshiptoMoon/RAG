
from src.log.log_config import setup_logger
import functools
import time

logger = setup_logger("agent_graph")

def retry_on_exception(max_retries=2, delay=1):
	"""
	工具调用重试装饰器
	"""
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			for attempt in range(max_retries):
				try:
					return func(*args, **kwargs)
				except Exception as e:
					logger.warning(f"重试 {func.__name__} 第{attempt+1}次: {e}")
					time.sleep(delay)
			return {"error": f"{func.__name__} 多次重试失败"}
		return wrapper
	return decorator

def with_timeout(timeout_sec):
	"""
	超时装饰器（伪实现，建议生产用多线程/asyncio）
	"""
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			start = time.time()
			result = func(*args, **kwargs)
			if time.time() - start > timeout_sec:
				logger.error(f"{func.__name__} 超时")
				return {"error": "超时"}
			return result
		return wrapper
	return decorator
