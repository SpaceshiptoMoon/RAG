"""agents.tool_agent

ToolAgent 使用工具注册表来执行外部工具调用，返回工具结果。
"""
from typing import Any, Dict
from ..tools.tool_registry import ToolRegistry
import logging
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

logger = logging.getLogger(__name__)


class ToolAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        # 简单的幂等性缓存：{key: result}
        self._idempotent_cache: Dict[str, Any] = {}
        # executor 用于支持超时
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _make_idempotency_key(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        key_obj = {"tool": tool_name, "args": args, "kwargs": kwargs}
        # ensure JSON serializable; fallback to repr
        try:
            key_str = json.dumps(key_obj, sort_keys=True, default=repr)
        except Exception:
            key_str = repr(key_obj)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def call_tool(self, tool_name: str, *args, timeout: float = 10.0, retries: int = 2, backoff: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Call a registered tool with validation, timeout, retries and optional idempotency caching.

        Args:
            tool_name: 注册的工具名
            *args, **kwargs: 传给 tool.call 的参数
            timeout: 单次调用的超时时间（秒）
            retries: 出错时的重试次数（不包含首次调用）
            backoff: 指数退避基数（秒）
        """
        tool = self.registry.get(tool_name)
        if not tool:
            logger.error(f"Tool {tool_name} not found")
            return {"error": "tool not found"}

        # 参数校验
        try:
            tool.validate_args(*args, **kwargs)
        except ValueError as ve:
            logger.error(f"Tool {tool_name} validation failed: {ve}")
            return {"error": f"validation error: {ve}"}

        # idempotency cache check
        idemp_key = None
        if getattr(tool, "idempotent", False):
            try:
                idemp_key = self._make_idempotency_key(tool_name, args, kwargs)
                if idemp_key in self._idempotent_cache:
                    logger.debug(f"Returning cached result for tool {tool_name}")
                    return {"result": self._idempotent_cache[idemp_key], "cached": True}
            except Exception:
                # ignore cache errors
                idemp_key = None

        attempt = 0
        last_err = None
        while attempt <= retries:
            attempt += 1
            try:
                future = self._executor.submit(tool.call, *args, **kwargs)
                res = future.result(timeout=timeout)
                # save to cache if idempotent
                if idemp_key is not None:
                    try:
                        self._idempotent_cache[idemp_key] = res
                    except Exception:
                        logger.debug("Failed to write idempotent cache, continuing")
                return {"result": res}
            except FutureTimeout:
                last_err = f"timeout after {timeout}s"
                logger.warning(f"Tool {tool_name} attempt {attempt} timeout")
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Tool {tool_name} attempt {attempt} failed: {e}")

            # backoff before next attempt
            if attempt <= retries:
                sleep_time = backoff * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        logger.error(f"Tool {tool_name} failed after {attempt} attempts: {last_err}")
        return {"error": last_err}
