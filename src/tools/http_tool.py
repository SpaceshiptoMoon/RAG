"""tools.http_tool

简单的 HTTP 请求工具实现示例。
"""
from typing import Dict, Any
from .base_tool import BaseTool

try:
    import requests
except Exception:
    requests = None


class HTTPTool(BaseTool):
    def __init__(self, name: str = "http"):
        super().__init__(name)
        # 默认不假定幂等，具体调用时根据 method 决定
        self.idempotent = False

    def call(self, url: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
        # basic validation
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a valid HTTP method string")

        # GET/HEAD are idempotent
        if method.upper() in ("GET", "HEAD"):
            self.idempotent = True
        else:
            self.idempotent = False

        if requests is None:
            return {"error": "requests not installed"}

        resp = requests.request(method, url, **kwargs)
        return {"status_code": resp.status_code, "text": resp.text}
