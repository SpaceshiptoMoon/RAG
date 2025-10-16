"""tools.search_tool

示例搜索工具（简单封装 HTTP 搜索或内部索引）。
"""
from typing import List, Dict, Optional, Any
from .base_tool import BaseTool
from src.utils.search import GoogleSearch


class SearchTool(BaseTool):
    """搜索工具：可选地封装 GoogleSearch，也支持无外部依赖时的模拟返回。

    构造器可接受一个已经初始化的 GoogleSearch 实例，或传入 api_key 和 base_url
    来创建内部的 GoogleSearch。若都未提供，工具会退回到模拟结果（便于测试）。
    """
    def __init__(self, name: str = "search", google_search: Optional[GoogleSearch] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(name)
        # search is read-only -> idempotent
        self.idempotent = True

        if google_search is not None:
            self._searcher = google_search
        elif api_key and base_url:
            self._searcher = GoogleSearch(api_key, base_url)
        else:
            # 未配置外部搜索，保持 None 并返回模拟数据
            self._searcher = None

    def validate_args(self, query: str, limit: int = 10) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

    def call(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """执行搜索并返回标准化的结果列表。

        当内部有已配置的 `GoogleSearch` 时，调用其 `search` 方法；
        由于 `GoogleSearch.search` 可能返回字符串或基于 JSON 的 items 列表，
        这里做容错与标准化，最终返回 list[dict]，每项包含 title/url/snippet。
        """
        # 参数校验
        self.validate_args(query, limit)

        if self._searcher is None:
            # 回退：返回模拟结果，保持兼容性（便于本地开发/测试）
            return [{"title": f"Result for {query}", "url": "http://example.com", "snippet": ""}]

        raw = self._searcher.search(query)

        # 如果 GoogleSearch 返回字符串（拼接的结果），则作为单条 snippet 返回
        if isinstance(raw, str):
            return [{"title": query, "url": "", "snippet": raw}]

        # 否则尝试把 raw 解析为可迭代的条目并标准化
        results: List[Dict[str, Any]] = []
        try:
            for item in raw:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("name") or ""
                    url = item.get("link") or item.get("url") or ""
                    snippet = item.get("snippet") or item.get("description") or ""
                else:
                    # 非 dict 项，直接转为字符串放入 snippet
                    title = str(item)
                    url = ""
                    snippet = str(item)
                results.append({"title": title, "url": url, "snippet": snippet})
        except Exception:
            # 如果解析失败，作为单条文本返回原始内容的字符串表示
            return [{"title": query, "url": "", "snippet": str(raw)}]

        return results
