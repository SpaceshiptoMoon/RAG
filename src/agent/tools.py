from langchain.agents import tool
from src.utils.search import GoogleSearch

@tool
def google_search(query: str) -> str:
    """
    使用谷歌搜索引擎获取搜索结果
    Args:
    query (str): 搜索查询字符串
    """
    search = GoogleSearch()
    return search.search(query=query)