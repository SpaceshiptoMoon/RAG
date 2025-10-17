
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.log.log_config import setup_logger
from src.utils.search import GoogleSearch

logger = setup_logger(__name__)


def get_search() -> GoogleSearch:
    from dotenv import load_dotenv
    import os 
    load_dotenv()
    api_key = os.getenv("GOOGLE_SEARCH_API")
    base_url = "https://google-search72.p.rapidapi.com/search"
    return GoogleSearch(api_key, base_url)




class Searcher(BaseModel):
    query: str = Field(..., description="你要查询的问题或者信息", examples=["今天的天气如何？", "今天股票行情如何？"])
    num: int = Field(default=3, description="需要返回几条搜索结果,例如:3,则返回前3条搜索结果", examples=[3, 1])


@tool(
    name_or_callable="searcher",
    description="使用谷歌搜索引擎获取搜索结果，当你需要获得外部信息或者实时信息时，你可以使用它。参数: query(str), num(int)",
    # return_direct=True  # 先注释，便于排查
)
def searcher(query:str, num:int):
    search = get_search()
    result = search.search(query, num)
    return result
