from pydantic import BaseModel, Field
from typing import Union, Dict, Any
from decimal import Decimal
from langchain_core.tools import tool
from src.log.log_config import setup_logger
import httpx

logger = setup_logger(__name__)



# 通用数值类型（支持原生 float、int、Decimal 等）
Number = Union[float, int, Decimal]

class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名称，非空字符串", examples=["北京", "上海"])


@tool(
    name_or_callable="weather",
    description="天气查询工具，参数: city(str)",
    args_schema=WeatherArgs,
    # return_direct=True  # 先注释，便于排查
)
def weather(city: str) -> Dict[str, Any]:
    """
    查询指定城市天气，结构化输出
    """
    if not isinstance(city, str) or not city.strip():
        return {"error": "参数 city 必须为非空字符串"}
    try:
        resp = httpx.get(f"https://wttr.in/{city.strip()}?format=3", timeout=5.0)
        resp.raise_for_status()
        return {"result": resp.text.strip()}
    except Exception as e:
        return {"error": f"天气查询失败: {e}"}