from src.agent.agent import DeepAgent
from src.models.llm import Ollama_Model
from dotenv import load_dotenv
load_dotenv()
# llm = Ollama_Model()
# print(llm.chat("你是谁"))
agent = DeepAgent()
result = agent.query("现在广东的台风怎么样了")
print("查询结果:", result)