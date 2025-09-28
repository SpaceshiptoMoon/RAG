"""
main.py - FastAPI主应用
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

from src.knowlage_agent import KnowledgeAgentSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="智能知识问答系统API",
    description="基于Agent和RAG的智能问答系统",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求响应模型
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    strategy: str
    confidence: float
    sources: List[str]
    metadata: dict

class DocumentAddRequest(BaseModel):
    file_path: str

class DocumentAddResponse(BaseModel):
    success: bool
    message: str
    file_path: str

# 全局系统实例
knowledge_system = None

@app.on_event("startup")
async def startup_event():
    """应用启动初始化"""
    global knowledge_system
    try:
        vector_store_path = "./vector_db"  # 可配置化
        knowledge_system = KnowledgeAgentSystem(vector_store_path)
        logger.info("知识问答系统启动成功")
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise

@app.get("/")
async def root():
    """根端点健康检查"""
    return {"message": "智能知识问答系统API服务运行中", "status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """
    查询知识库端点
    """
    try:
        if not knowledge_system:
            raise HTTPException(status_code=503, detail="系统未就绪")
        
        result = knowledge_system.process_query(
            question=request.question,
            top_k=request.top_k
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"查询处理错误: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@app.post("/documents/add", response_model=DocumentAddResponse)
async def add_document(request: DocumentAddRequest):
    """
    添加文档到知识库
    """
    try:
        if not knowledge_system:
            raise HTTPException(status_code=503, detail="系统未就绪")
        
        result = knowledge_system.add_document(request.file_path)
        return DocumentAddResponse(**result)
        
    except Exception as e:
        logger.error(f"文档添加错误: {e}")
        raise HTTPException(status_code=500, detail=f"文档添加失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy" if knowledge_system else "unhealthy",
        "components": {
            "llm": "available",
            "agent": "available",
            "vector_db": "available" if knowledge_system and hasattr(knowledge_system.vectorizer, 'vector_store') else "unavailable"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )