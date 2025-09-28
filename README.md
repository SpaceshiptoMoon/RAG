# <center>RAG知识库与智能体问答系统</center>

# 项目简介

这是一个基于LangChain和LangGraph框架构建的RAG（检索增强生成）知识库与智能体问答系统。项目集成了文档检索、语义搜索和多轮对话能力，专门用于处理企业年度报告、研究文档等专业资料的知识问答任务。

核心特性：
• 📚 多格式文档支持：支持PDF、TXT、Markdown等格式的文档处理

• 🔍 语义检索：基于向量数据库的智能相似度搜索

• 🤖 智能体对话：采用LangGraph构建的多轮对话工作流

• 🧠 上下文记忆：保持对话历史，实现连贯的多轮交互

• ⚡ 生产就绪：包含API服务、配置管理和完整部署结构

# 技术架构

核心技术栈

• 框架：LangChain + LangGraph（工作流编排）

• 语言模型：支持多种LLM接口（配置于src/models/）

• 向量数据库：ChromaDB（持久化存储）

• 文档处理：LangChain文档加载与文本分割

• Web框架：FastAPI（API服务）

# 系统架构图

```
项目结构基于有向图工作流设计：
用户输入 → 文档检索 → 智能体处理 → 响应生成
    ↑          ↓          ↓          ↓
状态管理 ← 条件分支 ← 工具调用 ← 模型推理
```

# 项目结构详解

``` tree 
RAG/
├── config/                 # 配置文件
│   ├── appsettings.json   # 应用设置
│   └── config.py          # 配置管理
├── data/                  # 文档数据源
│   ├── *.pdf             # 企业年度报告
│   └── research_report_*.md # 研究报告
├── src/                   # 核心源代码
│   ├── agent/            # 智能体实现
│   │   ├── agent.py      # 主智能体
│   │   ├── router_agent.py # 路由智能体
│   │   └── tools.py      # 工具函数
│   ├── docs_read/        # 文档读取模块
│   ├── models/           # 模型配置
│   │   ├── embedding.py  # 嵌入模型
│   │   └── llm.py        # 语言模型
│   ├── prompt/           # 提示词模板
│   ├── utils/            # 工具函数
│   ├── vector/           # 向量数据库
│   └── knowlage_agent.py # 知识代理入口
├── serve/                # API服务层
│   └── routers/          # API路由
├── vector_db/            # 向量数据库存储
│   └── chroma.sqlite3    # ChromaDB数据库文件
└── 配置文件
    ├── .env              # 环境变量
    ├── requirements.txt  # Python依赖
    └── README.md         # 项目说明
```

# 快速开始

环境要求

• Python 3.11+（LangGraph要求）

• 必要的API密钥（在.env文件中配置）

安装步骤

1. 克隆项目
git clone <项目地址>
cd RAG


2. 创建虚拟环境
conda create -n rag python=3.12
conda activate rag


3. 安装依赖
pip install -r requirements.txt


4. 配置环境变量
在.env文件中配置必要的API密钥：

OPENAI_API_KEY=your_openai_key
SILICONFLOW_API_KEY=your_siliconflow_key



#运行系统

1. 启动主程序
python main.py


2. 测试系统
python test.py


3. 启动API服务
cd serve
python -m uvicorn main:app --reload


核心功能说明

1. 文档处理流水线

• 自动文档解析：支持多种格式的文档解析和文本提取

• 智能文本分割：根据语义进行合理的文本分块

• 向量化存储：将文本转换为向量并存入ChromaDB

2. 智能体工作流

系统采用LangGraph构建的有状态工作流，包含以下核心节点：
• 意图识别：分析用户问题意图

• 文档检索：从向量数据库检索相关文档

• 响应生成：基于检索结果生成回答

• 状态管理：维护对话上下文和记忆

3. 检索增强生成（RAG）

```python
# 示例检索流程
query = "上海市场分析"
results = vector_store.similarity_search(query, k=3)
```


## 使用方法
``` python
基本问答

from src.knowlage_agent import KnowledgeAgentSystem

# 初始化系统
knowledge_system = KnowledgeAgentSystem("vector_db")

# 进行问答
response = knowledge_system.query("请分析聚灿光电2021年财务情况")
print(response)
```

## 高级功能

多轮对话支持

系统支持基于LangGraph的多轮对话，保持对话上下文：
```python
# 第一次查询
response1 = agent.query("什么是人工智能？")

# 后续查询（保持上下文）
response2 = agent.query("它在医疗领域有什么应用？")
```

工作流可视化与调试

项目集成LangSmith进行工作流追踪和调试：
```
# 启动LangSmith监控
langgraph dev --tunnel
```

## 性能优化

检索优化

• 相似度阈值：设置最小相似度分数过滤无关结果

• 分块策略优化：根据文档类型调整文本分块大小

• 索引优化：定期优化向量数据库索引

对话长度管理

使用LangChain的trim_messages工具管理对话长度，避免超出模型上下文限制。

## 故障排除

常见问题

1. 向量数据库连接失败
   • 检查vector_db目录权限

   • 确认ChromaDB服务正常运行

2. API密钥错误
   • 验证.env文件中的API密钥配置

   • 检查网络连接和API服务状态

3. 依赖冲突
   • 使用虚拟环境隔离项目依赖

   • 确保Python版本符合要求


# 许可证

本项目采用MIT许可证。详见LICENSE文件。


本项目基于LangChain和LangGraph框架构建，充分利用了现代AI技术栈的优势。