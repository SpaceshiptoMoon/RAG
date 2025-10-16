import importlib
import sys
import os

# 将项目根目录加入 sys.path，确保可以按包路径导入 src 下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

modules = [
    'src.agent_graph.nodes',
    'src.agent_graph.tools',
    'src.agent_graph.utils',
    'src.docs_read.data_read',
    'src.rag.rag_system',
    'src.rag.retriever',
    'src.rag.generator',
    'src.vector.milvus_db',
    'src.vector.vectorstore',
    'src.models.embedding',
]
for m in modules:
    try:
        importlib.import_module(m)
        print(f'OK: {m}')
    except Exception as e:
        print(f'ERROR importing {m}: {e}')
