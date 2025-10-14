import os
import logging
import json
from typing import List, Dict, Any, Optional, Union

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from dotenv import load_dotenv
import os

class MilvusManager:
    """基于官方 pymilvus 的 Milvus 管理类。

    提供：连接管理、集合（collection）增删改查、向量插入/删除/更新、索引管理、向量检索等。

    设计约定：
    - 默认集合 schema 为 (id:int64 primary auto_id, embedding:float_vector, text: varchar, metadata: varchar)
    - 向量字段名默认: 'embedding'
    - 文本字段名默认: 'text'
    - metadata 存为 JSON 字符串
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 19530, alias: str = "default"):
        self.host = host if host else os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port if port else int(os.getenv("MILVUS_PORT", 19530))
        self.alias = alias
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"MilvusManager_{self.alias}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def connect(self) -> None:
        """连接到 Milvus 服务器。"""
        uri = f"{self.host}:{self.port}"
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        self._logger.info(f"connected to Milvus {uri} (alias={self.alias})")

    def close(self) -> None:
        """断开连接。"""
        try:
            connections.disconnect(self.alias)
            self._logger.info("disconnected from Milvus")
        except Exception:
            pass

    # --- Collection (表) 操作 ---
    def has_collection(self, name: str) -> bool:
        return utility.has_collection(name, using=self.alias)

    def list_collections(self) -> List[str]:
        return utility.list_collections(using=self.alias)

    def create_collection(
        self,
        name: str,
        dimension: int,
        vector_field: str = "embedding",
        text_field: str = "text",
        metadata_field: str = "metadata",
        metric_type: str = "L2",
        shards_num: int = 2,
    ) -> Collection:
        """创建集合（如果已存在则返回已存在的 Collection 对象）。"""
        if self.has_collection(name):
            self._logger.warning(f"collection {name} exists")
            return Collection(name, using=self.alias)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name=metadata_field, dtype=DataType.VARCHAR, max_length=4096),
        ]

        schema = CollectionSchema(fields=fields, description=f"collection {name} schema")

        collection = Collection(name=name, schema=schema, using=self.alias, shards_num=shards_num)
        # 设置默认索引参数为 IVF_FLAT
        collection.create_index(field_name=vector_field, index_params={"index_type": "IVF_FLAT", "metric_type": metric_type, "params": {"nlist": 1024}})
        collection.load()
        self._logger.info(f"created collection {name} dim={dimension}")
        return collection

    def drop_collection(self, name: str) -> bool:
        if not self.has_collection(name):
            self._logger.warning(f"collection {name} not exists")
            return False
        utility.drop_collection(name, using=self.alias)
        self._logger.info(f"dropped collection {name}")
        return True

    def get_collection(self, name: str) -> Collection:
        if not self.has_collection(name):
            raise ValueError(f"collection {name} does not exist")
        return Collection(name, using=self.alias)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        coll = self.get_collection(name)
        stats = utility.get_collection_stats(name, using=self.alias)
        return {
            "name": name,
            "num_entities": coll.num_entities,
            "partitions": stats.get("partitions", []),
        }

    # --- 数据操作 ---
    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """插入向量及可选文本/metadata，返回分配/插入的主键列表。"""
        coll = self.get_collection(collection_name)
        n = len(vectors)
        texts = texts or ["" for _ in range(n)]
        metadatas = metadatas or [{} for _ in range(n)]

        # metadata 序列化为 json 字符串
        metadata_strs = [json.dumps(m, ensure_ascii=False) for m in metadatas]

        # 注意插入数据顺序需与 schema 中字段顺序（除 auto_id 主键外）一致
        entities = [vectors, texts, metadata_strs]

        res = coll.insert(entities)
        # 插入后可能需要 flush
        coll.flush()

        # 尝试从返回结果中获取主键
        try:
            pks = res.primary_keys
        except Exception:
            # fallback: 使用 num_entities 计算末尾 id 不是可靠方法，但保留空列表处理
            pks = []
        self._logger.info(f"inserted {n} entities into {collection_name}")
        return pks

    def delete(self, collection_name: str, expr: str) -> Dict[str, Any]:
        """按表达式删除数据，例如 "id in [1,2,3]" 或 "text == 'hello'"。"""
        coll = self.get_collection(collection_name)
        delete_res = coll.delete(expr)
        coll.flush()
        self._logger.info(f"deleted by expr from {collection_name}: {expr}")
        return delete_res

    def update(self, collection_name: str, primary_key: int, updates: Dict[str, Any]) -> None:
        """更新实体：要求 collection 的主键为非 auto_id（即用户可指定 id）。

        实现方式：直接插入带相同主键的实体（upsert 风格）
        """
        coll = self.get_collection(collection_name)
        pk_field = [f for f in coll.schema.fields if f.is_primary][0]
        if pk_field.auto_id:
            raise RuntimeError("collection primary key is auto_id=True; update by primary key is not supported")

        # 构造要插入/更新的数据，必须包含主键
        # 按 schema 顺序组织数据
        field_names = [f.name for f in coll.schema.fields if not f.is_primary]
        row = []
        for fname in field_names:
            row.append(updates.get(fname))

        # 插入时需要以列的形式传入主键和其它字段
        entities = [[primary_key], *[ [v] for v in row ]]
        coll.insert(entities)
        coll.flush()
        self._logger.info(f"updated entity {primary_key} in {collection_name}")

    # --- 索引与加载 ---
    def create_index(self, collection_name: str, field_name: str = "embedding", index_params: Optional[Dict] = None) -> None:
        coll = self.get_collection(collection_name)
        params = index_params or {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        coll.create_index(field_name=field_name, index_params=params)
        self._logger.info(f"created index on {collection_name}.{field_name}")

    def load_collection(self, collection_name: str) -> None:
        coll = self.get_collection(collection_name)
        coll.load()

    def release_collection(self, collection_name: str) -> None:
        coll = self.get_collection(collection_name)
        coll.release()

    # --- 搜索 ---
    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        fields: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        coll = self.get_collection(collection_name)
        search_params = params or {"metric_type": "L2", "params": {"nprobe": 10}}
        # 默认返回 text 和 metadata
        output_fields = fields or [f.name for f in coll.schema.fields if not f.is_primary]

        results = coll.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )

        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "entity": hit.entity,
                })
        return formatted


if __name__ == "__main__":
    # 简单使用示例（连接前请确保 Milvus 已启动）
    mgr = MilvusManager(host="192.168.2.83", port=19530)
    mgr.connect()

    coll = mgr.create_collection("example_vectors", dimension=128)

    import random
    vectors = [[random.random() for _ in range(128)] for _ in range(10)]
    texts = [f"doc_{i}" for i in range(10)]
    metadatas = [{"idx": i} for i in range(10)]

    pks = mgr.insert("example_vectors", vectors, texts, metadatas)
    print("inserted pks:", pks)

    q = [vectors[0]]
    res = mgr.search("example_vectors", q, top_k=3)
    print("search:", res)

    mgr.close()