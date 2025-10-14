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

load_dotenv()

class MilvusManager:
    """基于 pymilvus 的 Milvus 管理类。

    提供：连接管理、集合（collection）增删改查、向量插入/删除/更新、索引管理、向量检索等。

    设计约定：
    - 默认集合 schema 为 (id:int64 primary auto_id, embedding:float_vector, text: varchar, metadata: varchar)
    - 向量字段名默认: 'embedding'
    - 文本字段名默认: 'text'
    - metadata 存为 JSON 字符串
    """

    def __init__(self, host: str , port: int , alias: str = "default"):
        """
        初始化 MilvusManager 实例。

        Args:
            host: Milvus 服务主机地址，若为空则从环境变量 MILVUS_HOST 读取，默认 "127.0.0.1"。
            port: Milvus 服务端口，若为 0 或 None 则从环境变量 MILVUS_PORT 读取，默认 19530。
            alias: 连接别名，默认为 "default"，用于区分多个连接。

        Returns:
            None: 不返回值，完成对象初始化。
        """
        self.host = host if host else os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port if port else int(os.getenv("MILVUS_PORT", 19530))
        self.alias = alias
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        配置并返回用于 MilvusManager 的 logger。

        Args:
            None

        Returns:
            logging.Logger: 配置好的 logger 实例，名称格式为 `MilvusManager_{alias}`。
        """
        logger = logging.getLogger(f"MilvusManager_{self.alias}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def connect(self) -> None:
        """
        连接到 Milvus 服务器。

        Args:
            None

        Returns:
            None: 成功连接后无返回值；连接失败会抛出异常。
        """
        uri = f"{self.host}:{self.port}"
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        self._logger.info(f"connected to Milvus {uri} (alias={self.alias})")

    def close(self) -> None:
        """
        断开与 Milvus 的连接并清理资源。

        Args:
            None

        Returns:
            None: 无返回值；若断开过程中发生异常则被捕获并忽略。
        """
        try:
            connections.disconnect(self.alias)
            self._logger.info("disconnected from Milvus")
        except Exception:
            pass

    # --- Collection (表) 操作 ---
    def has_collection(self, name: str) -> bool:
        """
        判断指定名称的集合（collection）是否存在。

        Args:
            name: 要检查的集合名称。

        Returns:
            bool: 如果集合存在返回 True，否则返回 False。
        """
        return utility.has_collection(name, using=self.alias)

    def list_collections(self) -> List[str]:
        """
        列出当前连接下的所有集合名称。

        Args:
            None

        Returns:
            List[str]: 集合名称列表。
        """
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
        """
        创建一个新的集合（collection）。如果集合已存在则返回已存在的 Collection 对象。

        Args:
            name: 集合名称。
            dimension: 向量维度（embedding 向量维度）。
            vector_field: 向量字段名称，默认 'embedding'。
            text_field: 文本字段名称，默认 'text'。
            metadata_field: 用于存放 metadata 的字段名称，默认 'metadata'。
            metric_type: 度量类型，例如 'L2' 或 'IP'，默认 'L2'。
            shards_num: 集合分片数量，默认 2。

        Returns:
            Collection: 创建或已存在的 Milvus Collection 对象。
        """
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
        """
        删除指定名称的集合。

        Args:
            name: 要删除的集合名称。

        Returns:
            bool: 删除成功返回 True；若集合不存在返回 False。
        """
        if not self.has_collection(name):
            self._logger.warning(f"collection {name} not exists")
            return False
        utility.drop_collection(name, using=self.alias)
        self._logger.info(f"dropped collection {name}")
        return True

    def get_collection(self, name: str) -> Collection:
        """
        获取已存在的 Collection 对象。

        Args:
            name: 集合名称。

        Returns:
            Collection: 指定名称的 Collection 对象；若集合不存在则抛出 ValueError。
        """
        if not self.has_collection(name):
            raise ValueError(f"collection {name} does not exist")
        return Collection(name, using=self.alias)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """
        获取集合的统计信息。

        Args:
            name: 集合名称。

        Returns:
            Dict[str, Any]: 包含集合名称、实体数量（num_entities）和分区信息（partitions）的字典。
        """
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
        """
        向集合中插入向量及可选的文本和 metadata，并返回插入记录的主键列表。

        Args:
            collection_name: 目标集合名称。
            vectors: 向量列表，形状为 List[List[float]]。
            texts: 可选的文本列表，与 vectors 对应，默认空字符串列表。
            metadatas: 可选的 metadata 列表（字典），会被序列化为 JSON 字符串。

        Returns:
            List[int]: 插入后返回的主键列表；如果无法从返回结果获取主键则返回空列表。
        """
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
        """
        按照表达式从集合中删除实体。

        Args:
            collection_name: 目标集合名称。
            expr: 删除表达式，例如 "id in [1,2,3]" 或 "text == 'hello'"。

        Returns:
            Dict[str, Any]: Milvus 返回的删除结果对象（通常包含受影响实体的信息）。
        """
        coll = self.get_collection(collection_name)
        delete_res = coll.delete(expr)
        coll.flush()
        self._logger.info(f"deleted by expr from {collection_name}: {expr}")
        return delete_res

    def update(self, collection_name: str, primary_key: int, updates: Dict[str, Any]) -> None:
        """
        更新指定主键的实体（仅支持主键非 auto_id 的集合）。

        实现方式为 upsert：通过插入包含相同主键的新实体来覆盖原有数据。

        Args:
            collection_name: 目标集合名称。
            primary_key: 要更新的实体主键（必须是用户自定义主键，auto_id=False）。
            updates: 包含要更新字段及其新值的字典，字段名应与集合 schema 中非主键字段一致。

        Returns:
            None: 操作完成后无返回值；如果集合主键为 auto_id 会抛出 RuntimeError。
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
        """
        为指定集合的向量字段创建索引。

        Args:
            collection_name: 目标集合名称。
            field_name: 要创建索引的字段名，默认 'embedding'。
            index_params: 可选索引参数字典（index_type、metric_type、params 等），若无则使用默认 IVF_FLAT 参数。

        Returns:
            None: 创建完成后无返回值。
        """
        coll = self.get_collection(collection_name)
        params = index_params or {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        coll.create_index(field_name=field_name, index_params=params)
        self._logger.info(f"created index on {collection_name}.{field_name}")

    def load_collection(self, collection_name: str) -> None:
        """
        将集合加载到内存以便进行检索操作。

        Args:
            collection_name: 要加载的集合名称。

        Returns:
            None: 加载完成后无返回值。
        """
        coll = self.get_collection(collection_name)
        coll.load()

    def release_collection(self, collection_name: str) -> None:
        """
        释放集合占用的内存资源（卸载集合）。

        Args:
            collection_name: 要释放的集合名称。

        Returns:
            None: 释放完成后无返回值。
        """
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
        """
        在集合中执行向量检索并返回格式化结果。

        Args:
            collection_name: 目标集合名称。
            query_vectors: 待检索的向量列表（List[List[float]]）。
            top_k: 每个查询向量返回的最近邻数量，默认 10。
            filter_expr: 可选过滤表达式，用于在检索时过滤实体。
            fields: 可选的返回字段列表，默认返回 schema 中除主键外的所有字段（如 text、metadata）。
            params: 检索参数字典（例如 metric_type、params.nprobe 等），默认使用 L2 与 nprobe=10。

        Returns:
            List[Dict[str, Any]]: 检索到的结果列表，每个元素为字典，包含 'id'（实体主键）、'distance'（距离）和 'entity'（实体字段数据）。
        """
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