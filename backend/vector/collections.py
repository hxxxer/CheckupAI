"""
Milvus Lite Collection 初始化 / 删除
"""

from pymilvus import DataType, MilvusClient

from .config import ALL_SCHEMAS, MilvusLiteConfig


def _add_scalar_fields(schema, field_defs: list[dict]):
    for f in field_defs:
        dtype = getattr(DataType, f["dtype"])
        kwargs = {"field_name": f["name"], "datatype": dtype}
        if f.get("is_primary"):
            kwargs["is_primary"] = True
        elif dtype == DataType.JSON:
            pass  # JSON 类型无需额外参数
        elif "max_length" in f:
            kwargs["max_length"] = f["max_length"]
        schema.add_field(**kwargs)


def _add_vector_fields(schema, field_defs: list[dict]):
    for f in field_defs:
        if f["dtype"] == "FLOAT_VECTOR":
            schema.add_field(
                field_name=f["name"],
                datatype=DataType.FLOAT_VECTOR,
                dim=f["dim"],
            )
        elif f["dtype"] == "SPARSE_FLOAT_VECTOR":
            schema.add_field(
                field_name=f["name"],
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )


def _add_indexes(client: MilvusClient, collection_name: str, indexes: list[dict]):
    index_params = client.prepare_index_params()
    for idx in indexes:
        index_params.add_index(
            field_name=idx["field_name"],
            index_type=idx["index_type"],
            metric_type=idx["metric_type"],
        )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )


def init_collection(client: MilvusClient, schema_def: dict, drop_existing: bool = False) -> bool:
    """
    初始化单个 Collection

    Args:
        client: MilvusClient 实例
        schema_def: Schema 定义字典
        drop_existing: 是否删除已有 Collection 重建

    Returns:
        True 表示新创建，False 表示已存在
    """
    name = schema_def["name"]

    if client.has_collection(name):
        if drop_existing:
            client.drop_collection(name)
            print(f"[Milvus] 删除旧集合: {name}")
        else:
            print(f"[Milvus] 集合 {name} 已存在，跳过创建")
            return False

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
        description=schema_def.get("description", ""),
    )

    _add_scalar_fields(schema, schema_def["scalar_fields"])
    _add_vector_fields(schema, schema_def["vector_fields"])

    index_params = client.prepare_index_params()
    for idx_def in schema_def["indexes"]:
        index_params.add_index(
            field_name=idx_def["field_name"],
            index_type=idx_def["index_type"],
            metric_type=idx_def["metric_type"],
        )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )

    print(f"[Milvus] 创建集合成功: {name}")
    return True


def init_all_collections(
    client: MilvusClient,
    drop_existing: bool = False,
    schemas: list[dict] | None = None,
) -> dict[str, bool]:
    """
    初始化所有 Collection

    Args:
        client: MilvusClient 实例
        drop_existing: 是否删除已有 Collection 重建
        schemas: Schema 定义列表，默认使用 ALL_SCHEMAS

    Returns:
        {collection_name: created_flag}
    """
    if schemas is None:
        schemas = ALL_SCHEMAS

    results = {}
    for schema_def in schemas:
        created = init_collection(client, schema_def, drop_existing=drop_existing)
        results[schema_def["name"]] = created
    return results


def get_milvus_client(
    db_file: str | None = None,
    drop_existing: bool = False,
    fresh_start: bool = False,
) -> MilvusClient:
    """
    获取初始化的 MilvusClient 并确保所有 Collection 就绪

    Args:
        db_file: Milvus Lite DB 文件路径，默认从 config 读取
        drop_existing: 是否删除已有 Collection 重建
        fresh_start: 是否删除整个 DB 文件重新开始（解决残留锁问题）

    Returns:
        MilvusClient 实例
    """
    if db_file is None:
        db_file = MilvusLiteConfig.DB_FILE

    import os
    import glob as _glob

    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    # 清理 SQLite WAL/SHM 残留（上次崩溃或未正常关闭导致）
    for suffix in ("-wal", "-shm"):
        companion = db_file + suffix
        if os.path.isfile(companion):
            try:
                os.remove(companion)
                print(f"[Milvus] 清理残留文件: {companion}")
            except OSError:
                pass

    # 完全重建模式
    if fresh_start and os.path.isfile(db_file):
        try:
            os.remove(db_file)
            print(f"[Milvus] 删除旧数据库: {db_file}")
            for suffix in ("-wal", "-shm"):
                companion = db_file + suffix
                if os.path.isfile(companion):
                    os.remove(companion)
        except OSError as e:
            print(f"[Milvus] 无法删除旧数据库 ({e})，尝试继续...")

    client = MilvusClient(uri=db_file)
    print(f"[Milvus] 连接数据库: {db_file}")

    init_all_collections(client, drop_existing=drop_existing or fresh_start)

    return client
