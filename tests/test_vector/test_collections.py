"""
测试 Milvus Lite 数据库初始化与 Collection Schema
"""

import os
import sys
import tempfile

import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

pymilvus = pytest.importorskip("pymilvus", reason="pymilvus 未安装")
MilvusClient = pymilvus.MilvusClient
DataType = pymilvus.DataType


def test_config_schemas_valid():
    """验证所有 Collection Schema 定义合法（字段名无冲突、向量维度一致）"""
    from backend.vector.config import ALL_SCHEMAS, MilvusLiteConfig

    assert len(ALL_SCHEMAS) == 4

    collection_names = set()

    for schema in ALL_SCHEMAS:
        # 必须字段
        assert "name" in schema
        assert "scalar_fields" in schema
        assert "vector_fields" in schema
        assert "indexes" in schema

        # Collection 名称唯一
        assert schema["name"] not in collection_names, f"重复的 collection: {schema['name']}"
        collection_names.add(schema["name"])

        # 检查标量字段
        scalar_names = set()
        has_primary = False
        for f in schema["scalar_fields"]:
            assert f["name"] not in scalar_names, f"重复字段: {f['name']}"
            scalar_names.add(f["name"])
            if f.get("is_primary"):
                has_primary = True
        assert has_primary, f"{schema['name']} 缺少主键"

        # 检查向量字段
        for f in schema["vector_fields"]:
            assert f["name"] not in scalar_names, f"向量字段与标量字段重名: {f['name']}"
            if f["dtype"] == "FLOAT_VECTOR":
                assert f["dim"] == MilvusLiteConfig.EMBEDDING_DIM, (
                    f"{schema['name']}.{f['name']} 维度应为 {MilvusLiteConfig.EMBEDDING_DIM}"
                )

        # 检查索引与字段对应
        all_field_names = scalar_names | {f["name"] for f in schema["vector_fields"]}
        for idx in schema["indexes"]:
            assert idx["field_name"] in all_field_names, (
                f"{schema['name']} 索引字段 {idx['field_name']} 不在 schema 中"
            )


def test_collections_init_temp_db():
    """在临时 DB 中创建所有 Collection，验证成功"""
    from backend.vector.collections import init_all_collections
    from backend.vector.config import ALL_SCHEMAS

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test.db")
        client = MilvusClient(uri=db_file)

        results = init_all_collections(client, drop_existing=False)
        assert len(results) == 4

        for name, created in results.items():
            assert created is True
            assert client.has_collection(name), f"Collection {name} 应存在"


def test_collections_drop_and_recreate():
    """测试 drop_existing=True 重建 Collection"""
    from backend.vector.collections import init_all_collections
    from backend.vector.config import ALL_SCHEMAS

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test.db")
        client = MilvusClient(uri=db_file)

        # 第一次创建
        results1 = init_all_collections(client, drop_existing=False)
        assert all(results1.values())

        # 第二次不重建
        results2 = init_all_collections(client, drop_existing=False)
        assert not any(results2.values()), "不应重复创建"

        # 强制重建
        results3 = init_all_collections(client, drop_existing=True)
        assert all(results3.values()), "应全部重建"


def test_get_milvus_client_fresh_start():
    """测试 fresh_start 清理残留锁文件"""
    from backend.vector.collections import get_milvus_client

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test.db")

        # 模拟残留 WAL 文件
        open(db_file + "-wal", "w").close()
        open(db_file + "-shm", "w").close()
        assert os.path.isfile(db_file + "-wal")
        assert os.path.isfile(db_file + "-shm")

        # 初始化应清理残留
        client = get_milvus_client(db_file=db_file, fresh_start=True)

        assert not os.path.isfile(db_file + "-wal"), "WAL 残留应被清理"
        assert not os.path.isfile(db_file + "-shm"), "SHM 残留应被清理"

        # 验证 collection 存在
        for name in ["report_pages", "report_items", "knowledge_chunks", "medical_qa"]:
            assert client.has_collection(name), f"Collection {name} 应存在"


def test_config_values():
    """验证配置值合理性"""
    from backend.vector.config import MilvusLiteConfig

    assert MilvusLiteConfig.EMBEDDING_DIM == 1024
    assert MilvusLiteConfig.INDEX_TYPE == "FLAT"
    assert MilvusLiteConfig.METRIC_TYPE == "IP"
    assert MilvusLiteConfig.DROP_EXISTING is False
    assert len(MilvusLiteConfig.COLLECTION_REPORT_PAGES) > 0
