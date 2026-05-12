"""
Milvus Lite 配置与 Collection Schema 定义
参考 temp/qwen3-med_rag 项目模式
"""


import os
from pathlib import Path


class MilvusLiteConfig:
    """Milvus Lite 本地文件数据库配置"""

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DB_FILE = os.path.join(BASE_DIR, "data", "milvus_vectors.db")

    EMBEDDING_DIM = 1024  # BGE-M3 向量维度
    INDEX_TYPE = "FLAT"   # Milvus Lite 仅支持 FLAT
    METRIC_TYPE = "IP"    # 内积距离

    # VARCHAR 最大长度
    MAX_PK_LENGTH = 128
    MAX_SOURCE_LENGTH = 500
    MAX_DATE_LENGTH = 50
    MAX_ITEM_LENGTH = 200
    MAX_RESULT_LENGTH = 100
    MAX_UNIT_LENGTH = 50
    MAX_ABNORMAL_LENGTH = 20
    MAX_REFERENCE_LENGTH = 100
    MAX_TABLE_TITLE_LENGTH = 200
    MAX_SUMMARY_LENGTH = 2000
    MAX_CONTENT_LENGTH = 2000
    MAX_FILENAME_LENGTH = 200
    MAX_TEXT_LENGTH = 1024
    MAX_DOCUMENT_LENGTH = 1024
    MAX_DEPT_LENGTH = 64

    # Collection 名称
    COLLECTION_REPORT_PAGES = "report_pages"
    COLLECTION_REPORT_ITEMS = "report_items"
    COLLECTION_KNOWLEDGE_CHUNKS = "knowledge_chunks"
    COLLECTION_MEDICAL_QA = "medical_qa"

    DROP_EXISTING = False  # 默认不删除已有数据


# ==================== Collection Schema 定义 ====================

SCHEMA_REPORT_PAGES = {
    "name": MilvusLiteConfig.COLLECTION_REPORT_PAGES,
    "description": "体检报告页面完整信息，summary 向量化",
    "scalar_fields": [
        {"name": "pk", "dtype": "INT64", "is_primary": True},
        {"name": "report_source", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_SOURCE_LENGTH},
        {"name": "exam_date", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_DATE_LENGTH},
        {"name": "page_index", "dtype": "INT64"},
        {"name": "summary_text", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_SUMMARY_LENGTH},
        {"name": "page_data_json", "dtype": "JSON"},
    ],
    "vector_fields": [
        {"name": "summary_dense", "dim": MilvusLiteConfig.EMBEDDING_DIM, "dtype": "FLOAT_VECTOR"},
        {"name": "summary_sparse", "dtype": "SPARSE_FLOAT_VECTOR"},
    ],
    "indexes": [
        {"field_name": "summary_dense", "index_type": MilvusLiteConfig.INDEX_TYPE, "metric_type": MilvusLiteConfig.METRIC_TYPE},
        {"field_name": "summary_sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": MilvusLiteConfig.METRIC_TYPE},
    ],
}


SCHEMA_REPORT_ITEMS = {
    "name": MilvusLiteConfig.COLLECTION_REPORT_ITEMS,
    "description": "体检报告检验项目精确检索，item 名称向量化",
    "scalar_fields": [
        {"name": "pk", "dtype": "INT64", "is_primary": True},
        {"name": "report_source", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_SOURCE_LENGTH},
        {"name": "exam_date", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_DATE_LENGTH},
        {"name": "page_index", "dtype": "INT64"},
        {"name": "table_title", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_TABLE_TITLE_LENGTH},
        {"name": "item", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_ITEM_LENGTH},
        {"name": "result", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_RESULT_LENGTH},
        {"name": "unit", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_UNIT_LENGTH},
        {"name": "abnormal", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_ABNORMAL_LENGTH},
        {"name": "reference_range", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_REFERENCE_LENGTH},
    ],
    "vector_fields": [
        {"name": "item_dense", "dim": MilvusLiteConfig.EMBEDDING_DIM, "dtype": "FLOAT_VECTOR"},
        {"name": "item_sparse", "dtype": "SPARSE_FLOAT_VECTOR"},
    ],
    "indexes": [
        {"field_name": "item_dense", "index_type": MilvusLiteConfig.INDEX_TYPE, "metric_type": MilvusLiteConfig.METRIC_TYPE},
        {"field_name": "item_sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": MilvusLiteConfig.METRIC_TYPE},
    ],
}


SCHEMA_KNOWLEDGE_CHUNKS = {
    "name": MilvusLiteConfig.COLLECTION_KNOWLEDGE_CHUNKS,
    "description": "权威医学知识库分块，content 向量化",
    "scalar_fields": [
        {"name": "pk", "dtype": "INT64", "is_primary": True},
        {"name": "content", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_CONTENT_LENGTH},
        {"name": "source", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_SOURCE_LENGTH},
        {"name": "filename", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_FILENAME_LENGTH},
    ],
    "vector_fields": [
        {"name": "content_dense", "dim": MilvusLiteConfig.EMBEDDING_DIM, "dtype": "FLOAT_VECTOR"},
        {"name": "content_sparse", "dtype": "SPARSE_FLOAT_VECTOR"},
    ],
    "indexes": [
        {"field_name": "content_dense", "index_type": MilvusLiteConfig.INDEX_TYPE, "metric_type": MilvusLiteConfig.METRIC_TYPE},
        {"field_name": "content_sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": MilvusLiteConfig.METRIC_TYPE},
    ],
}


SCHEMA_MEDICAL_QA = {
    "name": MilvusLiteConfig.COLLECTION_MEDICAL_QA,
    "description": "医疗问答资料，参考 temp/qwen3-med_rag 项目",
    "scalar_fields": [
        {"name": "pk", "dtype": "INT64", "is_primary": True},
        {"name": "text", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_TEXT_LENGTH},
        {"name": "summary", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_SUMMARY_LENGTH},
        {"name": "document", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_DOCUMENT_LENGTH},
        {"name": "department", "dtype": "VARCHAR", "max_length": MilvusLiteConfig.MAX_DEPT_LENGTH},
    ],
    "vector_fields": [
        {"name": "summary_dense", "dim": MilvusLiteConfig.EMBEDDING_DIM, "dtype": "FLOAT_VECTOR"},
        {"name": "text_dense", "dim": MilvusLiteConfig.EMBEDDING_DIM, "dtype": "FLOAT_VECTOR"},
        {"name": "text_sparse", "dtype": "SPARSE_FLOAT_VECTOR"},
    ],
    "indexes": [
        {"field_name": "summary_dense", "index_type": MilvusLiteConfig.INDEX_TYPE, "metric_type": MilvusLiteConfig.METRIC_TYPE},
        {"field_name": "text_dense", "index_type": MilvusLiteConfig.INDEX_TYPE, "metric_type": MilvusLiteConfig.METRIC_TYPE},
        {"field_name": "text_sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": MilvusLiteConfig.METRIC_TYPE},
    ],
}


ALL_SCHEMAS = [
    SCHEMA_REPORT_PAGES,
    SCHEMA_REPORT_ITEMS,
    SCHEMA_KNOWLEDGE_CHUNKS,
    SCHEMA_MEDICAL_QA,
]
