"""
向量数据库模块 — BGE-M3 + Milvus Lite 统一入库与检索

Collection 说明：
- report_pages:   体检报告页面完整信息，summary 向量化
- report_items:   检验项目精确检索，item 名称向量化
- knowledge_chunks: 权威医学知识库分块
- medical_qa:     医疗问答资料
"""

from .collections import get_milvus_client, init_all_collections
from .config import MilvusLiteConfig, ALL_SCHEMAS
from .embeddings import BGEM3Embedder, get_bgem3_model
from .ingest_knowledge import ingest_knowledge_chunks, load_chunks_json
from .ingest_qa import ingest_qa_from_csv
from .ingest_report import ingest_checkup_report
