"""
端到端测试：OCR 解析 → 向量入库 → 精确检索 + 语义检索

需要 PaddleOCR 环境 + vLLM 服务，不可用时自动跳过
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from backend.config import settings

# ---- 辅助 ----

def _vllm_available():
    try:
        import requests
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        resp = requests.get(f"{base_url}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _ocr_env_available():
    """检查 settings.toml 中 OCR 环境是否就绪"""
    return (
        settings.ocr_python
        and os.path.isfile(settings.ocr_python)
    )


def _build_pk(raw: str) -> int:
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


def _save_ocr_debug(results):
    """OCR 解析结果为空时保存到 temp.json 供排查"""
    from dataclasses import asdict
    from datetime import datetime

    # 自定义 encoder 处理 dataclass / datetime / tuple / Enum
    class _DebugEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return super().default(obj)

    dump_path = os.path.join(settings.project_root, "temp.json")
    data = json.loads(json.dumps([asdict(r) for r in results], cls=_DebugEncoder))
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[E2E] OCR 解析结果已保存至 {dump_path}")


# ---- 端到端测试 ----

@pytest.mark.skipif(not _ocr_env_available(), reason="OCR 环境不可用（检查 settings.toml）")
@pytest.mark.skipif(not _vllm_available(), reason="vLLM 服务不可用")
def test_ocr_parse_and_ingest():
    """完整链路：OCR 解析 → 入库 → 精确检索 → 语义检索"""
    import tempfile

    from pymilvus import MilvusClient

    from backend.ocr import parse_checkup
    from backend.vector.collections import init_all_collections
    from backend.vector.config import ALL_SCHEMAS
    from backend.vector.embeddings import BGEM3Embedder
    from backend.vector.ingest_report import ingest_checkup_report

    test_files_path = os.path.join(
        settings.project_root, "tests", "test_ocr", "test_files"
    )

    if not os.path.isdir(test_files_path):
        pytest.skip(f"测试图片目录不存在: {test_files_path}")

    # ---- Step 1: OCR 解析 ----
    print("\n[E2E] Step 1: OCR 解析...")
    results = parse_checkup(test_files_path)
    assert len(results) > 0, "OCR 解析结果不应为空"
    print(f"[E2E] 解析完成: {len(results)} 份文件, {sum(len(r.pages) for r in results)} 页")

    # 检查关键字段非空
    total_items = 0
    total_summaries = 0
    for result in results:
        for page in result.pages:
            # table items
            for table in page.tables:
                total_items += len(table.items)
            # summary
            if page.text_analyses and page.text_analyses.summary:
                total_summaries += 1

    if total_items == 0:
        _save_ocr_debug(results)
    assert total_items > 0, "应至少解析出一些检验项目"
    print(f"[E2E] 检验项目: {total_items}, 页面摘要: {total_summaries}")

    # ---- Step 2: 入库 ----
    print("[E2E] Step 2: 向量化并入库...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_e2e.db")
        client = MilvusClient(uri=db_file)
        init_all_collections(client, drop_existing=False)

        ingest_result = ingest_checkup_report(client, results)
        assert ingest_result["pages"] > 0, "应至少入库一个页面"

        if ingest_result["items"] == 0:
            _save_ocr_debug(results)
        assert ingest_result["items"] > 0, "应至少入库一个检验项目"
        print(f"[E2E] 入库: {ingest_result['pages']} 页, {ingest_result['items']} 项")

        # ---- Step 3: 精确检索 ----
        print("[E2E] Step 3: 精确检索 report_items...")

        # 从解析结果中取出一个实际存在的 item 名
        sample_items = []
        for result in results:
            for page in result.pages:
                for table in page.tables:
                    for titem in table.items:
                        if titem.item and titem.item.strip():
                            sample_items.append(titem.item.strip())

        # 取第一个 item 做精确查询
        target_item = sample_items[0]
        query_result = client.query(
            collection_name="report_items",
            filter=f'item == "{target_item}"',
            output_fields=["*"],
        )
        assert len(query_result) > 0, f"精确查询 '{target_item}' 应有结果"
        for row in query_result:
            assert row["item"] == target_item, f"返回的 item 应为 '{target_item}'"
            assert "result" in row
            assert "reference_range" in row
            assert "table_title" in row
        print(f"[E2E] 精确查询 '{target_item}': {len(query_result)} 条")

        # 对每个 item 做精确查询，验证都能命中
        all_queried = 0
        missed = 0
        for item_name in set(sample_items):
            results_q = client.query(
                collection_name="report_items",
                filter=f'item == "{item_name}"',
                output_fields=["pk"],
            )
            if results_q:
                all_queried += 1
            else:
                missed += 1
        assert missed == 0, f"{missed} 个 item 精确查询未命中"
        print(f"[E2E] 全部 {all_queried} 个唯一 item 精确查询验证通过")

        # ---- Step 4: 语义检索 ----
        print("[E2E] Step 4: 语义检索 report_pages...")

        embedder = BGEM3Embedder()
        query_vec = embedder.encode_dense(["血常规肝功能"])[0]

        search_result = client.search(
            collection_name="report_pages",
            data=[query_vec],
            anns_field="summary_dense",
            limit=3,
            output_fields=["summary_text", "exam_date", "page_index"],
        )[0]

        assert len(search_result) > 0, "语义检索应有结果"
        for hit in search_result:
            entity = hit.get("entity", {})
            assert "summary_text" in entity or "page_data_json" in entity
        print(f"[E2E] 语义检索: {len(search_result)} 条")

        # ---- Step 5: page_data_json 完整性 ----
        print("[E2E] Step 5: 验证 page_data_json 完整性...")

        full_result = client.query(
            collection_name="report_pages",
            filter="pk > 0",
            output_fields=["page_data_json", "summary_text"],
            limit=1,
        )
        assert len(full_result) > 0

        page_data = full_result[0].get("page_data_json")
        assert isinstance(page_data, dict), "page_data_json 应为 dict"

        # 检查关键嵌套字段存在
        if "text_analyses" in page_data:
            ta = page_data["text_analyses"]
            assert "personal_info" in ta
            assert "positive_findings" in ta

        if "tables" in page_data:
            for table in page_data["tables"]:
                assert "title" in table or "raw_md" in table or "items" in table

        if "regions" in page_data:
            assert isinstance(page_data["regions"], list)

        print("[E2E] page_data_json 结构完整")
        print("[E2E] ✅ 端到端测试全部通过")
