"""
从已有 OCR 原始结果加载 → 解析 → 入库 → 精确检索 + 语义检索

跳过 OCR subprocess，直接从 runner.load_result() 开始。
需要 vLLM 服务（表格路由+解析+文本分析），不可用时自动跳过。

使用前将 OCR_OUTPUT_DIR 改为实际 OCR 输出目录。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from backend.config import settings

# ============================================================
# 硬编码：上一次 parse_checkup() 产生的 OCR 输出目录
# 格式: data/sensitive/ocr_output/{timestamp}/
# 部署机跑之前手动改这里
# ============================================================
OCR_OUTPUT_DIR = os.path.join(
    settings.project_root,
    "data/sensitive/ocr_output/20250513_000000",  # TODO: 替换为实际时间戳
)


# ---- 辅助 ----

def _vllm_available():
    try:
        import requests
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        resp = requests.get(f"{base_url}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _save_ocr_debug(results, label: str = ""):
    """OCR 解析结果为空时保存到 temp.json 供排查"""
    from dataclasses import asdict
    from datetime import datetime

    class _DebugEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return super().default(obj)

    filename = f"temp{'_' + label if label else ''}.json"
    dump_path = os.path.join(settings.project_root, filename)
    data = json.loads(json.dumps([asdict(r) for r in results], cls=_DebugEncoder))
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[LoadOCR] OCR 解析结果已保存至 {dump_path}")


# ---- 测试 ----

@pytest.mark.skipif(not _vllm_available(), reason="vLLM 服务不可用")
def test_load_ocr_and_ingest():
    """从已有 OCR 原始结果加载 → 解析 → 入库 → 检索验证"""
    import tempfile

    from pymilvus import MilvusClient

    from backend.llm import text_analyzer
    from backend.ocr import PaddleOCRRunner, UniversalParser
    from backend.vector.collections import init_all_collections
    from backend.vector.embeddings import BGEM3Embedder
    from backend.vector.ingest_report import ingest_checkup_report

    if not os.path.isdir(OCR_OUTPUT_DIR):
        pytest.skip(f"OCR 输出目录不存在: {OCR_OUTPUT_DIR}")

    # ---- Step 1: 加载 OCR 原始结果 ----
    print(f"\n[LoadOCR] Step 1: 加载 OCR 原始结果...")
    runner = PaddleOCRRunner()
    raw_outputs = runner.load_result(OCR_OUTPUT_DIR)
    assert len(raw_outputs) > 0, "OCR 原始结果不应为空"
    print(f"[LoadOCR] 加载完成: {len(raw_outputs)} 个文件, "
          f"{sum(len(r.pages) for r in raw_outputs)} 页")

    # ---- Step 2: UniversalParser 解析 ----
    print("[LoadOCR] Step 2: UniversalParser 解析...")
    parser = UniversalParser()
    structured_data = parser.parse(raw_outputs)
    assert len(structured_data) > 0, "解析结果不应为空"

    # ---- Step 3: text_analyzer 文本分析 ----
    print("[LoadOCR] Step 3: text_analyzer 文本分析...")
    results = text_analyzer.analyze(structured_data)
    assert len(results) > 0, "分析结果不应为空"

    # 检查关键字段非空
    total_items = 0
    total_summaries = 0
    for result in results:
        for page in result.pages:
            for table in page.tables:
                total_items += len(table.items)
            if page.text_analyses and page.text_analyses.summary:
                total_summaries += 1

    if total_items == 0:
        _save_ocr_debug(results, label="no_items")
    assert total_items > 0, "应至少解析出一些检验项目"
    print(f"[LoadOCR] 检验项目: {total_items}, 页面摘要: {total_summaries}")

    # ---- Step 4: 入库 ----
    print("[LoadOCR] Step 4: 向量化并入库...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_load_ocr.db")
        client = MilvusClient(uri=db_file)
        init_all_collections(client, drop_existing=False)

        ingest_result = ingest_checkup_report(client, results)
        assert ingest_result["pages"] > 0, "应至少入库一个页面"

        if ingest_result["items"] == 0:
            _save_ocr_debug(results, label="no_ingested_items")
        assert ingest_result["items"] > 0, "应至少入库一个检验项目"
        print(f"[LoadOCR] 入库: {ingest_result['pages']} 页, {ingest_result['items']} 项")

        # ---- Step 5: 精确检索 ----
        print("[LoadOCR] Step 5: 精确检索 report_items...")

        sample_items = []
        for result in results:
            for page in result.pages:
                for table in page.tables:
                    for titem in table.items:
                        if titem.item and titem.item.strip():
                            sample_items.append(titem.item.strip())

        target_item = sample_items[0]
        query_result = client.query(
            collection_name="report_items",
            filter=f'item == "{target_item}"',
            output_fields=["*"],
        )
        assert len(query_result) > 0, f"精确查询 '{target_item}' 应有结果"
        for row in query_result:
            assert row["item"] == target_item
            assert "result" in row
            assert "reference_range" in row
            assert "table_title" in row
        print(f"[LoadOCR] 精确查询 '{target_item}': {len(query_result)} 条")

        # 全部 item 验证
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
        print(f"[LoadOCR] 全部 {all_queried} 个唯一 item 精确查询验证通过")

        # ---- Step 6: 语义检索 ----
        print("[LoadOCR] Step 6: 语义检索 report_pages...")

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
        print(f"[LoadOCR] 语义检索: {len(search_result)} 条")

        # ---- Step 7: page_data_json 完整性 ----
        print("[LoadOCR] Step 7: 验证 page_data_json 完整性...")

        full_result = client.query(
            collection_name="report_pages",
            filter="pk > 0",
            output_fields=["page_data_json", "summary_text"],
            limit=1,
        )
        assert len(full_result) > 0

        page_data = full_result[0].get("page_data_json")
        assert isinstance(page_data, dict), "page_data_json 应为 dict"

        if "text_analyses" in page_data:
            ta = page_data["text_analyses"]
            assert "personal_info" in ta
            assert "positive_findings" in ta

        if "tables" in page_data:
            for table in page_data["tables"]:
                assert "title" in table or "raw_md" in table or "items" in table

        if "regions" in page_data:
            assert isinstance(page_data["regions"], list)

        print("[LoadOCR] page_data_json 结构完整")
        print("[LoadOCR] ✅ 测试全部通过")
