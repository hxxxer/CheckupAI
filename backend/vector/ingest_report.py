"""
体检报告数据入库
OCRResult → report_pages + report_items

精确检索需求：
- report_items.item 标量字段支持 expr 精确匹配（如 item == "血糖"）
- report_items.item_dense/item_sparse 支持语义扩展
- report_pages 通过 summary 向量召回 → page_data_json 获取该页全部信息
"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from pymilvus import MilvusClient

from .config import MilvusLiteConfig
from .embeddings import BGEM3Embedder


class _DataclassEncoder(json.JSONEncoder):
    """支持 dataclass、datetime、tuple、Enum 的 JSON 编码器"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)


def _build_pk(raw: str) -> int:
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


def _page_to_dict(page) -> dict:
    """将 Page dataclass 转为可 JSON 序列化的 dict"""
    return json.loads(json.dumps(asdict(page), cls=_DataclassEncoder))


def ingest_checkup_report(
    client: MilvusClient,
    ocred_results: list,
    embedder: BGEM3Embedder | None = None,
) -> dict:
    """
    将体检报告 OCR 解析结果入库

    Args:
        client: MilvusClient 实例
        ocred_results: parse_checkup() 返回的 list[OCRResult]
        embedder: BGE-M3 编码器，None 则自动创建

    Returns:
        {"pages": count, "items": count}
    """
    if embedder is None:
        embedder = BGEM3Embedder()

    cfg = MilvusLiteConfig()
    pages_data = []
    items_data = []

    # 收集所有需要向量化的文本
    summaries: List[str] = []
    item_names: List[str] = []

    for result in ocred_results:
        source_path = result.source_path
        exam_date = ""

        if result.pages:
            personal_info = result.pages[0].text_analyses.personal_info if result.pages[0].text_analyses else None
            if personal_info and personal_info.exam_date:
                exam_date = personal_info.exam_date

        for page in result.pages:
            page_idx = page.page_index

            text_analysis = page.text_analyses
            summary_text = text_analysis.summary if text_analysis else ""

            # 收集 summary 用于向量化
            summaries.append(summary_text)

            # 整页转 dict（Milvus JSON 字段需要 Python dict，非 JSON 字符串）
            page_dict = _page_to_dict(page)

            pages_data.append({
                "source_path": source_path,
                "exam_date": exam_date,
                "page": page,
                "page_idx": page_idx,
                "summary_text": summary_text,
                "page_dict": page_dict,
            })

            # 提取表格检验项目
            for table in page.tables:
                table_title = table.title or ""
                for titem in table.items:
                    item_name = (titem.item or "").strip()
                    if not item_name:
                        continue

                    item_names.append(item_name)

                    items_data.append({
                        "source_path": source_path,
                        "exam_date": exam_date,
                        "page_idx": page_idx,
                        "table_title": table_title,
                        "item_name": item_name,
                        "result": titem.result or "",
                        "unit": titem.unit or "",
                        "abnormal": titem.abnormal or "",
                        "reference_range": titem.reference_range or "",
                    })

    # ===== 批量向量化 =====
    print(f"[ingest_report] 向量化 {len(summaries)} 个 summary...")
    summary_dense, summary_sparse = embedder.encode_hybrid(summaries)

    print(f"[ingest_report] 向量化 {len(item_names)} 个 item 名称...")
    # TODO: 后续可对 item + result + unit 联合向量化
    item_dense, item_sparse = embedder.encode_hybrid(item_names)

    # ===== 入库 report_pages =====
    page_rows = []
    for i, pd in enumerate(pages_data):
        pk = _build_pk(f"{pd['source_path']}_{pd['page_idx']}")
        page_rows.append({
            "pk": pk,
            "report_source": pd["source_path"],
            "exam_date": pd["exam_date"],
            "page_index": pd["page_idx"],
            "summary_text": pd["summary_text"],
            "page_data_json": pd["page_dict"],
            "summary_dense": summary_dense[i] if i < len(summary_dense) else [],
            "summary_sparse": summary_sparse[i] if i < len(summary_sparse) else {},
        })

    if page_rows:
        insert_result = client.insert(
            collection_name=cfg.COLLECTION_REPORT_PAGES,
            data=page_rows,
        )
        print(f"[ingest_report] report_pages 入库: {insert_result['insert_count']} 行")

    # ===== 入库 report_items =====
    item_rows = []
    for i, it in enumerate(items_data):
        pk = _build_pk(f"{it['source_path']}_{it['page_idx']}_{it['item_name']}")
        item_rows.append({
            "pk": pk,
            "report_source": it["source_path"],
            "exam_date": it["exam_date"],
            "page_index": it["page_idx"],
            "table_title": it["table_title"],
            "item": it["item_name"],
            "result": it["result"],
            "unit": it["unit"],
            "abnormal": it["abnormal"],
            "reference_range": it["reference_range"],
            "item_dense": item_dense[i] if i < len(item_dense) else [],
            "item_sparse": item_sparse[i] if i < len(item_sparse) else {},
        })

    if item_rows:
        insert_result = client.insert(
            collection_name=cfg.COLLECTION_REPORT_ITEMS,
            data=item_rows,
        )
        print(f"[ingest_report] report_items 入库: {insert_result['insert_count']} 行")

    return {"pages": len(page_rows), "items": len(item_rows)}
