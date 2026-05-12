"""
医疗问答资料入库
CSV → medical_qa

参考 temp/qwen3-med_rag/src/emb.py 模式
输入 CSV 格式：department, title, ask, answer
"""

import hashlib
from typing import Dict, List

import pandas as pd
from pymilvus import MilvusClient

from .config import MilvusLiteConfig
from .embeddings import BGEM3Embedder


def _build_pk(raw: str) -> int:
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


def ingest_qa_from_csv(
    client: MilvusClient,
    csv_path: str,
    embedder: BGEM3Embedder | None = None,
) -> dict:
    """
    将医疗问答 CSV 入库

    Args:
        client: MilvusClient 实例
        csv_path: CSV 文件路径
        embedder: BGE-M3 编码器

    Returns:
        {"inserted": count}
    """
    if embedder is None:
        embedder = BGEM3Embedder()

    cfg = MilvusLiteConfig()

    df = pd.read_csv(csv_path)
    print(f"[ingest_qa] 读取 CSV: {csv_path}, 共 {len(df)} 行")

    summary_texts: List[str] = []
    full_texts: List[str] = []
    rows: List[Dict] = []

    for _, row in df.iterrows():
        department = str(row.get("department", ""))
        title = str(row.get("title", ""))
        ask = str(row.get("ask", ""))
        answer = str(row.get("answer", ""))

        summary = title
        document = answer
        text = f"【{department}】{title}\n{ask}\n{answer}"

        pk = _build_pk(f"{department}_{title}")

        summary_texts.append(summary)
        full_texts.append(text)
        rows.append({
            "pk": pk,
            "text": text,
            "summary": summary,
            "document": document,
            "department": department,
        })

    # 向量化 summary (Dense only — for summary_dense field)
    print(f"[ingest_qa] 向量化 {len(summary_texts)} 个 summary...")
    summary_dense = embedder.encode_dense(summary_texts)

    # 向量化 text (Dense + Sparse — for text_dense and text_sparse fields)
    print(f"[ingest_qa] 向量化 {len(full_texts)} 个 text...")
    text_dense, text_sparse = embedder.encode_hybrid(full_texts)

    for i, row in enumerate(rows):
        row["summary_dense"] = summary_dense[i] if i < len(summary_dense) else []
        row["text_dense"] = text_dense[i] if i < len(text_dense) else []
        row["text_sparse"] = text_sparse[i] if i < len(text_sparse) else {}

    if rows:
        insert_result = client.insert(
            collection_name=cfg.COLLECTION_MEDICAL_QA,
            data=rows,
        )
        print(f"[ingest_qa] medical_qa 入库: {insert_result['insert_count']} 行")
        return {"inserted": insert_result["insert_count"]}

    return {"inserted": 0}
