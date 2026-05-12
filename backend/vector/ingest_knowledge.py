"""
权威知识库入库
chunks.json → knowledge_chunks

输入格式：[
    {
        "content": "医学知识文本...",
        "metadata": {"source": "/path/to/file.md", "filename": "file.md"}
    },
    ...
]
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from pymilvus import MilvusClient

from .config import MilvusLiteConfig
from .embeddings import BGEM3Embedder


def _build_pk(raw: str) -> int:
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


def load_chunks_json(chunks_path: str) -> List[Dict]:
    """从 chunks.json 加载分块数据"""
    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks.json 不存在: {chunks_path}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[ingest_knowledge] 加载 {len(chunks)} 个分块 from {chunks_path}")
    return chunks


def ingest_knowledge_chunks(
    client: MilvusClient,
    chunks: List[Dict] | None = None,
    chunks_path: str | None = None,
    embedder: BGEM3Embedder | None = None,
) -> dict:
    """
    将知识库分块入库

    Args:
        client: MilvusClient 实例
        chunks: 分块数据列表，与 chunks_path 二选一
        chunks_path: chunks.json 路径，默认 data/knowledge_base/final_chunks/chunks.json
        embedder: BGE-M3 编码器

    Returns:
        {"inserted": count}
    """
    if embedder is None:
        embedder = BGEM3Embedder()

    if chunks is None:
        if chunks_path is None:
            chunks_path = str(
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "knowledge_base"
                / "final_chunks"
                / "chunks.json"
            )
        chunks = load_chunks_json(chunks_path)

    if not chunks:
        print("[ingest_knowledge] 无数据可入库")
        return {"inserted": 0}

    cfg = MilvusLiteConfig()

    contents = []
    rows = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if not content:
            continue

        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "")
        filename = metadata.get("filename", "")

        pk = _build_pk(f"{source}_{content[:100]}")
        contents.append(content)
        rows.append({
            "pk": pk,
            "content": content,
            "source": source,
            "filename": filename,
        })

    print(f"[ingest_knowledge] 向量化 {len(contents)} 个 content...")
    dense_vecs, sparse_vecs = embedder.encode_hybrid(contents)

    for i, row in enumerate(rows):
        row["content_dense"] = dense_vecs[i] if i < len(dense_vecs) else []
        row["content_sparse"] = sparse_vecs[i] if i < len(sparse_vecs) else {}

    if rows:
        insert_result = client.insert(
            collection_name=cfg.COLLECTION_KNOWLEDGE_CHUNKS,
            data=rows,
        )
        print(f"[ingest_knowledge] knowledge_chunks 入库: {insert_result['insert_count']} 行")
        return {"inserted": insert_result["insert_count"]}

    return {"inserted": 0}
