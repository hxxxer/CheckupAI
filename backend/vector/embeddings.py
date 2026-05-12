"""
BGE-M3 嵌入模型包装器
单例加载，支持 Dense + Sparse 向量编码
参考 temp/qwen3-med_rag/src/embed/ 模式
"""

import os
from typing import Dict, List

from FlagEmbedding import BGEM3FlagModel

_bgem3_instance = None


def get_bgem3_model(device: str = "cuda") -> BGEM3FlagModel:
    """获取 BGE-M3 单例"""
    global _bgem3_instance
    if _bgem3_instance is None:
        model_path = os.getenv("BGE_M3_MODEL_PATH", "BAAI/bge-m3")
        print(f"[BGE-M3] 加载模型: {model_path}")
        _bgem3_instance = BGEM3FlagModel(
            model_path,
            device=device,
            use_fp16=True,
        )
        print("[BGE-M3] 模型加载完成")
    return _bgem3_instance


class BGEM3Embedder:
    """BGE-M3 编码器：同时输出 Dense 和 Sparse 向量"""

    def __init__(self, device: str = "cuda"):
        self.model = get_bgem3_model(device)

    def encode_dense(self, texts: List[str], batch_size: int = 12) -> List[List[float]]:
        """仅生成 Dense 向量"""
        if not texts:
            return []
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            batch_size=batch_size,
        )
        return output["dense_vecs"].tolist()

    def encode_hybrid(self, texts: List[str], batch_size: int = 12):
        """同时生成 Dense + Sparse 向量

        Returns:
            (dense_vecs: List[List[float]], sparse_vecs: List[Dict[int, float]])
        """
        if not texts:
            return [], []
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            batch_size=batch_size,
        )
        dense = output["dense_vecs"].tolist()
        sparse = output["lexical_weights"]
        return dense, sparse

    def encode_sparse_query(self, text: str) -> Dict[int, float]:
        """为查询生成 Sparse 向量"""
        output = self.model.encode(
            [text],
            return_sparse=True,
            return_dense=False,
        )
        return output["lexical_weights"][0]
