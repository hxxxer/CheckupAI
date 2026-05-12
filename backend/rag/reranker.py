"""
Reranker 模块
使用 BGE-reranker-v2-m3 对检索结果重排序

输入格式：documents 为 list[dict]，每项必须含 "text" 字段
"""

from FlagEmbedding import FlagReranker


class BGEReranker:
    """BGE-Reranker 重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
        """
        对文档列表重排序

        Args:
            query: 查询文本
            documents: 文档列表，每项必须含 "text" 字段
            top_k: 返回前 k 条，None 则返回全部

        Returns:
            重排序后的文档列表，增加 "rerank_score" 字段
        """
        if not documents:
            return []

        pairs = [[query, doc.get("text", "")] for doc in documents]
        scores = self.reranker.compute_score(pairs, normalize=True)

        if isinstance(scores, float):
            scores = [scores]

        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k:
            return reranked[:top_k]
        return reranked
