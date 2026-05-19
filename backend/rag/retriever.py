"""
MedicalRAG — 三路检索编排

检索流程：
1. QueryRewriter 改写查询 + 判断是否需要报告 + 联想指标
2. 指标统一化（复用 medical_terms_map.json）
3. 三路并行检索：
   A. report_items 精确匹配 → 全部返回
   B. report_pages 向量检索 → 按 exam_date 取最近一份
   C. knowledge_chunks + medical_qa 向量检索 → rerank → 各取 top 3
"""

from typing import Any, Dict, List

from pymilvus import MilvusClient

from backend.llm import QueryRewriter
from backend.llm.utils import medical_term_normalizer
from backend.vector.config import MilvusLiteConfig
from backend.vector.embeddings import BGEM3Embedder

from .reranker import BGEReranker


class MedicalRAG:
    """体检报告 RAG 检索器"""

    def __init__(
        self,
        client: MilvusClient,
        embedder: BGEM3Embedder | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: BGEReranker | None = None,
    ):
        self.client = client
        self.cfg = MilvusLiteConfig()
        self.embedder = embedder or BGEM3Embedder()
        self.query_rewriter = query_rewriter  # lazily loaded
        self.reranker = reranker or BGEReranker()

    def _get_query_rewriter(self) -> QueryRewriter:
        if self.query_rewriter is None:
            from backend.llm import query_rewriter as qr
            self.query_rewriter = qr
        return self.query_rewriter

    def _normalize_indicators(self, indicators: List[str]) -> List[str]:
        """用 MedicalTermNormalizer 统一化指标名"""
        return medical_term_normalizer.normalize_list(indicators)

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedder.encode_dense([text])[0]

    # ---------- 路径 A: report_items 精确匹配 ----------

    def _retrieve_report_items(self, indicators: List[str]) -> List[Dict]:
        """返回全部匹配的检验项目"""
        if not indicators:
            return []

        all_items = []
        seen = set()

        for ind in indicators:
            try:
                results = self.client.query(
                    collection_name=self.cfg.COLLECTION_REPORT_ITEMS,
                    filter=f'item == "{ind}"',
                    output_fields=["*"],
                )
                for row in results:
                    key = f"{row.get('report_source')}_{row.get('page_index')}_{row.get('item')}"
                    if key not in seen:
                        seen.add(key)
                        all_items.append(row)
            except Exception as e:
                print(f"[MedicalRAG] report_items 查询失败 (indicator={ind}): {e}")

        return all_items

    # ---------- 路径 B: report_pages 向量检索 ----------

    def _retrieve_report_page(self, query_vec: List[float]) -> Dict | None:
        """向量检索 report_pages → 按 exam_date 降序取最近一份"""
        try:
            results = self.client.search(
                collection_name=self.cfg.COLLECTION_REPORT_PAGES,
                data=[query_vec],
                anns_field="summary_dense",
                limit=10,
                output_fields=["*"],
            )[0]  # 取第一条查询结果
        except Exception as e:
            print(f"[MedicalRAG] report_pages 向量检索失败: {e}")
            return None

        if not results:
            return None

        # 提取实体，按 exam_date 降序
        hits_with_date = []
        for hit in results:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            if not entity:
                continue
            exam_date = entity.get("exam_date", "")
            hits_with_date.append((exam_date, entity))

        # exam_date 降序，空日期排最后
        hits_with_date.sort(
            key=lambda x: (x[0] if x[0] else ""),
            reverse=True,
        )

        return hits_with_date[0][1] if hits_with_date else None

    # ---------- 路径 C: 向量检索 + rerank ----------

    def _retrieve_and_rerank(
        self,
        query: str,
        query_vec: List[float],
        collection: str,
        anns_field: str,
        output_fields: List[str],
        retrieve_k: int = 10,
        rerank_k: int = 3,
        text_field: str = "text",
    ) -> List[Dict]:
        """通用：向量检索 → 转为通用格式 → rerank → top k"""
        try:
            results = self.client.search(
                collection_name=collection,
                data=[query_vec],
                anns_field=anns_field,
                limit=retrieve_k,
                output_fields=output_fields,
            )[0]
        except Exception as e:
            print(f"[MedicalRAG] {collection} 检索失败: {e}")
            return []

        if not results:
            return []

        docs = []
        for hit in results:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            if not entity:
                continue
            doc = dict(entity)
            doc["text"] = doc.get(text_field, "")
            doc["_distance"] = hit.get("distance", 0) if isinstance(hit, dict) else 0
            docs.append(doc)

        return self.reranker.rerank(query, docs, top_k=rerank_k)

    # ---------- 主入口 ----------

    def retrieve(
        self,
        query: str,
        skip_qa: bool = False,
        knowledge_rerank_k: int = 3,
    ) -> Dict[str, Any]:
        """
        执行三路检索

        Args:
            query: 用户原始输入（口语化）
            skip_qa: 跳过问答资料检索（报告模式用）
            knowledge_rerank_k: 权威知识 rerank 后取 top k

        Returns:
            {
                "rewritten_query": str,
                "need_report": bool,
                "enable_rag": bool,
                "indicators": [str, ...],
                "report_items": [dict, ...],
                "report_page": dict | None,
                "knowledge_chunks": [dict, ...],
                "medical_qa": [dict, ...],
            }
        """
        # Step 1: 查询改写
        qr = self._get_query_rewriter()
        rewrite_result = qr.rewrite(query)

        rewritten = rewrite_result["rewritten"]
        need_report = rewrite_result["need_report"]
        enable_rag = rewrite_result["enable_rag"]
        raw_indicators = rewrite_result["indicators"]

        # Step 2: 指标统一化
        indicators = self._normalize_indicators(raw_indicators)

        # RAG 路由：不需要检索知识库则直接返回空结果
        if not enable_rag:
            return {
                "rewritten_query": rewritten,
                "need_report": need_report,
                "enable_rag": enable_rag,
                "indicators": indicators,
                "report_items": [],
                "report_page": None,
                "knowledge_chunks": [],
                "medical_qa": [],
            }

        # Step 3: 生成向量（一次，共用）
        query_vec = self._get_embedding(rewritten)

        # 路径 A: 精确匹配 report_items
        report_items = []
        if need_report and indicators:
            report_items = self._retrieve_report_items(indicators)

        # 路径 B: 向量检索 report_pages → 最近一份
        report_page = None
        if need_report:
            report_page = self._retrieve_report_page(query_vec)

        # 路径 C1: knowledge_chunks（始终检索 10 条，rerank 后取 knowledge_rerank_k 条）
        knowledge_chunks = self._retrieve_and_rerank(
            query=rewritten,
            query_vec=query_vec,
            collection=self.cfg.COLLECTION_KNOWLEDGE_CHUNKS,
            anns_field="content_dense",
            output_fields=["content", "source", "filename"],
            retrieve_k=10,
            rerank_k=knowledge_rerank_k,
            text_field="content",
        )

        # 路径 C2: medical_qa（报告模式下跳过）
        medical_qa = []
        if not skip_qa:
            medical_qa = self._retrieve_and_rerank(
                query=rewritten,
                query_vec=query_vec,
                collection=self.cfg.COLLECTION_MEDICAL_QA,
                anns_field="summary_dense",
                output_fields=["summary", "document", "department", "text"],
                retrieve_k=10,
                rerank_k=3,
                text_field="document",
            )

        return {
            "rewritten_query": rewritten,
            "need_report": need_report,
            "enable_rag": enable_rag,
            "indicators": indicators,
            "report_items": report_items,
            "report_page": report_page,
            "knowledge_chunks": knowledge_chunks,
            "medical_qa": medical_qa,
        }
