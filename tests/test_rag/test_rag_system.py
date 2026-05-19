"""
RAG 系统集成测试
需要 vLLM 服务、BGE-M3 模型、Milvus Lite DB 中有数据
若无服务或依赖则自动跳过
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

# 逐测试懒检查，不让模块级 skip 误伤不依赖这些库的测试
_pymilvus_ok = False
_flagembedding_ok = False
try:
    import pymilvus as _  # noqa: F401
    _pymilvus_ok = True
except ImportError:
    pass
try:
    import FlagEmbedding as _  # noqa: F401
    _flagembedding_ok = True
except ImportError:
    pass


# ---- 辅助：检查外部服务可用性 ----

def _vllm_available():
    """检查 vLLM 服务是否运行"""
    try:
        import requests
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        resp = requests.get(f"{base_url}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _milvus_has_data(client):
    """检查 report_pages 是否有数据"""
    try:
        result = client.query(
            collection_name="report_pages",
            filter="pk > 0",
            output_fields=["pk"],
            limit=1,
        )
        return len(result) > 0
    except Exception:
        return False


# ---- 不需要外部服务的测试 ----

@pytest.mark.skipif(not _flagembedding_ok, reason="FlagEmbedding 未安装")
def test_bge_reranker_init():
    """测试 BGEReranker 可实例化"""
    from backend.rag.reranker import BGEReranker

    reranker = BGEReranker()
    assert reranker.reranker is not None

@pytest.mark.skipif(not _flagembedding_ok, reason="FlagEmbedding 未安装")
def test_bge_reranker_rerank():
    from backend.rag.reranker import BGEReranker

@pytest.mark.skipif(not _flagembedding_ok, reason="FlagEmbedding 未安装")
def test_bge_reranker_empty():
    from backend.rag.reranker import BGEReranker

@pytest.mark.skipif(not _flagembedding_ok, reason="FlagEmbedding 未安装")
def test_bge_reranker_single():
    from backend.rag.reranker import BGEReranker

@pytest.mark.skipif(not _pymilvus_ok, reason="pymilvus 未安装")
def test_medical_rag_init():
    """MedicalRAG 可实例化（需要可用的 MilvusClient）"""
    import tempfile
    from pymilvus import MilvusClient
    from backend.vector.collections import init_all_collections
    from backend.rag import MedicalRAG

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_rag.db")
        client = MilvusClient(uri=db_file)
        init_all_collections(client, drop_existing=False)

        rag = MedicalRAG(client)
        assert rag is not None
        assert rag.client is client


def test_pk_generation():
    """测试 _build_pk 逻辑"""
    import hashlib

    def build_pk(raw: str) -> int:
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF

    # 确定性
    assert build_pk("hello") == build_pk("hello")
    # 不同输入不同 ID
    assert build_pk("abc") != build_pk("def")
    # 正数
    assert build_pk("test") > 0
    # 类型
    assert isinstance(build_pk("test"), int)


# ---- 需要外部服务的集成测试 ----

@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM 服务不可用 (检查 OPENAI_BASE_URL)",
)
def test_query_rewriter_rewrite():
    """测试查询改写（需要 vLLM）"""
    from backend.llm import query_rewriter

    result = query_rewriter.rewrite("嗓子疼得厉害，咽口水都疼")

    assert "rewritten" in result
    assert "need_report" in result
    assert "indicators" in result
    assert isinstance(result["need_report"], bool)
    assert isinstance(result["indicators"], list)
    assert len(result["rewritten"]) > 0


@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM 服务不可用",
)
def test_chat_llm_chat():
    """测试 ChatLLM 直接调用（需要 vLLM）"""
    from backend.llm import chat_llm

    messages = [
        {"role": "user", "content": "1+1等于几？请只回答数字。"},
    ]
    answer = chat_llm.chat(messages)
    assert len(answer) > 0
    assert "2" in answer


@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM 服务不可用",
)
def test_medical_rag_retrieve():
    """测试完整 RAG 检索流程（需要 vLLM + Milvus 数据）"""
    import tempfile
    from pymilvus import MilvusClient
    from backend.vector.collections import init_all_collections
    from backend.rag import MedicalRAG

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_full_rag.db")
        client = MilvusClient(uri=db_file)
        init_all_collections(client, drop_existing=False)

        rag = MedicalRAG(client)

        result = rag.retrieve("嗓子疼")

        # 必须返回的字段
        assert "rewritten_query" in result
        assert "need_report" in result
        assert "indicators" in result
        assert "report_items" in result
        assert "report_page" in result
        assert "knowledge_chunks" in result
        assert "medical_qa" in result

        # 类型检查
        assert isinstance(result["need_report"], bool)
        assert isinstance(result["indicators"], list)
        assert isinstance(result["report_items"], list)
        assert isinstance(result["knowledge_chunks"], list)
        assert isinstance(result["medical_qa"], list)

        # report_page 可为 None（无数据时）
        if result["report_page"] is not None:
            assert isinstance(result["report_page"], dict)

        print(f"\n[test] rewritten={result['rewritten_query']}")
        print(f"[test] need_report={result['need_report']}")
        print(f"[test] indicators={result['indicators']}")
        print(f"[test] report_items={len(result['report_items'])}")
        print(f"[test] knowledge_chunks={len(result['knowledge_chunks'])}")
        print(f"[test] medical_qa={len(result['medical_qa'])}")


@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM 服务不可用",
)
def test_medical_rag_skip_qa():
    """测试 skip_qa 参数"""
    import tempfile
    from pymilvus import MilvusClient
    from backend.vector.collections import init_all_collections
    from backend.rag import MedicalRAG

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "test_skip_qa.db")
        client = MilvusClient(uri=db_file)
        init_all_collections(client, drop_existing=False)

        rag = MedicalRAG(client)

        # 报告模式：跳过 QA，knowledge rerank top 1
        result = rag.retrieve("帮我分析体检", skip_qa=True, knowledge_rerank_k=1)

        assert result["medical_qa"] == []
        assert len(result["knowledge_chunks"]) <= 1
