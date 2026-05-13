"""
CheckupAI Backend Service
FastAPI + LangChain RAG Chain + Milvus Lite + vLLM

启动流程（模块级，import 时执行）：
1. 检查/初始化 Milvus Lite 向量数据库
2. 若 report_pages 为空 → OCR 解析测试图片 → 入库
3. 初始化 MedicalRAG（QueryRewriter + BGE-M3 + BGEReranker）
4. 初始化 ChatLLM（已在 backend.llm.chat_llm 模块级完成）
5. 构建场景（ChatScenario + ReportScenario）
6. 启动 FastAPI
"""

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.llm import chat_llm
from backend.rag import MedicalRAG
from backend.scenarios import ChatScenario, ReportScenario
from backend.vector import get_milvus_client

# ============================================================
# 模块级启动
# ============================================================

print("=" * 50)
print("[CheckupAI] 启动中...")

# ---- Step 1: 检查/初始化向量数据库 ----
print("[CheckupAI] 检查向量数据库...")
client = get_milvus_client()
print("[CheckupAI] 向量数据库就绪")

# ---- Step 2: 若 report_pages 为空则 OCR 解析测试图片入库 ----
try:
    count_result = client.query(
        collection_name="report_pages",
        expr="pk > 0",
        output_fields=["pk"],
        limit=1,
    )
    report_pages_empty = len(count_result) == 0
except Exception:
    report_pages_empty = True

if report_pages_empty:
    print("[CheckupAI] report_pages 为空，开始 OCR 解析测试图片...")
    test_files_path = os.path.join(settings.project_root, "tests/test_ocr/test_files/")

    if os.path.isdir(test_files_path):
        try:
            from backend.ocr import parse_checkup
            from backend.vector import ingest_checkup_report

            results = parse_checkup(test_files_path)
            if results:
                ingest_checkup_report(client, results)
                print(f"[CheckupAI] 体检报告入库完成 ({len(results)} 份文件)")
            else:
                print("[CheckupAI] 警告：OCR 解析结果为空")
        except Exception as e:
            print(f"[CheckupAI] 警告：OCR 解析/入库失败 ({e})，跳过")
    else:
        print(f"[CheckupAI] 警告：测试图片目录不存在 ({test_files_path})，跳过 OCR")
else:
    print("[CheckupAI] report_pages 已有数据，跳过 OCR")

# ---- Step 2b: 若 knowledge_chunks 为空则从 chunks.json 入库 ----
try:
    count_result = client.query(
        collection_name="knowledge_chunks",
        expr="pk > 0",
        output_fields=["pk"],
        limit=1,
    )
    kc_empty = len(count_result) == 0
except Exception:
    kc_empty = True

if kc_empty:
    chunks_path = os.path.join(
        settings.project_root, "data", "knowledge_base", "final_chunks", "chunks.json"
    )
    if os.path.isfile(chunks_path):
        print(f"[CheckupAI] knowledge_chunks 为空，入库 chunks.json...")
        try:
            from backend.vector import ingest_knowledge_chunks
            ingest_knowledge_chunks(client, chunks_path=str(chunks_path))
            print("[CheckupAI] 知识库入库完成")
        except Exception as e:
            print(f"[CheckupAI] 警告：知识库入库失败 ({e})，跳过")
    else:
        print(f"[CheckupAI] 提示：chunks.json 不存在 ({chunks_path})，跳过知识库入库")
else:
    print("[CheckupAI] knowledge_chunks 已有数据，跳过")

# ---- Step 2c: 若 medical_qa 为空则从 qa.csv 入库 ----
try:
    count_result = client.query(
        collection_name="medical_qa",
        expr="pk > 0",
        output_fields=["pk"],
        limit=1,
    )
    qa_empty = len(count_result) == 0
except Exception:
    qa_empty = True

if qa_empty:
    qa_csv_path = os.path.join(settings.project_root, "data", "qa.csv")
    if os.path.isfile(qa_csv_path):
        print(f"[CheckupAI] medical_qa 为空，入库 qa.csv...")
        try:
            from backend.vector import ingest_qa_from_csv
            ingest_qa_from_csv(client, csv_path=str(qa_csv_path))
            print("[CheckupAI] 问答资料入库完成")
        except Exception as e:
            print(f"[CheckupAI] 警告：问答资料入库失败 ({e})，跳过")
    else:
        print(f"[CheckupAI] 提示：qa.csv 不存在 ({qa_csv_path})，跳过问答资料入库")
else:
    print("[CheckupAI] medical_qa 已有数据，跳过")

# ---- Step 3: 初始化 MedicalRAG ----
print("[CheckupAI] 初始化 MedicalRAG...")
rag = MedicalRAG(client)
print("[CheckupAI] MedicalRAG 就绪")

# ---- Step 4: LLM 初始化（已在 backend.llm.chat_llm 模块级完成） ----
print(f"[CheckupAI] LLM 就绪 (model={chat_llm.model})")

# ---- Step 5: 构建场景 ----
print("[CheckupAI] 构建场景...")
chat_scenario = ChatScenario(settings.llm_chat_prompt, chat_llm)
report_scenario = ReportScenario(settings.llm_report_prompt, chat_llm)
print("[CheckupAI] 场景就绪 (chat + report)")

# ---- Step 6: 启动 FastAPI ----
app = FastAPI(title="CheckupAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[CheckupAI] 启动完成")
print("=" * 50)


# ============================================================
# Pydantic Models
# ============================================================

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieval: Dict[str, Any]


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def read_root():
    return {"message": "CheckupAI API is running"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """问答入口：三路检索 → LLM 生成回答"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    retrieval = rag.retrieve(request.question)

    try:
        answer = chat_scenario.invoke(retrieval, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")

    return ChatResponse(answer=answer, retrieval=retrieval)


@app.post("/api/report", response_model=ChatResponse)
async def generate_report(request: ChatRequest):
    """报告生成入口：三路检索（跳过问答，权威 rerank top 1）→ LLM 生成结构化报告"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    retrieval = rag.retrieve(request.question, skip_qa=True, knowledge_rerank_k=1)

    try:
        report = report_scenario.invoke(retrieval, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")

    return ChatResponse(answer=report, retrieval=retrieval)


@app.get("/api/health")
def health_check():
    """健康检查：返回 DB 状态和各 Collection 行数"""
    status = {"status": "healthy", "collections": {}}
    try:
        for coll_name in [
            "report_pages", "report_items",
            "knowledge_chunks", "medical_qa",
        ]:
            try:
                count_result = client.query(
                    collection_name=coll_name,
                    expr="pk > 0",
                    output_fields=["pk"],
                    limit=1,
                )
                status["collections"][coll_name] = "有数据" if count_result else "空"
            except Exception:
                status["collections"][coll_name] = "查询失败"
    except Exception as e:
        status["status"] = f"degraded: {e}"
    return status


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
