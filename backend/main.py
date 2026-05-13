"""
CheckupAI Backend Service
FastAPI + LangChain RAG Chain + Milvus Lite + vLLM

启动流程（模块级，import 时执行）：
1. 检查/初始化 Milvus Lite 向量数据库
2. 若 report_pages 为空 → OCR 解析测试图片 → 入库
3. 初始化 MedicalRAG（QueryRewriter + BGE-M3 + BGEReranker）
4. 初始化 vLLM OpenAI 客户端
5. 构建 LangChain Chain
6. 启动 FastAPI
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from backend.config import settings
from backend.llm import chat_llm
from backend.rag import MedicalRAG, BGEReranker
from backend.vector import get_milvus_client

# ============================================================
# 模块级启动：检查 DB → OCR 入库 → 初始化组件 → 构建 Chain
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

# ---- Step 3: 初始化 MedicalRAG ----
print("[CheckupAI] 初始化 MedicalRAG...")
rag = MedicalRAG(client)
print("[CheckupAI] MedicalRAG 就绪")

# ---- Step 4: LLM 初始化（已在 backend.llm.chat_llm 模块级完成） ----
print(f"[CheckupAI] LLM 就绪 (model={chat_llm.model})")

# ---- Step 5: 构建 LangChain Chain ----

SYSTEM_PROMPT = (
    "你是一个专业的AI医疗健康助手，名字叫CheckupAI。\n"
    "你的任务是基于提供的检索资料，为用户提供专业、准确、易懂的健康咨询回答。\n\n"
    "要求：\n"
    "1. 如果检索到了用户的体检报告数据，优先结合用户的个人指标进行分析\n"
    "2. 引用权威医学知识时，标注来源\n"
    "3. 如果检索资料不足以回答，请如实告知，不要编造\n"
    "4. 回答末尾可以给出进一步的健康建议或就医指引\n"
    "5. 严禁给出具体的药物处方或剂量建议"
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user",
     "检索资料：\n"
     "{context}\n\n"
     "用户问题：{question}\n\n"
     "请提供专业回答："),
])


chain = PROMPT_TEMPLATE | RunnableLambda(chat_llm.invoke) | StrOutputParser()
print("[CheckupAI] LangChain Chain 构建完成")

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
# Context 格式化
# ============================================================

def _format_report_items(report_items: List[Dict]) -> str:
    """格式化体检检验项目"""
    if not report_items:
        return ""
    lines = ["【用户体检报告相关指标】"]
    for item in report_items:
        name = item.get("item", "N/A")
        result = item.get("result", "")
        unit = item.get("unit", "")
        ref = item.get("reference_range", "")
        abnormal = item.get("abnormal", "")
        table = item.get("table_title", "")

        line = f"- {name}: {result} {unit}"
        if ref:
            line += f" (参考范围: {ref})"
        if abnormal:
            line += f" [异常: {abnormal}]"
        if table:
            line += f"（来自{table}）"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _format_report_page(page: Dict | None) -> str:
    """格式化体检报告页面摘要"""
    if not page:
        return ""
    lines = ["【体检报告页面信息】"]
    exam_date = page.get("exam_date", "")
    summary = page.get("summary_text", "")
    if exam_date:
        lines.append(f"体检日期: {exam_date}")
    if summary:
        lines.append(f"摘要: {summary}")

    page_data = page.get("page_data_json", {})
    if isinstance(page_data, dict):
        findings = page_data.get("text_analyses", {}).get("positive_findings", [])
        if findings:
            lines.append("异常发现:")
            for f in findings[:5]:
                f_type = f.get("type", "")
                f_text = f.get("text", "")
                lines.append(f"  [{f_type}] {f_text}")

    return "\n".join(lines) + "\n"


def _format_chunks(chunks: List[Dict], label: str, text_field: str = "content") -> str:
    """通用：格式化检索文档（知识库 / 问答资料）"""
    if not chunks:
        return ""
    lines = [f"【{label}】"]
    for i, doc in enumerate(chunks, 1):
        score = doc.get("rerank_score", 0)
        text = doc.get(text_field, "")
        source = doc.get("source", "") or doc.get("filename", "") or doc.get("summary", "")
        lines.append(f"[{i}] (score:{score:.2f}) {text}")
        if source:
            lines.append(f"    来源: {source}")
    return "\n".join(lines) + "\n"


def _format_context(retrieval: Dict[str, Any]) -> str:
    """将检索结果格式化为 LLM 完整上下文"""
    parts = []

    report_items = retrieval.get("report_items", [])
    if report_items:
        parts.append(_format_report_items(report_items))

    report_page = retrieval.get("report_page")
    if report_page:
        parts.append(_format_report_page(report_page))

    knowledge = retrieval.get("knowledge_chunks", [])
    if knowledge:
        parts.append(_format_chunks(knowledge, "权威医学知识", "content"))

    qa = retrieval.get("medical_qa", [])
    if qa:
        parts.append(_format_chunks(qa, "相关问答参考", "document"))

    return "\n".join(parts) if parts else "无相关检索资料"


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
    """主问答入口：三路检索 → LLM 生成回答"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # Step 1: RAG 检索
    retrieval = rag.retrieve(request.question)

    # Step 2: 格式化上下文
    context = _format_context(retrieval)

    # Step 3: LLM 生成
    try:
        answer = chain.invoke({
            "context": context,
            "question": request.question,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")

    return ChatResponse(answer=answer, retrieval=retrieval)


@app.get("/api/health")
def health_check():
    """健康检查：返回 DB 状态和各 Collection 行数"""
    status = {"status": "healthy", "collections": {}}
    try:
        for coll_name in [
            "report_pages",
            "report_items",
            "knowledge_chunks",
            "medical_qa",
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
