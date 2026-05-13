"""
BaseScenario — 场景基类
子类化实现不同场景的 Prompt 和 Context 格式化
"""

import tomllib
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class BaseScenario:
    """场景基类：加载 TOML Prompt → 构建 Chain → 提供 format_context + invoke"""

    def __init__(self, prompt_path: str, llm):
        self.llm = llm
        self.system_content, self.user_template = self._load_prompt(prompt_path)
        self.chain = self._build_chain()

    def _load_prompt(self, path: str) -> tuple[str, str]:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return (
            data.get("system", {}).get("content", ""),
            data.get("user", {}).get("template", ""),
        )

    def _build_chain(self):
        template = ChatPromptTemplate.from_messages([
            ("system", self.system_content),
            ("user", self.user_template),
        ])
        return template | RunnableLambda(self.llm.invoke) | StrOutputParser()

    # ---------- 默认 Context 格式化（问答场景） ----------

    @staticmethod
    def _format_report_items(report_items: List[Dict]) -> str:
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

    @staticmethod
    def _format_report_page(page: Dict | None) -> str:
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
                    lines.append(f"  [{f.get('type', '')}] {f.get('text', '')}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_chunks(chunks: List[Dict], label: str, text_field: str = "content") -> str:
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

    def format_context(self, retrieval: Dict[str, Any]) -> str:
        """默认：4 段拼接（问答场景）"""
        parts = []
        items = retrieval.get("report_items", [])
        if items:
            parts.append(self._format_report_items(items))
        page = retrieval.get("report_page")
        if page:
            parts.append(self._format_report_page(page))
        knowledge = retrieval.get("knowledge_chunks", [])
        if knowledge:
            parts.append(self._format_chunks(knowledge, "权威医学知识", "content"))
        qa = retrieval.get("medical_qa", [])
        if qa:
            parts.append(self._format_chunks(qa, "相关问答参考", "document"))
        return "\n".join(parts) if parts else "无相关检索资料"

    def invoke(self, retrieval: Dict[str, Any], question: str, history: str = "") -> str:
        context = self.format_context(retrieval)
        return self.chain.invoke({
            "history": history,
            "context": context,
            "question": question,
        })
