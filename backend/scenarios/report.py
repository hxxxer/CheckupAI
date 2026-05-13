"""
ReportScenario — 报告生成场景
结构化输出，完整表格 + 全字段
"""

from typing import Any, Dict, List

from .base import BaseScenario


class ReportScenario(BaseScenario):
    """报告生成：完整检验项目表格，结构化章节输出"""

    def format_context(self, retrieval: Dict[str, Any]) -> str:
        parts = []

        # 报告页面完整信息（展开）
        page = retrieval.get("report_page")
        if page:
            parts.append(self._format_report_page_full(page))

        # 全部检验项目（完整表格，不截断）
        items = retrieval.get("report_items", [])
        if items:
            parts.append(self._format_report_items_full(items))

        # 权威知识 + 问答（保留）
        knowledge = retrieval.get("knowledge_chunks", [])
        if knowledge:
            parts.append(self._format_chunks(knowledge, "权威医学知识", "content"))
        qa = retrieval.get("medical_qa", [])
        if qa:
            parts.append(self._format_chunks(qa, "相关问答参考", "document"))

        return "\n".join(parts) if parts else "无相关检索资料"

    # ---------- 完整格式化 ----------

    @staticmethod
    def _format_report_page_full(page: Dict) -> str:
        """完整页面信息：个人信息 + 全部异常发现 + 全量文本"""
        if not page:
            return ""
        lines = ["【体检报告完整信息】"]
        exam_date = page.get("exam_date", "")
        if exam_date:
            lines.append(f"体检日期: {exam_date}")

        page_data = page.get("page_data_json", {})
        if isinstance(page_data, dict):
            # 个人信息
            analyses = page_data.get("text_analyses", {})
            if analyses:
                pi = analyses.get("personal_info", {})
                if pi:
                    parts = []
                    if pi.get("name"):
                        parts.append(f"姓名: {pi['name']}")
                    if pi.get("gender"):
                        parts.append(f"性别: {pi['gender']}")
                    if pi.get("age"):
                        parts.append(f"年龄: {pi['age']}")
                    if parts:
                        lines.append("个人信息: " + ", ".join(parts))

                # 全部异常发现
                findings = analyses.get("positive_findings", [])
                if findings:
                    lines.append(f"异常发现（共 {len(findings)} 项）:")
                    for f in findings:
                        lines.append(f"  [{f.get('type', '')}] {f.get('text', '')}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_report_items_full(items: List[Dict]) -> str:
        """完整检验项目表格，全部展示，表格化"""
        if not items:
            return ""
        lines = ["【全部检验项目】"]

        # Markdown 表格
        header = "| 项目 | 结果 | 单位 | 参考范围 | 异常 | 来源 |"
        sep = "|------|------|------|----------|------|------|"
        rows = [header, sep]
        for item in items:
            name = item.get("item", "")
            result = item.get("result", "")
            unit = item.get("unit", "")
            ref = item.get("reference_range", "")
            abnormal = item.get("abnormal", "")
            table = item.get("table_title", "")
            rows.append(f"| {name} | {result} | {unit} | {ref} | {abnormal} | {table} |")

        lines.extend(rows)
        return "\n".join(lines) + "\n"
