"""
查询改写模块
使用 LLM 将用户口语化问题改写为标准医学查询，并联想相关检验指标
"""

import os
from typing import Any, Dict, List

from backend.config import settings
from .base_llm import BaseLLM
from .utils import safe_json_parse


class QueryRewriter(BaseLLM):
    """查询改写器：去口语化 + 判断是否需要报告/RAG + 联想检验指标"""

    def __init__(
        self,
        prompt_path: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B",
        enable_thinking: bool = False,
    ):
        super().__init__(
            prompt_path=prompt_path,
            base_url=base_url,
            api_key=api_key,
            model=model,
            enable_thinking=enable_thinking,
        )

    def rewrite(self, query: str) -> Dict[str, Any]:
        """
        改写用户查询

        Args:
            query: 用户原始口语化输入

        Returns:
            {
                "rewritten": "标准化改写查询",
                "need_report": True/False,
                "enable_rag": True/False,
                "indicators": ["指标1", "指标2", ...]
            }
        """
        prompt = self._load_prompt()
        content = self._call_llm(prompt, query)
        result = self._parse_json_response(content)

        if result is None:
            return {
                "rewritten": query,
                "need_report": False,
                "enable_rag": True,
                "indicators": [],
            }

        return {
            "rewritten": result.get("rewritten", query),
            "need_report": result.get("need_report", False),
            "enable_rag": result.get("enable_rag", True),
            "indicators": result.get("indicators", []),
        }


# ===== 模块级单例 =====

base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

_query_rewriter = None


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter(
            prompt_path=settings.llm_query_rewriter_prompt,
            base_url=base_url,
            api_key=api_key,
            model="Qwen3.5-4B",
        )
    return _query_rewriter


query_rewriter = get_query_rewriter()
