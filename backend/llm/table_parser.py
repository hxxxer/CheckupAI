import json
import os
import re
import tomllib
from typing import Any, Union

from openai import OpenAI

from backend.config import settings
from .base_llm import BaseLLM
from .utils import safe_json_parse


class TableParserLLM(BaseLLM):
    def __init__(
        self,
        prompt_path,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B",
        enable_thinking: bool = False
    ):
        """
        初始化客户端

        Args:
            base_url: vLLM API地址
            api_key: API密钥（vLLM默认为EMPTY）
            model: 模型名称
        """
        super().__init__(
            prompt_path=prompt_path,
            base_url=base_url,
            api_key=api_key,
            model=model,
            enable_thinking=enable_thinking
        )

    def parse(self, table_data: dict) -> Union[list, None]:
        """
        使用 LLM 解析表格数据并解析为 JSON
        :param table_data: table_html_to_md 返回的结构化数据，包含：
            - table_count: 表格数量
            - tables: 表格列表，每个表格包含：
              - matrix: 二维矩阵
              - segments: 分段信息
              - is_double_column: 是否双栏
              - markdown: Markdown 字符串
              - html: HTML 字符串
              - context: 第一个表格前的上下文文本（可选）
        :return: 解析后的JSON结果
        """
        if not table_data or not table_data.get("tables"):
            return []

        # 构建Prompt
        prompt = self._load_prompt()

        results = []

        for table in table_data["tables"]:
            context = table.get("context", "")
            context = f"【表格上方原始文本】:\n{context}\n\n" if context else ""

            md_content = table["markdown"]
            html_content = table["html"]

            user_content = (
                f"{context}"
                f"【Markdown】:\n{md_content}\n\n"
                # f"【HTML源码】:\n{html_content}"
            )

            # 调用 LLM
            content = self._call_llm(prompt, user_content)
            
            # 解析 JSON
            parsed_result = self._parse_json_response(content)

            # 安全解析JSON
            results.append(parsed_result)

        # print("提取结果:", content)

        return results


# 获取环境变量
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

_table_parser = None


def get_table_parser() -> TableParserLLM:
    global _table_parser
    if _table_parser is None:
        _table_parser = TableParserLLM(prompt_path=settings.llm_table_prompt,
                                       base_url=base_url,
                                       api_key=api_key,
                                       model="Qwen3.5-4B")
    return _table_parser


table_parser = get_table_parser()
