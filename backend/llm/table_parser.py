import json
import os
import re
import tomllib
from typing import Any, Union, Dict, List

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
        # 加载医学名词映射表
        self.medical_terms_map = self._load_medical_terms_map()

    def _load_medical_terms_map(self) -> Dict[str, str]:
        """
        加载医学名词统一化映射表
        """
        map_file_path = os.path.join(os.path.dirname(__file__), "medical_terms_map.json")
        if os.path.exists(map_file_path):
            try:
                with open(map_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告：加载医学名词统一表失败 {e}")
                return {}
        return {}

    def normalize_item(self, raw_item: str) -> str:
        """将原始 item 名称转换为标准名称，若无映射则保留原值（或可改为 None/自定义）"""
        if raw_item is None or not isinstance(raw_item, str):
            return raw_item
        
        # 去除首尾空格，可考虑大小写归一化
        cleaned = raw_item.strip()
        if not cleaned:
            return raw_item
        
        return self.medical_terms_map.get(cleaned, raw_item)  # 找不到则返回原值

    def normalize_json(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        对解析结果中的医学名词进行统一化
        针对 LLM 输出格式 {"title": ..., "table": ["item": ..., ...]} 进行优化
        """
        if not self.medical_terms_map or not isinstance(input_json, dict):
            return input_json
        
        output = input_json.copy()  # 浅拷贝，保留其他字段
        if "table" in output and isinstance(output["table"], list):
            new_table = []
            for record in output["table"]:
                if isinstance(record, dict) and "item" in record:
                    record_copy = record.copy()
                    record_copy["item"] = self.normalize_item(record["item"])
                    new_table.append(record_copy)
                else:
                    new_table.append(record)  # 非预期结构则原样保留
            output["table"] = new_table
        return output

    def parse(self, table_data: dict) -> Union[list, None]:
        """
        使用 LLM 解析表格数据并解析为 JSON

        Args:
            table_data: table_html_to_md 返回的结构化数据中的表格，包含：
            - matrix: 二维矩阵
            - segments: 分段信息
            - is_double_column: 是否双栏
            - markdown: Markdown 字符串
            - html: HTML 字符串
            - context: 第一个表格前的上下文文本（可选）

        Returns:
            解析后的JSON结果
        """
        if not table_data or not table_data.get("tables"):
            return []

        # 构建Prompt
        prompt = self._load_prompt()

        context = table_data.get("context", "")
        context = f"【表格上方原始文本】:\n{context}\n\n" if context else ""

        md_content = table_data["markdown"]
        html_content = table_data["html"]

        user_content = (
            f"{context}"
            f"【Markdown】:\n{md_content}\n\n"
            # f"【HTML源码】:\n{html_content}"
        )

        # 调用 LLM
        content = self._call_llm(prompt, user_content)
        
        # 解析 JSON
        parsed_result = self._parse_json_response(content)

        # 医学名词统一化
        if parsed_result:
            parsed_result = self.normalize_json(parsed_result)

        # print("提取结果:", content)

        return parsed_result


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
