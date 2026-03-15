"""
文本分析模块 - 使用 LLM 从 OCR 文本中提取有效信息

功能：
1. 提取个人信息（姓名、性别、年龄、体检号、体检日期）
2. 提取阳性结果（病变、异常指标、医生备注）
3. 生成健康总结
"""

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from backend.config import settings
from backend.llm import safe_json_parse
from backend.ocr import RawOCRResult, RawPage


class TextAnalyzer:
    """使用 LLM 分析体检报告文本内容"""

    def __init__(
        self,
        prompt_path: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3-4B"
    ):
        """
        初始化文本分析器

        Args:
            prompt_path: Prompt 配置文件路径
            base_url: vllm server 地址
            api_key: API Key（本地服务不需要）
            model_name: 模型名称
        """
        self.prompt_path = prompt_path
        self._prompt = None

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

        # 采样参数配置
        self.sampling_params = {
            "temperature": 1.0,          # 温度
            "top_p": 0.95,               # 核采样
            "presence_penalty": 2.0,     # 存在惩罚
            "max_tokens": 32768,         # 最大生成长度
            "extra_body": {
                "top_k": 40,             # Top-K采样
            }
        }

    def _load_prompt(self) -> str:
        """加载 Prompt 配置"""
        if self._prompt is None:
            with open(self.prompt_path, "rb") as f:
                raw_prompt = tomllib.load(f)

            prompt = raw_prompt.get("prompt", {}).get("role", "") + "\n\n"

            if 'rules' in raw_prompt:
                prompt += "规则：\n"
                for key, rule in raw_prompt["rules"].items():
                    prompt += f"{key}：{rule}\n"
                prompt += "\n"

            if 'output' in raw_prompt:
                prompt += "输出格式：\n"
                prompt += raw_prompt.get("format", "") + "\n\n"

            if 'examples' in raw_prompt:
                prompt += "示例：\n"
                for key, example in raw_prompt["examples"].items():
                    prompt += f"例子{key} - {example.get('title', '')}\n"
                    prompt += "输入：\n"
                    prompt += example.get("input", "") + "\n"
                    prompt += "输出：\n"
                    prompt += example.get("output", "") + "\n\n"

            prompt += raw_prompt.get("task", {}).get("instruction", "")
            prompt += "\n\n"
            self._prompt = prompt

        return self._prompt

    def analyze(
        self,
        ocr_results: List[RawOCRResult]
    ) -> List[Dict[str, Any]]:
        """
        分析 OCR 结果中的文本内容

        Args:
            ocr_results: OCR 解析结果列表

        Returns:
            分析结果列表，每个文件一个结果
        """
        results = []

        for ocr_result in ocr_results:
            file_result = {
                "source_path": ocr_result.source_path,
                "pages": []
            }

            # 逐页分析
            for page in ocr_result.pages:
                page_result = self._analyze_page(page)
                file_result["pages"].append(page_result)

            results.append(file_result)

        return results

    def _analyze_page(self, page: RawPage) -> Dict[str, Any]:
        """
        分析单个页面的文本

        Args:
            page: RawPage 对象

        Returns:
            分析结果字典
        """
        # 构建输入文本
        input_text = self._build_page_text(page)

        # 如果没有文本，返回空结果
        if not input_text.strip():
            return {
                "page_index": page.page_index,
                "personal_info": {},
                "positive_findings": [],
                "summary": ""
            }

        # 调用 LLM
        prompt = self._load_prompt()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.sampling_params
        )

        content = response.choices[0].message.content

        # 解析 JSON 结果
        result = safe_json_parse(content)

        # 添加页码信息
        if result is None:
            result = {
                "personal_info": {},
                "positive_findings": [],
                "summary": ""
            }

        result["page_index"] = page.page_index

        return result

    def _build_page_text(self, page: RawPage) -> str:
        """
        合并一页的所有 regions 文本

        Args:
            page: RawPage 对象

        Returns:
            合并后的文本字符串
        """
        lines = []
        for region in sorted(page.regions, key=lambda x: x.index):
            text = region.text.strip()
            if text:
                lines.append(f"[{region.index}|{region.label}] {text}")

        return "\n".join(lines)


# 获取环境变量
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "not-needed")

_text_analyzer = None


def get_text_analyzer() -> TextAnalyzer:
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer(prompt_path=settings.llm_text_prompt,
                                      base_url=base_url,
                                      api_key=api_key,
                                      model="Qwen3-4B")
    return _text_analyzer


text_analyzer = get_text_analyzer()
