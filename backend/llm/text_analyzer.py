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
from .utils import safe_json_parse
from backend.ocr import OCRResult, Page, TextAnalysis, PersonalInfo, PositiveFinding


class TextAnalyzer:
    """使用 LLM 分析体检报告文本内容"""

    def __init__(
        self,
        prompt_path: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B"
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
            "max_tokens": 32768,         # 最大生成长度
            "temperature": 0.7,          # 温度
            "top_p": 0.8,               # 核采样
            "presence_penalty": 1.5,     # 存在惩罚
            "extra_body": {
                "top_k": 20,             # Top-K采样
                "chat_template_kwargs": {"enable_thinking": False},
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
        ocr_results: List[OCRResult]
    ) -> None:
        """
        分析 OCR 结果中的文本内容，且原地修改 OCR 结果

        Args:
            ocr_results: OCR 解析结果列表
        """
        for ocr_result in ocr_results:
            # 逐页分析
            for page in ocr_result.pages:
                page.text_analyses = self._analyze_page(page)

    def _analyze_page(self, page: Page) -> TextAnalysis:
        """
        分析单个页面的文本

        Args:
            page: RawPage 对象

        Returns:
            TextAnalysis格式的分析结果
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
        result_dict = safe_json_parse(content)

        # 添加页码信息
        if result is None:
            result = {
                "personal_info": {},
                "positive_findings": [],
                "summary": ""
            }

        # 解析个人信息
        personal_info_dict = result_dict.get("personal_info", {})
        personal_info = PersonalInfo(
            name=personal_info_dict.get("name"),
            gender=personal_info_dict.get("gender"),
            age=personal_info_dict.get("age"),
            exam_date=personal_info_dict.get("exam_date")
        )

        # 解析阳性发现
        positive_findings = []
        for finding_dict in result_dict.get("positive_findings", []):
            finding = PositiveFinding(
                text=finding_dict.get("text", ""),
                region_index=finding_dict.get("region_index", 0),
                type=finding_dict.get("type", "检验异常")
            )
            positive_findings.append(finding)

        result = TextAnalysis(
            has_abnormal_findings=result_dict.get(
                "has_abnormal_findings", False),
            personal_info=personal_info,
            positive_findings=positive_findings,
            summary=result_dict.get("summary")
        )

        return result

    def _build_page_text(self, page: Page) -> str:
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
                                      model="Qwen3.5-4B")
    return _text_analyzer


text_analyzer = get_text_analyzer()
