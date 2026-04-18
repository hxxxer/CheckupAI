"""
LLM 服务基类 - 提供通用的 LLM 调用功能
"""

import os
import tomllib
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from backend.config import settings
from .utils import safe_json_parse


class BaseLLM:
    """LLM 服务基类，提供通用的初始化和 Prompt 加载功能"""

    def __init__(
        self,
        prompt_path: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B",
        enable_thinking: bool = False
    ):
        """
        初始化 LLM 服务

        Args:
            prompt_path: Prompt 配置文件路径（TOML格式）
            base_url: vLLM API地址
            api_key: API密钥（vLLM默认为EMPTY）
            model: 模型名称
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
            "max_tokens": 32768,
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        } if enable_thinking else {
            "max_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }

    def _load_prompt(self) -> str:
        """
        从 TOML 文件加载并构建 Prompt
        
        Returns:
            构建好的 Prompt 字符串
            
        Raises:
            FileNotFoundError: Prompt 文件不存在
            ValueError: TOML 文件格式错误
        """
        if self._prompt is None:
            try:
                with open(self.prompt_path, "rb") as f:
                    raw_prompt = tomllib.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt文件缺失: {self.prompt_path}")
            except tomllib.TOMLDecodeError as e:
                raise ValueError(f"TOML文件格式不合法: {self.prompt_path}: {e}")

            prompt = raw_prompt.get("prompt", {}).get("role", "") + "\n\n"

            if 'rules' in raw_prompt:
                prompt += "规则：\n"
                for key, rule in raw_prompt["rules"].items():
                    prompt += f"{key}：{rule}\n"
                prompt += "\n"

            if 'output' in raw_prompt:
                prompt += "输出格式：\n"
                prompt += raw_prompt.get("title", "") + "\n"
                prompt += raw_prompt.get("description", "") + "\n\n"

            if 'format' in raw_prompt:
                prompt += "输出格式：\n"
                prompt += raw_prompt.get("format", "") + "\n\n"

            if 'examples' in raw_prompt:
                prompt += "以下为若干个示例，涵盖绝大部分特殊情况处理方式：\n"
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

    def _call_llm(
        self,
        system_content: str,
        user_content: str
    ) -> str:
        """
        调用 LLM 获取响应内容

        Args:
            system_content: 系统提示词
            user_content: 用户输入内容

        Returns:
            LLM 返回的文本内容

        Raises:
            RuntimeError: API 调用失败
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.sampling_params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API调用失败: {str(e)}")

    def _parse_json_response(self, content: str) -> Union[Dict, List, None]:
        """
        解析 LLM 返回的 JSON 响应

        Args:
            content: LLM 返回的文本内容

        Returns:
            解析后的 JSON 对象（dict/list）或 None
        """
        return safe_json_parse(content)