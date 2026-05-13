"""
ChatLLM — vLLM 对话客户端
封装 OpenAI 兼容 API 调用，支持 LangChain Runnable 协议
"""

import os
from typing import Any, Dict, List

from openai import OpenAI

from .base_llm import BaseLLM


class ChatLLM(BaseLLM):
    """vLLM 对话服务，OpenAI 兼容协议，同时支持 LangChain LCEL"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen3.5-4B",
        enable_thinking: bool = True,
    ):
        super().__init__(
            prompt_path='',
            base_url=base_url,
            api_key=api_key,
            model=model,
            enable_thinking=enable_thinking
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        直接调用 chat completion

        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

        Returns:
            LLM 回复文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.sampling_params,
        )
        return response.choices[0].message.content or ""

    def invoke(self, prompt_value: Any) -> str:
        """
        LangChain Runnable 协议入口
        接收 ChatPromptTemplate 生成的 prompt_value，提取 messages 后调用 vLLM
        """
        messages = [
            {"role": m.type, "content": m.content}
            for m in prompt_value.to_messages()
        ]
        return self.chat(messages)


# ===== 模块级单例 =====

base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

_chat_llm = None


def get_chat_llm() -> ChatLLM:
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = ChatLLM(
            base_url=base_url,
            api_key=api_key,
            model="Qwen3.5-4B",
            enable_thinking=True
        )
    return _chat_llm


chat_llm = get_chat_llm()
